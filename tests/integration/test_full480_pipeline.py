"""Original full480 execution mechanisms, with explicit frozen artifacts."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


def test_original_full480_whole_step_graph_and_frame_growth(combat_factory):
    """The old model is a porting oracle, never the current training policy."""
    assert torch.cuda.is_available(), 'full480 qualification requires CUDA and its explicit artifacts'
    from rl.search.full480.run_device import cpu_bridge
    from rl.search.full480.combo_runtime import BattleArena, BattlePool
    from rl.policies.token_regret import OnlineTokenRegretPolicy, encode_public_roots, pack_rows

    torch.manual_seed(20261111)
    torch.set_float32_matmul_precision('highest')
    cpu = cpu_bridge()
    envs = [combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 4 + ['BASH'],
                           seed=20261111 + i, combat_action_contract='autoregressive-v2')
            for i in range(2)]
    with torch.inference_mode():
        net = OnlineTokenRegretPolicy(785).cuda().eval()
        flat = SimpleNamespace(net=net, precision='tf32x3',
            lookup=net.symbol_lift(net.symbol.weight).detach(),
            head_weight=torch.cat((net.source_score.weight, net.left.weight,
                                  net.target_score.weight, net.right.weight), 0).detach())
        arena = BattleArena(2)
        pool = None
        try:
            versions = arena.load({i: cpu.State(env.thick.bc).snapshot()
                                   for i, env in enumerate(envs)})
            pool = BattlePool(arena, 16, net=net, flat=flat, frame=32, tile=32,
                              frame_tokens=4, frame_queries=4)
            rows = {}
            for i, env in enumerate(envs):
                row = cpu.State(env.thick.bc).root_plan(20261111 + i, 16, 1., 4, False)
                row.update(cpu.inventory(env.thick.gc))
                row['mirror'] = (i, *versions[i])
                rows[i] = row
            pool.submit(rows)
            pool.observe()
            raw = pool.public()
            pool.score()
            for row, tokens in zip(raw, encode_public_roots(raw), strict=True):
                row['tokens'] = tokens
            expected = net.forward_packed(*pack_rows(raw, 'cuda')).cpu().numpy()
            metadata = pool.meta[:pool.count].cpu().numpy()
            scores = pool.scores.cpu().numpy()
            for i, row in enumerate(raw):
                count, _, start, _ = map(int, metadata[i])
                if count > 1:
                    np.testing.assert_allclose(scores[start:start + count], expected[i, :count],
                                               atol=3e-5, rtol=1e-5)
            pool.apply()
            pool.refresh()
            temporary = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cuda')
            temporary.fill_(7)
            del temporary
            torch.cuda.empty_cache()
            completed = {}
            for _ in range(8):
                completed.update(pool.receive())
                if not pool.pending:
                    break
            assert set(completed) == set(rows)
            assert all(r['simulations'] == 16 for r in completed.values())
            statistics = pool.graph_scorer.statistics()
            assert statistics['overflow_retries'] > 0
            assert statistics['step_captures'] > 0 and statistics['step_replays'] > 0
            assert statistics['workspace_alias_version'] == 1
            assert not pool.page_leases and not pool.ledger.pending
            pool.close()
            assert pool.state_pages.closed
        finally:
            if pool is not None and not pool.closed:
                pool.graph_scorer.close()
                pool.launch('stream_destroy', [pool.arena, pool.constructed], [pool.n], grid=pool.n)
                pool.launch('stream_destroy', [pool.roots, pool.root_constructed], [pool.m], grid=pool.m)
                torch.cuda.synchronize()
                pool.state_pages.release(list(pool.page_leases.values()))
                pool.page_leases.clear()
                pool.state_pages.close()
                pool.modules.references -= 1
                pool.closed = True
            arena.close()
        assert arena.modules.references == 0 and arena.modules.closed


def test_full480_current_deck_fused_heads_and_owned_graphs(latent_env, monkeypatch):
    """Current eager policy is the independent oracle for the old fused path."""
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.search.full480.combo_model import DeckWeights, GraphScorer
    from env.core import encode

    assert torch.cuda.is_available()
    torch.manual_seed(20261111)
    for _ in range(128):
        if latent_env.thick.bc is not None:
            break
        latent_env.step(int(np.flatnonzero(latent_env.action_masks())[0]))
    assert latent_env.thick.bc is not None
    policy = DeckLatentPolicy(latent_env.observation_space, latent_env.action_space,
        lambda _: 5e-5, d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
        reward_flow='sequential-v1', deck_combat_encoder='transformer64x2').cuda()
    policy.set_training_mode(False)
    core = policy.core
    with torch.no_grad():
        # Exercise trained, nonuniform target and subset factors too.
        for p in (core.combat.complete_head.source_query.weight,
                  core.combat.complete_head.global_query.weight,
                  core.combat.complete_head.subset.choice_key.weight,
                  core.combat.complete_head.subset.stop.weight):
            p.normal_(0, .075)
        data = {k: torch.as_tensor(v[None], device='cuda') for k, v in latent_env._obs().items()}
        context = core.prepare_search_context(data)
        def no_encoder(*args, **kwargs):
            raise AssertionError('inner fused execution recomputed the large context encoder')
        monkeypatch.setattr(core.context, 'forward', no_encoder)
        flat = DeckWeights(core)
        scorer = GraphScorer(flat)
        try:
            for batch_size in (1, 3, 7):
                obs = {k: v.expand(batch_size, *v.shape[1:]).clone() for k, v in data.items()}
                obs['_combat_context'] = context.combat.expand(batch_size, -1).clone()
                # Synthetic head contracts cover every canonical index, empty
                # entity slots, PAD queries and increasing subset prefixes.
                obs['combat_legal'].fill_(1)
                obs['select_native_indices'][:, :5] = torch.arange(1, 6, device='cuda')
                obs['select_hand_mask'][:, 1] = 1
                obs['select_hand_mask'][:, 4] = 1
                obs['pick_remaining'].fill_(3 / encode.PICK_SCALE)
                if batch_size > 1:
                    obs['combat_legal'][1, 1:91] = 0
                    obs['combat_legal'][1, 6] = 1  # sole no-target conditional
                    obs['combat_legal'][1, 187:192] = 0
                tokens, present, spans, required = core.combat.token_inputs(obs)
                count, length, _ = tokens.shape
                # Drop key-only padding, but retain all consumed query slots.
                selected_slots = list(range(21)) + list(range(25, 121))
                packed, masks, queries, maps, starts, lens, b1, b2 = [], [], [], [], [], [], [], []
                nt = nq = 0
                for i in range(count):
                    retained = torch.where(present[i] | torch.tensor(
                        [j in selected_slots for j in range(length)], device='cuda'))[0]
                    logical = {int(j): pos for pos, j in enumerate(retained.cpu().tolist())}
                    q = [nt + logical[j] for j in selected_slots]
                    mapping = torch.full((161,), -1, dtype=torch.int32, device='cuda')
                    mapping[selected_slots] = torch.arange(len(q), device='cuda', dtype=torch.int32)
                    starts.append([nt, nq, len(b1), len(b2)])
                    lens.append([len(retained), len(q), (len(retained)+15)//16, (len(q)+15)//16])
                    b1.extend((i, j) for j in range(0, len(retained), 16))
                    b2.extend((i, j) for j in range(0, len(q), 16))
                    packed.append(tokens[i, retained]); masks.append(present[i, retained]); queries.extend(q); maps.append(mapping)
                    nt += len(retained); nq += len(q)
                features = torch.zeros((nt, 128), device='cuda'); features[:, :64] = torch.cat(packed)
                symbols = torch.zeros(nt, dtype=torch.int64, device='cuda')
                query = torch.tensor(queries, dtype=torch.int64, device='cuda')
                starts = torch.tensor(starts, dtype=torch.int64, device='cuda')
                lens = torch.tensor(lens, dtype=torch.int64, device='cuda')
                b1, b2 = [torch.tensor(x, dtype=torch.int32, device='cuda') for x in (b1, b2)]
                actions = obs['combat_legal'].nonzero()
                owners = actions[:, 0].int().contiguous()
                keys = torch.full((len(actions), 10), -1, dtype=torch.int64, device='cuda')
                keys[:, 0] = actions[:, 1]
                meta = torch.zeros((count, 4), dtype=torch.int32, device='cuda')
                meta[:, 0] = obs['combat_legal'].sum(1).int()
                state = torch.zeros((count, 32), device='cuda')
                unused = torch.empty((len(actions), 24), device='cuda')
                scores = torch.empty(len(actions), device='cuda')
                remaining = (obs['pick_remaining'] * encode.PICK_SCALE).round().int().contiguous()
                scorer.bind_deck(obs['combat_legal'], obs['select_hand_mask'], remaining)
                args = (flat, features, symbols, symbols, torch.cat(masks), query, starts, lens,
                    b1, b2, state, unused, keys, owners, keys, owners, meta, torch.stack(maps), scores, len(actions))
                for replay in range(3):
                    if replay == 1:
                        temporary = torch.empty(64*1024*1024, dtype=torch.uint8, device='cuda')
                        temporary.fill_(3); del temporary
                        torch.cuda.empty_cache()
                    if replay == 2:
                        core.combat.end.bias.add_(.03125)
                        with pytest.raises(RuntimeError, match='drain and refresh'):
                            scorer(*args)
                        flat.refresh()
                    expected, values, _ = core.combat(obs)
                    scorer(*args)
                    torch.testing.assert_close(scores, expected[actions[:, 0], actions[:, 1]], atol=3e-5, rtol=1e-5)
                    torch.testing.assert_close(scorer.buffers['coef'][:count, 12], values.flatten(), atol=3e-5, rtol=1e-5)
            assert scorer.counts['captures'] >= 3 and scorer.counts['replays'] >= 9
            assert scorer.counts['evictions'] >= 2
        finally:
            scorer.close()
        assert scorer.graph is None and not scorer.buffers and not scorer.weights
