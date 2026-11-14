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
                if batch_size == 7:
                    obs['combat_legal'][-1] = 0
                    obs['combat_legal'][-1, 0] = 1
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
                    if batch_size == 7 and i == count - 1:
                        mapping.fill_(2**30)  # forced rows own no query map
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
                    nonforced = meta[:, 0] > 1
                    reference = expected[actions[:, 0], actions[:, 1]].clone()
                    reference[~nonforced[actions[:, 0]]] = 0
                    torch.testing.assert_close(scores, reference, atol=3e-5, rtol=1e-5)
                    torch.testing.assert_close(scorer.buffers['coef'][:count, 12][nonforced], values.flatten()[nonforced], atol=3e-5, rtol=1e-5)
            assert scorer.counts['captures'] >= 3 and scorer.counts['replays'] >= 9
            assert scorer.counts['evictions'] >= 2
        finally:
            scorer.close()
        assert scorer.graph is None and not scorer.buffers and not scorer.weights


def test_full480_current_native_observer_and_packed_model(latent_env):
    """The original frame reads real native states, against the Gym consumer."""
    import slaythespire as sts
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.search.full480.combo_model import DeckWeights
    from rl.search.full480.deck_inputs import DeckInputTables
    from rl.search.full480.run_device import cpu_bridge, artifact_directory
    from rl.search.full480.combo_runtime import BattleArena, BattlePool

    assert torch.cuda.is_available()
    torch.manual_seed(20261111)
    for _ in range(128):
        if latent_env.thick.bc is not None:
            break
        latent_env.step(int(np.flatnonzero(latent_env.action_masks())[0]))
    assert latent_env.thick.bc is not None
    deck = ['WARCRY', 'BURNING_PACT', 'HEADBUTT', 'TRUE_GRIT', 'DEFEND_RED',
            'BATTLE_TRANCE', 'STRIKE_RED', 'DEFEND_RED', 'BASH', 'EXHUME']
    latent_env.thick.gc.set_deck([(name, 0) for name in deck])
    latent_env.thick.gc.set_potion_state(['Gamblers Brew', 'Liquid Memories'])
    latent_env.thick.bc = sts.create_battle(latent_env.thick.gc, 'Jaw Worm')
    latent_env._invalidate_backend_cache()
    policy = DeckLatentPolicy(latent_env.observation_space, latent_env.action_space,
        lambda _: 5e-5, d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
        reward_flow='sequential-v1', deck_combat_encoder='transformer64x2').cuda()
    policy.set_training_mode(False)
    cpu = cpu_bridge()
    with torch.no_grad():
        flat = DeckWeights(policy.core)
        flat.input_tables = DeckInputTables(policy.core, latent_env._obs())
        assert flat.input_tables.header() == (artifact_directory()/'deck/deck_schema.h').read_text()
        arena = BattleArena(1)
        pool = None
        checked = selections = prefixes = 0
        try:
            pool = BattlePool(arena, 1, net=flat.net, flat=flat, frame=1, tile=32,
                              frame_tokens=4, frame_queries=4)
            # Read-only borrowed mirror address; no synthetic search or sampler.
            pool.arena[0] = arena.arena.data_ptr()
            pool.live[0] = 0; pool.logical_ids[0] = 0
            for _ in range(64):
                if latent_env.thick.bc is None or latent_env._phase() != 'combat' or latent_env._at_combat_exit_anchor():
                    break
                observation = latent_env._obs()
                native = cpu.State(latent_env.thick.bc).snapshot()
                if not arena.versions:
                    assert native['portable'], {k: v for k, v in native.items() if k != 'words'}
                    arena.load({0: native})
                obs = {k: torch.as_tensor(v[None], device='cuda') for k, v in observation.items()}
                context = policy.core.prepare_search_context(obs)
                pool.root_inputs[0] = torch.cat((context.combat[0], obs['run_keys'][0], obs['ascension'][0]))
                pool.chain_inputs[0].copy_(pool.root_inputs[0])
                pool.root_relics[0] = obs['relics'][0].int()
                pool.root_counters[0] = obs['relic_counters'][0]
                prefix = sum(1 << i for i in (latent_env._multi or ()))
                prefixes += int(prefix != 0)
                pool.selection_prefix[0] = prefix
                summary = pool.frame_pipeline.run(0, 1)
                a, ne, _, _, _, nt, nq, _, _, real = summary
                w = pool.frame_pipeline.buffers
                torch.testing.assert_close(w['legal'][0].bool(), obs['combat_legal'][0].bool())
                if real:
                    expected, values = policy.core.combat_with_context(obs, context)
                    action_ids = pool.keys[:a, 0].long()
                    torch.testing.assert_close(pool.scores[:a], expected[0, action_ids], atol=3e-5, rtol=1e-5)
                    torch.testing.assert_close(pool.frame_pipeline.model.buffers['coef'][0, 12], values[0, 0], atol=3e-5, rtol=1e-5)
                    tokens, _, _, _ = flat.net.token_inputs(dict(obs, _combat_context=context.combat))
                    count = int(pool.meta[0, 1])
                    physical = pool.chunks[0, torch.arange(count, device='cuda')//64] + torch.arange(count, device='cuda') % 64
                    slots = pool.ids[physical].long()
                    torch.testing.assert_close(pool.numbers[physical], tokens[0, slots], atol=3e-5, rtol=1e-5)
                    checked += 1
                    selections += int(observation['select_context'][0] != 0)
                legal = np.flatnonzero(latent_env.action_masks())
                action = int(next((x for x in legal if 1 <= x < 187), legal[0]))
                _, _, _, _, info = latent_env.step(action)
                if info.get('simulator_transition') is not False:
                    arena.step({0: (*arena.versions[0], int(info['native_action']))})
            assert checked >= 8 and selections >= 1 and prefixes >= 1, (checked, selections, prefixes)
            assert pool.frame_pipeline.counts['overflow_retries'] > 0
            assert pool.frame_pipeline.counts['replays'] >= checked
        finally:
            if pool is not None:
                pool.arena.zero_()
                pool.close()
            arena.close()
        assert arena.modules.references == 0 and arena.modules.closed


def test_full480_pcg64_matches_numpy_and_owns_each_chain():
    from rl.search.full480.combo_runtime import BattleArena
    from rl.search.full480.finite_plan import paired_worlds
    assert torch.cuda.is_available()
    words = np.concatenate([paired_worlds(seed, actions, budget)
        for seed, actions, budget in [(20261111, 3, 64), (0, 197, 64), (2**63-1, 2, 256)]])
    assert len(paired_worlds(0, 197, 64)) == 1
    # Include duplicate worlds and rollover-prone 128-bit carries; independent
    # lane state must not become one shared mutable random stream.
    states = np.repeat(words[:, 1:], 2, axis=0).copy()
    expected = []
    for row in states:
        generator = np.random.PCG64()
        state = generator.state
        state['state'].update(state=int(row[0]) | int(row[1]) << 64,
                              inc=int(row[2]) | int(row[3]) << 64)
        generator.state = state
        expected.append(generator.random_raw(200))
    arena = BattleArena(1, deck=True)
    try:
        device = arena.upload(states, np.uint64)
        output = arena.arr((len(states), 200), torch.int64)
        arena.launch('paired_pcg_probe', [device, output], [len(states), 200], grid=len(states))
        np.testing.assert_array_equal(output.cpu().numpy().view(np.uint64), expected)
        temporary = torch.empty(64 * 1024 * 1024, device='cuda', dtype=torch.uint8)
        temporary.fill_(9)
        del temporary
        torch.cuda.empty_cache()
        device.copy_(arena.upload(states, np.uint64))
        arena.launch('paired_pcg_probe', [device, output], [len(states), 200], grid=len(states))
        np.testing.assert_array_equal(output.cpu().numpy().view(np.uint64), expected)
    finally:
        arena.close()
    assert arena.modules.references == 0 and arena.modules.closed


@pytest.mark.parametrize('scenario', ['prefix', 'growth', 'escape'])
def test_full480_current_paired_states_prefixes_and_graph_retry(latent_env, scenario, tmp_path):
    from rl.search.full480.combo_runtime import BattleArena, BattlePool
    from rl.search.full480.combo_model import DeckWeights
    from rl.search.full480.deck_inputs import DeckInputTables
    from rl.search.full480.finite_plan import PAIRED_ANCHOR, paired_worlds
    from rl.search.full480.run_device import cpu_bridge
    from rl.policies.deck_latent import DeckLatentPolicy
    assert torch.cuda.is_available()
    torch.manual_seed(20261111)
    for _ in range(128):
        if latent_env.thick.bc is not None:
            break
        latent_env.step(int(np.flatnonzero(latent_env.action_masks())[0]))
    import slaythespire as sts
    latent_env.thick.gc.set_deck([(name, 0) for name in ['WARCRY', 'BURNING_PACT', 'HEADBUTT', 'TRUE_GRIT',
        'DEFEND_RED', 'BATTLE_TRANCE', 'STRIKE_RED', 'DEFEND_RED', 'BASH', 'EXHUME']])
    latent_env.thick.gc.set_potion_state(['Gamblers Brew', 'Liquid Memories'])
    latent_env.thick.bc = sts.create_battle(latent_env.thick.gc, 'Jaw Worm')
    if scenario != 'prefix':
        latent_env.thick.gc.set_deck([('RITUAL_DAGGER' if scenario == 'growth' else 'STRIKE_RED', 0)] * 5)
        latent_env.thick.gc.set_potion_state(['Smoke Bomb'] if scenario == 'escape' else [])
        latent_env.thick.bc = sts.create_battle(latent_env.thick.gc, 'Jaw Worm')
        if scenario == 'growth':
            latent_env.thick.bc.set_player_strength(999)
    latent_env._invalidate_backend_cache()
    policy = DeckLatentPolicy(latent_env.observation_space, latent_env.action_space,
        lambda _: 5e-5, d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
        reward_flow='sequential-v1', deck_combat_encoder='transformer64x2').cuda()
    policy.set_training_mode(False)
    cpu = cpu_bridge()
    core = policy.core
    branches = []
    with torch.no_grad():
        obs = latent_env._obs()
        data = {k: torch.as_tensor(v[None], device='cuda') for k, v in obs.items()}
        context = core.prepare_search_context(data)
        flat = DeckWeights(core)
        flat.input_tables = DeckInputTables(core, obs)
        mapping = sorted(latent_env._joint_policy_actions())
        worlds = paired_worlds(20261111, len(mapping), 2 * len(mapping))
        budget = len(mapping) * len(worlds)
        row = cpu.State(latent_env.thick.bc).root_plan(20261111, budget, 1., 8, False)
        semantic = dict(zip(row['action_bits'], row['semantic'], strict=True))
        row.update(cpu.inventory(latent_env.thick.gc))
        row.update(contract=PAIRED_ANCHOR, worlds=worlds, prefix=0,
            action_bits=np.array([bits for _, bits in mapping], np.uint32),
            semantic=np.array([semantic[bits] for _, bits in mapping], np.uint64),
            order=np.arange(len(mapping)), context_inputs=torch.cat((context.combat,
                data['run_keys'], data['ascension']), 1)[0].cpu().numpy(),
            relics=obs['relics'], relic_counters=obs['relic_counters'],
            anchor_context=context.anchor[0].cpu().numpy(),observation={k:v.copy() for k,v in obs.items()})
        arena = BattleArena(1, deck=True)
        pool = None
        try:
            versions = arena.load({0: row['snapshot']})
            row['mirror'] = (0, *versions[0])
            pool = BattlePool(arena, budget, net=flat.net, flat=flat, frame=budget, tile=32,
                             frame_tokens=4, frame_queries=4)
            for world in worlds[:, 0]:
                for action, _ in mapping:
                    branch = latent_env.fork_run(worldline=int(world))
                    branch.thick.bc.nosl_redeterminize(int(world))
                    branch._invalidate_backend_cache()
                    branch.step(action)
                    branches.append([branch, np.random.default_rng(int(world)^0x504F4C494359), 1])
            pool.submit({0: row})
            def compare_states():
                ids = [i for i, (e, _, _) in enumerate(branches) if e.thick.bc is not None]
                words, meta, _ = pool.snapshot(ids)
                for j, i in enumerate(ids):
                    snap = cpu.State(branches[i][0].thick.bc).snapshot()
                    np.testing.assert_array_equal(words[j, :len(snap['words'])], snap['words'], err_msg=str(i))
                    assert int(pool.selection_prefix[i]) == sum(1 << j for j in (branches[i][0]._multi or ()))
            compare_states()
            for tick in range(8):
                active = torch.nonzero(pool.alive).flatten().cpu().tolist()
                if not active:
                    break
                expected_draws = {}
                for i in active:
                    env, rng, depth = branches[i]
                    mask = env.action_masks()
                    legal = np.flatnonzero(mask)
                    if len(legal) == 1:
                        action = legal[0]
                        expected_draws[i] = -1.
                    else:
                        ob = {k: torch.as_tensor(v[None], device='cuda') for k, v in env._obs().items()}
                        branch_context = core.prepare_search_context(ob) if scenario == 'growth' else context
                        logits, _ = core.combat_with_context(ob, branch_context)
                        scores = logits[0, legal].cpu().double().numpy()
                        probabilities = np.exp(scores-scores.max())
                        probabilities /= probabilities.sum()
                        before = rng.bit_generator.state
                        action = rng.choice(legal, p=probabilities)
                        reference = np.random.default_rng()
                        reference.bit_generator.state = before
                        expected_draws[i] = reference.random()
                    env.step(int(action))
                    branches[i][2] = depth + 1
                if tick == 1:
                    temporary = torch.empty(64 * 1024 * 1024, device='cuda', dtype=torch.uint8)
                    temporary.fill_(3)
                    del temporary
                    torch.cuda.empty_cache()
                pool.tick()
                np.testing.assert_array_equal(pool.draws[active].cpu().numpy(), list(expected_draws.values()))
                compare_states()
            assert not pool.active_count
            gold = []
            for branch, _, _ in branches:
                if branch.thick.is_over:
                    gold.append(0.)
                else:
                    leaf = {k:torch.as_tensor(v[None],device='cuda') for k,v in branch._obs().items()}
                    branch_context = core.prepare_search_context(leaf) if scenario == 'growth' else context
                    if branch._at_combat_exit_anchor():
                        value = core.anchor_with_context(leaf,branch_context)
                    else:
                        _, value = core.combat_with_context(leaf,branch_context)
                    gold.append(float(value.item()))
            result = pool.reap()[0]
            np.testing.assert_allclose(result['values'],np.array(gold).reshape(len(worlds),-1).mean(0),atol=3e-5,rtol=1e-5)
            assert result['simulations'] == budget
            if scenario == 'prefix':
                assert pool.stats[:, 11].sum().item() > 0, 'AR prefix path was not exercised'
            else:
                assert pool.frame_pipeline.counts['anchor_value_rows'] > 0
                if scenario == 'growth':
                    assert pool.frame_pipeline.counts['context_refresh_rows'] > 0
            assert pool.graph_scorer.statistics()['step_overflow_retries'] > 0
            assert pool.graph_scorer.statistics()['step_replays'] > 0
            pool.close()
            arena.close()
            # Exercise the original CPU owner and its real FIFO GPU service on
            # exactly the same root/worlds, including pre-existing callbacks.
            from rl.search.full480.service_devicestepfull480141 import GpuService
            from rl.search.full480.cpu_boundary import CpuFallback
            from rl.algorithms.upper_probe import state_digest
            import ctypes
            from rl.search.full480.run_device import lib,check
            # The oracle in this parent just unloaded every simulator module.
            # Match the normal learner's stack before starting a second CUDA
            # context; retain the worker's original 64 KiB simulator stack.
            stack=ctypes.c_size_t();lib.cuCtxGetLimit.argtypes=[ctypes.POINTER(ctypes.c_size_t),ctypes.c_int]
            check(lib.cuCtxGetLimit(ctypes.byref(stack),0),'read idle oracle stack')
            check(lib.cuCtxSetLimit(0,1024),'release idle oracle stack')
            torch.cuda.empty_cache()
            path=tmp_path/'frozen-policy.pt';torch.save(policy,path)
            service=GpuService(1,path,expected_digest=state_digest(policy.state_dict()),
                prototypes=obs,frozen_policy=policy,budget_cap=budget,frame=budget,tile=32)
            try:
                cfg=sts.SmallPlayoutConfig();cfg.simulations=budget;cfg.playout_depth=8;cfg.paired_worldlines=True
                fallback=CpuFallback([(0,'hallway',cfg,20261111)],[latent_env],service,
                    capacity=min(8,budget),threads=2,rows={0:row})
                completed=[]
                import time
                deadline=time.monotonic()+90
                while not fallback.complete:
                    assert time.monotonic()<deadline,'original CPU fallback exceeded test budget'
                    completed.extend(fallback.step())
                    if fallback.pending is not None:service.poll(.001)
                assert len(completed)==1
                actual=completed[0][1]
                for key in ('action_bits','visits','terminal_visits','statistics','playout_lengths'):
                    np.testing.assert_array_equal(actual[key],result[key],err_msg=key)
                np.testing.assert_allclose(actual['values'],result['values'],atol=3e-5,rtol=1e-5)
                status=service.statistics()
                assert status['fallback_graph']['captures']>0
                assert status['upper_graph']['context_cache_hits']>0
                service.close()
            finally:
                if not service.closed:service.close(cancel='test cleanup')
                check(lib.cuCtxSetLimit(0,stack.value),'restore oracle stack')
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
            for branch, _, _ in branches:
                branch.close()
