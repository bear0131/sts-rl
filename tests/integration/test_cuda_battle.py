"""Resident simulation parity, including device-owned queued continuations."""
import gc

import numpy as np
import pytest
import torch
import slaythespire as sts

from env.core.run_env import _nosl_compact_lut
from env.search.cuda_battle import CudaBattleArena
from env.core.cuda_observation import CudaObservationDecoder


pytestmark = [pytest.mark.native, pytest.mark.cuda, pytest.mark.timeout(120)]


def test_resident_permanent_misc_matches_native_rounding_and_refreshes_only_real_changes():
    import triton
    from env.core.cuda_observation import canonical_misc
    from env.core.outer_return import MISC_SCALE
    from env.search import resident_step_kernels as K
    values = np.r_[np.arange(512), 2**24-1, 2**24, 2**24+1, 2**31-1,
        np.random.default_rng(20261111).integers(0,2**31-1,1024)].astype(np.int32)
    n = len(values)
    canonical = torch.as_tensor((values.astype(np.float64)/MISC_SCALE).astype(np.float32), device='cuda')
    mask = torch.zeros((n,198),device='cuda',dtype=torch.int32); mask[:,:2] = 1
    cats = torch.zeros((n,226),device='cuda',dtype=torch.int32)
    misc = torch.full((n,96),-1,device='cuda',dtype=torch.int32); misc[:,0] = torch.as_tensor(values,device='cuda')
    expected = torch.zeros((n,96),device='cuda'); expected[:,0] = canonical
    deck = torch.zeros((n,96),device='cuda',dtype=torch.int64); deck[:,0] = 1
    owners = torch.arange(n,device='cuda',dtype=torch.int64); frame = owners.int()
    count = torch.tensor([n],device='cuda',dtype=torch.int32)
    depth = torch.ones(n,device='cuda',dtype=torch.int32); outcome = torch.zeros_like(depth)
    kind,forced,need,changed = [torch.zeros_like(depth) for _ in range(4)]
    status = torch.zeros(10,device='cuda',dtype=torch.int32); offset = torch.zeros((),device='cuda',dtype=torch.int64)
    def classify():
        status.zero_()
        K.classify[(triton.cdiv(n,32),)](outcome,mask,cats,misc,expected,deck,owners,
            frame,count,depth,kind,forced,need,changed,status,offset,n,200,MISC_SCALE,32,num_warps=4)
    torch.testing.assert_close(canonical_misc(misc[:,0]),canonical,rtol=0,atol=0)
    classify()
    assert not changed.any(), 'unchanged native integer values must reuse root context'
    misc[:100,0] += 1
    classify()
    assert changed[:100].all() and not changed[100:].any()
    expected[:100,0] = canonical_misc(misc[:100,0])
    classify()
    assert not changed.any(), 'a committed refresh must not repeat on the same integer state'


def test_resident_relic_lookup_keeps_static_inventory_and_unbounded_live_fields():
    from rl.policies.combat_summary import AdditiveCombatInputs
    from rl.policies.shared_wholerun import LearnedCardTable
    from rl.policies.resident_ragged import frozen_relic_summary
    torch.manual_seed(20261111)
    net = AdditiveCombatInputs(LearnedCardTable(12, 8), 9, width=64).cuda().eval()
    live = torch.tensor([False, True, False, True, True, False, True, False, False], device='cuda')
    with torch.no_grad():
        rid = torch.arange(9, device='cuda')
        table = net.relic_terms(rid[:, None], torch.arange(256, device='cuda')[None, :].float()/10)
        for batch in (17, 257, 33):
            ids = torch.tensor([[1, 2, 3, 0, 4, 6]], device='cuda').repeat(batch, 1)
            original = torch.tensor([[1., .9, 0., 99., .2, -.7]], device='cuda').repeat(batch, 1)
            static = (net.relic_terms(ids, original)*(~live[ids])[..., None]).sum(1)
            frozen = static.clone()
            counters = original.clone()
            # Consumed active flag, ordinary counts, fractional/out-of-table
            # values and negative counters must all retain the original basis.
            counters[:, 0] = torch.arange(batch, device='cuda') % 2
            counters[:, 2] = torch.arange(batch, device='cuda').float()/10
            counters[:, 4] = 100.1
            counters[::2, 5] = .035
            expected = net.relic_terms(ids, counters).sum(1)
            actual = frozen_relic_summary(ids, counters, static, live, table,
                net.relic_identity.weight, net.relic_counter.weight)
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(static, frozen, rtol=0, atol=0)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            replay = frozen_relic_summary(ids, counters, static, live, table,
                net.relic_identity.weight, net.relic_counter.weight)
            torch.testing.assert_close(replay, actual, rtol=0, atol=0)


def test_resident_stack_limit_restored_after_last_arena(cuda_battle_artifacts):
    import ctypes as c
    folder, manifest, _ = cuda_battle_artifacts
    torch.empty(1, device='cuda')
    driver = c.CDLL('libcuda.so.1')
    driver.cuCtxGetLimit.argtypes = [c.POINTER(c.c_size_t), c.c_int]
    def stack_limit():
        value = c.c_size_t()
        assert driver.cuCtxGetLimit(c.byref(value), 0) == 0
        return value.value
    original = stack_limit()
    def make():
        return CudaBattleArena(folder/'deck.cubin', manifest['sha256']['deck.cubin'], 1,
            card_count=_nosl_compact_lut()['layout']['card_count'])
    # Simulate unused allocator storage after a full PPO update. A cold arena
    # succeeding says nothing about the update -> next collection boundary.
    cached_bytes = min(12*1024**3, torch.cuda.mem_get_info()[0]//2)
    temporary = torch.empty(cached_bytes, dtype=torch.uint8, device='cuda')
    del temporary
    reserved_before = torch.cuda.memory_reserved()
    a, b = make(), make()
    try:
        assert torch.cuda.memory_reserved() < reserved_before
        assert stack_limit() >= 80 * 1024
        a.close()
        assert stack_limit() >= 80 * 1024
        b.close()
        assert stack_limit() == original
        gc.collect()
        torch.cuda.empty_cache()
        with_again = make()
        with_again.close()
        assert stack_limit() == original
    finally:
        a.close()
        b.close()


def test_resident_host_snapshot_does_not_cap_generated_piles(complete_combat_factory,
                                                            cuda_battle_artifacts):
    from tests.bridge_fixtures import native_card, native_card_state
    _, _, bridge = cuda_battle_artifacts
    env = complete_combat_factory(['STRIKE_RED', 'DEFEND_RED'], seed=42,
                                  deck_latent_context=True)
    try:
        piles = {zone: [native_card('DAZED', 1 + index * 333 + j, base_cost=-2, cost=-2)
                        for j in range(333)]
                 for index, zone in enumerate(('draw', 'discard', 'exhaust'))}
        env.thick.bc.set_card_state(native_card_state(hand=[native_card('STRIKE_RED', 0)],
            next_uid=1000, **piles))
        before = env.thick.bc.nosl_future_key()
        snapshot = bridge.snapshot(env.thick.bc)
        assert snapshot['max_vector'] == 333
        assert len(snapshot['words']) > 8192
        assert snapshot['portable'] == (getattr(bridge, 'transport_version', 1) == 2)
        assert env.thick.bc.nosl_future_key() == before
        again = bridge.snapshot(env.thick.bc)
        np.testing.assert_array_equal(snapshot['words'], again['words'])
    finally:
        env.close()


def test_resident_variable_transport_preserves_large_piles_and_forks(complete_combat_factory,
                                                                   cuda_battle_artifacts):
    from tests.bridge_fixtures import native_card, native_card_state
    folder, manifest, bridge = cuda_battle_artifacts
    assert bridge.transport_version == 2, 'test requires the explicit dynamic transport build'
    env = complete_combat_factory(['STRIKE_RED', 'DEFEND_RED'], seed=42,
                                  deck_latent_context=True)
    arena = None
    branch = None
    try:
        piles = {zone: [native_card('DAZED', 1 + index * 333 + j, base_cost=-2, cost=-2)
                        for j in range(333)]
                 for index, zone in enumerate(('draw', 'discard', 'exhaust'))}
        env.thick.bc.set_card_state(native_card_state(hand=[native_card('STRIKE_RED', 0)],
            next_uid=1000, **piles))
        env._invalidate_backend_cache()
        root = bridge.snapshot(env.thick.bc)
        arena = CudaBattleArena(folder/'deck.cubin', manifest['sha256']['deck.cubin'], 2,
            card_count=_nosl_compact_lut()['layout']['card_count'], graph_capacity=1)
        arena.load([root], [0])
        arena.fork([0], [1], [31])
        branch = env.fork_run(worldline=31)
        branch.thick.bc.nosl_redeterminize(31)
        branch._invalidate_backend_cache()
        bc = branch.thick.bc
        for turn in range(8):
            arena.observe(1)
            raw = arena.observe(2)
            words, metadata = arena.snapshot(2)
            reference = bridge.snapshot(bc)
            np.testing.assert_array_equal(words[0, :len(root['words'])], root['words'])
            np.testing.assert_array_equal(words[1, :metadata[1, 0]], reference['words'])
            if branch._at_combat_exit_anchor() or branch.thick.is_over:
                break
            mask = branch.action_masks()[branch._policy_offset:]
            np.testing.assert_array_equal(raw['mask'][1].cpu(), mask)
            action = int(np.flatnonzero(mask)[0])
            branch.step(action + branch._policy_offset)
            arena.apply([-1, action])
            if turn == 2:
                arena.clear_graphs()
                gc.collect()
                torch.cuda.empty_cache()
        # Replacing a heap-backed imported state must release its old piles.
        arena.load([root], [0])
        words, _ = arena.snapshot(1)
        np.testing.assert_array_equal(words[0], root['words'])
    finally:
        if arena is not None:
            arena.close()
        if branch is not None:
            branch.close()
        env.close()


def test_resident_fused_fp64_sampling_preserves_masked_actions_and_padding():
    from env.search.resident_step_kernels import sample_fp64
    from rl.search.resident_deck import ResidentDeckScorer
    torch.manual_seed(20261111)
    n, actions = 2048, 198
    logits = torch.randn(n, actions, device='cuda') * 7
    mask = torch.rand(n+3, actions, device='cuda') > .85
    mask[:, 0] = True
    rows = torch.randperm(n, device='cuda')
    offset = torch.tensor(3, device='cuda', dtype=torch.int64)
    uniforms = torch.rand(n, device='cuda', dtype=torch.float64)
    expected = ResidentDeckScorer._sample(logits, mask[rows+offset], uniforms)
    result = torch.empty(n, device='cuda', dtype=torch.int32)
    status = torch.zeros(10, device='cuda', dtype=torch.int32)
    status[0] = n-9
    sample_fp64[(n,)](logits, mask, rows, offset, uniforms, result, status, actions, 256, num_warps=4)
    torch.testing.assert_close(result[:n-9], expected[:n-9], rtol=0, atol=0)
    assert torch.count_nonzero(result[n-9:]) == 0
    status[7] = 1
    sample_fp64[(n,)](logits, mask, rows, offset, uniforms, result, status, actions, 256, num_warps=4)
    assert torch.count_nonzero(result) == 0


def test_resident_frame_preserves_separate_worker_random_streams(complete_combat_factory, cuda_battle_artifacts, monkeypatch):
    from env.search.resident_chains import root_payload, resident_anchor_search
    from rl.search.resident_deck import ResidentDeckScorer
    from rl.policies.deck_latent import DeckLatentPolicy
    roots = [complete_combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH', 'SHRUG_IT_OFF'],
        seed=s, deck_latent_context=True, card_language=True) for s in (42, 91)]
    torch.manual_seed(20261111)
    policy = DeckLatentPolicy(roots[0].observation_space, roots[0].action_space, lambda _: 5e-5,
        d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1',
        deck_combat_encoder='transformer64x2').cuda().eval()
    folder, manifest, bridge = cuda_battle_artifacts
    payloads = [root_payload(r, bridge) for r in roots]
    snapshots = [r.export_training_state() for r in roots]
    arena = CudaBattleArena(folder/'deck.cubin', manifest['sha256']['deck.cubin'], 258,
        card_count=_nosl_compact_lut()['layout']['card_count'])
    score = ResidentDeckScorer(policy)
    try:
        serial, work = [], []
        for root, payload in zip(roots, payloads):
            score.roots([root._obs()])
            # Even repeated base seeds in different real steps restart their
            # worker stream; batching must never concatenate these generators.
            values, counts = resident_anchor_search([payload], [64], score, arena,
                seed=20261111, groups=[0], max_depth=32, wave_capacity=256)
            serial.extend(values); work.append(counts)
        score.close()
        from rl.search.resident_step import ResidentStepPipeline
        monkeypatch.setattr(ResidentStepPipeline, 'frame_capacity', 32)
        score = ResidentDeckScorer(policy)
        score.roots([r._obs() for r in roots])
        seed = int(np.random.SeedSequence([20261111, 0]).generate_state(1, dtype=np.uint64)[0])
        merged, counts = resident_anchor_search(payloads, [64, 64], score, arena, seed=0,
            groups=[(0, 0), (1, 0)], group_seeds={(0, 0): seed, (1, 0): seed},
            max_depth=32, wave_capacity=256)
        for a, b in zip(serial, merged):
            for key in ('actions', 'visits'):
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_allclose(a['values'], b['values'], rtol=3e-5, atol=3e-6)
        for key in ('chains', 'atomic_steps', 'policy_rows', 'forced_rows', 'anchor_rows', 'cutoff_rows', 'terminal_rows'):
            assert counts[key] == sum(w[key] for w in work)
        assert score.step_pipeline.stats['replays'] > score.step_pipeline.stats['host_reads']
        assert snapshots == [r.export_training_state() for r in roots]
    finally:
        score.close(); arena.close()


def test_resident_fused_blocks_preserve_fp32_outputs_across_shapes():
    from rl.policies.entity_action_playout import EntityBlock
    from rl.policies.resident_transformer import frozen_block
    torch.manual_seed(20261111)
    block = EntityBlock(64, 4).cuda().eval()
    with torch.no_grad():
        for batch, length in ((3, 32), (64, 40), (3, 128), (3, 32)):
            x = torch.randn(batch, length, 64, device='cuda')
            present = torch.rand(batch, length, device='cuda') > .3
            present[:, 0] = True
            expected = block(x, present)
            actual = frozen_block(block, x, present)
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_resident_ragged_blocks_preserve_selected_queries_and_empty_padding():
    from rl.policies.entity_action_playout import EntityBlock
    from rl.policies.resident_ragged import frozen_ragged_blocks
    torch.manual_seed(20261111)
    blocks = torch.nn.ModuleList([EntityBlock(64, 4), EntityBlock(64, 4)]).cuda().eval()
    with torch.no_grad():
        for batch, length, count in ((3, 32, 3), (16, 128, 9), (3, 40, 0), (3, 32, 2)):
            x = torch.randn(batch, length, 64, device='cuda')
            present = torch.rand(batch, length, device='cuda') > .6
            present[:, 0] = True
            queries = torch.rand(batch, length, device='cuda') > .7
            queries[:, 0] = True
            expected = blocks[1](blocks[0](x, present), present)
            actual = frozen_ragged_blocks(blocks, x.clone(), present, queries,
                torch.tensor([count], device='cuda', dtype=torch.int32))
            selected = queries & (torch.arange(batch, device='cuda') < count)[:, None]
            torch.testing.assert_close(actual[selected], expected[selected], rtol=3e-5, atol=3e-6)
            assert torch.count_nonzero(actual[~selected]) == 0
            norm = torch.nn.LayerNorm(64).cuda().eval()
            norm.weight.uniform_(.5, 1.5)
            norm.bias.uniform_(-.1, .1)
            normalized = frozen_ragged_blocks(blocks, x, present, queries,
                torch.tensor([count], device='cuda', dtype=torch.int32), norm=norm)
            torch.testing.assert_close(normalized, norm(actual), rtol=3e-5, atol=3e-6)


def test_resident_indexed_piles_match_full_histograms_with_holes():
    from rl.policies.resident_ragged import frozen_indexed_piles, frozen_pile_summary
    torch.manual_seed(20261111)
    cards, start, physical = 73, 11, 19
    scalars = torch.randint(0, 333, (physical, start+3*cards+7), device='cuda').float()
    upgraded = torch.randint(0, 128, (physical, 6, cards), device='cuda').float()
    rows = torch.tensor([17, 3, 11, 0, 9], device='cuda')
    base = torch.randn(cards, 64, device='cuda')*.01
    delta = torch.randn_like(base)*.01; magnitude = torch.randn(64, device='cuda')*.01
    with torch.no_grad():
        counts = scalars[rows, start:start+3*cards].reshape(-1, 3, cards)
        expected = frozen_pile_summary(counts, upgraded[rows, :3], upgraded[rows, 3:]/10,
                                       base, delta, magnitude)
        for active in (5, 3, 0):
            actual = frozen_indexed_piles(scalars, upgraded, rows, torch.tensor(active, device='cuda'),
                                         base, delta, magnitude, start, 10.)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got[:active], want[:active], rtol=3e-5, atol=3e-6)
                assert torch.count_nonzero(got[active:]) == 0


@pytest.mark.timeout(180)
def test_resident_compact_features_preserve_slots_queries_and_retry(complete_combat_factory, monkeypatch):
    from rl.policies.deck_latent import DeckLatentPolicy
    from rl.search.resident_deck import ResidentDeckScorer
    root = complete_combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH', 'WARCRY'], seed=42,
        deck_latent_context=True, card_language=True)
    torch.manual_seed(20261111)
    policy = DeckLatentPolicy(root.observation_space, root.action_space, lambda _: 5e-5,
        d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1',
        deck_combat_encoder='transformer64x2').cuda().eval()
    scorer = ResidentDeckScorer(policy)
    net = policy.core.combat
    try:
        scorer.roots([root._obs()])
        for batch, width, active in ((1, 1, 1), (33, 16, 17), (17, 96, 17), (33, 1, 0)):
            obs = {k: torch.as_tensor(v, device='cuda')[None].repeat(batch, *([1]*v.ndim))
                   for k, v in root._obs().items()}
            obs['select_native_indices'].zero_();obs['select_card_ids'].zero_()
            obs['combat_legal'][:, 91:187] = 0
            # Interior holes and a legal PAD-ID/handle-only query are distinct.
            obs['select_native_indices'][:, width-1] = 9
            obs['combat_legal'][:, 91+width-1] = 1
            obs['select_card_ids'][:, 0] = obs['hand_ids'][:, 0]
            obs['select_card_upgrades'][:, 0] = 1000.
            obs['select_card_upgrades'][:, width-1] = -1.
            obs['hand_upgrades'][:, 0] = 1000.
            obs['hand_ids'][:, 3] = 0
            obs['_combat_context'] = scorer.context.combat.expand(batch, -1)
            obs['_choice_width'] = width
            with torch.no_grad():
                obs['_combat_summary'] = net.summaries.rebuild(obs)
                reference, entities = net.encode(obs)
                resident = dict(obs, _resident_rows=torch.tensor(active, device='cuda'),
                    _resident_card_projection=(scorer.projected_cards, scorer.card_upgrade),
                    _resident_monster_projection=scorer.monster_tables)
                status = torch.zeros(8, dtype=torch.int32, device='cuda')
                resident['_resident_workspace'] = (1, 1, status)
                # The combined route must not silently rebuild dense entities.
                def dense_forbidden(*args, **kwargs):
                    raise AssertionError('resident path rebuilt dense projected entities')
                with monkeypatch.context() as patch:
                    patch.setattr(net, '_dynamic_inputs', dense_forbidden)
                    failed, _ = net.encode(resident)
                    needed = status.cpu().tolist()
                    assert bool(needed[7]) == bool(active)
                    assert torch.isfinite(failed).all()
                    resident['_resident_workspace'] = (max(1, needed[5]), max(1, needed[6]), status)
                    actual, blocks = net.encode(resident)
                assert status[7].item() == 0
                torch.testing.assert_close(actual[:active], reference[:active], rtol=3e-5, atol=3e-6)
                for key, mask in (('hand', obs['hand_ids'] != 0), ('monster', obs['monster_id'] != 0),
                                  ('potion', obs['potions'] != 0)):
                    mask = mask[:active]
                    torch.testing.assert_close(blocks[key][:active][mask], entities[key][:active][mask],
                                               rtol=3e-5, atol=3e-6)
                required = (obs['select_native_indices'] > 0) | obs['combat_legal'][:, 91:187].bool()
                mask = required[:active, :width]
                torch.testing.assert_close(blocks['select'][:active][mask],
                    entities['select'][:active, :width][mask], rtol=3e-5, atol=3e-6)
                from rl.policies.resident_ragged import frozen_selection_pointer
                state, compact = net.encode(dict(resident, _resident_project_selection=True))
                assert compact['select'].shape[1] == 0
                pointer = frozen_selection_pointer(net,state,compact['_resident_selection'])
                expected_pointer = net._pointer(actual,blocks['select'])
                torch.testing.assert_close(pointer[:active],expected_pointer[:active],rtol=3e-5,atol=3e-6)
            gc.collect();torch.cuda.empty_cache()
    finally:
        scorer.close()


def test_resident_ragged_capacity_retry_preserves_selected_outputs():
    from rl.policies.entity_action_playout import EntityBlock
    from rl.policies.resident_ragged import frozen_ragged_blocks
    torch.manual_seed(20261111)
    blocks = torch.nn.ModuleList([EntityBlock(64, 4), EntityBlock(64, 4)]).cuda().eval()
    with torch.no_grad():
        x = torch.randn(17, 128, 64, device='cuda')
        present = torch.rand(17, 128, device='cuda') > .6
        present[:, 0] = True
        queries = torch.rand_like(present, dtype=torch.float32) > .7
        count = torch.tensor([11], device='cuda', dtype=torch.int32)
        status = torch.zeros(10, device='cuda', dtype=torch.int32)
        before = blocks[1](blocks[0](x, present), present)
        failed = frozen_ragged_blocks(blocks, x.clone(), present, queries, count, (1, 1, status))
        nt, nq, overflow = status[5:8].cpu().tolist()
        assert overflow == 1 and nt > nq > 1 and torch.count_nonzero(failed) == 0
        actual = frozen_ragged_blocks(blocks, x.clone(), present, queries, count, (nt, nq, status))
        assert status[7] == 0
        selected = queries & (torch.arange(17, device='cuda') < 11)[:, None]
        torch.testing.assert_close(actual[selected], before[selected], rtol=3e-5, atol=3e-6)
        assert torch.count_nonzero(actual[~selected]) == 0


@pytest.mark.parametrize('graph_capacity', [0, 1])
def test_resident_forks_atomic_masks_rng_and_graph_lifetime(complete_combat_factory, cuda_battle_artifacts,
                                                         graph_capacity):
    folder, manifest, bridge = cuda_battle_artifacts
    roots = [complete_combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH', 'TRUE_GRIT', 'WARCRY', 'HEADBUTT',
        'BATTLE_TRANCE', 'ARMAMENTS', 'OFFERING', 'SEEING_RED'], seed=seed,
        deck_latent_context=True) for seed in (42, 91)]
    initial = [bridge.snapshot(e.thick.bc) for e in roots]
    arena = CudaBattleArena(folder / 'deck.cubin', manifest['sha256']['deck.cubin'], 6,
        card_count=_nosl_compact_lut()['layout']['card_count'], graph_capacity=graph_capacity)
    branches = []
    try:
        with pytest.raises(ValueError, match='queues cannot'):
            arena.load([dict(initial[0], queue_size=1, portable=False)], [0])
        arena.load(initial, [0, 0])
        sources, destinations, worlds = [0, 0, 1, 1], [2, 3, 4, 5], [31, 31, 53, 67]
        for source, world in zip(sources, worlds, strict=True):
            branch = roots[source].fork_run(worldline=world)
            branch.thick.bc.nosl_redeterminize(world)
            branch._invalidate_backend_cache()
            branches.append(branch)
        arena.fork(sources, destinations, worlds)
        envs = roots + branches
        roots_obs = [env._obs() for env in envs]
        root_tensors = {k: torch.as_tensor(np.stack([o[k] for o in roots_obs]), device='cuda')
                        for k in roots_obs[0]}
        decoder = CudaObservationDecoder(roots[0].observation_space)
        battle_refs = [e.thick.bc for e in envs]
        rng = np.random.default_rng(20261111)
        atomic_steps, queued_rows, skipped_features = 0, 0, 0
        for turn in range(200):
            # Alternate shapes against capacity one, including a live queued
            # continuation. Graph retirement must not destroy simulator state.
            arena.observe(2)
            raw = arena.observe(6)
            if 'observe_live' in arena.functions:
                expected_raw = {k: v.clone() for k, v in raw.items()}
                expected_bits = arena.bits[:6].clone()
                for key, value in raw.items():
                    if key != 'prefix':
                        value.fill_(99)
                arena._launch('observe_live', arena.observe_args+[
                    torch.ones(6, device='cuda', dtype=torch.int32),
                    torch.zeros((), device='cuda', dtype=torch.int64)], 6)
                for key in raw:
                    torch.testing.assert_close(raw[key], expected_raw[key], rtol=0, atol=0)
                torch.testing.assert_close(arena.bits[:6], expected_bits, rtol=0, atol=0)
            if 'observe_search' in arena.functions:
                for at_cutoff in (False, True):
                    for key, value in raw.items():
                        if key not in ('prefix', 'mask', 'outcome'):
                            value.fill_(99)
                    arena._launch('observe_search', arena.observe_args+[
                        arena._upload(np.arange(6)), arena._upload([6]),
                        arena._upload([0], np.int64), arena._upload([200 if at_cutoff else 0]*6),
                        arena._upload([200])], 6)
                    need = (expected_raw['outcome'] == 0) & (
                        at_cutoff | (expected_raw['mask'].sum(1) > 1))
                    skipped_features += int((~need).sum())
                    for key, value in raw.items():
                        if key in ('prefix', 'mask', 'outcome'):
                            torch.testing.assert_close(value, expected_raw[key], rtol=0, atol=0)
                        else:
                            torch.testing.assert_close(value[need], expected_raw[key][need], rtol=0, atol=0)
                            assert torch.all(value[~need] == 99)
                for key, value in raw.items():
                    value.copy_(expected_raw[key])
                torch.testing.assert_close(arena.bits[:6], expected_bits, rtol=0, atol=0)
            if 'observe_indexed' in arena.functions:
                # Nonzero offset and holes: indexed observation must neither
                # reorder fields nor touch roots and excluded sibling slots.
                selected_rows = torch.tensor([2, 4, 5], device='cuda')
                for key, value in raw.items():
                    if key != 'prefix':
                        value[selected_rows] = 99
                arena._launch('observe_indexed', arena.observe_args+[
                    arena._upload([0, 2, 3]), arena._upload([3]),
                    arena._upload([2], np.int64)], 6, dispatch_n=3)
                for key in raw:
                    torch.testing.assert_close(raw[key], expected_raw[key], rtol=0, atol=0)
            decoded = {k: v.cpu().numpy() for k, v in decoder(raw, root_tensors,
                torch.arange(6, device='cuda')).items()}
            anchored = decoder.anchor(raw, root_tensors, torch.arange(6, device='cuda'))
            for key, value in anchored.items():
                expected = decoded[key][:, :2] if key == 'player_scalars' else decoded[key]
                np.testing.assert_array_equal(value.cpu().numpy(), expected)
            observed = {key: value.cpu().numpy() for key, value in raw.items()}
            words, metadata = arena.snapshot(6)
            for i, bc in enumerate(battle_refs):
                state = bridge.snapshot(bc)
                np.testing.assert_array_equal(metadata[i], [len(state['words']), state['queue_size'],
                    state['max_vector'], state['input_state'], state['outcome']])
                np.testing.assert_array_equal(words[i, :len(state['words'])], state['words'])
                queued_rows += state['queue_size'] > 0
            for i, root in enumerate(initial):
                np.testing.assert_array_equal(words[i, :len(root['words'])], root['words'])
            if turn == 0:
                np.testing.assert_array_equal(words[2, :metadata[2, 0]], words[3, :metadata[3, 0]])
            active = [i for i in range(2, 6) if not envs[i].thick.is_over
                      and not envs[i]._at_combat_exit_anchor()]
            if not active:
                break
            packed = sts.pack_neural_roots([battle_refs[i] for i in active],
                [sum(1 << k for k in (envs[i]._multi or ())) for i in active], None)
            np.testing.assert_array_equal(observed['categorical'][active], packed['categorical'])
            np.testing.assert_array_equal(observed['scalars'][active], packed['scalars'])
            selected = np.full(6, -1, np.int32)
            for i in active:
                env = envs[i]
                reference = env._obs()
                for key, value in decoded.items():
                    np.testing.assert_allclose(value[i], reference[key], rtol=1e-6, atol=1e-7,
                                               err_msg=f'{turn=} {i=} {key=}')
                mask = env.action_masks()[env._policy_offset:]
                np.testing.assert_array_equal(observed['mask'][i], mask)
                assert observed['prefix'][i] == sum(1 << k for k in (env._multi or ()))
                action = int(rng.choice(np.flatnonzero(mask)))
                selected[i] = action
                env.step(action + env._policy_offset)
                atomic_steps += 1
            if turn % 2 and 'apply_compact' in arena.functions:
                arena.selected[:6].copy_(arena._upload(selected))
                arena._launch('apply_compact', arena.apply_args+[
                    arena._upload(np.flatnonzero(selected >= 0)-2), arena._upload([len(active)]),
                    arena._upload([2], np.int64)], 6, dispatch_n=len(active))
            else:
                arena.apply(selected)
            if turn == 3:
                arena.clear_graphs()
                gc.collect()
                temporary = torch.empty(8 * 1024**2, device='cuda')
                del temporary
                torch.cuda.empty_cache()
        else:
            pytest.fail('resident parity exceeded the fixed 200-action depth')
        assert atomic_steps > 10 and queued_rows > 0
        if 'observe_search' in arena.functions:
            assert skipped_features > 0
        if graph_capacity:
            assert arena.counts['graph_evictions'] > 2 and arena.counts['graph_replays'] > 10
        else:
            assert arena.counts['graph_replays'] == 0
    finally:
        arena.close()
        for branch in branches:
            branch.close()
    with pytest.raises(RuntimeError, match='closed'):
        arena.observe(2)


def test_resident_anchor_batches_bound_projection_and_keep_exact_cache(
        complete_combat_factory, cuda_battle_artifacts, monkeypatch):
    from env.search.resident_chains import root_payload
    from rl.search.resident_deck import ResidentDeckScorer
    from rl.policies.deck_latent import DeckLatentPolicy
    root = complete_combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH'], seed=42,
        deck_latent_context=True, card_language=True)
    torch.manual_seed(20261111)
    policy = DeckLatentPolicy(root.observation_space, root.action_space, lambda _: 5e-5,
        d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1',
        deck_combat_encoder='transformer64x2').cuda().eval()
    folder, manifest, bridge = cuda_battle_artifacts
    payload = root_payload(root, bridge)
    arena = CudaBattleArena(folder/'deck.cubin', manifest['sha256']['deck.cubin'], 1,
        card_count=_nosl_compact_lut()['layout']['card_count'])
    scorer = ResidentDeckScorer(policy)
    before = root.export_training_state()
    try:
        arena.load([payload['snapshot']], [payload['prefix']])
        raw = {k: v.repeat(257, *([1] * (v.ndim-1))) for k, v in arena.observe(1).items()}
        scorer.roots([root._obs()])
        owners = torch.zeros(257, device='cuda', dtype=torch.long)
        rows = np.arange(256, -1, -1)
        with torch.no_grad():
            obs = scorer.decoder.anchor(raw, scorer.root, owners)
            relics = policy.core.combat.summaries.relic_terms(obs['relics'], obs['relic_counters']).sum(1)
            expected = policy.core.anchor_with_context(obs, scorer.context.anchor[owners], relics).flatten()
        lengths = []
        original = scorer._forward
        def bounded(batch, width, anchor):
            lengths.append(len(batch['context']))
            assert lengths[-1] <= 17
            return original(batch, width, anchor)
        monkeypatch.setattr(scorer, 'anchor_batch_size', 17)
        monkeypatch.setattr(scorer, '_forward', bounded)
        rng = torch.cuda.get_rng_state().clone()
        _, actual = scorer.score(raw, owners, rows, anchor=True)
        torch.testing.assert_close(actual, expected[rows], rtol=3e-5, atol=3e-6)
        misses = scorer.anchor_cache.stats['misses']
        assert misses == 1 and len(lengths) == 16 and lengths[-1] == 2
        # A changed chunk boundary and temporary allocator release retain exact
        # values/cache identity without forwarding previously scored anchors.
        torch.cuda.empty_cache()
        def unexpected_projection(*args, **kwargs):
            raise AssertionError('an exact cache hit rebuilt relic counter features')
        counter_terms = policy.core.combat.summaries.relic_counter_terms
        monkeypatch.setattr(policy.core.combat.summaries, 'relic_counter_terms', unexpected_projection)
        _, replay = scorer.score(raw, owners, rows[:33], anchor=True)
        torch.testing.assert_close(replay, actual[:33], rtol=0, atol=0)
        assert scorer.anchor_cache.stats['misses'] == misses
        assert torch.equal(rng, torch.cuda.get_rng_state())
        assert before == root.export_training_state()
        # Re-admission of byte-identical context remains a hit. A different
        # context at the same root slot must miss and use that new GPU row.
        from rl.policies.deck_latent import SearchContext
        original_context = scorer.context
        monkeypatch.setattr(policy.core.combat.summaries, 'relic_counter_terms', counter_terms)
        scorer.roots([root._obs()], original_context)
        monkeypatch.setattr(policy.core.combat.summaries, 'relic_counter_terms', unexpected_projection)
        _, same = scorer.score(raw, owners, rows[:1], anchor=True)
        torch.testing.assert_close(same, actual[:1], rtol=0, atol=0)
        assert scorer.anchor_cache.stats['misses'] == misses
        monkeypatch.setattr(policy.core.combat.summaries, 'relic_counter_terms', counter_terms)
        changed_context = SearchContext(original_context.combat, original_context.anchor + .25)
        scorer.roots([root._obs()], changed_context)
        _, changed = scorer.score(raw, owners, rows[:1], anchor=True)
        with torch.no_grad():
            expected_changed = policy.core.anchor_with_context(
                {k:v[:1] for k,v in obs.items()}, changed_context.anchor, relics[:1]).flatten()
        torch.testing.assert_close(changed, expected_changed, rtol=3e-5, atol=3e-6)
        assert scorer.anchor_cache.stats['misses'] == misses + 1
        # Force intern eviction while the value cache still has the old ID.
        # Returning to old bytes gets a fresh ID, never a stale value alias.
        scorer.anchor_context_limit = 256 + original_context.anchor[0].numel()*4
        scorer.roots([root._obs()], SearchContext(original_context.combat, original_context.anchor + .5))
        previous_id = int(scorer.anchor_context_ids[0])
        scorer.roots([root._obs()], original_context)
        assert int(scorer.anchor_context_ids[0]) > previous_id
        _, restored = scorer.score(raw, owners, rows[:1], anchor=True)
        torch.testing.assert_close(restored, actual[:1], rtol=3e-5, atol=3e-6)
        assert scorer.anchor_context_bytes <= scorer.anchor_context_limit
    finally:
        scorer.close()
        arena.close()


@pytest.mark.timeout(180)
def test_resident_paired_search_matches_native_and_preserves_root(complete_combat_factory, cuda_battle_artifacts):
    from env.search.anchored_chains import paired_anchor_search
    from env.search.resident_chains import root_payload, resident_anchor_search
    from rl.search.resident_deck import ResidentDeckScorer
    from rl.search.deck_latent import DeckLatentScorer
    from rl.policies.deck_latent import DeckLatentPolicy
    root = complete_combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH', 'TRUE_GRIT', 'WARCRY',
        'HEADBUTT', 'BATTLE_TRANCE', 'ARMAMENTS', 'OFFERING', 'SEEING_RED'], seed=42,
        deck_latent_context=True, card_language=True)
    torch.manual_seed(20261111)
    policy = DeckLatentPolicy(root.observation_space, root.action_space, lambda _: 5e-5,
        d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1',
        deck_combat_encoder='transformer64x2').cuda().eval()
    folder, manifest, bridge = cuda_battle_artifacts
    payload = root_payload(root, bridge)
    before = root.export_training_state()
    rng_before = torch.cuda.get_rng_state().clone()
    expected, work = paired_anchor_search([root], [64], DeckLatentScorer(policy, [root._obs()]),
        seed=20261111, max_depth=200, max_active=256, timeout_seconds=90)
    arena = CudaBattleArena(folder/'deck.cubin', manifest['sha256']['deck.cubin'], 257,
                           card_count=_nosl_compact_lut()['layout']['card_count'])
    score = ResidentDeckScorer(policy, graph_capacity=2)
    try:
        for capacity in (256, 32):
            score.roots([root._obs()])
            actual, stats = resident_anchor_search([payload], [64], score, arena,
                seed=20261111, max_depth=200, wave_capacity=capacity, timeout_seconds=90)
            for key in ('chains', 'atomic_steps', 'policy_rows', 'forced_rows', 'anchor_rows', 'cutoff_rows', 'terminal_rows'):
                assert stats[key] == work[key], (key, stats[key], work[key])
            for key in ('actions', 'visits'):
                np.testing.assert_array_equal(actual[0][key], expected[0][key])
            np.testing.assert_allclose(actual[0]['values'], expected[0]['values'], rtol=3e-5, atol=3e-6)
            if score.step_pipeline is not None:
                assert score.step_pipeline.stats['replays'] > 0
                score.step_pipeline.capacity = 1
                score.step_pipeline.close()
            arena.clear_graphs()
            gc.collect()
            temporary = torch.empty(8*1024**2, device='cuda')
            del temporary
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        assert before == root.export_training_state()
        assert torch.equal(rng_before, torch.cuda.get_rng_state())
        # A smaller next wave reuses the same padded graph, while its unused
        # observation rows have no validity contract. Never gather stale rows.
        score.roots([root._obs()])
        arena.load([payload['snapshot']] * 33, [payload['prefix']] * 33)
        arena.observe(33)
        owner = torch.zeros(32, dtype=torch.long, device='cuda')
        score.bind_wave(arena, owner, 1)
        raw = {k: v[1:] for k, v in arena.observe(33).items()}
        meta = score.metadata(raw, owner)
        uniforms = torch.full((32,), .375, dtype=torch.float64, device='cuda')
        expected = tuple(x.clone() for x in score.score(raw, owner, np.arange(32),
            metadata=meta, uniforms=uniforms))
        arena.categorical[18:33].fill_(-100000)
        owner = owner[:17]
        score.bind_wave(arena, owner, 1)
        small = {k: v[:17] for k, v in raw.items()}
        meta = score.metadata(small, owner)
        actual = score.score(small, owner, np.arange(17), metadata=meta, uniforms=uniforms[:17])
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want[:17], rtol=0, atol=0)
        arena.observe(33)
        # Trailing PAD width changes graph identity; interior empty slots must
        # survive. Compare against the complete untrimmed head consumers.
        base = root._obs()
        present = np.flatnonzero(base['relics'])
        assert len(present)
        for slot in (0, 7, 0):
            obs = {k: v.copy() for k, v in base.items()}
            obs['relics'].fill(0); obs['relic_counters'].fill(0)
            obs['relics'][slot] = base['relics'][present[0]]
            obs['relic_counters'][slot] = base['relic_counters'][present[0]]
            score.roots([obs])
            raw = arena.observe(1)
            owners = torch.zeros(1, dtype=torch.long, device='cuda')
            full = score.decoder(raw, {k: torch.as_tensor(v[None], device='cuda')
                                      for k, v in obs.items()}, owners)
            with torch.no_grad():
                from unittest.mock import patch
                net = policy.core.combat
                head_input = dict(full, _combat_context=score.context.combat)
                complete_projection = net._dynamic_inputs
                with patch.object(net, '_dynamic_inputs',
                                  lambda obs, table=None: complete_projection(head_input, table)):
                    old_global, old_entities = net.encode(head_input)
                new_global, new_entities = net.encode(head_input)
                torch.testing.assert_close(new_global, old_global, rtol=3e-5, atol=3e-6)
                for key in old_entities:
                    torch.testing.assert_close(new_entities[key], old_entities[key], rtol=3e-5, atol=3e-6)
                expected = policy.core.combat_with_context(full, score.context)
                actual = score.score(raw, owners, [0])
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(got.flatten(), want.flatten(), rtol=3e-5, atol=3e-6)
        with torch.no_grad():
            next(policy.core.parameters()).add_(0.)
        with pytest.raises(RuntimeError, match='weights changed'):
            score.score({}, None, [])
    finally:
        score.close()
        arena.close()
