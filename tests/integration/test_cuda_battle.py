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
        atomic_steps, queued_rows = 0, 0
        for turn in range(200):
            # Alternate shapes against capacity one, including a live queued
            # continuation. Graph retirement must not destroy simulator state.
            arena.observe(2)
            raw = arena.observe(6)
            decoded = {k: v.cpu().numpy() for k, v in decoder(raw, root_tensors,
                torch.arange(6, device='cuda')).items()}
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
            arena.clear_graphs()
            gc.collect()
            temporary = torch.empty(8*1024**2, device='cuda')
            del temporary
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        assert before == root.export_training_state()
        assert torch.equal(rng_before, torch.cuda.get_rng_state())
    finally:
        score.close()
        arena.close()
