"""COMBAT06-ROLLOUT: actual root observations retain the complete public schema."""
import copy
import numpy as np
import pytest
import slaythespire as sts

from env.core.native_observation import NativeRootObservationBatch
from env.core.run_env import SlayTheSpireEnv

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


def assert_roots(encoder, envs):
    expected = [env._obs() for env in envs]
    actual = encoder.encode(envs)
    assert set(actual) == set(expected[0])
    for row, reference in enumerate(expected):
        for key, value in reference.items():
            assert actual[key].dtype == value.dtype
            np.testing.assert_array_equal(actual[key][row], value, err_msg=f'{row}/{key}')
            assert actual[key][row].tobytes() == value.tobytes(), f'{row}/{key} bytes'
    return actual


@pytest.mark.parametrize('language', [False, True])
def test_root_batch_all_fields_mutable_globals_and_ownership(combat_factory, language):
    envs = [combat_factory(['STRIKE_RED']*5+['DEFEND_RED']*4+['BASH'],
                          encounter, seed=42+i, card_language=language,
                          combat_action_contract='autoregressive-v2')
            for i, encounter in enumerate(('Two Louse', 'Automaton', 'Collector'))]
    encoder = NativeRootObservationBatch(envs[0])
    before = copy.deepcopy(assert_roots(encoder, envs))
    for i, env in enumerate(envs):
        env.thick.gc.gold = 101+i
        env.thick.gc.set_deck([('SEARING_BLOW', 130), ('STRIKE_RED', 1)])
        env.thick.bc.apply_custom_state({
            'player': {'hp': 21+i, 'max_hp': 83+i, 'gold': 199+i},
            'hand': [('SEARING_BLOW', 130, 1), ('STRIKE_RED', 1)],
            'draw': [('BASH', 1), ('BASH', 0), ('DEFEND_RED', 2)],
            'discard': [('ANGER', 0)],
        })
        if i == 0:
            env.thick.bc.apply_custom_state({'monsters': [{'hp': 0}, {'hp': 13}]})
        env._invalidate_backend_cache()
    actual = assert_roots(encoder, envs)
    assert not np.array_equal(before['hand_damage'], actual['hand_damage'])
    saved = copy.deepcopy(actual)
    another = encoder.encode(envs)
    for key in actual:
        another[key][...] = 0
        np.testing.assert_array_equal(actual[key], saved[key], err_msg=key)


def test_root_batch_pending_selection_and_nonhand_handles(combat_factory):
    envs=[]
    for card in ('PURITY', 'SECRET_TECHNIQUE'):
        env=combat_factory([card]*5, combat_action_contract='autoregressive-v2', card_language=True)
        env.thick.bc.apply_custom_state({'hand':[card, 'STRIKE_RED', 'BASH'],
                                        'draw':['STRIKE_RED']*12+['DEFEND_RED','SHRUG_IT_OFF']})
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0))
        env._invalidate_backend_cache();envs.append(env)
    encoder=NativeRootObservationBatch(envs[0]);assert_roots(encoder,envs)
    envs[0]._multi.add(1)
    envs[0]._invalidate_thin_cache()
    assert_roots(encoder,envs)


@pytest.mark.parametrize('hp', [0, 25])
def test_exploder_public_intent_hides_explosion_without_changing_native_damage(combat_factory, hp):
    env = combat_factory(['DEFEND_RED']*10, encounter='Three Shapes', seed=4,
                         combat_action_contract='autoregressive-v2', card_language=True)
    monsters = env.thick.bc.to_dict()['monsters']
    slot = next(i for i, m in enumerate(monsters) if m['id'] == 'EXPLODER')
    patches = [{} for _ in monsters]
    patches[slot] = {'hp': hp, 'move': 'EXPLODER_EXPLODE'}
    env.thick.bc.apply_custom_state({'monsters': patches})
    env._invalidate_backend_cache()
    before = env.thick.bc.to_dict()
    assert before['monsters'][slot]['base_damage'] == 30
    batch = assert_roots(NativeRootObservationBatch(env), [env])
    assert batch['monster_intent_cat'][0, slot] == 0
    np.testing.assert_array_equal(batch['monster_scalars'][0, slot, 4:6], [0, 0])
    assert env.thick.bc.to_dict() == before


def test_dead_monster_negative_power_keeps_canonical_positive_zero(combat_factory):
    env = combat_factory(['DEFEND_RED']*10, encounter='Two Louse', seed=42,
                         combat_action_contract='autoregressive-v2')
    env.thick.bc.apply_custom_state({'monsters': [
        {'hp': 0, 'powers': [('STRENGTH', -1)]}, {}]})
    env._invalidate_backend_cache()
    before = env.thick.bc.to_dict()
    assert dict(before['monsters'][0]['powers'])['Strength'] == -1
    batch = assert_roots(NativeRootObservationBatch(env), [env])
    assert not np.signbit(batch['monster_buffs'][0, 0]).any()
    assert env.thick.bc.to_dict() == before


def test_root_batch_full_run_observations_and_public_context():
    envs=[SlayTheSpireEnv(combat_action_contract='autoregressive-v2',
             card_language=True, global_map_view=True, deck_size_observation=True,
             neow_observation=True, event_option_observation=True, outer_return_context=True,
             combat_value_objective='combat-outcome-means-v1') for _ in range(8)]
    rng=np.random.default_rng(9001000)
    try:
        for i,env in enumerate(envs):env.reset(seed=9001000+i)
        encoder=NativeRootObservationBatch(envs[0]);checked=0
        for _ in range(256):
            for env in envs:
                _,_,done,truncated,_=env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
                if done or truncated:env.reset()
            active=[env for env in envs if env.thick.bc is not None]
            if active:
                assert_roots(encoder,active);checked+=len(active)
        assert checked>1000
    finally:
        for env in envs:env.close()


def test_root_packer_rejects_bad_identity_and_preserves_state(combat_factory):
    env=combat_factory(['STRIKE_RED']*5,combat_action_contract='autoregressive-v2')
    before=env.thick.bc.to_dict()
    with pytest.raises(ValueError,match='disagree'):
        sts.pack_neural_roots([env.thick.bc],[])
    with pytest.raises(ValueError,match='prefix'):
        sts.pack_neural_roots([env.thick.bc],[1024])
    sts.pack_neural_roots([env.thick.bc],[0])
    assert env.thick.bc.to_dict()==before


def assert_nested_equal(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.fixture
def native_wrapper_factories():
    import gymnasium as gym
    from rl.training.environments import CheckpointObservationCompat, TrainingStateWrapper
    from rl.training.reward import ShapedRewardWrapper

    def factory(index, *, outer_return_context):
        def make():
            base = SlayTheSpireEnv(combat_action_contract='autoregressive-v2',
                 card_language=True, global_map_view=True, deck_size_observation=True,
                 neow_observation=True, event_option_observation=True, outer_return_context=outer_return_context,
                 combat_value_objective='combat-outcome-means-v1')
            expected = gym.spaces.Dict({key: space for key, space in base.observation_space.spaces.items()
                                       if key != 'post_enemy_turn_boundary'})
            env = gym.wrappers.TimeLimit(ShapedRewardWrapper(base), max_episode_steps=41+index)
            return TrainingStateWrapper(CheckpointObservationCompat(env, expected))
        return make
    return lambda count, outer_return_context=True: [
        factory(i, outer_return_context=outer_return_context) for i in range(count)]


def test_vector_wrappers_autoreset_labels_journals_and_restore(monkeypatch, native_wrapper_factories):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.training.native_vec_env import NativeSearchVecEnv
    factories = native_wrapper_factories(8)
    reference, candidate = DummyVecEnv(factories), NativeSearchVecEnv(factories, threads=2)
    rng = np.random.default_rng(9001000)
    try:
        assert candidate.native_root_observations is not None
        pending_batches = []
        indices = {id(env.unwrapped): index for index, env in enumerate(candidate.envs)}
        encode = candidate.native_root_observations.encode
        def record_pending(envs):
            pending_batches.append(tuple(indices[id(env)] for env in envs))
            return encode(envs)
        monkeypatch.setattr(candidate.native_root_observations, 'encode', record_pending)
        for vec in (reference, candidate):
            vec.seed(9001000)
        assert_nested_equal(reference.reset(), candidate.reset())
        for step in range(192):
            masks = reference.env_method('action_masks')
            actions = np.array([rng.choice(np.flatnonzero(mask)) for mask in masks])
            expected = reference.step(actions)
            actual = candidate.step(actions)
            assert_nested_equal(expected, actual)
            assert_nested_equal(reference.env_method('action_masks'), candidate.env_method('action_masks'))
            if step == 91:
                saved = candidate.env_method('export_training_state')
                assert_nested_equal(reference.env_method('export_training_state'), saved)
                for env, state in zip(candidate.envs, saved, strict=True):
                    env.restore_training_state(state)
        assert_nested_equal(reference.env_method('export_training_state'), candidate.env_method('export_training_state'))
        assert all(not env.unwrapped._defer_combat_observation for env in candidate.envs)
        assert any(len(rows) < len(factories) and any(b-a > 1 for a,b in zip(rows, rows[1:]))
                   for rows in pending_batches), 'fixture must mix noncontiguous combat rows with other phases/reset'
    finally:
        reference.close(); candidate.close()


@pytest.mark.parametrize('threads', [1, 4])
def test_small_playout_overlap_matches_serial_transitions_samples_and_restore(native_wrapper_factories, threads):
    from rl.training.native_vec_env import NativeSearchVecEnv
    from rl.algorithms.small_playout import SmallPlayoutLearner
    from rl.policies.small_playout import export_online_search_pair
    from env.core import vocab as V
    pair = export_online_search_pair(SmallPlayoutLearner(seed=19).policy, collect_samples=True)
    pair.update(common_simulations=8, elite_simulations=8, boss_simulations=8, playout_depth=8)
    reference = NativeSearchVecEnv(native_wrapper_factories(4), threads=threads)
    candidate = NativeSearchVecEnv(native_wrapper_factories(4), threads=threads)
    rng = np.random.default_rng(9001000)
    overlap_steps = 0
    try:
        for vec in (reference, candidate):
            vec.seed(9001000)
            vec.configure_small_playout(pair)
        observations = reference.reset()
        assert_nested_equal(observations, candidate.reset())
        for step in range(128):
            masks = reference.env_method('action_masks')
            assert_nested_equal(masks, candidate.env_method('action_masks'))
            actions = np.array([rng.choice(np.flatnonzero(mask)) for mask in masks])
            combat = [i for i in range(4) if observations['phase'][i, 0] == V.PHASE_VOCAB_INDEX['combat']]
            if combat:
                calls = {i: ((), {'state_nonce': 9001000 + step * 4 + i}) for i in combat}
                expected_decisions = reference.env_method_batch('small_playout_decision', calls)
                assert candidate.prepare_small_playout_steps(actions, [i for i in range(4) if i not in combat])
                actual_decisions = candidate.env_method_batch('small_playout_decision', calls)
                assert_nested_equal(expected_decisions, actual_decisions)
                overlap_steps += bool(candidate._prefetched_steps)
                for i, decision in expected_decisions.items():
                    actions[i] = decision['action']
            expected, actual = reference.step(actions), candidate.step(actions)
            assert_nested_equal(expected, actual)
            observations = expected[0]
            assert not candidate._prefetched_steps and candidate._small_step_plan is None
            if step == 63:
                snapshots = candidate.env_method('export_training_state')
                assert_nested_equal(reference.env_method('export_training_state'), snapshots)
                for env, state in zip(candidate.envs, snapshots, strict=True):
                    env.restore_training_state(state)
        assert overlap_steps > 10
        assert_nested_equal(reference.take_small_playout_samples(), candidate.take_small_playout_samples())
        assert_nested_equal(reference.env_method('export_training_state'), candidate.env_method('export_training_state'))
        for key, value in reference.small_playout_stats().items():
            if not key.endswith('search_seconds') and not key.startswith('overlap/'):
                np.testing.assert_equal(value, candidate.small_playout_stats()[key])
        assert reference.small_playout_stats()['overlap/prefetched_steps'] == 0
        assert candidate.small_playout_stats()['overlap/prefetched_steps'] > 0
    finally:
        reference.close(); candidate.close()


def test_unknown_observation_wrapper_keeps_serial_path():
    import gymnasium as gym
    from rl.training.native_vec_env import NativeSearchVecEnv

    class InspectObservation(gym.ObservationWrapper):
        def observation(self, observation):
            assert observation and 'phase' in observation
            return observation

    vec = NativeSearchVecEnv([lambda: InspectObservation(
        SlayTheSpireEnv(combat_action_contract='autoregressive-v2'))], threads=1)
    try:
        assert vec.native_root_observations is None
        vec.seed(42); vec.reset()
        for _ in range(32):
            vec.step([int(np.flatnonzero(vec.env_method('action_masks')[0])[0])])
    finally:
        vec.close()


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('omit_value_observations', [False, True])
def test_independent_games_match_recorded_transitions_and_samples(
        native_wrapper_factories, threads, omit_value_observations):
    from contextlib import nullcontext
    from env.search.small_playout import SmallPlayoutStream
    from rl.training.native_vec_env import NativeSearchVecEnv
    from rl.algorithms.small_playout import SmallPlayoutLearner
    from rl.policies.small_playout import export_online_search_pair
    from env.core import vocab as V
    pair = export_online_search_pair(SmallPlayoutLearner(seed=19).policy, collect_samples=True)
    pair.update(common_simulations=8, elite_simulations=8, boss_simulations=8, playout_depth=8)
    factories = native_wrapper_factories(4, outer_return_context=not omit_value_observations)
    reference = NativeSearchVecEnv(factories, threads=threads)
    candidate = NativeSearchVecEnv(factories, threads=1)
    history = [[] for _ in range(4)]
    rng = np.random.default_rng(9001000)
    try:
        for vec in (reference, candidate):
            vec.seed(9001000)
            vec.configure_small_playout(pair)
        obs = reference.reset()
        assert_nested_equal(obs, candidate.reset())
        if not omit_value_observations:
            with pytest.raises(ValueError, match='outer-return'):
                with candidate.omit_unused_combat_value_observations():
                    pass
        for step in range(128):
            masks = reference.env_method('action_masks')
            actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            combat = [i for i in range(4) if obs['phase'][i, 0] == V.PHASE_VOCAB_INDEX['combat']]
            decisions = reference.env_method_batch('small_playout_decision',
                {i: ((), dict(state_nonce=9001000+step*4+i)) for i in combat})
            for i, decision in decisions.items():
                actions[i] = decision['action']
            obs, rewards, dones, infos = reference.step(actions)
            for i in range(4):
                history[i].append((i in combat, int(actions[i]), masks[i],
                    ({k: v[i].copy() for k, v in obs.items()}, rewards[i], dones[i], infos[i])))
        steps = [0]*4
        ready, progressed = set(range(4)), set()
        scope = candidate.omit_unused_combat_value_observations() if omit_value_observations else nullcontext()
        with scope, SmallPlayoutStream(candidate.small_playout_batch, threads=threads) as stream:
            while ready or stream.pending:
                actions, calls = {}, {}
                for i in sorted(ready, reverse=True):
                    combat, action, mask, _ = history[i][steps[i]]
                    assert_nested_equal(candidate.env_method('action_masks', indices=i)[0], mask)
                    if combat:
                        calls[i] = ((), dict(state_nonce=9001000+steps[i]*4+i))
                    else:
                        actions[i] = action
                ready.clear()
                stream.submit(calls)
                if calls:
                    with pytest.raises(RuntimeError, match='outstanding'):
                        stream.submit(calls)
                decisions = stream.receive(timeout=.1 if not actions else 0.)
                actions.update({i: d['action'] for i, d in decisions.items()})
                if not actions:
                    continue
                rows = list(actions)
                for i in rows:
                    assert actions[i] == history[i][steps[i]][1]
                obs, rewards, dones, infos = candidate.step_subset(actions)
                for row, i in enumerate(rows):
                    actual = ({k: v[row] for k, v in obs.items()}, rewards[row], dones[row], infos[row])
                    expected = history[i][steps[i]][3]
                    if omit_value_observations:
                        omitted = {'nosl_value_boundary_observation', 'nosl_value_terminal_observation',
                                   'nosl_value_terminal_max_hp'}
                        assert not omitted.intersection(infos[row])
                        expected = (*expected[:3], {k: v for k, v in expected[3].items() if k not in omitted})
                    assert_nested_equal(actual, expected)
                    steps[i] += 1
                    if steps[i] < 128:
                        ready.add(i)
                progressed.add(tuple(steps))
        assert steps == [128]*4
        assert any(len(set(s)) > 1 for s in progressed), 'games must advance independently'
        order = lambda samples: sorted(samples, key=lambda sample: sample['env'])
        assert_nested_equal(order(reference.take_small_playout_samples()), order(candidate.take_small_playout_samples()))
        assert_nested_equal(reference.env_method('export_training_state'), candidate.env_method('export_training_state'))
        for key, value in reference.small_playout_stats().items():
            if 'seconds' not in key and not key.startswith('overlap/'):
                np.testing.assert_equal(value, candidate.small_playout_stats()[key])
    finally:
        reference.close(); candidate.close()
