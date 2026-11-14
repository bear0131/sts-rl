"""COMBAT06-ROLLOUT: actual root observations retain the complete public schema."""
import copy
import numpy as np
import pytest
import slaythespire as sts

from env.core.native_observation import NativeRootObservationBatch
from env.core.run_env import SlayTheSpireEnv

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


def assert_overworld_projection(env, monkeypatch):
    from env.core import game_env
    assert env.thick.bc is None
    canonical = env.thick.obs_canonical()
    with monkeypatch.context() as patch:
        patch.setattr(env.thick, 'encoding_global_observation', lambda **kwargs: None)
        expected = env._obs()
    def forbidden(*args, **kwargs):
        raise AssertionError('numeric globals must not rebuild the Python inventory')
    with monkeypatch.context() as patch:
        patch.setattr(game_env, 'normalize_sim_global', forbidden)
        actual = env._obs()
        target = {key: np.full_like(value, 99) for key, value in expected.items()}
        assert env._obs(_out=target) is target
    for observed in (actual, target):
        assert observed.keys() == expected.keys()
        for key in expected:
            assert observed[key].dtype == expected[key].dtype, key
            assert observed[key].shape == expected[key].shape, key
            assert observed[key].tobytes() == expected[key].tobytes(), key
            assert not np.shares_memory(observed[key], expected[key]), key
    assert env.thick.obs_canonical() == canonical
    assert env.thick.encoding_canonical()['global']['deck'] == canonical['global']['deck']
    return actual


@pytest.mark.parametrize('seed', [21, 42101])
@pytest.mark.parametrize('full,public,language,deck_size', [
    (False, False, False, False), (True, False, False, True),
    (True, False, True, False), (True, True, True, True)])
def test_overworld_globals_full_runs_preserve_every_public_field(
        monkeypatch, seed, full, public, language, deck_size):
    assert hasattr(sts, 'OverworldGlobalObservationEncoder'), 'rebuild the explicitly selected extension'
    env = SlayTheSpireEnv(global_map_view=full, upper_public_context=public,
        card_language=language, deck_size_observation=deck_size, neow_observation=True,
        event_option_observation=True, combat_action_contract='autoregressive-v2')
    seen = set()
    try:
        env.reset(seed=seed)
        env.thick.gc.skip_battles = True
        for _ in range(650):
            seen.add((env.thick.gc.act, env._phase()))
            if env.thick.bc is not None:
                assert env.thick.encoding_global_observation(
                    card_language=language, deck_size_observation=deck_size) is None
            elif (not full and env._phase() == 'map'
                  and len(env.thick.gc.legal_map_options()) > 3):
                # The retained three-slot schema deliberately rejects a map
                # it cannot represent; optimization must preserve that error.
                with monkeypatch.context() as patch:
                    patch.setattr(env.thick, 'encoding_global_observation', lambda **kwargs: None)
                    with pytest.raises(ValueError, match='options.nodes.*exceeding capacity 3'):
                        env._obs()
                with pytest.raises(ValueError, match='options.nodes.*exceeding capacity 3'):
                    env._obs()
            else:
                assert_overworld_projection(env, monkeypatch)
            if env.thick.is_over:
                break
            # Use the same real semantic route as the reward projection
            # corpus; first codec ID can instead skip offers or pick fights.
            env.thick.step(env.thick.legal_semantic_actions()[0])
            env._invalidate_backend_cache()
        assert env.thick.is_over
        assert {1, 2, 3}.issubset({act for act, phase in seen})
        assert {'event', 'map', 'combat_reward', 'card_select', 'boss_relic', 'rest', 'shop'}.issubset(
            {phase for act, phase in seen})
        # Explicit Act-4 boundary fixture: keep the final live inventory and
        # previous map, as the task's existing terminal projection requires.
        env.thick.bc = None
        env.thick.gc.act = 4
        env._invalidate_backend_cache()
        assert env._phase() == 'none'
        assert_overworld_projection(env, monkeypatch)
    finally:
        env.close()


@pytest.mark.parametrize('phase', ['rest', 'shop', 'event', 'combat_reward'])
def test_overworld_globals_read_live_inventory_and_legacy_fallback(
        upper_search_env_factory, monkeypatch, phase):
    from env.core import action_space as A
    assert hasattr(sts, 'OverworldGlobalObservationEncoder'), 'rebuild the explicitly selected extension'
    env = upper_search_env_factory(phase)
    before = assert_overworld_projection(env, monkeypatch)
    gc = env.thick.gc
    gc.set_deck([('SEARING_BLOW', 130), ('BASH', 1), ('BASH', 0), ('STRIKE_RED', 1)])
    gc.set_relic_state([dict(id='Ancient Tea Set', counter=1), dict(id='Pen Nib', counter=7)])
    gc.set_potion_state(['Fire Potion', 'EMPTY_POTION_SLOT', 'Block Potion'])
    gc.cur_hp, gc.max_hp, gc.gold = 257, 301, 1234
    env._invalidate_backend_cache()
    actual = assert_overworld_projection(env, monkeypatch)
    assert actual['deck_ids'].tobytes() != before['deck_ids'].tobytes()
    assert actual['run_hp'].tobytes() != before['run_hp'].tobytes()
    with monkeypatch.context() as patch:
        patch.delattr(sts, 'OverworldGlobalObservationEncoder')
        fallback = env._obs()
    for key in actual:
        assert actual[key].tobytes() == fallback[key].tobytes(), key
        fallback[key].fill(99)
    later = env._obs()
    for key in actual:
        assert actual[key].tobytes() == later[key].tobytes(), key
    if phase == 'shop':
        env.step(A.seg_base('SHOP_REMOVE'))
        assert env._phase() == 'card_select'
        assert_overworld_projection(env, monkeypatch)


def assert_roots(encoder, envs):
    expected = [env._obs() for env in envs]
    actual = encoder.encode(envs)
    progress = encoder.encode_progress(envs)
    for key, value in progress.items():
        assert value.dtype == actual[key].dtype and value.tobytes() == actual[key].tobytes()
        assert not np.shares_memory(value, actual[key])
    assert set(actual) == set(expected[0])
    for row, reference in enumerate(expected):
        for key, value in reference.items():
            assert actual[key].dtype == value.dtype
            np.testing.assert_array_equal(actual[key][row], value, err_msg=f'{row}/{key}')
            assert actual[key][row].tobytes() == value.tobytes(), f'{row}/{key} bytes'
    return actual


def test_deck_context_root_upgrades_map_and_direct_destination():
    env = SlayTheSpireEnv(global_map_view=True, upper_public_context=True, card_language=True, event_option_observation=True,
        combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True,
        combat_action_contract='autoregressive-v2')
    try:
        env.reset(seed=20261111)
        for _ in range(64):
            if env.thick.bc is not None:
                break
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert env.thick.bc is not None
        env.thick.bc.apply_custom_state({'draw': [('SEARING_BLOW', 13), ('BASH', 1)],
            'discard': [('BASH', 0), ('BASH', 1)], 'exhaust': [('ANGER', 1)]})
        env._invalidate_backend_cache()
        encoder = NativeRootObservationBatch(env)
        owned = assert_roots(encoder, [env])
        assert np.count_nonzero(owned['pile_upgraded_counts']) == 4
        assert owned['map_full_rooms'].any()
        target = {k: np.full_like(v, 99) for k, v in owned.items()}
        encoder.encode([env], destination=target)
        for k in owned:
            assert target[k].tobytes() == owned[k].tobytes(), k
    finally:
        env.close()


@pytest.mark.parametrize('language', [False, True])
def test_root_batch_all_fields_mutable_globals_and_ownership(combat_factory, language):
    envs = [combat_factory(['STRIKE_RED']*5+['DEFEND_RED']*4+['BASH'],
                          encounter, seed=42+i, card_language=language,
                          combat_action_contract='autoregressive-v2')
            for i, encounter in enumerate(('Two Louse', 'Automaton', 'Collector'))]
    encoder = NativeRootObservationBatch(envs[0])
    assert encoder.global_encoder is not None, 'rebuild the explicitly selected native global encoder'
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


def test_native_globals_read_live_metadata_and_preserve_legacy_projection(combat_factory, monkeypatch):
    from env.core import native_observation
    envs = [combat_factory(['STRIKE_RED']*5+['DEFEND_RED']*4+['BASH'],
                          seed=41+i, card_language=True, combat_action_contract='autoregressive-v2')
            for i in range(2)]
    encoder = NativeRootObservationBatch(envs[0])
    assert encoder.global_encoder is not None, 'new native projection must actually run'
    assert_roots(encoder, envs)
    for i, env in enumerate(envs):
        env.thick.gc.set_deck([('BASH', 1), ('STRIKE_RED', 0), ('BASH', 0),
                              ('SEARING_BLOW', 130), ('STRIKE_RED', 1)])
        env.thick.gc.set_relic_state([dict(id='Pen Nib', counter=7+i),
                                     dict(id='Ancient Tea Set', counter=1)])
        env.thick.gc.gold = 234+i
        env.thick.bc.apply_custom_state({'player': {'hp': 17+i, 'max_hp': 97+i, 'gold': 432+i}})
        env._invalidate_backend_cache()
    expected = assert_roots(encoder, envs)
    with monkeypatch.context() as patch:
        patch.delattr(sts, 'CombatGlobalObservationEncoder')
        legacy = NativeRootObservationBatch(envs[0])
        assert legacy.global_encoder is None
        reference = legacy.encode(envs)
        for key in expected:
            assert expected[key].tobytes() == reference[key].tobytes(), key

    def forbidden(*args, **kwargs):
        raise AssertionError('native global projection called the Python canonicalizer')

    with monkeypatch.context() as patch:
        patch.setattr(native_observation, 'normalize_sim_global', forbidden)
        actual = encoder.encode(list(reversed(envs)))
        for key in actual:
            assert actual[key].tobytes() == expected[key][::-1].tobytes(), key
        # Reuse the encoder at a different batch size and after mutating a
        # returned row; neither cached metadata nor shared storage may leak.
        saved = {key: value.copy() for key, value in actual.items()}
        smaller = encoder.encode(envs[:1])
        for key in smaller:
            smaller[key].fill(99)
            assert actual[key].tobytes() == saved[key].tobytes(), key


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
    from rl.training.reward import ShapedRewardWrapper, StrategicRewardWrapper

    def factory(index, *, outer_return_context, strategic, hide_boundary, deck_latent):
        def make():
            base = SlayTheSpireEnv(combat_action_contract='autoregressive-v2',
                 card_language=True, global_map_view=True, deck_size_observation=True,
                 neow_observation=True, event_option_observation=True, outer_return_context=outer_return_context,
                 combat_value_objective='combat-outcome-means-v1', upper_public_context=deck_latent,
                 combat_exit_anchor=deck_latent, deck_latent_context=deck_latent,
                 reward_flow='sequential-v1' if deck_latent else 'full')
            expected = gym.spaces.Dict({key: space for key, space in base.observation_space.spaces.items()
                                       if not hide_boundary or key != 'post_enemy_turn_boundary'})
            wrapper = StrategicRewardWrapper if strategic else ShapedRewardWrapper
            env = gym.wrappers.TimeLimit(wrapper(base), max_episode_steps=41+index)
            return TrainingStateWrapper(CheckpointObservationCompat(env, expected))
        return make
    return lambda count, outer_return_context=True, strategic=False, hide_boundary=True, deck_latent=False: [
        factory(i, outer_return_context=outer_return_context, strategic=strategic,
                hide_boundary=hide_boundary, deck_latent=deck_latent) for i in range(count)]


@pytest.mark.parametrize('hide_boundary', [False, True])
@pytest.mark.parametrize(('strategic', 'deck_latent'), [(False, False), (True, False), (True, True)])
def test_vector_wrappers_autoreset_labels_journals_and_restore(
        monkeypatch, native_wrapper_factories, strategic, deck_latent, hide_boundary):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.training.native_vec_env import NativeSearchVecEnv
    factories = native_wrapper_factories(8, outer_return_context=not strategic, strategic=strategic,
                                         hide_boundary=hide_boundary, deck_latent=deck_latent)
    reference, candidate = DummyVecEnv(factories), NativeSearchVecEnv(factories, threads=2)
    rng = np.random.default_rng(9001000)
    try:
        assert candidate.native_root_observations is not None
        pending_batches, direct_batches = [], []
        indices = {id(env.unwrapped): index for index, env in enumerate(candidate.envs)}
        encode = candidate.native_root_observations.encode
        def record_pending(envs, **kwargs):
            pending_batches.append(tuple(indices[id(env)] for env in envs))
            direct_batches.append(kwargs.get('destination') is not None)
            return encode(envs, **kwargs)
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
        assert any(direct_batches) == (not hide_boundary)
        assert not all(direct_batches), 'mixed noncontiguous rows must retain the scatter path'
    finally:
        reference.close(); candidate.close()


def test_upper_vector_step_reuses_rows_but_public_results_remain_independent(
        monkeypatch, native_wrapper_factories):
    from env.core import encode
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.training.native_vec_env import NativeSearchVecEnv

    factories = native_wrapper_factories(2, outer_return_context=False)
    reference, candidate = DummyVecEnv(factories), NativeSearchVecEnv(factories, threads=1)
    try:
        for vec in (reference, candidate):
            vec.seed(42); vec.reset()
        actions = np.array([np.flatnonzero(mask)[0] for mask in reference.env_method('action_masks')])
        expected = reference.step(actions)
        assert not expected[2].any()
        assert all(env.unwrapped._phase() not in ('combat', 'none') for env in reference.envs)
        assert all(info.get('simulator_transition') is not False for info in expected[3])
        for env in candidate.envs:
            base = env.unwrapped
            base.thick.rmap.encoded_view(-1, -1,
                next_columns=base.thick.gc.legal_map_options(), global_map_view=True)
        def unexpected(*args, **kwargs):
            raise AssertionError('upper vector step must not allocate/copy a temporary observation')
        with monkeypatch.context() as patch:
            patch.setattr(encode, '_empty', unexpected)
            patch.setattr(candidate, '_save_obs', unexpected)
            actual = candidate.step(actions)
        assert_nested_equal(expected, actual)
        retained = copy.deepcopy(actual)
        for env in candidate.envs:
            assert env.unwrapped._step_observation_buffer is None
            assert not env.unwrapped._step_observation_buffer_used
        for key in actual[0]:
            assert not np.shares_memory(actual[0][key], candidate.buf_obs[key])
        candidate.reset()
        assert_nested_equal(retained, actual)
    finally:
        reference.close(); candidate.close()


def test_vector_observation_write_failure_releases_private_destination(
        monkeypatch, native_wrapper_factories):
    from env.core import encode
    from rl.training.native_vec_env import NativeSearchVecEnv

    vec = NativeSearchVecEnv(native_wrapper_factories(1, outer_return_context=False), threads=1)
    try:
        vec.seed(42); vec.reset()
        action = np.flatnonzero(vec.env_method('action_masks')[0])[0]
        def fail(*args, **kwargs):
            raise RuntimeError('injected observation write failure')
        with monkeypatch.context() as patch:
            patch.setattr(encode, '_encode_options', fail)
            with pytest.raises(RuntimeError, match='injected observation write failure'):
                vec.step(np.array([action]))
        base = vec.envs[0].unwrapped
        assert base._step_observation_buffer is None and not base._step_observation_buffer_used
        assert not base._defer_combat_observation
        public = base._obs()
        for key in vec.buf_obs:
            assert not np.shares_memory(public[key], vec.buf_obs[key])
        assert base.observation_space.contains(public)
    finally:
        vec.close()


def test_upper_buffered_timelimit_preserves_terminal_before_autoreset(native_wrapper_factories):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.training.native_vec_env import NativeSearchVecEnv

    factories = native_wrapper_factories(2, outer_return_context=False)
    reference, candidate = DummyVecEnv(factories), NativeSearchVecEnv(factories, threads=1)
    try:
        for vec in (reference, candidate):
            for env in vec.envs:
                while not isinstance(env, gym.wrappers.TimeLimit):
                    env = env.env
                env._max_episode_steps = 1
            vec.seed(42); vec.reset()
        actions = np.array([np.flatnonzero(mask)[0] for mask in reference.env_method('action_masks')])
        expected, actual = reference.step(actions), candidate.step(actions)
        assert_nested_equal(expected, actual)
        assert actual[2].all() and all(info['TimeLimit.truncated'] for info in actual[3])
        for info in actual[3]:
            assert 'terminal_observation' in info
            for key, value in info['terminal_observation'].items():
                assert not np.shares_memory(value, candidate.buf_obs[key])
        retained = copy.deepcopy(actual)
        candidate.reset()
        assert_nested_equal(retained, actual)
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


@pytest.mark.parametrize('seed', [42, 42101])
def test_strategic_projection_matches_complete_natural_cohort(native_wrapper_factories, monkeypatch, seed):
    import gymnasium as gym
    from rl.algorithms.small_playout import SmallPlayoutLearner
    from rl.algorithms.upper_ppo import UPPER_KEYS, COMBAT_PHASE
    from rl.policies.small_playout import export_online_search_pair
    from rl.training.native_vec_env import NativeSearchVecEnv

    pair = export_online_search_pair(SmallPlayoutLearner(seed=19).policy, collect_samples=True)
    pair.update(common_simulations=8, elite_simulations=8, boss_simulations=8, playout_depth=8)
    factories = native_wrapper_factories(2, outer_return_context=False)
    reference, candidate = [NativeSearchVecEnv(factories, threads=2) for _ in range(2)]
    rng = np.random.default_rng(seed)
    projected, upper, terminals, mixed = 0, 0, 0, 0
    try:
        for vec in (reference, candidate):
            for env in vec.envs:
                while not isinstance(env, gym.wrappers.TimeLimit):
                    env = env.env
                env._max_episode_steps = 6000
            vec.configure_small_playout(pair)
        assert_nested_equal(reference.begin_episode_rollout(seed), candidate.begin_episode_rollout(seed))
        def forbidden(*args):
            pytest.fail('strategic combat collection must not construct full neural roots')
        monkeypatch.setattr(candidate.native_root_observations, 'encode', forbidden)
        with (reference.omit_unused_combat_value_observations(),
              candidate.omit_unused_combat_value_observations(),
              candidate.strategic_observations(UPPER_KEYS.intersection(candidate.keys)) as step):
            for tick in range(1200):
                active = list(reversed(np.flatnonzero(reference.episode_rollout_active).tolist()))
                if not active:
                    break
                if tick % 3 == 0:
                    active = active[:1]
                masks = {i: reference.envs[i].get_wrapper_attr('action_masks')() for i in active}
                actions = {i: int(rng.choice(np.flatnonzero(masks[i]))) for i in active}
                calls = {i: ((), dict(state_nonce=seed+tick*2+i)) for i in active
                         if reference.buf_obs['phase'][i, 0] == COMBAT_PHASE}
                expected_decisions = reference.env_method_batch('small_playout_decision', calls)
                actual_decisions = candidate.env_method_batch('small_playout_decision', calls)
                assert_nested_equal(expected_decisions, actual_decisions)
                for i, decision in expected_decisions.items():
                    actions[i] = decision['action']
                full, rewards, dones, infos = reference.step_subset(actions)
                obs, actual_rewards, actual_dones, actual_infos = step(actions)
                assert_nested_equal((rewards, dones, infos), (actual_rewards, actual_dones, actual_infos))
                phases = full['phase'][:, 0]
                mixed += int(np.any(phases == COMBAT_PHASE) and np.any(phases != COMBAT_PHASE))
                for row, i in enumerate(active):
                    keys = {'phase', 'act', 'floor'} if phases[row] == COMBAT_PHASE else UPPER_KEYS.intersection(full)
                    for key in keys:
                        assert_nested_equal(full[key][row], obs[key][row])
                        assert not np.shares_memory(obs[key], candidate.buf_obs[key])
                    projected += int(phases[row] == COMBAT_PHASE)
                    upper += int(phases[row] != COMBAT_PHASE)
                    terminals += int(dones[row])
            assert not reference.episode_rollout_active.any(), 'fixed natural cohort budget exhausted'
            assert candidate.strategic_observation_rows_omitted > 0
        assert projected and upper and mixed and terminals == 2
        assert_nested_equal(reference._obs_from_buf(), candidate._obs_from_buf())
        assert_nested_equal(reference.env_method('export_training_state'), candidate.env_method('export_training_state'))
        assert_nested_equal(reference.take_small_playout_samples(), candidate.take_small_playout_samples())
        for key, value in reference.small_playout_stats().items():
            if 'seconds' not in key:
                np.testing.assert_equal(value, candidate.small_playout_stats()[key])
        assert candidate._strategic_observation_keys is None and not candidate._deferred_observation_rows
    finally:
        reference.close(); candidate.close()


@pytest.mark.parametrize('exit_mode', ['normal', 'cancel', 'exception', 'reading_callback'])
def test_strategic_scope_restores_live_complete_observations(native_wrapper_factories, exit_mode):
    import gymnasium as gym
    from types import SimpleNamespace
    from rl.algorithms.independent_rollout import _strategic_observations
    from rl.algorithms.upper_ppo import COMBAT_PHASE
    from rl.training.distributed_sb3 import NoopCallback
    from rl.training.native_vec_env import NativeSearchVecEnv

    class ReadingCallback(NoopCallback):
        def _on_step(self):
            assert self.model._last_obs.keys() == self.model.env.buf_obs.keys()
            return True

    vec = NativeSearchVecEnv(native_wrapper_factories(1, outer_return_context=False), threads=1)
    try:
        wrapper = vec.envs[0]
        while not isinstance(wrapper, gym.wrappers.TimeLimit):
            wrapper = wrapper.env
        wrapper._max_episode_steps = 6000
        model = SimpleNamespace(_last_obs=vec.begin_episode_rollout(42), env=vec, num_timesteps=0)
        for _ in range(32):
            if model._last_obs['phase'][0, 0] == COMBAT_PHASE:
                break
            action = int(np.flatnonzero(vec.env_method('action_masks')[0])[0])
            model._last_obs = vec.step_subset({0: action})[0]
        assert model._last_obs['phase'][0, 0] == COMBAT_PHASE
        callback = ReadingCallback() if exit_mode == 'reading_callback' else NoopCallback()
        callback.model = model
        def run():
            with vec.omit_unused_combat_value_observations(), _strategic_observations(model, vec, callback) as step:
                # End turn is an actual transition, not an autoregressive prefix.
                obs, _, _, _ = step({0: vec.envs[0].unwrapped._policy_offset})
                for key, value in obs.items():
                    model._last_obs[key][:] = value
                if exit_mode == 'reading_callback':
                    assert callback.on_step()
                else:
                    assert vec.strategic_observation_rows_omitted == 1
                if exit_mode == 'cancel':
                    return False
                if exit_mode == 'exception':
                    raise RuntimeError('callback failure')
                return True
        if exit_mode == 'exception':
            with pytest.raises(RuntimeError, match='callback failure'):
                run()
        else:
            assert run() == (exit_mode != 'cancel')
        expected = vec.envs[0].unwrapped._obs()
        assert model._last_obs.keys() == vec.buf_obs.keys()
        for key in vec.buf_obs:
            assert_nested_equal(model._last_obs[key][0], expected[key])
            assert_nested_equal(vec.buf_obs[key][0], expected[key])
        assert vec._strategic_observation_keys is None and not vec._deferred_observation_rows
        assert vec.envs[0].unwrapped._emit_nosl_value_observations
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


def test_root_destination_preserves_rows_and_owned_default_output(combat_factory):
    envs = [combat_factory(['STRIKE_RED', 'DEFEND_RED'] * 5, seed=seed,
                          combat_action_contract='autoregressive-v2', card_language=True)
            for seed in (42, 43)]
    encoder = NativeRootObservationBatch(envs[0])
    expected = assert_roots(encoder, envs)
    saved = {key: value.copy() for key, value in expected.items()}
    destination = {key: np.full_like(value, 99) for key, value in expected.items()}
    actual = encoder.encode(envs, destination=destination)
    assert_nested_equal(expected, actual)
    for key in actual:
        assert np.shares_memory(actual[key], destination[key])
        assert not np.shares_memory(actual[key], expected[key])
    encoder.encode(list(reversed(envs)), destination=destination)
    for key in expected:
        assert_nested_equal(destination[key], saved[key][::-1])
    assert_nested_equal(expected, saved)
    before = {key: value.copy() for key, value in destination.items()}
    with pytest.raises(ValueError, match='exactly one row'):
        encoder.encode(envs[:1], destination=destination)
    assert_nested_equal(destination, before)
