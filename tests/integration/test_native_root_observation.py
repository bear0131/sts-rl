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


def test_vector_wrappers_autoreset_labels_journals_and_restore():
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.training.native_vec_env import NativeSearchVecEnv
    from rl.training.environments import CheckpointObservationCompat, TrainingStateWrapper
    from rl.training.reward import ShapedRewardWrapper

    def factory(index):
        def make():
            base = SlayTheSpireEnv(combat_action_contract='autoregressive-v2',
                 card_language=True, global_map_view=True, deck_size_observation=True,
                 neow_observation=True, event_option_observation=True, outer_return_context=True,
                 combat_value_objective='combat-outcome-means-v1')
            expected = gym.spaces.Dict({key: space for key, space in base.observation_space.spaces.items()
                                       if key != 'post_enemy_turn_boundary'})
            env = gym.wrappers.TimeLimit(ShapedRewardWrapper(base), max_episode_steps=41+index)
            return TrainingStateWrapper(CheckpointObservationCompat(env, expected))
        return make

    factories = [factory(i) for i in range(8)]
    reference, candidate = DummyVecEnv(factories), NativeSearchVecEnv(factories, threads=2)
    rng = np.random.default_rng(9001000)
    try:
        assert candidate.native_root_observations is not None
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
