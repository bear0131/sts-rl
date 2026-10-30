"""Public run context survives upper screens, policy filtering and recovery."""
import numpy as np
import pytest
import torch
from env.core import encode, vocab as V
from env.core.run_env import SlayTheSpireEnv
from rl.policies.entity_joint import EntityJointPolicy

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


@pytest.mark.parametrize('schema', ['legacy', 'global_map', 'public_context'])
def test_act3_victory_returns_terminal_observation(act3_boss_map_root, schema):
    from env.core import action_space as A
    env = SlayTheSpireEnv(global_map_view=schema != 'legacy',
        upper_public_context=schema == 'public_context',
        event_option_observation=schema == 'public_context')
    try:
        env.reset(seed=42)
        x = act3_boss_map_root(env)
        obs, reward, done, cut, info = env.step(A.seg_base('MAP_NODE') + x)
        assert done and not cut and reward == 1.
        assert info['run_stats']['act'] == 4
        assert env.thick.phase() == 'none'
        assert obs['phase'][0] == V.PHASE_VOCAB_INDEX['none']
        assert env.observation_space.contains(obs) and not env.action_masks().any()
        assert not obs['map_node_rooms'].any()
        if schema != 'legacy':
            assert not obs['map_full_rooms'].any()
        assert obs['run_hp'][0] > 0 and obs['deck_ids'].any()
    finally:
        env.close()


def test_native_vector_preserves_victory_terminal_before_reset(act3_boss_map_root):
    from env.core import action_space as A
    from rl.training.native_vec_env import NativeSearchVecEnv
    vec = NativeSearchVecEnv([lambda: SlayTheSpireEnv(global_map_view=True,
        upper_public_context=True, event_option_observation=True,
        combat_action_contract='autoregressive-v2')], threads=1)
    try:
        vec.seed(42); vec.reset()
        x = act3_boss_map_root(vec.envs[0])
        obs, rewards, dones, infos = vec.step(np.array([A.seg_base('MAP_NODE') + x]))
        assert dones[0] and rewards[0] == 1.
        assert infos[0]['terminal_observation']['phase'][0] == V.PHASE_VOCAB_INDEX['none']
        assert not infos[0]['TimeLimit.truncated']
        assert infos[0]['run_stats']['act'] == 4
        assert obs['phase'][0, 0] != V.PHASE_VOCAB_INDEX['none']
        assert vec.envs[0].thick.gc.act == 1
    finally:
        vec.close()


@pytest.mark.parametrize('phase', ['combat_reward', 'card_select', 'event', 'shop', 'rest', 'chest', 'boss_relic', 'map'])
def test_all_upper_screens_preserve_context_and_network_reads_map(monkeypatch, phase):
    from env.core import game_env
    env = SlayTheSpireEnv(global_map_view=True, upper_public_context=True, card_language=True,
        combat_action_contract='autoregressive-v2', deck_size_observation=True,
        neow_observation=True, event_option_observation=True)
    try:
        env.reset(seed=42)
        original = game_env.normalize_sim_overworld
        def screen(gc, **kw):
            c = original(gc, **kw)
            c['phase'], c['options'] = phase, {}
            return c
        monkeypatch.setattr(game_env, 'normalize_sim_overworld', screen)
        obs = env._obs()
        assert env.observation_space.contains(obs)
        assert obs['map_full_rooms'].any() and obs['map_full_features'][:, 4].any()
        reference = encode.encode(dict(phase='map', options=env.thick.public_map_context()), global_map_view=True)
        for key in ('map_full_rooms', 'map_full_features'):
            np.testing.assert_array_equal(obs[key], reference[key])
        for key in ('deck_ids', 'relics', 'run_hp', 'next_boss'):
            assert obs[key].any(), key
        policy = EntityJointPolicy(env.observation_space, env.action_space, lambda _: 3e-4,
            entity_config=dict(width=32, layers=1, heads=4, quantiles=8),
            upper_config=dict(width=32, layers=1, heads=4))
        tensors = {k: torch.as_tensor(v[None]) for k, v in obs.items()}
        tensors['phase'].fill_(V.PHASE_VOCAB_INDEX[phase])
        policy.core.actor_critic(tensors)[1].sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in policy.core.overworld.encoder.map_dag.parameters())
        old_space = encode.obs_space(global_map_view=True)
        assert 'upper_public_context' not in old_space.spaces
    finally:
        env.close()


def test_real_upper_route_context_and_exact_environment_restore():
    env = SlayTheSpireEnv(global_map_view=True, upper_public_context=True, card_language=True,
        combat_action_contract='autoregressive-v2', deck_size_observation=True,
        neow_observation=True, event_option_observation=True)
    restored = SlayTheSpireEnv(global_map_view=True, upper_public_context=True, card_language=True,
        combat_action_contract='autoregressive-v2', deck_size_observation=True,
        neow_observation=True, event_option_observation=True)
    try:
        env.reset(seed=42)
        # A normal short battle is replayable; no custom states or skip flags.
        rng = np.random.default_rng(21)
        phases = set()
        checked_native = False
        for step in range(150):
            obs = env._obs(); phase = env._phase(); phases.add(phase)
            if phase != 'combat':
                assert obs['map_full_rooms'].any(), phase
            elif not checked_native:
                from env.core.native_observation import NativeRootObservationBatch
                native = NativeRootObservationBatch(env).encode([env])
                for key in obs:
                    np.testing.assert_array_equal(native[key][0], obs[key])
                checked_native = True
            if step == 30:
                saved = env.export_training_state()
                actual, _ = restored.restore_training_state(saved)
                for k in obs:
                    np.testing.assert_array_equal(actual[k], obs[k])
            _, _, done, cut, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            if done or cut:
                break
        assert {'event', 'map', 'combat'} <= phases
        assert checked_native
    finally:
        env.close(); restored.close()


def test_event_public_targets_and_selection_origin(monkeypatch):
    from env.core import game_env
    env = SlayTheSpireEnv(global_map_view=True, upper_public_context=True,
        event_option_observation=True, card_language=True, combat_action_contract='autoregressive-v2')
    try:
        env.reset(seed=42)
        before = env._obs()
        original = game_env.normalize_sim_overworld
        def selection(gc, **kw):
            c = original(gc, **kw)
            c['phase'] = 'card_select'
            c['options'] = dict(cards=[('Bash', 0)], select=dict(context='TRANSFORM',
                pick_remaining=2, can_pick_zero=False, can_pick_any=False, selected=[]))
            return c
        monkeypatch.setattr(game_env, 'normalize_sim_overworld', selection)
        after = env._obs()
        np.testing.assert_array_equal(after['event_id'], before['event_id'])
        assert after['select_context'][0] != 0 and after['pick_remaining'][0] > 0
        for key in ('map_full_rooms', 'map_full_features'):
            np.testing.assert_array_equal(after[key], before[key])
    finally:
        env.close()


def test_dynamic_event_entities_reach_policy_inputs():
    options = dict(event='Falling', event_options=[0, 1, 2], event_choices=[
        dict(index=0, kind='lose_card', card=('Offering', 1)),
        dict(index=1, kind='lose_relic', relic='Anchor'),
        dict(index=2, kind='lose_potion', potion='Fairy Potion')])
    env = SlayTheSpireEnv(global_map_view=True, upper_public_context=True,
                         event_option_observation=True, card_language=True,
                         combat_action_contract='autoregressive-v2')
    try:
        env.reset(seed=42)
        c = dict(phase='event', options=options, public_map=env.thick.public_map_context())
        obs = encode.encode(c, global_map_view=True, upper_public_context=True,
                            event_option_observation=True, card_language=True)
        for key in ('event_choice_card_ids', 'event_choice_relic_ids', 'event_choice_potion_ids'):
            assert obs[key].any() and key in encode.upper_observation_keys(V.PHASE_VOCAB_INDEX['event'], obs)
        assert obs['event_choice_card_upgrades'][0] == pytest.approx(1/encode.UPGRADE_SCALE)
        policy = EntityJointPolicy(env.observation_space, env.action_space, lambda _: 3e-4,
            entity_config=dict(width=32, layers=1, heads=4, quantiles=8),
            upper_config=dict(width=32, layers=1, heads=4))
        tensors = {k: torch.as_tensor(v[None]) for k, v in obs.items()}
        policy.core.actor_critic(tensors)[1].sum().backward()
        encoder = policy.core.overworld.encoder
        for module in (encoder.event_kind_embedding, encoder.event_detail_projection):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    finally:
        env.close()
