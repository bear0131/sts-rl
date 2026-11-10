from copy import deepcopy

import numpy as np
import pytest
import torch
from stable_baselines3.common.utils import obs_as_tensor

from env.core import vocab
from env.core.run_env import SlayTheSpireEnv
from rl.policies.deck_latent import DeckLatentPolicy


pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


@pytest.fixture
def latent_env():
    env = SlayTheSpireEnv(card_language=True, global_map_view=True, upper_public_context=True,
        event_option_observation=True, combat_action_contract='autoregressive-v2',
        combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True)
    env.reset(seed=20261111)
    yield env
    env.close()


def policy_for(env):
    torch.manual_seed(20261111)
    return DeckLatentPolicy(env.observation_space, env.action_space, lambda _: 5e-5,
        d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1')


def tensor_obs(env, repeats=1):
    return obs_as_tensor({k: np.repeat(v[None], repeats, 0) for k, v in env._obs().items()}, 'cpu')


def enter_combat(env):
    for _ in range(128):
        if env.thick.bc is not None:
            return
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    raise AssertionError('could not enter combat with bounded legal actions')


def test_public_context_present_in_combat_and_checkpoint_replay(latent_env):
    env = latent_env
    enter_combat(env)
    obs = env._obs()
    assert obs['deck_ids'].any()
    assert obs['map_full_rooms'].any()
    assert obs['map_full_features'][..., 2].any()  # current node
    assert obs['deck_latent_context'].item() == 1
    state = env.export_training_state()
    env.step(int(np.flatnonzero(env.action_masks())[0]))
    env.restore_training_state(state)
    for key, value in obs.items():
        np.testing.assert_array_equal(env._obs()[key], value)
    incompatible = deepcopy(state); incompatible['deck_latent_context'] = False
    with pytest.raises(ValueError, match='deck latent'):
        env.restore_training_state(incompatible)


def test_all_phases_gradients_cache_invalidation_and_direct_search_interface(latent_env):
    env = latent_env
    policy = policy_for(env)
    for combat in (False, True):
        if combat:
            enter_combat(env)
        obs = tensor_obs(env, 2)
        policy.set_training_mode(True)
        logits, values = policy.core.actor_critic(obs)
        assert logits.shape == (2, env.action_space.n)
        assert values.shape == (2, 1)
        (values.square().sum() + logits[logits > -1e8].square().mean()).backward()
        assert policy.core.context.card_projection.weight.grad.abs().sum() > 0
        policy.zero_grad(set_to_none=True)
        policy.set_training_mode(False)
        with torch.no_grad():
            reference = policy.core.actor_critic(obs)
            cached = policy.core.actor_critic(obs)
            for a, b in zip(reference, cached, strict=True):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            obs['relic_counters'] = obs['relic_counters'] + .1
            z = policy.core.run_context(obs)
            torch.testing.assert_close(z, policy.core._context_cache[2], rtol=0, atol=0)
            obs['outer_deck_misc'][:, 0] += .2
            changed = policy.core.run_context(obs)
            assert not torch.allclose(z, changed)
            policy.core.context.roles.add_(.03)
            refreshed = policy.core.run_context(obs)
            torch.testing.assert_close(refreshed, policy.core.context(obs), atol=2e-6, rtol=2e-5)
            if combat:
                direct = policy.core.combat_with_context(obs, refreshed)
                routed = policy.core.actor_critic(obs)
                for a, b in zip(direct, routed, strict=True):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_anchor_uses_branch_hp_and_ignores_unrevealed_rewards(latent_env):
    env = latent_env
    enter_combat(env)
    policy = policy_for(env)
    policy.set_training_mode(False)
    obs = tensor_obs(env)
    obs['combat_exit_anchor'][0, 0] = 1.
    with torch.no_grad():
        z = policy.core.run_context(obs)
        projected = policy.core.anchor.context_projection(z)
        baseline = policy.core.anchor_with_context(obs, projected)
        obs['run_hp'] += .5
        obs['reward_gold'] += 1
        obs['offer_card_ids'] += 1
        torch.testing.assert_close(policy.core.anchor_with_context(obs, projected), baseline, rtol=0, atol=0)
        obs['player_scalars'][:, 0] += .3
        assert not torch.allclose(policy.core.anchor_with_context(obs, projected), baseline)
        logits, value = policy.core.actor_critic(obs)
        torch.testing.assert_close(value, policy.core.anchor_with_context(obs, projected))
        assert not logits.any()


def test_duplicate_context_gradients_equal_full_recomputation(latent_env):
    core = policy_for(latent_env).core
    obs = tensor_obs(latent_env, 3)
    obs['outer_deck_misc'][1, 0] = .4
    params = tuple(core.context.parameters())
    a = core.run_context(obs)
    ga = torch.autograd.grad(a.square().sum(), params, allow_unused=True)
    b = core.context(obs)
    gb = torch.autograd.grad(b.square().sum(), params, allow_unused=True)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    for x, y in zip(ga, gb, strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, atol=3e-5, rtol=1e-3)


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('transport', ['dummy', 'native-subproc'])
def test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path, search, transport):
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.packed_ppo import configure_packed_updates
    from rl.training.checkpoints import save_bundle, restore_learner
    from rl.training.environments import TrainingStateWrapper
    from rl.training.reward import StrategicRewardWrapper
    from rl.policies.checkpoint import load_checkpoint

    def factory():
        return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(SlayTheSpireEnv(
            card_language=True, global_map_view=True, upper_public_context=True,
            event_option_observation=True, combat_action_contract='autoregressive-v2',
            combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True)), 6000))
    def vector():
        if transport == 'dummy':
            return DummyVecEnv([factory, factory])
        from rl.training.native_subproc_vec_env import NativeSubprocVecEnv
        return NativeSubprocVecEnv([factory, factory], envs_per_worker=1, start_method='spawn')
    env = vector()
    model = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
        n_steps=32, batch_size=32, n_epochs=1, device='cpu', gamma=1., gae_lambda=1.,
        learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1'))
    configure_packed_updates(model, dict(token_budget=1024, max_rows=8, precision='fp32'))
    if search:
        from dataclasses import asdict
        from rl.algorithms.search_auxiliary import SearchTrainingConfig
        model.policy.combat_value_target = 'pre-reward-anchor'
        model.policy_kwargs['combat_value_target'] = 'pre-reward-anchor'
        model.deck_search_config = asdict(SearchTrainingConfig(fraction=1., normal_budget=2,
            boss_budget=2, max_depth=2, max_active=16, timeout_seconds=30.))
        model.deck_search_snapshot_dir = str(tmp_path / 'weights')
    model.learn(128)
    assert model.num_timesteps == 128
    if search:
        assert model.deck_search_last_stats['searched_roots'] > 0
        assert len(model.rollout_buffer.search_targets) > 0
        assert model._deck_search_loss_normalization > 0
        assert model.deck_search_last_stats['atomic_steps'] > model.deck_search_last_stats['searched_roots']
    source = save_bundle(model, tmp_path/'first', rank=0, world_size=1, run_id='deck-latent-test',
                         wandb_id=None, config={'ascension': 0})
    loaded = load_checkpoint(source/'model.zip', device='cpu')
    assert isinstance(loaded.policy, DeckLatentPolicy)
    if search:
        assert loaded.deck_search_config == model.deck_search_config
        assert loaded.deck_search_rng == model.deck_search_rng
    target = vector()
    restored = AccumulatingMaskablePPO.load(source/'model.zip', env=target, device='cpu')
    restored._setup_learn(64, reset_num_timesteps=False)
    restore_learner(restored, source, rank=0, world_size=1)
    assert restored._recovery_environment_restored
    for a, b in zip(model.policy.parameters(), restored.policy.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    restored.learn(128, reset_num_timesteps=False)
    assert restored.num_timesteps == 256
    assert save_bundle(restored, tmp_path/'resumed', rank=0, world_size=1,
        run_id='deck-latent-test', wandb_id=None, config={'ascension': 0}).is_dir()
    env.close(); target.close()


def test_paired_shadow_search_batches_real_anchors_without_mutating_root(latent_env):
    from env.search.anchored_chains import paired_anchor_search
    from rl.search.deck_latent import DeckLatentScorer
    env = latent_env
    enter_combat(env)
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    obs = env._obs()
    before = env._array_mapping_digest(obs)
    root_key = env.thick.bc.nosl_future_key()
    journal = deepcopy(env._episode_action_journal)
    policy = policy_for(env); policy.set_training_mode(False)
    scorer = DeckLatentScorer(policy, [obs])
    seen = []
    def score(requests):
        seen.extend(r.kind for r in requests)
        result = scorer(requests)
        with torch.no_grad():
            for request, (logit, value) in zip(requests, result, strict=True):
                full = obs_as_tensor({k: v[None] for k, v in request.observation.items()}, 'cpu')
                expected_logit, expected_value = policy.core.actor_critic(full)
                np.testing.assert_allclose(value, expected_value.item(), rtol=3e-5, atol=3e-6)
                if logit is not None:
                    np.testing.assert_allclose(logit, expected_logit[0], rtol=3e-5, atol=3e-6)
        return result
    count = np.count_nonzero(env.action_masks())
    result, stats = paired_anchor_search([env], [count * 2], score, seed=20261111,
        max_depth=12, max_active=16, timeout_seconds=30)
    assert stats['chains'] == count * 2
    assert stats['max_active'] > 1
    assert (result[0]['visits'] == 2).all()
    assert np.isfinite(result[0]['values']).all()
    assert 'anchor' in seen
    assert env._array_mapping_digest(env._obs()) == before
    assert env.thick.bc.nosl_future_key() == root_key
    assert env._episode_action_journal == journal
    # Changing the frozen policy must invalidate the entire teacher.
    with torch.no_grad():
        policy.core.combat.end.bias.add_(.1)
    with pytest.raises(RuntimeError, match='weights changed'):
        scorer([])


def test_shadow_teaching_preserves_real_actions_rewards_and_sampling_rng():
    from dataclasses import asdict
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.packed_ppo import configure_packed_updates
    from rl.algorithms.search_auxiliary import SearchTrainingConfig
    from rl.training.reward import StrategicRewardWrapper

    class Capture(BaseCallback):
        def _on_step(self):
            return True
        def _on_rollout_end(self):
            self.actions = self.model.rollout_buffer.actions.copy()
            self.rewards = self.model.rollout_buffer.rewards.copy()
            self.rng = torch.get_rng_state().clone()

    captures = []
    for searched in (False, True):
        env = DummyVecEnv([lambda: StrategicRewardWrapper(SlayTheSpireEnv(
            card_language=True, global_map_view=True, upper_public_context=True,
            event_option_observation=True, combat_action_contract='autoregressive-v2',
            combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True))])
        model = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
            n_steps=24, batch_size=24, n_epochs=1, device='cpu', gamma=1., gae_lambda=1.,
            learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
            policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
                reward_flow='sequential-v1', combat_value_target='pre-reward-anchor'))
        configure_packed_updates(model, dict(token_budget=1024, max_rows=8, precision='fp32'))
        if searched:
            model.deck_search_config = asdict(SearchTrainingConfig(fraction=1., normal_budget=2,
                boss_budget=2, max_depth=2, max_active=16, timeout_seconds=30.))
        callback = Capture()
        model.learn(24, callback=callback)
        if searched:
            assert model.deck_search_last_stats['searched_roots'] > 0
        captures.append(callback)
        env.close()
    np.testing.assert_array_equal(captures[0].actions, captures[1].actions)
    np.testing.assert_array_equal(captures[0].rewards, captures[1].rewards)
    torch.testing.assert_close(captures[0].rng, captures[1].rng, rtol=0, atol=0)


@pytest.mark.parametrize('card,zone', [('ARMAMENTS', 1), ('SECRET_TECHNIQUE', 2), ('HEADBUTT', 3)])
def test_root_selection_identity_depends_on_candidate_zone(latent_env, card, zone):
    import slaythespire as sts
    from env.search.anchored_chains import paired_anchor_search
    env = latent_env
    enter_combat(env)
    env.thick.bc.apply_custom_state(dict(hand=[card, 'DEFEND_RED', 'STRIKE_RED'],
        draw=['DEFEND_RED', 'SHRUG_IT_OFF', 'STRIKE_RED'], discard=['BASH', 'ANGER']))
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._invalidate_backend_cache()
    assert env._obs()['select_zone'].item() == zone
    before = env._array_mapping_digest(env._obs())
    legal = np.flatnonzero(env.action_masks())
    result, stats = paired_anchor_search([env], [2 * len(legal)],
        lambda requests: [(None, 0.) for r in requests], seed=20261111, max_depth=1,
        max_active=16, timeout_seconds=10)
    assert (result[0]['visits'] == 2).all()
    assert env._array_mapping_digest(env._obs()) == before


def test_anchor_validation_uses_completed_natural_games_and_marks_limits(latent_env):
    from types import SimpleNamespace
    from rl.evaluation.anchor_readiness import anchor_readiness
    from rl.evaluation.runner import EvaluationConfig, Scenario
    policy = policy_for(latent_env)
    model = SimpleNamespace(policy=policy, observation_space=latent_env.observation_space,
        action_space=latent_env.action_space, device=torch.device('cpu'),
        nosl_action_group_contract='individual-actions')
    model.predict = policy.predict
    def factory():
        return SlayTheSpireEnv(card_language=True, global_map_view=True, upper_public_context=True,
            event_option_observation=True, combat_action_contract='autoregressive-v2',
            combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True)
    # Real environment lifecycle with a strict small cap: no incomplete future
    # may become an anchor's "actual" return or a natural win/loss.
    result = anchor_readiness(model, Scenario('latent-readiness', factory),
                             EvaluationConfig(episodes=2, seed=20261111, num_envs=2, max_steps=8, timeout=10.))
    assert result['natural'] == 0
    assert result['censored'] == 2
    assert result['anchors'] == 0
    natural = anchor_readiness(model, Scenario('latent-readiness', factory),
        EvaluationConfig(episodes=2, seed=20261111, num_envs=2, max_steps=6000, timeout=30.))
    assert natural['natural'] == 2 and natural['censored'] == 0
    assert natural['anchors'] > 0
