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


def test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path):
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
    env = DummyVecEnv([factory, factory])
    model = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
        n_steps=32, batch_size=32, n_epochs=1, device='cpu', gamma=1., gae_lambda=1.,
        learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1'))
    configure_packed_updates(model, dict(token_budget=1024, max_rows=8, precision='fp32'))
    model.learn(128)
    assert model.num_timesteps == 128
    source = save_bundle(model, tmp_path/'first', rank=0, world_size=1, run_id='deck-latent-test',
                         wandb_id=None, config={'ascension': 0})
    loaded = load_checkpoint(source/'model.zip', device='cpu')
    assert isinstance(loaded.policy, DeckLatentPolicy)
    target = DummyVecEnv([factory, factory])
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
