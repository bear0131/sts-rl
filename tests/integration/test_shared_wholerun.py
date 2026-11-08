from copy import deepcopy

import numpy as np
import pytest
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from env.core.run_env import SlayTheSpireEnv
from env.core import vocab
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.shared_wholerun import SharedWholeRunPolicy
from rl.policies.checkpoint import load_checkpoint
from rl.training.reward import StrategicRewardWrapper
from rl.training.environments import TrainingStateWrapper
from rl.training.checkpoints import save_bundle, restore_learner

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


def make_env():
    return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(SlayTheSpireEnv(
        card_language=True, global_map_view=True, combat_action_contract='autoregressive-v2')),
        max_episode_steps=6000))


def make_model(env, device):
    return AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261108,
        n_steps=32, batch_size=32, n_epochs=1, device=device,
        gamma=1., gae_lambda=1., learning_rate=1e-4,
        reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=.9))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_ppo_two_rounds_and_resume(tmp_path, device):
    assert device != 'cuda' or torch.cuda.is_available(), 'explicit CUDA qualification required'
    env = DummyVecEnv([make_env, make_env])
    model = make_model(env, device)
    model.set_logger(configure(folder=None, format_strings=[]))
    recorded = {}
    original_record = model.logger.record
    def record(key, value, *args, **kwargs):
        recorded[key] = value
        original_record(key, value, *args, **kwargs)
    model.logger.record = record
    core = model.policy.core
    assert core.combat.transformer is core.overworld.encoder.transformer
    assert core.combat.card is core.upper_tokenizer.card
    params = [p for g in model.policy.optimizer.param_groups for p in g['params']]
    assert len(params) == len({id(p) for p in params})
    before = deepcopy(model.policy.state_dict())
    model.learn(total_timesteps=128)
    assert model.num_timesteps == 128 and model._n_updates == 2
    assert recorded['pure_rl/combat_samples'] > 0
    assert not model.nosl_is_az and not model.hierarchical_combat_hp
    assert not hasattr(model.policy, 'upper_optimizer')
    for prefix in ('core.combat.transformer.', 'core.combat.end.', 'core.overworld.fixed_head.'):
        assert any(not torch.equal(v, before[k]) for k, v in model.policy.state_dict().items()
                   if k.startswith(prefix)), prefix
    assert recorded['pure_rl/search_simulations'] == 0
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    bundle = save_bundle(model, tmp_path/'recovery', rank=0, world_size=1,
                         run_id='pure-test', wandb_id=None, config={})
    assert bundle is not None
    model.save(tmp_path/'policy.zip')
    loaded = load_checkpoint(tmp_path/'policy.zip', device=device)
    assert isinstance(loaded, AccumulatingMaskablePPO)
    assert loaded.policy.core.combat.transformer is loaded.policy.core.overworld.encoder.transformer
    obs = model._last_obs
    masks = np.stack(env.env_method('action_masks'))
    for policy in (model.policy, loaded.policy): policy.set_training_mode(False)
    x, _ = model.predict(obs, action_masks=masks, deterministic=True)
    y, _ = loaded.predict(obs, action_masks=masks, deterministic=True)
    np.testing.assert_array_equal(x, y)
    resumed_env = DummyVecEnv([make_env, make_env])
    loaded.set_env(resumed_env, force_reset=False)
    restore_learner(loaded, bundle, rank=0, world_size=1)
    optimizer = loaded.policy.optimizer
    old_steps = {p: int(s['step']) for p, s in optimizer.state.items()}
    shared = next(loaded.policy.core.combat.transformer.parameters())
    loaded.learn(total_timesteps=128, reset_num_timesteps=False)
    assert loaded.num_timesteps == 256
    # A phase-specific head need not occur in both batches. Shared backbone
    # moments must advance; every existing head must retain its prior steps.
    assert int(optimizer.state[shared]['step']) > old_steps[shared]
    assert all(int(optimizer.state[p]['step']) >= step for p, step in old_steps.items())
    env.close(); resumed_env.close()
