"""Exercise the production collector → learner → logging → checkpoint lifecycle."""
import json
import sys
from types import SimpleNamespace

import pytest
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.training import distributed_sb3
from rl.training.reward import ShapedRewardWrapper

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(90)]


@pytest.mark.parametrize('nosl', [False, True])
def test_training_reaches_wandb_and_checkpoint(tmp_path, monkeypatch, short_combat_env_type, nosl):
    def make_env(*_):
        return DummyVecEnv([lambda: TimeLimit(
            ShapedRewardWrapper(short_combat_env_type()), max_episode_steps=12)] * 2)

    env = make_env()
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy, env, n_steps=16, batch_size=8, n_epochs=1,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4), device='cpu', seed=42,
        nosl_is_az=nosl, nosl_is_az_cpp=nosl, nosl_is_az_inner_only=nosl,
        nosl_is_az_simulations=4, nosl_is_az_depth=0, nosl_is_az_updates=1,
        nosl_is_az_replay_capacity=128, nosl_is_az_batch_size=16,
        nosl_is_az_micro_batch_size=8, nosl_is_az_leaf_batch_size=1,
        hierarchical_combat_hp=nosl)
    checkpoint = tmp_path / 'initial.zip'
    model.save(checkpoint)
    env.close()
    payloads, finished, identities = [], [], []
    def init(**kwargs):
        identities.append(kwargs)
        return SimpleNamespace(finish=lambda: finished.append(True))
    # Replace only the external service; the production payload construction runs.
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=init, config=SimpleNamespace(update=lambda *a, **kw: None),
        define_metric=lambda *a, **kw: None,
        log=lambda payload: payloads.append(json.loads(json.dumps(payload, allow_nan=False)))))
    monkeypatch.setattr(distributed_sb3, 'make_env', make_env)
    output = tmp_path / 'trained.zip'
    monkeypatch.setattr(sys, 'argv', ['distributed_sb3', '--rank', '0', '--world-size', '1',
        '--checkpoint', str(checkpoint), '--output', str(output), '--updates', '2',
        '--n-envs', '2', '--n-steps', '16', '--batch-size', '8', '--n-epochs', '1',
        '--gradient-accumulation-steps', '1', '--device', 'cpu', '--vec', 'dummy',
        '--reasoning-updates', '0', '--reasoning-coef', '0', '--no-interactive-reasoning',
        '--remote-root', str(tmp_path / 'transport'), '--ssh-control-dir', str(tmp_path / 'ssh'),
        '--run-id', 'ci-lifecycle', '--wandb', '--wandb-id', 'ci-existing-run'])
    distributed_sb3.main()
    assert len(payloads) == 2
    assert [p['global_step'] for p in payloads] == [32, 64]
    assert [p['legacy_step'] for p in payloads] == [1, 2]
    assert all(p['dist/avg_grad_sec'] == 0 for p in payloads)
    assert all('game/layer_5_win_rate' in p for p in payloads)
    assert identities[0]['id'] == 'ci-existing-run'
    assert finished == [True]
    restored = AccumulatingMaskablePPO.load(output, device='cpu')
    assert restored.num_timesteps == 64
    assert restored.policy.optimizer.state
