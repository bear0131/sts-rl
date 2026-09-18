"""Architecture replacement preserves the upper, resets progress and roundtrips."""

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from tools.checkpoints.replace_combat import replace_with_mlp

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(90)]


def test_replace_roundtrip_and_seed(tmp_path):
    env = DummyVecEnv([SlayTheSpireEnv])
    try:
        model = AccumulatingMaskablePPO(
            SplitStSPointerPolicy, env, n_steps=2, batch_size=2, n_epochs=1,
            policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4), seed=42, device="cpu",
            nosl_is_az=True, nosl_is_az_inner_only=True, nosl_is_az_cpp=True,
            nosl_is_az_upper_strategy="rules", nosl_is_az_updates=40,
            nosl_is_az_simulations=128, nosl_is_az_depth=0, nosl_is_az_replay_capacity=64,
        )
        model.configure_nosl_is_az_inner_only()
        for p in model.policy.core.combat.parameters():
            p.grad = torch.ones_like(p)
        model.policy.optimizer.step()
        assert model.policy.optimizer.state
        model.num_timesteps, model._n_updates = 1024, 3
        original = tmp_path / "original.zip"
        model.save(original)
        frozen = {key: value.clone() for key, value in model.policy.state_dict().items()
                  if not key.startswith("core.combat.")}
        obs = env.reset()
        masks = np.stack(env.env_method("action_masks"))
        upper_before = model.predict(obs, deterministic=True, action_masks=masks)[0]
        replace_with_mlp(model, seed=0)
        assert model.num_timesteps == model._n_updates == 0
        assert model._nosl_is_az_replay is None and not model.policy.optimizer.state
        assert model.nosl_is_az_updates == 40 and model.nosl_is_az_simulations == 128
        for key, before in frozen.items():
            assert torch.equal(model.policy.state_dict()[key], before)
        np.testing.assert_array_equal(upper_before, model.predict(
            obs, deterministic=True, action_masks=masks)[0])
        mlp = tmp_path / "mlp.zip"
        model.save(mlp)
        restored = AccumulatingMaskablePPO.load(mlp, device="cpu")
        repeated = replace_with_mlp(AccumulatingMaskablePPO.load(original, device="cpu"), seed=0)
        assert restored.policy_kwargs["combat_arch"] == "mlp"
        for key, before in model.policy.state_dict().items():
            assert torch.equal(restored.policy.state_dict()[key], before)
            assert torch.equal(repeated.policy.state_dict()[key], before)
        restored.configure_nosl_is_az_inner_only()
        assert all(not p.requires_grad for p in restored.policy.core.overworld.parameters())
        assert all(p.requires_grad for p in restored.policy.core.combat.parameters())
    finally:
        env.close()
