"""Frozen upper decisions and native combat coexist in real NOSL collection."""
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(90)]


@pytest.mark.parametrize("upper", ["model", "rules"])
def test_upper_rules_fallback_and_native_search(tmp_path, monkeypatch, short_combat_env_type, upper):
    calls, sampled_modes = [], []

    class StaticUpper(SlayTheSpireEnv):
        def step(self, action):
            assert self.action_masks()[int(action)]
            return self._obs(), 0.0, False, False, self._info()

        def overworld_rule_decision(self):
            calls.append("rule")
            return super().overworld_rule_decision()

    class FallbackUpper(StaticUpper):
        def overworld_rule_decision(self):
            calls.append("fallback")
            return None

    class Combat(short_combat_env_type):
        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.thick.bc.apply_custom_state({
                "hand": ["STRIKE_RED"] * 5, "monsters": [{"hp": 1}],
            })
            self._invalidate_backend_cache()
            return self._obs(), self._info()

        def nosl_is_az_cpp_begin(self, *args, **kwargs):
            calls.append("combat")
            return super().nosl_is_az_cpp_begin(*args, **kwargs)

        def overworld_rule_decision(self):
            assert self._phase() != "combat", "upper rules must not handle combat"
            calls.append("postcombat")
            return super().overworld_rule_decision()

    env = DummyVecEnv([StaticUpper, FallbackUpper, Combat])
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy, env, n_steps=4, batch_size=6, n_epochs=1,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4), seed=42, device="cpu",
        nosl_is_az=True, nosl_is_az_inner_only=True, nosl_is_az_cpp=True,
        nosl_is_az_upper_strategy=upper, nosl_is_az_simulations=4,
        nosl_is_az_depth=0, nosl_is_az_leaf_batch_size=1,
        nosl_is_az_replay_capacity=64, nosl_is_az_batch_size=8,
        nosl_is_az_micro_batch_size=4,
    )
    model.configure_nosl_is_az_inner_only()
    initial_upper = {k: v.clone() for k, v in model.policy.state_dict().items()
                     if k.startswith("core.overworld.")}
    assert initial_upper
    dist_type = type(model.policy.action_dist)
    get_actions = dist_type.get_actions

    def tracked_actions(self, deterministic=False):
        sampled_modes.append(deterministic)
        return get_actions(self, deterministic=deterministic)

    monkeypatch.setattr(dist_type, "get_actions", tracked_actions)

    class CheckActions(BaseCallback):
        def _on_step(self):
            state = self.locals
            actions = state["actions_np"]
            assert state["action_masks"][np.arange(3), actions].all()
            if upper == "rules":
                assert actions[0] == env.envs[0].overworld_rule_decision()
                with torch.no_grad():
                    expected = state["distribution"].mode().cpu().numpy()
                assert actions[1] == expected[1]
            return True

    try:
        model.learn(24, callback=CheckActions())
        assert model.num_timesteps == 24
        assert sampled_modes and all(mode == (upper == "rules") for mode in sampled_modes)
        assert "combat" in calls
        counts = model._nosl_is_az_rollout_upper_counts
        if upper == "rules":
            assert counts["overworld_rule_decisions"] >= 4
            assert counts["overworld_fallback_decisions"] == 4
        else:
            assert set(calls) == {"combat"}
            assert counts == {"overworld_rule_decisions": 0, "overworld_fallback_decisions": 0}
        # Fight-local labels may remain pending under this fixed action budget.
        assert (model._nosl_is_az_replay.size +
                sum(len(fight) for fight in model._nosl_is_az_pending_fights)) > 0
        for key, before in initial_upper.items():
            assert torch.equal(model.policy.state_dict()[key], before)
        checkpoint = tmp_path / "upper.zip"
        model.save(checkpoint)
        restored = AccumulatingMaskablePPO.load(checkpoint, device="cpu")
        assert restored.nosl_is_az_upper_strategy == upper
        obs = env.reset()
        masks = np.stack(env.env_method("action_masks"))
        np.testing.assert_array_equal(
            model.predict(obs, deterministic=True, action_masks=masks)[0],
            restored.predict(obs, deterministic=True, action_masks=masks)[0],
        )
    finally:
        env.close()
