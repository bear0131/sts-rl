"""Exercise a saved pre-boundary schema through current inference consumers."""
import sys

import gymnasium as gym
import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from env.core.run_env import SlayTheSpireEnv
from rl.policies.model import observation_for_policy
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.search.combat_value_search import CombatValueSearch, CombatValueSearchConfig
from apps.watch import _action_dist
from rl.evaluation import eval as evaluate

pytestmark = pytest.mark.native


class LegacyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            key: space for key, space in env.observation_space.spaces.items()
            if key not in {"post_enemy_turn_boundary", "select_card_costs",
                           "select_native_indices", "select_zone", "pending_card_slot",
                           "pending_potion_slot"}
        })

    def observation(self, obs):
        return {key: obs[key] for key in self.observation_space.spaces}


@pytest.fixture
def legacy_checkpoint(tmp_path):
    env = LegacyObservation(SlayTheSpireEnv())
    try:
        model = MaskablePPO(SplitStSPointerPolicy, env, n_steps=2, batch_size=2,
                            policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4),
                            seed=42, device="cpu")
        path = tmp_path / "legacy.zip"
        model.save(path)
        yield path
    finally:
        env.close()


def test_legacy_checkpoint_prediction_watch_and_search(legacy_checkpoint, combat_factory):
    model = MaskablePPO.load(legacy_checkpoint, device="cpu")
    env = combat_factory(["STRIKE_RED"] * 5)
    obs, mask = env._obs(), env.action_masks()
    assert "post_enemy_turn_boundary" in obs
    projected = observation_for_policy(model, obs)
    assert "post_enemy_turn_boundary" not in projected
    assert "post_enemy_turn_boundary" in obs  # no mutation of live observation
    action = int(model.predict(projected, action_masks=mask, deterministic=True)[0])
    assert mask[action]
    probabilities, value = _action_dist(model, obs, mask)
    assert np.isfinite(value) and np.isfinite(probabilities).all()
    result = CombatValueSearch(model, CombatValueSearchConfig(
        depth=1, beam_width=2, policy_top_k=1, node_budget=8,
    )).search(env, obs, mask)
    assert mask[result.action]


def test_formal_eval_loads_legacy_checkpoint(legacy_checkpoint, monkeypatch):
    class TwoStepEnv(SlayTheSpireEnv):
        def reset(self, **kwargs):
            self.test_steps = 0
            return super().reset(**kwargs)

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            self.test_steps += 1
            return obs, reward, terminated, truncated or self.test_steps == 2, info

    monkeypatch.setattr(evaluate, "SlayTheSpireEnv", TwoStepEnv)
    monkeypatch.setattr(sys, "argv", ["rl.evaluation.eval", str(legacy_checkpoint),
                                    "--episodes", "1", "--device", "cpu"])
    evaluate.main()


def test_adapter_rejects_missing_required_field(legacy_checkpoint):
    model = MaskablePPO.load(legacy_checkpoint, device="cpu")
    obs = model.observation_space.sample()
    del obs["phase"]
    with pytest.raises(ValueError, match="phase"):
        observation_for_policy(model, obs)


def test_legacy_weights_migrate_to_joint_with_named_feature_columns(legacy_checkpoint, combat_factory, tmp_path):
    import torch
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.training.trainer import _warm_start_policy
    from env.core import native_actions as C
    source = MaskablePPO.load(legacy_checkpoint, device="cpu")
    env = combat_factory(["STRIKE_RED"] * 5, combat_action_contract="joint-v2")
    target = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, n_steps=2, batch_size=2,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4), seed=43, device="cpu")
    assert _warm_start_policy(target, source, allow_combat_contract_migration=True) > 0
    assert target.policy.optimizer.state == {}
    assert target._nosl_is_az_replay is None
    for key, value in source.policy.state_dict().items():
        if not key.startswith("core.combat.encoder.global_"):
            torch.testing.assert_close(target.policy.state_dict()[key], value, rtol=0, atol=0)
    encoder = target.policy.core.combat.encoder
    offset = sum(np.prod(encoder.observation_space[k].shape) for k in
                 encoder.global_scalar_keys[:encoder.global_scalar_keys.index("post_enemy_turn_boundary")])
    assert torch.count_nonzero(encoder.global_projection.weight[:, int(offset)]) == 0
    path = tmp_path / "joint-migration.zip"
    target.save(path)
    loaded = AccumulatingMaskablePPO.load(path, env=env, device="cpu")
    mask = env.action_masks()
    action = int(loaded.predict(env._obs(), action_masks=mask, deterministic=True)[0])
    assert mask[action] and action >= C.JOINT_END
    assert loaded.combat_action_contract == "joint-v2"
