"""Real PPO updates, optimizer persistence, and same-machine determinism."""
import numpy as np
import pytest
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from rl.training.reward import ShapedRewardWrapper
from rl.training.environments import TrainingStateWrapper
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy

pytestmark = [pytest.mark.native, pytest.mark.training]


def train_once(hierarchical, env_type=SlayTheSpireEnv, **algorithm_kwargs):
    # Force autoresets/truncation through the same wrapper ordering as WholeRun.
    env = DummyVecEnv([lambda: TrainingStateWrapper(TimeLimit(
        ShapedRewardWrapper(env_type()), max_episode_steps=12,
    ))] * 2)
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy, env, n_steps=16, batch_size=8, n_epochs=1,
        gradient_accumulation_steps=2, hierarchical_combat_hp=hierarchical,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4), seed=42, device="cpu",
        **algorithm_kwargs,
    )
    initial = {k: v.clone() for k, v in model.policy.state_dict().items()}
    model.learn(64)
    assert model.num_timesteps == 64
    assert any(not torch.equal(v, initial[k]) for k, v in model.policy.state_dict().items())
    if algorithm_kwargs.get("nosl_is_az"):
        assert any(k.startswith("core.combat.") and not torch.equal(v, initial[k])
                   for k, v in model.policy.state_dict().items())
    assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
    return model, env


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("hierarchical", [False, True])
def test_optimizer_checkpoint_and_prediction_roundtrip(tmp_path, hierarchical):
    model, env = train_once(hierarchical)
    try:
        checkpoint = tmp_path / "smoke.zip"
        model.save(checkpoint)
        restored = AccumulatingMaskablePPO.load(checkpoint, device="cpu")
        assert_nested_equal(model.policy.state_dict(), restored.policy.state_dict())
        assert_nested_equal(model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict())
        obs = env.reset()
        masks = np.stack(env.env_method("action_masks"))
        a = model.predict(obs, deterministic=True, action_masks=masks)[0]
        b = restored.predict(obs, deterministic=True, action_masks=masks)[0]
        np.testing.assert_array_equal(a, b)
    finally:
        env.close()


@pytest.mark.parametrize("hierarchical", [False, True])
def test_repeated_cpu_training_is_bitwise_equal(hierarchical):
    first, env1 = train_once(hierarchical)
    try:
        second, env2 = train_once(hierarchical)
        try:
            assert_nested_equal(first.policy.state_dict(), second.policy.state_dict())
            assert_nested_equal(first.policy.optimizer.state_dict(), second.policy.optimizer.state_dict())
            np.testing.assert_array_equal(first.rollout_buffer.actions, second.rollout_buffer.actions)
            np.testing.assert_array_equal(first.rollout_buffer.rewards, second.rollout_buffer.rewards)
        finally:
            env2.close()
    finally:
        env1.close()


@pytest.mark.parametrize("mode,strict", [("puct", False), ("full_gumbel", True)])
def test_native_nosl_search_targets_reach_learner(mode, strict, short_combat_env_type):
    model, env = train_once(
        True, env_type=short_combat_env_type,
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_depth=0,
        nosl_is_az_simulations=4, nosl_is_az_replay_capacity=128,
        nosl_is_az_micro_batch_size=8, nosl_is_az_leaf_batch_size=4,
        nosl_is_az_search_mode=mode, nosl_is_az_paper_full_gumbel=strict,
    )
    try:
        replay = model._nosl_is_az_replay
        assert replay is not None and replay.size > 0
        assert model.nosl_is_az_leaf_batch_size == (1 if strict else 4)
        assert getattr(model, "nosl_is_az_gumbel_interior_min_visits", 1) == (
            0 if strict else 1
        )
        assert replay.total_added >= replay.size
        assert np.isfinite(replay.value_targets[:replay.size]).all()
        assert model.policy.optimizer.state
        hp_order = model._nosl_is_az_rollout_hp_order
        assert hp_order["count"] > 0
        assert 0 <= hp_order["matches"] <= hp_order["count"]
        assert hp_order["fraction"] == pytest.approx(
            hp_order["matches"] / hp_order["count"]
        )
    finally:
        env.close()


@pytest.mark.parametrize("end_turn_as_leaf", [False, True])
def test_real_value_boundaries_reach_replay_independent_of_search_leaf_mode(
    short_combat_env_type, end_turn_as_leaf
):
    class BoundaryThenWinEnv(short_combat_env_type):
        def __init__(self):
            super().__init__()
            self._force_end_turn = False

        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self.thick.bc.apply_custom_state({"monsters": [{"hp": 1}]})
            self._force_end_turn = True
            self._invalidate_backend_cache()
            return self._obs(), self._info()

        def action_masks(self):
            mask = super().action_masks()
            if self._phase() != "combat" or self._pending is not None:
                return mask
            restricted = np.zeros_like(mask)
            if self._force_end_turn:
                restricted[A.END_TURN] = True
                return restricted
            raw = self.thick.bc.to_dict()
            attack_index = next(
                index
                for index, card in enumerate(raw["hand"])
                if card[0] in {"STRIKE_RED", "BASH"}
            )
            restricted[A._PLAY + attack_index] = True
            return restricted

        def step(self, action):
            result = super().step(action)
            if "nosl_value_boundary_observation" in result[4]:
                self._force_end_turn = False
                self._invalidate_thin_cache()
            return result

    env = DummyVecEnv([BoundaryThenWinEnv])
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy,
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        gradient_accumulation_steps=1,
        hierarchical_combat_hp=True,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4),
        seed=42,
        device="cpu",
        nosl_is_az=True,
        nosl_is_az_cpp=True,
        nosl_is_az_depth=0,
        nosl_is_az_simulations=1,
        nosl_is_az_replay_capacity=32,
        nosl_is_az_micro_batch_size=4,
        nosl_is_az_leaf_batch_size=1,
        nosl_is_az_end_turn_as_leaf=end_turn_as_leaf,
    )
    try:
        model.learn(4)
        replay = model._nosl_is_az_replay
        assert replay is not None and replay.size >= 5
        weights = replay.policy_weights[:replay.size]
        value_only = np.flatnonzero(weights == 0)
        assert value_only.size == 2
        boundaries = replay.observations["post_enemy_turn_boundary"][value_only, 0]
        assert sorted(boundaries.tolist()) == [0.0, 1.0]
        present = replay.observations["monster_scalars"][value_only, :, 0].sum(axis=1)
        assert sorted(present.tolist()) == [0.0, 1.0]
        assert not replay.action_masks[value_only].any()
        assert not replay.policy_targets[value_only].any()
    finally:
        env.close()
