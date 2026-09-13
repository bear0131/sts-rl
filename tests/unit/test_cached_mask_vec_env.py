from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import VecMonitor

from rl.training.cached_mask_vec_env import (
    CachedMaskSubprocVecEnv,
    GroupedCachedMaskSubprocVecEnv,
)
from rl.training.distributed_sb3 import build_parser as build_distributed_parser


class _MaskInfoEnv(gym.Env):
    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(0, 2, shape=(1,), dtype=np.int64)

    def __init__(self) -> None:
        self.position = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        return np.array([0]), {"action_mask": np.array([True, False])}

    def step(self, action):
        self.position += 1
        done = self.position == 2
        mask = np.array([False, not done])
        return np.array([self.position]), 0.0, done, False, {
            "action_mask": mask
        }

    def action_masks(self):
        raise AssertionError("cached info mask should avoid worker method calls")


def _make_mask_info_env():
    return _MaskInfoEnv()


def test_cached_masks_follow_step_and_autoreset_info():
    base = CachedMaskSubprocVecEnv(
        [_make_mask_info_env], start_method="spawn"
    )
    env = VecMonitor(base)
    try:
        env.reset()
        np.testing.assert_array_equal(get_action_masks(env), [[True, False]])

        env.step(np.array([0]))
        np.testing.assert_array_equal(get_action_masks(env), [[False, True]])

        _obs, _reward, done, _info = env.step(np.array([1]))
        assert done.tolist() == [True]
        np.testing.assert_array_equal(get_action_masks(env), [[True, False]])
    finally:
        env.close()


def test_grouped_cached_masks_preserve_logical_order():
    base = GroupedCachedMaskSubprocVecEnv(
        [_make_mask_info_env, _make_mask_info_env, _make_mask_info_env],
        envs_per_worker=2,
        start_method="spawn",
    )
    env = VecMonitor(base)
    try:
        assert len(base.processes) == 2
        env.reset()
        np.testing.assert_array_equal(
            get_action_masks(env),
            [[True, False], [True, False], [True, False]],
        )
        env.step(np.array([0, 0, 0]))
        np.testing.assert_array_equal(
            get_action_masks(env),
            [[False, True], [False, True], [False, True]],
        )
        # Distinct calls in one group must be routed back to their logical
        # indices without exposing the group's local numbering.
        values = base.env_method_batch(
            "get_wrapper_attr",
            {
                0: (("position",), {}),
                1: (("position",), {}),
                2: (("position",), {}),
            },
        )
        assert values == {0: 1, 1: 1, 2: 1}
    finally:
        env.close()


def test_distributed_parser_accepts_grouped_worker_geometry():
    args = build_distributed_parser().parse_args([
        "--rank", "0",
        "--world-size", "1",
        "--checkpoint", "model.zip",
        "--n-envs", "288",
        "--envs-per-worker", "12",
        "--nosl-is-az-replay-max-old-fraction", "0.9",
    ])
    assert args.n_envs == 288
    assert args.envs_per_worker == 12
    assert args.nosl_is_az_replay_max_old_fraction == 0.9
