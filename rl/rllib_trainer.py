"""Distributed RLlib PPO trainer for the whole-run STS environment.

This is an alternative backend to ``rl.trainer``.  It reuses the environment,
reward, observation encoder, and StSPolicyCore, while RLlib owns distributed
sampling, multi-GPU PPO updates, metrics, and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from rl.pointer_policy import StSPolicyCore
from rl.reward import ShapedRewardWrapper
from rl.tasks.wholerun import _MAX_EPISODE_STEPS


ENV_NAME = "sts-whole-run-masked-v1"
MODEL_NAME = "sts-pointer-core-v1"
MASK_LOG_FLOOR = -1.0e9


class RLLibActionMaskEnv(gym.Wrapper):
    """Expose RLlib's standard ``action_mask + observations`` contract."""

    def __init__(self, config=None):
        config = dict(config or {})
        env = SlayTheSpireEnv(
            ascension=int(config.get("ascension", 0)),
            global_map_view=bool(config.get("global_map_view", True)),
            card_language=bool(config.get("card_language", False)),
            combat_language=bool(config.get("combat_language", False)),
            deck_size_observation=bool(
                config.get("deck_size_observation", True)
            ),
        )
        env = ShapedRewardWrapper(env)
        env = TimeLimit(env, max_episode_steps=_MAX_EPISODE_STEPS)
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "observations": env.observation_space,
            "action_mask": spaces.Box(
                low=0.0, high=1.0, shape=(A.N_ACTIONS,), dtype=np.float32
            ),
        })

    def _masked(self, observation):
        mask = self.env.get_wrapper_attr("action_masks")()
        return {
            "observations": observation,
            "action_mask": np.asarray(mask, dtype=np.float32),
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._masked(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._masked(observation), reward, terminated, truncated, info
        )


class StSRLLibTorchModel(TorchModelV2, torch.nn.Module):
    """RLlib adapter around the same pointer actor-critic used by SB3."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 **kwargs):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)
        if num_outputs != A.N_ACTIONS:
            raise ValueError(
                f"RLlib requested {num_outputs} logits, expected {A.N_ACTIONS}"
            )
        base_space = obs_space["observations"]
        cfg = model_config.get("custom_model_config", {})
        self.core = StSPolicyCore(
            base_space,
            d_model=int(cfg.get("d_model", 256)),
            n_layers=int(cfg.get("n_layers", 4)),
            n_heads=int(cfg.get("n_heads", 8)),
            card_language=bool(cfg.get("card_language", False)),
            combat_language=bool(cfg.get("combat_language", False)),
            map_dag=bool(cfg.get("map_dag", True)),
        )
        self._last_values = None

    def forward(self, input_dict, state, seq_lens):
        wrapped = input_dict["obs"]
        logits, values = self.core.actor_critic(wrapped["observations"])
        mask = wrapped["action_mask"].to(logits.dtype)
        logits = logits.masked_fill(mask <= 0.0, MASK_LOG_FLOOR)
        self._last_values = values.reshape(-1)
        return logits, state

    def value_function(self):
        if self._last_values is None:
            raise RuntimeError("value_function called before forward")
        return self._last_values


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train whole-run PPO with RLlib distributed sampling."
    )
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--envs-per-runner", type=int, default=4)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--rollout-fragment-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-coeff", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--global-map-view", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--map-dag", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--deck-size-observation",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--card-language", action="store_true")
    parser.add_argument("--combat-language", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("rl/runs/rllib"))
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save every N RLlib training iterations.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--ray-address", default=None,
                        help="Ray cluster address; omit to start a local cluster.")
    return parser


def _sampled_steps(result):
    return int(
        result.get("num_env_steps_sampled_lifetime")
        or result.get("num_env_steps_sampled")
        or result.get("timesteps_total")
        or 0
    )


def _metric(result, *paths, default=None):
    for path in paths:
        value = result
        for key in path.split("/"):
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    return default


def _summary(result, elapsed):
    sample_ms = _metric(result, "timers/sample_time_ms", default=0.0) or 0.0
    learn_ms = _metric(result, "timers/learn_time_ms", default=0.0) or 0.0
    sync_ms = _metric(
        result, "timers/synch_weights_time_ms", default=0.0
    ) or 0.0
    summary = {
        "iteration": result.get("training_iteration"),
        "steps": _sampled_steps(result),
        "episode_return_mean": _metric(
            result,
            "env_runners/episode_return_mean",
            "episode_reward_mean",
        ),
        "episode_len_mean": _metric(
            result,
            "env_runners/episode_len_mean",
            "episode_len_mean",
        ),
        "sample_throughput": _metric(
            result,
            "env_runners/num_env_steps_sampled_per_second",
            "num_env_steps_sampled_throughput_per_sec",
            "num_env_steps_sampled_per_second",
        ),
        "sample_seconds": round(float(sample_ms) / 1000.0, 3),
        "learn_seconds": round(float(learn_ms) / 1000.0, 3),
        "sync_weights_seconds": round(float(sync_ms) / 1000.0, 3),
        "cpu_percent": _metric(result, "perf/cpu_util_percent"),
        "iteration_seconds": round(elapsed, 3),
    }
    summary["bottleneck"] = (
        "sampling/cpu" if sample_ms >= learn_ms else "learning/gpu"
    )
    return summary


def main():
    args = build_parser().parse_args()
    if args.train_batch_size % args.minibatch_size:
        raise ValueError("--minibatch-size must divide --train-batch-size")

    register_env(ENV_NAME, lambda config: RLLibActionMaskEnv(config))
    ModelCatalog.register_custom_model(MODEL_NAME, StSRLLibTorchModel)
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    run_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    algo = None
    try:
        if args.resume:
            algo = Algorithm.from_checkpoint(str(args.resume.resolve()))
        else:
            env_config = {
                "ascension": args.ascension,
                "global_map_view": args.global_map_view,
                "card_language": args.card_language,
                "combat_language": args.combat_language,
                "deck_size_observation": args.deck_size_observation,
            }
            config = (
                PPOConfig()
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .environment(
                    ENV_NAME,
                    env_config=env_config,
                    disable_env_checking=False,
                )
                .framework("torch")
                .experimental(_disable_preprocessor_api=True)
                .env_runners(
                    num_env_runners=args.num_env_runners,
                    num_envs_per_env_runner=args.envs_per_runner,
                    rollout_fragment_length=args.rollout_fragment_length,
                    batch_mode="truncate_episodes",
                )
                .resources(num_gpus=args.num_gpus)
                .debugging(seed=args.seed, log_sys_usage=True)
                .training(
                    gamma=args.gamma,
                    lambda_=args.gae_lambda,
                    lr=args.learning_rate,
                    clip_param=args.clip_range,
                    entropy_coeff=args.entropy_coeff,
                    vf_loss_coeff=args.vf_coeff,
                    train_batch_size=args.train_batch_size,
                    minibatch_size=args.minibatch_size,
                    num_epochs=args.num_epochs,
                    model={
                        "custom_model": MODEL_NAME,
                        "custom_model_config": {
                            "d_model": args.d_model,
                            "n_layers": args.n_layers,
                            "n_heads": args.n_heads,
                            "card_language": args.card_language,
                            "combat_language": args.combat_language,
                            "map_dag": args.map_dag,
                        },
                    },
                )
            )
            algo = config.build_algo()

        while True:
            started = time.perf_counter()
            result = algo.train()
            summary = _summary(result, time.perf_counter() - started)
            print("[rllib] " + json.dumps(summary, ensure_ascii=False), flush=True)
            iteration = int(result.get("training_iteration", 0))
            if args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
                saved = algo.save(str(run_dir / "checkpoints"))
                print(f"[rllib] checkpoint={saved.checkpoint.path}", flush=True)
            if summary["steps"] >= args.steps:
                break

        saved = algo.save(str(run_dir / "final"))
        print(f"[rllib] final={saved.checkpoint.path}", flush=True)
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
