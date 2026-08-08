"""Distributed RLlib PPO trainer for the whole-run STS environment.

This is an alternative backend to ``rl.trainer``.  It reuses the environment,
reward, observation encoder, and StSPolicyCore, while RLlib owns distributed
sampling, multi-GPU PPO updates, metrics, and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.torch_policy import (
    TorchPolicy,
    _directStepOptimizerSingleton,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.tune.registry import register_env

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from rl.pointer_policy import StSPolicyCore
from rl.reward import ShapedRewardWrapper
from rl.tasks.wholerun import _MAX_EPISODE_STEPS


ENV_NAME = "sts-whole-run-masked-v1"
MODEL_NAME = "sts-pointer-core-v1"
MASK_LOG_FLOOR = -1.0e9
RAY_RUNTIME_EXCLUDES = [
    ".git/",
    ".venv/",
    "venv/",
    "wandb/",
    "rl/runs/",
    "rl/bughunt_out/",
    "bridge/client/logs/",
    "**/__pycache__/",
    "*.zip",
    "sim/sts_lightspeed/build/CMakeFiles/",
    "sim/sts_lightspeed/build/**/*.o",
    "sim/sts_lightspeed/build/**/*.obj",
    "sim/sts_lightspeed/build/*.exe",
    "sim/sts_lightspeed/build/*.dll",
    "sim/sts_lightspeed/build-combat-language/",
    "sim/sts_lightspeed/cmake-build-*/",
]


def _fragment_length(value):
    if value == "auto":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("rollout fragment length must be positive")
    return parsed


def _ray_runtime_env():
    """Package worker code without historical runs or checkpoint archives."""
    # The simulator extension is intentionally gitignored but must be shipped
    # to Ray workers, so use our explicit excludes instead of .gitignore.
    os.environ.setdefault("RAY_RUNTIME_ENV_IGNORE_GITIGNORE", "1")
    return {
        "working_dir": str(Path(__file__).resolve().parents[1]),
        "excludes": RAY_RUNTIME_EXCLUDES,
        "env_vars": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
        },
    }


class StSRLLibCallbacks(DefaultCallbacks):
    """Publish the same whole-run outcome metrics used by the SB3 trainer."""

    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for()
        stats = info.get("run_stats") if info else None
        if stats is None:
            return
        act = int(stats["act"])
        floor = int(stats["floor"])
        won = act > 3 or stats["outcome"] == "PLAYER_VICTORY"
        episode.custom_metrics["game/win_rate"] = float(won)
        episode.custom_metrics["game/floor"] = floor
        episode.custom_metrics["game/deck_size"] = int(stats["deck_size"])
        for number in (1, 2, 3):
            episode.custom_metrics[f"game/act_{number}_won"] = float(act > number)
        for threshold in range(5, 55, 5):
            episode.custom_metrics[f"game/floor_won/{threshold:02d}"] = float(
                floor > threshold
            )


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
            neow_observation=bool(config.get("neow_observation", True)),
            event_option_observation=bool(
                config.get("event_option_observation", True)
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
    parser.add_argument(
        "--task", choices=["wholerun"], default="wholerun",
        help="Compatibility with rl.trainer; RLlib currently trains wholerun.",
    )
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--envs-per-runner", type=int, default=4)
    parser.add_argument(
        "--num-cpus-per-env-runner", type=int, default=1,
        help="Ray CPU slots reserved for each remote env-runner actor.",
    )
    parser.add_argument(
        "--max-env-runners", type=int, default=8,
        help="Maximum remote env-runner actors before folding extra envs into "
             "vectorized runners. Use 0 to disable this single-node cap.",
    )
    parser.add_argument(
        "--num-cpus-for-main-process", type=int, default=0,
        help="Ray CPU slots reserved for the local driver/learner process. "
             "Use 0 on one-GPU boxes to let all CPUs host env runners.",
    )
    parser.add_argument(
        "--create-local-env-runner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also create a local env runner in the driver process. Off by "
             "default to avoid a hidden extra sampler competing with learning.",
    )
    parser.add_argument(
        "--sample-timeout-s", type=float, default=120.0,
        help="Seconds RLlib waits for slow env-runner sample requests.",
    )
    parser.add_argument(
        "--auto-fragment-max", type=int, default=64,
        help="Largest rollout fragment length chosen by --rollout-fragment-length "
             "auto. Smaller fragments keep slow map-DAG samplers returning "
             "before RLlib's sample timeout.",
    )
    parser.add_argument(
        "--max-requests-in-flight-per-env-runner", type=int, default=1,
        help="Limit queued sample requests per runner to keep CPU and memory "
             "pressure predictable.",
    )
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument(
        "--rollout-fragment-length", type=_fragment_length, default="auto",
        help="Steps sampled per runner fragment; auto matches train batch size.",
    )
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--minibatch-size", "--batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", "--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", "--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coeff", "--vf-coef", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--global-map-view", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--map-dag", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--deck-size-observation",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neow-observation",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event-option-observation",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--card-language", action="store_true")
    parser.add_argument("--combat-language", action="store_true")
    parser.add_argument(
        "--arch", choices=["auto", "transformer-pointer"], default="auto",
        help="RLlib uses the shared transformer-pointer policy core.",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("rl/runs/rllib"))
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save every N RLlib training iterations.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="Warm-start policy weights from an SB3 .zip checkpoint.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-grad-freq", type=int, default=0)
    parser.add_argument("--wandb-save-freq", type=int, default=50_000)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--profile-performance", action="store_true")
    parser.add_argument("--metrics-window", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-every", type=int, default=None,
        help="SB3-compatible checkpoint interval in sampled environment steps.",
    )
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--vec", choices=["subproc", "dummy"], default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--ray-address", default=None,
                        help="Ray cluster address; omit to start a local cluster.")
    return parser


def _load_sb3_checkpoint(path: Path, device: str):
    """Load and describe a pointer-policy SB3 checkpoint for migration."""
    from rl.accumulating_ppo import AccumulatingMaskablePPO
    from rl.model import (
        uses_card_language,
        uses_combat_language,
        uses_deck_size_observation,
        uses_event_option_observation,
        uses_global_map_observation,
        uses_map_dag,
        uses_neow_observation,
    )
    from rl.pointer_policy import StSPointerPolicy

    if not path.is_file():
        raise FileNotFoundError(f"SB3 checkpoint not found: {path}")
    model = AccumulatingMaskablePPO.load(str(path), device=device)
    if not isinstance(model.policy, StSPointerPolicy):
        raise ValueError(
            "RLlib can migrate only SB3 transformer-pointer checkpoints; "
            f"{path} contains {type(model.policy).__name__}"
        )
    cfg = dict(getattr(model.policy, "_core_cfg", {}))
    return SimpleNamespace(
        model=model,
        core=model.policy.core,
        num_timesteps=int(model.num_timesteps),
        global_map_view=uses_global_map_observation(model),
        map_dag=uses_map_dag(model),
        deck_size_observation=uses_deck_size_observation(model),
        neow_observation=uses_neow_observation(model),
        event_option_observation=uses_event_option_observation(model),
        card_language=uses_card_language(model),
        combat_language=uses_combat_language(model),
        d_model=int(cfg.get("d_model", 256)),
        n_layers=int(cfg.get("n_layers", 4)),
        n_heads=int(cfg.get("n_heads", 8)),
    )


def _apply_sb3_weights(algo, checkpoint, freeze_encoder=False):
    """Copy a shared StSPolicyCore from SB3 into RLlib and sync samplers."""
    core = algo.get_policy().model.core
    source = checkpoint.core.state_dict()
    target = core.state_dict()
    incompatible = []
    initialized = []
    for key, value in source.items():
        if key not in target:
            incompatible.append(f"unexpected source parameter {key}")
            continue
        if value.shape == target[key].shape:
            target[key] = value
            continue
        is_appended_role = (
            key.endswith("role_emb.weight")
            and value.ndim == 2
            and target[key].ndim == 2
            and target[key].shape[0] >= value.shape[0]
            and target[key].shape[1] == value.shape[1]
        )
        if is_appended_role:
            target[key][:value.shape[0]] = value
            initialized.append(f"{key}[{value.shape[0]}:]")
            continue
        is_appended_global_scalar = (
            key.endswith("tok.global.weight")
            and value.ndim == 2
            and target[key].ndim == 2
            and target[key].shape[0] == value.shape[0]
            and target[key].shape[1] == value.shape[1] + 1
        )
        if is_appended_global_scalar:
            target[key][:, :value.shape[1]] = value
            initialized.append(f"{key}[:, -1]")
            continue
        incompatible.append(
            f"shape mismatch for {key}: checkpoint {tuple(value.shape)}, "
            f"target {tuple(target[key].shape)}"
        )
    for key in target.keys() - source.keys():
        is_neow = any(marker in key for marker in (
            ".shared.neow_bonus.",
            ".shared.neow_drawback.",
            ".tok.neow.",
            ".pos.neow",
        ))
        is_event_option = any(marker in key for marker in (
            ".shared.event_option.",
            ".tok.event_option.",
            ".pos.event_option",
        ))
        is_card_language = any(marker in key for marker in (
            ".shared.card.token_embedding.",
            ".shared.card.local_encoder.",
            ".shared.card.program_position",
            ".shared.card.effect_scale",
            ".pile_card_language.",
            ".exhaust_card_language.",
        ))
        is_combat_language = ".combat_language" in key
        is_map_dag = ".map_dag." in key or key.endswith(".map_dag")
        if (
            is_neow or is_event_option or is_card_language
            or is_combat_language or is_map_dag
        ):
            initialized.append(key)
        else:
            incompatible.append(f"missing source parameter {key}")
    if incompatible:
        details = "\n  - ".join(incompatible)
        raise ValueError(
            "SB3 checkpoint geometry does not match the requested RLlib model: "
            f"\n  - {details}"
        )
    core.load_state_dict(target)
    if initialized:
        print(
            f"[rllib] initialized {len(initialized)} new policy tensors",
            flush=True,
        )
    if freeze_encoder:
        for parameter in core.encoder.parameters():
            parameter.requires_grad_(False)
    runner_group = getattr(algo, "env_runner_group", None)
    if runner_group is None:
        runner_group = getattr(algo, "workers", None)
        if callable(runner_group):
            runner_group = runner_group()
    if runner_group is not None:
        runner_group.sync_weights()


def _accumulating_apply_gradients(policy, gradients):
    """Average micro-batch gradients before one RLlib optimizer step."""
    if gradients != _directStepOptimizerSingleton:
        raise ValueError("gradient accumulation requires direct optimizer steps")

    parameters = list(policy.model.parameters())
    if policy._sts_accumulated_gradients is None:
        policy._sts_accumulated_gradients = [
            None if parameter.grad is None else torch.zeros_like(parameter.grad)
            for parameter in parameters
        ]
    for index, parameter in enumerate(parameters):
        if parameter.grad is None:
            continue
        buffer = policy._sts_accumulated_gradients[index]
        if buffer is None:
            buffer = torch.zeros_like(parameter.grad)
            policy._sts_accumulated_gradients[index] = buffer
        buffer.add_(parameter.grad.detach())

    policy._sts_accumulated_micro_batches += 1
    if policy._sts_accumulated_micro_batches < policy._sts_accumulation_steps:
        return

    scale = 1.0 / policy._sts_accumulated_micro_batches
    for parameter, accumulated in zip(
        parameters, policy._sts_accumulated_gradients
    ):
        parameter.grad = None if accumulated is None else accumulated.mul(scale)
    policy_base = (
        TorchPolicyV2 if isinstance(policy, TorchPolicyV2) else TorchPolicy
    )
    policy_base.apply_gradients(policy, _directStepOptimizerSingleton)
    policy._sts_optimizer_steps += 1
    policy._sts_accumulated_micro_batches = 0
    policy._sts_accumulated_gradients = None


def _install_gradient_accumulation(algo, steps):
    if steps == 1:
        return
    policy = algo.get_policy()
    if not isinstance(policy, (TorchPolicy, TorchPolicyV2)):
        raise TypeError(
            "--gradient-accumulation-steps requires RLlib TorchPolicy"
        )
    policy._sts_accumulation_steps = steps
    policy._sts_accumulated_micro_batches = 0
    policy._sts_accumulated_gradients = None
    policy._sts_optimizer_steps = 0
    policy.apply_gradients = MethodType(_accumulating_apply_gradients, policy)
    print(
        f"[rllib] accumulating {steps} micro-batches per optimizer step",
        flush=True,
    )


def _normalize_compat_args(args):
    """Translate rl.trainer spellings and checkpoint-derived model geometry."""
    if args.init_from and args.resume:
        raise ValueError("--init-from and --resume are mutually exclusive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.n_envs is not None:
        args.num_env_runners = args.n_envs if args.vec != "dummy" else 0
        args.envs_per_runner = 1 if args.vec != "dummy" else args.n_envs
    if args.num_env_runners <= 0 and not args.create_local_env_runner:
        args.create_local_env_runner = True
        print(
            "[rllib] enabled --create-local-env-runner because no remote "
            "env runners were requested",
            flush=True,
        )
    if args.n_steps is not None:
        args.rollout_fragment_length = args.n_steps
    if args.gradient_accumulation_steps > 1:
        args.train_batch_size = (
            args.minibatch_size * args.gradient_accumulation_steps
        )
    if args.device == "cpu":
        args.num_gpus = 0.0

    sb3_source = None
    sb3_resume_steps = 0
    sb3_path = args.init_from
    if args.resume and args.resume.suffix.lower() == ".zip":
        sb3_path = args.resume
    if sb3_path:
        load_device = "cpu" if args.device == "auto" else args.device
        sb3_source = _load_sb3_checkpoint(sb3_path, load_device)
        sb3_resume_steps = sb3_source.num_timesteps if args.resume else 0
        for name in (
            "global_map_view", "map_dag", "deck_size_observation",
            "neow_observation", "event_option_observation",
            "card_language", "combat_language", "d_model", "n_layers",
            "n_heads",
        ):
            old = getattr(args, name)
            new = getattr(sb3_source, name)
            if args.init_from and isinstance(old, bool) and isinstance(new, bool):
                new = old or new
            if old != new:
                print(f"[rllib] inferred --{name.replace('_', '-')}={new} from {sb3_path}")
                setattr(args, name, new)
    return sb3_source, sb3_resume_steps


def _plan_distribution(args):
    """Choose a runner layout that fits the available CPU budget.

    RLlib env runners are Ray actors, so too many 1-env actors can stall a
    single machine by exhausting CPU slots and increasing scheduling overhead.
    We keep the total env count stable and fold excess runners back into a
    smaller number of vectorized runners when needed.
    """
    cpu_total = int(ray.cluster_resources().get("CPU", os.cpu_count() or 1))
    runner_cpu = max(1, int(args.num_cpus_per_env_runner))
    main_cpu = max(0, int(args.num_cpus_for_main_process))
    requested_runners = max(0, int(args.num_env_runners))
    requested_envs_per_runner = max(1, int(args.envs_per_runner))
    requested_total_envs = requested_runners * requested_envs_per_runner
    if requested_runners == 0:
        args.envs_per_runner = requested_envs_per_runner
        return {
            "cpu_total": cpu_total,
            "runner_cpu": runner_cpu,
            "main_cpu": main_cpu,
            "max_env_runners": 0,
            "requested_total_envs": requested_envs_per_runner,
            "planned_total_envs": requested_envs_per_runner,
        }
    actor_cap = (
        requested_runners if args.max_env_runners <= 0
        else max(1, int(args.max_env_runners))
    )

    # Keep one CPU free for the driver only if the user explicitly asked for it.
    usable_cpu_slots = max(1, cpu_total - main_cpu)
    max_runners = min(actor_cap, max(1, usable_cpu_slots // runner_cpu))
    planned_runners = min(requested_runners, max_runners)
    planned_envs_per_runner = max(
        requested_envs_per_runner,
        (requested_total_envs + planned_runners - 1) // planned_runners,
    )

    changed = (
        planned_runners != requested_runners
        or planned_envs_per_runner != requested_envs_per_runner
    )
    if changed:
        print(
            "[rllib] adjusted env layout from "
            f"{requested_runners}x{requested_envs_per_runner} "
            f"to {planned_runners}x{planned_envs_per_runner} "
            f"for CPU budget {cpu_total} and actor cap {actor_cap} "
            f"(main={main_cpu}, runner={runner_cpu})",
            flush=True,
        )

    args.num_env_runners = planned_runners
    args.envs_per_runner = planned_envs_per_runner
    return {
        "cpu_total": cpu_total,
        "runner_cpu": runner_cpu,
        "main_cpu": main_cpu,
        "max_env_runners": actor_cap,
        "requested_total_envs": requested_total_envs,
        "planned_total_envs": planned_runners * planned_envs_per_runner,
    }


def _set_auto_rollout_fragment_length(args):
    if args.rollout_fragment_length != "auto":
        return
    sampling_width = max(1, args.num_env_runners) * args.envs_per_runner
    if args.train_batch_size % sampling_width:
        return
    samples_per_runner = args.train_batch_size // sampling_width
    max_fragment = max(1, int(args.auto_fragment_max))
    args.rollout_fragment_length = next(
        fragment
        for fragment in range(min(max_fragment, samples_per_runner), 0, -1)
        if samples_per_runner % fragment == 0
    )
    print(
        "[rllib] auto rollout fragment length="
        f"{args.rollout_fragment_length} ({sampling_width} concurrent envs)",
        flush=True,
    )


def _sampled_steps(result):
    return int(_number_metric(
        result,
        "num_env_steps_sampled_lifetime",
        "env_runners/num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "env_runners/num_env_steps_sampled",
        "timesteps_total",
        "env_runners/timesteps_total",
        default=0,
    ) or 0)


def _trained_steps(result):
    return int(_number_metric(
        result,
        "num_env_steps_trained_lifetime",
        "learners/default_policy/num_env_steps_trained_lifetime",
        "info/num_env_steps_trained",
        "num_agent_steps_trained_lifetime",
        default=0,
    ) or 0)


def _agent_steps(result):
    return int(_number_metric(
        result,
        "num_agent_steps_sampled_lifetime",
        "env_runners/num_agent_steps_sampled_lifetime",
        "num_agent_steps_sampled",
        "env_runners/num_agent_steps_sampled",
        default=0,
    ) or 0)


def _episodes_total(result):
    return int(_number_metric(
        result,
        "num_episodes_lifetime",
        "env_runners/num_episodes_lifetime",
        "episodes_total",
        "env_runners/episodes_total",
        "episodes_this_iter",
        "env_runners/num_episodes",
        default=0,
    ) or 0)


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


def _number_metric(result, *paths, default=None):
    value = _metric(result, *paths, default=default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary(result, elapsed):
    sample_ms = _number_metric(
        result, "timers/sample_time_ms", "timers/sample_time_ms_mean",
        default=0.0,
    ) or 0.0
    learn_ms = _number_metric(
        result, "timers/learn_time_ms", "timers/learn_time_ms_mean",
        default=0.0,
    ) or 0.0
    sync_ms = _number_metric(
        result, "timers/synch_weights_time_ms",
        "timers/synch_weights_time_ms_mean",
        default=0.0
    ) or 0.0
    sampled_steps = _sampled_steps(result)
    summary = {
        "iteration": result.get("training_iteration"),
        "steps": sampled_steps,
        "agent_steps": _agent_steps(result),
        "trained_steps": _trained_steps(result),
        "episodes_total": _episodes_total(result),
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
        "sample_throughput": _number_metric(
            result,
            "env_runners/num_env_steps_sampled_per_second",
            "env_runners/num_env_steps_sampled_throughput_per_sec",
            "env_runners/num_env_steps_sampled_lifetime_throughput_per_sec",
            "num_env_steps_sampled_throughput_per_sec",
            "num_env_steps_sampled_per_second",
        ),
        "sample_seconds": round(float(sample_ms) / 1000.0, 3),
        "learn_seconds": round(float(learn_ms) / 1000.0, 3),
        "sync_weights_seconds": round(float(sync_ms) / 1000.0, 3),
        "cpu_percent": _metric(result, "perf/cpu_util_percent"),
        "iteration_seconds": round(elapsed, 3),
    }
    # Match SB3's time/fps semantics: all sampled steps divided by the
    # complete train iteration wall time (sampling + learning + sync).
    previous_steps = getattr(_summary, "_previous_steps", None)
    iter_steps = _number_metric(
        result,
        "num_env_steps_sampled_this_iter",
        "env_runners/num_env_steps_sampled_this_iter",
        "env_runners/num_env_steps_sampled",
        default=None,
    )
    if iter_steps is None and previous_steps is not None:
        iter_steps = summary["steps"] - previous_steps
    if iter_steps is None:
        iter_steps = summary["steps"]
    iter_steps = max(0.0, float(iter_steps or 0.0))
    if summary["sample_throughput"] in (None, 0.0) and sample_ms > 0 and iter_steps > 0:
        summary["sample_throughput"] = round(iter_steps / (sample_ms / 1000.0), 2)
    summary["total_throughput"] = round(
        float(iter_steps) / elapsed, 2
    ) if elapsed > 0 else 0.0
    _summary._previous_steps = summary["steps"]
    summary["iteration_steps"] = int(iter_steps)
    summary["no_samples"] = iter_steps <= 0
    custom = _metric(result, "env_runners/custom_metrics", "custom_metrics",
                     default={}) or {}
    for source, destination in (
        ("game/win_rate_mean", "game/win_rate"),
        ("game/floor_mean", "game/floor_mean"),
        ("game/deck_size_mean", "game/deck_size_mean"),
        ("game/act_1_won_mean", "game/act_1_won_mean"),
        ("game/act_2_won_mean", "game/act_2_won_mean"),
        ("game/act_3_won_mean", "game/act_3_won_mean"),
    ):
        if source in custom:
            summary[destination] = custom[source]
    summary["bottleneck"] = (
        "sampling/cpu" if sample_ms >= learn_ms else "learning/gpu"
    )
    return summary


def main():
    args = build_parser().parse_args()
    sb3_source, sb3_resume_steps = _normalize_compat_args(args)
    if args.train_batch_size % args.minibatch_size:
        raise ValueError("--minibatch-size must divide --train-batch-size")
    micro_batches = args.train_batch_size // args.minibatch_size
    if micro_batches % args.gradient_accumulation_steps:
        raise ValueError(
            "--gradient-accumulation-steps must divide the number of "
            "micro-batches in --train-batch-size"
        )

    register_env(ENV_NAME, lambda config: RLLibActionMaskEnv(config))
    ModelCatalog.register_custom_model(MODEL_NAME, StSRLLibTorchModel)
    ray.init(
        address=args.ray_address,
        ignore_reinit_error=True,
        runtime_env=_ray_runtime_env(),
    )
    distribution = _plan_distribution(args)
    _set_auto_rollout_fragment_length(args)
    print(
        "[rllib] distribution="
        + json.dumps({
            "num_env_runners": args.num_env_runners,
            "envs_per_runner": args.envs_per_runner,
            "total_envs": args.num_env_runners * args.envs_per_runner,
            "rollout_fragment_length": args.rollout_fragment_length,
            "num_cpus_per_env_runner": args.num_cpus_per_env_runner,
            "num_cpus_for_main_process": args.num_cpus_for_main_process,
            "create_local_env_runner": args.create_local_env_runner,
            "sample_timeout_s": args.sample_timeout_s,
            "max_requests_in_flight_per_env_runner":
                args.max_requests_in_flight_per_env_runner,
            **distribution,
        }, ensure_ascii=False),
        flush=True,
    )

    run_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    algo = None
    wandb_run = None
    try:
        if args.wandb:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
                dir=str(run_dir),
            )

        if args.resume and sb3_source is None:
            algo = Algorithm.from_checkpoint(str(args.resume.resolve()))
        else:
            env_config = {
                "ascension": args.ascension,
                "global_map_view": args.global_map_view,
                "card_language": args.card_language,
                "combat_language": args.combat_language,
                "deck_size_observation": args.deck_size_observation,
                "neow_observation": args.neow_observation,
                "event_option_observation": args.event_option_observation,
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
                .callbacks(StSRLLibCallbacks)
                .experimental(_disable_preprocessor_api=True)
                .env_runners(
                    num_env_runners=args.num_env_runners,
                    create_local_env_runner=args.create_local_env_runner,
                    num_envs_per_env_runner=args.envs_per_runner,
                    num_cpus_per_env_runner=args.num_cpus_per_env_runner,
                    rollout_fragment_length=args.rollout_fragment_length,
                    batch_mode="truncate_episodes",
                    sample_timeout_s=args.sample_timeout_s,
                    max_requests_in_flight_per_env_runner=(
                        args.max_requests_in_flight_per_env_runner
                    ),
                )
                .resources(
                    num_cpus_for_main_process=args.num_cpus_for_main_process,
                    num_gpus=args.num_gpus,
                )
                .debugging(seed=args.seed, log_sys_usage=True)
                .reporting(
                    metrics_num_episodes_for_smoothing=args.metrics_window,
                )
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
            if sb3_source is not None:
                _apply_sb3_weights(algo, sb3_source, args.freeze_encoder)
                mode = "resume" if sb3_resume_steps else "warm-start"
                print(
                    f"[rllib] {mode} loaded {len(sb3_source.core.state_dict())} "
                    f"policy tensors from {args.resume or args.init_from}",
                    flush=True,
                )

        _install_gradient_accumulation(
            algo, args.gradient_accumulation_steps
        )

        if args.freeze_encoder and sb3_source is None:
            for parameter in algo.get_policy().model.core.encoder.parameters():
                parameter.requires_grad_(False)
        if wandb_run is not None and args.wandb_grad_freq > 0:
            wandb_run.watch(
                algo.get_policy().model,
                log="gradients",
                log_freq=args.wandb_grad_freq,
            )

        next_step_checkpoint = (
            args.save_every if args.save_every and args.save_every > 0 else None
        )
        next_wandb_checkpoint = (
            args.wandb_save_freq
            if wandb_run is not None and args.wandb_save_freq > 0
            else None
        )
        while True:
            started = time.perf_counter()
            result = algo.train()
            summary = _summary(result, time.perf_counter() - started)
            summary["steps"] += sb3_resume_steps
            if args.gradient_accumulation_steps > 1:
                policy = algo.get_policy()
                summary["train/effective_batch_size"] = args.train_batch_size
                summary["train/optimizer_steps"] = policy._sts_optimizer_steps
            print("[rllib] " + json.dumps(summary, ensure_ascii=False), flush=True)
            if wandb_run is not None:
                wandb_run.log(summary, step=summary["steps"])
            iteration = int(result.get("training_iteration", 0))
            if args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
                saved = algo.save(str(run_dir / "checkpoints"))
                print(f"[rllib] checkpoint={saved.checkpoint.path}", flush=True)
            step_due = (
                next_step_checkpoint is not None
                and summary["steps"] >= next_step_checkpoint
            )
            wandb_due = (
                next_wandb_checkpoint is not None
                and summary["steps"] >= next_wandb_checkpoint
            )
            if step_due or wandb_due:
                saved = algo.save(str(run_dir / "step_checkpoints"))
                print(f"[rllib] checkpoint={saved.checkpoint.path}", flush=True)
                if step_due:
                    while summary["steps"] >= next_step_checkpoint:
                        next_step_checkpoint += args.save_every
                if wandb_due:
                    while summary["steps"] >= next_wandb_checkpoint:
                        next_wandb_checkpoint += args.wandb_save_freq
            if summary["steps"] >= args.steps:
                break

        saved = algo.save(str(run_dir / "final"))
        print(f"[rllib] final={saved.checkpoint.path}", flush=True)
    finally:
        if algo is not None:
            algo.stop()
        if wandb_run is not None:
            wandb_run.finish()
        ray.shutdown()


if __name__ == "__main__":
    main()
