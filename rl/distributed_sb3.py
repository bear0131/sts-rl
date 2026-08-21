"""Synchronous multi-machine SB3 PPO probe/trainer.

Each rank runs the same script on one machine:

  1. collect a local rollout with the current policy,
  2. compute PPO gradients on local minibatches,
  3. at every gradient-accumulation boundary, exchange gradients with all peers,
  4. average gradients and take the same optimizer step on every rank.

This intentionally avoids Ray/RLlib.  It uses SSH/SCP as a simple transport and
therefore works with the Autodl public port mappings already used for training.
The all-reduce is O(world_size^2), which is fine for the 2-4 machine scale this
experiment targets.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rl.accumulating_ppo import (
    AccumulatingMaskablePPO,
    ppo_loss_components,
)
from rl.cached_mask_vec_env import CachedMaskSubprocVecEnv
from rl.ring_allreduce import SshRingAllReduce
from rl.tasks.wholerun import WholeRunTask

# Register custom policies before loading checkpoints.
from rl import pointer_policy as _pointer_policy  # noqa: F401
from rl import split_policy as _split_policy  # noqa: F401
from rl import model as model_utils


class NoopCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


@dataclass(frozen=True)
class Peer:
    rank: int
    host: str
    port: int
    user: str = "root"


def parse_peer(text: str) -> Peer:
    """Parse rank=user@host:port or rank=host:port."""
    rank_text, target = text.split("=", 1)
    rank = int(rank_text)
    if "@" in target:
        user, host_port = target.split("@", 1)
    else:
        user, host_port = "root", target
    host, port_text = host_port.rsplit(":", 1)
    return Peer(rank=rank, user=user, host=host, port=int(port_text))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--peer",
        action="append",
        default=[],
        help="Peer endpoint, repeatable: rank=root@host:port. Include every "
        "other rank; self is ignored if present.",
    )
    parser.add_argument("--ssh-key", default="~/.ssh/sts_dist_test_ed25519")
    parser.add_argument("--ssh-control-dir", default="/tmp/sts_dist_ssh_mux")
    parser.add_argument("--remote-root", default="/tmp/sts_dist_sb3")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Absolute global env-step target. When set, overrides --updates "
        "using checkpoint num_timesteps plus world_size*n_envs*n_steps/update.",
    )
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--hierarchical-combat-hp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use fight-local terminal-HP returns for combat and macro combat "
             "transitions for overworld. Omission preserves checkpoint mode.",
    )
    parser.add_argument("--combat-hp-gamma", type=float, default=None)
    parser.add_argument("--combat-hp-gae-lambda", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--metrics-window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help="PyTorch float32 matmul precision. 'high' enables TF32 tensor cores "
             "on supported NVIDIA GPUs; use 'highest' for strict FP32.",
    )
    parser.add_argument(
        "--amp-mode",
        choices=("off", "bf16"),
        default="off",
        help="Optional CUDA autocast for rollout inference and PPO backward. "
             "BF16 keeps parameters and synchronized gradients in FP32.",
    )
    parser.add_argument("--vec", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--start-method", default="fork")
    parser.add_argument("--save-every-updates", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-prefix", default="maskable_ppo")
    parser.add_argument("--max-checkpoints", type=int, default=20)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--wandb-step-base",
        type=int,
        default=None,
        help="Legacy chart-step value at --wandb-step-base-timesteps. Use "
        "with --wandb-step-per-update to continue an existing chart axis.",
    )
    parser.add_argument(
        "--wandb-step-base-timesteps",
        type=int,
        default=None,
        help="Global env-step anchor corresponding to --wandb-step-base.",
    )
    parser.add_argument(
        "--wandb-step-per-update",
        type=int,
        default=1,
        help="Amount to advance W&B Step for each global rollout update.",
    )
    parser.add_argument("--barrier-timeout", type=float, default=600.0)
    # Accepted so existing launch scripts keep working; atomic rename makes the
    # old post-copy stability delay unnecessary.
    parser.add_argument(
        "--stable-file-seconds",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gradient-transport",
        choices=("scp", "ssh-ring"),
        default="scp",
        help="Gradient exchange backend. ssh-ring keeps one persistent SSH "
             "tunnel per rank and avoids gradient files.",
    )
    parser.add_argument(
        "--ring-port",
        type=int,
        default=29750,
        help="Loopback listener port forwarded by the ssh-ring transport.",
    )
    return parser


def run_cmd(cmd: list[str], *, timeout: float | None = None) -> None:
    subprocess.run(cmd, check=True, timeout=timeout)


def ssh_options(peer: Peer, key: Path, control_dir: Path) -> list[str]:
    return [
        "-i",
        str(key),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=600",
        "-o",
        f"ControlPath={control_dir}/%C",
    ]


def ssh_cmd(
    peer: Peer,
    key: Path,
    control_dir: Path,
    command: str,
    *,
    timeout: float | None = None,
) -> None:
    run_cmd(
        [
            "ssh",
            *ssh_options(peer, key, control_dir),
            "-p",
            str(peer.port),
            f"{peer.user}@{peer.host}",
            command,
        ],
        timeout=timeout,
    )


def scp_to(
    peer: Peer,
    key: Path,
    control_dir: Path,
    src: Path,
    dst: str,
    *,
    timeout: float | None = None,
) -> None:
    run_cmd(
        [
            "scp",
            *ssh_options(peer, key, control_dir),
            "-P",
            str(peer.port),
            str(src),
            f"{peer.user}@{peer.host}:{dst}",
        ],
        timeout=timeout,
    )


def wait_for_sized_file(path: Path, timeout: float, expected_size: int) -> None:
    """Wait for an atomically published peer file of the expected size."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if path.exists():
            size = path.stat().st_size
            if size == expected_size:
                return
            raise RuntimeError(
                f"peer gradient file has size {size}, expected {expected_size}: {path}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for peer file: {path}")


def observation_flags(model: AccumulatingMaskablePPO) -> dict[str, bool]:
    return {
        "global_map_view": model_utils.uses_global_map_observation(model),
        "card_language": model_utils.uses_card_language(model),
        "combat_language": model_utils.uses_combat_language(model),
        "deck_size_observation": model_utils.uses_deck_size_observation(model),
        "neow_observation": model_utils.uses_neow_observation(model),
        "event_option_observation": (
            model_utils.uses_event_option_observation(model)
        ),
    }


def make_env(model: AccumulatingMaskablePPO, args: argparse.Namespace):
    flags = observation_flags(model)

    class EnvArgs:
        pass

    env_args = EnvArgs()
    env_args.ascension = args.ascension
    env_args.global_map_view = flags["global_map_view"]
    env_args.card_language = flags["card_language"]
    env_args.combat_language = flags["combat_language"]
    env_args.deck_size_observation = flags["deck_size_observation"]
    env_args.neow_observation = flags["neow_observation"]
    env_args.event_option_observation = flags["event_option_observation"]
    env_args.map_dag = True

    env_fns = [WholeRunTask().make_env_fn(env_args) for _ in range(args.n_envs)]
    if args.vec == "subproc":
        venv = CachedMaskSubprocVecEnv(
            env_fns, start_method=args.start_method
        )
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)
    venv.seed(args.seed + args.rank * 1_000_003)
    return venv


def sync_file_to_peers(
    *,
    local_file: Path,
    remote_name: str,
    peers: list[Peer],
    key: Path,
    control_dir: Path,
    remote_root: str,
    timeout: float,
) -> tuple[float, float]:
    started = time.perf_counter()
    for peer in peers:
        remote_tmp = f"{remote_root}/{remote_name}.tmp"
        remote_final = f"{remote_root}/{remote_name}"
        scp_to(peer, key, control_dir, local_file, remote_tmp, timeout=timeout)
        ssh_cmd(
            peer,
            key,
            control_dir,
            f"mv {shlex.quote(remote_tmp)} {shlex.quote(remote_final)}",
        )
    elapsed = time.perf_counter() - started
    mbps = local_file.stat().st_size * len(peers) / 1024 / 1024 / max(elapsed, 1e-9)
    return elapsed, mbps


def all_rank_file_barrier(
    *,
    tag: str,
    rank: int,
    peers: list[Peer],
    key: Path,
    control_dir: Path,
    local_root: Path,
    remote_root: str,
    run_id: str,
    timeout: float,
) -> None:
    marker = local_root / f"barrier_rank{rank}_{tag}.ready"
    marker.write_bytes(b"ready")
    remote_name = f"rank{rank}/{marker.name}"
    sync_file_to_peers(
        local_file=marker,
        remote_name=remote_name,
        peers=peers,
        key=key,
        control_dir=control_dir,
        remote_root=f"{remote_root}/{run_id}",
        timeout=timeout,
    )
    peer_markers = [
        Path(remote_root)
        / run_id
        / f"rank{peer.rank}"
        / f"barrier_rank{peer.rank}_{tag}.ready"
        for peer in peers
    ]
    for path in peer_markers:
        wait_for_sized_file(path, timeout=timeout, expected_size=5)
    for path in [marker, *peer_markers]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def gradient_parameters(model: AccumulatingMaskablePPO) -> list[th.nn.Parameter]:
    return [
        parameter
        for parameter in model.policy.parameters()
        if parameter.requires_grad
    ]


def pack_gradients(model: AccumulatingMaskablePPO) -> th.Tensor:
    """Pack every trainable gradient into one CPU FP32 vector."""
    parameters = gradient_parameters(model)
    parts = [
        (
            parameter.grad.detach().reshape(-1)
            if parameter.grad is not None
            else th.zeros_like(parameter).reshape(-1)
        )
        for parameter in parameters
    ]
    flat = th.cat(parts).detach().cpu()
    if flat.dtype != th.float32:
        raise TypeError(f"distributed gradients must be float32, got {flat.dtype}")
    return flat


def save_gradients(flat: th.Tensor, path: Path) -> int:
    """Publish a fixed-layout raw gradient vector for SCP transport."""
    flat.numpy().tofile(path)
    return path.stat().st_size


def install_flat_gradients(
    model: AccumulatingMaskablePPO, averaged: th.Tensor
) -> None:
    averaged_device = averaged.to(model.device)
    with th.no_grad():
        offset = 0
        for parameter in gradient_parameters(model):
            end = offset + parameter.numel()
            parameter.grad = averaged_device[offset:end].view_as(parameter)
            offset = end
        if offset != averaged_device.numel():
            raise RuntimeError(
                f"consumed {offset} averaged gradients, got "
                f"{averaged_device.numel()}"
            )


def average_peer_gradients(
    model: AccumulatingMaskablePPO,
    local_flat: th.Tensor,
    peer_gradients: list[tuple[int, Path]],
    *,
    rank: int,
    world_size: int,
) -> float:
    started = time.perf_counter()
    by_rank = {rank: local_flat}
    for peer_rank, path in peer_gradients:
        peer_array = np.fromfile(path, dtype=np.float32)
        if peer_array.size != local_flat.numel():
            raise RuntimeError(
                f"peer gradient has {peer_array.size} values, expected "
                f"{local_flat.numel()}: {path}"
            )
        by_rank[peer_rank] = th.from_numpy(peer_array)
    if set(by_rank) != set(range(world_size)):
        raise RuntimeError(
            f"gradient ranks are {sorted(by_rank)}, expected "
            f"{list(range(world_size))}"
        )
    total = th.zeros_like(local_flat)
    for source_rank in range(world_size):
        total.add_(by_rank[source_rank])
    total.mul_(1.0 / world_size)
    install_flat_gradients(model, total)
    return time.perf_counter() - started


def checkpoint_step(path: Path, prefix: str) -> int | None:
    stem = path.stem
    start = f"{prefix}_"
    end = "_steps"
    if not stem.startswith(start) or not stem.endswith(end):
        return None
    text = stem[len(start):-len(end)]
    return int(text) if text.isdigit() else None


def prune_checkpoints(directory: Path, prefix: str, limit: int) -> None:
    if limit < 1 or not directory.exists():
        return
    checkpoints = []
    for path in directory.glob(f"{prefix}_*_steps.zip"):
        step = checkpoint_step(path, prefix)
        if step is not None:
            checkpoints.append((step, path))
    checkpoints.sort()
    for _step, path in checkpoints[:max(0, len(checkpoints) - limit)]:
        path.unlink()


def logger_scalars(model: AccumulatingMaskablePPO) -> dict[str, float]:
    scalars = {}
    for key, value in model.logger.name_to_value.items():
        if key.startswith(("game/", "rollout/", "speed/", "hierarchical/")):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalars[key] = float(value)
    return scalars


def wandb_step_for_timesteps(
    *,
    global_timesteps: int,
    global_rollout_size: int,
    base_step: int | None,
    base_timesteps: int | None,
    step_per_update: int,
) -> int | None:
    if base_step is None and base_timesteps is None:
        return None
    if base_step is None or base_timesteps is None:
        raise ValueError(
            "--wandb-step-base and --wandb-step-base-timesteps must be set together"
        )
    if step_per_update < 1:
        raise ValueError("--wandb-step-per-update must be positive")

    elapsed = global_timesteps - base_timesteps
    updates, remainder = divmod(elapsed, global_rollout_size)
    if elapsed < 0 or remainder:
        raise ValueError(
            f"global timesteps {global_timesteps} do not align with W&B Step "
            f"anchor {base_timesteps} and rollout size {global_rollout_size}"
        )
    return base_step + updates * step_per_update


def ppo_microbatch_backward(
    model: AccumulatingMaskablePPO,
    rollout_data,
    clip_range: float,
    clip_range_vf: float | None,
) -> dict[str, float | None]:
    actions = rollout_data.actions
    if isinstance(model.action_space, spaces.Discrete):
        actions = rollout_data.actions.long().flatten()

    with model.autocast_context():
        values, log_prob, entropy = model.policy.evaluate_actions(
            rollout_data.observations,
            actions,
            action_masks=rollout_data.action_masks,
        )
    values = values.flatten()
    components = ppo_loss_components(
        hierarchical=model.hierarchical_combat_hp,
        normalize_advantage=model.normalize_advantage,
        observations=rollout_data.observations,
        advantages=rollout_data.advantages,
        returns=rollout_data.returns,
        old_values=rollout_data.old_values,
        old_log_prob=rollout_data.old_log_prob,
        values=values,
        log_prob=log_prob,
        entropy=entropy,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
    )
    loss = (
        components.policy_loss
        + model.ent_coef * components.entropy_loss
        + model.vf_coef * components.value_loss
    )
    loss.backward()
    return {
        "loss": float(loss.detach().cpu()),
        "combat_value_loss": (
            float(components.combat_value_loss.detach().cpu())
            if components.combat_value_loss is not None else None
        ),
        "overworld_value_loss": (
            float(components.overworld_value_loss.detach().cpu())
            if components.overworld_value_loss is not None else None
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    th.set_float32_matmul_precision(args.float32_matmul_precision)
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank must be in [0, world_size)")
    if args.world_size == 1 and args.peer:
        raise ValueError("--peer is meaningless for world_size=1")
    if args.gradient_transport == "ssh-ring" and args.world_size < 2:
        raise ValueError("--gradient-transport ssh-ring requires world_size >= 2")
    if not 1 <= args.ring_port <= 65535:
        raise ValueError("--ring-port must be in [1, 65535]")

    peers = sorted((parse_peer(p) for p in args.peer), key=lambda p: p.rank)
    peer_ranks = {p.rank for p in peers}
    expected = set(range(args.world_size)) - {args.rank}
    if peer_ranks != expected:
        raise ValueError(f"peers must be ranks {sorted(expected)}, got {sorted(peer_ranks)}")

    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size:
        raise ValueError("--batch-size must divide n_envs*n_steps")
    if (rollout_size // args.batch_size) % args.gradient_accumulation_steps:
        raise ValueError("micro-batches must divide gradient_accumulation_steps")

    run_id = args.run_id or f"{int(time.time())}"
    local_root = Path(args.remote_root) / run_id / f"rank{args.rank}"
    local_root.mkdir(parents=True, exist_ok=True)
    for path in local_root.glob("*"):
        if path.is_file():
            path.unlink()

    key = Path(args.ssh_key).expanduser()
    control_dir = Path(args.ssh_control_dir) / run_id / f"rank{args.rank}"
    control_dir.mkdir(parents=True, exist_ok=True)
    for peer in peers:
        ssh_cmd(
            peer,
            key,
            control_dir,
            f"mkdir -p {shlex.quote(str(Path(args.remote_root) / run_id / f'rank{args.rank}'))}",
        )

    # Load once without env to infer obs flags, then load again with env and
    # custom_objects so SB3 rebuilds the rollout buffer at the requested shape.
    probe_model = AccumulatingMaskablePPO.load(args.checkpoint, device=args.device)
    source_hierarchical = bool(
        getattr(probe_model, "hierarchical_combat_hp", False)
    )
    source_combat_gamma = float(
        getattr(probe_model, "combat_hp_gamma", 1.0)
    )
    source_combat_gae_lambda = float(
        getattr(probe_model, "combat_hp_gae_lambda", 0.95)
    )
    if args.hierarchical_combat_hp is None:
        args.hierarchical_combat_hp = source_hierarchical
    if args.combat_hp_gamma is None:
        args.combat_hp_gamma = source_combat_gamma
    if args.combat_hp_gae_lambda is None:
        args.combat_hp_gae_lambda = source_combat_gae_lambda
    if not 0.0 <= args.combat_hp_gamma <= 1.0:
        raise ValueError("--combat-hp-gamma must be in [0, 1]")
    if not 0.0 <= args.combat_hp_gae_lambda <= 1.0:
        raise ValueError("--combat-hp-gae-lambda must be in [0, 1]")
    env = make_env(probe_model, args)
    del probe_model
    model = AccumulatingMaskablePPO.load(
        args.checkpoint,
        env=env,
        device=args.device,
        amp_mode=args.amp_mode,
        custom_objects={
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "hierarchical_combat_hp": args.hierarchical_combat_hp,
            "combat_hp_gamma": args.combat_hp_gamma,
            "combat_hp_gae_lambda": args.combat_hp_gae_lambda,
        },
    )
    reset_count = model.configure_hierarchical_combat_hp(
        enabled=args.hierarchical_combat_hp,
        combat_gamma=args.combat_hp_gamma,
        combat_gae_lambda=args.combat_hp_gae_lambda,
        reset_combat_value=(
            bool(args.hierarchical_combat_hp) != source_hierarchical
        ),
        reset_seed=args.seed,
    )
    if reset_count:
        print(
            f"[rank {args.rank}] combat value objective changed; reset "
            f"{reset_count} value parameter tensors",
            flush=True,
        )
    model.set_random_seed(args.seed + args.rank * 1_000_003)

    ring_transport = None
    if args.gradient_transport == "ssh-ring":
        next_rank = (args.rank + 1) % args.world_size
        next_peer = next(peer for peer in peers if peer.rank == next_rank)

        def ring_ready_barrier() -> None:
            all_rank_file_barrier(
                tag="ssh_ring_listening",
                rank=args.rank,
                peers=peers,
                key=key,
                control_dir=control_dir,
                local_root=local_root,
                remote_root=args.remote_root,
                run_id=run_id,
                timeout=args.barrier_timeout,
            )

        ring_transport = SshRingAllReduce(
            rank=args.rank,
            world_size=args.world_size,
            next_host=next_peer.host,
            next_port=next_peer.port,
            next_user=next_peer.user,
            ssh_key=key,
            listen_port=args.ring_port,
            timeout=args.barrier_timeout,
            ready_barrier=ring_ready_barrier,
        )
        print(
            f"[rank {args.rank}] persistent SSH gradient ring connected to "
            f"rank {next_rank}",
            flush=True,
        )

    start_timesteps = int(model.num_timesteps)
    global_rollout_size = rollout_size * args.world_size
    if args.total_steps is not None:
        remaining = args.total_steps - start_timesteps
        if remaining <= 0:
            raise ValueError(
                f"--total-steps {args.total_steps} <= checkpoint step "
                f"{start_timesteps}"
            )
        args.updates = int(np.ceil(remaining / global_rollout_size))

    wandb_run = None
    if args.wandb and args.rank == 0:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_id,
            name=args.wandb_name,
            resume="allow" if args.wandb_id else None,
        )
        wandb.config.update(
            {
                **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "start_timesteps": start_timesteps,
                "global_rollout_size": global_rollout_size,
            },
            allow_val_change=True,
        )
        wandb.define_metric("time/total_timesteps")
        if args.wandb_step_base is not None:
            wandb.define_metric("legacy_step")
            for metric in (
                "game/*",
                "rollout/*",
                "speed/*",
                "hierarchical/*",
                "train/*",
                "dist/*",
                "global_step",
                "time/total_timesteps",
            ):
                wandb.define_metric(metric, step_metric="legacy_step")

    callback = CallbackList(
        [WholeRunTask().metrics_callback(args)] if args.rank == 0 else [NoopCallback()]
    )
    total_timesteps, callback = model._setup_learn(
        total_timesteps=args.updates * rollout_size,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name=f"dist_sb3_rank{args.rank}",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())

    records = []
    try:
        for update in range(args.updates):
            if model.device.type == "cuda":
                th.cuda.synchronize(model.device)
            started = time.perf_counter()
            ok = model.collect_rollouts(
                model.env,
                callback,
                model.rollout_buffer,
                n_rollout_steps=args.n_steps,
            )
            if model.device.type == "cuda":
                th.cuda.synchronize(model.device)
            rollout_sec = time.perf_counter() - started
            if not ok:
                raise RuntimeError("collect_rollouts returned False")

            model.policy.set_training_mode(True)
            model._update_learning_rate(model.policy.optimizer)
            clip_range = model.clip_range(model._current_progress_remaining)
            clip_range_vf = (
                model.clip_range_vf(model._current_progress_remaining)
                if model.clip_range_vf is not None
                else None
            )

            losses = []
            combat_value_losses = []
            overworld_value_losses = []
            sync_seconds = 0.0
            sync_mbps = []
            avg_seconds = 0.0
            gradient_pack_seconds = 0.0
            step_seconds = 0.0
            backward_seconds = 0.0
            grad_bytes = 0
            optimizer_steps = 0

            for epoch in range(args.n_epochs):
                accumulated = 0
                model.policy.optimizer.zero_grad()
                for rollout_data in model.rollout_buffer.get(args.batch_size):
                    backward_started = time.perf_counter()
                    batch_result = ppo_microbatch_backward(
                        model,
                        rollout_data,
                        clip_range,
                        clip_range_vf,
                    )
                    losses.append(batch_result["loss"])
                    if batch_result["combat_value_loss"] is not None:
                        combat_value_losses.append(
                            batch_result["combat_value_loss"]
                        )
                    if batch_result["overworld_value_loss"] is not None:
                        overworld_value_losses.append(
                            batch_result["overworld_value_loss"]
                        )
                    if model.device.type == "cuda":
                        th.cuda.synchronize(model.device)
                    backward_seconds += time.perf_counter() - backward_started
                    accumulated += 1
                    if accumulated != args.gradient_accumulation_steps:
                        continue

                    gradients = [
                        parameter.grad
                        for parameter in model.policy.parameters()
                        if parameter.grad is not None
                    ]
                    th._foreach_mul_(gradients, 1.0 / accumulated)

                    pack_started = time.perf_counter()
                    local_flat = pack_gradients(model)
                    grad_bytes = local_flat.numel() * local_flat.element_size()
                    gradient_pack_seconds += time.perf_counter() - pack_started
                    paths_to_remove = []

                    if ring_transport is not None:
                        ring_started = time.perf_counter()
                        local_flat, bytes_sent = ring_transport.average(local_flat)
                        elapsed = time.perf_counter() - ring_started
                        sync_seconds += elapsed
                        sync_mbps.append(
                            bytes_sent / 1024 / 1024 / max(elapsed, 1e-9)
                        )
                        avg_started = time.perf_counter()
                        install_flat_gradients(model, local_flat)
                        avg_seconds += time.perf_counter() - avg_started
                    else:
                        sync_tag = (
                            f"u{update:06d}_e{epoch:03d}_s{optimizer_steps:06d}"
                        )
                        grad_path = local_root / (
                            f"grad_rank{args.rank}_{sync_tag}.pt"
                        )
                        save_gradients(local_flat, grad_path)
                        remote_name = (
                            f"rank{args.rank}/grad_rank{args.rank}_{sync_tag}.pt"
                        )
                        elapsed, mbps = sync_file_to_peers(
                            local_file=grad_path,
                            remote_name=remote_name,
                            peers=peers,
                            key=key,
                            control_dir=control_dir,
                            remote_root=f"{args.remote_root}/{run_id}",
                            timeout=args.barrier_timeout,
                        )
                        sync_seconds += elapsed
                        sync_mbps.append(mbps)

                        peer_paths = [
                            Path(args.remote_root)
                            / run_id
                            / f"rank{peer.rank}"
                            / f"grad_rank{peer.rank}_{sync_tag}.pt"
                            for peer in peers
                        ]
                        wait_started = time.perf_counter()
                        for path in peer_paths:
                            wait_for_sized_file(
                                path,
                                timeout=args.barrier_timeout,
                                expected_size=grad_bytes,
                            )
                        sync_seconds += time.perf_counter() - wait_started

                        avg_seconds += average_peer_gradients(
                            model,
                            local_flat,
                            list(zip(
                                (peer.rank for peer in peers),
                                peer_paths,
                                strict=True,
                            )),
                            rank=args.rank,
                            world_size=args.world_size,
                        )
                        paths_to_remove = [grad_path, *peer_paths]

                    if model.device.type == "cuda":
                        th.cuda.synchronize(model.device)
                    step_started = time.perf_counter()
                    th.nn.utils.clip_grad_norm_(
                        model.policy.parameters(),
                        model.max_grad_norm,
                    )
                    model.policy.optimizer.step()
                    model.policy.optimizer.zero_grad()
                    if model.device.type == "cuda":
                        th.cuda.synchronize(model.device)
                    step_seconds += time.perf_counter() - step_started
                    for path in paths_to_remove:
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass

                    accumulated = 0
                    optimizer_steps += 1

            record = {
                "rank": args.rank,
                "world_size": args.world_size,
                "gradient_transport": args.gradient_transport,
                "update": update,
                "start_timesteps": start_timesteps,
                "global_timesteps": start_timesteps + (update + 1) * global_rollout_size,
                "transitions_local": rollout_size,
                "transitions_global": rollout_size * args.world_size,
                "rollout_sec": rollout_sec,
                "rollout_steps_per_sec_local": rollout_size / rollout_sec,
                "loss_mean": float(np.mean(losses)) if losses else 0.0,
                "optimizer_steps": optimizer_steps,
                "sync_sec": sync_seconds,
                "sync_mbps_mean": float(np.mean(sync_mbps)) if sync_mbps else 0.0,
                "backward_sec": backward_seconds,
                "gradient_pack_sec": gradient_pack_seconds,
                "avg_grad_sec": avg_seconds,
                "optimizer_step_sec": step_seconds,
                "grad_bytes": grad_bytes,
                "total_sec_rank": (
                    rollout_sec
                    + backward_seconds
                    + gradient_pack_seconds
                    + sync_seconds
                    + avg_seconds
                    + step_seconds
                ),
            }
            if args.hierarchical_combat_hp:
                record["combat_value_loss_mean"] = (
                    float(np.mean(combat_value_losses))
                    if combat_value_losses else 0.0
                )
                record["overworld_value_loss_mean"] = (
                    float(np.mean(overworld_value_losses))
                    if overworld_value_losses else 0.0
                )
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

            global_timesteps = int(record["global_timesteps"])
            # Keep saved checkpoints/resumes on the global env-step axis rather
            # than SB3's local rank counter.
            model.num_timesteps = global_timesteps
            if wandb_run is not None:
                import wandb

                wandb_payload = {
                        "global_step": global_timesteps,
                        "time/total_timesteps": global_timesteps,
                        "dist/rank0_total_sec": record["total_sec_rank"],
                        "dist/rollout_sec": record["rollout_sec"],
                        "dist/backward_sec": record["backward_sec"],
                        "dist/sync_sec": record["sync_sec"],
                        "dist/gradient_pack_sec": record["gradient_pack_sec"],
                        "dist/avg_grad_sec": record["avg_grad_sec"],
                        "dist/optimizer_step_sec": record["optimizer_step_sec"],
                        "dist/sync_mbps": record["sync_mbps_mean"],
                        "dist/global_steps_per_second": (
                            record["transitions_global"]
                            / max(record["total_sec_rank"], 1e-9)
                        ),
                        "train/loss": record["loss_mean"],
                        "train/optimizer_steps": record["optimizer_steps"],
                        "train/gradient_accumulation_steps": (
                            args.gradient_accumulation_steps * args.world_size
                        ),
                        "train/effective_batch_size": (
                            args.batch_size
                            * args.gradient_accumulation_steps
                            * args.world_size
                        ),
                        "train/effective_batch_size_global": (
                            args.batch_size
                            * args.gradient_accumulation_steps
                            * args.world_size
                        ),
                    }
                if args.hierarchical_combat_hp:
                    wandb_payload.update({
                        "train/combat_value_loss": record[
                            "combat_value_loss_mean"
                        ],
                        "train/overworld_value_loss": record[
                            "overworld_value_loss_mean"
                        ],
                    })
                wandb_payload.update(logger_scalars(model))
                legacy_step = wandb_step_for_timesteps(
                    global_timesteps=global_timesteps,
                    global_rollout_size=global_rollout_size,
                    base_step=args.wandb_step_base,
                    base_timesteps=args.wandb_step_base_timesteps,
                    step_per_update=args.wandb_step_per_update,
                )
                if legacy_step is not None:
                    # W&B's internal _step is append-only and cannot be moved
                    # backwards when resuming an existing run. A custom step
                    # metric preserves the legacy chart axis without changing
                    # the real global environment-step counters.
                    wandb_payload["legacy_step"] = legacy_step
                wandb.log(wandb_payload)

            if args.output is not None and (
                args.save_every_updates
                and (update + 1) % args.save_every_updates == 0
            ):
                model.save(str(args.output.with_suffix("")))
            if (
                args.rank == 0
                and args.checkpoint_dir is not None
                and args.save_every_updates
                and (update + 1) % args.save_every_updates == 0
            ):
                args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                prune_checkpoints(
                    args.checkpoint_dir,
                    args.checkpoint_prefix,
                    args.max_checkpoints - 1,
                )
                path = (
                    args.checkpoint_dir
                    / f"{args.checkpoint_prefix}_{global_timesteps}_steps"
                )
                model.save(str(path))
                prune_checkpoints(
                    args.checkpoint_dir,
                    args.checkpoint_prefix,
                    args.max_checkpoints,
                )

        if args.output is not None:
            model.save(str(args.output.with_suffix("")))
        if args.rank == 0 and args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            final_step = start_timesteps + len(records) * global_rollout_size
            model.num_timesteps = final_step
            path = (
                args.checkpoint_dir
                / f"{args.checkpoint_prefix}_{final_step}_steps"
            )
            model.save(str(path))
            prune_checkpoints(
                args.checkpoint_dir,
                args.checkpoint_prefix,
                args.max_checkpoints,
            )
    finally:
        callback.on_training_end()
        if ring_transport is not None:
            ring_transport.close()
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    summary = {
        "rank": args.rank,
        "world_size": args.world_size,
        "updates": args.updates,
        "mean_total_sec_rank": float(np.mean([r["total_sec_rank"] for r in records])),
        "mean_rollout_sec": float(np.mean([r["rollout_sec"] for r in records])),
        "mean_backward_sec": float(np.mean([r["backward_sec"] for r in records])),
        "mean_sync_sec": float(np.mean([r["sync_sec"] for r in records])),
        "mean_optimizer_step_sec": float(np.mean([r["optimizer_step_sec"] for r in records])),
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
