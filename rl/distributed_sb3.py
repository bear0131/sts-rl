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
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from rl.accumulating_ppo import AccumulatingMaskablePPO
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
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--metrics-window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--barrier-timeout", type=float, default=600.0)
    parser.add_argument("--stable-file-seconds", type=float, default=0.25)
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


def wait_stable(path: Path, timeout: float, stable_for: float) -> None:
    start = time.perf_counter()
    last_size = -1
    last_change = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if path.exists():
            size = path.stat().st_size
            if size != last_size:
                last_size = size
                last_change = time.perf_counter()
            elif size > 0 and time.perf_counter() - last_change >= stable_for:
                return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for stable file: {path}")


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
        venv = SubprocVecEnv(env_fns, start_method=args.start_method)
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


def save_gradients(model: AccumulatingMaskablePPO, path: Path) -> int:
    grads = {
        name: parameter.grad.detach().cpu()
        for name, parameter in model.policy.named_parameters()
        if parameter.grad is not None
    }
    th.save(grads, path)
    return path.stat().st_size


def average_peer_gradients(
    model: AccumulatingMaskablePPO,
    paths: list[Path],
    *,
    world_size: int,
) -> float:
    started = time.perf_counter()
    totals: dict[str, th.Tensor] = {}
    for path in paths:
        peer = th.load(path, map_location="cpu")
        for name, grad in peer.items():
            if name not in totals:
                totals[name] = grad
            else:
                totals[name].add_(grad)

    with th.no_grad():
        for name, parameter in model.policy.named_parameters():
            if parameter.grad is not None:
                total = parameter.grad.detach().cpu()
                if name in totals:
                    total = total + totals[name]
                averaged = total / world_size
                parameter.grad.copy_(averaged.to(parameter.grad.device))
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
        if key.startswith(("game/", "rollout/", "speed/")):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalars[key] = float(value)
    return scalars


def ppo_microbatch_backward(
    model: AccumulatingMaskablePPO,
    rollout_data,
    clip_range: float,
    clip_range_vf: float | None,
) -> float:
    actions = rollout_data.actions
    if isinstance(model.action_space, spaces.Discrete):
        actions = rollout_data.actions.long().flatten()

    values, log_prob, entropy = model.policy.evaluate_actions(
        rollout_data.observations,
        actions,
        action_masks=rollout_data.action_masks,
    )
    values = values.flatten()
    advantages = rollout_data.advantages
    if model.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = th.exp(log_prob - rollout_data.old_log_prob)
    policy_loss = -th.min(
        advantages * ratio,
        advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
    ).mean()
    if clip_range_vf is None:
        values_pred = values
    else:
        values_pred = rollout_data.old_values + th.clamp(
            values - rollout_data.old_values,
            -clip_range_vf,
            clip_range_vf,
        )
    value_loss = F.mse_loss(rollout_data.returns, values_pred)
    entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
    loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
    loss.backward()
    return float(loss.detach().cpu())


def main() -> None:
    args = build_parser().parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank must be in [0, world_size)")
    if args.world_size == 1 and args.peer:
        raise ValueError("--peer is meaningless for world_size=1")

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
    env = make_env(probe_model, args)
    del probe_model
    model = AccumulatingMaskablePPO.load(
        args.checkpoint,
        env=env,
        device=args.device,
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
        },
    )
    model.set_random_seed(args.seed + args.rank * 1_000_003)

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
            sync_seconds = 0.0
            sync_mbps = []
            avg_seconds = 0.0
            step_seconds = 0.0
            backward_seconds = 0.0
            grad_bytes = 0
            optimizer_steps = 0

            for epoch in range(args.n_epochs):
                accumulated = 0
                model.policy.optimizer.zero_grad()
                for rollout_data in model.rollout_buffer.get(args.batch_size):
                    backward_started = time.perf_counter()
                    losses.append(
                        ppo_microbatch_backward(
                            model,
                            rollout_data,
                            clip_range,
                            clip_range_vf,
                        )
                    )
                    if model.device.type == "cuda":
                        th.cuda.synchronize(model.device)
                    backward_seconds += time.perf_counter() - backward_started
                    accumulated += 1
                    if accumulated != args.gradient_accumulation_steps:
                        continue

                    for parameter in model.policy.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(1.0 / accumulated)

                    sync_tag = f"u{update:06d}_e{epoch:03d}_s{optimizer_steps:06d}"
                    grad_path = local_root / f"grad_rank{args.rank}_{sync_tag}.pt"
                    grad_bytes = save_gradients(model, grad_path)

                    remote_name = f"rank{args.rank}/grad_rank{args.rank}_{sync_tag}.pt"
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
                        wait_stable(
                            path,
                            timeout=args.barrier_timeout,
                            stable_for=args.stable_file_seconds,
                        )
                    sync_seconds += time.perf_counter() - wait_started

                    avg_seconds += average_peer_gradients(
                        model,
                        peer_paths,
                        world_size=args.world_size,
                    )

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

                    accumulated = 0
                    optimizer_steps += 1

            record = {
                "rank": args.rank,
                "world_size": args.world_size,
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
                "avg_grad_sec": avg_seconds,
                "optimizer_step_sec": step_seconds,
                "grad_bytes": grad_bytes,
                "total_sec_rank": (
                    rollout_sec
                    + backward_seconds
                    + sync_seconds
                    + avg_seconds
                    + step_seconds
                ),
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

            global_timesteps = int(record["global_timesteps"])
            # Keep saved checkpoints/resumes on the global env-step axis rather
            # than SB3's local rank counter.
            model.num_timesteps = global_timesteps
            if wandb_run is not None:
                import wandb

                wandb_payload = {
                        "time/total_timesteps": global_timesteps,
                        "dist/rank0_total_sec": record["total_sec_rank"],
                        "dist/rollout_sec": record["rollout_sec"],
                        "dist/backward_sec": record["backward_sec"],
                        "dist/sync_sec": record["sync_sec"],
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
                wandb_payload.update(logger_scalars(model))
                # Match rl.trainer's W&B indexing: do not pass step= here and
                # do not bind all metrics to time/total_timesteps. The original
                # SB3 path lets W&B/TensorBoard advance the run step naturally.
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
