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

from rl.algorithms.updates import ppo_forward, ppo_chunked_backward, optimizer_step, reasoning_microbatch_backward
from rl.algorithms.ppo import configure_advantage_normalization, prepare_advantage_normalization
from rl.algorithms.packed_ppo import configure_packed_updates
from rl.training.ppo_overlap import distributed_packed_epoch
from rl.algorithms.upper_ppo import enabled as joint_upper, combat_update, update_upper
from rl.algorithms.update_schedule import parse_update_schedule
from rl.search.native import NATIVE_TIMING_KEYS, WORKER_TIMING_KEYS, apply_strict_training_settings
from rl.search.rollout_inference import ROLLOUT_INFERENCE_METRICS

import argparse
import hashlib
import json
import random
import shlex
import subprocess
import time
from itertools import count
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from env.core import vocab as V
from rl.algorithms.accumulating_ppo import (
    AccumulatingMaskablePPO,
    NOSL_IS_AZ_SEARCH_MODES,
)
from rl.training.environments import make_env
from rl.training.checkpoints import read_bundle, restore_learner, save_bundle, prune_bundles
from rl.training.ring_allreduce import SshRingAllReduce, ParallelRingAllReduce
from rl.training.monitoring import HP_ORDER_PREFIX, hp_order_fraction
from rl.tasks.wholerun import WholeRunTask

# Register custom policies before loading checkpoints.
from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401
from rl.policies import semantic_reasoning_policy as _semantic_reasoning_policy  # noqa: F401


class NoopCallback(BaseCallback):
    _uses_rollout_observations = False

    def _on_step(self) -> bool:
        return True


class RolloutProgressCallback(NoopCallback):
    """Low-frequency progress for unattended resource-monitored collections."""
    _uses_rollout_observations = False

    def __init__(self, rank, steps):
        super().__init__()
        self.rank, self.steps = rank, steps

    def _on_rollout_start(self):
        self.started, self.completed = time.perf_counter(), 0

    def _on_step(self):
        self.completed += 1
        if self.completed % 128 == 0 or self.completed == self.steps:
            record = dict(rank=self.rank, vector_steps=self.completed, target_steps=self.steps,
                          seconds=time.perf_counter()-self.started, at=time.time())
            if getattr(self.model, 'episode_rollout_config', None):
                record.update(target_steps=None,
                    completed_games=int(self.model.env.episode_rollout_finished.sum()),
                    target_games=self.model.env.num_envs)
            independent = getattr(self.model, '_independent_rollout_progress', None)
            if independent is not None:
                record.pop('vector_steps')
                record.update(transition_batches=self.completed,
                    env_steps=independent['simulator_transitions'] + independent['selection_prefix_decisions'],
                    simulator_transitions=independent['simulator_transitions'])
            runtime = getattr(self.model, '_rollout_inference', None)
            if runtime is not None:
                record['inference'] = runtime.statistics()
            print('ROLLOUT_PROGRESS ' + json.dumps(record, sort_keys=True), flush=True)
        return True


class StartupStaggerMetricsCallback(NoopCallback):
    def __init__(self) -> None:
        super().__init__()
        self.completed_floors: list[int] = []
        self.completed_wins: list[int] = []
        self.synthetic_releases = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            if info.get("startup_stagger_hold"):
                self.synthetic_releases += int(info.get("startup_stagger_remaining") == 0)
                continue
            if "episode" not in info or "run_stats" not in info:
                continue
            stats = info["run_stats"]
            self.completed_floors.append(int(stats["floor"]))
            self.completed_wins.append(int(
                stats["act"] > 3 or stats.get("outcome") == "PLAYER_VICTORY"
            ))
        return True


def run_startup_stagger(model, *, span_steps: int, rollout_steps: int) -> dict:
    """Advance staggered environments while discarding every warmup sample."""
    if span_steps == 0:
        return {}
    if span_steps < 0 or span_steps % rollout_steps:
        raise ValueError("startup stagger span must be a multiple of n_steps")
    callback = StartupStaggerMetricsCallback()
    callback.init_callback(model)
    callback.on_training_start({}, {})
    start_timesteps = int(model.num_timesteps)
    started = time.perf_counter()
    try:
        for _ in range(span_steps // rollout_steps):
            if not model.collect_rollouts(
                model.env,
                callback,
                model.rollout_buffer,
                n_rollout_steps=rollout_steps,
            ):
                raise RuntimeError("startup stagger collection returned False")
    finally:
        callback.on_training_end()
    discarded_transitions = int(model.num_timesteps) - start_timesteps
    expected_transitions = int(model.env.num_envs) * span_steps
    if discarded_transitions != expected_transitions:
        raise RuntimeError(
            "startup stagger transition mismatch: "
            f"expected {expected_transitions}, got {discarded_transitions}"
        )
    states = model.env.env_method("startup_stagger_state")
    if any(int(state["remaining_hold_steps"]) for state in states):
        raise RuntimeError("startup stagger ended before every environment was released")

    # Keep the warmed environment/RNG state, but make the training boundary
    # indistinguishable from a fresh collector to replay, Adam and counters.
    model.num_timesteps = start_timesteps
    model.rollout_buffer.reset()
    model._wait_for_reasoning_samples()
    model._reasoning_rollout_samples = []
    model._nosl_is_az_samples = []
    model._nosl_is_az_pending_fights = [[] for _ in range(model.env.num_envs)]
    for name in ("ep_info_buffer", "ep_success_buffer"):
        buffer = getattr(model, name, None)
        if buffer is not None:
            buffer.clear()
    logger = getattr(model, "logger", None)
    for name in ("name_to_value", "name_to_count", "name_to_excluded"):
        values = getattr(logger, name, None)
        if values is not None:
            values.clear()

    delays = [int(state["initial_hold_steps"]) for state in states]
    ages = [int(state["steps_since_reset"]) for state in states]
    floors = [state.get("floor") for state in states]
    active_floors = [int(value) for value in floors if value is not None]
    recent_completed = callback.completed_floors[-100:]
    return {
        "span_steps": int(span_steps),
        "rollout_blocks": int(span_steps // rollout_steps),
        "discarded_transitions_local": discarded_transitions,
        "delay_min": min(delays),
        "delay_mean": float(np.mean(delays)),
        "delay_max": max(delays),
        "age_min": min(ages),
        "age_mean": float(np.mean(ages)),
        "age_max": max(ages),
        "active_floor_min": min(active_floors) if active_floors else None,
        "active_floor_mean": float(np.mean(active_floors)) if active_floors else None,
        "active_floor_max": max(active_floors) if active_floors else None,
        "completed_episode_count": len(callback.completed_floors),
        "completed_floor_mean_last_100": (
            float(np.mean(recent_completed)) if recent_completed else None
        ),
        "completed_win_rate_last_100": (
            float(np.mean(callback.completed_wins[-100:]))
            if recent_completed else None
        ),
        "synthetic_release_count": callback.synthetic_releases,
        "wall_sec": time.perf_counter() - started,
    }


def run_startup_occupancy(
    model, *, budget_steps: int, rollout_steps: int, seed: int
) -> dict:
    """Start from independent live occupancy samples without a warmup update."""
    if budget_steps == 0:
        return {}
    if budget_steps < 0 or budget_steps % rollout_steps:
        raise ValueError("occupancy budget must be a positive multiple of n_steps")
    if not getattr(model, "nosl_is_az", False):
        raise ValueError("occupancy startup currently requires NOSL-IS-AZ collection")
    callback = NoopCallback()
    callback.init_callback(model)
    callback.on_training_start({}, {})
    start_timesteps = int(model.num_timesteps)
    started = time.perf_counter()
    replay = getattr(model, "_nosl_is_az_replay", None)
    replay_cursor_before = None if replay is None else (
        int(replay.position), int(replay.size), int(replay.total_added)
    )
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = th.get_rng_state()
    cuda_state = (
        th.cuda.get_rng_state_all() if model.device.type == "cuda" else None
    )
    warmup_seed = int(seed) & 0xFFFFFFFF
    np.random.seed(warmup_seed)
    random.seed(warmup_seed)
    th.manual_seed(warmup_seed)
    if model.device.type == "cuda":
        th.cuda.manual_seed_all(warmup_seed)
    old_nonce_rng = getattr(model, "_startup_occupancy_nonce_rng", None)
    model._startup_occupancy_nonce_rng = np.random.default_rng(
        np.random.SeedSequence((warmup_seed, 0x4E4F4E43))
    )
    model._startup_occupancy_sampling = True
    try:
        total_blocks = budget_steps // rollout_steps
        # Occupancy startup intentionally does not emit normal training
        # metrics. Keep a low-frequency heartbeat so a long startup is
        # distinguishable from a stalled worker without changing sampling.
        progress_interval = max(1, min(8, total_blocks))
        for block in range(total_blocks):
            states = model.env.env_method("startup_occupancy_state")
            if all(state["accepted"] for state in states):
                break
            if not model.collect_rollouts(
                model.env, callback, model.rollout_buffer,
                n_rollout_steps=rollout_steps,
            ):
                raise RuntimeError("occupancy collection returned False")
            completed_blocks = block + 1
            if (
                completed_blocks % progress_interval == 0
                or completed_blocks == total_blocks
            ):
                states = model.env.env_method("startup_occupancy_state")
                accepted = sum(bool(state["accepted"]) for state in states)
                ages = [int(state["age"]) for state in states]
                print(
                    "STARTUP_OCCUPANCY_PROGRESS " + json.dumps({
                        "blocks": completed_blocks,
                        "blocks_total": total_blocks,
                        "accepted_envs": accepted,
                        "total_envs": len(states),
                        "active_age_mean": float(np.mean(ages)) if ages else 0.0,
                        "active_age_max": max(ages) if ages else 0,
                        "real_transitions_local": int(sum(
                            state["real_steps"] for state in states
                        )),
                        "elapsed_sec": time.perf_counter() - started,
                    }, sort_keys=True),
                    flush=True,
                )
    finally:
        callback.on_training_end()
        model._startup_occupancy_sampling = False
        model._startup_occupancy_nonce_rng = old_nonce_rng
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        th.set_rng_state(torch_state)
        if cuda_state is not None:
            th.cuda.set_rng_state_all(cuda_state)
    states = model.env.env_method("startup_occupancy_state")
    if not all(state["accepted"] for state in states):
        raise RuntimeError(
            "occupancy startup exhausted its budget without all environments "
            "accepted; no formal collection or optimizer update was started"
        )
    model.env.env_method("startup_occupancy_release")
    if hasattr(model.env, "episode_lengths"):
        model.env.episode_lengths[:] = [state["age"] for state in states]
    if hasattr(model.env, "episode_returns"):
        model.env.episode_returns[:] = [
            state["episode_return"] for state in states
        ]
    model._last_episode_starts = np.asarray(
        [state["age"] == 0 for state in states], dtype=bool
    )
    discarded = int(model.num_timesteps) - start_timesteps
    if discarded % (model.env.num_envs * rollout_steps):
        raise RuntimeError("occupancy startup transition count is misaligned")
    model.num_timesteps = start_timesteps
    model.rollout_buffer.reset()
    model._wait_for_reasoning_samples()
    model._reasoning_rollout_samples = []
    discarded_completed_roots = len(model._nosl_is_az_samples)
    discarded_pending_roots = sum(
        len(roots) for roots in model._nosl_is_az_pending_fights
    )
    model._nosl_is_az_samples = []
    # Formal sampling begins at the accepted state. Warmup roots before that
    # boundary must not later enter replay when the current fight terminates.
    model._nosl_is_az_pending_fights = [[] for _ in range(model.env.num_envs)]
    replay_cursor_after = None if replay is None else (
        int(replay.position), int(replay.size), int(replay.total_added)
    )
    if replay_cursor_after != replay_cursor_before:
        raise RuntimeError("occupancy startup mutated replay before formal training")
    for name in ("ep_info_buffer", "ep_success_buffer"):
        buffer = getattr(model, name, None)
        if buffer is not None:
            buffer.clear()
    logger = getattr(model, "logger", None)
    for name in ("name_to_value", "name_to_count", "name_to_excluded"):
        values = getattr(logger, name, None)
        if values is not None:
            values.clear()
    return {
        "discarded_transitions_local": discarded,
        "collection_blocks": discarded // (model.env.num_envs * rollout_steps),
        "rejected_episodes": sum(state["rejected_episodes"] for state in states),
        "real_transitions_local": sum(state["real_steps"] for state in states),
        "discarded_completed_combat_roots": discarded_completed_roots,
        "discarded_pending_combat_roots": discarded_pending_roots,
        "replay_size": 0 if replay is None else int(replay.size),
        "max_age": int(states[0]["max_age"]),
        "active_age_mean": float(np.mean([state["age"] for state in states])),
        "active_age_max": max(state["age"] for state in states),
        "wall_sec": time.perf_counter() - started,
    }


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
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@", description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument('--completed-game-budget', type=int, default=None,
                        help='Stop and save at the first collection reaching this natural full-game count (small-playout only).')
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument('--prepare-combat-outer', type=Path,
                        help='Construct a fresh lower actor/critic and frozen upper from this profile, save step zero, then exit')
    parser.add_argument('--material-directory', type=Path)
    parser.add_argument('--material-sha256')
    parser.add_argument(
        "--card-vector-checkpoint",
        type=Path,
        default=None,
        help="Freeze the anonymous card-rule encoder loaded from this v7 "
        "checkpoint while initializing the WholeRun policy.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--continuous", action="store_true",
                        help="Keep training until coordinated STOP; supports fixed-rate shared whole-run PPO, inner-only NOSL or joint-upper small-playout, with full recovery saves.")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Absolute global env-step target. When set, overrides --updates "
        "using checkpoint num_timesteps plus world_size*n_envs*n_steps/update.",
    )
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument('--boss-opening-dataset', type=Path,
                        help='Opening-only Act 2 Boss curriculum; 2:1:1 Champ/Automaton/Collector actors')
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument('--episode-rollout', action='store_true',
                        help='One complete fresh game per environment per rollout; no auto-replacement.')
    parser.add_argument('--episode-rollout-max-steps', type=int, default=6000)
    parser.add_argument('--episode-rollout-max-seconds', type=float, default=1800.)
    parser.add_argument('--independent-rollout-config', type=Path, default=None,
        help='JSON thread/coalescing/RNG configuration; checkpointed, complete natural cohorts only. '
             'A trained source requires an explicit complete-boundary fork to migrate.')
    parser.add_argument('--iteration-timeout-seconds', type=float, default=0.,
                        help='Hard round wall limit; timeout exits 124, resume only prior common save.')
    parser.add_argument('--episode-rollout-censored', choices=('error', 'exclude'), default=None)
    parser.add_argument(
        "--startup-stagger-steps",
        type=int,
        default=0,
        help="Before the first update, hold each environment for an independent "
        "stratified-random delay in [0, N], run N policy steps, then discard "
        "all warmup samples. N must be a multiple of --n-steps.",
    )
    parser.add_argument(
        "--startup-occupancy-budget", type=int, default=0,
        help="Maximum pre-training real environment steps per env, in n_steps "
        "blocks. Rejection-sample episode ages from the live frozen-policy "
        "occupancy distribution. Fail closed if any env remains unaccepted.",
    )
    parser.add_argument(
        "--startup-occupancy-max-age", type=int, default=6000,
        help="Exclusive upper bound for sampled startup ages. Values below "
        "the environment episode cap sample the occupancy distribution "
        "truncated to this many decisions without truncating formal episodes.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument('--small-playout-batch-size', type=int, default=None,
                        help='Online compact learner effective batch per rank; preserves upper PPO.')
    parser.add_argument('--small-playout-micro-batch-size', type=int, default=None,
                        help='Online compact learner forward/backward chunk size only.')
    parser.add_argument(
        "--upper-micro-batch-size", type=int, default=None,
        help="Override only strategic PPO's forward/backward chunk size; "
             "each epoch still accumulates all completed upper samples.",
    )
    parser.add_argument(
        "--upper-gae-lambda", type=float, default=None,
        help="Override strategic PPO GAE lambda (0..1); omission preserves the checkpoint value.",
    )
    parser.add_argument(
        "--upper-epochs", type=int, default=None,
        help="Strategic PPO passes per collection; retains the configured KL stopping policy. "
             "Omission preserves the checkpoint value.",
    )
    for flag, kind in (('learning-rate', float), ('value-learning-rate', float),
                       ('effective-batch-size', int), ('critic-effective-batch-size', int), ('critic-epochs', int)):
        parser.add_argument('--upper-' + flag, type=kind, default=None,
                            help='Unified upper override; omission preserves checkpoint configuration and Adam state.')
    parser.add_argument('--upper-critic-target', choices=('mc', 'frozen-lambda'), default=None,
                        help='Unified upper critic labels only; frozen-lambda uses collection-time values. '
                             'Omission preserves the checkpoint target and all Adam state.')
    parser.add_argument('--upper-measure-value-loss', action=argparse.BooleanOptionalAction, default=None,
                        help='Measure frozen full-cohort critic MSE before/after unified upper updates; '
                             'adds forward passes. Omission preserves checkpoint configuration.')
    parser.add_argument(
        "--upper-target-kl", type=float, default=None,
        help="Strategic PPO stops above 1.5 times this positive KL target; 0 disables KL early stopping. "
             "Omission preserves the checkpoint value; KL and nonfinite checks remain active.",
    )
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument('--upper-auxiliary-search', choices=('public-v1', 'off'), default=None,
                        help='Optional public upper search supervision; omission preserves checkpoint configuration.')
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Override the checkpoint gradient norm limit. Use 'inf' to "
        "disable gradient-norm scaling while retaining norm diagnostics.",
    )
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--hierarchical-combat-hp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use fight-local terminal-HP returns for combat and macro combat "
             "transitions for overworld. Omission preserves checkpoint mode.",
    )
    parser.add_argument(
        "--nosl-is-az-boss-transition-target",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt in to post-transition act-boss terminal HP targets; "
             "omission preserves the checkpoint setting.",
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
        "--card-language",
        action="store_true",
        help="Expose tokenized anonymous card programs in observations.",
    )
    parser.add_argument(
        "--combat-language",
        action="store_true",
        help="Expose tokenized enemy intents and live combat status bars.",
    )
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
    parser.add_argument("--vec", choices=("subproc", "dummy", "native", "native-subproc"), default="subproc")
    parser.add_argument('--native-search-threads', type=int, default=4)
    parser.add_argument("--start-method", default="fork")
    parser.add_argument(
        "--envs-per-worker",
        type=int,
        default=1,
        help="Independent environments/search trees hosted by one subprocess. "
             "Values above 1 batch worker IPC while preserving independent "
             "simulator state and NOSL trees.",
    )
    parser.add_argument("--save-every-updates", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--training-state-dir", type=Path, help="Per-rank policy/Adam/replay/RNG recovery generations")
    parser.add_argument("--resume-state", type=Path, help="Explicit complete per-rank recovery generation")
    parser.add_argument('--ppo-advantage-normalization', choices=('group256', 'rollout'),
        help='Shared PPO only: retain checkpoint setting, or choose legacy groups/global rollout moments')
    parser.add_argument('--ppo-update-config', type=Path,
        help='Shared pure PPO physical batching JSON: token_budget, max_rows, precision; omitted inherits checkpoint')
    parser.add_argument('--deck-training-schedule', type=Path,
        help='Deck latent preparation/anchor/search schedule JSON; omitted inherits checkpoint')
    parser.add_argument('--fork-public-lower',choices=('root','turn','state'),default=None,
                        help='On a lower-only fork, add a public adapter refreshed per real root or simulated state')
    parser.add_argument('--fork-public-weights',type=Path,default=None,
                        help='Optional parent-matched lower weights/Adam from the public-context probe')
    parser.add_argument('--training-scope', choices=('all', 'lower'), default=None,
                        help='Train all configured learners, or only the online compact lower policy. '
                             'Omitted inherits the checkpoint; changing a resumed scope requires a fork.')
    parser.add_argument(
        "--fork-state",
        type=Path,
        help="Restore a complete recovery generation while requiring new run "
        "and W&B identities; the source bundle remains unchanged.",
    )
    parser.add_argument(
        "--fork-reset-environments",
        action="store_true",
        help="With --fork-state, keep learner/replay/RNG state but reset the "
        "freshly constructed VecEnv. Complete natural cohorts can instead grow explicitly.",
    )
    parser.add_argument('--fork-grow-environments', action='store_true',
                        help='With --fork-state and --episode-rollout, preserve the complete natural '
                             'unified cohort and append new seeded environments; never reset source slots.')
    parser.add_argument('--resume-grow-environments', action='store_true',
                        help='Pure whole-run PPO only: retain live source slots and append seeded slots.')
    parser.add_argument(
        "--resume-reset-environments",
        action="store_true",
        help="With --resume-state, preserve learner/Adam/replay/RNG but reset "
        "in-flight environments when a simulator rule change makes their "
        "action journals incompatible.",
    )
    parser.add_argument("--stop-file", type=Path, help="Request a coordinated save and stop at the next update boundary")
    parser.add_argument("--checkpoint-prefix", default="maskable_ppo")
    parser.add_argument("--max-checkpoints", type=int, default=20)
    parser.add_argument('--checkpoint-keep-round-multiple', type=int, default=0,
                        help='For complete cohorts, retain latest divisible rollout round plus current; '
                             '0 keeps the normal count-based policy. Requires saving every round.')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--leaf-inference", choices=("eager", "cuda-graph"), default="eager",
                        help="Runtime combat leaf backend; search and update semantics are unchanged")
    parser.add_argument('--leaf-amp-mode', choices=('off', 'bf16', 'fp16'), default=None,
                        help='CUDA leaf graph precision override; default follows --amp-mode; learner/root precision stays unchanged')
    parser.add_argument('--leaf-compile', action=argparse.BooleanOptionalAction, default=True,
                        help='Compile graph forward; disable explicitly on platforms without Triton')
    parser.add_argument("--leaf-graph-profile-interval", type=int, default=0,
                        help="Sample CUDA stream stage timings every N leaf calls; 0 disables")
    parser.add_argument("--leaf-graph-batch-multiple", type=int, default=0,
                        help="Pad independent leaf rows to this multiple; 0 preserves exact batch shapes")
    parser.add_argument("--rollout-resource-interval", type=float, default=0,
                        help="Sample process-tree CPU/RSS and device GPU every N seconds for rollout and full iteration including update/sync/save; 0 disables")
    parser.add_argument("--wandb-project", default="sts-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--reasoning-coef", type=float, default=0.2,
        help="Auxiliary autoregressive reasoning loss coefficient.",
    )
    parser.add_argument(
        "--interactive-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override trace capture for the environment. Use --no-"
        "interactive-reasoning for the split-pointer card-vector run.",
    )
    parser.add_argument("--reasoning-batch-size", type=int, default=128)
    parser.add_argument("--reasoning-updates", type=int, default=4)
    parser.add_argument(
        "--nosl-is-az-upper-strategy", choices=("model", "rules"), default=None,
        help="Frozen upper decision strategy in inner-only NOSL: sampled model "
             "or overworld-v1 rules with deterministic model fallback.",
    )
    parser.add_argument(
        "--nosl-is-az-combat-strategy",
        choices=("model", "turn-planner", "small-playout"),
        default=None,
        help="Combat actor: checkpoint model search or fixed model-free turn planner.",
    )
    parser.add_argument("--rulebase-planner-worlds", type=int, default=None)
    parser.add_argument("--rulebase-planner-nodes", type=int, default=None)
    parser.add_argument("--rulebase-planner-beam", type=int, default=None)
    parser.add_argument("--rulebase-planner-depth", type=int, default=None)
    parser.add_argument("--rulebase-planner-turns", type=int, default=None)
    parser.add_argument(
        "--rulebase-planner-leaf-profile",
        choices=("legacy-v8", "time-clock-v9"),
        default=None,
    )
    parser.add_argument("--rulebase-planner-nonce", type=int, default=None)
    parser.add_argument(
        "--nosl-is-az-simulations",
        type=int,
        default=None,
        help="Override checkpoint NOSL simulation budget.",
    )
    parser.add_argument(
        "--nosl-is-az-depth",
        type=int,
        default=None,
        help="Override the checkpoint's NOSL search depth; 0 enables unlimited "
        "AlphaZero-style node expansion.",
    )
    parser.add_argument(
        "--nosl-is-az-end-turn-as-leaf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether End Turn stops search at the post-enemy-turn "
             "value boundary.",
    )
    parser.add_argument("--nosl-is-az-fixed-worldline", action=argparse.BooleanOptionalAction,
                        default=None, help="Override fixed future RNG within each search tree.")
    parser.add_argument("--disable-entity-transition-auxiliary", action="store_true",
                        help="Stop collecting and training the entity combat action-effect head.")
    parser.add_argument(
        "--nosl-is-az-prior-temperature",
        type=float,
        default=None,
        help="Override checkpoint policy-logit temperature for native search.",
    )
    parser.add_argument(
        "--nosl-is-az-search-mode",
        choices=tuple(NOSL_IS_AZ_SEARCH_MODES),
        default=None,
        help="Override native search with puct, gumbel, or full_gumbel.",
    )
    parser.add_argument(
        "--nosl-is-az-paper-full-gumbel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the canonical strict-paper Full Gumbel/MuZero settings.",
    )
    parser.add_argument(
        "--nosl-is-az-gumbel-candidates",
        type=int,
        default=None,
        help="Override root Gumbel top-k size; 0 means all legal actions.",
    )
    parser.add_argument("--nosl-is-az-gumbel-scale", type=float, default=None)
    parser.add_argument('--nosl-is-az-merge-equivalent-actions',
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--nosl-is-az-gumbel-value-scale", type=float, default=None
    )
    parser.add_argument(
        "--nosl-is-az-gumbel-max-visit-init", type=float, default=None
    )
    parser.add_argument(
        "--nosl-is-az-gumbel-rescale-q",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--nosl-is-az-gumbel-use-mixed-value",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--nosl-is-az-gumbel-interior-uniformity",
        type=float,
        default=None,
        help="Full-Gumbel non-root smoothing strength in [0,1].",
    )
    parser.add_argument(
        "--nosl-is-az-gumbel-interior-coverage-fraction",
        type=float,
        default=None,
        help="Prior-independent Full-Gumbel interior coverage fraction in [0,1].",
    )
    parser.add_argument(
        "--nosl-is-az-gumbel-interior-value-mode",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Interior V aggregation: 0=mean, 1=max, 2=progressive mean-to-max.",
    )
    parser.add_argument(
        "--nosl-is-az-gumbel-interior-max-mix-visits",
        type=float,
        default=None,
        help="Visit scale at which progressive interior V reaches max.",
    )
    parser.add_argument(
        "--nosl-is-az-batch-size",
        type=int,
        default=None,
        help="Requested effective AlphaZero root batch; replay mode samples up "
             "to this many local roots, while no-replay mode consumes all "
             "available fresh roots and uses this as the recorded global cap.",
    )
    parser.add_argument(
        "--nosl-is-az-updates",
        type=int,
        default=None,
        help="Joint policy/value updates per collection (warmup adds value updates).",
    )
    parser.add_argument("--nosl-is-az-update-schedule",
                        type=parse_update_schedule, default=None,
                        help="constant, value-warmup, or a schedule JSON object.")
    parser.add_argument("--nosl-is-az-policy-coef", type=float, default=None,
                        help="NOSL group cross-entropy weight; omission preserves the checkpoint.")
    parser.add_argument("--nosl-is-az-value-coef", type=float, default=None,
                        help="NOSL value/distribution loss weight; omission preserves the checkpoint.")
    parser.add_argument(
        "--nosl-is-az-replay",
        dest="nosl_is_az_replay_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the labelled-combat replay buffer. Omission preserves "
             "the checkpoint setting; legacy checkpoints default to enabled.",
    )
    parser.add_argument(
        "--nosl-is-az-train-shared-embeddings",
        action="store_true",
        help="Reset shared card/relic/potion embeddings with the run seed and "
             "train them alongside the combat specialist in inner-only NOSL.",
    )
    parser.add_argument(
        "--nosl-is-az-micro-batch-size",
        type=int,
        default=None,
        help="Maximum replay roots in one forward/backward pass; gradients are "
        "sample-weighted and accumulated into one effective AZ batch.",
    )
    parser.add_argument(
        "--nosl-is-az-replay-capacity",
        type=int,
        default=None,
        help="Per-rank FIFO replay capacity in labelled combat roots.",
    )
    parser.add_argument(
        "--nosl-is-az-replay-min-size",
        type=int,
        default=None,
        help="Minimum local replay roots required before AZ updates begin.",
    )
    parser.add_argument(
        "--nosl-is-az-replay-max-old-fraction",
        type=float,
        default=None,
        help="Maximum fraction of each replay batch drawn from before the "
             "current collection; lower values reserve more batch entries "
             "for the newest roots.",
    )
    parser.add_argument(
        "--nosl-is-az-leaf-batch-size",
        type=int,
        default=None,
        help="Override C++ leaves returned per worker round; larger values "
              "reduce worker synchronization and increase GPU batch size.",
    )
    parser.add_argument(
        "--nosl-is-az-inference-batch-size",
        type=int,
        default=None,
        help="Target GPU batch assembled from independent worker trees; this "
             "does not change per-tree leaf concurrency.",
    )
    parser.add_argument(
        "--nosl-is-az-inference-max-wait-ms",
        type=float,
        default=None,
        help="Fixed-mode partial-batch delay, or adaptive-mode straggler "
             "watchdog floor. Omission uses the checkpoint value or 0.5 ms.",
    )
    parser.add_argument(
        "--nosl-is-az-inference-schedule",
        choices=("fixed_wait", "adaptive"),
        default=None,
        help="Cross-tree GPU batching policy. Adaptive mode uses active-tree "
             "capacity and backpressure while preserving one leaf per tree.",
    )
    parser.add_argument("--nosl-is-az-dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--nosl-is-az-dirichlet-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of reasoning microbatches accumulated before one optimizer step.",
    )
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
    parser.add_argument(
        "--ring-streams", type=int, choices=(1, 2, 4, 8), default=1,
        help="Independent SSH streams per ring edge; uses consecutive listener ports.",
    )
    parser.add_argument(
        "--profile-cuda-memory",
        action="store_true",
        help="Record per-micro-batch CUDA allocated/reserved peaks and the "
             "observation mix that produced the largest peak.",
    )
    parser.add_argument(
        "--phasewise-backward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Backpropagate combat and overworld halves of one logical "
             "minibatch sequentially. This preserves the phase-balanced PPO "
             "objective while releasing one specialist's activations early.",
    )
    return parser


def normalize_checkpoint_path(path: Path) -> Path:
    """SB3's loader appends .zip itself; accept either spelling at the CLI."""
    return path.with_suffix("") if path.suffix.lower() == ".zip" else path


def run_cmd(cmd: list[str], *, timeout: float | None = None) -> None:
    subprocess.run(cmd, check=True, timeout=timeout)


def ssh_control_directory(root: Path, run_id: str, rank: int) -> Path:
    # OpenSSH adds a 40-character %C hash and a 17-character temporary suffix.
    # Experiment names must not determine the length of a Unix socket pathname.
    identity = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    return Path(root) / f'{identity}-r{rank}'


def ssh_options(peer: Peer, key: Path, control_dir: Path) -> list[str]:
    if len(str(control_dir).encode()) + 1 + 40 + 17 >= 104:
        raise ValueError('SSH control socket path too long; use a shorter --ssh-control-dir')
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
    cmd = [
        "ssh",
        *ssh_options(peer, key, control_dir),
        "-p",
        str(peer.port),
        f"{peer.user}@{peer.host}",
        command,
    ]
    # Peer SSH daemons occasionally reject a burst of rank-startup sessions
    # (kex_exchange_identification). Retry transient setup/file-barrier calls
    # without changing the steady-state persistent ring transport.
    for attempt in range(4):
        try:
            run_cmd(cmd, timeout=timeout)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))


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


def count_nonfinite_gradients(flat: th.Tensor) -> int:
    """Scan the transport's CPU FP32 view without Torch's temporary passes."""
    if flat.device.type != "cpu" or flat.dtype != th.float32:
        raise TypeError("gradient diagnostics require a CPU float32 tensor")
    return int(np.count_nonzero(~np.isfinite(flat.detach().numpy())))


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
        if key.startswith((
            "game/", "boss/", "upper/", "rollout/", "speed/", "hierarchical/", "nosl_is_az/", "outer_return/", "frozen_upper/", "pure_rl/", "deck_search/"
        )):
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
    *,
    profile_cuda_memory: bool = False,
    phasewise_backward: bool = False,
) -> dict[str, object]:
    cuda_profile = None
    if profile_cuda_memory:
        if model.device.type != "cuda":
            raise ValueError("CUDA memory profiling requires a CUDA device")
        th.cuda.reset_peak_memory_stats(model.device)
        cuda_profile = {
            "allocated_before": th.cuda.memory_allocated(model.device),
            "reserved_before": th.cuda.memory_reserved(model.device),
        }

    if getattr(model, 'nosl_supervision_contract', None) == 'shared-wholerun-ppo-room-v1':
        loss_value = ppo_chunked_backward(model, rollout_data, clip_range, clip_range_vf)
        if cuda_profile is not None:
            cuda_profile.update(allocated_after_backward=th.cuda.memory_allocated(model.device),
                peak_allocated=th.cuda.max_memory_allocated(model.device),
                peak_reserved=th.cuda.max_memory_reserved(model.device))
        return dict(loss=loss_value, combat_value_loss=None, overworld_value_loss=None,
                    cuda_profile=cuda_profile)
    phase = rollout_data.observations["phase"].long().reshape(-1)
    valid = th.isfinite(rollout_data.advantages) & th.isfinite(
        rollout_data.returns
    )
    combat = phase == V.PHASE_VOCAB_INDEX["combat"]
    if bool(getattr(model, "nosl_is_az", False)):
        valid = valid & ~combat
    if phasewise_backward and model.hierarchical_combat_hp:
        index_groups = [
            th.nonzero(valid & group, as_tuple=False).flatten()
            for group in (combat, ~combat)
        ]
        index_groups = [indices for indices in index_groups if indices.numel()]
    else:
        index_groups = [None]

    if not index_groups:
        # Match the combined objective's differentiable zero when every row
        # is unresolved or supervised by the combat learner instead of PPO.
        index_groups = [None]

    group_scale = 1.0 / len(index_groups)
    loss_value = 0.0
    combat_value_loss = None
    overworld_value_loss = None
    peak_forward_allocated = 0
    peak_observation_mix = None
    for indices in index_groups:
        def after_forward(observations):
            nonlocal peak_forward_allocated, peak_observation_mix
            if cuda_profile is not None:
                allocated = th.cuda.memory_allocated(model.device)
                if allocated > peak_forward_allocated:
                    peak_forward_allocated = allocated
                    peak_observation_mix = cuda_observation_mix(observations)

        components, loss = ppo_forward(model, rollout_data, clip_range, clip_range_vf,
                                       indices=indices, after_forward=after_forward)
        loss = group_scale * loss
        loss_value += float(loss.detach().cpu())
        if components.combat_value_loss is not None:
            combat_value_loss = float(
                components.combat_value_loss.detach().cpu()
            )
        if components.overworld_value_loss is not None:
            overworld_value_loss = float(
                components.overworld_value_loss.detach().cpu()
            )
        loss.backward()

    if cuda_profile is not None:
        cuda_profile.update({
            "allocated_after_backward": th.cuda.memory_allocated(model.device),
            "allocated_after_forward": peak_forward_allocated,
            "peak_allocated": th.cuda.max_memory_allocated(model.device),
            "peak_reserved": th.cuda.max_memory_reserved(model.device),
            "observation_mix": peak_observation_mix,
        })
    return {
        "loss": loss_value,
        "combat_value_loss": combat_value_loss,
        "overworld_value_loss": overworld_value_loss,
        "cuda_profile": cuda_profile,
    }


def cuda_observation_mix(
    observations: dict[str, th.Tensor],
) -> dict[str, object]:
    """Describe the routed samples and dynamic token widths in one batch."""
    phase = observations["phase"].long().reshape(-1)
    phase_counts = {
        V.PHASE_VOCAB[phase_id]: int((phase == phase_id).sum().item())
        for phase_id in range(len(V.PHASE_VOCAB))
        if bool((phase == phase_id).any())
    }

    def active_width(key: str, indices: th.Tensor | None = None) -> int:
        values = observations[key]
        if indices is not None:
            values = values.index_select(0, indices)
        active_columns = values.long().ne(0).any(dim=0).nonzero(
            as_tuple=False
        )
        return (
            int(active_columns[-1, 0].item()) + 1
            if active_columns.numel() else 1
        )

    combat_id = V.PHASE_VOCAB_INDEX["combat"]
    combat_indices = th.nonzero(phase == combat_id, as_tuple=False).flatten()
    widths: dict[str, int] = {}
    if combat_indices.numel():
        widths["combat_relic"] = active_width("relics", combat_indices)
    for phase_id in range(len(V.PHASE_VOCAB)):
        if phase_id == combat_id:
            continue
        indices = th.nonzero(phase == phase_id, as_tuple=False).flatten()
        if not indices.numel():
            continue
        name = V.PHASE_VOCAB[phase_id]
        widths[f"{name}_deck"] = active_width("deck_ids", indices)
        widths[f"{name}_relic"] = active_width("relics", indices)
        if name == "card_select":
            widths["card_select_options"] = active_width(
                "select_card_ids", indices
            )
    return {"phase_counts": phase_counts, "active_widths": widths}




def synchronize_gradient_step(
    model: AccumulatingMaskablePPO,
    *,
    rank: int,
    world_size: int,
    peers: list[Peer],
    key: Path,
    control_dir: Path,
    local_root: Path,
    remote_root: str,
    run_id: str,
    tag: str,
    ring_transport,
    barrier_timeout: float,
) -> dict[str, float]:
    """Average current gradients over all ranks using the selected transport."""
    if world_size == 1 and ring_transport is None:
        if rank != 0 or peers:
            raise ValueError('single-rank gradients require rank zero and no peers')
        started = time.perf_counter()
        parameters = gradient_parameters(model)
        flat = th.cat([(p.grad.detach() if p.grad is not None else th.zeros_like(p)).reshape(-1)
                       for p in parameters])
        if flat.dtype != th.float32:
            raise TypeError(f'distributed gradients must be float32, got {flat.dtype}')
        bad = ~th.isfinite(flat)
        nonfinite = int(bad.sum().item())
        first_parameter = None
        if nonfinite:
            index = int(th.nonzero(bad, as_tuple=False)[0, 0])
            offset = 0
            for name, p in model.policy.named_parameters():
                if p.requires_grad:
                    offset += p.numel()
                    if index < offset:
                        first_parameter = name
                        break
        pack_seconds = time.perf_counter() - started
        started = time.perf_counter()
        # Preserve the file transport's zero + grad, then * 1 arithmetic,
        # including signed zero. Materialize unused gradients: Adam momentum
        # must still advance these parameters exactly as on ordinary resume.
        averaged = th.zeros_like(flat).add_(flat).mul_(1.)
        install_flat_gradients(model, averaged)
        if flat.is_cuda:
            th.cuda.synchronize(flat.device)
        return dict(gradient_pack_sec=pack_seconds, sync_sec=time.perf_counter() - started,
                    sync_mbps=0., grad_bytes=float(flat.numel() * flat.element_size()),
                    local_nonfinite_gradients=float(nonfinite),
                    synchronized_nonfinite_gradients=float(nonfinite),
                    first_local_nonfinite_parameter=first_parameter)
    pack_started = time.perf_counter()
    local_flat = pack_gradients(model)
    local_nonfinite = count_nonfinite_gradients(local_flat)
    first_nonfinite_parameter = None
    if local_nonfinite:
        first_index = int(th.nonzero(~th.isfinite(local_flat), as_tuple=False)[0, 0])
        offset = 0
        for name, parameter in model.policy.named_parameters():
            if not parameter.requires_grad:
                continue
            offset += parameter.numel()
            if first_index < offset:
                first_nonfinite_parameter = name
                break
    pack_seconds = time.perf_counter() - pack_started
    grad_bytes = local_flat.numel() * local_flat.element_size()
    sync_started = time.perf_counter()
    sync_mbps = 0.0
    paths_to_remove = []
    if ring_transport is not None:
        local_flat, bytes_sent = ring_transport.average(local_flat)
        sync_mbps = bytes_sent / 1024 / 1024 / max(
            time.perf_counter() - sync_started, 1e-9
        )
        install_flat_gradients(model, local_flat)
    else:
        grad_path = local_root / f"grad_rank{rank}_{tag}.pt"
        save_gradients(local_flat, grad_path)
        remote_name = f"rank{rank}/grad_rank{rank}_{tag}.pt"
        _elapsed, sync_mbps = sync_file_to_peers(
            local_file=grad_path,
            remote_name=remote_name,
            peers=peers,
            key=key,
            control_dir=control_dir,
            remote_root=f"{remote_root}/{run_id}",
            timeout=barrier_timeout,
        )
        peer_paths = [
            Path(remote_root) / run_id / f"rank{peer.rank}"
            / f"grad_rank{peer.rank}_{tag}.pt"
            for peer in peers
        ]
        for path in peer_paths:
            wait_for_sized_file(path, timeout=barrier_timeout, expected_size=grad_bytes)
        average_peer_gradients(
            model,
            local_flat,
            list(zip((peer.rank for peer in peers), peer_paths, strict=True)),
            rank=rank,
            world_size=world_size,
        )
        local_flat = pack_gradients(model)
        paths_to_remove = [grad_path, *peer_paths]
    synchronized_nonfinite = count_nonfinite_gradients(local_flat)
    for path in paths_to_remove:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return {
        "gradient_pack_sec": pack_seconds,
        "sync_sec": time.perf_counter() - sync_started,
        "sync_mbps": sync_mbps,
        "grad_bytes": float(grad_bytes),
        "local_nonfinite_gradients": float(local_nonfinite),
        "synchronized_nonfinite_gradients": float(synchronized_nonfinite),
        "first_local_nonfinite_parameter": first_nonfinite_parameter,
    }


def save_nonfinite_nosl_diagnostic(
    model: AccumulatingMaskablePPO, *, local_root: Path,
    update: int, inner_update: int, rank: int,
    local_nonfinite_gradients: int, first_nonfinite_parameter: str | None,
) -> Path:
    """Persist the exact offending input and finite pre-step state on its rank."""
    directory = local_root / "nonfinite-nosl-diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"step-{int(model.num_timesteps):012d}-u{update:06d}"
        f"-i{inner_update:04d}-rank{rank}.pt"
    )
    payload = {
        "rank": rank,
        "global_step": int(model.num_timesteps),
        "update": update,
        "inner_update": inner_update,
        "local_nonfinite_gradients": local_nonfinite_gradients,
        "first_nonfinite_parameter": first_nonfinite_parameter,
        "batch": getattr(model, "_nosl_is_az_diagnostic_batch", None),
        "numpy_rng_before_sample": getattr(model, "_nosl_is_az_diagnostic_rng", None),
        "fresh_start_id": getattr(model, "_nosl_is_az_diagnostic_fresh_start_id", None),
        "policy": {name: tensor.detach().cpu().clone()
                   for name, tensor in model.policy.state_dict().items()},
        "optimizer": model.policy.optimizer.state_dict(),
    }
    temporary = path.with_suffix(".tmp")
    th.save(payload, temporary, pickle_protocol=5)
    temporary.replace(path)
    return path


def average_scalar_dict(
    values: dict[str, float], ring_transport,
) -> dict[str, float]:
    """Average a fixed scalar schema over ranks when the SSH ring is active."""
    if ring_transport is None:
        return values
    keys = tuple(sorted(values))
    tensor = th.tensor([values[key] for key in keys], dtype=th.float32)
    averaged, _bytes = ring_transport.average(tensor, schema=keys)
    return {key: float(value) for key, value in zip(keys, averaged.tolist())}


@combat_update
def distributed_nosl_is_az_update(
    model: AccumulatingMaskablePPO,
    *,
    rank: int,
    world_size: int,
    peers: list[Peer],
    key: Path,
    control_dir: Path,
    local_root: Path,
    remote_root: str,
    run_id: str,
    update: int,
    ring_transport,
    barrier_timeout: float,
    prepare_replay: bool = True,
    consume_fresh_samples: bool = True,
    update_count: int | None = None,
    value_only_updates: int = 0,
    inner_update: int = 0,
) -> dict[str, object]:
    """Run synchronized inner-only NOSL updates.

    Each rank first accumulates its local AZ loss gradient without stepping
    Adam.  The flat gradient is then averaged through the same ring used by
    ordinary PPO before every rank takes one identical optimizer step.  The
    sample-count weighting keeps ranks with fewer completed combats from
    contributing an oversized mean gradient.  When ``model.nosl_is_az_updates``
    is greater than one, each configured update is a separate replay sample,
    all-reduce, and Adam step; this preserves the semantics of multiple local
    optimizer steps in the distributed path.
    """
    if not model.nosl_is_az or not model.nosl_is_az_inner_only:
        raise ValueError("distributed NOSL update requires inner-only mode")
    configured_updates = int(model.nosl_is_az_updates if update_count is None else update_count)
    if configured_updates < 1:
        raise ValueError("distributed NOSL updates must be positive")
    if not 0 <= value_only_updates <= configured_updates:
        raise ValueError("invalid value-only update budget")

    # The one-step implementation below deliberately keeps its collective
    # ordering small and explicit.  Re-enter it once per requested inner
    # update, so every Adam step is synchronized across all ranks.  Replay is
    # prepared only on the first pass; this is important for the legacy
    # non-replay mode where fresh roots are one-shot samples.
    if configured_updates > 1:
        results = [
            distributed_nosl_is_az_update(
                model,
                rank=rank,
                world_size=world_size,
                peers=peers,
                key=key,
                control_dir=control_dir,
                local_root=local_root,
                remote_root=remote_root,
                run_id=run_id,
                update=update,
                ring_transport=ring_transport,
                barrier_timeout=barrier_timeout,
                prepare_replay=prepare_replay if inner_update == 0 else False,
                consume_fresh_samples=False,
                update_count=1,
                value_only_updates=int(inner_update < value_only_updates),
                inner_update=inner_update,
            )
            for inner_update in range(configured_updates)
        ]

        if (
            consume_fresh_samples
            and not bool(getattr(model, "nosl_is_az_replay_enabled", True))
        ):
            # Every synchronized optimizer step above must see the same fresh
            # collection.  Consume it only after the complete update group.
            model._nosl_is_az_fresh_samples = []

        # Preserve the logger-facing shape while combining repeated updates.
        # Losses and model diagnostics are root-weighted; timings and counts
        # are additive; replay/configuration fields describe the final pass.
        if not results:
            return {}
        aggregated = dict(results[-1])
        metric_keys = {
            "policy_loss", "value_loss", "target_entropy",
            "policy_entropy", "target_policy_kl", "policy_cross_entropy",
            "target_effective_actions", "target_top1_agreement",
            "legal_actions_mean", "illegal_target_mass", "target_sum_error",
            "value_mae", "value_bias", "finite_fraction",
            "replay_old_fraction", "replay_current_fraction",
        }
        if getattr(model, '_combat_distribution', False):
            from rl.algorithms.combat_distribution import metrics_for
            metric_keys.update(metrics_for(model.nosl_supervision_contract))
        additive_keys = {
            "roots", "global_roots", "gradient_batches", "updates", "metric_count",
            "replay_sampled_roots", "backward_sec", "sync_sec",
            "optimizer_step_sec", "skipped_nonfinite_updates",
            "local_nonfinite_gradients", "synchronized_nonfinite_gradients",
            "value_only_updates", "joint_updates",
        }
        weights = [max(0.0, float(row.get("metric_count", 0.0))) for row in results]
        weight_total = sum(weights)
        for key in aggregated:
            values = [row.get(key) for row in results]
            if not all(isinstance(value, (int, float, np.number)) for value in values):
                continue
            numeric = [float(value) for value in values]
            if key in additive_keys:
                aggregated[key] = float(sum(numeric))
            elif key in metric_keys and weight_total > 0.0:
                aggregated[key] = float(
                    sum(value * weight for value, weight in zip(numeric, weights))
                    / weight_total
                )
        # Combine means and standard deviations through their second moments.
        # Averaging standard deviations is incorrect when update batches have
        # different means (as replay batches generally do).
        for mean_key, std_key in (
            ("value_target_mean", "value_target_std"),
            ("pred_value_mean", "pred_value_std"),
        ):
            if weight_total <= 0.0:
                aggregated[mean_key] = 0.0
                aggregated[std_key] = 0.0
                continue
            mean = sum(
                float(row.get(mean_key, 0.0)) * weight
                for row, weight in zip(results, weights)
            ) / weight_total
            second_moment = sum(
                (
                    float(row.get(std_key, 0.0)) ** 2
                    + float(row.get(mean_key, 0.0)) ** 2
                ) * weight
                for row, weight in zip(results, weights)
            ) / weight_total
            aggregated[mean_key] = float(mean)
            aggregated[std_key] = float(np.sqrt(max(
                0.0, second_moment - mean * mean
            )))
        populated = [
            row for row, weight in zip(results, weights) if weight > 0.0
        ]
        if populated:
            aggregated["value_target_min"] = float(min(
                float(row["value_target_min"]) for row in populated
            ))
            aggregated["value_target_max"] = float(max(
                float(row["value_target_max"]) for row in populated
            ))
        else:
            aggregated["value_target_min"] = 0.0
            aggregated["value_target_max"] = 0.0
        # These values are repeated configuration or final replay state.  Keep
        # them from the final inner update, while averaging per-step scalars
        # whose value is meaningful over the whole collection update.
        for key in ("sync_mbps", "gradient_norm_preclip"):
            values = [float(row.get(key, 0.0)) for row in results]
            aggregated[key] = float(np.mean(values)) if values else 0.0
        aggregated["grad_bytes"] = int(max(int(row.get("grad_bytes", 0)) for row in results))
        return aggregated

    if prepare_replay:
        model._prepare_nosl_is_az_replay()
    local_count = model._nosl_is_az_replay_train_count()
    count_average = average_scalar_dict(
        {"nosl_is_az_valid_roots": float(local_count)}, ring_transport
    )
    # ``average_scalar_dict`` is a collective only for the SSH ring.  The
    # legacy file transport has no scalar collective, so do not pretend its
    # local count was already averaged across ranks (that would inflate every
    # diagnostic by ``world_size``).
    scalar_world = float(world_size if ring_transport is not None else 1)
    global_count = float(
        count_average["nosl_is_az_valid_roots"] * scalar_world
    )
    gradient_scale = (
        float(scalar_world * local_count / global_count)
        if global_count > 0.0
        else 0.0
    )

    if model.device.type == "cuda":
        th.cuda.synchronize(model.device)
    backward_started = time.perf_counter()
    az_stats = model._train_nosl_is_az(
        step_optimizer=False,
        gradient_scale=gradient_scale,
        prepare_replay=False,
        consume_fresh_samples=consume_fresh_samples,
        update_count=1,
        value_only_updates=value_only_updates,
    )
    if model.device.type == "cuda":
        th.cuda.synchronize(model.device)
    backward_seconds = time.perf_counter() - backward_started

    # The local trainer keeps exact per-sample sums for diagnostics.  Reduce
    # those small vectors once so loss/entropy/value statistics are weighted by
    # the number of roots contributed by each rank.  This is separate from the
    # large parameter-gradient reduction below and is negligible in size.
    diagnostic_names = (
        "target_entropy",
        "policy_entropy",
        "target_policy_kl",
        "policy_cross_entropy",
        "target_effective_actions",
        "target_top1_agreement",
        "legal_actions",
        "illegal_target_mass",
        "target_sum_error",
        "value_target",
        "pred_value",
        "value_mae",
        "value_bias",
        "finite_fraction",
    )
    if getattr(model, '_combat_distribution', False):
        from rl.algorithms.combat_distribution import metrics_for
        diagnostic_names += metrics_for(model.nosl_supervision_contract)
    local_metric_sums = getattr(model, "_nosl_is_az_metric_sums", {})
    local_metric_sumsq = getattr(model, "_nosl_is_az_metric_sumsq", {})
    legacy_metric_count = float(getattr(
        model, "_nosl_is_az_metric_count", 0.0
    ))
    local_metric_counts = getattr(model, "_nosl_is_az_metric_counts", {})
    diagnostic_payload = {}
    for name in diagnostic_names:
        diagnostic_payload[f"sum:{name}"] = float(local_metric_sums.get(name, 0.0))
        diagnostic_payload[f"sumsq:{name}"] = float(
            local_metric_sumsq.get(name, 0.0)
        )
        diagnostic_payload[f"count:{name}"] = float(
            local_metric_counts.get(name, legacy_metric_count)
        )
    diagnostic_global = average_scalar_dict(diagnostic_payload, ring_transport)

    def diagnostic_count(name: str) -> float:
        return float(diagnostic_global[f"count:{name}"] * scalar_world)

    def diagnostic_mean(name: str) -> float:
        count = diagnostic_count(name)
        if count <= 0.0:
            return 0.0
        return float(
            diagnostic_global[f"sum:{name}"] * scalar_world / count
        )

    def diagnostic_std(name: str) -> float:
        count = diagnostic_count(name)
        if count <= 0.0:
            return 0.0
        mean = diagnostic_mean(name)
        second = float(
            diagnostic_global[f"sumsq:{name}"] * scalar_world / count
        )
        return float(np.sqrt(max(0.0, second - mean * mean)))

    diagnostic_values = {
        **{name:diagnostic_mean(name) for name in diagnostic_names[14:]},
        "metric_count": diagnostic_count("value_target"),
        "target_entropy": diagnostic_mean("target_entropy"),
        "policy_entropy": diagnostic_mean("policy_entropy"),
        "target_policy_kl": diagnostic_mean("target_policy_kl"),
        "policy_cross_entropy": diagnostic_mean("policy_cross_entropy"),
        "target_effective_actions": diagnostic_mean("target_effective_actions"),
        "target_top1_agreement": diagnostic_mean("target_top1_agreement"),
        "legal_actions_mean": diagnostic_mean("legal_actions"),
        "illegal_target_mass": diagnostic_mean("illegal_target_mass"),
        "target_sum_error": diagnostic_mean("target_sum_error"),
        "value_target_mean": diagnostic_mean("value_target"),
        "value_target_std": diagnostic_std("value_target"),
        "value_target_min": float(az_stats.get("value_target_min", 0.0)),
        "value_target_max": float(az_stats.get("value_target_max", 0.0)),
        "pred_value_mean": diagnostic_mean("pred_value"),
        "pred_value_std": diagnostic_std("pred_value"),
        "value_mae": diagnostic_mean("value_mae"),
        "value_bias": diagnostic_mean("value_bias"),
        "finite_fraction": diagnostic_mean("finite_fraction"),
    }

    sync_seconds = 0.0
    sync_mbps = 0.0
    grad_bytes = 0
    optimizer_step_seconds = 0.0
    gradient_norm_preclip = 0.0
    skipped_nonfinite = 0.0
    local_nonfinite_gradients = 0.0
    synchronized_nonfinite_gradients = 0.0
    if global_count > 0.0:
        sync = synchronize_gradient_step(
            model,
            rank=rank,
            world_size=world_size,
            peers=peers,
            key=key,
            control_dir=control_dir,
            local_root=local_root,
            remote_root=remote_root,
            run_id=run_id,
            tag=f"u{update:06d}_nosl_i{inner_update:04d}",
            ring_transport=ring_transport,
            barrier_timeout=barrier_timeout,
        )
        sync_seconds = float(sync["sync_sec"])
        sync_mbps = float(sync["sync_mbps"])
        grad_bytes = int(sync["grad_bytes"])
        local_nonfinite_gradients = float(sync.get("local_nonfinite_gradients", 0.0))
        synchronized_nonfinite_gradients = float(sync.get("synchronized_nonfinite_gradients", 0.0))
        if synchronized_nonfinite_gradients:
            # Every rank received the same averaged vector.  Discard this
            # update on every rank before Adam or clipping can poison weights.
            model.policy.optimizer.zero_grad()
            skipped_nonfinite = 1.0
            model._nosl_is_az_skipped_nonfinite_updates = int(getattr(
                model, "_nosl_is_az_skipped_nonfinite_updates", 0,
            )) + 1
            consecutive = int(getattr(model,
                "_nosl_is_az_consecutive_nonfinite_updates", 0)) + 1
            model._nosl_is_az_consecutive_nonfinite_updates = consecutive
            diagnostic_path = None
            if consecutive == 1 and local_nonfinite_gradients:
                if getattr(model, "_nosl_is_az_diagnostic_batch", None) is not None:
                    diagnostic_path = save_nonfinite_nosl_diagnostic(
                        model, local_root=local_root, update=update,
                        inner_update=inner_update, rank=rank,
                        local_nonfinite_gradients=int(local_nonfinite_gradients),
                        first_nonfinite_parameter=sync.get("first_local_nonfinite_parameter"),
                    )
            print("NONFINITE_NOSL_GRADIENT " + json.dumps({
                "rank": rank, "global_step": int(model.num_timesteps),
                "inner_update": inner_update,
                "local_nonfinite_gradients": int(local_nonfinite_gradients),
                "synchronized_nonfinite_gradients": int(synchronized_nonfinite_gradients),
                "first_local_nonfinite_parameter": sync.get("first_local_nonfinite_parameter"),
                "consecutive_skips": consecutive,
                "diagnostic_path": str(diagnostic_path) if diagnostic_path else None,
            }, sort_keys=True), flush=True)
            if consecutive >= 3:
                raise FloatingPointError(
                    "three consecutive synchronized NOSL gradients were nonfinite; "
                    "stopped before modifying model or optimizer"
                )
        else:
            if model.device.type == "cuda":
                th.cuda.synchronize(model.device)
            step_started = time.perf_counter()
            gradient_norm_preclip = float(optimizer_step(model, value_only=bool(value_only_updates)))
            model._nosl_is_az_consecutive_nonfinite_updates = 0
            if model.device.type == "cuda":
                th.cuda.synchronize(model.device)
            optimizer_step_seconds = time.perf_counter() - step_started
        for name in (
            "_nosl_is_az_diagnostic_batch", "_nosl_is_az_diagnostic_rng",
            "_nosl_is_az_diagnostic_fresh_start_id",
        ):
            model.__dict__.pop(name, None)

    # Record the value installed in the optimizer rather than only the CLI
    # request.  This catches accidental schedule/checkpoint overrides in a
    # resumed distributed run.
    effective_learning_rate = float(
        model.policy.optimizer.param_groups[0].get("lr", 0.0)
    )
    rollout_timings = getattr(model, "_nosl_is_az_rollout_timings", {})

    result = {
        "roots": float(az_stats["roots"]),
        "policy_loss": float(az_stats["policy_loss"]),
        "value_loss": float(az_stats["value_loss"]),
        "updates": 1.0 if global_count > 0.0 and not skipped_nonfinite else 0.0,
        "value_only_updates": float(global_count > 0.0 and value_only_updates > 0 and not skipped_nonfinite),
        "joint_updates": float(global_count > 0.0 and value_only_updates == 0 and not skipped_nonfinite),
        "skipped_nonfinite_updates": skipped_nonfinite,
        "skipped_nonfinite_updates_total": float(getattr(model,
            "_nosl_is_az_skipped_nonfinite_updates", 0)),
        "local_nonfinite_gradients": local_nonfinite_gradients,
        "synchronized_nonfinite_gradients": synchronized_nonfinite_gradients,
        "gradient_batches": float(az_stats.get("gradient_batches", 0.0)),
        "global_roots": global_count,
        "backward_sec": backward_seconds,
        "sync_sec": sync_seconds,
        "sync_mbps": sync_mbps,
        "grad_bytes": grad_bytes,
        "optimizer_step_sec": optimizer_step_seconds,
        "gradient_norm_preclip": gradient_norm_preclip,
        "learning_rate": effective_learning_rate,
    }
    # Collection timing is produced before this synchronized learner update.
    # Carry it explicitly so a logger dump/reset cannot hide the breakdown
    # from the distributed record or W&B.
    for name in (
        "root_policy_seconds",
        "combat_observation_rows_omitted",
        "cpp_begin_rpc_seconds",
        "cpp_leaf_decode_seconds",
        "cpp_leaf_model_seconds",
        "cpp_backprop_rpc_seconds",
        "cpp_stream_wait_seconds",
        "env_step_seconds",
        "selection_prefix_decisions",
        "simulator_transitions",
    ) + WORKER_TIMING_KEYS + NATIVE_TIMING_KEYS:
        result[name] = float(rollout_timings.get(name, 0.0))
    result.update(diagnostic_values)
    # Aggregate the objective scalars with the same root weighting used by the
    # diagnostic values.  The gradient itself is already weighted before the
    # parameter ring reduction.
    local_policy_loss_sum = float(
        getattr(model, "_nosl_is_az_policy_loss_sum", 0.0)
    )
    local_value_loss_sum = float(
        getattr(model, "_nosl_is_az_value_loss_sum", 0.0)
    )
    local_policy_weight_sum = float(
        getattr(
            model,
            "_nosl_is_az_policy_weight_sum",
            az_stats.get("metric_count", diagnostic_count("value_target")),
        )
    )
    loss_payload = average_scalar_dict(
        {
            "policy_sum": local_policy_loss_sum,
            "value_sum": local_value_loss_sum,
            "policy_count": local_policy_weight_sum,
            "value_count": float(az_stats.get(
                "metric_count", diagnostic_count("value_target")
            )),
        },
        ring_transport,
    )
    policy_loss_count = float(loss_payload["policy_count"] * scalar_world)
    value_loss_count = float(loss_payload["value_count"] * scalar_world)
    if policy_loss_count > 0.0:
        result["policy_loss"] = float(
            loss_payload["policy_sum"] * scalar_world / policy_loss_count
        )
    if value_loss_count > 0.0:
        result["value_loss"] = float(
            loss_payload["value_sum"] * scalar_world / value_loss_count
        )
    result.update({
        key: float(az_stats[key])
        for key in (
            "fresh_roots",
            "dropped_unlabeled_roots",
            "gradient_batches",
            "batch_size_requested",
            "batch_size_used",
            "batch_fill_ratio",
            "replay_allocated_mb",
            "replay_capacity",
            "replay_fill_ratio",
            "replay_old_fraction",
            "replay_current_fraction",
            "replay_max_old_fraction",
            "replay_ready",
            "replay_sampled_roots",
            "replay_size",
            "replay_enabled",
            "rollout_value_gt_current_hp_ge_future_hp_fraction",
            "rollout_value_gt_current_hp_ge_future_hp_count",
            "rollout_value_gt_current_hp_ge_future_hp_matches",
        )
        if key in az_stats
    })
    return result


# These keys are intentionally fixed.  A ring all-reduce is a collective
# operation, so every rank must contribute the same vector even when one rank
# has not completed an episode during the current rollout.
_DISTRIBUTED_GAME_METRIC_KEYS = (
    "game/win_rate",
    "game/floor_mean",
    "game/deck_size_mean",
    "game/act_1_won_mean",
    "game/act_2_won_mean",
    "game/act_3_won_mean",
    *(f"game/floor_won/{floor:02d}" for floor in range(5, 55, 5)),
)

_DISTRIBUTED_GAME_COUNT_KEYS = (
    "game/episodes_window",
    "game/episodes_total",
    "game/startup_episodes_excluded_total",
    "game/chest_opened_window",
    "game/chest_opportunities_window",
    "game/chest_relic_taken_window",
    "game/chest_relic_opportunities_window",
    'boss/censored_episodes_total',
    *(f'boss/{boss}/{metric}' for boss in ('CHAMP', 'AUTOMATON', 'COLLECTOR')
      for metric in ('episodes_window', 'wins_window')),
)


def _prepare_distributed_game_scalars(
    callback_scalars: dict[str, float],
) -> dict[str, float]:
    """Weight rolling game means by the number of represented episodes."""
    game_window = float(callback_scalars.get("game/episodes_window", 0.0))
    values = {
        key: float(callback_scalars.get(key, 0.0)) * game_window
        for key in _DISTRIBUTED_GAME_METRIC_KEYS
    }
    values.update({
        key: float(callback_scalars.get(key, 0.0))
        for key in _DISTRIBUTED_GAME_COUNT_KEYS
    })
    return values


def _finalize_distributed_game_scalars(
    aggregate_scalars: dict[str, float], *, world_size: int,
) -> None:
    """Convert rank-mean weighted sums and counts into global statistics."""
    game_window_mean = aggregate_scalars["game/episodes_window"]
    for key in _DISTRIBUTED_GAME_METRIC_KEYS:
        aggregate_scalars[key] = (
            aggregate_scalars[key] / game_window_mean
            if game_window_mean > 0.0 else 0.0
        )
    for key in _DISTRIBUTED_GAME_COUNT_KEYS:
        aggregate_scalars[key] *= world_size
    for rate, numerator, denominator in _CHEST_RATE_COUNTS:
        count = aggregate_scalars[denominator]
        if count:
            aggregate_scalars[rate] = aggregate_scalars[numerator] / count


_CHEST_RATE_COUNTS = (
    ("game/chest_open_rate", "game/chest_opened_window", "game/chest_opportunities_window"),
    ("game/chest_relic_take_rate", "game/chest_relic_taken_window", "game/chest_relic_opportunities_window"),
)


def _add_game_metrics_to_wandb_payload(
    payload: dict[str, object], record: dict[str, object],
) -> None:
    payload.update({key: record[key] for key in _DISTRIBUTED_GAME_COUNT_KEYS
                    if key in record and (not key.startswith('boss/') or
                                          record.get('episode_scope') == 'boss-opening')})
    for rate, numerator, denominator in _CHEST_RATE_COUNTS:
        if record.get(denominator, 0):
            payload[rate] = record[numerator] / record[denominator]
    if float(record["game/episodes_window"]) <= 0.0:
        return
    payload.update({key: record[key] for key in _DISTRIBUTED_GAME_METRIC_KEYS})
    if record.get('episode_scope') == 'boss-opening':
        payload['boss/win_rate'] = record['game/win_rate']
        for key in list(payload):
            if key.startswith('game/act_') or key.startswith('game/floor_'):
                del payload[key]
        for boss in ('CHAMP', 'AUTOMATON', 'COLLECTOR'):
            count = record[f'boss/{boss}/episodes_window']
            if count:
                payload[f'boss/{boss}/win_rate'] = record[f'boss/{boss}/wins_window'] / count
        return
    for floor in range(5, 55, 5):
        payload[f"game/layer_{floor}_win_rate"] = record[
            f"game/floor_won/{floor:02d}"
        ]
    for layer in (1, 2, 3):
        value = payload.get(f"game/act_{layer}_won_mean")
        if value is not None:
            payload[f"game/layer_{layer}_win_rate"] = value


# NOSL metrics are emitted by the local collector/trainer.  Keep a fixed key
# set so every rank participates in the same scalar all-reduce even when a
# rank has no completed combat in a particular rollout.
_DISTRIBUTED_NOSL_METRIC_KEYS = (
    "nosl_is_az/schedule_round",
    "nosl_is_az/scheduled_updates",
    "nosl_is_az/scheduled_value_only_updates",
    "nosl_is_az/value_only_updates",
    "nosl_is_az/joint_updates",
    "nosl_is_az/overworld_rule_decisions",
    "nosl_is_az/overworld_fallback_decisions",
    "nosl_is_az/forced_end_turn_roots",
    "nosl_is_az/combat_observation_rows_omitted",
    "nosl_is_az/rulebase_lower_roots",
    "nosl_is_az/rulebase_lower_decisions",
    "nosl_is_az/rulebase_lower_expanded_nodes",
    "nosl_is_az/rulebase_lower_search_seconds",
    "nosl_is_az/rulebase_lower_batch_seconds",
    "nosl_is_az/root_value_predictions_skipped",
    "nosl_is_az/combat_roots",
    "nosl_is_az/combat_policy_roots",
    "nosl_is_az/value_only_samples",
    "nosl_is_az/completed_combat_roots",
    "nosl_is_az/completed_combats",
    "nosl_is_az/incomplete_combat_roots",
    "nosl_is_az/pending_combat_roots",
    "nosl_is_az/collected_value_boundaries",
    "nosl_is_az/collected_combat_terminals",
    "nosl_is_az/cpp_leaf_batches",
    "nosl_is_az/cpp_leaf_evaluations",
    "nosl_is_az/root_policy_seconds",
    "nosl_is_az/cpp_begin_rpc_seconds",
    "nosl_is_az/cpp_leaf_decode_seconds",
    "nosl_is_az/cpp_leaf_model_seconds",
    *(f"nosl_is_az/{name}" for name in ROLLOUT_INFERENCE_METRICS),
    "nosl_is_az/cpp_backprop_rpc_seconds",
    "nosl_is_az/cpp_stream_wait_seconds",
    "nosl_is_az/cpp_ready_empty_wait_seconds",
    *(f"nosl_is_az/{name}" for name in WORKER_TIMING_KEYS),
    *(f"nosl_is_az/{name}" for name in NATIVE_TIMING_KEYS),
    "nosl_is_az/inference_batch_size_mean",
    "nosl_is_az/inference_batch_size_max",
    "nosl_is_az/inference_batch_target_mean",
    "nosl_is_az/inference_batch_target_current",
    "nosl_is_az/inference_ready_backlog_mean",
    "nosl_is_az/inference_ready_backlog_max",
    "nosl_is_az/inference_leaf_queue_wait_ms_mean",
    "nosl_is_az/inference_leaf_worker_wait_ms_mean",
    "nosl_is_az/inference_full_flushes",
    "nosl_is_az/inference_drained_flushes",
    "nosl_is_az/inference_watchdog_flushes",
    "nosl_is_az/inference_controller_changes",
    "nosl_is_az/inference_controller_increases",
    "nosl_is_az/inference_controller_decreases",
    "nosl_is_az/inference_controller_rollbacks",
    "nosl_is_az/inference_watchdog_ms",
    "nosl_is_az/env_step_seconds",
    "nosl_is_az/selection_prefix_decisions",
    "nosl_is_az/simulator_transitions",
    "nosl_is_az/collection_seconds",
    "nosl_is_az/leaf_graph_captures",
    "nosl_is_az/leaf_graph_hits",
    "nosl_is_az/leaf_graph_evictions",
    "nosl_is_az/leaf_graph_cached_graphs",
    "nosl_is_az/leaf_graph_capture_seconds",
    "nosl_is_az/leaf_graph_copy_seconds",
    "nosl_is_az/leaf_graph_output_seconds",
    "nosl_is_az/leaf_graph_profile_samples",
    "nosl_is_az/leaf_graph_h2d_ms",
    "nosl_is_az/leaf_graph_replay_ms",
    "nosl_is_az/leaf_graph_d2h_ms",
    "nosl_is_az/dropped_unlabeled_roots",
    "nosl_is_az/roots",
    "nosl_is_az/policy_loss",
    "nosl_is_az/value_loss",
    "nosl_is_az/updates",
    "nosl_is_az/fresh_roots",
    "nosl_is_az/batch_size_requested",
    "nosl_is_az/batch_size_used",
    "nosl_is_az/batch_fill_ratio",
    "nosl_is_az/gradient_batches",
    "nosl_is_az/replay_allocated_mb",
    "nosl_is_az/replay_capacity",
    "nosl_is_az/replay_fill_ratio",
    "nosl_is_az/replay_old_fraction",
    "nosl_is_az/replay_current_fraction",
    "nosl_is_az/replay_max_old_fraction",
    "nosl_is_az/replay_ready",
    "nosl_is_az/replay_sampled_roots",
    "nosl_is_az/replay_size",
    "nosl_is_az/replay_enabled",
    "nosl_is_az/metric_count",
    "nosl_is_az/target_entropy",
    "nosl_is_az/policy_entropy",
    "nosl_is_az/target_policy_kl",
    "nosl_is_az/policy_cross_entropy",
    "nosl_is_az/target_effective_actions",
    "nosl_is_az/target_top1_agreement",
    "nosl_is_az/legal_actions_mean",
    "nosl_is_az/illegal_target_mass",
    "nosl_is_az/target_sum_error",
    "nosl_is_az/value_target_mean",
    "nosl_is_az/value_target_std",
    "nosl_is_az/value_target_min",
    "nosl_is_az/value_target_max",
    "nosl_is_az/pred_value_mean",
    "nosl_is_az/pred_value_std",
    "nosl_is_az/value_mae",
    "nosl_is_az/value_bias",
    "nosl_is_az/finite_fraction",
    "nosl_is_az/rollout_value_gt_current_hp_ge_future_hp_fraction",
    "nosl_is_az/rollout_value_gt_current_hp_ge_future_hp_count",
    "nosl_is_az/rollout_value_gt_current_hp_ge_future_hp_matches",
    "nosl_is_az/gradient_norm_preclip",
    "nosl_is_az/learning_rate",
    "nosl_is_az/win_bce",
    "nosl_is_az/hp_mse",
    "nosl_is_az/hp_pinball",
    "nosl_is_az/utility_mse",
    "nosl_is_az/transition_loss",
    "nosl_is_az/win_probability",
    "nosl_is_az/win_target",
    "nosl_is_az/hp_given_win",
)


def apply_unified_upper_update_overrides(model, args):
    """Apply explicit unified settings after loading Adam, without resetting moments."""
    from dataclasses import asdict, replace
    changes = {name: getattr(args, 'upper_' + name) for name in
               ('learning_rate', 'value_learning_rate', 'effective_batch_size', 'critic_effective_batch_size', 'critic_epochs')
               if getattr(args, 'upper_' + name, None) is not None}
    if any(not np.isfinite(v) or v <= 0 for v in changes.values()):
        raise ValueError('upper optimization overrides must be finite and positive')
    if getattr(args, 'upper_measure_value_loss', None) is not None:
        changes['measure_value_loss'] = args.upper_measure_value_loss
    if not changes:
        return
    core = getattr(model.policy, 'core', None)
    cfg = getattr(core, 'upper_config', None)
    if cfg is None or cfg.architecture != 'upper-latent-actions-v1':
        raise ValueError('upper optimization overrides require unified upper')
    updated = replace(cfg, **changes)
    groups = model.policy.upper_optimizer.param_groups
    networks = (core.upper_actor, core.upper_critic)
    if len(groups) != 2 or any(
        {id(p) for p in group['params']} != {id(p) for p in network.parameters()}
        for group, network in zip(groups, networks)
    ):
        raise ValueError('unexpected unified upper Adam parameter groups')
    # load_state_dict restores old group learning rates as well as moments.
    # Set only explicitly requested rates after that restoration.
    if 'learning_rate' in changes:
        groups[0]['lr'] = updated.learning_rate
    if 'value_learning_rate' in changes or ('learning_rate' in changes and not cfg.value_learning_rate):
        groups[1]['lr'] = updated.value_learning_rate or updated.learning_rate
    core.upper_config = updated
    model.policy.upper_config = asdict(updated)
    model.policy_kwargs['upper_config'] = asdict(updated)
    args.upper_config = asdict(updated)


def configure_training_scope(model, args):
    """Persist the learner boundary without changing the collection/search policy."""
    saved = getattr(model, 'training_scope', 'all')
    requested = args.training_scope or saved
    if saved not in ('all', 'lower') or requested not in ('all', 'lower'):
        raise ValueError('unknown checkpoint training scope')
    if args.resume_state is not None and requested != saved:
        raise ValueError('changing training scope requires --fork-state')
    if requested == 'lower' and not (
        joint_upper(model) and hasattr(model.policy, 'small_playout')
        and model.nosl_is_az and model.nosl_is_az_inner_only
        and model.nosl_is_az_combat_strategy == 'small-playout'
        and not args.nosl_is_az_replay_enabled and args.vec == 'native'
        and args.reasoning_updates == 0 and args.reasoning_coef == 0
        and not args.interactive_reasoning
        and getattr(model.policy, 'upper_auxiliary_config', None) is None
        and not getattr(model, '_frozen_upper', False)
    ):
        raise ValueError('lower training scope requires native online small-playout, joint upper, '
                         'inner-only NOSL, and no replay, reasoning or upper auxiliary training')
    model.training_scope = args.training_scope = requested


def main() -> None:
    args = build_parser().parse_args()
    if not np.isfinite(args.iteration_timeout_seconds) or args.iteration_timeout_seconds < 0:
        raise ValueError('iteration timeout must be finite and nonnegative')
    if args.leaf_amp_mode is not None and args.leaf_inference != 'cuda-graph':
        raise ValueError('--leaf-amp-mode requires --leaf-inference cuda-graph')
    if args.prepare_combat_outer is not None:
        from rl.training.combat_outer import prepare
        return prepare(args)
    if args.checkpoint is None and args.resume_state is None and args.fork_state is None:
        raise ValueError('supply a checkpoint or explicitly prepare new combat material')
    if args.upper_micro_batch_size is not None and args.upper_micro_batch_size < 1:
        raise ValueError('--upper-micro-batch-size must be positive')
    if args.upper_epochs is not None and args.upper_epochs < 1:
        raise ValueError('--upper-epochs must be positive')
    if args.upper_gae_lambda is not None and not 0.0 <= args.upper_gae_lambda <= 1.0:
        raise ValueError('--upper-gae-lambda must be finite and between 0 and 1')
    if args.upper_target_kl is not None and (not np.isfinite(args.upper_target_kl) or args.upper_target_kl < 0):
        raise ValueError('--upper-target-kl must be finite and nonnegative (0 disables early stopping)')
    for name in ('small_playout_batch_size', 'small_playout_micro_batch_size'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError('--' + name.replace('_', '-') + ' must be positive')
    if args.fork_public_lower and (args.fork_state is None or args.training_scope!='lower'):
        raise ValueError('--fork-public-lower requires --fork-state and --training-scope lower')
    if args.fork_public_weights and not args.fork_public_lower:
        raise ValueError('--fork-public-weights requires --fork-public-lower')
    if args.resume_state is not None and args.fork_state is not None:
        raise ValueError("--resume-state and --fork-state are mutually exclusive")
    if args.fork_reset_environments and args.fork_state is None:
        raise ValueError("--fork-reset-environments requires --fork-state")
    if args.resume_reset_environments and args.resume_state is None:
        raise ValueError("--resume-reset-environments requires --resume-state")
    if args.resume_grow_environments and (args.resume_state is None or args.resume_reset_environments
                                          or args.episode_rollout):
        raise ValueError('--resume-grow-environments requires ordinary PPO resume without reset')
    if args.fork_grow_environments and (
            args.fork_state is None or not args.episode_rollout
            or args.fork_reset_environments or args.resume_reset_environments):
        raise ValueError('--fork-grow-environments requires --fork-state/--episode-rollout without environment reset')
    recovery_state = args.resume_state or args.fork_state
    if recovery_state is not None:
        manifest = read_bundle(recovery_state, rank=args.rank, world_size=args.world_size)
    if args.resume_state is not None:
        if args.run_id is not None and args.run_id != manifest['run_id']:
            raise ValueError('resume run ID differs from recovery manifest')
        if args.wandb_id is not None and args.wandb_id != manifest['wandb_id']:
            raise ValueError('resume W&B ID differs from recovery manifest')
        args.run_id, args.wandb_id = manifest['run_id'], manifest['wandb_id']
        args.checkpoint = args.resume_state / 'model.zip'
    elif args.fork_state is not None:
        if not args.run_id or args.run_id == manifest['run_id']:
            raise ValueError("--fork-state requires a new explicit --run-id")
        if args.wandb and (
            not args.wandb_id or args.wandb_id == manifest['wandb_id']
        ):
            raise ValueError("--fork-state with W&B requires a new --wandb-id")
        args.fork_source_run_id = manifest['run_id']
        args.fork_source_wandb_id = manifest['wandb_id']
        args.fork_source_global_step = manifest['global_step']
        args.checkpoint = args.fork_state / 'model.zip'
    if args.continuous:
        if args.total_steps is not None or args.updates != 1:
            raise ValueError('--continuous cannot have --total-steps or a finite --updates target')
        if args.stop_file is None or args.training_state_dir is None:
            raise ValueError('--continuous requires --stop-file and --training-state-dir')
        if args.learning_rate is None or args.learning_rate <= 0:
            raise ValueError('--continuous requires an explicit positive fixed learning rate')
    if args.stop_file is not None and args.stop_file.exists():
        raise ValueError('remove the previous stop request before resuming')
    if args.training_state_dir is not None and (args.save_every_updates < 1 or args.max_checkpoints < 2):
        raise ValueError('learner recovery requires save-every-updates >= 1 and max-checkpoints >= 2')
    if args.checkpoint_keep_round_multiple < 0:
        raise ValueError('--checkpoint-keep-round-multiple must be nonnegative')
    if args.checkpoint_keep_round_multiple and (args.training_state_dir is None or args.save_every_updates != 1):
        raise ValueError('round-multiple retention requires recovery bundles saved every round')
    if args.training_state_dir is not None and args.wandb and not args.wandb_id:
        raise ValueError('learner recovery with W&B requires an explicit --wandb-id')
    if args.stop_file is not None and args.world_size > 1 and args.gradient_transport != 'ssh-ring':
        raise ValueError('coordinated stop requires ssh-ring for multiple ranks')
    args.checkpoint = normalize_checkpoint_path(args.checkpoint)
    th.set_float32_matmul_precision(args.float32_matmul_precision)
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank must be in [0, world_size)")
    if args.world_size == 1 and args.peer:
        raise ValueError("--peer is meaningless for world_size=1")
    if args.gradient_transport == "ssh-ring" and args.world_size < 2:
        raise ValueError("--gradient-transport ssh-ring requires world_size >= 2")
    if not 1 <= args.ring_port <= 65536 - args.ring_streams:
        raise ValueError("all --ring-port listener ports must be in [1, 65535]")
    if args.ring_streams != 1 and args.gradient_transport != "ssh-ring":
        raise ValueError("--ring-streams requires --gradient-transport ssh-ring")
    if args.reasoning_gradient_accumulation_steps < 1:
        raise ValueError(
            "--reasoning-gradient-accumulation-steps must be at least 1"
        )
    if (
        args.max_grad_norm is not None
        and (
            np.isnan(args.max_grad_norm)
            or args.max_grad_norm < 0.0
        )
    ):
        raise ValueError("--max-grad-norm must be non-negative or inf")
    if args.envs_per_worker < 1:
        raise ValueError("--envs-per-worker must be at least 1")
    for name in ("nosl_is_az_policy_coef", "nosl_is_az_value_coef"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.startup_stagger_steps < 0:
        raise ValueError("--startup-stagger-steps must be non-negative")
    if args.startup_stagger_steps % args.n_steps:
        raise ValueError("--startup-stagger-steps must be a multiple of --n-steps")
    if args.startup_occupancy_budget < 0 or args.startup_occupancy_budget % args.n_steps:
        raise ValueError("--startup-occupancy-budget must be a non-negative multiple of --n-steps")
    if args.startup_occupancy_max_age < 1:
        raise ValueError("--startup-occupancy-max-age must be positive")
    if args.startup_occupancy_budget and args.startup_stagger_steps:
        raise ValueError("occupancy sampling and startup stagger are mutually exclusive")
    if args.nosl_is_az_simulations is not None and args.nosl_is_az_simulations < 1:
        raise ValueError("--nosl-is-az-simulations must be positive")
    if args.nosl_is_az_depth is not None and args.nosl_is_az_depth < 0:
        raise ValueError("--nosl-is-az-depth must be non-negative")
    if (
        args.nosl_is_az_prior_temperature is not None
        and (
            not np.isfinite(args.nosl_is_az_prior_temperature)
            or args.nosl_is_az_prior_temperature <= 0.0
        )
    ):
        raise ValueError(
            "--nosl-is-az-prior-temperature must be positive and finite"
        )
    if (
        args.nosl_is_az_gumbel_candidates is not None
        and args.nosl_is_az_gumbel_candidates < 0
    ):
        raise ValueError("--nosl-is-az-gumbel-candidates must be non-negative")
    for name in (
        "nosl_is_az_gumbel_scale",
        "nosl_is_az_gumbel_value_scale",
        "nosl_is_az_gumbel_max_visit_init",
    ):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(
                f"--{name.replace('_', '-')} must be non-negative and finite"
            )
    if (
        args.nosl_is_az_gumbel_interior_uniformity is not None
        and not 0.0 <= args.nosl_is_az_gumbel_interior_uniformity <= 1.0
    ):
        raise ValueError(
            "--nosl-is-az-gumbel-interior-uniformity must be in [0, 1]"
        )
    if (
        args.nosl_is_az_gumbel_interior_coverage_fraction is not None
        and not 0.0 <= args.nosl_is_az_gumbel_interior_coverage_fraction <= 1.0
    ):
        raise ValueError(
            "--nosl-is-az-gumbel-interior-coverage-fraction must be in [0, 1]"
        )
    if (
        args.nosl_is_az_gumbel_interior_max_mix_visits is not None
        and (
            not np.isfinite(args.nosl_is_az_gumbel_interior_max_mix_visits)
            or args.nosl_is_az_gumbel_interior_max_mix_visits <= 0.0
        )
    ):
        raise ValueError(
            "--nosl-is-az-gumbel-interior-max-mix-visits must be positive and finite"
        )
    if args.nosl_is_az_batch_size is not None and args.nosl_is_az_batch_size < 1:
        raise ValueError("--nosl-is-az-batch-size must be positive")
    if args.nosl_is_az_updates is not None and args.nosl_is_az_updates < 1:
        raise ValueError("--nosl-is-az-updates must be positive")
    if (
        args.nosl_is_az_micro_batch_size is not None
        and args.nosl_is_az_micro_batch_size < 1
    ):
        raise ValueError("--nosl-is-az-micro-batch-size must be positive")
    if (
        args.nosl_is_az_replay_capacity is not None
        and args.nosl_is_az_replay_capacity < 1
    ):
        raise ValueError("--nosl-is-az-replay-capacity must be positive")
    if (
        args.nosl_is_az_replay_min_size is not None
        and args.nosl_is_az_replay_min_size < 1
    ):
        raise ValueError("--nosl-is-az-replay-min-size must be positive")
    if (
        args.nosl_is_az_replay_max_old_fraction is not None
        and (
            not np.isfinite(args.nosl_is_az_replay_max_old_fraction)
            or not 0.0 <= args.nosl_is_az_replay_max_old_fraction <= 1.0
        )
    ):
        raise ValueError(
            "--nosl-is-az-replay-max-old-fraction must be in [0, 1]"
        )
    if (
        args.nosl_is_az_leaf_batch_size is not None
        and args.nosl_is_az_leaf_batch_size < 1
    ):
        raise ValueError("--nosl-is-az-leaf-batch-size must be positive")
    if (
        args.nosl_is_az_inference_batch_size is not None
        and args.nosl_is_az_inference_batch_size < 1
    ):
        raise ValueError("--nosl-is-az-inference-batch-size must be positive")
    if (
        args.nosl_is_az_inference_max_wait_ms is not None
        and (
            not np.isfinite(args.nosl_is_az_inference_max_wait_ms)
            or args.nosl_is_az_inference_max_wait_ms < 0.0
        )
    ):
        raise ValueError(
            "--nosl-is-az-inference-max-wait-ms must be finite and non-negative"
        )
    if args.nosl_is_az_dirichlet_alpha < 0.0:
        raise ValueError("--nosl-is-az-dirichlet-alpha must be non-negative")
    if not 0.0 <= args.nosl_is_az_dirichlet_epsilon <= 1.0:
        raise ValueError("--nosl-is-az-dirichlet-epsilon must be in [0, 1]")
    if args.nosl_is_az_dirichlet_epsilon > 0.0 and args.nosl_is_az_dirichlet_alpha <= 0.0:
        raise ValueError("Dirichlet alpha must be positive when epsilon > 0")

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
    control_dir = ssh_control_directory(Path(args.ssh_control_dir), run_id, args.rank)
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
    pure_ppo = probe_model.nosl_supervision_contract == 'shared-wholerun-ppo-room-v1'
    if args.resume_grow_environments and not pure_ppo:
        raise ValueError('--resume-grow-environments only supports shared whole-run PPO')
    source_ppo_round = (getattr(probe_model, 'ppo_rollouts_completed',
        args.wandb_step_base if args.wandb_step_base is not None else
        probe_model.num_timesteps // (probe_model.n_envs * probe_model.n_steps * args.world_size))
        if pure_ppo else None)
    if args.upper_auxiliary_search is not None and not joint_upper(probe_model):
        raise ValueError('--upper-auxiliary-search requires joint strategic PPO')
    if args.upper_micro_batch_size is not None and not joint_upper(probe_model):
        raise ValueError('--upper-micro-batch-size requires joint strategic PPO')
    if args.upper_epochs is not None and not joint_upper(probe_model):
        raise ValueError('--upper-epochs requires joint strategic PPO')
    if args.upper_critic_target is not None and not joint_upper(probe_model):
        raise ValueError('--upper-critic-target requires unified upper')
    if args.upper_gae_lambda is not None and not joint_upper(probe_model):
        raise ValueError('--upper-gae-lambda requires joint strategic PPO')
    if args.upper_target_kl is not None and not joint_upper(probe_model):
        raise ValueError('--upper-target-kl requires joint strategic PPO')
    small_overrides = {key: getattr(args, 'small_playout_' + key)
                       for key in ('batch_size', 'micro_batch_size')
                       if getattr(args, 'small_playout_' + key) is not None}
    if small_overrides and not hasattr(probe_model.policy, 'small_playout'):
        raise ValueError('small playout batch overrides require an online compact checkpoint')
    args.policy_architecture = type(probe_model.policy).__name__
    args.combat_supervision_contract = getattr(probe_model,'nosl_supervision_contract','scalar-hp-v1')
    if getattr(getattr(probe_model.policy,'core',None),'combat_only',False) or joint_upper(probe_model) or getattr(probe_model, '_frozen_upper', False):
        args.entity_config = dict(probe_model.policy.entity_config)
        if joint_upper(probe_model):
            args.upper_config = dict(probe_model.policy.upper_config)
            args.upper_auxiliary_config = getattr(probe_model.policy, 'upper_auxiliary_config', None)
            if args.upper_auxiliary_search == 'off':
                args.upper_auxiliary_config = None
            elif args.upper_auxiliary_search == 'public-v1':
                from dataclasses import asdict
                from rl.algorithms.upper_auxiliary import UpperAuxiliaryConfig
                args.upper_auxiliary_config = asdict(UpperAuxiliaryConfig())
            if args.upper_auxiliary_config is not None and not probe_model.nosl_is_az:
                raise ValueError('public upper search currently requires the joint NOSL collector')
            args.small_playout_config = getattr(probe_model.policy, 'small_playout_config', None)
            if small_overrides:
                from dataclasses import asdict
                from rl.policies.small_playout import OnlinePlayoutConfig
                args.small_playout_config = asdict(OnlinePlayoutConfig(
                    **(dict(args.small_playout_config) | small_overrides)))
            if args.upper_micro_batch_size is not None:
                args.upper_config['micro_batch_size'] = args.upper_micro_batch_size
            if args.upper_epochs is not None:
                args.upper_config['epochs'] = args.upper_epochs
            if args.upper_critic_target is not None:
                if args.upper_config.get('architecture', 'legacy') != 'upper-latent-actions-v1':
                    raise ValueError('--upper-critic-target requires unified upper')
                args.upper_config['critic_target'] = args.upper_critic_target
            if args.upper_gae_lambda is not None:
                args.upper_config['gae_lambda'] = args.upper_gae_lambda
            if args.upper_target_kl is not None:
                args.upper_config['target_kl'] = args.upper_target_kl or None
            if args.world_size > 1 and args.gradient_transport != 'ssh-ring':
                raise ValueError('joint strategic PPO requires ring gradient synchronization')
            if args.boss_opening_dataset:
                raise ValueError('joint training requires whole-run environments')
        if args.disable_entity_transition_auxiliary:
            args.entity_config['auxiliary_loss_coef'] = 0.0
    elif args.disable_entity_transition_auxiliary:
        raise ValueError('--disable-entity-transition-auxiliary requires EntityCombatPolicy')
    source_nosl_is_az = bool(getattr(probe_model, "nosl_is_az", False))
    if args.boss_opening_dataset and (not source_nosl_is_az or
            args.combat_supervision_contract != 'combat-outcome-quantiles-v1'):
        raise ValueError('Boss specialization requires entity NOSL outcome supervision')
    source_nosl_is_az_replay_enabled = bool(
        getattr(probe_model, "nosl_is_az_replay_enabled", True)
    )
    source_nosl_inner_only = bool(
        getattr(probe_model, "nosl_is_az_inner_only", False)
    )
    if source_nosl_is_az and not source_nosl_inner_only:
        raise RuntimeError(
            "Distributed NOSL-IS-AZ currently supports inner-only combat "
            "training; enable --nosl-is-az-inner-only in the checkpoint."
        )
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
    if args.nosl_is_az_replay_enabled is None:
        args.nosl_is_az_replay_enabled = source_nosl_is_az_replay_enabled
    if args.nosl_is_az_boss_transition_target is None:
        args.nosl_is_az_boss_transition_target = bool(
            getattr(probe_model, "nosl_is_az_boss_transition_target", False)
        )
    if not 0.0 <= args.combat_hp_gamma <= 1.0:
        raise ValueError("--combat-hp-gamma must be in [0, 1]")
    if not 0.0 <= args.combat_hp_gae_lambda <= 1.0:
        raise ValueError("--combat-hp-gae-lambda must be in [0, 1]")
    # Preserve the checkpoint's policy geometry while allowing a fresh/random
    # split-pointer checkpoint to replace its card table with the frozen v7
    # anonymous rule encoder.  This is intentionally a policy-construction
    # option, not a trainable warm-start.
    policy_kwargs = dict(getattr(probe_model, "policy_kwargs", {}) or {})
    if not policy_kwargs:
        checkpoint_core_cfg = getattr(
            getattr(probe_model, "policy", None), "_core_cfg", None
        )
        if isinstance(checkpoint_core_cfg, dict):
            policy_kwargs = dict(checkpoint_core_cfg)
    if args.card_vector_checkpoint is not None:
        if isinstance(policy_kwargs.get("entity_config"), dict):
            policy_kwargs["entity_config"] = dict(policy_kwargs["entity_config"])
            policy_kwargs["entity_config"]["card_vector_checkpoint"] = str(
                args.card_vector_checkpoint
            )
        else:
            policy_kwargs["card_vector_checkpoint"] = str(
                args.card_vector_checkpoint
            )
    if joint_upper(probe_model):
        policy_kwargs['upper_config'] = dict(args.upper_config)
        policy_kwargs['upper_auxiliary_config'] = args.upper_auxiliary_config
        if args.small_playout_config is not None:
            policy_kwargs['small_playout_config'] = dict(args.small_playout_config)
    env = make_env(probe_model, args)
    del probe_model
    custom_objects = {
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "policy_kwargs": policy_kwargs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "hierarchical_combat_hp": args.hierarchical_combat_hp,
        "nosl_is_az_boss_transition_target": (
            args.nosl_is_az_boss_transition_target
        ),
        "combat_hp_gamma": args.combat_hp_gamma,
        "combat_hp_gae_lambda": args.combat_hp_gae_lambda,
        "nosl_is_az_replay_enabled": args.nosl_is_az_replay_enabled,
        "reasoning_coef": args.reasoning_coef,
        "reasoning_batch_size": args.reasoning_batch_size,
        "reasoning_updates": args.reasoning_updates,
        "reasoning_gradient_accumulation_steps": (
            args.reasoning_gradient_accumulation_steps
        ),
    }
    if args.max_grad_norm is not None:
        custom_objects["max_grad_norm"] = args.max_grad_norm
    for name in ("nosl_is_az_policy_coef", "nosl_is_az_value_coef"):
        if getattr(args, name) is not None:
            custom_objects[name] = getattr(args, name)
    model = AccumulatingMaskablePPO.load(
        args.checkpoint,
        env=env,
        device=args.device,
        amp_mode=args.amp_mode,
        custom_objects=custom_objects,
    )
    apply_unified_upper_update_overrides(model, args)
    if args.max_grad_norm is not None:
        print(
            f"[rank {args.rank}] max gradient norm override: "
            f"{model.max_grad_norm} "
            f"({'disabled' if np.isinf(model.max_grad_norm) else 'enabled'})",
            flush=True,
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
    if args.disable_entity_transition_auxiliary:
        from rl.policies.entity_combat import disable_entity_transition_auxiliary
        disable_entity_transition_auxiliary(model)
        print(f"[rank {args.rank}] entity action-effect auxiliary disabled", flush=True)
    if source_nosl_inner_only:
        if args.nosl_is_az_train_shared_embeddings and recovery_state is None:
            embedding_count = model.reset_nosl_is_az_shared_embeddings(args.seed)
            print(
                f"[rank {args.rank}] reset {embedding_count} shared semantic "
                "embedding tables with the run seed",
                flush=True,
            )
        frozen, trainable = model.configure_nosl_is_az_inner_only(
            train_shared_embeddings=args.nosl_is_az_train_shared_embeddings
        )
        print(
            f"[rank {args.rank}] NOSL-IS-AZ inner-only: froze "
            f"{frozen:,} parameters; {trainable:,} parameters trainable",
            flush=True,
        )
    upper_strategy = args.nosl_is_az_upper_strategy or getattr(
        model, "nosl_is_az_upper_strategy", "model"
    )
    if upper_strategy not in ("model", "rules"):
        raise ValueError("NOSL-IS-AZ upper strategy must be model or rules")
    if getattr(getattr(model.policy,'core',None),'combat_only',False) and upper_strategy != 'rules':
        raise ValueError('combat-only model requires rules upper strategy')
    if joint_upper(model) and upper_strategy != 'model':
        raise ValueError('joint strategic PPO requires sampled model upper actions')
    if upper_strategy == "rules" and not (
        model.nosl_is_az and model.nosl_is_az_inner_only
    ):
        raise ValueError("--nosl-is-az-upper-strategy rules requires inner-only NOSL")
    model.nosl_is_az_upper_strategy = args.nosl_is_az_upper_strategy = upper_strategy
    for name in ("nosl_is_az_policy_coef", "nosl_is_az_value_coef"):
        if getattr(args, name) is not None and not model.nosl_is_az:
            raise ValueError(f"--{name.replace('_', '-')} requires a NOSL-IS-AZ checkpoint")
        setattr(args, name, float(getattr(model, name)))
    from env.search.overworld_rules import OVERWORLD_RULE_VERSION
    args.nosl_is_az_upper_strategy_version = (
        OVERWORLD_RULE_VERSION if upper_strategy == "rules" else "checkpoint-model"
    )
    fallback_label = ('frozen-masked-argmax' if getattr(model, '_frozen_upper', False)
                      else 'disabled' if getattr(getattr(model.policy,'core',None),'combat_only',False)
                      else 'masked-argmax' if upper_strategy == 'rules' else 'sampled')
    print(f"[rank {args.rank}] NOSL upper strategy: {upper_strategy} "
          f"version={args.nosl_is_az_upper_strategy_version} "
          f"fallback={fallback_label}", flush=True)
    combat_strategy = args.nosl_is_az_combat_strategy or getattr(
        model, "nosl_is_az_combat_strategy", "model"
    )
    if combat_strategy not in ("model", "turn-planner", "small-playout"):
        raise ValueError("NOSL-IS-AZ combat strategy must be model or turn-planner")
    if combat_strategy in ("turn-planner", "small-playout"):
        if not joint_upper(model):
            raise ValueError("turn-planner combat requires joint strategic PPO")
        if not (model.nosl_is_az and model.nosl_is_az_inner_only):
            raise ValueError("turn-planner combat requires inner-only NOSL collection")
        if args.nosl_is_az_replay_enabled:
            raise ValueError("turn-planner combat requires --no-nosl-is-az-replay")
    model.nosl_is_az_combat_strategy = args.nosl_is_az_combat_strategy = combat_strategy
    if args.completed_game_budget is not None and (args.completed_game_budget < 1 or combat_strategy != 'small-playout'
                                                  or args.training_state_dir is None):
        raise ValueError('completed-game-budget needs small-playout and full recovery saving')
    planner_defaults = {
        "worlds": 2,
        "nodes": 8192,
        "beam": 8,
        "depth": 16,
        "turns": 1,
        "leaf_profile": "legacy-v8",
        "nonce": 20260918,
    }
    for suffix, default in planner_defaults.items():
        arg_name = f"rulebase_planner_{suffix}"
        value = getattr(args, arg_name)
        if value is None:
            value = getattr(model, arg_name, default)
        setattr(args, arg_name, value)
        setattr(model, arg_name, value)
    from env.search.turn_planner import TurnPlannerConfig
    from env.search.turn_policy import rulebase_turn_policy_version
    planner_config = TurnPlannerConfig(
        worlds=int(args.rulebase_planner_worlds),
        node_budget=int(args.rulebase_planner_nodes),
        beam_width=int(args.rulebase_planner_beam),
        max_depth=int(args.rulebase_planner_depth),
        lookahead_turns=int(args.rulebase_planner_turns),
        leaf_profile=str(args.rulebase_planner_leaf_profile),
        master_nonce=int(args.rulebase_planner_nonce),
    )
    args.nosl_is_az_combat_strategy_version = (
        rulebase_turn_policy_version(planner_config)
        if combat_strategy == "turn-planner"
        else 'sts1-small-playout-pair-v1' if combat_strategy == 'small-playout'
        else "checkpoint-model+configured-search"
    )
    if combat_strategy in ("turn-planner", "small-playout"):
        frozen, upper_trainable = model.configure_rulebase_lower()
        print(
            f"[rank {args.rank}] fixed combat strategy: "
            f"{args.nosl_is_az_combat_strategy_version} "
            f"config={planner_config} froze={frozen:,} upper_trainable={upper_trainable:,}",
            flush=True,
        )
        if combat_strategy == 'small-playout':
            from rl.training.small_playout import configure_collection
            configure_collection(model, model.get_env())
            args.small_playout_search_contract = model.small_playout_search_contract
            if getattr(model.policy, 'small_playout_config', None) and model.policy.small_playout_config.get('context_mode') in ('token-root-64x2', 'token-budget-64x2'):
                args.nosl_is_az_combat_strategy_version = args.small_playout_search_contract
                if model.policy.small_playout_config['context_mode'] == 'token-budget-64x2':
                    from rl.policies.search_budget import budget_manifest
                    args.search_budget_config = model.search_budget_config
                    print(f'[rank {args.rank}] learned paired-chain budgets {budget_manifest()}', flush=True)
                else:
                    print(f'[rank {args.rank}] paired chains 250/250/1000, Transformer roots, learned CPU guide', flush=True)
            else:
                args.nosl_is_az_combat_strategy_version = ('sts1-small-playout-online-terminal-v2'
                    if hasattr(model.policy, 'small_playout') else 'sts1-small-playout-pair-v1')
                print(f'[rank {args.rank}] compact CPU search common/elite/boss=250/250/1000, '
                      'playout cap=200, tree cap=1024; legacy NOSL leaf settings are inactive', flush=True)
    else:
        print(f"[rank {args.rank}] combat strategy: model", flush=True)
    if args.nosl_is_az_updates is not None:
        if not bool(getattr(model, "nosl_is_az", False)):
            raise ValueError(
                "--nosl-is-az-updates requires a NOSL-IS-AZ checkpoint"
            )
        model.nosl_is_az_updates = int(args.nosl_is_az_updates)
        print(
            f"[rank {args.rank}] NOSL-IS-AZ optimizer updates per collection: "
            f"{model.nosl_is_az_updates}",
            flush=True,
        )
    elif bool(getattr(model, "nosl_is_az", False)):
        # Make the effective checkpoint value visible in W&B and diagnostics.
        args.nosl_is_az_updates = int(getattr(model, "nosl_is_az_updates", 1))
    from rl.algorithms.update_schedule import CombatUpdateSchedule
    update_schedule = CombatUpdateSchedule.from_config(
        args.nosl_is_az_update_schedule or getattr(model, 'nosl_is_az_update_schedule', None))
    update_schedule.validate_model(model)
    args.nosl_is_az_update_schedule = model.nosl_is_az_update_schedule = update_schedule.config()
    from rl.training import episode_rollout
    episode_config = (dict(games_per_rank=args.n_envs, seed=args.seed,
                          max_steps=args.episode_rollout_max_steps,
                          max_seconds=args.episode_rollout_max_seconds,
                          censored_games=args.episode_rollout_censored or
                            (getattr(model, 'episode_rollout_config', None) or {}).get('censored_games','error')) if args.episode_rollout else
                      getattr(model, 'episode_rollout_config', None))
    if episode_config is not None:
        if args.total_steps is not None or args.startup_stagger_steps or args.boss_opening_dataset:
            raise ValueError('whole-game cohorts reject fixed-step targets, stagger and opening-only datasets')
        if model.amp_mode != 'off' or not model.policy.core.upper_config.minibatch_size:
            raise ValueError('whole-game cohorts require FP32 strategic minibatch PPO')
        th.set_float32_matmul_precision('highest')
        th.backends.cuda.matmul.allow_tf32 = False
        th.backends.cudnn.allow_tf32 = False
        args.episode_rollout = True
        episode_rollout.configure(model, episode_config, rank=args.rank, world_size=args.world_size,
                                  grow_environments=args.fork_grow_environments)
    args.episode_rollout_config = episode_config
    from rl.algorithms.independent_rollout import configure as configure_independent
    independent_config = (json.loads(args.independent_rollout_config.read_text(encoding='utf-8'))
                          if args.independent_rollout_config is not None else None)
    args.independent_rollout_config = configure_independent(
        model, independent_config, fork=args.fork_state is not None)
    configure_training_scope(model, args)
    args.ppo_advantage_normalization = configure_advantage_normalization(
        model, args.ppo_advantage_normalization)
    configure_packed_updates(model, json.loads(args.ppo_update_config.read_text(encoding='utf-8'))
        if args.ppo_update_config is not None else None)
    args.ppo_update_config = model.ppo_update_config
    from rl.algorithms.deck_schedule import configure as configure_deck_schedule
    deck_schedule = configure_deck_schedule(model,
        json.loads(args.deck_training_schedule.read_text(encoding='utf-8'))
        if args.deck_training_schedule is not None else None)
    args.deck_training_schedule = getattr(model, 'deck_training_schedule', None)
    if args.ppo_update_config is not None and args.world_size > 1 and args.gradient_transport != 'ssh-ring':
        raise ValueError('packed multi-rank PPO requires ssh-ring scalar consensus')
    if (args.ppo_advantage_normalization == 'rollout' and args.world_size > 1
            and args.gradient_transport != 'ssh-ring'):
        raise ValueError('global rollout advantage normalization requires ssh-ring transport')
    if args.independent_rollout_config is not None:
        if args.reasoning_updates or args.reasoning_coef or args.interactive_reasoning:
            raise ValueError('independent cohorts do not support reasoning training')
    if episode_config is not None:
        args.episode_rollout_seed_offset = getattr(model, 'episode_rollout_seed_offset', 0)
        args.episode_rollout_growth = getattr(model, 'episode_rollout_growth', None)
    if args.checkpoint_keep_round_multiple and episode_config is None:
        raise ValueError('round-multiple retention requires complete-game cohorts')
    continuous_joint = (args.continuous and joint_upper(model)
        and getattr(model, 'nosl_is_az_combat_strategy', None) == 'small-playout'
        and hasattr(model.policy, 'small_playout')
        and model.policy.core.upper_config.learning_rate > 0
        and model.policy.small_playout_config['learning_rate'] > 0)
    if args.continuous and not ((pure_ppo and args.reasoning_updates == 0) or (
        getattr(model, 'nosl_is_az', False)
        and getattr(model, 'nosl_is_az_inner_only', False)
        and (not joint_upper(model) or continuous_joint)
        and args.reasoning_updates == 0
    )):
        raise ValueError('--continuous supports fixed-rate shared whole-run PPO, inner-only NOSL or joint-upper small-playout without reasoning updates')
    model.nosl_is_az_replay_enabled = bool(args.nosl_is_az_replay_enabled)
    print(
        f"[rank {args.rank}] NOSL-IS-AZ replay enabled: "
        f"{model.nosl_is_az_replay_enabled}",
        flush=True,
    )
    if args.nosl_is_az_simulations is not None:
        if not bool(getattr(model, "nosl_is_az", False)):
            raise ValueError("--nosl-is-az-simulations requires a NOSL checkpoint")
        model.nosl_is_az_simulations = int(args.nosl_is_az_simulations)
        print(
            f"[rank {args.rank}] NOSL-IS-AZ simulations override: "
            f"{model.nosl_is_az_simulations}",
            flush=True,
        )
    model.nosl_is_az_dirichlet_alpha = float(args.nosl_is_az_dirichlet_alpha)
    model.nosl_is_az_dirichlet_epsilon = float(args.nosl_is_az_dirichlet_epsilon)
    print(
        f"[rank {args.rank}] NOSL root Dirichlet noise: "
        f"alpha={model.nosl_is_az_dirichlet_alpha} "
        f"epsilon={model.nosl_is_az_dirichlet_epsilon}",
        flush=True,
    )
    if args.nosl_is_az_depth is not None:
        if not bool(getattr(model, "nosl_is_az", False)):
            raise ValueError(
                "--nosl-is-az-depth requires a NOSL-IS-AZ checkpoint"
            )
        model.nosl_is_az_depth = int(args.nosl_is_az_depth)
        print(
            f"[rank {args.rank}] NOSL-IS-AZ depth override: "
            f"{model.nosl_is_az_depth} "
            "(0=unlimited node expansion)",
            flush=True,
        )
    if args.nosl_is_az_end_turn_as_leaf is not None:
        if not bool(getattr(model, "nosl_is_az", False)):
            raise ValueError(
                "--nosl-is-az-end-turn-as-leaf requires a NOSL-IS-AZ checkpoint"
            )
        model.nosl_is_az_end_turn_as_leaf = bool(
            args.nosl_is_az_end_turn_as_leaf
        )
        print(
            f"[rank {args.rank}] NOSL End Turn leaf override: "
            f"{model.nosl_is_az_end_turn_as_leaf}",
            flush=True,
        )
    gumbel_overrides = (
        args.nosl_is_az_fixed_worldline,
        args.nosl_is_az_prior_temperature,
        args.nosl_is_az_search_mode,
        args.nosl_is_az_paper_full_gumbel,
        args.nosl_is_az_gumbel_candidates,
        args.nosl_is_az_gumbel_scale,
        args.nosl_is_az_gumbel_value_scale,
        args.nosl_is_az_gumbel_max_visit_init,
        args.nosl_is_az_gumbel_rescale_q,
        args.nosl_is_az_gumbel_use_mixed_value,
        args.nosl_is_az_gumbel_interior_uniformity,
        args.nosl_is_az_gumbel_interior_coverage_fraction,
        args.nosl_is_az_gumbel_interior_value_mode,
        args.nosl_is_az_gumbel_interior_max_mix_visits,
    )
    if any(value is not None for value in gumbel_overrides) and not bool(
        getattr(model, "nosl_is_az", False)
    ):
        raise ValueError("NOSL search overrides require a NOSL-IS-AZ checkpoint")
    mode = (
        args.nosl_is_az_search_mode
        if args.nosl_is_az_search_mode is not None
        else str(getattr(model, "nosl_is_az_search_mode", "puct"))
    )
    if mode not in NOSL_IS_AZ_SEARCH_MODES:
        mode = "puct"
    if mode != "puct" and not bool(getattr(model, "nosl_is_az_cpp", False)):
        raise ValueError("Gumbel NOSL modes require a C++ NOSL checkpoint")
    model.nosl_is_az_search_mode = mode
    model.nosl_is_az_root_search_mode = NOSL_IS_AZ_SEARCH_MODES[mode]
    gumbel_fields = (
        ("nosl_is_az_merge_equivalent_actions", True, bool),
        ("nosl_is_az_fixed_worldline", False, bool),
        ("nosl_is_az_prior_temperature", 1.0, float),
        ("nosl_is_az_paper_full_gumbel", False, bool),
        ("nosl_is_az_gumbel_candidates", 0, int),
        ("nosl_is_az_gumbel_scale", 1.0, float),
        ("nosl_is_az_gumbel_value_scale", 0.1, float),
        ("nosl_is_az_gumbel_max_visit_init", 50.0, float),
        ("nosl_is_az_gumbel_rescale_q", True, bool),
        ("nosl_is_az_gumbel_use_mixed_value", True, bool),
        ("nosl_is_az_gumbel_interior_uniformity", 0.0, float),
        ("nosl_is_az_gumbel_interior_coverage_fraction", 0.0, float),
        ("nosl_is_az_gumbel_interior_value_mode", 2, int),
        ("nosl_is_az_gumbel_interior_max_mix_visits", 64.0, float),
    )
    for name, default, convert in gumbel_fields:
        requested = getattr(args, name)
        value = getattr(model, name, default) if requested is None else requested
        value = convert(value)
        setattr(model, name, value)
        setattr(args, name, value)
    if bool(getattr(model, "nosl_is_az_paper_full_gumbel", False)):
        apply_strict_training_settings(model, args)
    args.nosl_is_az_search_mode = mode
    from rl.algorithms.action_groups import CONTRACT
    model.nosl_action_group_contract = (CONTRACT if model.nosl_is_az_merge_equivalent_actions
                                        else 'individual-actions')
    if model.nosl_is_az_fixed_worldline and not getattr(model, 'nosl_is_az_cpp', False):
        raise ValueError('fixed-worldline training requires the C++ search backend')
    if bool(getattr(model, "nosl_is_az", False)):
        print(
            f"[rank {args.rank}] NOSL search: mode={mode} "
            f"root_mode={model.nosl_is_az_root_search_mode} "
            f"paper_full_gumbel={model.nosl_is_az_paper_full_gumbel} "
            f"fixed_worldline={model.nosl_is_az_fixed_worldline} "
            f"prior_temperature={model.nosl_is_az_prior_temperature} "
            f"candidates={model.nosl_is_az_gumbel_candidates} "
            f"gumbel_scale={model.nosl_is_az_gumbel_scale} "
            f"value_scale={model.nosl_is_az_gumbel_value_scale} "
            f"max_visit_init={model.nosl_is_az_gumbel_max_visit_init} "
            f"rescale_q={model.nosl_is_az_gumbel_rescale_q} "
            f"mixed_value={model.nosl_is_az_gumbel_use_mixed_value} "
            f"coverage={model.nosl_is_az_gumbel_interior_coverage_fraction} "
            f"value_mode={model.nosl_is_az_gumbel_interior_value_mode} "
            f"max_mix_visits={model.nosl_is_az_gumbel_interior_max_mix_visits}",
            flush=True,
        )
    if args.nosl_is_az_batch_size is not None:
        if not bool(getattr(model, "nosl_is_az", False)):
            raise ValueError(
                "--nosl-is-az-batch-size requires a NOSL-IS-AZ checkpoint"
            )
        model.nosl_is_az_batch_size = int(args.nosl_is_az_batch_size)
        print(
            f"[rank {args.rank}] NOSL-IS-AZ batch size override: "
            f"{model.nosl_is_az_batch_size}",
            flush=True,
        )
    model.nosl_is_az_micro_batch_size = int(
        getattr(model, "nosl_is_az_micro_batch_size", 2048)
    )
    model.nosl_is_az_replay_capacity = int(
        getattr(model, "nosl_is_az_replay_capacity", 131072)
    )
    model.nosl_is_az_replay_min_size = int(
        getattr(model, "nosl_is_az_replay_min_size", 1)
    )
    model.nosl_is_az_replay_max_old_fraction = float(
        getattr(model, "nosl_is_az_replay_max_old_fraction", 1.0)
    )
    if args.nosl_is_az_micro_batch_size is not None:
        model.nosl_is_az_micro_batch_size = int(args.nosl_is_az_micro_batch_size)
    if args.nosl_is_az_replay_capacity is not None:
        model.nosl_is_az_replay_capacity = int(args.nosl_is_az_replay_capacity)
    if args.nosl_is_az_replay_min_size is not None:
        model.nosl_is_az_replay_min_size = int(args.nosl_is_az_replay_min_size)
    if args.nosl_is_az_replay_max_old_fraction is not None:
        model.nosl_is_az_replay_max_old_fraction = float(
            args.nosl_is_az_replay_max_old_fraction
        )
    if model.nosl_is_az_replay_min_size > model.nosl_is_az_replay_capacity:
        raise ValueError(
            "NOSL replay min size cannot exceed replay capacity"
        )
    if bool(getattr(model, "nosl_is_az", False)):
        # W&B should record the effective checkpoint-or-CLI values rather than
        # nullable override arguments.
        args.nosl_is_az_batch_size = model.nosl_is_az_batch_size
        args.nosl_is_az_micro_batch_size = model.nosl_is_az_micro_batch_size
        args.nosl_is_az_replay_capacity = model.nosl_is_az_replay_capacity
        args.nosl_is_az_replay_min_size = model.nosl_is_az_replay_min_size
        args.nosl_is_az_replay_max_old_fraction = (
            model.nosl_is_az_replay_max_old_fraction
        )
        print(
            f"[rank {args.rank}] NOSL-IS-AZ replay: "
            f"capacity={model.nosl_is_az_replay_capacity} "
            f"min_size={model.nosl_is_az_replay_min_size} "
            f"max_old_fraction={model.nosl_is_az_replay_max_old_fraction} "
            f"effective_batch={model.nosl_is_az_batch_size} "
            f"micro_batch={model.nosl_is_az_micro_batch_size}",
            flush=True,
        )
    if args.nosl_is_az_leaf_batch_size is not None:
        if not bool(getattr(model, "nosl_is_az", False)):
            raise ValueError(
                "--nosl-is-az-leaf-batch-size requires a NOSL-IS-AZ checkpoint"
            )
        model.nosl_is_az_leaf_batch_size = int(
            args.nosl_is_az_leaf_batch_size
        )
        print(
            f"[rank {args.rank}] NOSL-IS-AZ leaf batch size override: "
            f"{model.nosl_is_az_leaf_batch_size}",
            flush=True,
        )
    inference_override = (
        args.nosl_is_az_inference_batch_size is not None
        or args.nosl_is_az_inference_max_wait_ms is not None
        or args.nosl_is_az_inference_schedule is not None
    )
    if inference_override and not bool(getattr(model, "nosl_is_az", False)):
        raise ValueError(
            "NOSL inference scheduling overrides require a NOSL-IS-AZ checkpoint"
        )
    if bool(getattr(model, "nosl_is_az", False)):
        model.nosl_is_az_inference_batch_size = int(getattr(
            model, "nosl_is_az_inference_batch_size", 32
        ))
        model.nosl_is_az_inference_max_wait_ms = float(getattr(
            model, "nosl_is_az_inference_max_wait_ms", 0.5
        ))
        model.nosl_is_az_inference_schedule = str(getattr(
            model, "nosl_is_az_inference_schedule", "fixed_wait"
        ))
        if args.nosl_is_az_inference_batch_size is not None:
            model.nosl_is_az_inference_batch_size = int(
                args.nosl_is_az_inference_batch_size
            )
        if args.nosl_is_az_inference_max_wait_ms is not None:
            model.nosl_is_az_inference_max_wait_ms = float(
                args.nosl_is_az_inference_max_wait_ms
            )
        if args.nosl_is_az_inference_schedule is not None:
            model.nosl_is_az_inference_schedule = str(
                args.nosl_is_az_inference_schedule
            )
        model._nosl_leaf_batch_controller = None
        args.nosl_is_az_inference_batch_size = (
            model.nosl_is_az_inference_batch_size
        )
        args.nosl_is_az_inference_max_wait_ms = (
            model.nosl_is_az_inference_max_wait_ms
        )
        args.nosl_is_az_inference_schedule = (
            model.nosl_is_az_inference_schedule
        )
        print(
            f"[rank {args.rank}] NOSL-IS-AZ cross-tree inference: "
            f"schedule={model.nosl_is_az_inference_schedule} "
            f"max_batch={model.nosl_is_az_inference_batch_size} "
            f"wait_or_watchdog_floor_ms="
            f"{model.nosl_is_az_inference_max_wait_ms}",
            flush=True,
        )
    # Restarted environments cannot restore native in-flight games. Choose a
    # new deterministic seed block rather than replaying the run's first games.
    model.set_random_seed(args.seed + args.rank * 1_000_003 +
                          (int(model.num_timesteps) if recovery_state is not None else 0))
    if getattr(model, '_frozen_upper', False):
        from rl.algorithms import outer_return
        outer_return.setup(model)
        args.outer_return_contract = model.outer_return_contract
    if args.leaf_inference == 'cuda-graph':
        from rl.search.inference import CombatLeafGraphs
        model._native_leaf_inference = CombatLeafGraphs(model.policy.core, model.device,
            amp=args.leaf_amp_mode if args.leaf_amp_mode is not None else args.amp_mode,
            compile_forward=args.leaf_compile,
            profile_interval=args.leaf_graph_profile_interval,
            batch_multiple=args.leaf_graph_batch_multiple,
            failure_path=(args.training_state_dir / f'failed-leaves-rank{args.rank}.npz')
            if args.training_state_dir else None)

    ring_transport = None
    if args.gradient_transport == "ssh-ring":
        next_rank = (args.rank + 1) % args.world_size
        next_peer = next(peer for peer in peers if peer.rank == next_rank)

        def ring_ready_barrier(stream=0) -> None:
            all_rank_file_barrier(
                tag=f"ssh_ring_listening_{stream}",
                rank=args.rank,
                peers=peers,
                key=key,
                control_dir=control_dir,
                local_root=local_root,
                remote_root=args.remote_root,
                run_id=run_id,
                timeout=args.barrier_timeout,
            )

        links = []
        try:
            for stream in range(args.ring_streams):
                links.append(SshRingAllReduce(
                    rank=args.rank,
                    world_size=args.world_size,
                    next_host=next_peer.host,
                    next_port=next_peer.port,
                    next_user=next_peer.user,
                    ssh_key=key,
                    listen_port=args.ring_port + stream,
                    timeout=args.barrier_timeout,
                    ready_barrier=lambda stream=stream: ring_ready_barrier(stream),
                ))
            ring_transport = links[0] if len(links) == 1 else ParallelRingAllReduce(links)
        except BaseException:
            for link in links:
                link.close()
            raise
        print(
            f"[rank {args.rank}] persistent SSH gradient ring connected to "
            f"rank {next_rank} over {args.ring_streams} stream(s)",
            flush=True,
        )

    start_timesteps = int(model.num_timesteps)
    global_rollout_size = rollout_size * args.world_size
    cumulative_global_steps = start_timesteps
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
        # Keep the legacy workspace charts usable even for fresh runs that do
        # not provide an explicit --wandb-step-base anchor.  W&B workspaces
        # remember a chart's custom x-axis; without this field those charts
        # show "no data" for otherwise valid metrics.
        wandb.define_metric("legacy_step")
        for metric in (
            "game/*",
            "rollout/*",
            "speed/*",
            "hierarchical/*",
            "train/*",
            "dist/*",
            "reasoning/*",
            "nosl_is_az/*",
            "boss/*",
            "upper/*",
            "outer_return/*",
            "global_step",
            "time/total_timesteps",
            "env_steps_per_second",
            "rule_prediction_accuracy",
        ):
            wandb.define_metric(metric, step_metric="legacy_step")

    callback = CallbackList([WholeRunTask().metrics_callback(args)])
    if args.rollout_resource_interval:
        callback.callbacks.append(RolloutProgressCallback(args.rank, args.n_steps))
    total_timesteps, callback = model._setup_learn(
        # SB3 setup initializes callbacks; the iterator below owns termination.
        # Continuous NOSL uses an explicit fixed rate, not a finite-horizon schedule.
        total_timesteps=(rollout_size if args.continuous else args.updates * rollout_size),
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name=f"dist_sb3_rank{args.rank}",
        progress_bar=False,
    )
    if pure_ppo:
        model.ppo_rollouts_completed = int(source_ppo_round)
    if recovery_state is not None:
        recovery_manifest = restore_learner(
            model,
            recovery_state,
            rank=args.rank,
            world_size=args.world_size,
            restore_environment=not bool(
                args.fork_reset_environments or args.resume_reset_environments),
            grow_environments=args.fork_grow_environments or args.resume_grow_environments,
        )
        if args.fork_grow_environments:
            print('ENVIRONMENT_EXPANDED_ON_FORK ' + json.dumps(
                model._recovery_environment_growth, sort_keys=True), flush=True)
        if args.resume_grow_environments:
            print('ENVIRONMENT_EXPANDED_ON_RESUME ' + json.dumps(
                model._recovery_environment_growth, sort_keys=True), flush=True)
        if args.fork_reset_environments or args.resume_reset_environments:
            print(
                ("ENVIRONMENT_RESET_ON_FORK " if args.fork_reset_environments
                 else "ENVIRONMENT_RESET_ON_RESUME ") + json.dumps({
                    "source_global_step": int(recovery_manifest["global_step"]),
                    "environments": int(model.env.num_envs),
                }, sort_keys=True),
                flush=True,
            )
    else:
        recovery_manifest = None
    if args.fork_public_lower:
        from rl.training.small_playout import fork_public_lower
        fork_public_lower(model,args.fork_public_lower,args.fork_public_weights)
        args.small_playout_config=dict(model.policy.small_playout_config)
    environment_restored = bool(
        recovery_manifest is not None
        and getattr(model, "_recovery_environment_restored", False)
    )
    if environment_restored:
        print(
            "ENVIRONMENT_RESTORED " + json.dumps({
                "global_step": int(model.num_timesteps),
                "schema_version": int(recovery_manifest["schema_version"]),
                "environments": int(model.env.num_envs),
            }, sort_keys=True),
            flush=True,
        )
    startup_stagger = {} if environment_restored else run_startup_stagger(
        model,
        span_steps=args.startup_stagger_steps,
        rollout_steps=args.n_steps,
    )
    if startup_stagger:
        startup_stagger["discarded_transitions_global"] = (
            startup_stagger["discarded_transitions_local"] * args.world_size
        )
        print(
            "STARTUP_STAGGER " + json.dumps(startup_stagger, sort_keys=True),
            flush=True,
        )
    # Startup occupancy warmup was removed from the production lifecycle.  Keep
    # the CLI fields readable for old argument files, but never spend training
    # time on discarded rollouts or mutate the formal sampling boundary.
    startup_occupancy = {}
    callback.on_training_start(locals(), globals())

    completed_updates = 0
    timing_sums = dict.fromkeys(('total_sec_rank', 'rollout_sec', 'backward_sec',
                                'sync_sec', 'optimizer_step_sec'), 0.)
    last_bundle_step = None
    training_failed = False
    resource_stack = ExitStack()
    from rl.training.failure_snapshot import install_truncation_recorder, save_failure_scene
    install_truncation_recorder(model, args.training_state_dir, rank=args.rank)
    from rl.training.iteration_budget import IterationBudget
    iteration_budget = IterationBudget(args.iteration_timeout_seconds, rank=args.rank,
                                       directory=args.training_state_dir)
    try:
        for update in (count() if args.continuous else range(args.updates)):
            iteration_budget.start(update)
            if deck_schedule is not None or getattr(model, 'deck_search_config', None) is not None:
                model.deck_search_snapshot_dir = str(local_root / 'search-weights')
            if deck_schedule is not None:
                from rl.training.deck_schedule import at_collection_boundary
                at_collection_boundary(model, rank=args.rank, world_size=args.world_size,
                    local_root=local_root, ascension=args.ascension,
                    average=lambda values: average_scalar_dict(values, ring_transport))
            if model.device.type == "cuda":
                th.cuda.synchronize(model.device)
            started = time.perf_counter()
            monitor = None
            rollout_resources = None
            if args.rollout_resource_interval:
                from rl.training.resource_monitor import ResourceMonitor
                monitor = resource_stack.enter_context(ResourceMonitor(
                    interval=args.rollout_resource_interval,
                    gpu=model.device.type=='cuda', torch_device=model.device,
                    thread_details=True))
            # This entry owns its metrics-only callbacks. Keep arbitrary
            # library callbacks' autograd behavior unchanged.
            with th.inference_mode() if pure_ppo else nullcontext():
                ok = model.collect_rollouts(
                    model.env,
                    callback,
                    model.rollout_buffer,
                    n_rollout_steps=args.n_steps,
                )
            if model.device.type == "cuda":
                th.cuda.synchronize(model.device)
            if monitor is not None:
                rollout_resources = monitor.snapshot()
            rollout_sec = time.perf_counter() - started
            if not ok:
                raise RuntimeError("collect_rollouts returned False")

            update_started = time.perf_counter()

            # The inference graphs are unused during learner backpropagation.
            # Release their private pools before the full training microbatch;
            # the compiled callable remains and graphs are captured again on
            # the next collection. This prevents persistent graph allocations
            # from consuming the learner's remaining VRAM.
            graph_release_sec = 0.0
            graph_release_memory = None
            if (args.leaf_inference == 'cuda-graph'
                    and hasattr(model.policy.core, 'graph_actor_critic')):
                release_started = time.perf_counter()
                allocated_before = th.cuda.memory_allocated(model.device)
                reserved_before = th.cuda.memory_reserved(model.device)
                model._native_leaf_inference.clear()
                import gc
                gc.collect()
                th.cuda.empty_cache()
                graph_release_memory = {
                    'allocated_before': allocated_before,
                    'allocated_after': th.cuda.memory_allocated(model.device),
                    'reserved_before': reserved_before,
                    'reserved_after': th.cuda.memory_reserved(model.device),
                }
                graph_release_sec = time.perf_counter() - release_started

            model.policy.set_training_mode(True)
            if args.training_scope != 'lower':
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
            gradient_pack_seconds = 0.0
            planned_update_stats = {}
            step_seconds = 0.0
            backward_seconds = 0.0
            grad_bytes = 0
            optimizer_steps = 0
            cuda_memory_peak = None
            if pure_ppo:
                prepare_advantage_normalization(model, world_size=args.world_size,
                    average=(lambda values: average_scalar_dict(values, ring_transport))
                    if ring_transport is not None else None)
                model._ppo_microbatch_stats = dict(chunks=0, rows=0, oom_retries=0,
                                                  min_rows=args.batch_size, max_rows=0)
                model._ppo_update_metric_sums = None
                model._ppo_update_metric_count = 0

            inner_only_nosl = bool(
                model.nosl_is_az and model.nosl_is_az_inner_only
            )
            distributed_nosl = bool(
                inner_only_nosl
                and getattr(model, "nosl_is_az_combat_strategy", "model") == "model"
            )
            nosl_update = None
            if distributed_nosl:
                completed_rollouts = start_timesteps // global_rollout_size + update
                scheduled_updates, scheduled_value_only = update_schedule.plan(
                    completed_rollouts, int(model.nosl_is_az_updates))
                nosl_update = distributed_nosl_is_az_update(
                    model,
                    rank=args.rank,
                    world_size=args.world_size,
                    peers=peers,
                    key=key,
                    control_dir=control_dir,
                    local_root=local_root,
                    remote_root=args.remote_root,
                    run_id=run_id,
                    update=update,
                    ring_transport=ring_transport,
                    barrier_timeout=args.barrier_timeout,
                    update_count=scheduled_updates,
                    value_only_updates=scheduled_value_only,
                )
                nosl_update.update(schedule_round=float(completed_rollouts + 1),
                    scheduled_updates=float(scheduled_updates),
                    scheduled_value_only_updates=float(scheduled_value_only))

            # Inner-only always delegates learning away from the ordinary PPO
            # buffer.  A learned lower uses the distributed NOSL update above;
            # a fixed rulebase lower has no lower learner at all.  In both
            # cases the only strategic update is the macro-action PPO below.
            for epoch in range(0 if inner_only_nosl else args.n_epochs):
                if model.ppo_update_config is not None:
                    offset = optimizer_steps
                    def sync_packed(step):
                        return synchronize_gradient_step(
                            model, rank=args.rank, world_size=args.world_size, peers=peers,
                            key=key, control_dir=control_dir, local_root=local_root,
                            remote_root=args.remote_root, run_id=run_id,
                            tag=f'u{update:06d}_e{epoch:03d}_s{offset + step:06d}',
                            ring_transport=ring_transport, barrier_timeout=args.barrier_timeout)
                    packed = distributed_packed_epoch(model, clip_range, clip_range_vf,
                        ring=ring_transport, synchronize=sync_packed,
                        average=lambda values: average_scalar_dict(values, ring_transport))
                    for name, value in packed.items():
                        if name.startswith(('planned_', 'graph_')):
                            planned_update_stats[name] = planned_update_stats.get(name, 0) + value
                            model.logger.record('pure_rl/update_' + name, planned_update_stats[name])
                    losses.extend(packed['losses'])
                    optimizer_steps += packed['optimizer_steps']
                    backward_seconds += packed['backward_sec']
                    gradient_pack_seconds += packed['gradient_pack_sec']
                    sync_seconds += packed['sync_sec']
                    sync_mbps.extend(packed['sync_mbps'])
                    grad_bytes = packed['grad_bytes']
                    step_seconds += packed['optimizer_step_sec']
                    model.logger.record('pure_rl/update_loss_scale', model.ppo_update_loss_scale)
                    model.logger.record('pure_rl/update_overflow_retries', packed['overflow_retries'])
                    for field in ('context_input_rows', 'context_unique_rows', 'context_oom_retries'):
                        model.logger.record('pure_rl/' + field, packed[field])
                    continue
                accumulated = 0
                model.policy.optimizer.zero_grad()
                for rollout_data in model.rollout_buffer.get(args.batch_size):
                    backward_started = time.perf_counter()
                    batch_result = ppo_microbatch_backward(
                        model,
                        rollout_data,
                        clip_range,
                        clip_range_vf,
                        profile_cuda_memory=args.profile_cuda_memory,
                        phasewise_backward=args.phasewise_backward,
                    )
                    batch_cuda_profile = batch_result["cuda_profile"]
                    if (
                        batch_cuda_profile is not None
                        and (
                            cuda_memory_peak is None
                            or batch_cuda_profile["peak_allocated"]
                            > cuda_memory_peak["peak_allocated"]
                        )
                    ):
                        cuda_memory_peak = batch_cuda_profile
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

                    sync = synchronize_gradient_step(
                        model, rank=args.rank, world_size=args.world_size, peers=peers,
                        key=key, control_dir=control_dir, local_root=local_root,
                        remote_root=args.remote_root, run_id=run_id,
                        tag=f"u{update:06d}_e{epoch:03d}_s{optimizer_steps:06d}",
                        ring_transport=ring_transport, barrier_timeout=args.barrier_timeout,
                    )
                    gradient_pack_seconds += sync["gradient_pack_sec"]
                    sync_seconds += sync["sync_sec"]
                    sync_mbps.append(sync["sync_mbps"])
                    grad_bytes = int(sync["grad_bytes"])
                    if model.device.type == "cuda":
                        th.cuda.synchronize(model.device)
                    step_started = time.perf_counter()
                    optimizer_step(model)
                    if model.device.type == "cuda":
                        th.cuda.synchronize(model.device)
                    step_seconds += time.perf_counter() - step_started
                    accumulated = 0
                    optimizer_steps += 1

            if nosl_update is not None:
                losses.append(
                    nosl_update["policy_loss"]
                    + model.nosl_is_az_value_coef * nosl_update["value_loss"]
                )
                optimizer_steps = int(nosl_update["updates"])
                backward_seconds = float(nosl_update["backward_sec"])
                sync_seconds = float(nosl_update["sync_sec"])
                sync_mbps.append(float(nosl_update["sync_mbps"]))
                grad_bytes = int(nosl_update["grad_bytes"])
                step_seconds = float(nosl_update["optimizer_step_sec"])

            def sync_small(step):
                return synchronize_gradient_step(model, rank=args.rank, world_size=args.world_size,
                    peers=peers, key=key, control_dir=control_dir, local_root=local_root,
                    remote_root=args.remote_root, run_id=run_id,
                    tag=f'u{update:06d}_small{step:04d}', ring_transport=ring_transport,
                    barrier_timeout=args.barrier_timeout)
            small_update = {}
            if hasattr(model.policy, 'small_playout'):
                from rl.algorithms.small_playout import update_online
                small_update = update_online(model, rank=args.rank, world_size=args.world_size,
                    average=lambda values: average_scalar_dict(values, ring_transport), synchronize=sync_small)
            upper_update = {}
            upper_started = time.perf_counter()
            if joint_upper(model) and args.training_scope == 'lower':
                from rl.algorithms.upper_ppo import finish_frozen_collection
                upper_update = finish_frozen_collection(model, world_size=args.world_size,
                    average=lambda values: average_scalar_dict(values, ring_transport))
            elif joint_upper(model):
                def sync_upper(epoch):
                    return synchronize_gradient_step(model, rank=args.rank, world_size=args.world_size,
                        peers=peers, key=key, control_dir=control_dir, local_root=local_root,
                        remote_root=args.remote_root, run_id=run_id,
                        tag=f'u{update:06d}_upper{epoch:03d}', ring_transport=ring_transport,
                        barrier_timeout=args.barrier_timeout)
                upper_update = update_upper(model, rank=args.rank, world_size=args.world_size,
                    average=lambda values: average_scalar_dict(values, ring_transport),
                    synchronize=sync_upper)
            elif pure_ppo:
                from rl.training import reward_monitor, campfire_monitor
                choice_counts = average_scalar_dict({**reward_monitor.counts(model),
                    **campfire_monitor.counts(model)}, ring_transport)
                upper_update = {**reward_monitor.metrics(choice_counts, args.world_size),
                    **campfire_monitor.metrics(choice_counts, args.world_size)}
            upper_seconds = time.perf_counter() - upper_started if joint_upper(model) else 0.0

            reasoning_records = []
            reasoning_sync_seconds = 0.0
            reasoning_sync_mbps = []
            reasoning_compile_wait_seconds = 0.0
            reasoning_compute_seconds = 0.0
            reasoning_step_seconds = 0.0
            reasoning_grad_bytes = 0
            reasoning_optimizer_steps = 0
            reasoning_accumulation_steps = max(
                1, int(args.reasoning_gradient_accumulation_steps)
            )
            model.policy.optimizer.zero_grad()
            for reasoning_update in range(args.reasoning_updates):
                group_position = reasoning_update % reasoning_accumulation_steps
                group_size = min(
                    reasoning_accumulation_steps,
                    args.reasoning_updates - reasoning_update + group_position,
                )
                reasoning_result = reasoning_microbatch_backward(
                    model,
                    args.reasoning_batch_size,
                    loss_scale=1.0 / group_size,
                )
                if reasoning_result is None:
                    reasoning_result = {
                        "loss": 0.0,
                        "token_accuracy": 0.0,
                        "supervised_tokens": 0.0,
                        "sampled_samples": 0.0,
                        "samples": 0.0,
                        "eligible_samples": 0.0,
                        "dropped_too_long": 0.0,
                        "environment_tokens": 0.0,
                        "atomic_trace_rate": 0.0,
                        "sequence_length_mean": 0.0,
                        "sequence_length_max": 0.0,
                        "compile_wait_sec": 0.0,
                        "compute_sec": 0.0,
                    }
                reasoning_compile_wait_seconds += reasoning_result.get(
                    "compile_wait_sec", 0.0
                )
                reasoning_compute_seconds += reasoning_result.get(
                    "compute_sec", 0.0
                )
                reasoning_records.append(reasoning_result)
                group_finished = (
                    group_position + 1 >= reasoning_accumulation_steps
                    or reasoning_update + 1 >= args.reasoning_updates
                )
                if group_finished:
                    reasoning_sync = synchronize_gradient_step(
                        model,
                        rank=args.rank,
                        world_size=args.world_size,
                        peers=peers,
                        key=key,
                        control_dir=control_dir,
                        local_root=local_root,
                        remote_root=args.remote_root,
                        run_id=run_id,
                        tag=(
                            f"u{update:06d}_reasoning"
                            f"{reasoning_optimizer_steps:03d}"
                        ),
                        ring_transport=ring_transport,
                        barrier_timeout=args.barrier_timeout,
                    )
                    reasoning_sync_seconds += reasoning_sync["sync_sec"]
                    reasoning_sync_mbps.append(reasoning_sync["sync_mbps"])
                    reasoning_grad_bytes = int(reasoning_sync["grad_bytes"])
                    if model.device.type == "cuda":
                        th.cuda.synchronize(model.device)
                    reasoning_step_started = time.perf_counter()
                    optimizer_step(model)
                    if model.device.type == "cuda":
                        th.cuda.synchronize(model.device)
                    reasoning_step_seconds += (
                        time.perf_counter() - reasoning_step_started
                    )
                    reasoning_optimizer_steps += 1

            reasoning_mean = {
                key: float(np.mean([row[key] for row in reasoning_records]))
                for key in (
                    "loss",
                    "token_accuracy",
                    "supervised_tokens",
                    "sampled_samples",
                    "samples",
                    "eligible_samples",
                    "dropped_too_long",
                    "environment_tokens",
                    "atomic_trace_rate",
                    "sequence_length_mean",
                    "sequence_length_max",
                )
            } if reasoning_records else {
                key: 0.0 for key in (
                    "loss",
                    "token_accuracy",
                    "supervised_tokens",
                    "sampled_samples",
                    "samples",
                    "eligible_samples",
                    "dropped_too_long",
                    "environment_tokens",
                    "atomic_trace_rate",
                    "sequence_length_mean",
                    "sequence_length_max",
                )
            }
            reasoning_mean["optimizer_steps"] = float(reasoning_optimizer_steps)
            reasoning_sampled_samples_total = float(
                sum(row["sampled_samples"] for row in reasoning_records)
            )

            # Complete learner wall time includes data gathering, all backward
            # passes, transport/waiting and optimizer work. Saves are separate.
            if model.device.type == "cuda":
                th.cuda.synchronize(model.device)
            update_seconds = time.perf_counter() - update_started

            actual_rollout_size = (int(model.episode_rollout_stats['transitions'])
                                   if episode_config is not None else rollout_size)
            global_actions = int(round(average_scalar_dict(
                {'actions': float(actual_rollout_size)}, ring_transport)['actions'] * args.world_size))
            rank_phase_seconds = average_scalar_dict({
                f'{phase}{i}': seconds if i == args.rank else 0.
                for phase, seconds in (('rollout', rollout_sec), ('update', update_seconds))
                for i in range(args.world_size)}, ring_transport)
            rollout_max_seconds = max(rank_phase_seconds[f'rollout{i}']
                                      for i in range(args.world_size)) * args.world_size
            update_max_seconds = max(rank_phase_seconds[f'update{i}']
                                     for i in range(args.world_size)) * args.world_size
            cumulative_global_steps += global_actions
            record = {
                "rank": args.rank,
                "world_size": args.world_size,
                "gradient_transport": args.gradient_transport,
                "update": update,
                "start_timesteps": start_timesteps,
                "global_timesteps": cumulative_global_steps,
                "transitions_local": actual_rollout_size,
                "transitions_global": global_actions,
                "rollout_sec": rollout_sec,
                "leaf_graph_release_sec": graph_release_sec,
                "leaf_graph_release_memory": graph_release_memory,
                "rollout_steps_per_sec_local": actual_rollout_size / rollout_sec,
                "rollout_sec_max": rollout_max_seconds,
                "update_sec": update_seconds,
                "update_sec_max": update_max_seconds,
                "rollout_steps_per_sec_global": global_actions / rollout_max_seconds,
                "loss_mean": float(np.mean(losses)) if losses else 0.0,
                "optimizer_steps": optimizer_steps,
                "sync_sec": sync_seconds,
                "sync_mbps_mean": float(np.mean(sync_mbps)) if sync_mbps else 0.0,
                "backward_sec": backward_seconds,
                "gradient_pack_sec": gradient_pack_seconds,
                # Averaging is included in sync_sec by the shared transport.
                # Retain the historical W&B field without double-counting it.
                "avg_grad_sec": 0.0,
                "optimizer_step_sec": step_seconds,
                "grad_bytes": grad_bytes,
                "reasoning_loss": reasoning_mean["loss"],
                "reasoning_token_accuracy": reasoning_mean["token_accuracy"],
                "reasoning_supervised_tokens": reasoning_mean[
                    "supervised_tokens"
                ],
                "reasoning_sampled_samples": reasoning_mean[
                    "sampled_samples"
                ],
                "reasoning_sampled_samples_total": (
                    reasoning_sampled_samples_total
                ),
                "reasoning_batch_size": float(args.reasoning_batch_size),
                "reasoning_updates_requested": float(args.reasoning_updates),
                "reasoning_gradient_accumulation_steps": float(
                    args.reasoning_gradient_accumulation_steps
                ),
                "reasoning_effective_batch_size": float(
                    args.reasoning_batch_size
                    * args.reasoning_gradient_accumulation_steps
                    * args.world_size
                ),
                "reasoning_samples": reasoning_mean["samples"],
                "reasoning_eligible_samples": reasoning_mean[
                    "eligible_samples"
                ],
                "reasoning_dropped_too_long": reasoning_mean[
                    "dropped_too_long"
                ],
                "reasoning_environment_tokens": reasoning_mean[
                    "environment_tokens"
                ],
                "reasoning_atomic_trace_rate": reasoning_mean[
                    "atomic_trace_rate"
                ],
                "reasoning_sequence_length_mean": reasoning_mean[
                    "sequence_length_mean"
                ],
                "reasoning_sequence_length_max": reasoning_mean[
                    "sequence_length_max"
                ],
                "reasoning_optimizer_steps": reasoning_mean[
                    "optimizer_steps"
                ],
                "reasoning_sync_sec": reasoning_sync_seconds,
                "reasoning_sync_mbps_mean": (
                    float(np.mean(reasoning_sync_mbps))
                    if reasoning_sync_mbps else 0.0
                ),
                "reasoning_compile_wait_sec": reasoning_compile_wait_seconds,
                "reasoning_compute_sec": reasoning_compute_seconds,
                "reasoning_step_sec": reasoning_step_seconds,
                "reasoning_grad_bytes": reasoning_grad_bytes,
                "reasoning_total_sec": (
                    reasoning_sync_seconds
                    + reasoning_compile_wait_seconds
                    + reasoning_compute_seconds
                    + reasoning_step_seconds
                ),
                "total_sec_rank": (
                    rollout_sec
                    + upper_seconds
                    + graph_release_sec
                    + backward_seconds
                    + gradient_pack_seconds
                    + sync_seconds
                    + step_seconds
                    + reasoning_sync_seconds
                    + reasoning_compile_wait_seconds
                    + reasoning_compute_seconds
                    + reasoning_step_seconds
                ),
            }
            if update == 0 and startup_stagger:
                record["startup_stagger"] = dict(startup_stagger)
            if update == 0 and startup_occupancy:
                record["startup_occupancy"] = dict(startup_occupancy)
            record["reasoning_time_fraction"] = (
                record["reasoning_total_sec"]
                / max(record["total_sec_rank"], 1e-9)
            )
            if args.hierarchical_combat_hp:
                record["combat_value_loss_mean"] = (
                    float(np.mean(combat_value_losses))
                    if combat_value_losses else 0.0
                )
                record["overworld_value_loss_mean"] = (
                    float(np.mean(overworld_value_losses))
                    if overworld_value_losses else 0.0
                )
            if cuda_memory_peak is not None:
                record["cuda_memory_peak"] = cuda_memory_peak
            if monitor is not None:
                record['rollout_resources'] = rollout_resources

            # Episode metrics are maintained independently by each callback.
            # Average the fixed set across ranks before printing or sending it
            # to W&B, so the dashboard represents the global rollout rather
            # than rank 0's local subset.
            if pure_ppo:
                from rl.algorithms.ppo_metrics import update_moments
                for key, value in update_moments(model).items():
                    model.logger.record(key, value)
                for key, value in model._ppo_microbatch_stats.items():
                    model.logger.record('pure_rl/backward_' + key, value)
                for key, value in dict(learning_rate=model.policy.optimizer.param_groups[0]['lr'],
                        epochs=args.n_epochs, microbatch_rows_requested=args.batch_size,
                        normalization_group_size=(model._ppo_advantage_stats['count']
                            if args.ppo_advantage_normalization == 'rollout'
                            else model.policy.ppo_advantage_group_size),
                        advantage_normalization_rollout=int(args.ppo_advantage_normalization == 'rollout'),
                        effective_batch_size_global=args.batch_size * args.gradient_accumulation_steps * args.world_size).items():
                    model.logger.record('pure_rl/' + key, value)
            callback_scalars = logger_scalars(model)
            upper_counts = getattr(model, "_nosl_is_az_rollout_upper_counts", {})
            record["nosl_is_az_upper_strategy"] = model.nosl_is_az_upper_strategy
            record["nosl_is_az_upper_strategy_version"] = args.nosl_is_az_upper_strategy_version
            record["nosl_is_az_combat_strategy"] = model.nosl_is_az_combat_strategy
            record["nosl_is_az_combat_strategy_version"] = args.nosl_is_az_combat_strategy_version
            for suffix in (
                "worlds", "nodes", "beam", "depth", "turns", "leaf_profile", "nonce"
            ):
                record[f"rulebase_planner_{suffix}"] = getattr(
                    args, f"rulebase_planner_{suffix}"
                )
            for name, value in upper_counts.items():
                record[name + "_local"] = float(value)
                callback_scalars["nosl_is_az/" + name] = float(value)
            if nosl_update is not None:
                callback_scalars.update({
                    f"nosl_is_az/{key}": float(value)
                    for key, value in nosl_update.items()
                    if isinstance(value, (int, float))
                })
            aggregate_scalars = {
                key: float(record[key])
                for key in (
                    "reasoning_loss",
                    "reasoning_token_accuracy",
                    "reasoning_supervised_tokens",
                    "reasoning_sampled_samples",
                    "reasoning_sampled_samples_total",
                    "reasoning_batch_size",
                    "reasoning_updates_requested",
                    "reasoning_gradient_accumulation_steps",
                    "reasoning_effective_batch_size",
                    "reasoning_samples",
                    "reasoning_eligible_samples",
                    "reasoning_dropped_too_long",
                    "reasoning_environment_tokens",
                    "reasoning_atomic_trace_rate",
                    "reasoning_sequence_length_mean",
                    "reasoning_sequence_length_max",
                    "reasoning_optimizer_steps",
                    "reasoning_compile_wait_sec",
                    "reasoning_compute_sec",
                    "reasoning_step_sec",
                    "reasoning_sync_sec",
                    "reasoning_total_sec",
                    "reasoning_time_fraction",
                )
            }
            aggregate_scalars.update(
                _prepare_distributed_game_scalars(callback_scalars)
            )
            aggregate_scalars.update({
                key: float(callback_scalars.get(key, 0.0))
                for key in _DISTRIBUTED_NOSL_METRIC_KEYS
            })
            if model.nosl_supervision_contract == 'shared-wholerun-ppo-room-v1':
                aggregate_scalars.update({key: value for key, value in callback_scalars.items()
                                          if key.startswith(('pure_rl/', 'deck_search/'))})
            if getattr(model, '_frozen_upper', False):
                model.policy.core.teacher.assert_integrity()
                aggregate_scalars.update({key: value for key, value in callback_scalars.items()
                                          if key.startswith(('outer_return/', 'frozen_upper/'))})
            if args.hierarchical_combat_hp:
                aggregate_scalars.update({
                    "combat_value_loss_mean": float(
                        record["combat_value_loss_mean"]
                    ),
                    "overworld_value_loss_mean": float(
                        record["overworld_value_loss_mean"]
                    ),
                })
            aggregate_scalars = average_scalar_dict(
                aggregate_scalars, ring_transport
            )
            _finalize_distributed_game_scalars(
                aggregate_scalars, world_size=args.world_size,
            )
            if pure_ppo:
                from rl.algorithms.ppo_metrics import finalize_metrics
                aggregate_scalars.update(finalize_metrics(aggregate_scalars))
            # Both counters underwent the same mean reduction, so their ratio
            # is sum(matches)/sum(count), even with unequal rank populations.
            # Keep the historical zero/count=0 logging convention for W&B.
            aggregate_scalars[HP_ORDER_PREFIX + '_fraction'] = (
                hp_order_fraction(aggregate_scalars) or 0.0
            )
            record.update(aggregate_scalars)
            if planned_update_stats:
                record['planned_ppo'] = planned_update_stats
            if pure_ppo:
                record['pure_rl/natural_games_global'] = int(round(
                    record['pure_rl/game_total/natural'] * args.world_size))
                record['pure_rl/censored_games_global'] = int(round(
                    record['pure_rl/game_total/censored'] * args.world_size))
                if record['pure_rl/game_total/start_step'] == 0:
                    record['completed_full_games_global'] = record['pure_rl/natural_games_global']
            record['training_scope'] = args.training_scope
            if deck_schedule is not None:
                record['deck_training_state'] = model.deck_training_state
                record['training_objective_contract'] = model.training_objective_contract
            record.update({'upper/' + key: value for key, value in upper_update.items()})
            record.update({'small_playout_train/' + key: value for key, value in small_update.items()})
            if combat_strategy == 'small-playout':
                record['small_playout_stats_rank'] = dict(getattr(model, '_small_playout_stats', {}))
                caps = average_scalar_dict({'turn_card_limit_decisions':sum(v for k,v in record['small_playout_stats_rank'].items()
                    if k.endswith('turn_card_limit_decisions'))},ring_transport)
                record['small_playout/turn_card_limit_decisions_global']=int(round(caps['turn_card_limit_decisions']*args.world_size))
                record['small_playout_collection_sha256'] = getattr(model, 'small_playout_collection_sha256', None)
                games = average_scalar_dict({'completed_games': float(getattr(model, 'small_playout_games_local', 0))}, ring_transport)
                record['completed_full_games_global'] = int(round(games['completed_games']*args.world_size))
                record['completed_game_budget'] = args.completed_game_budget
            if joint_upper(model):
                record['upper/total_sec_rank'] = upper_seconds
                record['upper/effective_batch_size_global'] = upper_update.get('effective_batch_size_global', upper_update.get('samples', 0.))
                record['upper/micro_batch_size'] = model.policy.upper_config['micro_batch_size']
            if episode_config is not None:
                record['episode_scope'] = 'complete-game-cohort'
                record['rollout_round'] = model.episode_rollouts_completed
                record['episode_rollout_rank'] = dict(model.episode_rollout_stats)
                record['episode_rollout_censored_slots_rank'] = model.episode_rollout_censored_slots
                record['games_per_rollout_global'] = args.n_envs * args.world_size
                cohort_means = average_scalar_dict({k: float(model.episode_rollout_stats[k])
                for k in ('completed_games', 'censored_games', 'excluded_upper_samples', 'excluded_lower_samples', 'wins', 'floor_sum')}, ring_transport)
                if cohort_means['completed_games']:
                    record['episode_rollout/win_rate'] = cohort_means['wins'] / cohort_means['completed_games']
                    record['episode_rollout/floor_mean'] = cohort_means['floor_sum'] / cohort_means['completed_games']
                record['episode_rollout/completed_games_global'] = int(round(cohort_means['completed_games'] * args.world_size))
                for key in ('censored_games','excluded_upper_samples','excluded_lower_samples'):
                    record['episode_rollout/'+key+'_global'] = int(round(cohort_means[key] * args.world_size))
                if record['episode_rollout/completed_games_global'] + record['episode_rollout/censored_games_global'] != args.n_envs * args.world_size:
                    raise RuntimeError('global full-game cohort size mismatch')
            if args.boss_opening_dataset:
                record['episode_scope'] = 'boss-opening'

            global_timesteps = int(record["global_timesteps"])
            if pure_ppo:
                model.ppo_rollouts_completed += 1
                record['rollout_round'] = model.ppo_rollouts_completed
            # Keep saved checkpoints/resumes on the global env-step axis rather
            # than SB3's local rank counter.
            model.num_timesteps = global_timesteps
            stop_requested = bool(average_scalar_dict(
                {'stop_requested': float(args.stop_file is not None and args.stop_file.exists())},
                ring_transport)['stop_requested'])
            if args.completed_game_budget is not None:
                stop_requested |= record['completed_full_games_global'] >= args.completed_game_budget
            save_due = bool(args.save_every_updates and (update + 1) % args.save_every_updates == 0)
            checkpoint_started = time.perf_counter()
            if args.output is not None and save_due:
                model.save(str(args.output.with_suffix("")))
            if (
                args.rank == 0
                and args.checkpoint_dir is not None
                and args.training_state_dir is None
                and save_due
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
            if args.training_state_dir is not None and (save_due or stop_requested):
                # Count-based retention bounds peak usage by pre-pruning.
                # Round anchors instead survive until the new common save.
                if not args.checkpoint_keep_round_multiple:
                    prune_bundles(
                        args.training_state_dir,
                        rank=args.rank,
                        keep=max(args.max_checkpoints - 1, 1),
                    )
                save_bundle(model, args.training_state_dir, rank=args.rank, world_size=args.world_size,
                    run_id=run_id, wandb_id=args.wandb_id, config=vars(args))
                # Round retention keeps the old common anchor/current until
                # every rank has published the replacement generation.
                if args.checkpoint_keep_round_multiple:
                    average_scalar_dict({'checkpoint_ready': 1.0}, ring_transport)
                prune_bundles(args.training_state_dir, rank=args.rank, keep=args.max_checkpoints,
                              round_multiple=args.checkpoint_keep_round_multiple)
                if not args.checkpoint_keep_round_multiple:
                    average_scalar_dict({'checkpoint_ready': 1.0}, ring_transport)
                last_bundle_step = int(model.num_timesteps)
            record['checkpoint_sec'] = time.perf_counter() - checkpoint_started
            if monitor is not None:
                if model.device.type == "cuda":
                    th.cuda.synchronize(model.device)
                resource_stack.close()
                record['iteration_resources'] = monitor.summary
            record['iteration_wall_sec'] = time.perf_counter() - started
            completed_updates += 1
            for key in timing_sums:
                timing_sums[key] += record[key]
            print(json.dumps(record, sort_keys=True), flush=True)
            if wandb_run is not None:
                import wandb

                wandb_payload = {
                        "global_step": global_timesteps,
                        "time/total_timesteps": global_timesteps,
                        "dist/rank0_total_sec": record["total_sec_rank"],
                        "dist/iteration_wall_sec": record["iteration_wall_sec"],
                        "dist/checkpoint_sec": record["checkpoint_sec"],
                        "dist/rollout_sec": record["rollout_sec"],
                        "dist/rollout_sec_max": record["rollout_sec_max"],
                        "dist/update_sec": record["update_sec"],
                        "dist/update_sec_max": record["update_sec_max"],
                        "speed/rollout_env_steps_per_second": record["rollout_steps_per_sec_global"],
                        "dist/leaf_graph_release_sec": record["leaf_graph_release_sec"],
                        "dist/backward_sec": record["backward_sec"],
                        "dist/sync_sec": record["sync_sec"],
                        "dist/gradient_pack_sec": record["gradient_pack_sec"],
                        "dist/avg_grad_sec": record["avg_grad_sec"],
                        "dist/optimizer_step_sec": record["optimizer_step_sec"],
                        "dist/sync_mbps": record["sync_mbps_mean"],
                        "dist/global_steps_per_second": (
                            record["transitions_global"]
                            / max(record["iteration_wall_sec"], 1e-9)
                        ),
                        "env_steps_per_second": (
                            record["transitions_global"]
                            / max(record["iteration_wall_sec"], 1e-9)
                        ),
                        "speed/env_steps_per_second": (
                            record["transitions_global"]
                            / max(record["iteration_wall_sec"], 1e-9)
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
                        "reasoning/loss": record["reasoning_loss"],
                        "reasoning/token_accuracy": record[
                            "reasoning_token_accuracy"
                        ],
                        "rule_prediction_accuracy": record[
                            "reasoning_token_accuracy"
                        ],
                        "reasoning/supervised_tokens": record[
                            "reasoning_supervised_tokens"
                        ],
                        "reasoning/sampled_samples": record[
                            "reasoning_sampled_samples"
                        ],
                        "reasoning/sampled_samples_total": record[
                            "reasoning_sampled_samples_total"
                        ],
                        "reasoning/batch_size": record["reasoning_batch_size"],
                        "reasoning/updates_requested": record[
                            "reasoning_updates_requested"
                        ],
                        "reasoning/gradient_accumulation_steps": record[
                            "reasoning_gradient_accumulation_steps"
                        ],
                        "reasoning/effective_batch_size": record[
                            "reasoning_effective_batch_size"
                        ],
                        "reasoning/samples": record["reasoning_samples"],
                        "reasoning/eligible_samples": record[
                            "reasoning_eligible_samples"
                        ],
                        "reasoning/dropped_too_long": record[
                            "reasoning_dropped_too_long"
                        ],
                        "reasoning/environment_tokens": record[
                            "reasoning_environment_tokens"
                        ],
                        "reasoning/atomic_trace_rate": record[
                            "reasoning_atomic_trace_rate"
                        ],
                        "reasoning/sequence_length_mean": record[
                            "reasoning_sequence_length_mean"
                        ],
                        "reasoning/sequence_length_max": record[
                            "reasoning_sequence_length_max"
                        ],
                        "reasoning/optimizer_steps": record[
                            "reasoning_optimizer_steps"
                        ],
                        "reasoning/compile_wait_sec": record[
                            "reasoning_compile_wait_sec"
                        ],
                        "reasoning/compute_sec": record[
                            "reasoning_compute_sec"
                        ],
                        "reasoning/step_sec": record[
                            "reasoning_step_sec"
                        ],
                        "reasoning/sync_sec": record["reasoning_sync_sec"],
                        "reasoning/sync_mbps": record[
                            "reasoning_sync_mbps_mean"
                        ],
                        "reasoning/total_sec": record["reasoning_total_sec"],
                        "reasoning/time_fraction": record[
                            "reasoning_time_fraction"
                        ],
                    }
                if update == 0 and startup_stagger:
                    wandb_payload.update({
                        f"startup_stagger/{key}": value
                        for key, value in startup_stagger.items()
                        if value is not None
                    })
                if update == 0 and startup_occupancy:
                    wandb_payload.update({
                        f"startup_occupancy/{key}": value
                        for key, value in startup_occupancy.items()
                    })
                if args.hierarchical_combat_hp:
                    wandb_payload.update({
                        "train/combat_value_loss": record[
                            "combat_value_loss_mean"
                        ],
                        "train/overworld_value_loss": record[
                            "overworld_value_loss_mean"
                        ],
                    })
                _add_game_metrics_to_wandb_payload(wandb_payload, record)
                if combat_strategy == 'small-playout':
                    # Ordinary PPO is disabled here; its placeholder batch is
                    # neither the compact learner nor the strategic full batch.
                    for key in ('train/effective_batch_size', 'train/effective_batch_size_global',
                                'train/gradient_accumulation_steps', 'train/optimizer_steps'):
                        wandb_payload.pop(key, None)
                    wandb_payload['game/completed_full_games_global'] = record['completed_full_games_global']
                    wandb_payload.update({'small_playout_rank0/'+key: value
                                          for key, value in record['small_playout_stats_rank'].items()})
                    wandb_payload['small_playout/turn_card_limit_decisions_global']=record['small_playout/turn_card_limit_decisions_global']
                wandb_payload.update({'upper/' + key: value for key, value in upper_update.items()})
                wandb_payload['training_scope'] = args.training_scope
                wandb_payload.update({'small_playout_train/' + key: value for key, value in small_update.items()})
                if joint_upper(model):
                    for key in ('upper/effective_batch_size_global', 'upper/micro_batch_size'):
                        wandb_payload[key] = record[key]
                wandb_payload.update({key: value for key, value in record.items()
                                      if key.startswith(('outer_return/', 'frozen_upper/', 'pure_rl/', 'deck_search/'))})
                if pure_ppo and 'completed_full_games_global' in record:
                    wandb_payload['game/completed_full_games_global'] = record['completed_full_games_global']
                wandb_payload.update({
                    key: record[key]
                    for key in _DISTRIBUTED_NOSL_METRIC_KEYS
                })
                if pure_ppo:
                    legacy_step = model.ppo_rollouts_completed
                elif episode_config is not None:
                    legacy_step = model.episode_rollouts_completed
                    wandb_payload['episode_rollout/round'] = legacy_step
                    wandb_payload['episode_rollout/games_global'] = args.n_envs * args.world_size
                    wandb_payload.update({k: v for k, v in record.items() if k.startswith('episode_rollout/')})
                else:
                    legacy_step = wandb_step_for_timesteps(
                        global_timesteps=global_timesteps,
                        global_rollout_size=global_rollout_size,
                        base_step=args.wandb_step_base,
                        base_timesteps=args.wandb_step_base_timesteps,
                        step_per_update=args.wandb_step_per_update,
                    )
                    if legacy_step is None:
                        # Use a monotonic global update index for fresh runs.  An
                        # explicit anchor still takes precedence above.
                        legacy_step = global_timesteps // global_rollout_size
                # W&B's internal _step is append-only and cannot be moved
                # backwards when resuming an existing run. A custom step
                # metric preserves the legacy chart axis without changing
                # the real global environment-step counters.
                wandb_payload["legacy_step"] = int(legacy_step)
                wandb.log(wandb_payload)

            if args.iteration_timeout_seconds:
                print('ROUND_COMPLETE ' + json.dumps(dict(global_step=global_timesteps,
                    round=getattr(model, 'episode_rollouts_completed', completed_updates),
                    wall_seconds=time.perf_counter()-started,
                    hard_limit_seconds=args.iteration_timeout_seconds)), flush=True)
            iteration_budget.cancel()
            if stop_requested:
                print(f'[rank {args.rank}] coordinated stop after {global_timesteps} global steps', flush=True)
                break

        if args.output is not None:
            model.save(str(args.output.with_suffix("")))
        # A boundary save already published this complete generation. Pruning
        # before another idempotent save would discard a fallback without
        # creating a replacement, shrinking retention on every clean exit.
        if args.training_state_dir is not None and last_bundle_step != int(model.num_timesteps):
            if not args.checkpoint_keep_round_multiple:
                prune_bundles(
                    args.training_state_dir,
                    rank=args.rank,
                    keep=max(args.max_checkpoints - 1, 1),
                )
            save_bundle(model, args.training_state_dir, rank=args.rank, world_size=args.world_size,
                run_id=run_id, wandb_id=args.wandb_id, config=vars(args))
            if args.checkpoint_keep_round_multiple:
                average_scalar_dict({'checkpoint_ready': 1.0}, ring_transport)
            prune_bundles(args.training_state_dir, rank=args.rank, keep=args.max_checkpoints,
                          round_multiple=args.checkpoint_keep_round_multiple)
        if args.rank == 0 and args.checkpoint_dir is not None and args.training_state_dir is None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            final_step = cumulative_global_steps
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
    except BaseException as error:
        training_failed = True
        save_failure_scene(model, args.training_state_dir, rank=args.rank, error=error)
        raise
    finally:
        iteration_budget.cancel()
        try:
            resource_stack.close()
        finally:
            callback.on_training_end()
            if ring_transport is not None:
                ring_transport.close()
            env.close()
            if wandb_run is not None:
                wandb_run.finish(exit_code=1 if training_failed else 0)

    summary = {
        "rank": args.rank,
        "world_size": args.world_size,
        "updates": completed_updates,
        "updates_requested": None if args.continuous else args.updates,
        "continuous": args.continuous,
        **{'mean_' + key: value / max(completed_updates, 1)
           for key, value in timing_sums.items()},
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
