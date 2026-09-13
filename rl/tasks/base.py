"""Task interface for the unified trainer (rl.training.trainer).

A *training* (the PPO algorithm + all its infrastructure: VecEnv, checkpoints,
wandb, resume/init-from, the learn() loop) is one axis; a *task* (which env to
build, which reward, which metrics to watch, which policy geometry) is the
orthogonal axis. rl.training.trainer owns the first; a Task owns the second.

To add a new task: subclass Task, implement the five hooks, and register it in
rl/tasks/__init__.py. The trainer gains `--task <name>` automatically; every
shared PPO hyperparameter is already configurable from the CLI.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any, Callable


class Task(ABC):
    """One training task. See module docstring for the trainer/task split.

    Subclasses set the three class attributes and implement the five hooks.
    Everything else (PPO build, VecEnv, callbacks, wandb, resume/init-from,
    the absolute-``--steps`` learn loop) is shared and lives in rl.training.trainer.
    """

    #: CLI selector (``--task <name>``) and identity used in logs.
    name: str
    #: Checkpoint filename prefix (CheckpointCallback name_prefix).
    ckpt_prefix: str
    #: Per-task overrides for shared-arg defaults, applied via
    #: ``parser.set_defaults(**defaults)`` so an explicit CLI flag still wins.
    #: Typical keys: ``steps``, ``metrics_window``, ``out``.
    defaults: dict[str, Any] = {}

    # --- hooks ---------------------------------------------------------------

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Register task-specific CLI args (e.g. combat's floor range, whole-run's
        --arch). Shared PPO/infra args are added by the trainer. Default: none."""

    @abstractmethod
    def make_env_fn(self, args: argparse.Namespace) -> Callable[[], Any]:
        """Return a *thunk* ``() -> gym.Env`` for one VecEnv worker. Reads
        whatever it needs (ascension, floor range, ...) off ``args``."""

    @abstractmethod
    def metrics_callback(self, args: argparse.Namespace):
        """Return the BaseCallback that logs this task's headline metrics."""

    @abstractmethod
    def build_policy(self, args: argparse.Namespace) -> tuple[Any, dict]:
        """Return ``(policy, policy_kwargs)`` for MaskablePPO on a *fresh* run.
        On --resume the policy is restored from the zip and this is unused."""

    @abstractmethod
    def tag(self, args: argparse.Namespace, run_id: str) -> str:
        """Return the run tag (wandb name / tensorboard log name / dir stem)."""
