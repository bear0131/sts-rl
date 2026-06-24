"""Train MaskablePPO on the whole-run SlayTheSpireEnv.

Run from repo root:

    python -m rl.train
    python -m rl.train --steps 1000000 --n-envs 8
    python -m rl.train --tensorboard
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch.nn as nn

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from env.core.run_env import SlayTheSpireEnv
from rl.model import StSMLPExtractor, StSTransformerExtractor
from rl.pointer_policy import StSPointerPolicy


class StSMetricsCallback(BaseCallback):
    """Log run-level metrics over a rolling window of finished episodes.

    Reads each episode boundary from the `info` dict: VecMonitor injects
    `info["episode"]` (reward `r`, length `l`) on episode end, and our env
    surfaces ground-truth game state in `info["run_stats"]` (see
    run_env.SlayTheSpireEnv._info). Emits under:

      rollout/  ep_rew_mean, ep_len_mean   (generic: reward + steps/episode)
      game/     win_rate, floor_mean, act_mean, deck_size_mean
      time/     elapsed_seconds            (GLOBAL wall-clock since learn(),
                                            not a per-episode delta)
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._rew = deque(maxlen=window)
        self._len = deque(maxlen=window)
        self._win = deque(maxlen=window)
        self._floor = deque(maxlen=window)
        self._act = deque(maxlen=window)
        self._deck = deque(maxlen=window)
        self._n_episodes = 0
        self._t0 = None

    def _on_training_start(self) -> None:
        self._t0 = time.monotonic()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue  # not an episode boundary for this sub-env
            self._n_episodes += 1
            self._rew.append(ep["r"])
            self._len.append(ep["l"])

            stats = info.get("run_stats")
            if stats is not None:
                won = stats["outcome"] == "PLAYER_VICTORY"
                self._win.append(1.0 if won else 0.0)
                self._floor.append(stats["floor"])
                self._act.append(stats["act"])
                self._deck.append(stats["deck_size"])
        return True

    def _on_rollout_end(self) -> None:
        # Record once per rollout (not per step) — these are rolling means, so
        # there is no value in re-logging the same window every transition.
        def mean(buf):
            return sum(buf) / len(buf) if buf else 0.0

        if self._rew:
            self.logger.record("rollout/ep_rew_mean", mean(self._rew))
            self.logger.record("rollout/ep_len_mean", mean(self._len))
        if self._win:
            self.logger.record("game/win_rate", mean(self._win))
            self.logger.record("game/floor_mean", mean(self._floor))
            self.logger.record("game/act_mean", mean(self._act))
        if self._deck:
            self.logger.record("game/deck_size_mean", mean(self._deck))
        self.logger.record("game/episodes_total", self._n_episodes)
        if self._t0 is not None:
            self.logger.record("time/elapsed_seconds", time.monotonic() - self._t0)


def make_env(*, ascension: int):
    """Return one shaped whole-run environment for DummyVecEnv."""

    from rl.reward import ShapedRewardWrapper

    def _thunk():
        env = SlayTheSpireEnv(ascension=ascension)
        env = ShapedRewardWrapper(env)
        return env

    return _thunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO on whole-run SlayTheSpireEnv."
    )

    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)

    parser.add_argument(
        "--vec",
        choices=["subproc", "dummy"],
        default="subproc",
        help="VecEnv backend. subproc = one process per env (true multi-core "
             "sampling; default). dummy = all envs in this process, serial "
             "(single core — use only for debugging).",
    )

    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")

    parser.add_argument(
        "--arch",
        choices=["mlp-flat", "transformer-flat", "transformer-pointer"],
        default="transformer-flat",
        help="Policy architecture, named <encoder>-<head>. The encoder maps obs to "
             "token/feature reps; the head maps those to the 149 action logits. "
             "mlp-flat = flat Linear extractor + flat head. transformer-flat = "
             "per-entity Transformer encoder + flat head (default). "
             "transformer-pointer = same Transformer encoder + per-entity POINTER "
             "head (only the head differs from transformer-flat; for the head A/B). "
             "New heads append as transformer-<head> (e.g. transformer-hier).",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)

    parser.add_argument(
        "--out",
        default="rl/runs/whole-run",
        help="Base directory; each run lands in <out>/<timestamp>/ "
             "(checkpoints, final model, optional TensorBoard logs).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100_000,
        help="Save a checkpoint about every N environment transitions.",
    )
    parser.add_argument("--tensorboard", action="store_true")

    parser.add_argument(
        "--metrics-window",
        type=int,
        default=100,
        help="Rolling window (episodes) for game/* and rollout/* metrics.",
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-grad-freq", type=int, default=0)
    parser.add_argument("--wandb-save-freq", type=int, default=50_000)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            "batch_size must divide n_envs * n_steps: "
            f"{args.batch_size} does not divide {rollout_size}"
        )

    # Unique run id (start timestamp) so each run gets its own out_dir and
    # wandb name — concurrent/repeated runs never clobber each other's
    # checkpoints, monitor.csv, or TensorBoard logs.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    out_dir = Path(args.out) / run_id
    checkpoint_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(ascension=args.ascension)
        for _ in range(args.n_envs)
    ]
    if args.vec == "subproc":
        # forkserver, not the default fork: forking a process that already
        # imported the native sts module + torch (threads, CUDA context) into a
        # child can deadlock. forkserver starts each worker from a clean-ish
        # process. spawn would also work but is slower to start.
        venv = SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=str(out_dir / "monitor.csv"))
    venv.seed(args.seed)

    # Callback runs once per vectorized step, so adjust for n_envs.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="maskable_ppo",
    )

    tag = f"whole-run-asc{args.ascension}-s{args.seed}-{run_id}"

    callbacks = [checkpoint_callback, StSMetricsCallback(window=args.metrics_window)]
    wandb_run = None

    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        config = vars(args).copy()
        config["tag"] = tag

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or tag,
            config=config,
            sync_tensorboard=True,
            save_code=True,
        )

        callbacks.append(
            WandbCallback(
                gradient_save_freq=args.wandb_grad_freq,
                model_save_path=str(out_dir / "wandb_models" / wandb_run.id),
                model_save_freq=args.wandb_save_freq,
                verbose=2,
            )
        )

    callback = CallbackList(callbacks)

    # sync_tensorboard needs SB3 to actually emit TensorBoard logs, so enable
    # the TensorBoard sink whenever either --tensorboard or --wandb is set.
    tensorboard_log = (
        str(out_dir / "tensorboard")
        if (args.tensorboard or args.wandb)
        else None
    )

    # Categorical id groups (card/monster/move/relic/... vocab indices) are
    # auto-detected from the observation space by the feature extractor — integer-
    # dtype Boxes are categorical, float Boxes are scalars (see model.py /
    # encode.obs_space()). No hand-maintained key list to drift out of sync.
    if args.arch == "transformer-pointer":
        # Pointer action head. The core owns the network (transformer encoder +
        # pointer/flat heads + value path), so no features_extractor/net_arch here
        # — only the encoder geometry. Same encoder geometry as transformer-flat so
        # the A/B isolates the action head.
        policy = StSPointerPolicy
        policy_kwargs = dict(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
    elif args.arch == "transformer-flat":
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=StSTransformerExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
            ),
            activation_fn=nn.GELU,
            net_arch=dict(
                pi=[256, 128],
                vf=[512, 256],
            ),
            ortho_init=False,
        )
    else:
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=StSMLPExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
            ),
            activation_fn=nn.GELU,
            net_arch=dict(
                pi=[256, 128],
                vf=[512, 256],
            ),
            ortho_init=False,
        )

    model = MaskablePPO(
        policy=policy,
        env=venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
    )

    print(
        "[train] whole-run "
        f"steps={args.steps} "
        f"n_envs={args.n_envs} "
        f"vec={args.vec} "
        f"rollout={rollout_size} "
        f"ascension={args.ascension}"
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            progress_bar=False,
            tb_log_name=tag,
        )

        final_path = out_dir / "final"
        model.save(str(final_path))
        print(f"[train] saved {final_path}.zip")
    except KeyboardInterrupt:
        # Ctrl+C: model.learn() bails out here. Save what we have — otherwise
        # we'd lose everything since the last periodic checkpoint.
        interrupted_path = out_dir / "interrupted"
        model.save(str(interrupted_path))
        print(f"[train] interrupted — saved {interrupted_path}.zip")
    finally:
        venv.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
