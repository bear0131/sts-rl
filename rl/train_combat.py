"""Pretrain the combat head on a random-combat GAUNTLET (SimCombatBackend).

Stage 1 of the combat-pretrain plan (doc/night_0625/combat_pretrain_decision.md):
each episode is ONE random loadout (random Ironclad deck, relics, starting HP)
fighting an ENDLESS stream of random encounters; HP carries over with no healing,
so the episode ends on death. Reward is +1 per fight cleared, nothing else
(CombatGauntletRewardWrapper) — dying's cost is the future clears it forfeits, so
HP value is learned implicitly. The network is the SAME StSPointerPolicy used by
the whole-run train.py, on the SAME obs/action space — so the resulting weights
warm-start global training via `python -m rl.train --init-from <combat>.zip
--freeze-encoder`.

Run from repo root:

    python -m rl.train_combat
    python -m rl.train_combat --steps 2000000 --n-envs 8 --pool normal
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from gymnasium.wrappers import TimeLimit
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

from env.core import factories
from env.core.run_env import SlayTheSpireEnv
from env.core.combat_backend import SimCombatBackend
from rl.reward import CombatGauntletRewardWrapper
from rl.pointer_policy import StSPointerPolicy


# A gauntlet episode spans MANY fights (HP attrition with no healing ends it by
# death well before this), so this cap is loose — it only guards a pathological
# non-terminating single-combat combo loop (0-reward infinite combo).
_MAX_EPISODE_STEPS = 4000


# Survival-curve thresholds: fraction of episodes that cleared at least x fights.
# Guesses for now (we don't yet know the mean) — bump if it routinely exceeds 18.
_CLEAR_THRESHOLDS = (3, 6, 9, 12, 15, 18)


class CombatMetricsCallback(BaseCallback):
    """Combats-cleared-per-episode + reward over a rolling window of finished
    gauntlet episodes. Reads info["combats_won"] / info["combat_act"] (set by
    CombatGauntletRewardWrapper when an episode ends) and VecMonitor's
    info["episode"]. Also logs per-act mean clears and a P(cleared >= x) curve."""

    def __init__(self, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self._rew = deque(maxlen=window)
        self._len = deque(maxlen=window)
        self._cleared = deque(maxlen=window)
        self._cleared_act = {a: deque(maxlen=window) for a in (1, 2, 3)}
        self._n = 0
        self._t0 = None

    def _on_training_start(self) -> None:
        self._t0 = time.monotonic()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            self._n += 1
            self._rew.append(ep["r"])
            self._len.append(ep["l"])
            won = float(info.get("combats_won", 0))
            self._cleared.append(won)
            act = info.get("combat_act")
            if act in self._cleared_act:
                self._cleared_act[act].append(won)
        return True

    def _on_rollout_end(self) -> None:
        def mean(buf):
            return sum(buf) / len(buf) if buf else 0.0

        if self._rew:
            self.logger.record("rollout/ep_rew_mean", mean(self._rew))
            self.logger.record("rollout/ep_len_mean", mean(self._len))
            self.logger.record("combat/cleared_per_episode", mean(self._cleared))
            # mean clears split by the act of the episode's level L.
            for act, buf in self._cleared_act.items():
                if buf:
                    self.logger.record(f"combat/cleared_act{act}", mean(buf))
            # survival curve: fraction of episodes that cleared at least x fights.
            n = len(self._cleared)
            for x in _CLEAR_THRESHOLDS:
                self.logger.record(
                    f"combat/clear_ge_{x}",
                    sum(1 for c in self._cleared if c >= x) / n,
                )
        self.logger.record("combat/episodes_total", self._n)
        if self._t0 is not None:
            self.logger.record("time/elapsed_seconds", time.monotonic() - self._t0)


def make_env(*, floor_lo: int, floor_hi: int, ascension: int):
    """Return one randomized combat-gauntlet env for the VecEnv: the whole-run thin
    SlayTheSpireEnv on a SimCombatBackend (so it speaks the SAME 149-action space
    as rl.train), +1-per-clear reward, step-capped."""

    def _builder(np_random):
        return factories.random_gauntlet_factory(
            np_random, ascension=ascension, floor_range=(floor_lo, floor_hi)
        )

    def _thunk():
        env = SlayTheSpireEnv(
            backend_factory=lambda: SimCombatBackend(_builder, ascension=ascension)
        )
        env = CombatGauntletRewardWrapper(env)
        env = TimeLimit(env, max_episode_steps=_MAX_EPISODE_STEPS)
        return env

    return _thunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain combat head on SimCombatEnv.")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--vec", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--floor-lo", type=int, default=1)
    parser.add_argument("--floor-hi", type=int, default=50)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--out", default="rl/runs/combat")
    parser.add_argument("--save-every", type=int, default=100_000)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--metrics-window", type=int, default=200)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-save-freq", type=int, default=50_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            f"batch_size must divide n_envs * n_steps: {args.batch_size} does not "
            f"divide {rollout_size}"
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / run_id
    checkpoint_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(floor_lo=args.floor_lo, floor_hi=args.floor_hi,
                 ascension=args.ascension)
        for _ in range(args.n_envs)
    ]
    if args.vec == "subproc":
        venv = SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=str(out_dir / "monitor.csv"))
    venv.seed(args.seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="combat_ppo",
    )

    tag = f"combat-f{args.floor_lo}-{args.floor_hi}-asc{args.ascension}-s{args.seed}-{run_id}"
    callbacks = [checkpoint_callback, CombatMetricsCallback(window=args.metrics_window)]
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
                model_save_path=str(out_dir / "wandb_models" / wandb_run.id),
                model_save_freq=args.wandb_save_freq,
                verbose=2,
            )
        )

    callback = CallbackList(callbacks)
    tensorboard_log = (
        str(out_dir / "tensorboard") if (args.tensorboard or args.wandb) else None
    )

    model = MaskablePPO(
        policy=StSPointerPolicy,
        env=venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.01,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        ),
    )

    print(
        f"[train_combat] steps={args.steps} n_envs={args.n_envs} vec={args.vec} "
        f"rollout={rollout_size} floors={args.floor_lo}-{args.floor_hi} "
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
        print(f"[train_combat] saved {final_path}.zip")
    except KeyboardInterrupt:
        interrupted_path = out_dir / "interrupted"
        model.save(str(interrupted_path))
        print(f"[train_combat] interrupted — saved {interrupted_path}.zip")
    finally:
        venv.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
