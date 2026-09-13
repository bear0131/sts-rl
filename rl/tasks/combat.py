"""Combat-gauntlet pretraining task.

Stage 1 of the combat-pretrain plan (docs/archive/doc/night_0625/combat_pretrain_decision.md):
each episode is ONE random loadout (random Ironclad deck, relics, starting HP)
fighting an ENDLESS stream of random encounters; HP carries over with no healing,
so the episode ends on death. Reward is +1 per fight cleared by default;
``--combat-hp-loss-coef`` optionally subtracts a net-HP-loss penalty when each
fight ends. The network is the combat specialist used inside the whole-run
split policy, on the same obs/action space, so its shared tokenizer, combat
encoder, policy head, and value head can warm-start global training via
``python -m rl.training.trainer --task wholerun --init-from <combat>.zip --freeze-encoder``.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback

from env.core import factories
from env.core.run_env import SlayTheSpireEnv
from env.core.combat_backend import SimCombatBackend
from rl.training.reward import CombatGauntletRewardWrapper
from rl.policies.split_policy import CombatStSPointerPolicy
from rl.tasks.base import Task


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
        self._hp_lost = deque(maxlen=window)
        self._hp_penalty = deque(maxlen=window)
        self._n = 0
        self._t0 = None
        self._last_speed_t = None
        self._last_speed_steps = 0
        self._start_steps = 0

    def _on_training_start(self) -> None:
        self._t0 = time.monotonic()
        self._last_speed_t = self._t0
        self._last_speed_steps = int(self.num_timesteps)
        self._start_steps = int(self.num_timesteps)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "combat_hp_lost" in info:
                self._hp_lost.append(float(info["combat_hp_lost"]))
                self._hp_penalty.append(float(info["combat_hp_penalty"]))
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
        if self._hp_lost:
            self.logger.record("combat/hp_lost_per_combat", mean(self._hp_lost))
            self.logger.record(
                "combat/hp_penalty_per_combat",
                mean(self._hp_penalty),
            )
        self.logger.record("combat/episodes_total", self._n)
        if self._t0 is not None:
            now = time.monotonic()
            elapsed = now - self._t0
            self.logger.record("time/elapsed_seconds", elapsed)
            trained_steps = int(self.num_timesteps) - self._start_steps
            if elapsed > 0:
                self.logger.record(
                    "speed/env_steps_per_second_lifetime",
                    trained_steps / elapsed,
                )
            if self._last_speed_t is not None:
                dt = now - self._last_speed_t
                d_steps = int(self.num_timesteps) - self._last_speed_steps
                if dt > 0:
                    self.logger.record(
                        "speed/env_steps_per_second",
                        d_steps / dt,
                    )
            self._last_speed_t = now
            self._last_speed_steps = int(self.num_timesteps)


class CombatTask(Task):
    name = "combat"
    ckpt_prefix = "combat_ppo"
    defaults = {
        "steps": 2_000_000,
        "metrics_window": 200,
        "out": "runs/combat",
        "n_layers": 4,
    }

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--floor-lo", type=int, default=1)
        parser.add_argument("--floor-hi", type=int, default=50)
        parser.add_argument(
            "--combat-hp-loss-coef",
            type=float,
            nargs="?",
            const=0.02,
            default=0.0,
            help="Subtract coefficient * net HP lost when each combat ends. "
            "Passing the flag without a value uses the recommended 0.02; "
            "default 0 keeps the original +1-per-win reward.",
        )

    def make_env_fn(self, args):
        floor_lo, floor_hi, ascension = args.floor_lo, args.floor_hi, args.ascension
        if args.combat_hp_loss_coef < 0:
            raise ValueError("--combat-hp-loss-coef must be non-negative")

        def _builder(np_random):
            return factories.random_gauntlet_factory(
                np_random, ascension=ascension, floor_range=(floor_lo, floor_hi)
            )

        def _thunk():
            env = SlayTheSpireEnv(
                backend_factory=lambda: SimCombatBackend(_builder, ascension=ascension),
                card_language=args.card_language,
                combat_language=args.combat_language,
            )
            env = CombatGauntletRewardWrapper(
                env,
                hp_loss_coef=args.combat_hp_loss_coef,
            )
            env = TimeLimit(env, max_episode_steps=_MAX_EPISODE_STEPS)
            return env

        return _thunk

    def metrics_callback(self, args):
        return CombatMetricsCallback(window=args.metrics_window)

    def build_policy(self, args):
        return CombatStSPointerPolicy, dict(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            card_language=args.card_language,
            combat_language=args.combat_language,
        )

    def tag(self, args, run_id):
        tag = (
            f"combat-f{args.floor_lo}-{args.floor_hi}-asc{args.ascension}"
            f"-s{args.seed}-{run_id}"
        )
        if args.combat_hp_loss_coef:
            tag += f"-hploss{args.combat_hp_loss_coef:g}"
        if args.card_language:
            tag += "-cardlang"
        if args.combat_language:
            tag += "-combatlang"
        return tag
