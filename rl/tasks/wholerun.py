"""Whole-run training task: MaskablePPO on the full 3-act SlayTheSpireEnv."""

from __future__ import annotations

import argparse
import time
from collections import deque

import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback

from env.core.run_env import SlayTheSpireEnv
from rl.model import StSMLPExtractor, StSTransformerExtractor
from rl.pointer_policy import StSPointerPolicy
from rl.tasks.base import Task


# Hard per-episode step cap (safety net for bug #1's infinite-combat / pathological
# loops). A legitimate full 3-act run is on the order of low-thousands of THIN-env
# steps — each targeted card play is 2 steps and each combat MULTI toggle is its own
# step — so test_game uses a 2000-step cap on the THICK (one-step-per-decision)
# layer and test_run_v2 a 5000-step "did not terminate" tripwire on the THIN layer.
# 6000 sits just above that thin-layer tripwire with comfortable headroom: a genuine
# win never reaches it, but no episode (infinite 0-reward combat combo, etc.) can run
# forever and poison the rollout buffer with one dead sub-env. TimeLimit sets
# truncated=True; VecMonitor's autoreset then resets the sub-env on the next step. It
# wraps OUTSIDE the env (and OUTSIDE ShapedRewardWrapper is fine — TimeLimit counts
# step() calls, which are exactly the steps the policy takes).
_MAX_EPISODE_STEPS = 6000


class StSMetricsCallback(BaseCallback):
    """Log run-level metrics over a rolling window of finished episodes.

    Reads each episode boundary from the `info` dict: VecMonitor injects
    `info["episode"]` (reward `r`, length `l`) on episode end, and our env
    surfaces ground-truth game state in `info["run_stats"]` (see
    run_env.SlayTheSpireEnv._info). Emits under:

      rollout/  ep_rew_mean, ep_len_mean   (generic: reward + steps/episode)
      game/     win_rate, floor_mean, act_{1,2,3}_won_mean, deck_size_mean
      game/floor_won/  {05,10,...,50}: pass-rate past floor x (detail section)
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
                # A win = cleared the run (act 3 boss bumps act to 4, leaving
                # outcome UNDECIDED), mirroring SimGameEnv.is_over (act > 3).
                # PLAYER_VICTORY only fires at the act-4 heart, which we never
                # reach — so testing it alone made win_rate stay 0.
                won = stats["act"] > 3 or stats["outcome"] == "PLAYER_VICTORY"
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
            # Per-act clear rate: clearing act x bumps act to x+1, so "won act x"
            # == final act > x. act_3 ~= win_rate; the ladder shows where runs die.
            for x in (1, 2, 3):
                self.logger.record(
                    f"game/act_{x}_won_mean",
                    mean([1.0 if a > x else 0.0 for a in self._act]),
                )
        if self._deck:
            self.logger.record("game/deck_size_mean", mean(self._deck))
        self.logger.record("game/episodes_total", self._n_episodes)
        # Per-5-floor pass-rate detail (fraction that got past floor x, final
        # floor > x). Kept under its own game/floor_won/ sub-section so the ten
        # curves cluster at the end instead of interleaving the headline metrics.
        if self._floor:
            for t in range(5, 55, 5):
                self.logger.record(
                    f"game/floor_won/{t:02d}",
                    mean([1.0 if f > t else 0.0 for f in self._floor]),
                )
        if self._t0 is not None:
            self.logger.record("time/elapsed_seconds", time.monotonic() - self._t0)


class WholeRunTask(Task):
    name = "wholerun"
    ckpt_prefix = "maskable_ppo"
    defaults = {
        "steps": 100_000,
        "metrics_window": 100,
        "out": "rl/runs/whole-run",
    }

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--global-map-view",
            action="store_true",
            help="Expose the complete current-act map DAG to the policy. Adds "
                 "105 room tokens with coordinates, edges, and reachability; "
                 "off by default for checkpoint compatibility.",
        )
        parser.add_argument(
            "--map-dag",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use the topology-aware DAG encoder for the full map. The old "
                 "flat map encoder remains available with --no-map-dag.",
        )
        parser.add_argument(
            "--deck-size-observation",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Add the normalized total held-card count to the global token.",
        )
        parser.add_argument(
            "--arch",
            choices=[
                "auto", "mlp-flat", "transformer-flat", "transformer-pointer",
            ],
            default="auto",
            help="Policy architecture, named <encoder>-<head>. The encoder maps obs "
                 "to token/feature reps; the head maps those to the 149 action "
                 "logits. auto = transformer-flat for a fresh run, or infer the "
                 "architecture from --init-from. mlp-flat = flat Linear "
                 "extractor + flat head. "
                 "transformer-flat = per-entity Transformer encoder + flat head "
                 "(default). transformer-pointer = same Transformer encoder + "
                 "per-entity POINTER head (only the head differs from "
                 "transformer-flat; for the head A/B). New heads append as "
                 "transformer-<head> (e.g. transformer-hier).",
        )

    def make_env_fn(self, args):
        from rl.reward import ShapedRewardWrapper

        ascension = args.ascension

        def _thunk():
            env = SlayTheSpireEnv(
                ascension=ascension,
                global_map_view=args.global_map_view,
                card_language=args.card_language,
                combat_language=args.combat_language,
                deck_size_observation=args.deck_size_observation,
            )
            env = ShapedRewardWrapper(env)
            # TimeLimit last so it sees the same steps the policy takes and its
            # truncated=True passes straight through ShapedRewardWrapper.step
            # (which forwards truncated untouched).
            env = TimeLimit(env, max_episode_steps=_MAX_EPISODE_STEPS)
            return env

        return _thunk

    def metrics_callback(self, args):
        return StSMetricsCallback(window=args.metrics_window)

    def build_policy(self, args):
        # Categorical id groups (card/monster/move/relic/... vocab indices) are
        # auto-detected from the observation space by the feature extractor —
        # integer-dtype Boxes are categorical, float Boxes are scalars (see
        # model.py / encode.obs_space()). No hand-maintained key list to drift.
        if args.arch == "auto":
            raise RuntimeError(
                "trainer must resolve --arch auto before building the policy"
            )
        if args.arch == "transformer-pointer":
            # Pointer action head. The core owns the network (transformer encoder
            # + pointer/flat heads + value path), so no features_extractor/net_arch
            # here — only the encoder geometry. Same encoder geometry as
            # transformer-flat so the A/B isolates the action head.
            return StSPointerPolicy, dict(
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                card_language=args.card_language,
                combat_language=args.combat_language,
                map_dag=args.map_dag,
            )
        elif args.arch == "transformer-flat":
            return "MultiInputPolicy", dict(
                features_extractor_class=StSTransformerExtractor,
                features_extractor_kwargs=dict(
                    features_dim=256,
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    card_language=args.card_language,
                    combat_language=args.combat_language,
                    map_dag=args.map_dag,
                ),
                activation_fn=nn.GELU,
                net_arch=dict(pi=[256, 128], vf=[512, 256]),
                ortho_init=False,
            )
        else:
            if args.card_language or args.combat_language:
                raise ValueError(
                    "--card-language/--combat-language require a transformer architecture"
                )
            return "MultiInputPolicy", dict(
                features_extractor_class=StSMLPExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                activation_fn=nn.GELU,
                net_arch=dict(pi=[256, 128], vf=[512, 256]),
                ortho_init=False,
            )

    def tag(self, args, run_id):
        map_tag = "-globalmap" if args.global_map_view else ""
        if args.global_map_view and args.map_dag:
            map_tag += "-dag"
        card_tag = "-cardlang" if args.card_language else ""
        if args.combat_language:
            card_tag += "-combatlang"
        deck_tag = "-decksize" if args.deck_size_observation else ""
        return (
            f"whole-run{map_tag}{card_tag}{deck_tag}-asc{args.ascension}"
            f"-s{args.seed}-{run_id}"
        )
