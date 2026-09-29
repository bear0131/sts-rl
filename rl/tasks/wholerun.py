"""Whole-run training task: MaskablePPO on the full 3-act SlayTheSpireEnv."""

from __future__ import annotations

import argparse
import time
from collections import deque

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback

from env.core.run_env import SlayTheSpireEnv
from rl.policies.split_policy import SplitStSPointerPolicy
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
        self._excluded_startup_episodes = 0
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
            ep = info.get("episode")
            if ep is None:
                continue  # not an episode boundary for this sub-env
            if info.get("startup_occupancy_incumbent_episode"):
                self._excluded_startup_episodes += 1
                continue
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
        self.logger.record("game/episodes_window", len(self._win))
        self.logger.record(
            "game/startup_episodes_excluded_total",
            self._excluded_startup_episodes,
        )
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


class WholeRunTask(Task):
    name = "wholerun"
    ckpt_prefix = "maskable_ppo"
    defaults = {
        "steps": 100_000,
        "metrics_window": 100,
        "out": "runs/whole-run",
        "n_layers": 8,
    }

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--combat-action-contract", choices=("legacy", "joint-v2"),
                            default=None, help="Complete combat policy; migrate old weights with --init-from.")
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
            "--neow-observation",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Expose the four Neow blessing choices as option tokens.",
        )
        parser.add_argument(
            "--event-option-observation",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Expose normal event legal choices as action-aligned option tokens.",
        )
        parser.add_argument(
            "--arch",
            choices=["auto", "split-pointer", "semantic-reasoning"],
            default="auto",
            help="Independent combat and overworld Transformer actor-critics "
                 "with shared card/relic/potion tokenizers. --n-layers is the "
                 "combined layer budget split between both specialists.",
        )
        parser.add_argument(
            "--interactive-reasoning",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Capture post-action simulator traces and train the anonymous "
                 "autoregressive reasoning decoder. Enabled automatically by "
                 "--arch semantic-reasoning.",
        )
        parser.add_argument("--reasoning-coef", type=float, default=0.2)
        parser.add_argument("--reasoning-batch-size", type=int, default=16)
        parser.add_argument("--reasoning-updates", type=int, default=1)
        parser.add_argument("--reasoning-layers", type=int, default=4)
        parser.add_argument("--card-program-layers", type=int, default=2)
        parser.add_argument(
            "--card-vector-checkpoint",
            default=None,
            help="Load the standalone v7 anonymous card vectors and freeze the "
                 "card-rule encoder for non-reasoning split-pointer wholerun "
                 "training. The checkpoint must contain card_encoder weights.",
        )
        parser.add_argument(
            "--max-reasoning-tokens",
            type=int,
            default=None,
            help="Deprecated initial learned-position capacity; reasoning "
            "transcripts are no longer length-limited.",
        )
        parser.add_argument(
            "--hierarchical-combat-hp",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Train combat as a fight-local option whose terminal reward "
                 "is remaining HP on the fixed observation scale. Overworld "
                 "returns skip combat and connect directly to the post-combat "
                 "state. On resume, omission preserves the checkpoint mode.",
        )
        parser.add_argument(
            "--nosl-is-az-boss-transition-target",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Opt in to post-transition act-boss terminal HP targets; "
                 "omission preserves the checkpoint setting on resume.",
        )
        parser.add_argument(
            "--combat-hp-gamma",
            type=float,
            default=None,
            help="Fight-local discount for hierarchical combat-HP mode "
                 "(default 1.0, or the checkpoint value on resume).",
        )
        parser.add_argument(
            "--combat-hp-gae-lambda",
            type=float,
            default=None,
            help="Fight-local GAE lambda (default 0.95, or the checkpoint "
            "value on resume).",
        )
        parser.add_argument(
            "--nosl-is-az",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Replace combat PPO updates and combat action sampling with "
                 "worker-side NOSL information-set PUCT targets. Overworld "
                 "continues to use PPO.",
        )
        parser.add_argument(
            "--nosl-is-az-inner-only",
            action="store_true",
            help="Freeze the shared and overworld policy and skip overworld "
                 "PPO updates; train only the combat specialist from search targets.",
        )
        parser.add_argument(
            "--nosl-is-az-cpp",
            action="store_true",
            help="Use the native C++ NOSL combat search/simulator while keeping "
                 "the Python whole-run learner and HP-labelled inner update.",
        )
        parser.add_argument(
            "--nosl-is-az-reset-combat",
            action="store_true",
            help="With --init-from, restore the trained outer policy but keep "
                 "the freshly initialized combat specialist.",
        )
        parser.add_argument("--nosl-is-az-simulations", type=int, default=8)
        parser.add_argument(
            "--nosl-is-az-depth",
            type=int,
            default=3,
            help="Maximum search depth; 0 removes the artificial depth cutoff "
                 "and expands until terminal/new-leaf states.",
        )
        parser.add_argument(
            "--nosl-is-az-end-turn-as-leaf",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Stop after End Turn and evaluate the resulting state with value only.",
        )
        parser.add_argument(
            "--nosl-is-az-rollout-steps",
            type=int,
            default=0,
            help="Deprecated compatibility option; AlphaZero-style NOSL uses "
                 "the value network at the leaf and performs no rollout.",
        )
        parser.add_argument("--nosl-is-az-c-puct", type=float, default=1.5)
        parser.add_argument("--nosl-is-az-temperature", type=float, default=1.0)
        parser.add_argument(
            "--nosl-is-az-prior-temperature",
            type=float,
            default=1.0,
            help="Temperature applied to policy logits before native search.",
        )
        parser.add_argument(
            "--nosl-is-az-search-mode",
            choices=("puct", "gumbel", "full_gumbel"),
            default="puct",
            help="Native tree policy: PUCT, legacy root Gumbel, or Full Gumbel AZ.",
        )
        parser.add_argument(
            "--nosl-is-az-paper-full-gumbel",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use canonical paper Full Gumbel AlphaZero settings and explicit sequential halving.",
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-candidates",
            type=int,
            default=0,
            help="Root Gumbel top-k size; 0 keeps all legal root actions.",
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-scale", type=float, default=1.0,
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-value-scale", type=float, default=0.1,
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-max-visit-init", type=float, default=50.0,
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-rescale-q",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-use-mixed-value",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-interior-uniformity",
            type=float,
            default=0.0,
            help="Full-Gumbel non-root smoothing strength in [0,1].",
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-interior-coverage-fraction",
            type=float,
            default=0.0,
            help="Prior-independent Full-Gumbel interior coverage fraction in [0,1].",
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-interior-value-mode",
            type=int,
            choices=(0, 1, 2),
            default=2,
            help="Interior V aggregation: 0=mean, 1=max, 2=progressive mean-to-max.",
        )
        parser.add_argument(
            "--nosl-is-az-gumbel-interior-max-mix-visits",
            type=float,
            default=64.0,
            help="Visit scale at which progressive interior V reaches max.",
        )
        parser.add_argument(
            "--nosl-is-az-dirichlet-alpha", type=float, default=0.0,
            help="Dirichlet concentration for root noise; 0 disables noise.",
        )
        parser.add_argument(
            "--nosl-is-az-dirichlet-epsilon", type=float, default=0.0,
            help="Root-prior noise mixture weight in [0, 1].",
        )
        parser.add_argument(
            "--nosl-is-az-master-nonce",
            type=lambda value: int(value, 0),
            default=0x4E4F534C4953415A,
        )
        parser.add_argument(
            "--nosl-is-az-batch-size",
            type=int,
            default=64,
            help="Effective number of replay roots sampled per AZ update.",
        )
        parser.add_argument(
            "--nosl-is-az-replay",
            dest="nosl_is_az_replay_enabled",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Store labelled combat roots in a replay buffer before AZ "
                 "updates. Omission preserves the checkpoint setting; an "
                 "older checkpoint without this field defaults to enabled.",
        )
        parser.add_argument(
            "--nosl-is-az-micro-batch-size",
            type=int,
            default=2048,
            help="Maximum roots per replay forward/backward pass.",
        )
        parser.add_argument(
            "--nosl-is-az-replay-capacity",
            type=int,
            default=131072,
            help="FIFO replay capacity in labelled combat roots.",
        )
        parser.add_argument(
            "--nosl-is-az-replay-min-size",
            type=int,
            default=1,
            help="Minimum replay roots required before AZ updates begin.",
        )
        parser.add_argument(
            "--nosl-is-az-replay-max-old-fraction",
            type=float,
            default=1.0,
            help="Maximum fraction of a replay batch drawn from before the "
                 "current collection. 1.0 preserves uniform replay.",
        )
        parser.add_argument(
            "--nosl-is-az-leaf-batch-size",
            type=int,
            default=64,
            help="Number of C++ AlphaZero leaf states returned per worker round; "
                  "larger values reduce IPC synchronization and use larger GPU batches.",
        )
        parser.add_argument(
            "--nosl-is-az-inference-batch-size",
            type=int,
            default=32,
            help="Target GPU batch assembled from independent search trees. "
                 "This does not increase per-tree leaf concurrency.",
        )
        parser.add_argument(
            "--nosl-is-az-inference-max-wait-ms",
            type=float,
            default=0.5,
            help="Fixed-mode queue delay or adaptive-mode watchdog floor.",
        )
        parser.add_argument(
            "--nosl-is-az-inference-schedule",
            choices=("fixed_wait", "adaptive"),
            default="fixed_wait",
            help="Cross-tree GPU batching policy; adaptive mode uses active "
                 "tree capacity and measured backpressure.",
        )
        parser.add_argument("--nosl-is-az-updates", type=int, default=1)
        parser.add_argument("--nosl-is-az-policy-coef", type=float, default=1.0)
        parser.add_argument("--nosl-is-az-value-coef", type=float, default=0.5)

    def make_env_fn(self, args):
        from rl.training.reward import ShapedRewardWrapper

        ascension = args.ascension
        interactive_reasoning = (
            args.arch == "semantic-reasoning"
            if args.interactive_reasoning is None
            else bool(args.interactive_reasoning)
        )

        def _thunk():
            env = SlayTheSpireEnv(
                ascension=ascension,
                global_map_view=args.global_map_view,
                card_language=(args.card_language or interactive_reasoning),
                combat_language=args.combat_language,
                deck_size_observation=args.deck_size_observation,
                neow_observation=args.neow_observation,
                event_option_observation=args.event_option_observation,
                interactive_reasoning=interactive_reasoning,
                combat_action_contract=getattr(args, "combat_action_contract", None) or "legacy",
                nosl_is_az_boss_transition_target=bool(
                    getattr(args, "nosl_is_az_boss_transition_target", False)
                ),
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
        if args.arch == "split-pointer":
            # The core owns both specialist encoders, policy heads, and value
            # heads, so no SB3 features_extractor/net_arch is needed here.
            return SplitStSPointerPolicy, dict(
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                card_language=args.card_language,
                combat_language=args.combat_language,
                map_dag=args.map_dag,
                card_vector_checkpoint=args.card_vector_checkpoint,
            )
        if args.arch == "semantic-reasoning":
            if args.interactive_reasoning is False:
                raise ValueError(
                    "semantic-reasoning requires interactive trace capture"
                )
            from rl.policies.semantic_reasoning_policy import SemanticReasoningPolicy

            return SemanticReasoningPolicy, dict(
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                card_program_layers=args.card_program_layers,
                reasoning_layers=args.reasoning_layers,
                max_reasoning_tokens=args.max_reasoning_tokens,
                combat_language=args.combat_language,
                map_dag=args.map_dag,
            )
        raise ValueError(f"unsupported whole-run architecture: {args.arch}")

    def tag(self, args, run_id):
        map_tag = "-globalmap" if args.global_map_view else ""
        if args.global_map_view and args.map_dag:
            map_tag += "-dag"
        card_tag = "-cardlang" if args.card_language else ""
        if args.combat_language:
            card_tag += "-combatlang"
        deck_tag = "-decksize" if args.deck_size_observation else ""
        event_tag = "-eventopt" if args.event_option_observation else ""
        hierarchy_tag = "-hier-hp" if args.hierarchical_combat_hp else ""
        az_tag = "-nosl-is-az" if getattr(args, "nosl_is_az", False) else ""
        if getattr(args, "nosl_is_az_inner_only", False):
            az_tag += "-inner-only"
        if getattr(args, "nosl_is_az_cpp", False):
            az_tag += "-cpp"
        if getattr(args, "nosl_is_az_reset_combat", False):
            az_tag += "-reset-combat"
        reasoning_tag = (
            "-semantic-reasoning" if args.arch == "semantic-reasoning" else ""
        )
        card_vector_tag = (
            "-cardvec" if getattr(args, "card_vector_checkpoint", None) else ""
        )
        return (
            f"whole-run{map_tag}{card_tag}{deck_tag}{event_tag}{hierarchy_tag}{az_tag}"
            f"{reasoning_tag}{card_vector_tag}"
            f"-asc{args.ascension}"
            f"-s{args.seed}-{run_id}"
        )
