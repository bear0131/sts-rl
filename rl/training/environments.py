"""Checkpoint-aware environment construction shared by training and benchmarks."""
from __future__ import annotations
import argparse
import hashlib
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies import model as model_utils
from rl.tasks.wholerun import WholeRunTask, _MAX_EPISODE_STEPS
from rl.training.cached_mask_vec_env import CachedMaskSubprocVecEnv, GroupedCachedMaskSubprocVecEnv


def startup_stagger_delays(
    *, seed: int, rank: int, n_envs: int, span_steps: int
) -> tuple[int, ...]:
    """Return reproducible stratified-random startup delays for one rank."""
    if n_envs < 1:
        raise ValueError("n_envs must be positive")
    if span_steps < 0:
        raise ValueError("span_steps must be non-negative")
    if span_steps == 0:
        return (0,) * n_envs
    seed_words = (
        int(seed) & 0xFFFFFFFF,
        int(rank) & 0xFFFFFFFF,
        0x53544752,
    )
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    positions = (np.arange(n_envs, dtype=np.float64) + rng.random(n_envs)) / n_envs
    rng.shuffle(positions)
    delays = np.rint(positions * span_steps).astype(np.int64)
    return tuple(int(value) for value in delays)


class StartupStaggerWrapper(gym.Wrapper):
    """Hold one environment at reset until its independent startup time."""

    def __init__(self, env: gym.Env, *, hold_steps: int):
        super().__init__(env)
        if hold_steps < 0:
            raise ValueError("hold_steps must be non-negative")
        self._initial_hold_steps = int(hold_steps)
        self._remaining_hold_steps = int(hold_steps)
        self._released = hold_steps == 0
        self._last_observation = None
        self._last_info: dict = {}
        self._steps_since_reset = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_observation = observation
        self._last_info = dict(info)
        self._steps_since_reset = 0
        return observation, info

    def step(self, action):
        if self._remaining_hold_steps > 0:
            self._remaining_hold_steps -= 1
            info = dict(self._last_info)
            info["startup_stagger_hold"] = True
            info["startup_stagger_remaining"] = self._remaining_hold_steps
            release = self._remaining_hold_steps == 0
            if release:
                # VecEnv autoreset makes the next observation an explicit
                # episode start, including when the delay spans all warmup.
                self._released = True
            return self._last_observation, 0.0, False, release, info

        observation, reward, terminated, truncated, info = self.env.step(action)
        self._last_observation = observation
        self._last_info = dict(info)
        self._steps_since_reset += 1
        return observation, reward, terminated, truncated, info

    def startup_stagger_state(self) -> dict[str, int | bool | None]:
        run_stats = self._last_info.get("run_stats") or {}
        return {
            "initial_hold_steps": self._initial_hold_steps,
            "remaining_hold_steps": self._remaining_hold_steps,
            "released": self._released,
            "steps_since_reset": self._steps_since_reset,
            "floor": run_stats.get("floor"),
            "act": run_stats.get("act"),
        }


class StartupOccupancyWrapper(gym.Wrapper):
    """Select a live state from the frozen policy's renewal occupancy measure.

    Draw an age uniformly from the episode cap and reject the whole attempt if
    the episode ends earlier. Each live (trajectory, age) pair then has equal
    unnormalized probability, so long episodes receive their natural weight.
    Accepted environments remain paused until formal collection begins.
    """

    def __init__(self, env: gym.Env, *, max_episode_steps: int, seed: int):
        super().__init__(env)
        if max_episode_steps < 1:
            raise ValueError("occupancy sampling requires a positive episode cap")
        self._max_episode_steps = int(max_episode_steps)
        self._age_rng = np.random.default_rng(seed)
        self._target_age = 0
        self._age = 0
        self._episode_return = 0.0
        self._accepted = False
        self._released = False
        self._rejected_episodes = 0
        self._real_steps = 0
        self._last_observation = None
        self._last_info: dict = {}

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_observation = observation
        self._last_info = dict(info)
        self._age = 0
        self._episode_return = 0.0
        if not self._accepted:
            self._target_age = int(self._age_rng.integers(self._max_episode_steps))
            self._accepted = self._target_age == 0
        return observation, info

    def step(self, action):
        if self._accepted and not self._released:
            info = {
                key: value for key, value in self._last_info.items()
                if key not in (
                    "nosl_value_boundary_observation",
                    "nosl_value_terminal_observation",
                    "reasoning_trace",
                    "episode",
                )
            }
            info["startup_occupancy_paused"] = True
            return (
                self._last_observation, 0.0, False, False, info,
            )
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._real_steps += 1
        self._episode_return += float(reward)
        self._last_observation = observation
        self._last_info = dict(info)
        self._age += 1
        if not self._accepted:
            if terminated or truncated:
                self._rejected_episodes += 1
            elif self._age == self._target_age:
                self._accepted = True
                info = {**info, "startup_occupancy_accepted": True}
        return observation, reward, terminated, truncated, info

    def startup_occupancy_release(self) -> dict[str, int | bool | float]:
        if not self._accepted:
            raise RuntimeError("cannot release an unaccepted occupancy sample")
        self._released = True
        return self.startup_occupancy_state()

    def startup_occupancy_state(self) -> dict[str, int | bool | float]:
        return {
            "accepted": self._accepted,
            "released": self._released,
            "max_age": self._max_episode_steps,
            "age": self._age,
            "episode_return": self._episode_return,
            "target_age": self._target_age,
            "rejected_episodes": self._rejected_episodes,
            "real_steps": self._real_steps,
        }


def startup_stagger_digest(delays: tuple[int, ...]) -> str:
    payload = ",".join(str(value) for value in delays).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class CheckpointObservationCompat(gym.ObservationWrapper):
    """Expose exactly the observation schema stored in a legacy checkpoint.

    Observation fields are append-only.  Newer environments can therefore
    emit fields that an older SB3 policy does not know about (currently the
    post-enemy-turn boundary marker).  SB3 checks spaces before loading, so
    filter the wrapper at the environment boundary instead of weakening that
    safety check globally.
    """

    def __init__(self, env, expected_space):
        super().__init__(env)
        expected = getattr(expected_space, "spaces", {})
        self._keys = frozenset(expected)
        self.observation_space = spaces.Dict({
            key: env.observation_space[key]
            for key in self._keys
            if key in env.observation_space.spaces
        })

    def observation(self, observation):
        return {
            key: value
            for key, value in observation.items()
            if key in self._keys
        }

def observation_flags(model: AccumulatingMaskablePPO) -> dict[str, bool]:
    core = getattr(model.policy, "core", None)
    interactive_reasoning = hasattr(core, "reasoning_loss")
    return {
        "global_map_view": model_utils.uses_global_map_observation(model),
        "card_language": model_utils.uses_card_language(model),
        "combat_language": model_utils.uses_combat_language(model),
        "deck_size_observation": model_utils.uses_deck_size_observation(model),
        "neow_observation": model_utils.uses_neow_observation(model),
        "event_option_observation": (
            model_utils.uses_event_option_observation(model)
        ),
        "interactive_reasoning": interactive_reasoning,
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
    interactive_reasoning = (
        flags["interactive_reasoning"]
        if args.interactive_reasoning is None
        else bool(args.interactive_reasoning)
    )
    env_args.arch = (
        "semantic-reasoning" if interactive_reasoning else "split-pointer"
    )
    env_args.interactive_reasoning = interactive_reasoning
    env_args.map_dag = True

    base_env_fn = WholeRunTask().make_env_fn(env_args)
    expected_space = getattr(model, "observation_space", None)

    stagger_steps = int(getattr(args, "startup_stagger_steps", 0))
    occupancy_budget = int(getattr(args, "startup_occupancy_budget", 0))
    occupancy_max_age = int(getattr(
        args, "startup_occupancy_max_age", _MAX_EPISODE_STEPS,
    ))
    if stagger_steps and occupancy_budget:
        raise ValueError("startup stagger and occupancy sampling are mutually exclusive")
    stagger_delays = startup_stagger_delays(
        seed=args.seed,
        rank=args.rank,
        n_envs=args.n_envs,
        span_steps=stagger_steps,
    )

    def compat_env_fn(hold_steps: int, env_index: int):
        def make():
            env = base_env_fn()
            if expected_space is not None:
                env = CheckpointObservationCompat(env, expected_space)
            if stagger_steps:
                env = StartupStaggerWrapper(env, hold_steps=hold_steps)
            if occupancy_budget:
                env = StartupOccupancyWrapper(
                    env, max_episode_steps=occupancy_max_age,
                    seed=int(np.random.SeedSequence((
                        int(args.seed) & 0xFFFFFFFF, int(args.rank), env_index,
                        0x4F434355,
                    )).generate_state(1)[0]),
                )
            return env

        return make

    env_fns = [compat_env_fn(delay, index) for index, delay in enumerate(stagger_delays)]
    if args.vec == "subproc":
        if args.envs_per_worker == 1:
            venv = CachedMaskSubprocVecEnv(
                env_fns, start_method=args.start_method
            )
        else:
            venv = GroupedCachedMaskSubprocVecEnv(
                env_fns,
                envs_per_worker=args.envs_per_worker,
                start_method=args.start_method,
            )
        print(
            f"[rank {args.rank}] subprocess workers={len(venv.remotes)} "
            f"logical_envs={args.n_envs} "
            f"envs_per_worker={args.envs_per_worker}"
        )
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)
    venv.startup_stagger_delays = stagger_delays
    args.startup_stagger_delays_sha256 = startup_stagger_digest(stagger_delays)
    args.startup_stagger_delay_min = min(stagger_delays)
    args.startup_stagger_delay_mean = float(np.mean(stagger_delays))
    args.startup_stagger_delay_max = max(stagger_delays)
    venv.seed(args.seed + args.rank * 1_000_003)
    return venv
