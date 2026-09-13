"""Checkpoint-aware environment construction shared by training and benchmarks."""
from __future__ import annotations
import argparse
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies import model as model_utils
from rl.tasks.wholerun import WholeRunTask
from rl.training.cached_mask_vec_env import CachedMaskSubprocVecEnv, GroupedCachedMaskSubprocVecEnv


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

    def compat_env_fn():
        env = base_env_fn()
        if expected_space is not None:
            env = CheckpointObservationCompat(env, expected_space)
        return env

    env_fns = [compat_env_fn for _ in range(args.n_envs)]
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
    venv.seed(args.seed + args.rank * 1_000_003)
    return venv
