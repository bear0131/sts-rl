"""Evaluate PPO and NOSL Flat-MC through one complete combat.

The starting combat state is reconstructed from a corpus record's episode seed
and action prefix.  Each evaluation trial gives PPO and Flat-MC the same
NOSL-redeterminized future, so the comparison is paired at the combat level.
Flat-MC is used at every thin combat decision; its internal rollouts receive
separate NOSL futures and use uniform legal actions after the forced root.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from env.core.run_env import SlayTheSpireEnv
from rl.policies import model as model_helpers
from rl.evaluation.nosl.collect_nosl_corpus import (
    _clone_redeterminized,
    _load_custom_objects,
    flat_mc_action,
)
from env.search.nosl_sampler import NOSLSampler

from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _env_from_model(model, ascension: int) -> SlayTheSpireEnv:
    return SlayTheSpireEnv(
        ascension=ascension,
        global_map_view=model_helpers.uses_global_map_observation(model),
        card_language=False,
        combat_language=model_helpers.uses_combat_language(model),
        deck_size_observation=model_helpers.uses_deck_size_observation(model),
        neow_observation=model_helpers.uses_neow_observation(model),
        event_option_observation=model_helpers.uses_event_option_observation(model),
    )


def _paired_branch(source, sampler: NOSLSampler, state_nonce: int, trial: int):
    """Make an isolated branch; both policies use the same trial nonce."""
    return _clone_redeterminized(source, sampler, state_nonce, trial)


def _outcome(branch) -> str:
    raw = branch.thick.bc.to_dict()
    # BattleContext outcomes are UNDECIDED=0, VICTORY=1, LOSS=2, ESCAPE=3.
    value = int(raw.get("outcome", 0))
    return {1: "victory", 2: "loss", 3: "escape"}.get(value, "undecided")


def _terminal_stats(branch, steps: int, timeout: bool) -> dict[str, Any]:
    raw = branch.thick.bc.to_dict()
    return {
        "terminal_hp": int(raw["player"]["hp"]),
        "projected_exit_hp": int(branch.thick.bc.projected_exit_hp(branch.thick.gc)),
        "outcome": _outcome(branch),
        "steps": steps,
        "timeout": bool(timeout),
    }


def _run_ppo(model, branch, *, max_steps: int) -> dict[str, Any]:
    obs = branch._obs()
    steps = 0
    terminated = False
    while not terminated and steps < max_steps:
        mask = branch.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _reward, terminated, truncated, _info = branch.step(
            int(np.asarray(action).item())
        )
        if truncated:
            break
        steps += 1
    return _terminal_stats(branch, steps, not terminated)


def _run_flat_mc(
    branch,
    *,
    sampler: NOSLSampler,
    state_nonce: int,
    samples: int,
    rollout_max_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    obs = branch._obs()
    steps = 0
    terminated = False
    search_decisions = 0
    while not terminated and steps < max_steps:
        mask = branch.action_masks()
        result = flat_mc_action(
            branch,
            mask,
            sampler=sampler,
            state_nonce=sampler.nonce(
                state_nonce=state_nonce, simulation_index=search_decisions
            ),
            samples=samples,
            max_steps=rollout_max_steps,
        )
        search_decisions += 1
        obs, _reward, terminated, truncated, _info = branch.step(result.action)
        if truncated:
            break
        steps += 1
    stats = _terminal_stats(branch, steps, not terminated)
    stats["search_decisions"] = search_decisions
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--card-vector-checkpoint", type=Path, default=None)
    parser.add_argument("--record-id", type=int, default=0)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0x53494E47)
    parser.add_argument("--master-nonce", type=lambda value: int(value, 0), default=0x4E4F534C_7631)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--mc-rollout-max-steps", type=int, default=128)
    parser.add_argument("--max-combat-steps", type=int, default=700)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_record(path: Path, record_id: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("record_id", -1)) == record_id:
                return record
    raise KeyError(f"record-id {record_id} not found in {path}")


def main() -> None:
    args = build_parser().parse_args()
    if args.trials < 1 or args.mc_samples < 1:
        raise ValueError("trials and mc-samples must be positive")
    if args.mc_rollout_max_steps < 1 or args.max_combat_steps < 1:
        raise ValueError("step limits must be positive")
    if not args.checkpoint.is_file() or not args.corpus.is_file():
        raise FileNotFoundError("checkpoint or corpus not found")
    custom_objects = _load_custom_objects(
        args.checkpoint, args.card_vector_checkpoint
    )
    model = MaskablePPO.load(
        args.checkpoint, device="cpu", custom_objects=custom_objects
    )
    record = _load_record(args.corpus, args.record_id)
    env = _env_from_model(model, args.ascension)
    obs, _info = env.reset(seed=int(record["episode_seed"]))
    for action in record["action_prefix"]:
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        if terminated or truncated:
            raise RuntimeError("corpus action prefix terminated before target state")
    if _info.get("phase") != "combat":
        raise RuntimeError("corpus record does not reconstruct a combat state")
    expected = record.get("observation")
    if expected is not None:
        for key, value in expected.items():
            if key in obs and not np.array_equal(np.asarray(value), np.asarray(obs[key])):
                raise RuntimeError(f"replay observation mismatch at field {key!r}")

    sampler = NOSLSampler(args.master_nonce)
    source = env
    state_nonce = sampler.nonce(
        state_nonce=int(record["episode_seed"]),
        simulation_index=int(record["thin_step"]),
    )
    trials: list[dict[str, Any]] = []
    started = time.perf_counter()
    for trial in range(args.trials):
        trial_nonce = sampler.nonce(state_nonce=state_nonce, simulation_index=trial)
        ppo_branch = _paired_branch(source, sampler, trial_nonce, trial)
        mc_branch = _paired_branch(source, sampler, trial_nonce, trial)
        ppo = _run_ppo(model, ppo_branch, max_steps=args.max_combat_steps)
        flat = _run_flat_mc(
            mc_branch,
            sampler=sampler,
            state_nonce=trial_nonce,
            samples=args.mc_samples,
            rollout_max_steps=args.mc_rollout_max_steps,
            max_steps=args.max_combat_steps,
        )
        trials.append(
            {
                "trial": trial,
                "future_nonce": int(trial_nonce),
                "ppo": ppo,
                "flat_mc": flat,
                "delta_flat_minus_ppo": {
                    "terminal_hp": flat["terminal_hp"] - ppo["terminal_hp"],
                    "projected_exit_hp": flat["projected_exit_hp"]
                    - ppo["projected_exit_hp"],
                },
            }
        )
        print(
            f"[single-combat] trial={trial + 1}/{args.trials} "
            f"ppo_hp={ppo['projected_exit_hp']} "
            f"flat_hp={flat['projected_exit_hp']} "
            f"ppo={ppo['outcome']} flat={flat['outcome']} "
            f"flat_decisions={flat['search_decisions']}",
            flush=True,
        )

    def mean(key: str, policy: str) -> float:
        return float(np.mean([trial[policy][key] for trial in trials]))

    summary = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "corpus": str(args.corpus.resolve()),
        "record_id": int(args.record_id),
        "episode_seed": int(record["episode_seed"]),
        "thin_step": int(record["thin_step"]),
        "objective": "projected terminal combat HP",
        "paired_nosl_future": True,
        "trials": trials,
        "aggregate": {
            "trials": len(trials),
            "ppo_mean_projected_exit_hp": mean("projected_exit_hp", "ppo"),
            "flat_mc_mean_projected_exit_hp": mean("projected_exit_hp", "flat_mc"),
            "mean_projected_exit_hp_delta": float(
                np.mean([trial["delta_flat_minus_ppo"]["projected_exit_hp"] for trial in trials])
            ),
            "ppo_victory_rate": float(
                np.mean([trial["ppo"]["outcome"] == "victory" for trial in trials])
            ),
            "flat_mc_victory_rate": float(
                np.mean([trial["flat_mc"]["outcome"] == "victory" for trial in trials])
            ),
            "flat_mc_mean_decisions": mean("search_decisions", "flat_mc"),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "config": {
            "trials": args.trials,
            "mc_samples_per_decision": args.mc_samples,
            "mc_rollout_max_steps": args.mc_rollout_max_steps,
            "max_combat_steps": args.max_combat_steps,
            "master_nonce": int(args.master_nonce),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[single-combat] wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
