"""Evaluate several combat policies on stratified full-combat corpus states."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from rl.evaluation.nosl.collect_nosl_corpus import (
    _clone_redeterminized,
    _load_custom_objects,
)
from rl.evaluation.nosl.eval_nosl_single_combat import (
    _env_from_model,
    _run_ppo,
)
from env.search.nosl_sampler import NOSLSampler
from rl.search.nosl_search_methods import (
    BeamConfig,
    MCTSConfig,
    NOSLBeamSearch,
    NOSLMCTS,
    random_action,
)

from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


METHODS = ("ppo", "random", "flat_mc", "beam", "mcts")


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


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("corpus contains no records")
    return records


def _floor_group(record: dict[str, Any]) -> tuple[int, int, int]:
    act = int(record.get("act") or (record.get("run_stats") or {}).get("act", 0))
    floor = int(record.get("floor") or (record.get("run_stats") or {}).get("floor", 0))
    # Keep exact floor in the group; the aggregate also reports act so floor
    # ladders remain comparable across acts.
    return act, floor, int(bool(record.get("is_boss", floor in (16, 33, 50))))


def _select_stratified(
    records: list[dict[str, Any]],
    limit: int,
    *,
    max_act: int | None = None,
    floors: set[int] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("phase") != "combat":
            continue
        act, floor, _is_boss = _floor_group(record)
        if max_act is not None and act > max_act:
            continue
        if floors is not None and floor not in floors:
            continue
        buckets[_floor_group(record)].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: int(record.get("record_id", 0)))
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    # Ensure a representative act boss is present whenever the requested
    # floor/capacity permits it; otherwise a round-robin floor scan can spend
    # its entire budget on early normal rooms.
    boss_keys = [key for key in keys if key[2]]
    for key in boss_keys:
        if len(selected) >= limit:
            break
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop(0))
    while keys and len(selected) < limit:
        next_keys: list[tuple[int, int, int]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) >= limit:
                    break
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _run_random(branch, *, sampler, state_nonce: int, max_steps: int) -> dict[str, Any]:
    obs = branch._obs()
    steps = 0
    terminated = False
    while not terminated and steps < max_steps:
        mask = np.asarray(branch.action_masks(), dtype=bool)
        action = random_action(
            mask, sampler=sampler, state_nonce=state_nonce, index=steps
        )
        obs, _reward, terminated, truncated, _info = branch.step(action)
        if truncated:
            break
        steps += 1
    raw = branch.thick.bc.to_dict()
    return {
        "terminal_hp": int(raw["player"]["hp"]),
        "projected_exit_hp": int(branch.thick.bc.projected_exit_hp(branch.thick.gc)),
        "outcome": {1: "victory", 2: "loss"}.get(int(raw.get("outcome", 0)), "undecided"),
        "steps": steps,
        "timeout": not terminated,
    }


def _run_search_method(
    branch,
    method,
    *,
    sampler,
    state_nonce: int,
    max_steps: int,
    flat_samples: int,
    rollout_steps: int,
    beam: NOSLBeamSearch,
    mcts: NOSLMCTS,
) -> dict[str, Any]:
    obs = branch._obs()
    steps = 0
    decisions = 0
    terminated = False
    while not terminated and steps < max_steps:
        mask = np.asarray(branch.action_masks(), dtype=bool)
        if method == "flat_mc":
            from rl.evaluation.nosl.collect_nosl_corpus import flat_mc_action

            action = flat_mc_action(
                branch,
                mask,
                sampler=sampler,
                state_nonce=sampler.nonce(
                    state_nonce=state_nonce, simulation_index=decisions
                ),
                samples=flat_samples,
                max_steps=rollout_steps,
            ).action
        elif method == "beam":
            action = beam.choose(
                branch,
                mask,
                sampler=sampler,
                state_nonce=state_nonce,
                decision_index=decisions,
            )
        elif method == "mcts":
            action = mcts.choose(
                branch,
                mask,
                sampler=sampler,
                state_nonce=state_nonce,
                decision_index=decisions,
            )
        else:
            raise ValueError(method)
        obs, _reward, terminated, truncated, _info = branch.step(action)
        decisions += 1
        if truncated:
            break
        steps += 1
    raw = branch.thick.bc.to_dict()
    return {
        "terminal_hp": int(raw["player"]["hp"]),
        "projected_exit_hp": int(branch.thick.bc.projected_exit_hp(branch.thick.gc)),
        "outcome": {1: "victory", 2: "loss"}.get(int(raw.get("outcome", 0)), "undecided"),
        "steps": steps,
        "timeout": not terminated,
        "search_decisions": decisions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--card-vector-checkpoint", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=24)
    parser.add_argument("--max-act", type=int, default=2)
    parser.add_argument(
        "--floors",
        default=None,
        help="Optional comma-separated exact floors to evaluate (e.g. 1,8,16,33).",
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--master-nonce", type=lambda value: int(value, 0), default=0x4E4F534C_7631)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--rollout-max-steps", type=int, default=128)
    parser.add_argument("--max-combat-steps", type=int, default=700)
    parser.add_argument("--beam-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--mcts-simulations", type=int, default=16)
    parser.add_argument("--mcts-depth", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_records < 1 or args.trials < 1 or args.mc_samples < 1:
        raise ValueError("max-records, trials and mc-samples must be positive")
    methods = tuple(dict.fromkeys(item.strip() for item in args.methods.split(",") if item.strip()))
    unknown = set(methods) - set(METHODS)
    if not methods or unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    floors = None
    if args.floors:
        floors = {int(value.strip()) for value in args.floors.split(",") if value.strip()}
        if not floors:
            raise ValueError("--floors must contain at least one integer")
    records = _select_stratified(
        _load_records(args.corpus),
        args.max_records,
        max_act=args.max_act,
        floors=floors,
    )
    if not records:
        raise ValueError("corpus contains no combat records")
    custom_objects = _load_custom_objects(args.checkpoint, args.card_vector_checkpoint)
    model = MaskablePPO.load(args.checkpoint, device="cpu", custom_objects=custom_objects)
    env = _env_from_model(model, args.ascension)
    sampler = NOSLSampler(args.master_nonce)
    beam = NOSLBeamSearch(
        BeamConfig(depth=args.beam_depth, width=args.beam_width, rollout_steps=args.rollout_max_steps)
    )
    mcts = NOSLMCTS(
        MCTSConfig(
            simulations=args.mcts_simulations,
            depth=args.mcts_depth,
            rollout_steps=args.rollout_max_steps,
        )
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for record_index, record in enumerate(records):
            obs, info = env.reset(seed=int(record["episode_seed"]))
            for action in record["action_prefix"]:
                obs, _reward, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    raise RuntimeError("record prefix terminated before combat")
            if info.get("phase") != "combat":
                raise RuntimeError("selected record does not reconstruct combat")
            expected = record.get("observation") or {}
            for key, value in expected.items():
                if key in obs and not np.array_equal(np.asarray(value), np.asarray(obs[key])):
                    raise RuntimeError(f"replay observation mismatch at {key!r}")
            base_nonce = sampler.nonce(
                state_nonce=int(record["episode_seed"]),
                simulation_index=int(record["thin_step"]),
            )
            for trial in range(args.trials):
                trial_nonce = sampler.nonce(state_nonce=base_nonce, simulation_index=trial)
                for method in methods:
                    branch = _clone_redeterminized(env, sampler, trial_nonce, trial)
                    if method == "ppo":
                        stats = _run_ppo(model, branch, max_steps=args.max_combat_steps)
                    elif method == "random":
                        stats = _run_random(
                            branch,
                            sampler=sampler,
                            state_nonce=trial_nonce,
                            max_steps=args.max_combat_steps,
                        )
                    else:
                        stats = _run_search_method(
                            branch,
                            method,
                            sampler=sampler,
                            state_nonce=trial_nonce,
                            max_steps=args.max_combat_steps,
                            flat_samples=args.mc_samples,
                            rollout_steps=args.rollout_max_steps,
                            beam=beam,
                            mcts=mcts,
                        )
                    rows.append(
                        {
                            "record_id": int(record.get("record_id", record_index)),
                            "episode": int(record.get("episode", 0)),
                            "episode_seed": int(record["episode_seed"]),
                            "act": int(record.get("act") or record["run_stats"]["act"]),
                            "floor": int(record.get("floor") or record["run_stats"]["floor"]),
                            "encounter": record.get("encounter", "UNKNOWN"),
                            "is_boss": bool(record.get("is_boss", False)),
                            "trial": trial,
                            "future_nonce": int(trial_nonce),
                            "method": method,
                            **stats,
                        }
                    )
            print(
                f"[methods] {record_index + 1}/{len(records)} "
                f"act={record.get('act')} floor={record.get('floor')} "
                f"encounter={record.get('encounter', 'UNKNOWN')}",
                flush=True,
            )
    finally:
        env.close()

    grouped: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[int, int, bool, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["act"], row["floor"], row["is_boss"], row["method"])].append(row)
    for (act, floor, is_boss, method), group_rows in sorted(groups.items()):
        grouped[f"act{act}/floor{floor}/{method}"] = {
            "act": act,
            "floor": floor,
            "is_boss": is_boss,
            "method": method,
            "trials": len(group_rows),
            "record_count": len({row["record_id"] for row in group_rows}),
            "mean_projected_exit_hp": float(np.mean([row["projected_exit_hp"] for row in group_rows])),
            "victory_rate": float(np.mean([row["outcome"] == "victory" for row in group_rows])),
            "loss_rate": float(np.mean([row["outcome"] == "loss" for row in group_rows])),
            "timeout_rate": float(np.mean([row["timeout"] for row in group_rows])),
            "mean_steps": float(np.mean([row["steps"] for row in group_rows])),
        }
    output = {
        "schema_version": 1,
        "objective": "projected terminal combat HP",
        "paired_nosl_future": True,
        "methods": methods,
        "records": len(records),
        "rows": rows,
        "grouped": grouped,
        "config": {
            "trials": args.trials,
            "mc_samples": args.mc_samples,
            "rollout_max_steps": args.rollout_max_steps,
            "max_combat_steps": args.max_combat_steps,
            "beam_depth": args.beam_depth,
            "beam_width": args.beam_width,
            "mcts_simulations": args.mcts_simulations,
            "mcts_depth": args.mcts_depth,
            "master_nonce": int(args.master_nonce),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "source_corpus": str(args.corpus.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(output), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[methods] wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
