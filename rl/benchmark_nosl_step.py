"""Benchmark one combat transition on real NOSL combat states.

The benchmark reports both the public thin-environment operation and the
underlying C++ ``BattleContext.execute`` operation.  Every measured branch is
created from a corpus replay and redeterminized with an independent NOSL
nonce; the live world-line RNG is never inspected.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from env.core import action_space as A
from env.core import actions as combat_actions
from env.core import combat_env
from env.core import game_actions as ga
from env.core.run_env import SlayTheSpireEnv
from rl import model as model_helpers
from rl.collect_nosl_corpus import _clone_redeterminized, _load_custom_objects
from rl.nosl_sampler import NOSLSampler

from . import pointer_policy as _pointer_policy  # noqa: F401
from . import split_policy as _split_policy  # noqa: F401


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


def _load_records(path: Path, max_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= max_records:
                    break
    return records


def _replay(env: SlayTheSpireEnv, record: dict[str, Any]) -> None:
    obs, info = env.reset(seed=int(record["episode_seed"]))
    for action in record["action_prefix"]:
        obs, _reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            raise RuntimeError("record prefix terminated before combat")
    if info.get("phase") != "combat":
        raise RuntimeError("record does not reconstruct a combat state")
    expected = record.get("observation") or {}
    for key, value in expected.items():
        if key in obs and not np.array_equal(np.asarray(value), np.asarray(obs[key])):
            raise RuntimeError(f"replay observation mismatch at {key!r}")


def _stats(values_ns: list[int]) -> dict[str, Any]:
    if not values_ns:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    values = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def _action_id(branch: SlayTheSpireEnv, *, end_turn: bool) -> int | None:
    """Pick a legal complete action, preferring END_TURN or a card/potion."""
    mask = np.asarray(branch.action_masks(), dtype=bool)
    legal = [int(value) for value in np.flatnonzero(mask)]
    if end_turn:
        return A.END_TURN if A.END_TURN in legal else (legal[0] if legal else None)
    for candidate in legal:
        semantic, _pending = A.resolve(
            candidate, branch._legal(), branch._pending, branch._phase()
        )
        if semantic is None or semantic[0] != ga.COMBAT:
            continue
        kind = combat_actions.decode(semantic[1])
        if kind[0] != combat_actions.KIND_END:
            return candidate
    return None


def _native_execute_time(
    source: SlayTheSpireEnv,
    sampler: NOSLSampler,
    nonce: int,
    index: int,
    action_id: int,
) -> int | None:
    branch = _clone_redeterminized(source, sampler, nonce, index)
    mask = np.asarray(branch.action_masks(), dtype=bool)
    if action_id < 0 or action_id >= len(mask) or not mask[action_id]:
        return None
    semantic, _pending = A.resolve(
        action_id, branch._legal(), branch._pending, branch._phase()
    )
    if semantic is None or semantic[0] != ga.COMBAT:
        return None
    sts_action = combat_env.to_sts_action(
        combat_actions.decode(semantic[1]), branch.thick.bc
    )
    started = time.perf_counter_ns()
    branch.thick.bc.execute(sts_action)
    return time.perf_counter_ns() - started


def _measure_record(
    source: SlayTheSpireEnv,
    sampler: NOSLSampler,
    record: dict[str, Any],
    *,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    state_nonce = sampler.nonce(
        state_nonce=int(record["episode_seed"]),
        simulation_index=int(record["thin_step"]),
    )
    all_iters = warmup + iterations
    fork_ns: list[int] = []
    mask_ns: list[int] = []
    step_ns: list[int] = []
    non_end_step_ns: list[int] = []
    execute_ns: list[int] = []
    non_end_execute_ns: list[int] = []
    for i in range(all_iters):
        nonce = sampler.nonce(state_nonce=state_nonce, simulation_index=i)

        started = time.perf_counter_ns()
        branch = _clone_redeterminized(source, sampler, nonce, i)
        elapsed = time.perf_counter_ns() - started
        if i >= warmup:
            fork_ns.append(elapsed)

        mask_started = time.perf_counter_ns()
        mask = np.asarray(branch.action_masks(), dtype=bool)
        mask_elapsed = time.perf_counter_ns() - mask_started
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise RuntimeError("benchmark state has no legal action")
        action_id = _action_id(branch, end_turn=True)
        if action_id is None:
            raise RuntimeError("benchmark state has no end-turn-compatible action")
        non_end_action = _action_id(branch, end_turn=False)
        if i >= warmup:
            mask_ns.append(mask_elapsed)

        # The thin step includes action resolution, C++ execute, post-step
        # legality checks, and observation/info construction.
        step_started = time.perf_counter_ns()
        branch.step(action_id)
        step_elapsed = time.perf_counter_ns() - step_started
        if i >= warmup:
            step_ns.append(step_elapsed)

        if non_end_action is not None:
            non_end_branch = _clone_redeterminized(
                source, sampler, nonce, i + all_iters
            )
            non_end_mask = np.asarray(non_end_branch.action_masks(), dtype=bool)
            if non_end_mask[non_end_action]:
                non_end_started = time.perf_counter_ns()
                non_end_branch.step(non_end_action)
                non_end_elapsed = time.perf_counter_ns() - non_end_started
            else:
                non_end_elapsed = None
        else:
            non_end_elapsed = None
        native_elapsed = _native_execute_time(
            source, sampler, nonce, i + 2 * all_iters, action_id
        )
        native_non_end_elapsed = (
            _native_execute_time(
                source, sampler, nonce, i + 3 * all_iters, non_end_action
            )
            if non_end_action is not None
            else None
        )
        if i >= warmup:
            if native_elapsed is not None:
                execute_ns.append(native_elapsed)
            if non_end_elapsed is not None:
                non_end_step_ns.append(non_end_elapsed)
            if native_non_end_elapsed is not None:
                non_end_execute_ns.append(native_non_end_elapsed)

    return {
        "act": int(record.get("act") or record["run_stats"]["act"]),
        "floor": int(record.get("floor") or record["run_stats"]["floor"]),
        "is_boss": bool(record.get("is_boss", False)),
        "encounter": record.get("encounter", "UNKNOWN"),
        "record_id": int(record.get("record_id", -1)),
        "thin_action_id": action_id,
        "non_end_action_count": len(non_end_step_ns),
        "fork_redeterminize": _stats(fork_ns),
        "action_masks": _stats(mask_ns),
        "thin_step": _stats(step_ns),
        "thin_step_non_end": _stats(non_end_step_ns),
        "native_execute": _stats(execute_ns) if execute_ns else None,
        "native_execute_non_end": _stats(non_end_execute_ns)
        if non_end_execute_ns
        else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--card-vector-checkpoint", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--master-nonce", type=lambda value: int(value, 0), default=0x4E4F534C_7631)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    custom_objects = _load_custom_objects(args.checkpoint, args.card_vector_checkpoint)
    model = MaskablePPO.load(args.checkpoint, device="cpu", custom_objects=custom_objects)
    env = _env_from_model(model, args.ascension)
    records = _load_records(args.corpus, args.max_records)
    if not records:
        raise ValueError("corpus contains no records")
    sampler = NOSLSampler(args.master_nonce)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, record in enumerate(records):
            _replay(env, record)
            row = _measure_record(
                env,
                sampler,
                record,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            rows.append(row)
            print(
                f"[bench] {index + 1}/{len(records)} act={row['act']} "
                f"floor={row['floor']} encounter={row['encounter']}",
                flush=True,
            )
    finally:
        env.close()
    output = {
        "schema_version": 1,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "records": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[bench] wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
