"""Benchmark repeated NOSL paths with one root redeterminization per path.

This is intentionally model-free.  It replays one combat record, gives the
player a controlled high-survivability/high-energy setup, and measures:

* root ``fork_combat + NOSL redeterminize``;
* native C++ ``BattleContext.execute`` transitions on one branch;
* thin ``search_step`` transitions, which include the Python action machine and
  next-mask construction but no observation/reward/info encoding.

The default END_TURN path keeps the combat alive and makes the benchmark a
fixed amount of work.  Every path receives an independent NOSL nonce; the
hidden future is not reset between steps within that path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from env.core import action_space as A
from env.core import actions as combat_actions
from env.core import combat_env
from env.core import game_actions as ga
from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.nosl.collect_nosl_corpus import _clone_redeterminized
from env.search.nosl_sampler import NOSLSampler


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _replay(env: SlayTheSpireEnv, record: dict[str, Any]) -> None:
    env.reset(seed=int(record["episode_seed"]))
    for action in record["action_prefix"]:
        _obs, _reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            raise RuntimeError("record prefix terminated before combat")
    if info.get("phase") != "combat":
        raise RuntimeError("record does not reconstruct combat")


def _make_invincible(env: SlayTheSpireEnv) -> dict[str, bool]:
    """Install a controlled setup through the trace-build debug setters."""
    bc = env.thick.bc
    required = (
        "set_player_block",
        "set_player_status",
        "set_player_strength",
        "set_player_energy",
    )
    missing = [name for name in required if not hasattr(bc, name)]
    if missing:
        raise RuntimeError(
            "benchmark requires STS_ENABLE_RULE_TRACE setters; missing "
            + ", ".join(missing)
        )
    bc.set_player_block(1_000_000)
    bc.set_player_status("BARRICADE", 1)
    bc.set_player_strength(1_000_000)
    bc.set_player_energy(1_000_000)
    env._invalidate_backend_cache()
    return {name: True for name in required}


def _stats(values_ns: list[int]) -> dict[str, float | int]:
    if not values_ns:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0}
    values = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def _end_turn_sts_action(branch):
    return combat_env.to_sts_action(
        combat_actions.decode(A.END_TURN), branch.thick.bc
    )


def _choose_safe_action(branch, mask: np.ndarray) -> int:
    if A.END_TURN < len(mask) and bool(mask[A.END_TURN]):
        return A.END_TURN
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        raise RuntimeError("controlled benchmark branch has no legal action")
    return int(legal[0])


def _measure_paths(
    source: SlayTheSpireEnv,
    sampler: NOSLSampler,
    *,
    path_count: int,
    path_steps: int,
    warmup_paths: int,
) -> dict[str, Any]:
    state_nonce = sampler.nonce(
        state_nonce=910000,
        simulation_index=0x50415448,
    )
    root_ns: list[int] = []
    native_total_ns: list[int] = []
    backend_total_ns: list[int] = []
    mask_backend_total_ns: list[int] = []
    search_total_ns: list[int] = []
    native_step_ns: list[int] = []
    backend_step_ns: list[int] = []
    fresh_mask_ns: list[int] = []
    search_step_ns: list[int] = []
    native_counts: list[int] = []
    backend_counts: list[int] = []
    mask_backend_counts: list[int] = []
    search_counts: list[int] = []

    total_paths = warmup_paths + path_count
    for path_index in range(total_paths):
        nonce = sampler.nonce(state_nonce=state_nonce, simulation_index=path_index)

        root_started = time.perf_counter_ns()
        root = _clone_redeterminized(source, sampler, nonce, path_index)
        root_elapsed = time.perf_counter_ns() - root_started

        # Native path: same root redeterminization, then only C++ transitions.
        native = root.fork_combat()
        native_action = _end_turn_sts_action(native)
        native_started = time.perf_counter_ns()
        native_steps = 0
        for _ in range(path_steps):
            if native.thick.bc.is_battle_over:
                break
            step_started = time.perf_counter_ns()
            native.thick.bc.execute(native_action)
            native_step_ns_value = time.perf_counter_ns() - step_started
            native_steps += 1
            if path_index >= warmup_paths:
                native_step_ns.append(native_step_ns_value)
        native_elapsed = time.perf_counter_ns() - native_started

        # Thick backend path: includes Python semantic-action dispatch around
        # the same native execute, but still omits thin resolution and masks.
        backend = root.fork_combat()
        backend_action = ga.combat(combat_actions.END_TURN)
        backend_started = time.perf_counter_ns()
        backend_steps = 0
        for _ in range(path_steps):
            if backend.thick.bc.is_battle_over:
                break
            step_started = time.perf_counter_ns()
            terminated = backend.thick.step_fast(backend_action)
            backend_step_ns_value = time.perf_counter_ns() - step_started
            backend_steps += 1
            if path_index >= warmup_paths:
                backend_step_ns.append(backend_step_ns_value)
            if terminated:
                break
        backend_elapsed = time.perf_counter_ns() - backend_started

        # Rebuild the thin legal mask after each direct backend transition. This
        # isolates the dominant state-to-legal-actions path from action resolve.
        mask_backend = root.fork_combat()
        mask_backend_started = time.perf_counter_ns()
        mask_backend_steps = 0
        for _ in range(path_steps):
            if mask_backend.thick.bc.is_battle_over:
                break
            mask_started = time.perf_counter_ns()
            mask = np.asarray(mask_backend.action_masks(), dtype=bool)
            fresh_mask_ns_value = time.perf_counter_ns() - mask_started
            action = _choose_safe_action(mask_backend, mask)
            terminated = mask_backend.thick.step_fast(backend_action)
            mask_backend._invalidate_backend_cache()
            mask_backend_steps += 1
            if path_index >= warmup_paths:
                fresh_mask_ns.append(fresh_mask_ns_value)
            if terminated:
                break
        mask_backend_elapsed = time.perf_counter_ns() - mask_backend_started

        # Thin search path: search_step performs action resolution and builds
        # the next legal mask, but deliberately skips obs/reward/info encoding.
        search = root.fork_combat()
        search_started = time.perf_counter_ns()
        search_steps = 0
        for _ in range(path_steps):
            if search.thick.bc.is_battle_over:
                break
            mask = np.asarray(search.action_masks(), dtype=bool)
            action = _choose_safe_action(search, mask)
            step_started = time.perf_counter_ns()
            terminated, _next_mask = search.search_step(action, validate=False)
            search_step_ns_value = time.perf_counter_ns() - step_started
            search_steps += 1
            if path_index >= warmup_paths:
                search_step_ns.append(search_step_ns_value)
            if terminated:
                break
        search_elapsed = time.perf_counter_ns() - search_started

        if path_index >= warmup_paths:
            root_ns.append(root_elapsed)
            native_total_ns.append(native_elapsed)
            backend_total_ns.append(backend_elapsed)
            mask_backend_total_ns.append(mask_backend_elapsed)
            search_total_ns.append(search_elapsed)
            native_counts.append(native_steps)
            backend_counts.append(backend_steps)
            mask_backend_counts.append(mask_backend_steps)
            search_counts.append(search_steps)

    def mean_or_zero(values: list[int]) -> float:
        return float(np.mean(values)) if values else 0.0

    measured_transitions = int(sum(search_counts))
    return {
        "path_count": path_count,
        "path_steps_requested": path_steps,
        "warmup_paths": warmup_paths,
        "measured_transitions": measured_transitions,
        "root_fork_redeterminize": _stats(root_ns),
        "native_cpp_path_total": _stats(native_total_ns),
        "native_cpp_transition": _stats(native_step_ns),
        "backend_step_fast_path_total": _stats(backend_total_ns),
        "backend_step_fast_transition": _stats(backend_step_ns),
        "fresh_action_mask_path_total": _stats(mask_backend_total_ns),
        "fresh_action_mask": _stats(fresh_mask_ns),
        "thin_search_path_total": _stats(search_total_ns),
        "thin_search_transition": _stats(search_step_ns),
        "native_transitions_per_path": mean_or_zero(native_counts),
        "backend_transitions_per_path": mean_or_zero(backend_counts),
        "mask_backend_transitions_per_path": mean_or_zero(mask_backend_counts),
        "search_transitions_per_path": mean_or_zero(search_counts),
        "search_transitions_per_second": (
            1000.0 / (_stats(search_step_ns)["mean_ms"])
            if search_step_ns and _stats(search_step_ns)["mean_ms"] > 0
            else 0.0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--record-id", type=int, default=None)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--path-steps", type=int, default=64)
    parser.add_argument("--warmup-paths", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, required=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.paths < 1 or args.path_steps < 1 or args.warmup_paths < 0:
        raise ValueError("paths/path-steps must be positive and warmup non-negative")
    records = _load_records(args.corpus)
    records = [record for record in records if record.get("phase") == "combat"]
    if args.record_id is not None:
        records = [
            record for record in records
            if int(record.get("record_id", -1)) == args.record_id
        ]
    else:
        records = [record for record in records if not record.get("is_boss", False)]
    if not records:
        raise ValueError("no matching combat record")
    record = records[0]

    env = SlayTheSpireEnv()
    sampler = NOSLSampler(0x4E4F534C_7631)
    started = time.perf_counter()
    try:
        _replay(env, record)
        controlled = _make_invincible(env)
        result = _measure_paths(
            env,
            sampler,
            path_count=args.paths,
            path_steps=args.path_steps,
            warmup_paths=args.warmup_paths,
        )
    finally:
        env.close()

    output = {
        "schema_version": 1,
        "objective": "fixed NOSL path throughput; one root redeterminization per path",
        "record_id": int(record.get("record_id", -1)),
        "episode_seed": int(record["episode_seed"]),
        "act": int(record.get("act") or record["run_stats"]["act"]),
        "floor": int(record.get("floor") or record["run_stats"]["floor"]),
        "encounter": record.get("encounter", "UNKNOWN"),
        "is_boss": bool(record.get("is_boss", False)),
        "controlled_setup": controlled,
        "nosl_root_only": True,
        "result": result,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True, indent=2))
    print(f"[bench-paths] wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
