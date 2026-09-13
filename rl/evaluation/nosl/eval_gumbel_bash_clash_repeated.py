"""Repeated Bash/Clash search comparison on the 7.68M checkpoint.

This harness keeps the public fixture controlled while varying both the public
seed (which changes the opening hand slot/layout) and the NOSL future nonce.
It compares ordinary PUCT with the native simplified Gumbel sequential
halving root allocator.  A single model is loaded for the whole matrix so the
experiment is reproducible and avoids checkpoint-load overhead.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from env.core import _BUILD  # noqa: F401
from rl.evaluation.nosl.collect_nosl_corpus import _load_custom_objects
from rl.evaluation.nosl.eval_nosl_order_sensitivity import (
    SCENARIOS,
    _jsonable,
    _make_env,
    _run_search,
)
from env.search.nosl_sampler import NOSLSampler


def _action_name(value: str) -> str:
    """Normalize slot-specific labels to the semantic Bash/Clash action."""
    value = str(value).upper()
    if "BASH" in value:
        return "BASH"
    if "CLASH" in value:
        return "CLASH"
    return value


def _action_for_semantic(result: dict[str, Any], card: str) -> bool:
    return _action_name(result["search"]["action"]) == card


def _visits_for(result: dict[str, Any], card: str) -> int:
    visits = result["search"].get("root_visits", {})
    return int(sum(int(v) for k, v in visits.items() if card in str(k).upper()))


def _q_for(result: dict[str, Any], card: str) -> float | None:
    root = result["search"].get("tree_root", {})
    for child in root.get("children", ()):
        if card.upper() in str(child.get("label", "")).upper():
            value = child.get("value")
            return None if value is None else float(value)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 23, 37, 49, 57, 71, 83, 97])
    parser.add_argument("--nonce-repeats", type=int, default=3)
    parser.add_argument("--simulations", nargs="+", type=int,
                        default=[64, 128, 256, 512])
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--leaf-batch", type=int, default=64)
    parser.add_argument("--gumbel-candidates", nargs="+", type=int,
                        default=[0, 2])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model = MaskablePPO.load(
        args.checkpoint,
        device="cpu",
        custom_objects=_load_custom_objects(args.checkpoint, None),
    )
    sampler = NOSLSampler(0x4E4F534C_7631)
    # Keep the draw pile non-unique so the evaluation probes future NOSL
    # branches rather than a deterministic one-card continuation.
    fillers = ("STRIKE_RED",) * 8
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    # One row is one independent search invocation.  For each seed the base
    # and swapped layouts are paired, so a semantic Bash/Clash comparison is
    # not confounded by which hand slot happened to contain the card.
    for seed in args.seeds:
        for swapped in (False, True):
            env = _make_env(
                model,
                SCENARIOS["bash_plus_clash"],
                swapped=swapped,
                seed=int(seed),
                filler_cards=fillers,
            )
            for repetition in range(max(1, int(args.nonce_repeats))):
                nonce = sampler.nonce(
                    state_nonce=int(seed),
                    simulation_index=100_000 + repetition,
                )
                for simulations in args.simulations:
                    methods = [("puct", 0, 0)]
                    methods.extend(
                        (f"gumbel_c{int(candidates)}", 1, int(candidates))
                        for candidates in args.gumbel_candidates
                    )
                    for method, root_mode, candidates in methods:
                        result = _run_search(
                            model,
                            env,
                            simulations=int(simulations),
                            depth=int(args.depth),
                            nonce=int(nonce),
                            leaf_batch=int(args.leaf_batch),
                            prior_mode="model",
                            root_search_mode=int(root_mode),
                            gumbel_candidates=int(candidates),
                        )
                        rows.append({
                            "seed": int(seed),
                            "swapped": bool(swapped),
                            "repetition": int(repetition),
                            "nonce": int(nonce),
                            "simulations": int(simulations),
                            "depth": int(args.depth),
                            "method": method,
                            "action": str(result["search"]["action"]),
                            "semantic_action": _action_name(
                                result["search"]["action"]
                            ),
                            "bash_selected": _action_for_semantic(result, "BASH"),
                            "clash_selected": _action_for_semantic(result, "CLASH"),
                            "bash_visits": _visits_for(result, "BASH"),
                            "clash_visits": _visits_for(result, "CLASH"),
                            "bash_q": _q_for(result, "BASH"),
                            "clash_q": _q_for(result, "CLASH"),
                            "root_value": float(result["search"]["root_value"]),
                        })
            del env

    # Aggregate separately by method/budget and preserve paired base/swapped
    # counts.  ``mean_q_delta`` ignores missing Q values defensively.
    aggregates: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["simulations"]))].append(row)
    for (method, simulations), group in sorted(grouped.items()):
        deltas = [
            float(r["bash_q"]) - float(r["clash_q"])
            for r in group
            if r["bash_q"] is not None and r["clash_q"] is not None
        ]
        aggregates.append({
            "method": method,
            "simulations": simulations,
            "trials": len(group),
            "bash_selection_rate": sum(bool(r["bash_selected"]) for r in group) / len(group),
            "clash_selection_rate": sum(bool(r["clash_selected"]) for r in group) / len(group),
            "mean_bash_visits": sum(int(r["bash_visits"]) for r in group) / len(group),
            "mean_clash_visits": sum(int(r["clash_visits"]) for r in group) / len(group),
            "mean_q_delta_bash_minus_clash": (sum(deltas) / len(deltas)) if deltas else None,
            "q_delta_trials": len(deltas),
        })
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "seeds": [int(s) for s in args.seeds],
        "nonce_repeats": int(args.nonce_repeats),
        "simulations": [int(s) for s in args.simulations],
        "depth": int(args.depth),
        "leaf_batch": int(args.leaf_batch),
        "scenario": "bash_plus_clash",
        "draw_is_non_unique": True,
        "aggregates": aggregates,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "rows": len(rows),
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
