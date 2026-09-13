"""Paired strict Bash/Clash evaluation for Gumbel-AlphaZero variants.

This evaluator is deliberately independent of the search implementation.  It
passes ``root_search_mode=1`` (the current simplified allocator) and
``root_search_mode=2`` (the full/completed-score allocator) through the same
Python adapter, so a native implementation can be swapped without changing
the experiment harness.  Every row uses the same public fixture and NOSL
nonce for all requested modes.

The fixture has exactly two legal PLAY_CARD actions at the root (Bash and
Clash) and energy two.  End Turn remains a native legal action and is reported
separately; it is never counted as a card-order success.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from sb3_contrib import MaskablePPO

from env.core import _BUILD  # noqa: F401
from rl.evaluation.nosl.collect_nosl_corpus import _load_custom_objects
from rl.evaluation.nosl.eval_gumbel_bash_clash_trials import (
    FILLER_PROFILES,
    _hand,
    _legal_cards,
    _make_trial_env,
)
from rl.evaluation.nosl.eval_nosl_order_sensitivity import _jsonable, _run_search
from rl.evaluation.nosl.eval_strict_bash_clash_sequence import _execute_selected
from env.search.nosl_sampler import NOSLSampler


def _classify(action: str) -> str:
    upper = str(action).upper()
    if "BASH" in upper:
        return "bash"
    if "CLASH" in upper:
        return "clash"
    # Native labels are human-readable ("{ end turn }") rather than the
    # thin action enum spelling, so accept both forms for diagnostics.
    if "END_TURN" in upper or "END TURN" in upper:
        return "end_turn"
    return "other"


def _strict_root(env) -> tuple[list[dict[str, Any]], list[str]]:
    raw = env.thick.bc.to_dict()
    if int(raw["player"]["energy"]) != 2:
        raise RuntimeError("strict root energy is not two")
    hand = _hand(env)
    legal = _legal_cards(env, hand)
    if sorted(legal) != ["BASH", "CLASH"]:
        raise RuntimeError(f"unexpected legal cards: {legal}")
    return hand, legal


def _edge_rows(search: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize native tree metadata while preserving optional full-Gumbel fields."""
    tree = search.get("tree_root") or {}
    rows: list[dict[str, Any]] = []
    for child in tree.get("children", ()):
        label = str(child.get("label", ""))
        semantic = _classify(label)
        row: dict[str, Any] = {
            "edge_index": int(child.get("edge_index", -1)),
            "label": label,
            "semantic": semantic,
            "prior": float(child.get("prior", 0.0)),
            "visits": int(child.get("visits", 0)),
            "child_visits": int(child.get("child_visits", 0)),
            "child_states": int(child.get("child_states", 0)),
            "q": float(child.get("value", 0.0)),
            "expandable": bool(child.get("expandable", False)),
        }
        # New native implementations may expose these directly.  Keeping
        # them optional makes the harness compatible with the current build.
        for key in (
            "raw_q", "gumbel", "gumbel_score", "completed_q", "completed_score",
            "candidate", "active_candidate", "normalized_q",
        ):
            if key in child:
                row[key] = _jsonable(child[key])
        rows.append(row)
    return rows


def _requested_mode(name: str) -> tuple[int, str]:
    if name == "puct":
        return 0, "puct"
    if name in {"gumbel", "simplified_gumbel"}:
        return 1, "simplified_gumbel"
    if name in {"full", "full_gumbel", "completed_score"}:
        return 2, "full_gumbel"
    raise ValueError(f"unknown mode {name}")


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["requested_mode"]),
                int(row["gumbel_candidates"]),
                int(row["simulations"]),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for (mode, candidates, simulations), items in sorted(groups.items()):
        n = len(items)
        def selected(name: str) -> int:
            return sum(1 for item in items if item.get("selected") == name)
        def avg(name: str) -> float | None:
            values = [float(item.get(name, 0)) for item in items]
            return mean(values) if values else None
        result.append({
            "requested_mode": mode,
            "gumbel_candidates": candidates,
            "simulations": simulations,
            "trials": n,
            "bash_selected": selected("bash"),
            "clash_selected": selected("clash"),
            "end_turn_selected": selected("end_turn"),
            "other_selected": selected("other"),
            "bash_selection_rate": selected("bash") / n if n else 0.0,
            "clash_selection_rate": selected("clash") / n if n else 0.0,
            "mean_bash_visits": avg("bash_visits"),
            "mean_clash_visits": avg("clash_visits"),
            "mean_end_turn_visits": avg("end_turn_visits"),
            "mean_observed_active_edges": avg("observed_active_edges"),
            "effective_modes": sorted({str(item.get("effective_mode", "")) for item in items}),
            "sequence_bash_clash": sum(
                1 for item in items if item.get("sequence") == "bash_clash"
            ),
            "sequence_clash_bash": sum(
                1 for item in items if item.get("sequence") == "clash_bash"
            ),
            "sequence_censored": sum(
                1 for item in items if item.get("sequence") == "censored"
            ),
        })
    return result


def _run_one(
    model,
    *,
    seed: int,
    swapped: bool,
    fillers: tuple[str, ...],
    nonce: int,
    second_nonce: int,
    simulations: int,
    depth: int,
    leaf_batch: int,
    requested_mode: str,
    mode_id: int,
    candidates: int,
    sequence: bool,
    prior_temperature: float,
    c_puct: float,
    gumbel_scale: float,
    gumbel_value_scale: float,
    gumbel_max_visit_init: float,
    gumbel_rescale_q: bool,
    gumbel_use_mixed_value: bool,
    paper_full_gumbel: bool,
) -> dict[str, Any]:
    env = _make_trial_env(model, seed=seed, swapped=swapped, fillers=fillers)
    hand, legal = _strict_root(env)
    first_result = _run_search(
        model,
        env,
        simulations=simulations,
        depth=depth,
        nonce=nonce,
        leaf_batch=leaf_batch,
        prior_mode="model",
        root_search_mode=mode_id,
        gumbel_candidates=candidates,
        prior_temperature=prior_temperature,
        c_puct=c_puct,
        gumbel_scale=gumbel_scale,
        gumbel_value_scale=gumbel_value_scale,
        gumbel_max_visit_init=gumbel_max_visit_init,
        gumbel_rescale_q=gumbel_rescale_q,
        gumbel_use_mixed_value=gumbel_use_mixed_value,
        paper_full_gumbel=paper_full_gumbel,
    )
    first_search = first_result["search"]
    first = _classify(first_search["action"])
    edges = _edge_rows(first_search)
    visits = {str(edge["semantic"]): int(edge["visits"]) for edge in edges}
    row: dict[str, Any] = {
        "seed": int(seed),
        "swapped": bool(swapped),
        "nonce": int(nonce),
        "second_nonce": int(second_nonce),
        "simulations": int(simulations),
        "depth": int(depth),
        "leaf_batch": int(leaf_batch),
        "requested_mode": requested_mode,
        "requested_root_search_mode": int(mode_id),
        "paper_full_gumbel": bool(paper_full_gumbel),
        "gumbel_candidates": int(candidates),
        # Keep the native mode label alongside the requested mode.  This is
        # useful when comparing an older simulator build that only supports
        # legacy Gumbel (mode 1) with the Full-Gumbel implementation (mode 2).
        "effective_mode": str(first_result.get("root_search_mode", "unknown")),
        "selected": first,
        "first_action": str(first_search["action"]),
        "root_value": float(first_search.get("root_value", 0.0)),
        "root_visit_policy": first_search.get("root_visit_policy", {}),
        "improved_policy": first_search.get("improved_policy", {}),
        "used_completed_policy": bool(
            first_search.get("used_completed_policy", False)
        ),
        "edges": edges,
        "hand": hand,
        "legal_cards": legal,
        "bash_visits": int(visits.get("bash", 0)),
        "clash_visits": int(visits.get("clash", 0)),
        "end_turn_visits": int(visits.get("end_turn", 0)),
        "observed_active_edges": sum(1 for edge in edges if int(edge["visits"]) > 0),
        "sequence": "censored",
        "second": None,
    }
    if sequence and first in {"bash", "clash"}:
        expected = "clash" if first == "bash" else "bash"
        _execute_selected(env, first_result)
        remaining = _legal_cards(env, _hand(env))
        # The strict fixture is expected to leave only the other card.  Keep
        # malformed states visible rather than silently calling them failures.
        if remaining != [expected.upper()]:
            row["sequence"] = "invalid_remaining"
            row["remaining_cards"] = remaining
        else:
            second_result = _run_search(
                model,
                env,
                simulations=simulations,
                depth=depth,
                nonce=second_nonce,
                leaf_batch=leaf_batch,
                prior_mode="model",
                root_search_mode=mode_id,
                gumbel_candidates=candidates,
                prior_temperature=prior_temperature,
                c_puct=c_puct,
                gumbel_scale=gumbel_scale,
                gumbel_value_scale=gumbel_value_scale,
                gumbel_max_visit_init=gumbel_max_visit_init,
                gumbel_rescale_q=gumbel_rescale_q,
                gumbel_use_mixed_value=gumbel_use_mixed_value,
                paper_full_gumbel=paper_full_gumbel,
            )
            second = _classify(second_result["search"]["action"])
            row["second"] = second
            row["second_action"] = str(second_result["search"]["action"])
            row["second_edges"] = _edge_rows(second_result["search"])
            row["second_improved_policy"] = second_result["search"].get(
                "improved_policy", {}
            )
            row["sequence"] = (
                f"{first}_{second}" if second == expected else "censored"
            )
    del env
    return row


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@", description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[219, 357, 454, 575, 605])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--simulations", nargs="+", type=int,
                        default=[64, 128, 256, 512])
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--leaf-batch", type=int, default=64)
    parser.add_argument("--modes", nargs="+", default=["puct", "gumbel", "full"],
                        choices=["puct", "gumbel", "full"])
    parser.add_argument("--gumbel-candidates", nargs="+", type=int,
                        default=[2, 3])
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--gumbel-scale", type=float, default=1.0)
    parser.add_argument("--gumbel-value-scale", type=float, default=0.1)
    parser.add_argument("--gumbel-max-visit-init", type=float, default=50.0)
    parser.add_argument("--no-gumbel-rescale-q", dest="gumbel_rescale_q",
                        action="store_false", default=True)
    parser.add_argument("--no-gumbel-mixed-value",
                        dest="gumbel_use_mixed_value", action="store_false",
                        default=True)
    parser.add_argument(
        "--paper-full-gumbel",
        action="store_true",
        help="Use the canonical paper settings and explicit sequential halving.",
    )
    parser.add_argument("--sequence", action="store_true",
                        help="also execute the selected first card and search the second card")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    model = MaskablePPO.load(
        args.checkpoint,
        device="cpu",
        custom_objects=_load_custom_objects(args.checkpoint, None),
    )
    fillers = FILLER_PROFILES["strict_root_normal_draw"]
    sampler = NOSLSampler(0x4E4F534C_7631)
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for swapped in (False, True):
            # Validate and retain only the requested strict fixture states.
            env = _make_trial_env(model, seed=int(seed), swapped=swapped, fillers=fillers)
            try:
                _strict_root(env)
            except RuntimeError:
                del env
                continue
            del env
            for repeat in range(int(args.repeats)):
                nonce = sampler.nonce(
                    state_nonce=(int(seed) << 1) | int(swapped),
                    simulation_index=500_000 + repeat,
                )
                second_nonce = sampler.nonce(
                    state_nonce=int(nonce), simulation_index=800_000 + repeat,
                )
                for simulations in args.simulations:
                    for mode_name in args.modes:
                        mode_id, mode_label = _requested_mode(mode_name)
                        # candidates is ignored by PUCT, but retaining the
                        # column makes paired matrices rectangular.
                        candidate_values = (
                            [0]
                            if not mode_id
                            else ([16] if args.paper_full_gumbel and mode_id == 2
                                  else args.gumbel_candidates)
                        )
                        for candidates in candidate_values:
                            print(
                                f"[full-gumbel] seed={seed} swapped={swapped} "
                                f"repeat={repeat} mode={mode_label} c={candidates} "
                                f"sim={simulations}", flush=True,
                            )
                            rows.append(_run_one(
                                model,
                                seed=int(seed),
                                swapped=swapped,
                                fillers=fillers,
                                nonce=int(nonce),
                                second_nonce=int(second_nonce),
                                simulations=int(simulations),
                                depth=int(args.depth),
                                leaf_batch=int(args.leaf_batch),
                                requested_mode=mode_label,
                                mode_id=mode_id,
                                candidates=int(candidates),
                                sequence=bool(args.sequence),
                                prior_temperature=float(args.prior_temperature),
                                c_puct=float(args.c_puct),
                                gumbel_scale=float(args.gumbel_scale),
                                gumbel_value_scale=float(args.gumbel_value_scale),
                                gumbel_max_visit_init=float(args.gumbel_max_visit_init),
                                gumbel_rescale_q=bool(args.gumbel_rescale_q),
                                gumbel_use_mixed_value=bool(args.gumbel_use_mixed_value),
                                paper_full_gumbel=(
                                    bool(args.paper_full_gumbel) and mode_id == 2
                                ),
                            ))

    if not rows:
        raise RuntimeError("no strict Bash+Clash states found")
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "profile": "strict_root_normal_draw",
        "seeds": [int(seed) for seed in args.seeds],
        "repeats": int(args.repeats),
        "simulations": [int(value) for value in args.simulations],
        "depth": int(args.depth),
        "leaf_batch": int(args.leaf_batch),
        "modes": list(args.modes),
        "gumbel_candidates": [int(value) for value in args.gumbel_candidates],
        "prior_temperature": float(args.prior_temperature),
        "c_puct": float(args.c_puct),
        "gumbel_scale": float(args.gumbel_scale),
        "gumbel_value_scale": float(args.gumbel_value_scale),
        "gumbel_max_visit_init": float(args.gumbel_max_visit_init),
        "gumbel_rescale_q": bool(args.gumbel_rescale_q),
        "gumbel_use_mixed_value": bool(args.gumbel_use_mixed_value),
        "paper_full_gumbel": bool(args.paper_full_gumbel),
        "sequence": bool(args.sequence),
        "rows": rows,
        "summary": _summary(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"rows={len(rows)} output={args.output.resolve()}")
    for item in payload["summary"]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
