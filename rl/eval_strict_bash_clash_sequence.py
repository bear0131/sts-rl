"""Strict two-card Bash/Clash order evaluation.

The opening state is accepted only when energy is exactly two and the only
legal PLAY_CARD entries are Bash and Clash.  Other cards may be present in the
hand, but they must be unaffordable; the draw pile remains populated so NOSL
future worlds are not collapsed to an empty deck.  Each trial searches the
root, executes the selected complete native card action, then searches the
resulting state once more.  This distinguishes a root preference from an
actual two-card order.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import slaythespire as sts
from sb3_contrib import MaskablePPO

from env.core import _BUILD  # noqa: F401
from env.core import action_space as A
from rl.collect_nosl_corpus import _load_custom_objects
from rl.eval_gumbel_bash_clash_trials import (
    FILLER_PROFILES,
    _hand,
    _legal_cards,
    _make_trial_env,
)
from rl.eval_nosl_order_sensitivity import _run_search
from rl.nosl_sampler import NOSLSampler


PAIR = {"bash", "clash"}


def _classify(action: str) -> str:
    """Keep End Turn distinct from an unrecognized action."""
    upper = str(action).upper()
    if "BASH" in upper:
        return "bash"
    if "CLASH" in upper:
        return "clash"
    if "END_TURN" in upper:
        return "end_turn"
    return "other"


def _strict_root(env) -> tuple[list[dict[str, Any]], list[str]]:
    raw = env.thick.bc.to_dict()
    energy = int(raw["player"]["energy"])
    hand = _hand(env)
    legal = _legal_cards(env, hand)
    if energy != 2:
        raise RuntimeError(f"strict root energy changed: {energy}")
    if sorted(legal) != ["BASH", "CLASH"]:
        raise RuntimeError(f"strict root has unexpected legal cards: {legal}")
    return hand, legal


def _execute_selected(env, search: dict[str, Any]) -> None:
    """Apply the complete native card+target action returned by the search."""
    bits = int(search["search"]["native_action_bits"])
    env.thick.bc.execute(sts.Action(bits))
    env._invalidate_backend_cache()


def _remaining_root(env, expected: str) -> list[str]:
    hand = _hand(env)
    legal = _legal_cards(env, hand)
    # End Turn is always a separate legal action; only the expected second card
    # may be playable at this point.  This rejects accidental third attacks.
    if legal != [expected]:
        raise RuntimeError(
            f"after first card expected only {expected}, got {legal}"
        )
    return legal


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["simulations"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (method, simulations), items in sorted(groups.items()):
        def count(name: str, field: str = "sequence") -> int:
            return sum(1 for item in items if item.get(field) == name)

        first_bash = count("bash", "first")
        first_clash = count("clash", "first")
        complete = [item for item in items if item.get("sequence") in {
            "bash_clash", "clash_bash"
        }]
        summaries.append({
            "method": method,
            "simulations": simulations,
            "trials": len(items),
            "root_first_bash": first_bash,
            "root_first_clash": first_clash,
            "root_end_turn": sum(
                item.get("first") == "end_turn" for item in items
            ),
            "root_other": sum(
                item.get("first") == "other" for item in items
            ),
            "bash_clash": count("bash_clash"),
            "clash_bash": count("clash_bash"),
            "second_end_turn": sum(
                item.get("second") == "end_turn" for item in items
            ),
            "second_other": sum(
                item.get("second") == "other" for item in items
            ),
            "second_censored": sum(
                item.get("second") == "censored" for item in items
            ),
            "second_other_or_censored": len(items) - len(complete),
            "bash_selection_rate": first_bash / len(items) if items else 0.0,
            "complete_sequence_rate": len(complete) / len(items) if items else 0.0,
            "mean_first_bash_visits": mean(
                float(item["first_bash_visits"]) for item in items
            ) if items else None,
            "mean_first_clash_visits": mean(
                float(item["first_clash_visits"]) for item in items
            ) if items else None,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[219, 357, 454, 575, 605],
        help="fixed seeds whose constructed roots satisfy the strict fixture",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--simulations", nargs="+", type=int,
                        default=[64, 128, 256, 512])
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--leaf-batch", type=int, default=64)
    parser.add_argument("--gumbel-candidates", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.seeds or args.repeats < 1:
        raise ValueError("--seeds must be non-empty and --repeats positive")

    model = MaskablePPO.load(
        args.checkpoint,
        device="cpu",
        custom_objects=_load_custom_objects(args.checkpoint, None),
    )
    fillers = FILLER_PROFILES["strict_root_normal_draw"]
    states: list[dict[str, Any]] = []
    for seed in args.seeds:
        for swapped in (False, True):
            env = _make_trial_env(
                model, seed=seed, swapped=swapped, fillers=fillers
            )
            try:
                hand, legal = _strict_root(env)
            except RuntimeError:
                # Most shuffled openings do not contain both cards.  They are
                # outside this fixture, not an experiment failure.
                del env
                continue
            raw = env.thick.bc.to_dict()
            states.append({
                "seed": int(seed),
                "swapped": bool(swapped),
                "energy": int(raw["player"]["energy"]),
                "hand": hand,
                "legal_cards": legal,
                "draw": [
                    {"id": str(card[0]), "up": int(card[1])}
                    for card in raw["draw"]
                ],
            })
    if not states:
        raise RuntimeError("no strict Bash+Clash roots found")

    sampler = NOSLSampler(0x4E4F534C_7631)
    rows: list[dict[str, Any]] = []
    methods = (("puct", 0, 0), ("gumbel", 1, int(args.gumbel_candidates)))
    for state_index, metadata in enumerate(states):
        for repetition in range(int(args.repeats)):
            root_nonce = sampler.nonce(
                state_nonce=(int(metadata["seed"]) << 1) |
                int(bool(metadata["swapped"])),
                simulation_index=300_000 + state_index * 1000 + repetition,
            )
            second_nonce = sampler.nonce(
                state_nonce=int(root_nonce),
                simulation_index=700_000 + repetition,
            )
            for simulations in args.simulations:
                for method, root_mode, candidates in methods:
                    env = _make_trial_env(
                        model,
                        seed=int(metadata["seed"]),
                        swapped=bool(metadata["swapped"]),
                        fillers=fillers,
                    )
                    _strict_root(env)
                    first_result = _run_search(
                        model, env, simulations=int(simulations),
                        depth=int(args.depth), nonce=int(root_nonce),
                        leaf_batch=int(args.leaf_batch), prior_mode="model",
                        root_search_mode=root_mode,
                        gumbel_candidates=candidates,
                    )
                    first = _classify(first_result["search"]["action"])
                    first_visits = first_result["search"].get("root_visits", {})
                    first_bash_visits = sum(
                        int(value) for key, value in first_visits.items()
                        if "BASH" in str(key).upper()
                    )
                    first_clash_visits = sum(
                        int(value) for key, value in first_visits.items()
                        if "CLASH" in str(key).upper()
                    )
                    row: dict[str, Any] = {
                        **metadata,
                        "state_index": state_index,
                        "repetition": repetition,
                        "nonce": int(root_nonce),
                        "method": method,
                        "simulations": int(simulations),
                        "depth": int(args.depth),
                        "gumbel_candidates": candidates,
                        "first": first,
                        "first_action": first_result["search"]["action"],
                        "first_bash_visits": int(first_bash_visits),
                        "first_clash_visits": int(first_clash_visits),
                        "second": "censored",
                        "sequence": "censored",
                        "second_root_visits": None,
                        "second_root_value": None,
                    }
                    if first not in PAIR:
                        rows.append(row)
                        continue

                    expected = "clash" if first == "bash" else "bash"
                    _execute_selected(env, first_result)
                    _remaining_root(env, expected.upper())
                    second_result = _run_search(
                        model, env, simulations=int(simulations),
                        depth=int(args.depth), nonce=int(second_nonce),
                        leaf_batch=int(args.leaf_batch), prior_mode="model",
                        root_search_mode=root_mode,
                        gumbel_candidates=candidates,
                    )
                    second = _classify(second_result["search"]["action"])
                    row["second"] = second
                    row["second_action"] = second_result["search"]["action"]
                    row["second_root_visits"] = (
                        second_result["search"].get("root_visits", {})
                    )
                    row["second_root_value"] = float(
                        second_result["search"]["root_value"]
                    )
                    if second == expected:
                        row["sequence"] = f"{first}_{second}"
                    else:
                        row["sequence"] = "censored"
                    rows.append(row)

    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "profile": "strict_root_normal_draw",
        "states": states,
        "repeats": int(args.repeats),
        "simulations": [int(value) for value in args.simulations],
        "depth": int(args.depth),
        "leaf_batch": int(args.leaf_batch),
        "gumbel_candidates": int(args.gumbel_candidates),
        "rows": rows,
        "summary": _summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"states={len(states)} rows={len(rows)} output={args.output.resolve()}")
    for item in payload["summary"]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
