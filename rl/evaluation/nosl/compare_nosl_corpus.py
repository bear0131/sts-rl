"""Aggregate the first NOSL-vs-PPO comparison from collector JSONL records."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("corpus contains no records")
    action_differences = 0
    margins: list[float] = []
    ppo_means: list[float] = []
    nosl_means: list[float] = []
    timeout_rates: list[float] = []
    actions_ppo: Counter[str] = Counter()
    actions_nosl: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    for record in records:
        ppo = int(record["ppo_action"])
        nosl = record.get("nosl") or {}
        chosen = int(nosl["action"])
        actions_ppo[str(ppo)] += 1
        actions_nosl[str(chosen)] += 1
        by_phase[str(record.get("phase", "unknown"))] += 1
        action_differences += int(ppo != chosen)
        means = {int(key): _finite(value) for key, value in (nosl.get("means_terminal_hp") or {}).items()}
        chosen_mean = means.get(chosen)
        ppo_mean = means.get(ppo)
        if chosen_mean is not None:
            nosl_means.append(chosen_mean)
        if ppo_mean is not None:
            ppo_means.append(ppo_mean)
        if chosen_mean is not None and ppo_mean is not None:
            margins.append(chosen_mean - ppo_mean)
        completed = {int(key): int(value) for key, value in (nosl.get("completed") or {}).items()}
        timeouts = {int(key): int(value) for key, value in (nosl.get("timeouts") or {}).items()}
        total = sum(completed.values()) + sum(timeouts.values())
        if total:
            timeout_rates.append(sum(timeouts.values()) / total)

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "schema_version": 1,
        "records": len(records),
        "phase_counts": dict(by_phase),
        "action_difference_rate": action_differences / len(records),
        "actions_ppo": dict(actions_ppo),
        "actions_nosl": dict(actions_nosl),
        "mean_terminal_hp_estimate_ppo_action": mean(ppo_means),
        "mean_terminal_hp_estimate_nosl_action": mean(nosl_means),
        "mean_nosl_gain_over_ppo_action": mean(margins),
        "positive_nosl_gain_fraction": (
            sum(value > 0 for value in margins) / len(margins)
            if margins
            else None
        ),
        "mean_mc_timeout_rate": mean(timeout_rates),
        "states_with_finite_ppo_estimate": len(ppo_means),
        "states_with_finite_nosl_estimate": len(nosl_means),
    }


def main() -> None:
    args = build_parser().parse_args()
    if not args.corpus.is_file():
        raise FileNotFoundError(args.corpus)
    records: list[dict[str, Any]] = []
    with args.corpus.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
    summary = summarize(records)
    summary["corpus"] = str(args.corpus.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"[nosl] wrote comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
