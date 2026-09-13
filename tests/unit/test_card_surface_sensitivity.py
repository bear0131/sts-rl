"""Probe whether an anonymous-card rule model responds to card-surface edits."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

from rl.evaluation.rules.explain_card_play import compare_model_outputs
import rl.rules.rule_data as rule_data


def surface(base_surface: str, base_damage: int, hits: int) -> str:
    """Replace the damage and hit literals while preserving card grammar."""
    result = re.sub(
        r"DEAL_DAMAGE NUM_BEGIN DIGIT_\d+ NUM_END",
        f"DEAL_DAMAGE NUM_BEGIN DIGIT_{base_damage} NUM_END",
        base_surface,
        count=1,
    )
    hit_match = re.search(
        r" TIMES NUM_BEGIN DIGIT_\d+ NUM_END(?= STMT_END)", result
    )
    if hits == 1 and hit_match:
        result = result[:hit_match.start()] + result[hit_match.end():]
    elif hits != 1 and hit_match:
        result = (
            result[:hit_match.start()]
            + f" TIMES NUM_BEGIN DIGIT_{hits} NUM_END"
            + result[hit_match.end():]
        )
    elif hits != 1:
        result = result.replace(
            " TARGET_ENEMY STMT_END",
            f" TARGET_ENEMY TIMES NUM_BEGIN DIGIT_{hits} NUM_END STMT_END",
            1,
        )
    return result


def build_variant(
    sample: dict, base_damage: int, hits: int,
    frozen_local_calculations: list[list[dict]],
    frozen_rule_steps: list[dict], card_name: str, base_surface: str,
) -> dict:
    variant = copy.deepcopy(sample)
    text = surface(base_surface, base_damage, hits)
    variant["rules"]["deck"] = (
        "DECK ; COUNT 10 " + text
    )
    # Keep the simulator-produced target fixed. This is a prefix-only
    # counterfactual: any change in the model response must come from reading
    # the altered card surface, rather than from changed supervision.
    variant["local_calculations"] = copy.deepcopy(frozen_local_calculations)
    variant["rule_steps"] = copy.deepcopy(frozen_rule_steps)
    original = rule_data.card_rule_sentence

    def overridden(card_id: str, upgrades: int = 0) -> str:
        if card_id == card_name and int(upgrades) == 0:
            return text
        return original(card_id, upgrades)

    rule_data.card_rule_sentence = overridden
    try:
        # compare_model_outputs calls sample_tokens synchronously, so the
        # temporary rule-language override changes only the constructed prefix;
        # simulator effects and teacher target remain fixed for sensitivity.
        return compare_model_outputs(
            variant,
            Path(args.run),
            args.checkpoint,
            device_name=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        rule_data.card_rule_sentence = original


def _digit_after(tokens: list[str], marker: str) -> tuple[int, str] | None:
    try:
        position = tokens.index(marker)
        begin = tokens.index("NUM_LSD_BEGIN", position)
        return begin + 1, tokens[begin + 1]
    except ValueError:
        return None


def main() -> None:
    global args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument(
        "--card-name", required=True,
        help="card id whose rule surface is being edited",
    )
    parser.add_argument(
        "--base-damage", type=int, required=True,
        help="unmodified damage literal in the sample surface",
    )
    parser.add_argument(
        "--hits", type=int, required=True,
        help="unmodified hit count (1 means no TIMES clause)",
    )
    parser.add_argument(
        "--modified-damage", type=int, action="append", required=True,
        help="replacement damage literal; may be supplied more than once",
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="last-stable.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    base_surface = sample["rules"]["deck"].split("DECK ; COUNT 10 ", 1)[1]
    frozen_local_calculations = rule_data.derive_local_calculations(sample)
    frozen_rule_steps = rule_data.derive_rule_steps(sample)
    results = {}
    variants = [(args.base_damage, args.hits)] + [
        (damage, args.hits) for damage in args.modified_damage
    ]
    for base_damage, hits in variants:
        key = f"surface_{base_damage}x{hits}"
        results[key] = build_variant(
            sample, base_damage, hits,
            frozen_local_calculations, frozen_rule_steps,
            args.card_name, base_surface,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    baseline = results[f"surface_{args.base_damage}x{args.hits}"]
    report = [
        "Anonymous-card surface sensitivity report",
        f"checkpoint_epoch={baseline['checkpoint_epoch']}",
        f"The teacher target is frozen to the real {args.card_name} simulator trace.",
        "Only the anonymous card surface in the input prefix changes.",
        "",
        "surface | teacher-forced base-damage prediction | "
        "teacher-forced target accuracy | autoregressive length | AR equal to baseline",
    ]
    for key, result in results.items():
        base_damage = _digit_after(
            result["teacher_target"], "CALCULATION_READ_BASE_DAMAGE"
        )
        predicted_base = (
            result["teacher_forced"][base_damage[0]]
            if base_damage is not None else "<missing>"
        )
        tf_accuracy = sum(
            want == got
            for want, got in zip(result["teacher_target"], result["teacher_forced"])
        ) / max(1, len(result["teacher_target"]))
        report.append(
            f"{key} | target={base_damage[1] if base_damage else '<missing>'}, "
            f"predicted={predicted_base} "
            f"at target position {base_damage[0] if base_damage else '<missing>'} | "
            f"{tf_accuracy:.3%} | {len(result['autoregressive'])} | "
            f"{result['autoregressive'] == baseline['autoregressive']}"
        )
    report.extend((
        "",
        "Teacher-forced differences versus the baseline surface:",
    ))
    for key, result in results.items():
        if key == f"surface_{args.base_damage}x{args.hits}":
            continue
        differences = [
            (position, baseline["teacher_forced"][position], result["teacher_forced"][position])
            for position, (left, right) in enumerate(zip(
                baseline["teacher_forced"], result["teacher_forced"]
            )) if left != right
        ]
        report.append(f"{key}: {differences}")
    args.out.with_suffix(".txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    for key, result in results.items():
        print(
            key,
            "epoch=", result["checkpoint_epoch"],
            "target=", len(result["teacher_target"]),
            "ar=", len(result["autoregressive"]),
            "first_ar_mismatch=", result["autoregressive_first_mismatch"],
        )


if __name__ == "__main__":
    main()
