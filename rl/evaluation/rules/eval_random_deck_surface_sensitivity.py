"""Counterfactual card-surface sensitivity on paired random decks.

For every pair, the simulator trace and target sequence stay fixed.  Only the
tested card's rule surface in the input prefix is changed, so a changed model
prediction is evidence that the model read the edited surface rather than a
different deck or a different outcome.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
from typing import Any, Sequence

from rl.evaluation.rules.eval_attack_damage_accuracy import _sample_mixed_deck
from rl.evaluation.rules.explain_card_play import (
    OpeningHandError,
    SandboxConfig,
    simulate_card_play,
)
import rl.rules.rule_data as rule_data


_NUMBER_AFTER_DAMAGE = re.compile(
    r"(?P<prefix>DEAL_DAMAGE\s+NUM_BEGIN\s+)"
    r"(?P<digits>(?:DIGIT_[0-9]\s*)+)NUM_END"
)
_TIMES_NUMBER = re.compile(
    r"(?P<prefix>\s+TIMES\s+NUM_BEGIN\s+)"
    r"(?P<digits>(?:DIGIT_[0-9]\s*)+)NUM_END"
)


def _number_tokens(value: int) -> str:
    if value < 0:
        raise ValueError("surface damage/hits must be non-negative")
    return " ".join(f"DIGIT_{digit}" for digit in str(value))


def replace_surface(
    surface: str, *, damage: int, hits: int | None = None,
) -> str:
    """Replace complete damage/hit number spans in a card-rule sentence."""
    result, count = _NUMBER_AFTER_DAMAGE.subn(
        lambda match: (
            f"{match.group('prefix')}{_number_tokens(damage)} NUM_END"
        ),
        surface,
        count=1,
    )
    if count != 1:
        raise ValueError(f"card surface has no fixed DEAL_DAMAGE number: {surface}")
    if hits is not None:
        result, hit_count = _TIMES_NUMBER.subn(
            lambda match: (
                f"{match.group('prefix')}{_number_tokens(hits)} NUM_END"
            ),
            result,
            count=1,
        )
        if hit_count == 0 and hits != 1:
            result = result.replace(
                " TARGET_ENEMY STMT_END",
                f" TARGET_ENEMY TIMES NUM_BEGIN {_number_tokens(hits)} NUM_END STMT_END",
                1,
            )
        elif hit_count == 1 and hits == 1:
            result = _TIMES_NUMBER.sub("", result, count=1)
    return result


def _surface_from_rules(sample: dict[str, Any], card_id: str, upgraded: bool) -> str:
    marker = f"CARD {card_id} VERSION {'upgraded' if upgraded else 'base'} "
    clauses = str(sample["rules"]["deck"]).split(" ; COUNT ")
    matches = [
        clause[clause.index("CARD "):]
        for clause in clauses
        if marker in clause and " CARD " in f" {clause}"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {marker!r} clause in random deck, found {len(matches)}"
        )
    return matches[0]


def _replace_deck_clause(
    deck_rules: str, original: str, replacement: str,
) -> str:
    if deck_rules.count(original) != 1:
        raise ValueError("target card rule clause is not unique in sampled deck")
    return deck_rules.replace(original, replacement, 1)


def _first_read_number(tokens: Sequence[str]) -> tuple[int, int | None]:
    try:
        marker = tokens.index("CALCULATION_READ_BASE_DAMAGE")
        begin = tokens.index("NUM_LSD_BEGIN", marker)
        end = tokens.index("NUM_END", begin)
    except ValueError:
        return -1, None
    digits = [token[-1] for token in tokens[begin + 1:end]
              if token.startswith("DIGIT_")]
    return begin, int("".join(reversed(digits))) if digits else None


def _first_difference(left: Sequence[str], right: Sequence[str]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _model_outputs(
    sample: dict[str, Any], run: Path, checkpoint: str, device: str,
    max_new_tokens: int | None,
) -> dict[str, Any]:
    # Reuse the established comparison implementation after applying the
    # temporary rule-language override below.
    from rl.evaluation.rules.explain_card_play import compare_model_outputs
    return compare_model_outputs(
        sample, run, checkpoint, device_name=device,
        max_new_tokens=max_new_tokens,
    )


def _variant(
    sample: dict[str, Any], *, card_id: str, upgraded: bool,
    original_surface: str, damage: int, hits: int | None,
    run: Path, checkpoint: str, device: str, max_new_tokens: int | None,
) -> dict[str, Any]:
    variant = copy.deepcopy(sample)
    edited = replace_surface(original_surface, damage=damage, hits=hits)
    variant["rules"]["deck"] = _replace_deck_clause(
        variant["rules"]["deck"], original_surface, edited,
    )
    original_rule = rule_data.card_rule_sentence

    def overridden(candidate_id: str, upgrades: int = 0) -> str:
        if candidate_id == card_id and int(upgrades) == int(upgraded):
            return edited
        return original_rule(candidate_id, upgrades)

    rule_data.card_rule_sentence = overridden
    try:
        output = _model_outputs(
            variant, run, checkpoint, device, max_new_tokens,
        )
    finally:
        rule_data.card_rule_sentence = original_rule
    read_position, target_value = _first_read_number(output["teacher_target"])
    predicted_value = None
    if read_position >= 0 and read_position < len(output["teacher_forced"]):
        _position, predicted_value = _first_read_number(
            output["teacher_forced"]
        )
        if _position != read_position:
            predicted_value = None
    return {
        "damage": damage,
        "hits": hits,
        "edited_surface": edited,
        "teacher_target_read": target_value,
        "teacher_forced_read": predicted_value,
        "teacher_forced_matches_edited_surface": (
            predicted_value == damage
        ),
        "teacher_forced": output["teacher_forced"],
        "autoregressive": output["autoregressive"],
        "teacher_target": output["teacher_target"],
        "autoregressive_first_mismatch": output[
            "autoregressive_first_mismatch"
        ],
    }


def run_pair(
    *, card_id: str, upgraded: bool, damage: int, hits: int | None,
    modified_damage: Sequence[int], modified_hits: Sequence[int | None],
    seed: int, trials: int, deck_attempts: int, opening_hand_attempts: int,
    enemy: str, player_energy: int, player_block: int,
    run: Path, checkpoint: str, device: str, max_new_tokens: int | None,
) -> list[dict[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    if len(modified_damage) != len(modified_hits):
        raise ValueError("modified damage/hits lists must have equal length")
    rows = []
    for trial in range(trials):
        simulation = None
        deck = None
        floor = None
        shuffle_seed = None
        for deck_attempt in range(deck_attempts):
            deck, floor = _sample_mixed_deck(rng, card_id, upgraded)
            shuffle_seed = int(rng.integers(0, (1 << 63) - opening_hand_attempts))
            try:
                simulation = simulate_card_play(SandboxConfig(
                    card_id=card_id, upgraded=upgraded, enemy=enemy,
                    seed=shuffle_seed, floor=floor,
                    player_energy=player_energy, player_block=player_block,
                    deck=deck, opening_hand_attempts=opening_hand_attempts,
                ))
            except OpeningHandError:
                continue
            break
        if simulation is None:
            raise OpeningHandError(
                f"failed to sample a playable random deck for {card_id}"
            )
        original_surface = _surface_from_rules(
            simulation["sample"], card_id, upgraded,
        )
        frozen_sample = copy.deepcopy(simulation["sample"])
        frozen_sample["local_calculations"] = rule_data.derive_local_calculations(
            frozen_sample
        )
        frozen_sample["rule_steps"] = rule_data.derive_rule_steps(frozen_sample)
        variants = [(damage, hits)] + list(zip(modified_damage, modified_hits))
        outputs = [
            _variant(
                frozen_sample, card_id=card_id, upgraded=upgraded,
                original_surface=original_surface,
                damage=variant_damage, hits=variant_hits,
                run=run, checkpoint=checkpoint, device=device,
                max_new_tokens=max_new_tokens,
            )
            for variant_damage, variant_hits in variants
        ]
        baseline = outputs[0]
        rows.append({
            "trial": trial,
            "deck_attempt": deck_attempt,
            "floor": floor,
            "shuffle_seed": simulation["actual_seed"],
            "deck": deck,
            "card_id": card_id,
            "upgraded": upgraded,
            "surface": original_surface,
            "variants": outputs,
            "changed_teacher_forced": [
                _first_difference(baseline["teacher_forced"], output["teacher_forced"])
                for output in outputs[1:]
            ],
            "changed_autoregressive": [
                _first_difference(baseline["autoregressive"], output["autoregressive"])
                for output in outputs[1:]
            ],
        })
    return rows


def _accuracy(rows: Sequence[dict[str, Any]], variant_index: int) -> dict[str, Any]:
    expected = []
    predicted = []
    edited = []
    changed_tf = 0
    changed_ar = 0
    for row in rows:
        variant = row["variants"][variant_index]
        expected.append(variant["teacher_target_read"])
        predicted.append(variant["teacher_forced_read"])
        edited.append(variant["damage"])
        if variant_index > 0 and row["changed_teacher_forced"][variant_index - 1] is not None:
            changed_tf += 1
        if variant_index > 0 and row["changed_autoregressive"][variant_index - 1] is not None:
            changed_ar += 1
    return {
        "target_values": expected,
        "teacher_forced_values": predicted,
        "teacher_forced_read_exact": sum(a == b for a, b in zip(expected, predicted)),
        "teacher_forced_read_total": len(expected),
        "teacher_forced_matches_edited_surface": sum(
            a == b for a, b in zip(edited, predicted)
        ),
        "edited_surface_values": edited,
        "teacher_forced_changed_vs_baseline": changed_tf,
        "autoregressive_changed_vs_baseline": changed_ar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True)
    parser.add_argument("--upgraded", action="store_true")
    parser.add_argument("--damage", type=int, required=True)
    parser.add_argument("--hits", type=int, default=None)
    parser.add_argument("--modified-damage", type=int, action="append", required=True)
    parser.add_argument("--modified-hits", type=int, action="append")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--deck-attempts", type=int, default=64)
    parser.add_argument("--opening-hand-attempts", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--enemy", default="Cultist")
    parser.add_argument("--player-energy", type=int, default=4)
    parser.add_argument("--player-block", type=int, default=10)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="last-strike-plus.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    modified_hits = args.modified_hits
    if modified_hits is None:
        modified_hits = [args.hits] * len(args.modified_damage)
    if len(modified_hits) != len(args.modified_damage):
        raise ValueError("--modified-hits must match --modified-damage count")
    rows = run_pair(
        card_id=args.card, upgraded=args.upgraded,
        damage=args.damage, hits=args.hits,
        modified_damage=args.modified_damage, modified_hits=modified_hits,
        seed=args.seed, trials=args.trials,
        deck_attempts=args.deck_attempts,
        opening_hand_attempts=args.opening_hand_attempts,
        enemy=args.enemy, player_energy=args.player_energy,
        player_block=args.player_block, run=args.run,
        checkpoint=args.checkpoint, device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    summary = {
        f"variant_{index}": _accuracy(rows, index)
        for index in range(len(rows[0]["variants"]))
    }
    result = {
        "settings": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "modified_hits": modified_hits,
        },
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    report = [
        "Random mixed-deck card surface sensitivity",
        f"card={args.card} version={'+' if args.upgraded else 'base'} trials={args.trials}",
        "Each row keeps the same random deck, state, shuffle seed, and simulator target.",
        "Only the target card surface in the input prefix changes.",
        "",
        "variant | frozen target READ_BASE_DAMAGE | edited surface damage | "
        "teacher-forced read | matches edited | matches frozen target | "
        "TF changed vs baseline | AR changed vs baseline",
    ]
    for index, (damage, hits) in enumerate(
        [(args.damage, args.hits)] + list(zip(args.modified_damage, modified_hits))
    ):
        stats = summary[f"variant_{index}"]
        report.append(
            f"{damage}x{hits} | {stats['target_values']} | {damage} | "
            f"{stats['teacher_forced_values']} | "
            f"{stats['teacher_forced_matches_edited_surface']}/"
            f"{stats['teacher_forced_read_total']} | "
            f"{stats['teacher_forced_read_exact']}/"
            f"{stats['teacher_forced_read_total']} | "
            f"{stats['teacher_forced_changed_vs_baseline']}/{args.trials} | "
            f"{stats['autoregressive_changed_vs_baseline']}/{args.trials}"
        )
    args.out.with_suffix(".md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(args.out.with_suffix(".md").resolve())
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
