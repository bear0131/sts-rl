"""Evaluate damage predictions for every Ironclad Attack and upgrade."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# This import selects the rule-trace simulator build before env.core is cached.
from rl.explain_card_play import (
    OpeningHandError,
    SandboxConfig,
    _load_compatible_tokenizer,
    _strip_unsupported_target_sections,
    simulate_card_play,
)
from env.core.card_language import card_program
from env.core.ironclad_cards import ATTACK, IRONCLAD_CARDS
from rl.rule_data import OUTPUT_BEGIN, RuleTokenizer, sample_tokens
from rl.rule_model import RuleWorldModel


def _number_span(tokens: Sequence[str], begin: int) -> tuple[int, int] | None:
    if begin >= len(tokens) or tokens[begin] not in ("NUM_BEGIN", "NUM_LSD_BEGIN"):
        return None
    try:
        end = tokens.index("NUM_END", begin + 1)
    except ValueError:
        return None
    return begin, end + 1


def _decode_number(tokens: Sequence[str]) -> int | None:
    if len(tokens) < 3 or tokens[0] not in ("NUM_BEGIN", "NUM_LSD_BEGIN"):
        return None
    if tokens[-1] != "NUM_END":
        return None
    negative = False
    digits: list[str] = []
    for token in tokens[1:-1]:
        if token == "SIGN_NEG" and not digits and not negative:
            negative = True
        elif token.startswith("DIGIT_") and len(token) == 7:
            digits.append(token[-1])
        else:
            return None
    if not digits:
        return None
    if tokens[0] == "NUM_LSD_BEGIN":
        digits.reverse()
    value = int("".join(digits))
    return -value if negative else value


def _aligned_number(
    target: Sequence[str], predicted: Sequence[str], begin: int,
) -> dict[str, Any]:
    span = _number_span(target, begin)
    if span is None:
        raise ValueError(f"invalid target number at token {begin}")
    left, right = span
    expected_tokens = list(target[left:right])
    aligned_predicted_tokens = list(predicted[left:right])
    predicted_span = _number_span(predicted, left)
    predicted_tokens = (
        list(predicted[predicted_span[0]:predicted_span[1]])
        if predicted_span is not None else aligned_predicted_tokens
    )
    return {
        "position": left,
        "expected": _decode_number(expected_tokens),
        "predicted": _decode_number(predicted_tokens),
        "correct": expected_tokens == aligned_predicted_tokens,
        "expected_tokens": expected_tokens,
        "predicted_tokens": predicted_tokens,
        "aligned_predicted_tokens": aligned_predicted_tokens,
    }


def _base_damage_checks(
    target: Sequence[str], predicted: Sequence[str],
) -> list[dict[str, Any]]:
    checks = []
    for position, token in enumerate(target):
        if token != "CALCULATION_READ_BASE_DAMAGE":
            continue
        if position + 2 >= len(target) or target[position + 1] != "CALC_FIELD_value":
            raise ValueError(
                f"READ_BASE_DAMAGE at {position} has no CALC_FIELD_value"
            )
        check = _aligned_number(target, predicted, position + 2)
        check["marker_position"] = position
        check["marker_correct"] = (
            position < len(predicted) and predicted[position] == token
        )
        checks.append(check)
    return checks


def _effect_blocks(tokens: Sequence[str]):
    cursor = 0
    while cursor < len(tokens):
        try:
            left = tokens.index("EFFECT_BEGIN", cursor)
            right = tokens.index("EFFECT_END", left + 1)
        except ValueError:
            return
        yield left, right
        cursor = right + 1


def _hp_damage_checks(
    target: Sequence[str], predicted: Sequence[str],
) -> list[dict[str, Any]]:
    checks = []
    for left, right in _effect_blocks(target):
        block = target[left:right]
        if "EFFECT_MONSTER_VALUE_CHANGE" not in block:
            continue
        if not any(
            block[index:index + 2] == ["FIELD_field", "VALUE_hp"]
            for index in range(max(0, len(block) - 1))
        ):
            continue
        try:
            delta = target.index("FIELD_delta", left, right)
        except ValueError:
            continue
        check = _aligned_number(target, predicted, delta + 1)
        if check["expected"] is not None:
            check["expected"] = abs(check["expected"])
        if check["predicted"] is not None:
            check["predicted"] = abs(check["predicted"])
        check["field_position"] = delta
        check["field_correct"] = (
            delta < len(predicted) and predicted[delta] == "FIELD_delta"
        )
        checks.append(check)
    return checks


def _surface_damage(card_id: str, upgraded: bool) -> str:
    program = card_program(card_id, upgraded)
    if "DEAL_DAMAGE_FROM" in program or "DEAL_DAMAGE_FORMULA" in program:
        return "dynamic"
    damage = IRONCLAD_CARDS[card_id].damage[int(upgraded)]
    if "PLUS" in program or "PER" in program:
        return f"{damage} + dynamic"
    try:
        times = program.index("TIMES")
    except ValueError:
        return str(damage)
    if times + 1 < len(program) and program[times + 1] == "NUM_BEGIN":
        span = _number_span(program, times + 1)
        hit_count = _decode_number(program[span[0]:span[1]]) if span else None
        return f"{damage} x {hit_count if hit_count is not None else 'X'}"
    return f"{damage} x X"


def _pad(rows: Sequence[Sequence[int]], pad_id: int, device: torch.device):
    return pad_sequence(
        [torch.tensor(row, dtype=torch.long) for row in rows],
        batch_first=True,
        padding_value=pad_id,
    ).to(device)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _rate(correct: int, total: int) -> str:
    return f"{correct}/{total} ({correct / total:.2%})" if total else "n/a"


def _values(checks: Sequence[dict[str, Any]], key: str) -> str:
    if not checks:
        return "-"
    return ", ".join("?" if check[key] is None else str(check[key]) for check in checks)


def _version_label(upgraded: bool) -> str:
    return "+" if upgraded else "base"


def _sample_mixed_deck(
    rng: np.random.Generator, card_id: str, upgraded: bool,
) -> tuple[tuple[tuple[str, int], ...], int]:
    """Sample the training deck-size distribution with one forced candidate."""
    floor = int(rng.integers(1, 51))
    remove_top = min(int(round(2.0 * 0.10 * floor)), 9)
    removed = int(rng.integers(0, remove_top + 1)) if remove_top else 0
    mean_added = 15.0 * (1.0 - math.exp(-0.06 * floor))
    added_top = int(round(2.0 * mean_added))
    added = int(rng.integers(0, added_top + 1)) if added_top else 0
    deck_size = 10 - removed + added

    filler_ids = [candidate for candidate in IRONCLAD_CARDS if candidate != card_id]
    deck = [
        (
            filler_ids[int(rng.integers(0, len(filler_ids)))],
            int(rng.integers(0, 2)),
        )
        for _ in range(deck_size - 1)
    ]
    target_position = int(rng.integers(0, deck_size))
    deck.insert(target_position, (card_id, int(upgraded)))
    return tuple(deck), floor


def _value_hist(checks: Sequence[dict[str, Any]], key: str) -> str:
    if not checks:
        return "-"
    counts = Counter(
        "?" if check[key] is None else str(check[key]) for check in checks
    )
    return ", ".join(
        f"{value} ({count}x)" if count > 1 else value
        for value, count in sorted(counts.items())
    )


def build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    settings = result["settings"]
    lines = [
        "# Ironclad Attack damage accuracy",
        "",
        f"- Checkpoint: `{settings['checkpoint']}` (epoch {settings['checkpoint_epoch']})",
        f"- Test state: energy={settings['player_energy']}, "
        f"block={settings['player_block']}, strength=0, enemy block=0, vulnerable=0",
        f"- Random mixed decks: {settings['trials_per_version']} per card version, "
        f"{settings['deck_size_min']}-{settings['deck_size_max']} cards; each deck "
        "contains exactly one copy of the tested card",
        f"- Fixed base-damage reads: {_rate(summary['base_read_correct'], summary['base_read_total'])}",
        f"- Base-version reads: "
        f"{_rate(summary['base_only_read_correct'], summary['base_only_read_total'])}; "
        f"exact trials: "
        f"{_rate(summary['base_only_trial_correct'], summary['base_only_trial_total'])}",
        f"- Upgraded-version reads: "
        f"{_rate(summary['upgraded_read_correct'], summary['upgraded_read_total'])}; "
        f"exact trials: "
        f"{_rate(summary['upgraded_trial_correct'], summary['upgraded_trial_total'])}",
        f"- Exact fixed-damage trials: "
        f"{_rate(summary['base_trial_correct'], summary['base_trial_total'])}",
        f"- Versions correct in every random deck: "
        f"{_rate(summary['robust_version_correct'], summary['robust_version_total'])}",
        f"- Teacher-forced HP-damage outputs: "
        f"{_rate(summary['hp_damage_correct'], summary['hp_damage_total'])}",
        f"- Exact HP-damage trials: "
        f"{_rate(summary['hp_trial_correct'], summary['hp_trial_total'])}",
        f"- Dynamic-formula trials without READ_BASE_DAMAGE: "
        f"{summary['dynamic_trial_total']}",
        "",
        "`READ_BASE_DAMAGE` is the primary metric: it tests whether the card-surface "
        "damage was copied into the local calculation. HP-damage output is auxiliary "
        "because teacher forcing supplies the gold calculation prefix before it.",
        "",
        "| Card | Version | Surface | Base target(s) | Base prediction(s) | "
        "Exact trials | HP target(s) | HP prediction(s) | Exact HP trials |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['card_id']} | {row['version']} | {row['surface_damage']} | "
            f"{_value_hist(row['base_checks'], 'expected')} | "
            f"{_value_hist(row['base_checks'], 'predicted')} | "
            f"{_rate(row['base_trial_correct'], row['base_trial_total'])} | "
            f"{_value_hist(row['hp_checks'], 'expected')} | "
            f"{_value_hist(row['hp_checks'], 'predicted')} | "
            f"{_rate(row['hp_trial_correct'], row['hp_trial_total'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--player-energy", type=int, default=4)
    parser.add_argument("--player-block", type=int, default=10)
    parser.add_argument("--enemy", default="Cultist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--opening-hand-attempts", type=int, default=512)
    parser.add_argument("--deck-attempts", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.opening_hand_attempts < 1:
        raise ValueError("--opening-hand-attempts must be positive")
    if args.deck_attempts < 1:
        raise ValueError("--deck-attempts must be positive")

    tokenizer_path = args.run / "tokenizer.json"
    tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_version = int(tokenizer_data.get("version", -1))
    tokenizer: RuleTokenizer = _load_compatible_tokenizer(tokenizer_path)
    model = RuleWorldModel.from_config(args.run / "model_config.json")
    checkpoint_path = args.run / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    device = _device(args.device)
    model.to(device).eval()

    rng = np.random.default_rng(args.seed)
    encoded = []
    for card_id, info in IRONCLAD_CARDS.items():
        if info.kind != ATTACK:
            continue
        for upgraded in (False, True):
            for trial in range(args.trials):
                simulation = None
                for deck_attempt in range(args.deck_attempts):
                    deck, floor = _sample_mixed_deck(
                        rng, card_id, upgraded,
                    )
                    shuffle_seed = int(rng.integers(
                        0, (1 << 63) - args.opening_hand_attempts,
                    ))
                    try:
                        simulation = simulate_card_play(SandboxConfig(
                            card_id=card_id,
                            upgraded=upgraded,
                            enemy=args.enemy,
                            seed=shuffle_seed,
                            floor=floor,
                            player_energy=args.player_energy,
                            player_strength=0,
                            enemy_block=0,
                            enemy_vulnerable=0,
                            player_block=args.player_block,
                            deck=deck,
                            opening_hand_attempts=args.opening_hand_attempts,
                        ))
                    except OpeningHandError:
                        continue
                    break
                if simulation is None:
                    raise OpeningHandError(
                        f"failed to construct a playable {card_id} "
                        f"{'+' if upgraded else 'base'} trial after "
                        f"{args.deck_attempts} random decks"
                    )
                prefix, full_target = sample_tokens(
                    simulation["sample"], output_version=tokenizer_version,
                )
                target = _strip_unsupported_target_sections(full_target, tokenizer)
                prefix_ids = tokenizer.encode(prefix)
                target_ids = tokenizer.encode(target)
                unknown_id = tokenizer.index["<UNK>"]
                unknowns = [
                    ("prefix", index, token)
                    for index, (token, token_id) in enumerate(zip(prefix, prefix_ids))
                    if token_id == unknown_id
                ] + [
                    ("target", index, token)
                    for index, (token, token_id) in enumerate(zip(target, target_ids))
                    if token_id == unknown_id
                ]
                if unknowns:
                    raise ValueError(f"{card_id} has unknown tokens: {unknowns[:5]}")
                encoded.append({
                    "card_id": card_id,
                    "upgraded": upgraded,
                    "trial": trial,
                    "deck_attempt": deck_attempt,
                    "floor": floor,
                    "deck": deck,
                    "shuffle_seed": simulation["actual_seed"],
                    "surface_damage": _surface_damage(card_id, upgraded),
                    "prefix_ids": prefix_ids,
                    "target_ids": target_ids,
                    "target": target,
                })

    # Sorting limits padding overhead for multi-hit cards with long targets.
    encoded.sort(key=lambda item: len(item["target_ids"]))
    trial_rows = []
    with torch.inference_mode():
        for begin in range(0, len(encoded), args.batch_size):
            chunk = encoded[begin:begin + args.batch_size]
            prefix_ids = _pad(
                [item["prefix_ids"] for item in chunk], tokenizer.pad_id, device
            )
            target_ids = _pad(
                [item["target_ids"] for item in chunk], tokenizer.pad_id, device
            )
            logits, _latent = model(prefix_ids, target_ids[:, :-1])
            predicted_ids = logits.argmax(dim=-1).cpu()
            for index, item in enumerate(chunk):
                target = item["target"]
                predicted = [OUTPUT_BEGIN] + tokenizer.decode(
                    predicted_ids[index, :len(item["target_ids"]) - 1].tolist()
                )
                base_checks = _base_damage_checks(target, predicted)
                hp_checks = _hp_damage_checks(target, predicted)
                trial_rows.append({
                    "card_id": item["card_id"],
                    "upgraded": item["upgraded"],
                    "trial": item["trial"],
                    "deck_attempt": item["deck_attempt"],
                    "floor": item["floor"],
                    "deck_size": len(item["deck"]),
                    "deck": item["deck"],
                    "shuffle_seed": item["shuffle_seed"],
                    "surface_damage": item["surface_damage"],
                    "target_tokens": len(target),
                    "base_checks": base_checks,
                    "base_exact": bool(base_checks) and all(
                        check["correct"] for check in base_checks
                    ),
                    "hp_checks": hp_checks,
                    "hp_exact": bool(hp_checks) and all(
                        check["correct"] for check in hp_checks
                    ),
                })

    trial_rows.sort(key=lambda row: (
        list(IRONCLAD_CARDS).index(row["card_id"]),
        row["upgraded"], row["trial"],
    ))
    rows = []
    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for trial_row in trial_rows:
        grouped.setdefault(
            (trial_row["card_id"], trial_row["upgraded"]), []
        ).append(trial_row)
    for (card_id, upgraded), trials in grouped.items():
        base_trials = [trial for trial in trials if trial["base_checks"]]
        hp_trials = [trial for trial in trials if trial["hp_checks"]]
        rows.append({
            "card_id": card_id,
            "upgraded": upgraded,
            "version": _version_label(upgraded),
            "surface_damage": trials[0]["surface_damage"],
            "base_checks": [
                check for trial in trials for check in trial["base_checks"]
            ],
            "base_trial_correct": sum(
                trial["base_exact"] for trial in base_trials
            ),
            "base_trial_total": len(base_trials),
            "base_all_trials_correct": bool(base_trials) and all(
                trial["base_exact"] for trial in base_trials
            ),
            "hp_checks": [
                check for trial in trials for check in trial["hp_checks"]
            ],
            "hp_trial_correct": sum(
                trial["hp_exact"] for trial in hp_trials
            ),
            "hp_trial_total": len(hp_trials),
            "trials": trials,
        })

    base_checks = [check for row in rows for check in row["base_checks"]]
    hp_checks = [check for row in rows for check in row["hp_checks"]]
    base_rows = [row for row in rows if row["base_checks"]]
    hp_rows = [row for row in rows if row["hp_checks"]]
    base_only_rows = [row for row in base_rows if not row["upgraded"]]
    upgraded_rows = [row for row in base_rows if row["upgraded"]]
    base_only_checks = [
        check for row in base_only_rows for check in row["base_checks"]
    ]
    upgraded_checks = [
        check for row in upgraded_rows for check in row["base_checks"]
    ]
    deck_sizes = [row["deck_size"] for row in trial_rows]
    result = {
        "settings": {
            "run": str(args.run),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "device": str(device),
            "player_energy": args.player_energy,
            "player_block": args.player_block,
            "enemy": args.enemy,
            "seed": args.seed,
            "trials_per_version": args.trials,
            "opening_hand_attempts": args.opening_hand_attempts,
            "deck_attempts": args.deck_attempts,
            "deck_size_min": min(deck_sizes),
            "deck_size_max": max(deck_sizes),
            "deck_distribution": (
                "rule-training floor-dependent size; uniformly sampled Ironclad "
                "fillers with replacement; exactly one tested card"
            ),
        },
        "summary": {
            "attack_cards": len({row["card_id"] for row in rows}),
            "versions": len(rows),
            "trials": len(trial_rows),
            "base_read_correct": sum(check["correct"] for check in base_checks),
            "base_read_total": len(base_checks),
            "base_only_read_correct": sum(
                check["correct"] for check in base_only_checks
            ),
            "base_only_read_total": len(base_only_checks),
            "base_only_trial_correct": sum(
                row["base_trial_correct"] for row in base_only_rows
            ),
            "base_only_trial_total": sum(
                row["base_trial_total"] for row in base_only_rows
            ),
            "upgraded_read_correct": sum(
                check["correct"] for check in upgraded_checks
            ),
            "upgraded_read_total": len(upgraded_checks),
            "upgraded_trial_correct": sum(
                row["base_trial_correct"] for row in upgraded_rows
            ),
            "upgraded_trial_total": sum(
                row["base_trial_total"] for row in upgraded_rows
            ),
            "base_trial_correct": sum(
                row["base_trial_correct"] for row in base_rows
            ),
            "base_trial_total": sum(
                row["base_trial_total"] for row in base_rows
            ),
            "robust_version_correct": sum(
                row["base_all_trials_correct"] for row in base_rows
            ),
            "robust_version_total": len(base_rows),
            "hp_damage_correct": sum(check["correct"] for check in hp_checks),
            "hp_damage_total": len(hp_checks),
            "hp_trial_correct": sum(row["hp_trial_correct"] for row in hp_rows),
            "hp_trial_total": sum(row["hp_trial_total"] for row in hp_rows),
            "dynamic_trial_total": sum(
                not row["base_checks"] for row in trial_rows
            ),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    report_path = args.out.with_suffix(".md")
    report_path.write_text(build_report(result), encoding="utf-8")
    print(report_path.resolve())
    print(json.dumps(result["summary"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
