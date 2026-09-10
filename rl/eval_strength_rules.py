"""Evaluate rule checkpoints on controlled random-Strength attack samples."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch

from rl.rule_data import (
    NEXT_STATE_BEGIN,
    OUTPUT_BEGIN,
    OUTPUT_END,
    RuleTokenizer,
    controlled_attack_calculations,
    critical_target_number_regions,
    read_jsonl,
    sample_tokens,
    transition_consistency_specs,
)
from rl.rule_model import RuleWorldModel
from rl.train_rules import (
    _loss_and_counts,
    _metrics,
    _transition_consistency_loss,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--greedy-samples", type=int, default=32)
    parser.add_argument("--greedy-max-tokens", type=int, default=256)
    parser.add_argument("--show-errors", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _pad(rows, pad_id, device):
    tensors = [torch.tensor(row, dtype=torch.long) for row in rows]
    return torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=pad_id
    ).to(device)


def _parse_number(tokens, position):
    if position >= len(tokens) or tokens[position] not in (
        "NUM_BEGIN", "NUM_LSD_BEGIN"
    ):
        return None
    least_significant_first = tokens[position] == "NUM_LSD_BEGIN"
    position += 1
    sign = 1
    if position < len(tokens) and tokens[position] == "SIGN_NEG":
        sign = -1
        position += 1
    digits = []
    while position < len(tokens) and tokens[position].startswith("DIGIT_"):
        digits.append(tokens[position][6:])
        position += 1
    if not digits or position >= len(tokens) or tokens[position] != "NUM_END":
        return None
    if least_significant_first:
        digits.reverse()
    return sign * int("".join(digits))


def _calculation_result(tokens, calculation_type):
    position = 0
    marker = f"CALCULATION_{calculation_type}"
    while position < len(tokens):
        try:
            calculation = tokens.index("CALCULATION_BEGIN", position)
            end = tokens.index("CALCULATION_END", calculation)
        except ValueError:
            return None
        if marker in tokens[calculation:end]:
            return _number_after(
                tokens, "CALC_FIELD_result", start=calculation,
                stop_marker="CALCULATION_END",
            )
        position = end + 1
    return None


def _calculation_checks(tokens, sample):
    calculations = sample.get("calculations")
    if calculations is None:
        calculations = controlled_attack_calculations(sample)
    return {
        calculation["type"]: _calculation_result(tokens, calculation["type"])
        == int(calculation["result"])
        for calculation in calculations
        if "result" in calculation
    }


def _number_after(tokens, marker, *, start=0, stop_marker=None):
    try:
        marker_position = tokens.index(marker, start)
    except ValueError:
        return None
    if stop_marker is not None:
        try:
            stop = tokens.index(stop_marker, start)
        except ValueError:
            stop = len(tokens)
        if marker_position >= stop:
            return None
    return _parse_number(tokens, marker_position + 1)


def _effect_loss(tokens, field):
    try:
        effects_end = tokens.index("<EFFECTS_END>")
    except ValueError:
        return None
    total = 0
    position = 0
    while position < effects_end:
        try:
            event = tokens.index("EFFECT_MONSTER_VALUE_CHANGE", position, effects_end)
            end = tokens.index("EFFECT_END", event, effects_end)
        except ValueError:
            break
        if f"VALUE_{field}" in tokens[event:end]:
            value = _number_after(
                tokens, "FIELD_delta", start=event, stop_marker="EFFECT_END"
            )
            if value is None:
                return None
            if value < 0:
                total -= value
        position = end + 1
    return total


def _aligned_damage(tokens):
    return _effect_loss(tokens, "hp")


def _aligned_next_monster_value(tokens, marker):
    try:
        state = tokens.index(NEXT_STATE_BEGIN)
        monster = tokens.index("MONSTER_BEGIN", state)
    except ValueError:
        return None
    return _number_after(tokens, marker, start=monster, stop_marker="MONSTER_END")


def _aligned_next_hp(tokens):
    return _aligned_next_monster_value(tokens, "HP")


def _aligned_next_block(tokens):
    return _aligned_next_monster_value(tokens, "BLOCK")


def _aligned_next_vulnerable(tokens):
    try:
        state = tokens.index(NEXT_STATE_BEGIN)
        monster = tokens.index("MONSTER_BEGIN", state)
        end = tokens.index("MONSTER_END", monster)
    except ValueError:
        return None
    position = monster
    while position < end:
        try:
            status = tokens.index("MONSTER_STATUS", position, end)
        except ValueError:
            return 0
        if status + 1 < end and tokens[status + 1] == "STATUS_Vulnerable":
            return _parse_number(tokens, status + 2)
        position = status + 1
    return 0


def _greedy_damage(tokens):
    return _effect_loss(tokens, "hp")


def _expected_next_hp(sample):
    return int(sample["next_state"]["combat"]["monsters"][0]["hp"])


def _expected_next_block(sample):
    return int(sample["next_state"]["combat"]["monsters"][0]["block"])


def main() -> None:
    args = build_parser().parse_args()
    samples = list(read_jsonl(args.data))
    if not samples:
        raise ValueError("evaluation data is empty")
    tokenizer_path = args.run / "tokenizer.json"
    output_version = int(json.loads(
        tokenizer_path.read_text(encoding="utf-8")
    ).get("version", -1))
    tokenizer = RuleTokenizer.load(tokenizer_path)
    model = RuleWorldModel.from_config(args.run / "model_config.json")
    checkpoint = torch.load(
        args.run / args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    run_config_path = args.run / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.exists()
        else {}
    )
    legacy_number_weight = float(run_config.get("critical_number_weight", 1.0))
    effect_number_weight = float(
        run_config.get("effect_number_weight") or legacy_number_weight
    )
    next_number_weight = float(
        run_config.get("next_number_weight") or legacy_number_weight
    )
    calculation_number_weight = float(
        run_config.get("calculation_number_weight") or effect_number_weight
    )
    consistency_loss_weight = float(run_config.get("consistency_loss_weight", 0.0))
    consistency_scale = float(run_config.get("consistency_scale", 10.0))
    device = _device(args.device)
    model.to(device).eval()

    encoded = []
    unknowns = 0
    for sample in samples:
        prefix, target = sample_tokens(sample, output_version=output_version)
        prefix_ids = tokenizer.encode(prefix)
        target_ids = tokenizer.encode(target)
        critical = critical_target_number_regions(target)
        specs = transition_consistency_specs(sample, target)
        unknowns += sum(token_id == tokenizer.index["<UNK>"] for token_id in prefix_ids)
        unknowns += sum(token_id == tokenizer.index["<UNK>"] for token_id in target_ids)
        encoded.append((prefix_ids, target_ids, target, critical, specs))

    totals = defaultdict(int)
    total_loss = 0.0
    total_consistency_loss = 0.0
    consistency_terms = 0
    damage_correct = next_hp_correct = 0
    status_experiment = all(
        "expected_block_loss" in sample.get("metadata", {}) for sample in samples
    )
    status_counts = defaultdict(int)
    calculation_counts = defaultdict(int)
    teacher_status_by_key = defaultdict(lambda: [0] * 7)
    greedy_status_by_key = defaultdict(lambda: [0] * 7)
    by_key = defaultdict(lambda: [0, 0, 0])
    digit_ids = torch.tensor(
        [tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)],
        device=device,
    )
    with torch.no_grad():
        for begin in range(0, len(samples), args.batch_size):
            chunk = encoded[begin:begin + args.batch_size]
            prefix_ids = _pad([item[0] for item in chunk], tokenizer.pad_id, device)
            target_ids = _pad([item[1] for item in chunk], tokenizer.pad_id, device)
            critical = _pad([item[3] for item in chunk], 0, device)
            logits, _latent = model(prefix_ids, target_ids[:, :-1])
            loss, counts = _loss_and_counts(
                logits, target_ids[:, 1:], tokenizer.pad_id,
                tokenizer.index[NEXT_STATE_BEGIN],
                critical[:, 1:], effect_number_weight, next_number_weight,
                calculation_number_weight,
            )
            consistency_loss, term_count = _transition_consistency_loss(
                logits, [item[4] for item in chunk], digit_ids, consistency_scale
            )
            total_loss += float(loss) * counts["loss_weight"]
            total_consistency_loss += float(consistency_loss) * term_count
            consistency_terms += term_count
            for key, value in counts.items():
                totals[key] += value
            predicted = logits.argmax(dim=-1).cpu()
            for row, (
                _prefix, target_row, target_tokens, _critical, _specs
            ) in enumerate(chunk):
                predicted_tokens = [target_tokens[0]] + tokenizer.decode(
                    predicted[row, :len(target_row) - 1].tolist()
                )
                sample = samples[begin + row]
                expected_damage = int(sample["metadata"]["expected_damage"])
                predicted_hp_loss = _aligned_damage(predicted_tokens)
                damage_ok = predicted_hp_loss == expected_damage
                predicted_next_hp = _aligned_next_hp(predicted_tokens)
                hp_ok = predicted_next_hp == _expected_next_hp(sample)
                damage_correct += int(damage_ok)
                next_hp_correct += int(hp_ok)
                calculation_checks = _calculation_checks(predicted_tokens, sample)
                if calculation_checks:
                    calculation_counts["teacher_all_total"] += 1
                    calculation_counts["teacher_all_hits"] += int(
                        all(calculation_checks.values())
                    )
                    for calculation_type, passed in calculation_checks.items():
                        calculation_counts[f"teacher_{calculation_type}_total"] += 1
                        calculation_counts[f"teacher_{calculation_type}_hits"] += int(
                            passed
                        )
                key = (int(sample["metadata"]["strength"]), sample["metadata"]["card"])
                by_key[key][0] += 1
                by_key[key][1] += int(damage_ok)
                by_key[key][2] += int(hp_ok)
                if status_experiment:
                    metadata = sample["metadata"]
                    predicted_block_loss = _effect_loss(predicted_tokens, "block")
                    expected_block_loss = int(metadata["expected_block_loss"])
                    expected_attack = int(metadata["calculated_attack_damage"])
                    predicted_attack = (
                        predicted_hp_loss + predicted_block_loss
                        if predicted_hp_loss is not None
                        and predicted_block_loss is not None
                        else None
                    )
                    checks = {
                        "attack": predicted_attack == expected_attack,
                        "block_loss": predicted_block_loss == expected_block_loss,
                        "hp_loss": damage_ok,
                        "next_block": _aligned_next_block(predicted_tokens)
                        == int(metadata["expected_next_block"]),
                        "next_hp": hp_ok,
                        "next_vulnerable": _aligned_next_vulnerable(predicted_tokens)
                        == int(metadata["expected_next_vulnerable"]),
                    }
                    for name, passed in checks.items():
                        status_counts[f"teacher_{name}"] += int(passed)
                    status_key = (
                        int(int(metadata["enemy_block"]) > 0),
                        int(int(metadata["enemy_vulnerable"]) > 0),
                        metadata["card"],
                    )
                    row = teacher_status_by_key[status_key]
                    row[0] += 1
                    for offset, name in enumerate(checks, 1):
                        row[offset] += int(checks[name])

    metrics = _metrics(total_loss, totals)
    metrics["cross_entropy_loss"] = metrics["loss"]
    metrics["consistency_loss"] = (
        total_consistency_loss / max(1, consistency_terms)
    )
    metrics["objective_loss"] = (
        metrics["cross_entropy_loss"]
        + consistency_loss_weight * metrics["consistency_loss"]
    )
    metrics.update({
        "samples": len(samples),
        "unknown_tokens": unknowns,
        "teacher_forced_damage_accuracy": damage_correct / len(samples),
        "teacher_forced_next_hp_accuracy": next_hp_correct / len(samples),
    })
    if status_experiment:
        for name in (
            "attack", "block_loss", "hp_loss", "next_block", "next_hp",
            "next_vulnerable",
        ):
            metrics[f"teacher_forced_{name}_accuracy"] = (
                status_counts[f"teacher_{name}"] / len(samples)
            )
    if calculation_counts["teacher_all_total"]:
        metrics["teacher_forced_all_calculations_accuracy"] = (
            calculation_counts["teacher_all_hits"]
            / calculation_counts["teacher_all_total"]
        )
        for key, total in sorted(calculation_counts.items()):
            if not key.startswith("teacher_") or not key.endswith("_total"):
                continue
            name = key[len("teacher_"):-len("_total")]
            if name == "all":
                continue
            metrics[f"teacher_forced_calculation_{name.lower()}_accuracy"] = (
                calculation_counts[f"teacher_{name}_hits"] / total
            )

    greedy_indices = list(range(len(samples)))
    random.Random(0).shuffle(greedy_indices)
    greedy_indices = greedy_indices[:min(args.greedy_samples, len(samples))]
    greedy_valid = greedy_correct = 0
    greedy_next_hp_valid = greedy_next_hp_correct = 0
    greedy_errors = []
    with torch.no_grad():
        for begin in range(0, len(greedy_indices), args.batch_size):
            indices = greedy_indices[begin:begin + args.batch_size]
            prefix_ids = _pad([encoded[index][0] for index in indices], tokenizer.pad_id, device)
            generated = model.generate(
                prefix_ids,
                output_begin_id=tokenizer.index[OUTPUT_BEGIN],
                output_end_id=tokenizer.index[OUTPUT_END],
                max_new_tokens=args.greedy_max_tokens,
            ).cpu()
            for row, index in enumerate(indices):
                predicted_tokens = tokenizer.decode(generated[row].tolist())
                predicted_damage = _greedy_damage(predicted_tokens)
                predicted_next_hp = _aligned_next_hp(predicted_tokens)
                expected_damage = int(samples[index]["metadata"]["expected_damage"])
                expected_next_hp = _expected_next_hp(samples[index])
                greedy_valid += int(predicted_damage is not None)
                greedy_correct += int(predicted_damage == expected_damage)
                greedy_next_hp_valid += int(predicted_next_hp is not None)
                greedy_next_hp_correct += int(predicted_next_hp == expected_next_hp)
                calculation_checks = _calculation_checks(
                    predicted_tokens, samples[index]
                )
                if calculation_checks:
                    calculation_counts["greedy_all_total"] += 1
                    calculation_counts["greedy_all_hits"] += int(
                        all(calculation_checks.values())
                    )
                    for calculation_type, passed in calculation_checks.items():
                        calculation_counts[f"greedy_{calculation_type}_total"] += 1
                        calculation_counts[f"greedy_{calculation_type}_hits"] += int(
                            passed
                        )
                status_error = False
                status_prediction = {}
                if status_experiment:
                    metadata = samples[index]["metadata"]
                    predicted_block_loss = _effect_loss(predicted_tokens, "block")
                    predicted_attack = (
                        predicted_damage + predicted_block_loss
                        if predicted_damage is not None
                        and predicted_block_loss is not None
                        else None
                    )
                    status_prediction = {
                        "predicted_attack_damage": predicted_attack,
                        "expected_attack_damage": int(
                            metadata["calculated_attack_damage"]
                        ),
                        "predicted_block_loss": predicted_block_loss,
                        "expected_block_loss": int(metadata["expected_block_loss"]),
                        "predicted_next_block": _aligned_next_block(predicted_tokens),
                        "expected_next_block": int(metadata["expected_next_block"]),
                        "predicted_next_vulnerable": _aligned_next_vulnerable(
                            predicted_tokens
                        ),
                        "expected_next_vulnerable": int(
                            metadata["expected_next_vulnerable"]
                        ),
                    }
                    status_checks = {
                        "attack": predicted_attack
                        == status_prediction["expected_attack_damage"],
                        "block_loss": predicted_block_loss
                        == status_prediction["expected_block_loss"],
                        "hp_loss": predicted_damage == expected_damage,
                        "next_block": status_prediction["predicted_next_block"]
                        == status_prediction["expected_next_block"],
                        "next_hp": predicted_next_hp == expected_next_hp,
                        "next_vulnerable": status_prediction[
                            "predicted_next_vulnerable"
                        ] == status_prediction["expected_next_vulnerable"],
                    }
                    for name, passed in status_checks.items():
                        status_counts[f"greedy_{name}"] += int(passed)
                    status_key = (
                        int(int(metadata["enemy_block"]) > 0),
                        int(int(metadata["enemy_vulnerable"]) > 0),
                        metadata["card"],
                    )
                    status_row = greedy_status_by_key[status_key]
                    status_row[0] += 1
                    for offset, name in enumerate(status_checks, 1):
                        status_row[offset] += int(status_checks[name])
                    before_monster = samples[index]["state"]["combat"][
                        "monsters"
                    ][0]
                    block_consistent = (
                        predicted_block_loss is not None
                        and status_prediction["predicted_next_block"] is not None
                        and status_prediction["predicted_next_block"]
                        == int(before_monster["block"]) - predicted_block_loss
                    )
                    hp_consistent = (
                        predicted_damage is not None
                        and predicted_next_hp is not None
                        and predicted_next_hp
                        == int(before_monster["hp"]) - predicted_damage
                    )
                    status_counts["greedy_effect_next_block_consistency"] += int(
                        block_consistent
                    )
                    status_counts["greedy_effect_next_hp_consistency"] += int(
                        hp_consistent
                    )
                    status_error = not (
                        all(status_checks.values())
                        and block_consistent
                        and hp_consistent
                    )
                if (
                    len(greedy_errors) < args.show_errors
                    and (
                        predicted_damage != expected_damage
                        or predicted_next_hp != expected_next_hp
                        or status_error
                    )
                ):
                    greedy_errors.append({
                        "sample": index,
                        "strength": int(samples[index]["metadata"]["strength"]),
                        "card": samples[index]["metadata"]["card"],
                        "expected_damage": expected_damage,
                        "predicted_damage": predicted_damage,
                        "expected_next_hp": expected_next_hp,
                        "predicted_next_hp": predicted_next_hp,
                        **status_prediction,
                    })
    metrics.update({
        "greedy_samples": len(greedy_indices),
        "greedy_valid_damage_rate": greedy_valid / max(1, len(greedy_indices)),
        "greedy_damage_accuracy": greedy_correct / max(1, len(greedy_indices)),
        "greedy_valid_next_hp_rate": greedy_next_hp_valid / max(1, len(greedy_indices)),
        "greedy_next_hp_accuracy": greedy_next_hp_correct / max(1, len(greedy_indices)),
    })
    if status_experiment:
        for name in (
            "attack", "block_loss", "hp_loss", "next_block", "next_hp",
            "next_vulnerable",
        ):
            metrics[f"greedy_{name}_accuracy"] = (
                status_counts[f"greedy_{name}"] / max(1, len(greedy_indices))
            )
        for field in ("block", "hp"):
            metrics[f"greedy_effect_next_{field}_consistency"] = (
                status_counts[f"greedy_effect_next_{field}_consistency"]
                / max(1, len(greedy_indices))
            )
    if calculation_counts["greedy_all_total"]:
        metrics["greedy_all_calculations_accuracy"] = (
            calculation_counts["greedy_all_hits"]
            / calculation_counts["greedy_all_total"]
        )
        for key, total in sorted(calculation_counts.items()):
            if not key.startswith("greedy_") or not key.endswith("_total"):
                continue
            name = key[len("greedy_"):-len("_total")]
            if name == "all":
                continue
            metrics[f"greedy_calculation_{name.lower()}_accuracy"] = (
                calculation_counts[f"greedy_{name}_hits"] / total
            )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not args.summary_only:
        print("strength card samples damage_accuracy next_hp_accuracy")
        for (strength, card), (count, damage_hits, hp_hits) in sorted(by_key.items()):
            print(
                f"{strength:2d} {card:8s} {count:4d} "
                f"{damage_hits / count:.4f} {hp_hits / count:.4f}"
            )
    if status_experiment:
        for label, grouped in (
            ("teacher_forced_by_status", teacher_status_by_key),
            ("greedy_by_status", greedy_status_by_key),
        ):
            print(label)
            print(
                "has_block has_vulnerable card samples attack block_loss hp_loss "
                "next_block next_hp next_vulnerable"
            )
            for (has_block, has_vulnerable, card), row in sorted(grouped.items()):
                count = row[0]
                accuracies = " ".join(f"{hits / count:.4f}" for hits in row[1:])
                print(
                    f"{has_block:9d} {has_vulnerable:14d} {card:8s} "
                    f"{count:4d} {accuracies}"
                )
    if greedy_errors:
        print("greedy_errors")
        for error in greedy_errors:
            print(json.dumps(error, sort_keys=True))


if __name__ == "__main__":
    main()
