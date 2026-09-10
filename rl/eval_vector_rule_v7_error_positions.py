"""Profile schema-v7 teacher-forced errors by token and delta-vector position."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch

from rl.make_vector_rule_v7_data import build_v7_record
from rl.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_VOCAB,
    StateVectorConfig,
    VectorRuleModelConfig,
    VectorRuleWorldModel,
)
from rl.train_rule_vectors_v7 import _model_inputs, _move_batch
from rl.vector_rule_v7_data import (
    V7_EXECUTOR_CONTRACT_VERSION,
    V7TokenCodec,
    VectorRuleV7Example,
    collate_vector_rule_v7_batch,
    examples_from_v7_record,
)


NUMBER_TOKENS = frozenset({
    "NUM_BEGIN", "NUM_LSD_BEGIN", "NUM_END", "SIGN_NEG", "DECIMAL",
    *(f"DIGIT_{digit}" for digit in range(10)),
})


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _world_slot_names(config: StateVectorConfig) -> list[str]:
    names: list[str] = []
    names.extend(f"player.{name}" for name in config.player_scalar_names)
    names.extend(f"player.buff.{name}" for name in config.player_buff_names)
    names.extend(f"player.counter.{name}" for name in config.counter_names)
    names.extend(f"player.relic.{name}" for name in config.relic_summary_names)
    names.extend(
        f"player.relic_counter[{index}]"
        for index in range(len(config.relic_presence_names))
    )
    for enemy in range(config.max_enemies):
        names.extend(
            f"enemy[{enemy}].{name}" for name in config.enemy_scalar_names
        )
        names.extend(
            f"enemy[{enemy}].buff.{name}"
            for name in config.monster_buff_names
        )
    names.extend(f"piles.{name}" for name in config.pile_names)
    if len(names) != config.world_dim:
        raise RuntimeError("world slot name construction does not match world_dim")
    return names


def _parameter_location(tokens: Sequence[str], index: int) -> tuple[str, str] | None:
    """Return (field, value-position) for one expected parameter token."""

    token = tokens[index]
    def contextual_field(field_index: int) -> str:
        field = tokens[field_index] if field_index >= 0 else "<START>"
        for cursor in range(field_index - 1, -1, -1):
            previous = tokens[cursor]
            if previous == "CALCULATION_BEGIN":
                break
            if (
                previous.startswith("CALCULATION_")
                and previous not in {"CALCULATION_BEGIN", "CALCULATION_END"}
            ):
                return f"{previous}/{field}"
        return field

    if token.startswith("VALUE_") or token.startswith("BOOL_"):
        return contextual_field(index - 1), "atomic_value"
    if token not in NUMBER_TOKENS:
        return None
    begin = index
    while begin > 0 and tokens[begin - 1] in NUMBER_TOKENS:
        begin -= 1
    field = contextual_field(begin - 1)
    offset = index - begin
    if token in {"NUM_BEGIN", "NUM_LSD_BEGIN"}:
        role = "number_begin"
    elif token == "NUM_END":
        role = "number_end"
    elif token == "SIGN_NEG":
        role = "sign"
    elif token == "DECIMAL":
        role = "decimal_point"
    elif token.startswith("DIGIT_"):
        role = f"digit_offset_{offset - 1}"
    else:
        role = f"offset_{offset}"
    return field, role


def _new_stat() -> dict[str, int]:
    return {"count": 0, "errors": 0}


def _update_stat(table: dict[str, dict[str, int]], key: str, correct: bool) -> None:
    row = table.setdefault(key, _new_stat())
    row["count"] += 1
    row["errors"] += int(not correct)


def _rank_table(
    table: Mapping[str, Mapping[str, int]], *, min_count: int = 1,
) -> list[dict[str, Any]]:
    rows = []
    for key, value in table.items():
        count = int(value["count"])
        errors = int(value["errors"])
        if count < min_count:
            continue
        rows.append({
            "key": key,
            "count": count,
            "errors": errors,
            "error_rate": errors / count,
        })
    return sorted(rows, key=lambda row: (-row["error_rate"], -row["errors"], row["key"]))


def _batches(
    *, records: int, seed: int, config: StateVectorConfig,
    codec: V7TokenCodec, batch_size: int, max_cards: int, max_relics: int,
    max_tokens: int,
) -> Iterable[list[VectorRuleV7Example]]:
    rng = random.Random(seed)
    pending: list[VectorRuleV7Example] = []
    for index in range(records):
        record = build_v7_record(index, config, rng)
        pending.extend(examples_from_v7_record(
            record, codec, config,
            card_vocab_size=len(CARD_RULE_VOCAB),
            relic_vocab_size=len(RELIC_RULE_VOCAB),
            max_cards=max_cards, max_relics=max_relics, max_tokens=max_tokens,
        ))
        while len(pending) >= batch_size:
            yield pending[:batch_size]
            del pending[:batch_size]
    if pending:
        yield pending


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cards", type=int, default=10)
    parser.add_argument("--max-relics", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top", type=int, default=30)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.records < 1 or args.batch_size < 1:
        raise ValueError("records and batch-size must be positive")
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if int(checkpoint.get("executor_contract_version", -1)) != V7_EXECUTOR_CONTRACT_VERSION:
        raise ValueError("checkpoint uses an incompatible executor contract")
    config = VectorRuleModelConfig.from_dict(checkpoint["config"])
    codec = V7TokenCodec.load(args.tokenizer)
    if config.decoder.token_vocab_size != len(codec.tokens):
        raise ValueError("checkpoint decoder vocabulary does not match tokenizer")
    model = VectorRuleWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scope_stats: dict[str, dict[str, int]] = {}
    token_stats: dict[str, dict[str, int]] = {}
    parameter_stats: dict[str, dict[str, int]] = {}
    parameter_position_stats: dict[str, dict[str, int]] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    total_tokens = correct_tokens = 0
    total_examples = 0

    world_names = _world_slot_names(config.state)
    width = len(world_names)
    active_count = torch.zeros(width, dtype=torch.long)
    active_abs_error = torch.zeros(width, dtype=torch.float64)
    active_sq_error = torch.zeros(width, dtype=torch.float64)
    active_within = torch.zeros(width, dtype=torch.long)
    active_sign = torch.zeros(width, dtype=torch.long)
    zero_count = torch.zeros(width, dtype=torch.long)
    zero_abs_prediction = torch.zeros(width, dtype=torch.float64)
    delta_events = 0

    collate = lambda rows: collate_vector_rule_v7_batch(rows, pad_id=codec.pad_id)
    for rows in _batches(
        records=args.records, seed=args.seed, config=config.state, codec=codec,
        batch_size=args.batch_size, max_cards=args.max_cards,
        max_relics=args.max_relics, max_tokens=args.max_tokens,
    ):
        batch = _move_batch(collate(rows), device)
        with torch.inference_mode():
            output = model(**_model_inputs(model, batch))
        predictions = output.logits.argmax(dim=-1).cpu()
        labels = batch["labels"].cpu()
        valid = labels.ne(codec.pad_id)
        correct = predictions.eq(labels) & valid
        total_tokens += int(valid.sum())
        correct_tokens += int(correct.sum())
        total_examples += len(rows)

        for row_index, example in enumerate(rows):
            expected_tokens = codec.decode(example.target_ids)
            length = len(expected_tokens) - 1
            row_predictions = predictions[row_index, :length].tolist()
            model_mask = example.target_model_mask or tuple(
                True for _ in example.target_ids
            )
            if len(model_mask) != len(expected_tokens):
                raise RuntimeError("target model mask does not match token length")
            for position in range(length):
                # Runtime-owned responses are masked out of training labels and
                # must not be counted as model prediction errors.
                if not model_mask[position + 1]:
                    continue
                expected = expected_tokens[position + 1]
                predicted = codec.tokens[row_predictions[position]]
                is_correct = expected == predicted
                _update_stat(scope_stats, example.scope_kind, is_correct)
                _update_stat(token_stats, expected, is_correct)
                location = _parameter_location(expected_tokens, position + 1)
                if location is not None:
                    field, role = location
                    _update_stat(parameter_stats, field, is_correct)
                    _update_stat(parameter_position_stats, f"{field}/{role}", is_correct)
                if not is_correct:
                    confusion[(expected, predicted)] += 1

        mask = batch["delta_mask"]
        if bool(mask.any()):
            predicted_delta = output.delta[mask].detach().cpu().double()
            target_delta = batch["delta_targets"][mask].detach().cpu().double()
            delta_events += target_delta.shape[0]
            active = target_delta.abs().ge(1.0e-8)
            zero = ~active
            error = predicted_delta - target_delta
            active_count += active.sum(dim=0)
            active_abs_error += (error.abs() * active).sum(dim=0)
            active_sq_error += (error.square() * active).sum(dim=0)
            active_within += ((error.abs() <= 0.005) & active).sum(dim=0)
            active_sign += ((predicted_delta.sign() == target_delta.sign()) & active).sum(dim=0)
            zero_count += zero.sum(dim=0)
            zero_abs_prediction += (predicted_delta.abs() * zero).sum(dim=0)

    delta_rows = []
    for index, name in enumerate(world_names):
        count = int(active_count[index])
        zeros = int(zero_count[index])
        if count == 0:
            continue
        delta_rows.append({
            "index": index,
            "name": name,
            "active_count": count,
            "active_mae": float(active_abs_error[index] / count),
            "active_rmse": float((active_sq_error[index] / count).sqrt()),
            "within_0.005": float(active_within[index] / count),
            "sign_accuracy": float(active_sign[index] / count),
            "zero_target_count": zeros,
            "zero_target_abs_prediction_mean": (
                float(zero_abs_prediction[index] / zeros) if zeros else None
            ),
        })
    delta_by_mae = sorted(
        delta_rows, key=lambda row: (-row["active_mae"], -row["active_count"])
    )
    delta_by_weighted_error = sorted(
        delta_rows,
        key=lambda row: (-(row["active_mae"] * row["active_count"]), -row["active_count"]),
    )
    confusion_rows = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in confusion.most_common(args.top)
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "device": str(device),
        "records": args.records,
        "scope_examples": total_examples,
        "tokens": total_tokens,
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "delta_events": delta_events,
        "scope_error_rates": _rank_table(scope_stats),
        "parameter_error_rates": _rank_table(parameter_stats),
        "parameter_position_error_rates": _rank_table(parameter_position_stats),
        "token_error_rates_min_20": _rank_table(token_stats, min_count=20),
        "top_confusions": confusion_rows,
        "delta_slots_by_active_mae": delta_by_mae,
        "delta_slots_by_total_active_error": delta_by_weighted_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print(report: Mapping[str, Any], top: int) -> None:
    print(
        f"checkpoint_step={report['checkpoint_step']} records={report['records']} "
        f"scopes={report['scope_examples']} tokens={report['tokens']} "
        f"token_accuracy={report['token_accuracy']:.6f} "
        f"delta_events={report['delta_events']}"
    )
    for title, key in (
        ("scope", "scope_error_rates"),
        ("parameter", "parameter_error_rates"),
        ("parameter_position", "parameter_position_error_rates"),
    ):
        print(f"\n[{title}]")
        for row in report[key][:top]:
            print(
                f"{row['key']} count={row['count']} errors={row['errors']} "
                f"error_rate={row['error_rate']:.6f}"
            )
    print("\n[delta_active_mae]")
    for row in report["delta_slots_by_active_mae"][:top]:
        print(
            f"{row['index']} {row['name']} count={row['active_count']} "
            f"mae={row['active_mae']:.8f} rmse={row['active_rmse']:.8f} "
            f"within_0.005={row['within_0.005']:.4f}"
        )
    print("\n[top_confusions]")
    for row in report["top_confusions"][:top]:
        print(f"{row['expected']} -> {row['predicted']} count={row['count']}")


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    _print(report, args.top)
    print(f"\nreport={args.output}")


if __name__ == "__main__":
    main()
