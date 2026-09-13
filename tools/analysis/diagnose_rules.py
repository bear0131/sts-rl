"""Diagnose where a combat-rule checkpoint is making prediction errors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

import torch

from rl.rules.rule_data import (
    CALCULATIONS_BEGIN,
    CALCULATIONS_END,
    CRITICAL_CALCULATION,
    CRITICAL_CAUSAL,
    CRITICAL_COUNTER,
    CRITICAL_EFFECT,
    CRITICAL_NEXT_STATE,
    CRITICAL_NONE,
    CRITICAL_STRUCTURE,
    CRITICAL_STEP,
    EFFECTS_BEGIN,
    EFFECTS_END,
    NEXT_STATE_BEGIN,
    NEXT_STATE_END,
    OUTPUT_BEGIN,
    OUTPUT_END,
    RULE_STEPS_BEGIN,
    RULE_STEPS_END,
    RuleTokenizer,
    read_jsonl,
    sample_tokens,
    target_supervision_regions,
    transition_consistency_specs,
)
from rl.rules.rule_model import RuleWorldModel
from rl.rules.training.train_rules import (
    _loss_and_counts,
    _metrics,
    _split_samples,
    _transition_consistency_loss,
)


REGION_NAMES = {
    CRITICAL_NONE: "other",
    CRITICAL_EFFECT: "effect_number",
    CRITICAL_NEXT_STATE: "next_number",
    CRITICAL_CALCULATION: "calculation_number",
    CRITICAL_STEP: "step_number",
    CRITICAL_CAUSAL: "causal",
    CRITICAL_COUNTER: "counter",
    CRITICAL_STRUCTURE: "structure",
}

SECTION_MARKERS = (
    ("rule_steps", RULE_STEPS_BEGIN, RULE_STEPS_END),
    ("effects", EFFECTS_BEGIN, EFFECTS_END),
    ("calculations", CALCULATIONS_BEGIN, CALCULATIONS_END),
    ("next_state", NEXT_STATE_BEGIN, NEXT_STATE_END),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--teacher-samples", type=int, default=256)
    parser.add_argument("--greedy-samples", type=int, default=12)
    parser.add_argument("--greedy-max-tokens", type=int, default=3072)
    parser.add_argument(
        "--greedy-action",
        action="append",
        help="Only autoregressively diagnose these action types; repeatable.",
    )
    parser.add_argument(
        "--greedy-card",
        action="append",
        help="Only autoregressively diagnose these card ids; repeatable.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--show-errors", type=int, default=8)
    parser.add_argument("--out", type=Path)
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _pad(rows: list[list[int]], pad_id: int, device: torch.device) -> torch.Tensor:
    tensors = [torch.tensor(row, dtype=torch.long) for row in rows]
    return torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=pad_id
    ).to(device)


def _counter_rate(counter: Counter, correct: str, total: str) -> dict[str, Any]:
    tokens = int(counter.get(total, 0))
    hits = int(counter.get(correct, 0))
    return {
        "correct": hits,
        "total": tokens,
        "accuracy": hits / tokens if tokens else None,
    }


def _top_rates(
    counters: dict[str, Counter],
    *,
    correct: str,
    total: str,
    top_k: int,
    min_total: int = 1,
    worst: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for name, counter in counters.items():
        count = int(counter.get(total, 0))
        if count < min_total:
            continue
        hits = int(counter.get(correct, 0))
        rows.append({
            "name": name,
            "correct": hits,
            "total": count,
            "accuracy": hits / count if count else None,
        })
    rows.sort(key=lambda row: (row["accuracy"], -row["total"]) if worst else (-row["total"], row["name"]))
    return rows[:top_k]


def _read_run_config(run: Path) -> dict[str, Any]:
    path = run / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _section_bounds(tokens: list[str], begin: str, end: str) -> tuple[int, int] | None:
    try:
        left = tokens.index(begin)
        right = tokens.index(end, left) + 1
    except ValueError:
        return None
    return left, right


def _section_ranges(
    tokens: list[str], begin: str, end: str,
) -> list[tuple[int, int]]:
    ranges = []
    position = 0
    while position < len(tokens):
        try:
            left = tokens.index(begin, position)
            right = tokens.index(end, left) + 1
        except ValueError:
            break
        ranges.append((left, right))
        position = right
    return ranges


def _section_positions(tokens: list[str]) -> dict[int, str]:
    positions: dict[int, str] = {}
    for name, begin, end in SECTION_MARKERS:
        for left, right in _section_ranges(tokens, begin, end):
            for position in range(left, right):
                positions[position] = name
    return positions


def _first_mismatch(expected: list[str], predicted: list[str]) -> int | None:
    for index, (want, got) in enumerate(zip(expected, predicted)):
        if want != got:
            return index
    if len(expected) != len(predicted):
        return min(len(expected), len(predicted))
    return None


def _exact_section(expected: list[str], predicted: list[str], begin: str, end: str) -> bool:
    expected_ranges = _section_ranges(expected, begin, end)
    predicted_ranges = _section_ranges(predicted, begin, end)
    return (
        [expected[left:right] for left, right in expected_ranges]
        == [predicted[left:right] for left, right in predicted_ranges]
    )


def _trim_output(tokens: list[str]) -> list[str]:
    try:
        return tokens[:tokens.index(OUTPUT_END) + 1]
    except ValueError:
        return tokens


def _sample_tags(sample: dict[str, Any]) -> list[str]:
    action = sample["action"]
    state = sample["state"]
    combat = state["combat"]
    player = combat["player"]
    tags = [f"action:{action.get('type', 'UNKNOWN')}"]
    card = action.get("card")
    if card is not None:
        tags.append(f"card:{card.get('id', 'UNKNOWN')}")
        tags.append(f"card_up:{card.get('id', 'UNKNOWN')}+{int(card.get('up', 0))}")
    player_powers = player.get("powers", {})
    tags.append(f"player_powers:{int(bool(player_powers))}")
    for name in sorted(player_powers):
        tags.append(f"player_status:{name}")
    monsters = combat["monsters"]
    tags.append(f"monster_block:{int(any(int(m.get('block', 0)) > 0 for m in monsters))}")
    monster_powers = {
        name
        for monster in monsters
        for name in monster.get("powers", {})
    }
    tags.append(f"monster_powers:{int(bool(monster_powers))}")
    for name in sorted(monster_powers):
        tags.append(f"monster_status:{name}")
    relics = state["global"].get("relics", [])
    if not relics:
        tags.append("relic_count:0")
    elif len(relics) == 1:
        tags.append("relic_count:1")
    elif len(relics) <= 3:
        tags.append("relic_count:2-3")
    else:
        tags.append("relic_count:4+")
    for relic in relics:
        tags.append(f"relic:{relic}")
    return tags


def _parse_number_at(
    tokens: list[str], position: int, stop: int | None = None
) -> tuple[int, int, tuple[int, ...], bool] | None:
    stop = len(tokens) if stop is None else min(stop, len(tokens))
    if position >= stop or tokens[position] not in ("NUM_BEGIN", "NUM_LSD_BEGIN"):
        return None
    least_significant_first = tokens[position] == "NUM_LSD_BEGIN"
    position += 1
    sign = 1
    if position < stop and tokens[position] == "SIGN_NEG":
        sign = -1
        position += 1
    digit_positions = []
    digits = []
    while position < stop and tokens[position].startswith("DIGIT_"):
        digit_positions.append(position)
        digits.append(tokens[position][len("DIGIT_"):])
        position += 1
    if not digits or position >= stop or tokens[position] != "NUM_END":
        return None
    if least_significant_first:
        digits.reverse()
    return sign * int("".join(digits)), position + 1, tuple(digit_positions), least_significant_first


def _number_after(
    tokens: list[str],
    marker: str,
    *,
    start: int,
    stop: int,
) -> int | None:
    try:
        marker_position = tokens.index(marker, start, stop)
    except ValueError:
        return None
    parsed = _parse_number_at(tokens, marker_position + 1, stop)
    return None if parsed is None else parsed[0]


def _value_after(
    tokens: list[str],
    marker: str,
    *,
    start: int,
    stop: int,
) -> str | None:
    try:
        marker_position = tokens.index(marker, start, stop)
    except ValueError:
        return None
    value_position = marker_position + 1
    if value_position >= stop:
        return None
    token = tokens[value_position]
    if token.startswith("VALUE_"):
        return token[len("VALUE_"):]
    return token


def _iter_blocks(
    tokens: list[str],
    section_begin: str,
    section_end: str,
    block_begin: str,
    block_end: str,
) -> list[tuple[int, int]]:
    out = []
    for left, right in _section_ranges(tokens, section_begin, section_end):
        position = left
        while position < right:
            try:
                begin = tokens.index(block_begin, position, right)
                end = tokens.index(block_end, begin, right) + 1
            except ValueError:
                break
            out.append((begin, end))
            position = end
    return out


def _iter_expected_numbers(tokens: list[str], regions: list[int]) -> list[dict[str, Any]]:
    sections = _section_positions(tokens)
    out = []
    position = 0
    while position < len(tokens):
        parsed = _parse_number_at(tokens, position)
        if parsed is None:
            position += 1
            continue
        value, end, digit_positions, _lsd = parsed
        region_ids = [regions[pos] for pos in digit_positions if pos < len(regions)]
        region = next((item for item in region_ids if item != CRITICAL_NONE), CRITICAL_NONE)
        out.append({
            "start": position,
            "end": end,
            "value": value,
            "region": REGION_NAMES[region],
            "section": sections.get(position, "other"),
            "marker": tokens[position - 1] if position > 0 else "",
        })
        position = end
    return out


def _record_number_errors(
    expected: list[str],
    predicted: list[str],
    regions: list[int],
    number_counts: dict[str, Counter],
) -> None:
    for number in _iter_expected_numbers(expected, regions):
        predicted_number = _parse_number_at(predicted, number["start"], number["end"])
        keys = [
            "all",
            f"section:{number['section']}",
            f"region:{number['region']}",
            f"marker:{number['marker']}",
        ]
        for key in keys:
            number_counts[key]["total"] += 1
        if predicted_number is None:
            for key in keys:
                number_counts[key]["unparseable"] += 1
            continue
        predicted_value = predicted_number[0]
        exact = predicted_value == number["value"]
        for key in keys:
            number_counts[key]["correct"] += int(exact)
            if not exact:
                number_counts[key]["absolute_error_sum"] += abs(predicted_value - number["value"])
                number_counts[key]["wrong"] += 1


def _event_type(tokens: list[str], begin: int, end: int, prefix: str) -> str:
    for token in tokens[begin:end]:
        if token in (f"{prefix}BEGIN", f"{prefix}END"):
            continue
        if token.startswith(prefix):
            return token[len(prefix):]
    return "UNKNOWN"


def _record_block_errors(
    expected: list[str],
    predicted: list[str],
    blocks: list[tuple[int, int]],
    group: dict[str, Counter],
    prefix: str,
) -> None:
    for begin, end in blocks:
        name = _event_type(expected, begin, end, prefix)
        row = group[name]
        row["blocks"] += 1
        same = expected[begin:end] == predicted[begin:min(end, len(predicted))]
        row["exact"] += int(same)
        for pos in range(begin, end):
            row["tokens"] += 1
            row["correct"] += int(pos < len(predicted) and predicted[pos] == expected[pos])


def _step_event_name(tokens: list[str], begin: int, end: int) -> str:
    # v5 effect blocks use STEP_CAUSE instead of the legacy STEP_EVENT marker.
    # Keep the legacy name when present, then fall back to the causal value and
    # finally the structural event type so reports remain actionable.
    return (
        _value_after(tokens, "STEP_EVENT", start=begin, stop=end)
        or _value_after(tokens, "STEP_CAUSE", start=begin, stop=end)
        or _event_type(tokens, begin, end, "EFFECT_")
        or "UNKNOWN"
    )


def _record_step_errors(
    expected: list[str],
    predicted: list[str],
    blocks: list[tuple[int, int]],
    group: dict[str, Counter],
) -> None:
    for begin, end in blocks:
        name = _step_event_name(expected, begin, end)
        row = group[name]
        row["blocks"] += 1
        same = expected[begin:end] == predicted[begin:min(end, len(predicted))]
        row["exact"] += int(same)
        for pos in range(begin, end):
            row["tokens"] += 1
            row["correct"] += int(pos < len(predicted) and predicted[pos] == expected[pos])


def _causal_step_blocks(tokens: list[str]) -> list[tuple[int, int]]:
    legacy = _iter_blocks(
        tokens, RULE_STEPS_BEGIN, RULE_STEPS_END, "STEP_BEGIN", "STEP_END"
    )
    if legacy:
        return legacy
    return _iter_blocks(
        tokens, EFFECTS_BEGIN, EFFECTS_END, "EFFECT_BEGIN", "EFFECT_END"
    )


def _expected_effect_aggregate(sample: dict[str, Any]) -> dict[str, int]:
    totals: Counter = Counter()
    for effect in sample.get("effects", ()):
        event_type = effect.get("type")
        field = effect.get("field")
        delta = effect.get("delta")
        if isinstance(delta, (int, float)):
            key = f"{event_type}:{field or effect.get('counter') or effect.get('status')}"
            totals[key] += int(delta)
        if event_type == "MONSTER_VALUE_CHANGE" and field in ("hp", "block") and isinstance(delta, (int, float)):
            slot = int(effect.get("slot", 0))
            totals[f"monster_{slot}_{field}_delta"] += int(delta)
        if event_type == "PLAYER_VALUE_CHANGE" and field in ("hp", "block", "energy") and isinstance(delta, (int, float)):
            totals[f"player_{field}_delta"] += int(delta)
        if event_type == "STATUS_CHANGE" and isinstance(delta, (int, float)):
            totals[f"status:{effect.get('owner')}:{effect.get('status')}"] += int(delta)
        if event_type == "COUNTER_CHANGE" and isinstance(delta, (int, float)):
            totals[f"counter:{effect.get('counter')}"] += int(delta)
    return dict(totals)


def _predicted_effect_aggregate(tokens: list[str]) -> dict[str, int] | None:
    blocks = _iter_blocks(tokens, EFFECTS_BEGIN, EFFECTS_END, "EFFECT_BEGIN", "EFFECT_END")
    if not blocks:
        return None
    totals: Counter = Counter()
    for begin, end in blocks:
        event_type = _event_type(tokens, begin, end, "EFFECT_")
        field = _value_after(tokens, "FIELD_field", start=begin, stop=end)
        delta = _number_after(tokens, "FIELD_delta", start=begin, stop=end)
        if delta is not None:
            key_value = field or _value_after(tokens, "FIELD_counter", start=begin, stop=end)
            if key_value is None:
                key_value = _value_after(tokens, "FIELD_status", start=begin, stop=end)
            totals[f"{event_type}:{key_value}"] += int(delta)
        if event_type == "MONSTER_VALUE_CHANGE" and field in ("hp", "block") and delta is not None:
            slot = _number_after(tokens, "FIELD_slot", start=begin, stop=end)
            slot = 0 if slot is None else slot
            totals[f"monster_{slot}_{field}_delta"] += int(delta)
        if event_type == "PLAYER_VALUE_CHANGE" and field in ("hp", "block", "energy") and delta is not None:
            totals[f"player_{field}_delta"] += int(delta)
        if event_type == "STATUS_CHANGE" and delta is not None:
            owner = _value_after(tokens, "FIELD_owner", start=begin, stop=end)
            status = _value_after(tokens, "FIELD_status", start=begin, stop=end)
            totals[f"status:{owner}:{status}"] += int(delta)
        if event_type == "COUNTER_CHANGE" and delta is not None:
            counter = _value_after(tokens, "FIELD_counter", start=begin, stop=end)
            totals[f"counter:{counter}"] += int(delta)
    return dict(totals)


def _record_aggregate_checks(
    sample: dict[str, Any],
    predicted: list[str],
    aggregate_counts: dict[str, Counter],
) -> None:
    expected = _expected_effect_aggregate(sample)
    got = _predicted_effect_aggregate(predicted)
    if got is None:
        aggregate_counts["effect_parse"]["total"] += 1
        return
    aggregate_counts["effect_parse"]["total"] += 1
    aggregate_counts["effect_parse"]["correct"] += 1
    for key, expected_value in expected.items():
        aggregate_counts[key]["total"] += 1
        aggregate_counts[key]["correct"] += int(got.get(key) == expected_value)


def _build_encoded(
    samples: list[dict[str, Any]],
    tokenizer: RuleTokenizer,
    output_version: int,
) -> list[dict[str, Any]]:
    encoded = []
    unknown = tokenizer.index["<UNK>"]
    for sample in samples:
        prefix, target = sample_tokens(
            sample, output_version=output_version,
        )
        prefix_ids = tokenizer.encode(prefix)
        target_ids = tokenizer.encode(target)
        regions = target_supervision_regions(target)
        specs = transition_consistency_specs(sample, target)
        encoded.append({
            "prefix_ids": prefix_ids,
            "target_ids": target_ids,
            "target": target,
            "regions": regions,
            "specs": specs,
            "unknowns": sum(value == unknown for value in prefix_ids)
            + sum(value == unknown for value in target_ids),
        })
    return encoded


def _teacher_forced_diagnostics(
    model: RuleWorldModel,
    tokenizer: RuleTokenizer,
    samples: list[dict[str, Any]],
    encoded: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    totals = Counter()
    section_counts: dict[str, Counter] = defaultdict(Counter)
    region_counts: dict[str, Counter] = defaultdict(Counter)
    group_counts: dict[str, Counter] = defaultdict(Counter)
    event_counts: dict[str, Counter] = defaultdict(Counter)
    calculation_counts: dict[str, Counter] = defaultdict(Counter)
    step_event_counts: dict[str, Counter] = defaultdict(Counter)
    number_counts: dict[str, Counter] = defaultdict(Counter)
    aggregate_counts: dict[str, Counter] = defaultdict(Counter)
    token_confusions: Counter = Counter()
    critical_confusions: Counter = Counter()
    section_first_mismatch: Counter = Counter()
    errors = []
    total_loss = 0.0
    total_consistency_loss = 0.0
    consistency_terms = 0
    effect_number_weight = float(
        run_config.get("effect_number_weight")
        or run_config.get("critical_number_weight", 1.0)
    )
    next_number_weight = float(
        run_config.get("next_number_weight")
        or run_config.get("critical_number_weight", 1.0)
    )
    calculation_number_weight = float(
        run_config.get("calculation_number_weight")
        or effect_number_weight
    )
    step_number_weight = float(run_config.get("step_number_weight", 1.0))
    causal_token_weight = float(run_config.get("causal_token_weight", 1.0))
    counter_token_weight = float(run_config.get("counter_token_weight", 1.0))
    structure_token_weight = float(run_config.get("structure_token_weight", 1.0))
    consistency_scale = float(run_config.get("consistency_scale", 10.0))
    digit_ids = torch.tensor(
        [tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)],
        device=device,
    )

    model.eval()
    with torch.no_grad():
        for begin in range(0, len(encoded), args.batch_size):
            chunk = encoded[begin:begin + args.batch_size]
            prefix_ids = _pad([item["prefix_ids"] for item in chunk], tokenizer.pad_id, device)
            target_ids = _pad([item["target_ids"] for item in chunk], tokenizer.pad_id, device)
            critical = _pad([item["regions"] for item in chunk], 0, device)
            logits, _latent = model(prefix_ids, target_ids[:, :-1])
            loss, counts = _loss_and_counts(
                logits,
                target_ids[:, 1:],
                tokenizer.pad_id,
                tokenizer.index[NEXT_STATE_BEGIN],
                critical[:, 1:],
                effect_number_weight,
                next_number_weight,
                calculation_number_weight,
                step_number_weight,
                causal_token_weight,
                counter_token_weight,
                structure_token_weight=structure_token_weight,
            )
            consistency_loss, term_count = _transition_consistency_loss(
                logits, [item["specs"] for item in chunk], digit_ids, consistency_scale
            )
            total_loss += float(loss) * counts["loss_weight"]
            total_consistency_loss += float(consistency_loss) * term_count
            consistency_terms += term_count
            totals.update(counts)
            predicted_ids = logits.argmax(dim=-1).cpu()
            for row, item in enumerate(chunk):
                sample = samples[begin + row]
                target = item["target"]
                predicted = [target[0]] + tokenizer.decode(
                    predicted_ids[row, :len(item["target_ids"]) - 1].tolist()
                )
                regions = item["regions"]
                sections = _section_positions(target)
                complete = target == predicted
                for pos in range(1, len(target)):
                    expected_token = target[pos]
                    predicted_token = predicted[pos] if pos < len(predicted) else "<MISSING>"
                    correct = expected_token == predicted_token
                    section_name = sections.get(pos, "other")
                    section_counts[section_name]["tokens"] += 1
                    section_counts[section_name]["correct"] += int(correct)
                    region_name = REGION_NAMES[regions[pos]]
                    region_counts[region_name]["tokens"] += 1
                    region_counts[region_name]["correct"] += int(correct)
                    if not correct:
                        token_confusions[(expected_token, predicted_token)] += 1
                        if regions[pos] != CRITICAL_NONE:
                            critical_confusions[(expected_token, predicted_token)] += 1
                for tag in _sample_tags(sample):
                    row_counter = group_counts[tag]
                    row_counter["samples"] += 1
                    row_counter["complete"] += int(complete)
                    row_counter["tokens"] += len(target) - 1
                    row_counter["correct"] += sum(
                        1 for pos in range(1, len(target))
                        if pos < len(predicted) and target[pos] == predicted[pos]
                    )
                    for name, left, right in SECTION_MARKERS:
                        if left not in target:
                            continue
                        key = f"{name}_exact"
                        row_counter[key] += int(_exact_section(target, predicted, left, right))
                mismatch = _first_mismatch(target, predicted)
                if mismatch is not None:
                    section_first_mismatch[sections.get(mismatch, "other")] += 1
                    if len(errors) < args.show_errors:
                        card = sample["action"].get("card", {})
                        errors.append({
                            "sample": begin + row,
                            "action_type": sample["action"].get("type"),
                            "card": card.get("id"),
                            "first_mismatch": mismatch,
                            "section": sections.get(mismatch, "other"),
                            "expected_context": target[max(0, mismatch - 8):mismatch + 16],
                            "predicted_context": predicted[max(0, mismatch - 8):mismatch + 16],
                        })
                _record_number_errors(target, predicted, regions, number_counts)
                _record_block_errors(
                    target,
                    predicted,
                    _iter_blocks(target, EFFECTS_BEGIN, EFFECTS_END, "EFFECT_BEGIN", "EFFECT_END"),
                    event_counts,
                    "EFFECT_",
                )
                _record_block_errors(
                    target,
                    predicted,
                    _iter_blocks(target, CALCULATIONS_BEGIN, CALCULATIONS_END, "CALCULATION_BEGIN", "CALCULATION_END"),
                    calculation_counts,
                    "CALCULATION_",
                )
                _record_step_errors(
                    target,
                    predicted,
                    _causal_step_blocks(target),
                    step_event_counts,
                )
                _record_aggregate_checks(sample, predicted, aggregate_counts)

    metrics = _metrics(total_loss, totals)
    metrics["cross_entropy_loss"] = metrics["loss"]
    metrics["consistency_loss"] = total_consistency_loss / max(1, consistency_terms)
    return {
        "token_metrics": {
            **metrics,
            "counts": {key: totals[key] for key in sorted(totals)},
        },
        "section_accuracy": {
            key: _counter_rate(value, "correct", "tokens")
            for key, value in sorted(section_counts.items())
        },
        "critical_region_accuracy": {
            key: _counter_rate(value, "correct", "tokens")
            for key, value in sorted(region_counts.items())
        },
        "worst_groups": _top_rates(
            group_counts,
            correct="correct",
            total="tokens",
            top_k=args.top_k,
            min_total=50,
            worst=True,
        ),
        "largest_groups": _top_rates(
            group_counts,
            correct="correct",
            total="tokens",
            top_k=args.top_k,
            min_total=50,
        ),
        "event_type_accuracy": {
            key: {
                "exact": _counter_rate(value, "exact", "blocks"),
                "token": _counter_rate(value, "correct", "tokens"),
            }
            for key, value in sorted(event_counts.items())
        },
        "calculation_type_accuracy": {
            key: {
                "exact": _counter_rate(value, "exact", "blocks"),
                "token": _counter_rate(value, "correct", "tokens"),
            }
            for key, value in sorted(calculation_counts.items())
        },
        "step_event_accuracy": {
            key: {
                "exact": _counter_rate(value, "exact", "blocks"),
                "token": _counter_rate(value, "correct", "tokens"),
            }
            for key, value in sorted(step_event_counts.items())
        },
        "number_accuracy": {
            key: {
                **_counter_rate(value, "correct", "total"),
                "unparseable": int(value.get("unparseable", 0)),
                "mae_on_wrong_parseable": (
                    value.get("absolute_error_sum", 0) / value["wrong"]
                    if value.get("wrong", 0)
                    else None
                ),
            }
            for key, value in sorted(number_counts.items())
        },
        "effect_aggregate_accuracy": {
            key: _counter_rate(value, "correct", "total")
            for key, value in sorted(aggregate_counts.items())
        },
        "top_token_confusions": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in token_confusions.most_common(args.top_k)
        ],
        "top_critical_confusions": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in critical_confusions.most_common(args.top_k)
        ],
        "first_mismatch_sections": dict(section_first_mismatch),
        "errors": errors,
    }


def _greedy_diagnostics(
    model: RuleWorldModel,
    tokenizer: RuleTokenizer,
    samples: list[dict[str, Any]],
    encoded: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.greedy_samples <= 0:
        return {"samples": 0}
    allowed_actions = set(args.greedy_action or ())
    allowed_cards = set(args.greedy_card or ())
    order = []
    for index, sample in enumerate(samples):
        action = sample["action"]
        card = action.get("card") or {}
        if allowed_actions and action.get("type") not in allowed_actions:
            continue
        if allowed_cards and card.get("id") not in allowed_cards:
            continue
        order.append(index)
    order.sort(key=lambda index: len(encoded[index]["target_ids"]))
    limit = min(args.greedy_samples, len(order))
    order = order[:limit]
    totals = Counter()
    group_counts: dict[str, Counter] = defaultdict(Counter)
    errors = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(order), max(1, args.batch_size)):
            indices = order[begin:begin + max(1, args.batch_size)]
            prefix = _pad([encoded[index]["prefix_ids"] for index in indices], tokenizer.pad_id, device)
            generated = model.generate(
                prefix,
                output_begin_id=tokenizer.index[OUTPUT_BEGIN],
                output_end_id=tokenizer.index[OUTPUT_END],
                max_new_tokens=args.greedy_max_tokens,
            ).cpu()
            for row, index in enumerate(indices):
                expected = encoded[index]["target"]
                predicted = _trim_output(tokenizer.decode(generated[row].tolist()))
                complete = predicted == expected
                checks = {
                    "complete": complete,
                    "terminated": bool(predicted and predicted[-1] == OUTPUT_END),
                }
                for name, left, right in SECTION_MARKERS:
                    if left not in expected:
                        continue
                    checks[f"{name}_exact"] = _exact_section(expected, predicted, left, right)
                totals["samples"] += 1
                for key, passed in checks.items():
                    totals[key] += int(passed)
                for tag in _sample_tags(samples[index]):
                    row_counter = group_counts[tag]
                    row_counter["samples"] += 1
                    for key, passed in checks.items():
                        row_counter[key] += int(passed)
                if not complete and len(errors) < args.show_errors:
                    mismatch = _first_mismatch(expected, predicted)
                    errors.append({
                        "sample": index,
                        "target_length": len(expected),
                        "predicted_length": len(predicted),
                        "first_mismatch": mismatch,
                        "expected_context": expected[max(0, (mismatch or 0) - 8):(mismatch or 0) + 16],
                        "predicted_context": predicted[max(0, (mismatch or 0) - 8):(mismatch or 0) + 16],
                    })
    sample_count = max(1, totals["samples"])
    return {
        "samples": totals["samples"],
        "metrics": {
            key: totals[key] / sample_count
            for key in ("complete", "terminated", "rule_steps_exact", "calculations_exact", "effects_exact", "next_state_exact")
        },
        "worst_groups": _top_rates(
            group_counts,
            correct="complete",
            total="samples",
            top_k=args.top_k,
            min_total=1,
            worst=True,
        ),
        "errors": errors,
    }


def _short_rate(row: dict[str, Any]) -> str:
    accuracy = row.get("accuracy")
    if accuracy is None:
        return "n/a"
    return f"{accuracy:.2%} ({row['correct']}/{row['total']})"


def _summary(report: dict[str, Any], top_k: int) -> str:
    teacher = report["teacher_forced"]
    lines = []
    lines.append("teacher_forced_sections")
    for name in ("rule_steps", "calculations", "effects", "next_state"):
        if name in teacher["section_accuracy"]:
            lines.append(f"  {name}: {_short_rate(teacher['section_accuracy'][name])}")
    lines.append("critical_regions")
    for name in (
        "step_number", "calculation_number", "effect_number", "next_number",
        "causal", "counter", "structure",
    ):
        if name in teacher["critical_region_accuracy"]:
            lines.append(f"  {name}: {_short_rate(teacher['critical_region_accuracy'][name])}")
    lines.append("worst_groups")
    for row in teacher["worst_groups"][:top_k]:
        lines.append(f"  {row['name']}: {_short_rate(row)}")
    lines.append("first_mismatch_sections")
    for name, count in sorted(
        teacher["first_mismatch_sections"].items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]:
        lines.append(f"  {name}: {count}")
    if report["greedy"]["samples"]:
        lines.append("greedy")
        for key, value in sorted(report["greedy"]["metrics"].items()):
            lines.append(f"  {key}: {value:.2%}")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    run_config = _read_run_config(args.run)
    validation_fraction = (
        args.validation_fraction
        if args.validation_fraction is not None
        else float(run_config.get("validation_fraction", 0.05))
    )
    seed = args.seed if args.seed is not None else int(run_config.get("seed", 0))
    validation_samples = (
        args.validation_samples
        if args.validation_samples is not None
        else int(run_config.get("validation_samples", 0) or 0)
    )
    samples = list(read_jsonl(args.data))
    _train_samples, validation, split_unit = _split_samples(samples, validation_fraction, seed)
    if validation_samples > 0:
        validation = validation[:validation_samples]
    if not validation:
        raise ValueError("empty validation split")

    tokenizer_path = args.run / "tokenizer.json"
    output_version = int(json.loads(
        tokenizer_path.read_text(encoding="utf-8")
    ).get("version", -1))
    tokenizer = RuleTokenizer.load(tokenizer_path)
    model = RuleWorldModel.from_config(args.run / "model_config.json")
    checkpoint = torch.load(args.run / args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    device = _device(args.device)
    model.to(device)
    encoded = _build_encoded(validation, tokenizer, output_version)
    unknowns = sum(item["unknowns"] for item in encoded)
    teacher_count = (
        min(args.teacher_samples, len(validation))
        if args.teacher_samples > 0
        else len(validation)
    )
    report = {
        "checkpoint": str(args.run / args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "data_samples": len(samples),
        "validation_samples": len(validation),
        "teacher_samples": teacher_count,
        "split_unit": split_unit,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "unknown_tokens": unknowns,
    }
    report["teacher_forced"] = _teacher_forced_diagnostics(
        model,
        tokenizer,
        validation[:teacher_count],
        encoded[:teacher_count],
        device,
        args,
        run_config,
    )
    report["greedy"] = _greedy_diagnostics(
        model, tokenizer, validation, encoded, device, args
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(_summary(report, min(args.top_k, 10)))
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
