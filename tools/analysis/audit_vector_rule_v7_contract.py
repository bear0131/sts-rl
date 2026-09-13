"""Audit schema-v7 contract-10 local sequences, masks, and delta alignment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from rl.rules.data.make_vector_rule_v7_data import (
    V7_EXECUTOR_CONTRACT_VERSION,
    build_v7_record,
)
from rl.rules.rule_vector_model import StateVectorConfig
from rl.rules.vector_rule_v7_data import V7TokenCodec


NUMBER_PREFIXES = ("NUM_", "DIGIT_", "SIGN_")
ENVIRONMENT_SPANS = {
    "<OBS_BEGIN>": "<OBS_END>",
}


def _walk(node: Mapping[str, Any], path: str = "root") -> Iterable[tuple[str, Mapping[str, Any]]]:
    yield path, node
    for index, child in enumerate(node.get("children", ())):
        yield from _walk(child, f"{path}/{index}")


def _validate_row(row: Mapping[str, Any], node: Mapping[str, Any]) -> None:
    tokens = tuple(str(token) for token in row["tokens"])
    mask = tuple(bool(value) for value in row["model_mask"])
    condition_length = int(row["condition_length"])
    if len(tokens) != len(mask):
        raise ValueError("local token/mask lengths differ")
    if any(mask[:condition_length]):
        raise ValueError("NodeSpec prefix contains model-owned labels")
    if any(any(char.isspace() for char in token) for token in tokens):
        raise ValueError("a logical token contains whitespace")

    environment = [False] * len(tokens)
    for begin, end in ENVIRONMENT_SPANS.items():
        active = False
        for index, token in enumerate(tokens):
            if token == begin:
                if active:
                    raise ValueError(f"nested {begin} span")
                active = True
            if active:
                environment[index] = True
            if token == end:
                if not active:
                    raise ValueError(f"unmatched {end}")
                active = False
        if active:
            raise ValueError(f"unclosed {begin} span")

    # READ_BIND closes a request and is assigned by the environment. The
    # source field itself remains model-owned so it still learns what value to
    # ask for. The binding name is environment-owned and only identifies the
    # value store entry used by later inline constants.
    for index, token in enumerate(tokens):
        if token == "READ_BIND" or (
            index > 0 and tokens[index - 1] == "READ_BIND"
        ):
            environment[index] = True
        if token == "<READ_REQUEST_END>":
            environment[index] = True
        if token == "CONST_VALUE":
            environment[index] = True
            for next_index in range(index + 1, len(tokens)):
                environment[next_index] = True
                if tokens[next_index] == "NUM_END":
                    break

    for index in range(condition_length, len(tokens)):
        expected = not environment[index]
        if mask[index] != expected:
            raise ValueError(
                f"ownership mismatch at {row['path']} token {index}: {tokens[index]}"
            )
        if environment[index] and tokens[index].startswith(NUMBER_PREFIXES) and mask[index]:
            raise ValueError("environment-owned number contributes token loss")

    forbidden = {
        "<READ_RESPONSE_BEGIN>", "<READ_RESPONSE_END>", "READ_VALUE",
        "<REGISTER_CONTEXT_BEGIN>", "<REGISTER_CONTEXT_END>",
        "<REGISTER_BEGIN>", "<REGISTER_END>", "REGISTER_NAME",
        "REGISTER_VALUE_BEGIN", "REGISTER_VALUE_END",
    }
    if forbidden.intersection(tokens):
        raise ValueError(f"contract-10 row still contains removed context tokens at {row['path']}")

    # Contract 10 expands both requested runtime values and earlier model
    # calculation results whenever they are reused as numeric operands.
    valued_registers = {
        tokens[index + 1]
        for index, token in enumerate(tokens[:-1])
        if token == "READ_BIND" and tokens[index + 1].startswith("VALUE_reg_")
    }
    result_fields = {"result", "source_result", "destination_result"}
    result_field_tokens = {f"CALC_FIELD_{field}" for field in result_fields}
    for index, token in enumerate(tokens[:-1]):
        if token not in result_field_tokens:
            continue
        register = tokens[index + 1]
        if not register.startswith("VALUE_reg_"):
            continue
        calculation_end = tokens.index("CALCULATION_END", index + 1)
        if "CALC_VALUE" in tokens[index + 2:calculation_end]:
            valued_registers.add(register)
    for index, token in enumerate(tokens):
        if token != "CONST_VALUE":
            continue
        if mask[index]:
            raise ValueError(f"inline constant is model-owned at {row['path']}")
        if index == 0 or not tokens[index - 1].startswith("VALUE_reg_"):
            raise ValueError(f"inline constant does not follow a register at {row['path']}")
        number_positions: list[int] = []
        for next_index in range(index + 1, len(tokens)):
            next_token = tokens[next_index]
            if next_token.startswith(NUMBER_PREFIXES):
                number_positions.append(next_index)
            if next_token == "NUM_END":
                break
            if next_token.startswith("CALC_FIELD_") or next_token == "CALCULATION_END":
                break
        if not number_positions or tokens[number_positions[-1]] != "NUM_END":
            raise ValueError(f"inline constant has no complete number at {row['path']}")
        if any(mask[position] for position in number_positions):
            raise ValueError(f"inline constant number is model-owned at {row['path']}")
    calc_begin = tokens.index("<CALCULATIONS_BEGIN>")
    for index in range(calc_begin + 1, len(tokens) - 1):
        token = tokens[index]
        if not token.startswith("VALUE_reg_") or token not in valued_registers:
            continue
        field = tokens[index - 1][len("CALC_FIELD_"):] if (
            index > 0 and tokens[index - 1].startswith("CALC_FIELD_")
        ) else ""
        if field in result_fields:
            continue
        if index + 1 >= len(tokens) or tokens[index + 1] != "CONST_VALUE":
            raise ValueError(f"valued register operand is missing inline constant at {row['path']}")

    delta_slots = node.get("delta_slots")
    expected_delta_count = (
        len(delta_slots) if isinstance(delta_slots, list)
        else int(node.get("delta") is not None)
    )
    delta_count = tokens.count("DELTA_SLOT")
    if delta_count != expected_delta_count:
        raise ValueError(f"DELTA_SLOT mismatch at {row['path']}")
    for index, token in enumerate(tokens):
        if token != "CALC_VALUE":
            continue
        number_tokens = []
        for next_index in range(index + 1, len(tokens)):
            next_token = tokens[next_index]
            if next_token == "CALCULATION_END" or next_token.startswith("CALC_FIELD_"):
                break
            if next_token.startswith(NUMBER_PREFIXES):
                number_tokens.append(next_index)
        if not number_tokens or not all(mask[position] for position in number_tokens):
            raise ValueError(f"CALC_VALUE number is not model-owned at {row['path']}")


def _validate_relic_counter_rows(
    record: Mapping[str, Any], rows: Iterable[Mapping[str, Any]],
) -> int:
    """Ensure counter regions can only address stateful, present relic slots."""

    relics = record.get("relics", ())
    if not isinstance(relics, list):
        raise ValueError("v7 relics must be a list")
    checked = 0
    for row in rows:
        if str(row.get("kind")) != "RELIC_COUNTER":
            continue
        tokens = tuple(str(token) for token in row.get("tokens", ()))
        slot_tokens = [token for token in tokens if token.startswith("ENTITY_RELIC_SLOT_")]
        if len(slot_tokens) != 1:
            raise ValueError(f"RELIC_COUNTER row {row.get('path')} has no unique relic slot")
        slot = int(slot_tokens[0].rsplit("_", 1)[-1])
        if slot < 0 or slot >= len(relics):
            raise ValueError(f"RELIC_COUNTER row {row.get('path')} addresses absent slot {slot}")
        relic = relics[slot]
        if not isinstance(relic, Mapping) or not bool(relic.get("present", False)):
            raise ValueError(f"RELIC_COUNTER row {row.get('path')} addresses a non-present relic")
        if not bool(relic.get("counter_present", False)):
            raise ValueError(f"RELIC_COUNTER row {row.get('path')} addresses a stateless relic")
        checked += 1
    return checked


def audit(*, samples: int, seed: int, vocab_samples: int, vocab_seed: int) -> dict[str, Any]:
    config = StateVectorConfig()
    vocab_rng = random.Random(vocab_seed)
    vocab_records = [
        build_v7_record(index, config, vocab_rng) for index in range(vocab_samples)
    ]
    codec = V7TokenCodec.from_records(vocab_records)

    kind_count: Counter[str] = Counter()
    kind_max_tokens: dict[str, int] = defaultdict(int)
    unknown: Counter[str] = Counter()
    total_nodes = 0
    total_delta_slots = 0
    total_relic_counter_regions = 0
    max_outer = {"decoder_input_length": 0, "token_length": 0, "record_index": -1}
    max_local = {
        "decoder_input_length": 0, "token_length": 0,
        "record_index": -1, "path": "", "kind": "",
    }
    max_owned_body = {"length": 0, "record_index": -1, "path": "", "kind": ""}
    max_repeat_body = {"length": 0, "record_index": -1}
    max_diagnostic = {"length": 0, "record_index": -1}

    rng = random.Random(seed)
    for index in range(samples):
        record = build_v7_record(index, config, rng)
        nodes = dict(_walk(record["effect_tree"]))
        outer_token_length = (
            int(record["lengths"]["outer_header_length"])
            + int(record["lengths"]["node_count"])
            + 6
        )
        if outer_token_length > max_outer["token_length"]:
            max_outer = {
                "decoder_input_length": outer_token_length - 1,
                "token_length": outer_token_length,
                "record_index": index,
            }
        repeat_length = int(record["lengths"]["repeat_static_body_length_max"])
        if repeat_length > max_repeat_body["length"]:
            max_repeat_body = {"length": repeat_length, "record_index": index}
        diagnostic_length = len(record["tokens"])
        if diagnostic_length > max_diagnostic["length"]:
            max_diagnostic = {"length": diagnostic_length, "record_index": index}

        for token in record["tokens"]:
            if token not in codec.token_to_id:
                unknown[str(token)] += 1
        for row in record["node_local_sequences"]:
            path = str(row["path"])
            kind = str(row["kind"])
            _validate_row(row, nodes[path])
            total_relic_counter_regions += _validate_relic_counter_rows(
                record, (row,)
            )
            total_nodes += 1
            kind_count[kind] += 1
            length = int(row["length"])
            owned_length = int(row["target_length"])
            kind_max_tokens[kind] = max(kind_max_tokens[kind], length)
            total_delta_slots += row["tokens"].count("DELTA_SLOT")
            if length > max_local["token_length"]:
                max_local = {
                    "decoder_input_length": length - 1,
                    "token_length": length, "record_index": index,
                    "path": path, "kind": kind,
                }
            if owned_length > max_owned_body["length"]:
                max_owned_body = {
                    "length": owned_length, "record_index": index,
                    "path": path, "kind": kind,
                }
            for token in row["tokens"]:
                if token not in codec.token_to_id:
                    unknown[str(token)] += 1

    return {
        "schema_version": 7,
        "executor_contract_version": V7_EXECUTOR_CONTRACT_VERSION,
        "samples": samples,
        "seed": seed,
        "vocab_samples": vocab_samples,
        "vocab_seed": vocab_seed,
        "vocab_size": len(codec.tokens),
        "total_nodes": total_nodes,
        "total_delta_slots": total_delta_slots,
        "total_relic_counter_regions": total_relic_counter_regions,
        "max_outer_input": max_outer,
        "max_node_local_input": max_local,
        "max_model_owned_body": max_owned_body,
        "max_repeat_static_body": max_repeat_body,
        "max_flat_diagnostic_tree": max_diagnostic,
        "node_kinds": {
            kind: {
                "count": kind_count[kind],
                "max_decoder_input_length": kind_max_tokens[kind] - 1,
                "max_token_length": kind_max_tokens[kind],
            }
            for kind in sorted(kind_count)
        },
        "unknown_token_occurrences": sum(unknown.values()),
        "unknown_tokens": dict(unknown.most_common()),
        "invariants": {
            "delta_slot_alignment": "passed",
            "read_binding_masking": "passed",
            "inline_constant_expansion": "passed",
            "calculation_result_ownership": "passed",
            "one_logical_token_per_item": "passed",
            "relic_counter_presence": "passed",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=4901)
    parser.add_argument("--vocab-samples", type=int, default=4_000)
    parser.add_argument("--vocab-seed", type=int, default=4900)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 1 or args.vocab_samples < 1:
        raise ValueError("sample counts must be positive")
    result = audit(
        samples=args.samples, seed=args.seed,
        vocab_samples=args.vocab_samples, vocab_seed=args.vocab_seed,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
