"""Analyze model-owned calculation results from a v7 evaluation report.

The regular v7 report compares the continuous ``DELTA_SLOT`` prediction with
the target state vector.  This diagnostic reads the already saved local token
traces and compares the numeric values emitted after ``CALC_VALUE`` instead.
It therefore separates arithmetic/register-result errors from errors in the
final continuous delta head without loading a checkpoint or running a model.

Example::

    python -m rl.evaluation.rules.eval_vector_rule_v7_intermediate \
        --input artifacts/archive/docs/simple-card-v7-eval-remote-latest.json \
        --json-output artifacts/local/simple-card-v7-intermediate-remote-latest.json \
        --markdown-output artifacts/local/simple-card-v7-intermediate-remote-latest.md
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _parse_number(tokens: Sequence[str], start: int) -> tuple[float | None, int]:
    """Decode one v7 numeric token sequence and return ``(value, next_pos)``."""

    if start >= len(tokens) or tokens[start] not in {"NUM_LSD_BEGIN", "NUM_BEGIN"}:
        return None, start
    lsd = tokens[start] == "NUM_LSD_BEGIN"
    index = start + 1
    negative = False
    if index < len(tokens) and tokens[index] == "SIGN_NEG":
        negative = True
        index += 1
    digits: list[int] = []
    decimal: list[str] = []
    while index < len(tokens) and tokens[index] != "NUM_END":
        token = tokens[index]
        if token.startswith("DIGIT_"):
            value = token.removeprefix("DIGIT_")
            if len(value) != 1 or not value.isdigit():
                return None, index
            if lsd:
                digits.append(int(value))
            else:
                decimal.append(value)
        elif token == "DECIMAL" and not lsd:
            decimal.append(".")
        else:
            return None, index
        index += 1
    if index >= len(tokens):
        return None, index
    if lsd:
        number = float(sum(value * (10 ** power) for power, value in enumerate(digits)))
    else:
        try:
            number = float("".join(decimal))
        except ValueError:
            return None, index + 1
    return (-number if negative else number), index + 1


def _calculation_blocks(tokens: Sequence[str]) -> list[dict[str, Any]]:
    """Parse calculation blocks independently of token-position alignment."""

    try:
        index = list(tokens).index("<CALCULATIONS_BEGIN>") + 1
    except ValueError:
        return []
    blocks: list[dict[str, Any]] = []
    while index < len(tokens) and tokens[index] != "<CALCULATIONS_END>":
        if tokens[index] != "CALCULATION_BEGIN":
            index += 1
            continue
        end = index + 1
        while end < len(tokens) and tokens[end] not in {
            "CALCULATION_END", "<CALCULATIONS_END>",
        }:
            end += 1
        body = list(tokens[index + 1:end])
        operation = next(
            (token.removeprefix("CALCULATION_") for token in body
             if token.startswith("CALCULATION_")
             and token not in {"CALCULATION_BEGIN", "CALCULATION_END"}),
            None,
        )
        result_register = None
        for position, token in enumerate(body[:-1]):
            if token == "CALC_FIELD_result" and body[position + 1].startswith("VALUE_"):
                result_register = body[position + 1].removeprefix("VALUE_")
                break
        values: list[float] = []
        position = 0
        while position < len(body):
            if body[position] == "CALC_VALUE":
                value, next_position = _parse_number(body, position + 1)
                if value is not None:
                    values.append(value)
                position = max(next_position, position + 1)
            else:
                position += 1
        blocks.append({
            "operation": operation,
            "result_register": result_register,
            "values": values,
        })
        index = end + 1
    return blocks


def _pair_blocks(
    expected: Sequence[Mapping[str, Any]], predicted: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
    """Pair by operation and occurrence so one wrong number does not shift all rows."""

    queues: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
    for block in predicted:
        queues[block.get("operation")].append(block)
    pairs = []
    for block in expected:
        operation = block.get("operation")
        candidates = queues[operation]
        pairs.append((block, candidates.pop(0) if candidates else None))
    return pairs


def _result_rows(case_name: str, node: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = _calculation_blocks(node.get("expected_tokens", ()))
    predicted = _calculation_blocks(node.get("predicted_tokens", ()))
    rows: list[dict[str, Any]] = []
    for block_index, (expected_block, predicted_block) in enumerate(
        _pair_blocks(expected, predicted)
    ):
        expected_values = list(expected_block.get("values", ()))
        predicted_values = list(predicted_block.get("values", ())) if predicted_block else []
        for value_index, expected_value in enumerate(expected_values):
            predicted_value = (
                predicted_values[value_index]
                if value_index < len(predicted_values) else None
            )
            rows.append({
                "case": case_name,
                "path": node.get("path"),
                "block_index": block_index,
                "operation": expected_block.get("operation"),
                "result_register": expected_block.get("result_register"),
                "expected": expected_value,
                "predicted": predicted_value,
                "correct": (
                    predicted_value is not None
                    and abs(float(expected_value) - float(predicted_value)) <= 1.0e-6
                ),
            })
    return rows


def _iter_local_nodes(report: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for case in report.get("cases", ()):
        case_name = str(case.get("name", ""))
        for node in case.get("evaluation", {}).get("local", ()):
            yield case_name, node


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    errors = [
        abs(float(row["expected"]) - float(row["predicted"]))
        for row in rows if row["predicted"] is not None
    ]
    non_terminal = [
        row for row in rows
        if "target_hp_after" not in str(row.get("result_register", ""))
    ]
    damage_amount = [
        row for row in rows
        if str(row.get("result_register", "")) in {
            "reg_unblocked_damage", "reg_damage_after_vulnerable",
        }
    ]

    def metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        item_correct = sum(bool(row["correct"]) for row in items)
        item_errors = [
            abs(float(row["expected"]) - float(row["predicted"]))
            for row in items if row["predicted"] is not None
        ]
        return {
            "results": len(items),
            "correct": item_correct,
            "accuracy": item_correct / max(len(items), 1),
            "mae": sum(item_errors) / max(len(item_errors), 1),
        }

    by_operation: dict[str, dict[str, Any]] = {}
    for operation in sorted({str(row.get("operation")) for row in rows}):
        by_operation[operation] = metrics([
            row for row in rows if str(row.get("operation")) == operation
        ])
    by_register: dict[str, dict[str, Any]] = {}
    for register in sorted({str(row.get("result_register")) for row in rows}):
        by_register[register] = metrics([
            row for row in rows if str(row.get("result_register")) == register
        ])
    return {
        "calculation_results": len(rows),
        "calculation_results_correct": correct,
        "calculation_result_accuracy": correct / max(len(rows), 1),
        "calculation_result_mae": sum(errors) / max(len(errors), 1),
        "non_terminal_results": metrics(non_terminal),
        "damage_amount_results": metrics(damage_amount),
        "by_operation": by_operation,
        "by_result_register": by_register,
        "wrong_results": [dict(row) for row in rows if not row["correct"]],
    }


def analyze(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        result
        for case_name, node in _iter_local_nodes(report)
        for result in _result_rows(case_name, node)
    ]
    return {
        "source_checkpoint": report.get("checkpoint"),
        "source_checkpoint_step": report.get("checkpoint_step"),
        "source_executor_contract_version": report.get("executor_contract_version"),
        "summary": _aggregate(rows),
        "results": rows,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# v7 中间计算结果诊断",
        "",
        f"source checkpoint: `{result['source_checkpoint']}`; step={result['source_checkpoint_step']}; contract={result['source_executor_contract_version']}",
        "",
        "这里比较的是 local 序列中每个 `CALC_VALUE` 的数值，不经过最后的 `DELTA_SLOT`。模型仍是远端评测时的 greedy 输出；本报告只重新解析已保存的 token trace。",
        "",
        f"- CALC_VALUE 数值准确率：{summary['calculation_results_correct']}/{summary['calculation_results']} = {summary['calculation_result_accuracy']:.4f}",
        f"- CALC_VALUE 数值 MAE：{summary['calculation_result_mae']:.4f}",
        f"- 排除最终 HP 计算后的准确率：{summary['non_terminal_results']['correct']}/{summary['non_terminal_results']['results']} = {summary['non_terminal_results']['accuracy']:.4f}",
        f"- 中间伤害量（`reg_unblocked_damage`/`reg_damage_after_vulnerable`）准确率：{summary['damage_amount_results']['correct']}/{summary['damage_amount_results']['results']} = {summary['damage_amount_results']['accuracy']:.4f}",
        "",
        "## 按算子",
        "",
        "| operator | correct / total | accuracy | MAE |",
        "|---|---:|---:|---:|",
    ]
    for operation, metrics in summary["by_operation"].items():
        lines.append(
            f"| `{operation}` | {metrics['correct']} / {metrics['results']} | {metrics['accuracy']:.4f} | {metrics['mae']:.4f} |"
        )
    lines.extend(("", "## 错误的中间结果", "", "| case | path | operator | result register | expected | predicted |", "|---|---|---|---|---:|---:|"))
    for row in summary["wrong_results"]:
        predicted = "missing" if row["predicted"] is None else f"{row['predicted']:g}"
        lines.append(
            f"| {row['case']} | {row['path']} | `{row['operation']}` | `{row['result_register']}` | {row['expected']:g} | {predicted} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    summary = result["summary"]
    print(
        "[v7-intermediate] results=%d correct=%d accuracy=%.6f mae=%.6f "
        "non_terminal=%.6f damage_amount=%.6f"
        % (
            summary["calculation_results"],
            summary["calculation_results_correct"],
            summary["calculation_result_accuracy"],
            summary["calculation_result_mae"],
            summary["non_terminal_results"]["accuracy"],
            summary["damage_amount_results"]["accuracy"],
        )
    )


if __name__ == "__main__":
    main()
