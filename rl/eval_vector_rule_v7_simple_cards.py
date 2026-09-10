"""Evaluate the previously used simple-card probes with the v7 evaluator.

The suite keeps one protocol and one set of metrics for Strike, Perfected
Strike, a tampered Strike rule, and Impervious.  Perfected Strike receives two
extra cases with zero and three Strike cards in the piles.  The current v7
contract intentionally hides pile composition, so the report records both the
game-semantic damage and the damage represented by the current teacher trace.
This makes the missing runtime count request visible instead of silently
calling it a model error.

Example::

    python -m rl.eval_vector_rule_v7_simple_cards \
        --checkpoint .tmp-vector-rule-v7-contract-v10-register-values-20260901/last.pt \
        --tokenizer .tmp-vector-rule-v7-contract-v10-register-values-20260901/tokenizer.json \
        --json-output docs/simple-card-v7-eval.json \
        --markdown-output docs/simple-card-v7-eval.md \
        --device cuda
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from env.core.card_language import TOKEN_INDEX as CARD_RULE_INDEX, card_program
from env.core.ironclad_cards import IRONCLAD_CARDS
from rl.eval_vector_rule_v7_cases import (
    EvaluationCase,
    _fixed_v6_record,
    evaluate_case,
)
from rl.make_vector_rule_v7_data import compile_v7_record
from rl.render_rule_vector_examples import _state
from rl.rule_vector_model import StateVectorConfig, VectorRuleModelConfig
from rl.vector_rule_v7_data import V7_EXECUTOR_CONTRACT_VERSION, V7TokenCodec


CARD_NAME_ZH = {
    "Strike_R": "\u6253\u51fb",
    "Perfected Strike": "\u5b8c\u7f8e\u6253\u51fb",
    "Impervious": "\u5c96\u7136\u4e0d\u52a8",
}


@dataclass(frozen=True)
class SimpleCardSpec:
    name: str
    card_id: str
    state: Mapping[str, Any]
    hand: tuple[tuple[str, int], ...]
    oracle_field: str
    game_effect_value: float
    note: str
    tamper_from: int | None = None
    tamper_to: int | None = None
    tamper_field: str = "damage"
    strike_count: int | None = None


def _combat_state(
    *,
    enemy_block: int = 0,
    enemy_powers: Mapping[str, int] | None = None,
    player_powers: Mapping[str, int] | None = None,
    draw: int = 0,
    discard: int = 0,
    exhaust: int = 0,
) -> dict[str, Any]:
    state = _state(
        energy=5,
        cards_played=0,
        attacks_played=0,
        sundial=0,
        enemy_hp=100,
        enemy_block=enemy_block,
        draw=draw,
        discard=discard,
        exhaust=exhaust,
    )
    state["combat"]["player"]["powers"] = dict(player_powers or {})
    state["combat"]["monsters"][0]["powers"] = dict(enemy_powers or {})
    return state


def _strike_piles(state: dict[str, Any], count: int) -> None:
    """Put exactly ``count`` Strike cards in the hidden combat piles."""

    state["combat"]["draw"] = [
        {"id": "Strike_R", "up": 0} for _ in range(count)
    ]
    state["combat"]["discard"] = []
    state["combat"]["exhaust"] = []


def _three_strike_state() -> dict[str, Any]:
    state = _combat_state(draw=3)
    _strike_piles(state, 3)
    return state


def build_specs() -> tuple[SimpleCardSpec, ...]:
    """Return the fixed probes used by the old simple-card evaluations."""

    hand = (("Strike_R", 0),)
    perfected_hand = (("Perfected Strike", 0),)
    impervious_hand = (("Impervious", 0),)
    return (
        SimpleCardSpec(
            "strike_base", "Strike_R", _combat_state(), hand,
            "enemy[0].hp", 6.0, "\u6253\u51fb\uff1a\u65e0\u6613\u4f24\u3001\u65e0\u683c\u6321\u3002",
        ),
        SimpleCardSpec(
            "strike_vulnerable", "Strike_R",
            _combat_state(enemy_powers={"Vulnerable": 2}), hand,
            "enemy[0].hp", 9.0, "\u6253\u51fb\uff1a\u654c\u4eba2\u5c42\u6613\u4f24\uff0c6\u70b9\u4f24\u5bb3\u6309\u6613\u4f24\u7ed3\u7b97\u4e3a9\u3002",
        ),
        SimpleCardSpec(
            "strike_block", "Strike_R", _combat_state(enemy_block=4), hand,
            "enemy[0].hp", 2.0, "\u6253\u51fb\uff1a\u654c\u4eba4\u70b9\u683c\u6321\uff0c\u53ea\u6709\u5269\u4f59\u4f24\u5bb3\u6263\u8840\u3002",
        ),
        SimpleCardSpec(
            "strike_vulnerable_block", "Strike_R",
            _combat_state(enemy_block=4, enemy_powers={"Vulnerable": 2}), hand,
            "enemy[0].hp", 5.0, "\u6253\u51fb\uff1a\u6613\u4f24\u4e0e\u683c\u6321\u540c\u65f6\u5b58\u5728\u3002",
        ),
        SimpleCardSpec(
            "strike_tampered_6_to_12", "Strike_R", _combat_state(), hand,
            "enemy[0].hp", 12.0, "\u6253\u51fb\uff1a\u5c06\u5361\u9762\u57fa\u7840\u4f24\u5bb36\u7be1\u6539\u4e3a12\u3002",
            tamper_from=6, tamper_to=12,
        ),
        SimpleCardSpec(
            "perfected_strike_zero_strikes", "Perfected Strike", _combat_state(),
            perfected_hand, "enemy[0].hp", 6.0,
            "\u5b8c\u7f8e\u6253\u51fb\uff1a\u9690\u85cf\u4e09\u5806\u4e2d0\u5f20Strike\uff0c\u8bed\u4e49\u4e0a\u671f\u5f854\u4f24\u5bb3\u3002",
            strike_count=0,
        ),
        SimpleCardSpec(
            "perfected_strike_three_strikes", "Perfected Strike", _three_strike_state(),
            perfected_hand, "enemy[0].hp", 12.0,
            "\u5b8c\u7f8e\u6253\u51fb\uff1a\u9690\u85cf\u4e09\u5806\u4e2d3\u5f20Strike\uff0c\u8bed\u4e49\u4e0a\u671f\u5f8512\u4f24\u5bb3\u3002",
            strike_count=3,
        ),
        SimpleCardSpec(
            "perfected_strike_vulnerable", "Perfected Strike",
            _combat_state(enemy_powers={"Vulnerable": 2}), perfected_hand,
            "enemy[0].hp", 9.0,
            "\u5b8c\u7f8e\u6253\u51fb\uff1a\u4e0d\u542bStrike\u65f6\u53d7\u6613\u4f24\u5f71\u54cd\uff0c\u5f53\u524dteacher\u57fa\u7840\u9879\u4e3a6\u3002",
            strike_count=0,
        ),
        SimpleCardSpec(
            "impervious_base", "Impervious", _combat_state(), impervious_hand,
            "player.block", 30.0, "\u5c96\u7136\u4e0d\u52a8\uff1a\u83b7\u5f9730\u70b9\u683c\u6321\u3002",
        ),
        SimpleCardSpec(
            "impervious_frail", "Impervious",
            _combat_state(player_powers={"Frail": 2}), impervious_hand,
            "player.block", 22.0,
            "\u5c96\u7136\u4e0d\u52a8+\u8106\u5f31\uff1a\u6e38\u620f\u8bed\u4e49\u4e3afloor(30*0.75)=22\uff0c\u5f53\u524dv7 teacher\u6682\u672a\u5e94\u7528Frail\u3002",
        ),
        SimpleCardSpec(
            "impervious_tampered_30_to_60", "Impervious", _combat_state(),
            impervious_hand, "player.block", 60.0,
            "\u5c96\u7136\u4e0d\u52a8\uff1a\u5c06\u5361\u9762\u683c\u632130\u7be1\u6539\u4e3a60\u3002",
            tamper_from=30, tamper_to=60, tamper_field="block",
        ),
    )


def _patched_card_program(card_id: str, old: int, new: int) -> list[int]:
    tokens = list(card_program(card_id, False))
    old_tokens = [f"DIGIT_{digit}" for digit in str(int(old))]
    new_tokens = [f"DIGIT_{digit}" for digit in str(int(new))]
    for index in range(len(tokens) - len(old_tokens) + 1):
        if tokens[index:index + len(old_tokens)] != old_tokens:
            continue
        if index == 0 or tokens[index - 1] not in {"NUM_BEGIN", "NUM_LSD_BEGIN"}:
            continue
        end = index + len(old_tokens)
        if end < len(tokens) and tokens[end] == "NUM_END":
            tokens[index:end] = new_tokens
            return [CARD_RULE_INDEX[token] for token in tokens]
    raise ValueError(f"numeric literal {old} not found in {card_id}")


def _build_case(spec: SimpleCardSpec, config: StateVectorConfig) -> EvaluationCase:
    original = IRONCLAD_CARDS[spec.card_id]
    if spec.tamper_from is not None:
        if spec.tamper_field == "damage":
            patched = replace(original, damage=(spec.tamper_to, spec.tamper_to))
        elif spec.tamper_field == "block":
            patched = replace(original, block=(spec.tamper_to, spec.tamper_to))
        else:
            raise ValueError(f"unsupported tamper field: {spec.tamper_field}")
        IRONCLAD_CARDS[spec.card_id] = patched
    try:
        raw, canonical = _fixed_v6_record(
            index=0,
            state=deepcopy(spec.state),
            action_id=spec.card_id,
            action_upgrade=0,
            hand=spec.hand,
            config=config,
            seed=0,
        )
        record = compile_v7_record(raw, config)
        if spec.tamper_from is not None:
            program = _patched_card_program(
                spec.card_id, spec.tamper_from, int(spec.tamper_to)
            )
            record["action"]["card_rule"] = program
            for item in record["hand"]:
                item["card_rule"] = program
        return EvaluationCase(
            name=spec.name,
            group="simple_card",
            canonical=canonical,
            record=record,
            action_id=spec.card_id,
            action_upgrade=0,
            hand=spec.hand,
            relics=(),
            note=spec.note,
        )
    finally:
        IRONCLAD_CARDS[spec.card_id] = original


def build_cases(config: StateVectorConfig) -> tuple[tuple[SimpleCardSpec, EvaluationCase], ...]:
    return tuple((spec, _build_case(spec, config)) for spec in build_specs())


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _walk_nodes(node: Mapping[str, Any]):
    yield node
    for child in node.get("children", ()):
        yield from _walk_nodes(child)


def _protocol_damage_base(record: Mapping[str, Any]) -> int | None:
    for node in _walk_nodes(record["effect_tree"]):
        if node.get("kind") != "DAMAGE":
            continue
        for row in node.get("calculations", ()):
            if row.get("type") == "READ_BASE_DAMAGE":
                return int(row["value"])
    return None


def _field_scale(field: str, config: StateVectorConfig) -> float:
    if field.endswith(".hp"):
        return config.state.hp_scale
    if field.endswith(".block"):
        return config.state.block_scale
    return 1.0


def _target_result(evaluation: Mapping[str, Any], field: str, config: VectorRuleModelConfig) -> dict[str, Any]:
    for node in evaluation["local"]:
        for delta in node["delta_slots"]:
            for slot in delta.get("active_slots", ()):
                if slot["name"] != field:
                    continue
                scale = _field_scale(field, config)
                expected_delta = float(slot["expected"]) * scale
                predicted_delta = float(slot["predicted"]) * scale
                return {
                    "field": field,
                    "expected_delta": expected_delta,
                    "predicted_delta": predicted_delta,
                    "absolute_error": abs(expected_delta - predicted_delta),
                    "expected_effect": -expected_delta if field.endswith(".hp") else expected_delta,
                    "predicted_effect": -predicted_delta if field.endswith(".hp") else predicted_delta,
                }
    return {
        "field": field,
        "expected_delta": None,
        "predicted_delta": None,
        "absolute_error": None,
        "expected_effect": None,
        "predicted_effect": None,
    }


def _case_row(
    spec: SimpleCardSpec,
    case: EvaluationCase,
    evaluation: Mapping[str, Any],
    config: VectorRuleModelConfig,
) -> dict[str, Any]:
    target = _target_result(evaluation, spec.oracle_field, config)
    protocol_base = _protocol_damage_base(case.record)
    dynamic = None
    if spec.strike_count is not None:
        dynamic = {
            "strike_count": spec.strike_count,
            "game_expected_damage": spec.game_effect_value,
            "protocol_base_damage": protocol_base,
            "protocol_expected_damage": protocol_base,
            "count_applied_by_teacher": (
                protocol_base == spec.game_effect_value
            ),
            "count_read_token_present": any(
                "STRIKE_COUNT" in str(token)
                for row in case.record["node_local_sequences"]
                for token in row["tokens"]
            ),
            "pile_composition_visible_to_model": False,
        }
    return {
        "name": spec.name,
        "card_id": spec.card_id,
        "card_name_zh": CARD_NAME_ZH.get(spec.card_id, spec.card_id),
        "note": spec.note,
        "oracle_field": spec.oracle_field,
        "game_expected_effect": spec.game_effect_value,
        "tamper": {
            "from": spec.tamper_from,
            "to": spec.tamper_to,
            "field": spec.tamper_field,
        },
        "input": {
            "hand": list(spec.hand),
            "enemy_block": case.canonical["combat"]["monsters"][0].get("block", 0),
            "enemy_powers": case.canonical["combat"]["monsters"][0].get("powers", {}),
            "player_powers": case.canonical["combat"]["player"].get("powers", {}),
            "draw_count": len(case.canonical["combat"].get("draw", ())),
            "discard_count": len(case.canonical["combat"].get("discard", ())),
            "exhaust_count": len(case.canonical["combat"].get("exhaust", ())),
            "pile_composition_hidden": True,
        },
        "target": target,
        "perfected_strike_diagnostic": dynamic,
        "evaluation": evaluation,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outer_scored = sum(row["evaluation"]["outer"]["metrics"]["scored_tokens"] for row in rows)
    outer_correct = sum(row["evaluation"]["outer"]["metrics"]["correct_tokens"] for row in rows)
    local_rows = [node for row in rows for node in row["evaluation"]["local"]]
    local_scored = sum(node["metrics"]["scored_tokens"] for node in local_rows)
    local_correct = sum(node["metrics"]["correct_tokens"] for node in local_rows)
    target_rows = [row["target"] for row in rows if row["target"]["absolute_error"] is not None]
    return {
        "cases": len(rows),
        "outer_token_accuracy": outer_correct / max(outer_scored, 1),
        "outer_exact_cases": sum(row["evaluation"]["outer"]["metrics"]["exact_match"] for row in rows),
        "local_token_accuracy": local_correct / max(local_scored, 1),
        "local_exact_nodes": sum(node["metrics"]["exact_match"] for node in local_rows),
        "local_nodes": len(local_rows),
        "target_effect_mae": sum(float(row["absolute_error"]) for row in target_rows) / max(len(target_rows), 1),
        "unknown_output_tokens": sum(
            row["evaluation"]["outer"]["unknown_output_tokens"]
            + sum(node["unknown_output_tokens"] for node in row["evaluation"]["local"])
            for row in rows
        ),
        "perfected_strike_count_cases": sum(
            row["perfected_strike_diagnostic"] is not None for row in rows
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# v7 \u7b80\u5355\u5361\u724c\u7edf\u4e00\u8bc4\u6d4b",
        "",
        f"checkpoint: `{report['checkpoint']}`; step={report['checkpoint_step']}; contract={report['executor_contract_version']}",
        "",
        f"评测统一使用 v7 contract {report['executor_contract_version']}：outer 自回归，local 使用真实 NodeSpec 前缀并自回归生成模型拥有的 token，delta 按目标状态槽比较。",
        "",
        f"- cases={summary['cases']}; outer token accuracy={summary['outer_token_accuracy']:.4f}; local token accuracy={summary['local_token_accuracy']:.4f}",
        f"- target effect MAE={summary['target_effect_mae']:.4f}; unknown tokens={summary['unknown_output_tokens']}",
        "",
        "| case | card | game expected effect | protocol expected | predicted effect | target abs error | outer acc | local acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        target = row["target"]
        evaluation = row["evaluation"]
        local_nodes = evaluation["local"]
        scored = sum(node["metrics"]["scored_tokens"] for node in local_nodes)
        correct = sum(node["metrics"]["correct_tokens"] for node in local_nodes)
        protocol = target["expected_effect"]
        lines.append(
            f"| {row['name']} | {row['card_name_zh']} | {row['game_expected_effect']:.1f} | "
            f"{protocol if protocol is not None else '-'} | "
            f"{target['predicted_effect'] if target['predicted_effect'] is not None else '-'} | "
            f"{target['absolute_error'] if target['absolute_error'] is not None else '-'} | "
            f"{evaluation['outer']['metrics']['token_accuracy']:.3f} | "
            f"{correct / max(scored, 1):.3f} |"
        )
    lines.extend((
        "",
        "## Perfected Strike 计数诊断",
        "",
        "当前协议隐藏牌堆组成，只提供 draw/discard/exhaust 数量；因此没有 STRIKE_COUNT 环境读取时，3 张 Strike 的语义值不会进入 teacher target。",
        "",
        "| case | hidden Strike count | game damage | protocol base damage | teacher applied count | count read token |",
        "|---|---:|---:|---:|---|---|",
    ))
    for row in report["cases"]:
        diagnostic = row["perfected_strike_diagnostic"]
        if diagnostic is None:
            continue
        lines.append(
            f"| {row['name']} | {diagnostic['strike_count']} | {diagnostic['game_expected_damage']:.1f} | "
            f"{diagnostic['protocol_base_damage']} | {diagnostic['count_applied_by_teacher']} | "
            f"{diagnostic['count_read_token_present']} |"
        )
    lines.extend(("", "## Case notes", ""))
    for row in report["cases"]:
        lines.extend((f"### {row['name']}", "", row["note"], ""))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-outer-tokens", type=int, default=512)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if int(checkpoint.get("executor_contract_version", -1)) != V7_EXECUTOR_CONTRACT_VERSION:
        raise ValueError("checkpoint uses an incompatible executor contract")
    config = VectorRuleModelConfig.from_dict(checkpoint["config"])
    codec = V7TokenCodec.load(args.tokenizer)
    if config.decoder.token_vocab_size != len(codec.tokens):
        raise ValueError("checkpoint decoder vocabulary does not match tokenizer")
    from rl.rule_vector_model import VectorRuleWorldModel

    model = VectorRuleWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    cases = build_cases(config.state)
    results = []
    max_outer_tokens = min(args.max_outer_tokens, config.decoder.max_tokens - 1)
    for index, (spec, case) in enumerate(cases, 1):
        print(f"[simple-card-v7] case={index}/{len(cases)} {spec.name}", flush=True)
        evaluation = evaluate_case(
            model, case, codec, config, device, max_outer_tokens,
            interactive_local=True,
        )
        results.append(_case_row(spec, case, evaluation, config))
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "schema_version": int(checkpoint.get("schema_version", -1)),
        "executor_contract_version": int(checkpoint.get("executor_contract_version", -1)),
        "device": str(device),
        "evaluation_protocol": {
            "outer": "free greedy autoregressive generation",
            "local": "ground-truth NodeSpec prefix plus contract-10 interactive register injection",
            "delta": "active state-slot real-value comparison",
        },
        "cases": results,
    }
    report["summary"] = _summary(results)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    summary = report["summary"]
    print(
        f"[simple-card-v7] step={report['checkpoint_step']} "
        f"outer_acc={summary['outer_token_accuracy']:.6f} "
        f"local_acc={summary['local_token_accuracy']:.6f} "
        f"target_effect_mae={summary['target_effect_mae']:.6f} "
        f"unknown={summary['unknown_output_tokens']}",
        flush=True,
    )
    print(f"json={args.json_output}\nmarkdown={args.markdown_output}")


if __name__ == "__main__":
    main()
