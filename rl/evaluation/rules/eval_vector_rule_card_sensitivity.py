"""Evaluate simple Ironclad cards against the v6-wide checkpoint.

The checkpoint beside ``.tmp-remote-v6-wide`` is schema-v7, but it uses
executor contract 5.  The current producer has moved on to contract 7, so
this evaluator contains a small contract-5 local-sequence adapter.  It keeps
the current effect tree and state vectors, while reproducing the operation
names and read/response ordering used by the checkpoint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch

from env.core.card_language import TOKEN_INDEX as CARD_RULE_INDEX, card_program
from env.core.ironclad_cards import IRONCLAD_CARDS
from rl.evaluation.rules.eval_vector_rule_v7_cases import (
    EvaluationCase,
    _fixed_v6_record,
    _world_slot_names,
    evaluate_case,
)
import rl.rules.data.make_vector_rule_v7_data as v7_generator
import rl.rules.vector_rule_v7_data as v7_data
from rl.rules.data.make_vector_rule_v7_data import compile_v7_record
from rl.evaluation.rules.render_rule_vector_examples import _state
from rl.rules.rule_vector_model import (
    StateVectorConfig,
    VectorRuleModelConfig,
    VectorRuleWorldModel,
    _calculation_atom,
    _calculation_number,
)


CONTRACT = 5


@dataclass(frozen=True)
class ExactCheckpointCodec:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(self.tokens)})
        object.__setattr__(self, "pad_id", self.token_to_id["<PAD>"])
        object.__setattr__(self, "unknown_token", "<UNK>")

    def encode(self, tokens: Sequence[str]) -> tuple[int, ...]:
        unknown = self.token_to_id["<UNK>"]
        return tuple(self.token_to_id.get(str(token), unknown) for token in tokens)

    def decode(self, ids: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.tokens[int(index)] for index in ids)


@dataclass(frozen=True)
class SensitivitySpec:
    name: str
    card_id: str
    state: Mapping[str, Any]
    description: str
    oracle_field: str
    tamper_from: int | None = None
    tamper_to: int | None = None
    tamper_field: str = "damage"


def _program(card_id: str, upgrade: int, *, old: int | None = None, new: int | None = None) -> tuple[int, ...]:
    tokens = list(card_program(card_id, bool(upgrade)))
    if old is not None:
        if new is None:
            raise ValueError("new numeric value is required")
        old_tokens = [f"DIGIT_{digit}" for digit in str(int(old))]
        new_tokens = [f"DIGIT_{digit}" for digit in str(int(new))]
        found = None
        for index in range(len(tokens) - len(old_tokens) + 1):
            if tokens[index:index + len(old_tokens)] != old_tokens:
                continue
            if index == 0 or tokens[index - 1] not in {"NUM_BEGIN", "NUM_LSD_BEGIN"}:
                continue
            if index + len(old_tokens) >= len(tokens) or tokens[index + len(old_tokens)] != "NUM_END":
                continue
            found = index
            break
        if found is None:
            raise ValueError(f"numeric literal {old} not found in {card_id}")
        tokens[found:found + len(old_tokens)] = new_tokens
    return tuple(CARD_RULE_INDEX[token] for token in tokens)


def _num(value: int | float) -> list[str]:
    return list(_calculation_number(value))


def _append_read(tokens: list[str], mask: list[bool], source: str, binding: str, value: int | float) -> None:
    request = [
        "<READ_REQUEST_BEGIN>", "READ_SOURCE", f"VALUE_{_calculation_atom(source)}",
        "READ_BIND", f"VALUE_{_calculation_atom(binding)}", "<READ_REQUEST_END>",
    ]
    response = [
        "<READ_RESPONSE_BEGIN>", "READ_BIND", f"VALUE_{_calculation_atom(binding)}",
        "READ_VALUE", *_num(value), "<READ_RESPONSE_END>",
    ]
    tokens.extend(request); mask.extend([True] * len(request))
    tokens.extend(response); mask.extend([False] * len(response))


def _append_operation(tokens: list[str], kind: str, fields: Sequence[tuple[str, str]], numeric: Mapping[str, int | float] | None = None) -> None:
    numeric = numeric or {}
    tokens.extend(["CALCULATION_BEGIN", f"CALCULATION_{kind}"])
    for key, value in fields:
        tokens.extend([f"CALC_FIELD_{_calculation_atom(key)}", f"VALUE_{_calculation_atom(value)}"])
        if key in numeric:
            tokens.extend(["CALC_VALUE", *_num(numeric[key])])
    tokens.append("CALCULATION_END")


def _apply_row(node: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next((row for row in node.get("calculations", ()) if row.get("type") == "APPLY_DELTA"), None)


def _header_tokens(node: Mapping[str, Any]) -> list[str]:
    tokens = ["<REGION_BEGIN>", "REGION_KIND", f"REGION_{node['kind']}"]
    if node.get("entity_ref") is not None:
        tokens.extend(["ENTITY_REF", str(node["entity_ref"])])
    for key, value in node.get("header_params", {}).items():
        tokens.append(str(key))
        if isinstance(value, bool):
            tokens.append(f"BOOL_{int(value)}")
        elif isinstance(value, (int, float)):
            tokens.extend(_num(value))
        else:
            tokens.append(f"VALUE_{_calculation_atom(value)}")
    return tokens


def _old_local_sequence(node: Mapping[str, Any]) -> tuple[list[str], list[bool]]:
    """Serialize the contract-5 body used by the checkpoint.

    Contract 5 exposes named read requests and masks only their exact response
    values.  Arithmetic operation names are typed by the node's purpose.
    """
    tokens = _header_tokens(node) + ["<NODE_EXECUTE>", "<CALCULATIONS_BEGIN>"]
    mask = [False] * len(tokens)
    kind = str(node["kind"])
    params = node.get("header_params", {})

    if kind in {"CARD_COMMIT", "CARD_PLAY"}:
        _append_operation(tokens, "EXECUTE_CHILDREN", [("scope", "node.children")])
    elif kind == "ATTACK":
        _append_operation(tokens, "EXECUTE_ATTACK_CHILDREN", [("targets", str(node.get("entity_ref"))), ("scope", "node.children")])
    elif kind == "COMPLETE":
        _append_operation(tokens, "COMPLETE_ACTION", [])
    elif kind == "DAMAGE":
        _old_damage(node, tokens, mask)
    else:
        row = _apply_row(node)
        if row is None:
            # Zero-cost cards still have a CARD_PAYMENT region, but there is
            # no state delta to calculate.  This is the contract-5 form used
            # by the checkpoint for free cards.
            if kind == "CARD_PAYMENT":
                tokens.extend(["<CALCULATIONS_END>"])
                mask.extend([True])
                if node.get("delta") is not None:
                    tokens.append("DELTA_SLOT"); mask.append(True)
                tokens.append("<REGION_END>"); mask.append(True)
                condition_length = tokens.index("<NODE_EXECUTE>") + 1
                final_mask = [True] * len(tokens)
                for index in range(condition_length):
                    final_mask[index] = False
                return tokens, final_mask
            raise ValueError(f"unsupported contract-5 node without delta row: {kind}")
        field = str(row["field"])
        before = "reg.field_before.0"
        after = "reg.field_after.0"
        _append_read(tokens, mask, field, before, row.get("left", 0))
        if kind in {"CARD_PAYMENT", "DRAW"}:
            operation = "SUBTRACT_FIELD"
        elif kind == "ENERGY":
            operation = "ADD_FIELD" if str(params.get("ENERGY_OP")) == "GAIN" else "SUBTRACT_FIELD"
        elif kind in {"CARD_DESTINATION", "STATE_CHANGE"}:
            operation = "ADD_FIELD"
        else:
            raise ValueError(f"unsupported contract-5 generic node: {kind}")
        right_key = {
            "CARD_PAYMENT": "PAYMENT_COST", "DRAW": "DRAW_COUNT",
            "ENERGY": "ENERGY_AMOUNT", "CARD_DESTINATION": "CARD_COUNT",
            "STATE_CHANGE": "CHANGE_AMOUNT",
        }[kind]
        _append_operation(tokens, operation, [
            ("field", field), ("left", before), ("right", f"header.{right_key}"),
            ("result", after),
        ], {"result": row["result"]})

    tokens.extend(["<CALCULATIONS_END>"])
    mask.extend([True])
    if node.get("delta") is not None:
        tokens.append("DELTA_SLOT"); mask.append(True)
    tokens.append("<REGION_END>"); mask.append(True)
    # Operation helpers append tokens directly; derive the final ownership
    # mask from the protocol markers so the mask always stays aligned.  Only
    # exact READ_RESPONSE spans are environment-owned after the immutable
    # header/NodeSpec prefix.
    condition_length = tokens.index("<NODE_EXECUTE>") + 1
    final_mask = [True] * len(tokens)
    for index in range(condition_length):
        final_mask[index] = False
    response = False
    for index, token in enumerate(tokens):
        if token == "<READ_RESPONSE_BEGIN>":
            response = True
        if response:
            final_mask[index] = False
        if token == "<READ_RESPONSE_END>":
            response = False
    return tokens, final_mask


def _old_damage(node: Mapping[str, Any], tokens: list[str], mask: list[bool]) -> None:
    params = node.get("header_params", {})
    calculations = list(node.get("calculations", ()))
    by_type = {str(row.get("type")): row for row in calculations}
    damage_ref = "header.BASE_DAMAGE"

    strength = by_type.get("READ_STRENGTH")
    if strength is not None and int(strength.get("value", 0)) != 0:
        add = by_type["ADD_STRENGTH"]
        _append_operation(tokens, "CHECK_PRESENCE", [("field", "player.buff.Strength"), ("result", "reg_has_strength")])
        _append_read(tokens, mask, "player.buff.Strength", "reg.strength", strength["value"])
        _append_operation(tokens, "ADD_STRENGTH", [("left", damage_ref), ("right", "reg.strength"), ("result", "reg.damage_after_strength")], {"result": add["result"]})
        damage_ref = "reg.damage_after_strength"

    multiplier = by_type.get("READ_DAMAGE_MULTIPLIER")
    if multiplier is not None:
        row = by_type["MULTIPLY_DAMAGE"]
        _append_read(tokens, mask, str(multiplier.get("source", "parent.relic_trigger.damage_multiplier")), "reg.damage_multiplier", multiplier["value"])
        _append_operation(tokens, "MULTIPLY_DAMAGE", [("left", damage_ref), ("right", "reg.damage_multiplier"), ("result", "reg.damage_after_multiplier")], {"result": row["result"]})
        damage_ref = "reg.damage_after_multiplier"

    vulnerable = by_type.get("READ_VULNERABLE")
    if vulnerable is not None and int(vulnerable.get("value", 0)) != 0:
        multiplied = by_type["MULTIPLY_VULNERABLE"]
        divided = by_type["FLOOR_DIVIDE_VULNERABLE"]
        entity = str(node.get("runtime_entity_ref") or node.get("entity_ref"))
        slot = entity.rsplit("_", 1)[-1] if "ENTITY_ENEMY_SLOT_" in entity else "0"
        field = f"enemy[{slot}].buff.Vulnerable"
        _append_operation(tokens, "CHECK_PRESENCE", [("field", field), ("result", "reg_has_vulnerable")])
        _append_operation(tokens, "MULTIPLY_VULNERABLE", [("left", damage_ref), ("result", "reg.damage_vulnerable_numerator")], {"result": multiplied["result"]})
        _append_operation(tokens, "FLOOR_DIVIDE_VULNERABLE", [("left", "reg.damage_vulnerable_numerator"), ("result", "reg.damage_after_vulnerable")], {"result": divided["result"]})
        damage_ref = "reg.damage_after_vulnerable"

    block = by_type.get("READ_TARGET_BLOCK")
    if block is not None and int(block.get("value", 0)) > 0:
        entity = str(node.get("runtime_entity_ref") or node.get("entity_ref"))
        slot = entity.rsplit("_", 1)[-1] if "ENTITY_ENEMY_SLOT_" in entity else "0"
        block_field = f"enemy[{slot}].block"
        _append_read(tokens, mask, block_field, "reg.target_block", block["value"])
        absorbed = by_type["BLOCK_ABSORB"]
        subtract_block = by_type["SUBTRACT_BLOCK"]
        subtract_damage = by_type["SUBTRACT_BLOCK_DAMAGE"]
        _append_operation(tokens, "CHECK_PRESENCE", [("field", f"enemy[{slot}].block_present"), ("result", "reg_has_block")])
        _append_operation(tokens, "BLOCK_ABSORB", [("damage", damage_ref), ("block", "reg.target_block"), ("absorbed", "reg.absorbed_damage")], {"result": absorbed["absorbed"]})
        _append_operation(tokens, "SUBTRACT_BLOCK", [("left", "reg.target_block"), ("right", "reg.absorbed_damage"), ("result", "reg.target_block_after")], {"result": subtract_block["result"]})
        _append_operation(tokens, "SUBTRACT_BLOCK_DAMAGE", [("left", damage_ref), ("right", "reg.absorbed_damage"), ("result", "reg.unblocked_damage")], {"result": subtract_damage["result"]})
        damage_ref = "reg.unblocked_damage"

    hp = by_type.get("READ_TARGET_HP")
    if hp is not None:
        entity = str(node.get("runtime_entity_ref") or node.get("entity_ref"))
        slot = entity.rsplit("_", 1)[-1] if "ENTITY_ENEMY_SLOT_" in entity else "0"
        _append_read(tokens, mask, f"enemy[{slot}].hp", "reg.target_hp", hp["value"])
        subtract = by_type["SUBTRACT_HP"]
        _append_operation(tokens, "SUBTRACT_HP", [("left", "reg.target_hp"), ("right", damage_ref), ("result", "reg.target_hp_after")], {"result": subtract["result"]})


def _adapt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(record)
    result["executor_contract_version"] = CONTRACT
    rows = []
    for row in result["node_local_sequences"]:
        node = _node_at_path(result["effect_tree"], str(row["path"]))
        tokens, mask = _old_local_sequence(node)
        rows.append({
            "path": row["path"], "kind": row["kind"], "tokens": tokens,
            "model_mask": mask, "length": len(tokens),
            "condition_length": tokens.index("<NODE_EXECUTE>") + 1,
            "target_length": sum(mask),
            "register_context_length": 0, "copy_target_length": 0,
            "calculation_length": len(tokens) - tokens.index("<CALCULATIONS_BEGIN>"),
        })
    result["node_local_sequences"] = rows
    result["lengths"] = dict(result.get("lengths", {}))
    result["lengths"]["node_local_length_max"] = max((len(row["tokens"]) for row in rows), default=0)
    result["lengths"]["register_context_length_max"] = 0
    return result


def _node_at_path(root: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    node: Mapping[str, Any] = root
    if path != "root":
        for part in path.split("/")[1:]:
            node = node["children"][int(part)]
    return node


def _patched_record(spec: SensitivitySpec, config: StateVectorConfig) -> dict[str, Any]:
    original = IRONCLAD_CARDS[spec.card_id]
    patched = original
    if spec.tamper_from is not None:
        if spec.tamper_field == "damage":
            patched = replace(original, damage=(spec.tamper_to, spec.tamper_to))
        elif spec.tamper_field == "block":
            patched = replace(original, block=(spec.tamper_to, spec.tamper_to))
        else:
            raise ValueError(f"unsupported tamper field: {spec.tamper_field}")
        IRONCLAD_CARDS[spec.card_id] = patched
    try:
        hand = ((spec.card_id, 0),)
        raw, _canonical = _fixed_v6_record(
            index=0, state=spec.state, action_id=spec.card_id,
            action_upgrade=0, hand=hand, config=config, seed=0,
        )
        record = compile_v7_record(raw, config)
        program = _program(spec.card_id, 0, old=spec.tamper_from, new=spec.tamper_to)
        record["action"]["card_rule"] = list(program)
        record["hand"][0]["card_rule"] = list(program)
        return _adapt_record(record)
    finally:
        if spec.tamper_from is not None:
            IRONCLAD_CARDS[spec.card_id] = original


def _build_specs() -> tuple[SensitivitySpec, ...]:
    def combat(
        *, energy: int = 5, enemy_block: int = 0,
        enemy_powers: Mapping[str, int] | None = None,
        player_powers: Mapping[str, int] | None = None,
        player_block: int = 0, draw: int = 5, discard: int = 0,
    ) -> dict[str, Any]:
        state = _state(
            energy=energy, cards_played=0, attacks_played=0, sundial=0,
            enemy_hp=100, enemy_block=enemy_block, draw=draw, discard=discard,
        )
        state["combat"]["player"]["block"] = player_block
        state["combat"]["player"]["powers"] = dict(player_powers or {})
        state["combat"]["monsters"][0]["powers"] = dict(enemy_powers or {})
        return state

    specs: list[SensitivitySpec] = []

    def add(card: str, label: str, state: dict[str, Any], description: str,
            field: str, old: int | None = None, new: int | None = None,
            tamper_field: str = "damage") -> None:
        specs.append(SensitivitySpec(
            label, card, state, description, field, old, new, tamper_field,
        ))

    # The original five damage/card-vector checks.
    add("Strike_R", "打击-基础", combat(draw=0), "无修正", "enemy[0].hp")
    add("Strike_R", "打击-敌人易伤", combat(enemy_powers={"Vulnerable": 2}, draw=0), "敌人有2层易伤", "enemy[0].hp")
    add("Strike_R", "打击-敌人格挡", combat(enemy_block=4, draw=0), "敌人有4点格挡", "enemy[0].hp")
    add("Strike_R", "打击-易伤加格挡", combat(enemy_block=4, enemy_powers={"Vulnerable": 2}, draw=0), "敌人有2层易伤和4点格挡", "enemy[0].hp")
    add("Strike_R", "打击-篡改6到12", combat(draw=0), "规则基础伤害从6改为12", "enemy[0].hp", 6, 12)
    add("Perfected Strike", "完美打击-基础", combat(draw=0), "牌堆不含Strike，基础伤害6", "enemy[0].hp")
    add("Perfected Strike", "完美打击-敌人易伤", combat(enemy_powers={"Vulnerable": 2}, draw=0), "牌堆不含Strike，敌人有2层易伤", "enemy[0].hp")
    add("Perfected Strike", "完美打击-篡改6到12", combat(draw=0), "规则基础伤害从6改为12", "enemy[0].hp", 6, 12)
    add("Impervious", "岿然不动-基础", combat(draw=0), "获得30点格挡", "player.block", 30, 30, "block")
    # The expected protocol remains 30 because the current oracle does not
    # implement Frail; the game-semantic value is documented separately.
    add("Impervious", "岿然不动-玩家脆弱", combat(player_powers={"Frail": 2}, draw=0), "玩家有脆弱，当前协议生成器仍按30计算", "player.block", 30, 30, "block")
    add("Impervious", "岿然不动-篡改30到60", combat(draw=0), "规则格挡从30改为60", "player.block", 30, 60, "block")

    # Additional cards with a single deterministic target and no random/query
    # or recursive repeat.  Each group includes a state modifier where it is
    # meaningful, plus one rule-literal sensitivity probe when the producer
    # obtains that literal from CardInfo.
    add("Bash", "痛击-基础", combat(), "造成8伤害并施加易伤", "enemy[0].hp")
    add("Bash", "痛击-敌人格挡", combat(enemy_block=4), "敌人有4点格挡", "enemy[0].hp")
    add("Bash", "痛击-篡改8到16", combat(), "规则基础伤害从8改为16", "enemy[0].hp", 8, 16)
    add("Defend_R", "防御-基础", combat(), "获得5点格挡", "player.block", 5, 5, "block")
    add("Defend_R", "防御-玩家脆弱", combat(player_powers={"Frail": 2}), "玩家有2层脆弱", "player.block", 5, 5, "block")
    add("Defend_R", "防御-篡改5到10", combat(), "规则格挡从5改为10", "player.block", 5, 10, "block")
    add("Pommel Strike", "剑柄打击-基础", combat(draw=5), "造成9伤害并抽1张牌", "enemy[0].hp")
    add("Pommel Strike", "剑柄打击-敌人易伤", combat(draw=5, enemy_powers={"Vulnerable": 2}), "敌人有2层易伤", "enemy[0].hp")
    add("Pommel Strike", "剑柄打击-篡改9到18", combat(draw=5), "规则基础伤害从9改为18", "enemy[0].hp", 9, 18)
    add("Shrug It Off", "耸肩无视-基础", combat(draw=5), "获得8点格挡并抽1张牌", "player.block", 8, 8, "block")
    add("Shrug It Off", "耸肩无视-玩家脆弱", combat(draw=5, player_powers={"Frail": 2}), "玩家有2层脆弱", "player.block", 8, 8, "block")
    add("Shrug It Off", "耸肩无视-篡改8到16", combat(draw=5), "规则格挡从8改为16", "player.block", 8, 16, "block")
    add("Inflame", "燃烧-基础", combat(), "获得2点力量", "player.buff.Strength")
    add("Clothesline", "金刚臂-基础", combat(), "造成12伤害并施加虚弱", "enemy[0].hp")
    add("Clothesline", "金刚臂-敌人格挡", combat(enemy_block=4), "敌人有4点格挡", "enemy[0].hp")
    add("Clothesline", "金刚臂-篡改12到24", combat(), "规则基础伤害从12改为24", "enemy[0].hp", 12, 24)
    add("Iron Wave", "铁斩波-基础", combat(), "造成5伤害并获得5点格挡", "enemy[0].hp")
    add("Iron Wave", "铁斩波-敌人易伤", combat(enemy_powers={"Vulnerable": 2}), "敌人有2层易伤", "enemy[0].hp")
    add("Uppercut", "上勾拳-基础", combat(), "造成13伤害并施加两种状态", "enemy[0].hp")
    add("Uppercut", "上勾拳-敌人格挡", combat(enemy_block=4), "敌人有4点格挡", "enemy[0].hp")
    add("Ghostly Armor", "幽灵铠甲-基础", combat(), "获得10点格挡并消耗", "player.block", 10, 10, "block")
    add("Ghostly Armor", "幽灵铠甲-篡改10到20", combat(), "规则格挡从10改为20", "player.block", 10, 20, "block")
    add("Power Through", "硬撑-基础", combat(), "获得15点格挡并加入伤口", "player.block", 15, 15, "block")
    add("Power Through", "硬撑-篡改15到30", combat(), "规则格挡从15改为30", "player.block", 15, 30, "block")
    add("True Grit", "坚毅-基础", combat(), "获得7点格挡并消耗一张牌", "player.block", 7, 7, "block")
    add("Thunderclap", "闪电霹雳-基础", combat(), "造成4伤害并施加易伤", "enemy[0].hp")
    add("Thunderclap", "闪电霹雳-敌人格挡", combat(enemy_block=4), "敌人有4点格挡", "enemy[0].hp")
    add("Thunderclap", "闪电霹雳-篡改4到8", combat(), "规则基础伤害从4改为8", "enemy[0].hp", 4, 8)
    add("Seeing Red", "盛怒-基础", combat(), "获得2点能量并消耗", "player.energy")
    add("Burning Pact", "燃烧契约-基础", combat(draw=5), "消耗1张牌并抽2张牌", "piles.draw_count")
    add("Burning Pact", "燃烧契约-篡改2到4", combat(draw=5), "规则抽牌数从2改为4", "piles.draw_count", 2, 4)
    return tuple(specs)


def _target_delta(result: Mapping[str, Any], kind: str) -> tuple[float | None, float | None]:
    for node in result["local"]:
        if node["kind"] != kind:
            continue
        for row in node["delta_slots"]:
            active = [slot for slot in row.get("active_slots", ()) if slot["name"].endswith(".hp") or slot["name"] == "player.block"]
            if active:
                return float(active[0]["expected"]), float(active[0]["predicted"])
    return None, None


def _target_metrics(result: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    """Find the requested state slot instead of assuming every card deals damage.

    Several Ironclad cards have a useful non-damage target (block, Strength,
    energy, or pile counts).  Matching the declared field keeps the report from
    silently selecting an unrelated DAMAGE node in a compound effect tree.
    """
    for node in result["local"]:
        for row in node["delta_slots"]:
            active = [slot for slot in row.get("active_slots", ()) if slot["name"] == field_name]
            if active:
                slot = active[0]
                return {
                    "field": slot["name"],
                    "expected_normalized": float(slot["expected"]),
                    "predicted_normalized": float(slot["predicted"]),
                    "absolute_error_normalized": float(slot["absolute_error"]),
                }
    return {"field": None, "expected_normalized": None,
            "predicted_normalized": None, "absolute_error_normalized": None}


def _to_real(value: float | None, field: str | None, config: StateVectorConfig) -> float | None:
    if value is None:
        return None
    if field == "player.block" or (field is not None and field.endswith(".block")):
        return value * config.block_scale
    if field is not None and field.endswith(".hp"):
        return value * config.hp_scale
    if field == "player.energy":
        return value * config.energy_scale
    if field is not None and field.startswith("piles."):
        return value * config.pile_scale
    return value


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# v6 wide 单牌数值与状态敏感性测试",
        "",
        f"checkpoint: `{report['checkpoint']}`，step={report['checkpoint_step']}，executor contract={report['executor_contract_version']}",
        "",
        "说明：本次使用 checkpoint 自带的 919-token 词表，并将当前 effect-tree 记录适配为训练时的 contract 5 local 语法。之前按旧 v6 平铺格式比较的报告无效。",
        "",
        "| 场景 | 卡牌 | 协议期望值 | 模型预测值 | local token 准确率 | active delta MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        target = row["target"]
        local = row["evaluation"]["local"]
        scored = sum(x["metrics"]["scored_tokens"] for x in local)
        correct = sum(x["metrics"]["correct_tokens"] for x in local)
        deltas = [d for x in local for d in x["delta_slots"]]
        active = sum(d["active_slot_count"] for d in deltas)
        err = sum(d["active_slot_mae"] * d["active_slot_count"] for d in deltas)
        expected = target["expected_real"]
        predicted = target["predicted_real"]
        expected_text = f"{expected:.2f}" if isinstance(expected, (int, float)) else "-"
        predicted_text = f"{predicted:.2f}" if isinstance(predicted, (int, float)) else "-"
        lines.append(f"| {row['name']} | {row['card_id']} | {expected_text} | {predicted_text} | {100*correct/max(scored,1):.1f}% | {err/max(active,1):.6f} |")
    lines.extend(["", "## 场景详情", ""])
    for row in report["cases"]:
        evaluation = row["evaluation"]
        target = row["target"]
        lines.extend([
            f"### {row['name']}", "",
            f"- {row['description']}。输入状态：{row['input']}。",
            f"- 卡牌规则数值篡改：{row['tamper_from']} -> {row['tamper_to']}。",
            f"- outer token accuracy={evaluation['outer']['metrics']['token_accuracy']:.4f}，exact={evaluation['outer']['metrics']['exact_match']}。",
            f"- 目标 delta：期望={target['expected_real']}，预测={target['predicted_real']}（归一化值分别为 {target['expected_normalized']}、{target['predicted_normalized']}）。",
        ])
    lines.extend(["", "## 口径注意", "", "岿然不动+脆弱的游戏语义应为 floor(30*0.75)=22 点格挡，但当前 `_effect_record` 没有实现 Frail 修正，因此本报告的协议期望仍是30；这不是模型误差。"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-outer-tokens", type=int, default=512)
    args = parser.parse_args()

    # Import-time constants are intentionally patched only for this compatibility
    # evaluator.  The current training producer remains contract 7.
    v7_data.V7_EXECUTOR_CONTRACT_VERSION = CONTRACT
    v7_generator.V7_EXECUTOR_CONTRACT_VERSION = CONTRACT

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if int(checkpoint.get("executor_contract_version", -1)) != CONTRACT:
        raise ValueError("checkpoint is not contract 5")
    config = VectorRuleModelConfig.from_dict(checkpoint["config"])
    payload = json.loads((args.checkpoint.parent / "tokenizer.json").read_text(encoding="utf-8"))
    codec = ExactCheckpointCodec(tuple(str(token) for token in payload["tokens"]))
    if config.decoder.token_vocab_size != len(codec.tokens):
        raise ValueError("checkpoint decoder vocabulary does not match tokenizer")
    model = VectorRuleWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    for index, spec in enumerate(_build_specs(), 1):
        record = _patched_record(spec, config.state)
        case = EvaluationCase(
            name=spec.name, group="card_sensitivity", canonical={}, record=record,
            action_id=spec.card_id, action_upgrade=0, hand=((spec.card_id, 0),),
            relics=(), note=spec.description,
        )
        # canonical is used only for the human-readable case description. The
        # evaluator itself consumes the anonymized record and effect tree.
        case = replace(case, canonical=spec.state)
        evaluation = evaluate_case(model, case, codec, config, device, min(args.max_outer_tokens, config.decoder.max_tokens - 1))
        target = _target_metrics(evaluation, spec.oracle_field)
        target["expected_real"] = _to_real(target["expected_normalized"], target["field"], config.state)
        target["predicted_real"] = _to_real(target["predicted_normalized"], target["field"], config.state)
        rows.append({
            "name": spec.name, "card_id": spec.card_id, "description": spec.description,
            "tamper_from": spec.tamper_from, "tamper_to": spec.tamper_to,
            "oracle_field": spec.oracle_field,
            "input": {"player_powers": spec.state["combat"]["player"].get("powers", {}), "enemy_powers": spec.state["combat"]["monsters"][0].get("powers", {}), "enemy_block": spec.state["combat"]["monsters"][0].get("block", 0)},
            "target": target,
            "evaluation": evaluation,
        })
        print(f"[{index}/{len(_build_specs())}] {spec.name}: outer={evaluation['outer']['metrics']['token_accuracy']:.3f}", flush=True)
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint.get("step"),
        "schema_version": checkpoint.get("schema_version"), "executor_contract_version": CONTRACT,
        "device": str(device), "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(f"report={args.output}")
    print(f"markdown={args.markdown}")


if __name__ == "__main__":
    main()
