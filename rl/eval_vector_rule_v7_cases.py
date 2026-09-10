"""Autoregressively evaluate the current v7 executor contract on fixed and random cases.

Outer headers are decoded freely.  Each local executor invocation receives its
ground-truth NodeSpec prefix, keeps model mistakes in subsequent context, and
uses environment-owned inline constants for requested runtime values. Random
target/card queries still pause for an environment observation.
The command writes a complete JSON artifact and a Chinese Markdown report.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch

from env.core.card_language import STATUS_CARDS
from env.core.ironclad_cards import IRONCLAD_CARDS
from env.core.relics import RELIC_NAME_ZH
from rl.make_vector_rule_training_data import (
    _effect_record,
    _program,
    _relic_program,
    build_record_with_canonical,
)
from rl.make_vector_rule_v7_data import _action_name, compile_v7_record
from rl.render_rule_vector_examples import _state
from rl.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_VOCAB,
    StateVectorConfig,
    TokenOwner,
    VectorValueBinder,
    VectorValueBinding,
    VectorRuleModelConfig,
    VectorRuleWorldModel,
    _parse_calculation_number,
    vectorize_state,
)
from rl.train_rule_vectors_v7 import _model_inputs, _move_batch
from rl.vector_rule_data import VECTOR_DATA_SCHEMA_VERSION
from rl.vector_rule_v7_data import (
    V7_EXECUTOR_CONTRACT_VERSION,
    V7TokenCodec,
    VectorRuleV7Example,
    collate_vector_rule_v7_batch,
    examples_from_v7_record,
)


DEFAULT_RANDOM_SEED = 20260831

POWER_NAME_ZH = {
    "Strength": "力量",
    "Dexterity": "敏捷",
    "Vulnerable": "易伤",
    "Weakened": "虚弱",
    "Frail": "脆弱",
    "Intangible": "无实体",
    "Artifact": "人工制品",
    "Barricade": "壁垒",
    "Corruption": "腐化",
    "Double Tap": "双发",
    "Brutality": "残暴",
    "Combust": "燃烧",
    "Dark Embrace": "黑暗之拥",
    "Demon Form": "恶魔形态",
    "Evolve": "进化",
    "Feel No Pain": "无惧疼痛",
    "Fire Breathing": "火焰吐息",
    "Flame Barrier": "火焰屏障",
    "Juggernaut": "勇猛",
    "Metallicize": "金属化",
    "Plated Armor": "多层护甲",
    "Poison": "中毒",
    "Rage": "狂怒",
    "Rupture": "撕裂",
    "Vigor": "活力",
}

COUNTER_NAME_ZH = {
    "happy_flower": "开心小花",
    "incense_burner": "香炉",
    "ink_bottle": "墨水瓶",
    "inserter": "机械臂",
    "nunchaku": "双节棍",
    "pen_nib": "钢笔尖",
    "sundial": "日晷",
    "necronomicon_used": "死灵之书本回合已触发",
    "cards_played": "已打出卡牌数",
    "attacks_played": "已打出攻击牌数",
    "skills_played": "已打出技能牌数",
    "orange_attack": "橙色药丸攻击标志",
    "orange_skill": "橙色药丸技能标志",
    "orange_power": "橙色药丸能力标志",
    "combust_hp_loss": "燃烧失血计数",
    "deva_form_energy_per_turn": "天人形态每回合能量",
    "echo_form_cards_doubled": "回响形态复制计数",
    "panache_counter": "华丽收场计数",
    "cards_discarded": "已弃牌数",
    "times_damaged": "受伤次数",
    "bomb_1": "炸弹倒计时1",
    "bomb_2": "炸弹倒计时2",
    "bomb_3": "炸弹倒计时3",
    "last_attack_unblocked_damage": "上次攻击未格挡伤害",
}

ENCOUNTER_KIND_ZH = {"normal": "普通", "elite": "精英", "boss": "首领"}


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    group: str
    canonical: Mapping[str, Any]
    record: Mapping[str, Any]
    action_id: str
    action_upgrade: int
    hand: tuple[tuple[str, int], ...]
    relics: tuple[tuple[str, float], ...]
    note: str = ""


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _vector_state_payload(state: Mapping[str, Any], config: StateVectorConfig) -> dict[str, Any]:
    encoded = vectorize_state(state, config)
    return {
        "player": [float(value) for value in encoded.player.tolist()],
        "enemies": [
            [float(value) for value in row]
            for row in encoded.enemies.tolist()
        ],
        "piles": [float(value) for value in encoded.piles.tolist()],
        "relic_counters": [
            float(value) for value in encoded.relic_counters.tolist()
        ],
    }


def _fixed_v6_record(
    *,
    index: int,
    state: Mapping[str, Any],
    action_id: str,
    action_upgrade: int,
    hand: Sequence[tuple[str, int]],
    config: StateVectorConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = deepcopy(state)
    canonical["combat"]["hand"] = [
        {"id": card_id, "up": int(upgrade)} for card_id, upgrade in hand
    ]
    action = {"id": action_id, "up": int(action_upgrade)}
    events = _effect_record(deepcopy(canonical), action, config, random.Random(seed))
    global_state = canonical.get("global", {})
    relic_ids = [str(value) for value in global_state.get("relics", ())]
    raw_counters = list(global_state.get("relic_counters", ()))
    counters = [
        float(raw_counters[i]) if i < len(raw_counters) else 0.0
        for i in range(len(relic_ids))
    ]
    record = {
        "schema_version": VECTOR_DATA_SCHEMA_VERSION,
        "state": _vector_state_payload(canonical, config),
        "action": {
            "card_rule": _program(action_id, action_upgrade),
            "upgrade": int(action_upgrade),
        },
        "hand": [
            {"card_rule": _program(card_id, upgrade), "upgrade": int(upgrade)}
            for card_id, upgrade in hand
        ],
        "relics": [
            {
                "rule": _relic_program(relic_id),
                "counter": counter / config.counter_scale,
            }
            for relic_id, counter in zip(relic_ids, counters)
        ],
        "events": events,
        "metadata": {
            "sample_index": index,
            "floor": int(global_state.get("floor", 0)),
            "act": int(global_state.get("act", 0)),
            "encounter_kind": str(global_state.get("encounter_kind", "fixed")),
            "encounter": str(global_state.get("encounter", "FixedExample")),
            "relics": relic_ids,
            "hand_size": len(hand),
        },
    }
    return record, canonical


def _relic_pairs(canonical: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    global_state = canonical.get("global", {})
    relic_ids = [str(value) for value in global_state.get("relics", ())]
    counters = list(global_state.get("relic_counters", ()))
    return tuple(
        (
            relic_id,
            float(counters[index]) if index < len(counters) else 0.0,
        )
        for index, relic_id in enumerate(relic_ids)
    )


def build_cases(
    config: StateVectorConfig,
    *,
    random_seed: int,
    random_cases: int,
) -> tuple[EvaluationCase, ...]:
    fly_hand = (("Dropkick", 0), ("Strike_R", 0), ("Defend_R", 0))
    fly_state = _state(
        energy=3,
        cards_played=0,
        attacks_played=0,
        sundial=0,
        enemy_hp=50,
        enemy_block=0,
        draw=5,
        discard=1,
        enemy_powers={"Vulnerable": 2},
    )
    fly_v6, fly_canonical = _fixed_v6_record(
        index=-2,
        state=fly_state,
        action_id="Dropkick",
        action_upgrade=0,
        hand=fly_hand,
        config=config,
        seed=0,
    )

    pommel_hand = (("Pommel Strike", 0), ("Strike_R", 0), ("Defend_R", 0))
    pommel_state = _state(
        energy=3,
        cards_played=0,
        attacks_played=0,
        sundial=2,
        enemy_hp=60,
        enemy_block=0,
        draw=0,
        discard=3,
        relics=("Sundial",),
        relic_counters=(2,),
    )
    pommel_v6, pommel_canonical = _fixed_v6_record(
        index=-1,
        state=pommel_state,
        action_id="Pommel Strike",
        action_upgrade=0,
        hand=pommel_hand,
        config=config,
        seed=0,
    )

    cases: list[EvaluationCase] = [
        EvaluationCase(
            name="固定例1：飞身踢，敌人有2层易伤",
            group="fixed",
            canonical=fly_canonical,
            record=compile_v7_record(fly_v6, config),
            action_id="Dropkick",
            action_upgrade=0,
            hand=fly_hand,
            relics=(),
            note="沿用既有固定回归例；敌人没有格挡，易伤参与伤害计算。",
        ),
        EvaluationCase(
            name="固定例2：剑柄打击，洗牌并触发日晷",
            group="fixed",
            canonical=pommel_canonical,
            record=compile_v7_record(pommel_v6, config),
            action_id="Pommel Strike",
            action_upgrade=0,
            hand=pommel_hand,
            relics=(("Sundial", 2.0),),
            note="抽牌堆为0、弃牌堆为3；日晷计数器为2。",
        ),
    ]

    rng = random.Random(random_seed)
    for index in range(random_cases):
        v6, canonical = build_record_with_canonical(index, config, rng)
        action_id = _action_name(v6)
        action_upgrade = int(v6["action"].get("upgrade", 0))
        hand = tuple(
            (str(card["id"]), int(card.get("up", 0)))
            for card in canonical["combat"].get("hand", ())
        )
        cases.append(EvaluationCase(
            name=f"随机例{index + 1}",
            group="random",
            canonical=canonical,
            record=compile_v7_record(v6, config),
            action_id=action_id,
            action_upgrade=action_upgrade,
            hand=hand,
            relics=_relic_pairs(canonical),
            note=f"由训练分布直接生成，随机种子={random_seed}，流内序号={index}；未筛选。",
        ))
    return tuple(cases)


def _token_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    free_length = positions is None
    scored_positions = (
        list(range(max(len(expected), len(predicted))))
        if free_length else [index for index in positions if index < len(expected)]
    )
    matches = sum(
        index < len(expected)
        and index < len(predicted)
        and expected[index] == predicted[index]
        for index in scored_positions
    )
    first = next(
        (
            index for index in scored_positions
            if index >= len(expected)
            or index >= len(predicted)
            or expected[index] != predicted[index]
        ),
        None,
    )
    return {
        "exact_match": first is None and (not free_length or len(expected) == len(predicted)),
        "expected_tokens": len(expected),
        "predicted_tokens": len(predicted),
        "scored_tokens": len(scored_positions),
        "correct_tokens": matches,
        "token_accuracy": matches / max(len(scored_positions), 1),
        "first_mismatch": first,
        "expected_at_mismatch": (
            expected[first] if first is not None and first < len(expected) else None
        ),
        "predicted_at_mismatch": (
            predicted[first] if first is not None and first < len(predicted) else None
        ),
    }


def _context(
    expected: Sequence[str], predicted: Sequence[str], index: int | None,
    *, radius: int = 4,
) -> dict[str, Any] | None:
    if index is None:
        return None
    begin = max(0, index - radius)
    end = min(max(len(expected), len(predicted)), index + radius + 1)
    return {
        "begin": begin,
        "end": end,
        "expected": list(expected[begin:end]),
        "predicted": list(predicted[begin:end]),
    }


def _prepare_model_inputs(
    model: VectorRuleWorldModel,
    example: VectorRuleV7Example,
    codec: V7TokenCodec,
    device: torch.device,
) -> dict[str, Any]:
    batch = collate_vector_rule_v7_batch([example], pad_id=codec.pad_id)
    batch = _move_batch(batch, device)
    inputs = _model_inputs(model, batch)
    inputs.pop("target_input_ids")
    return inputs


def _decode_outer(
    model: VectorRuleWorldModel,
    example: VectorRuleV7Example,
    codec: V7TokenCodec,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = _prepare_model_inputs(model, example, codec, device)
    with torch.inference_mode():
        generated, _deltas = model.generate(
            **inputs,
            output_begin_id=codec.token_to_id["<OUTPUT_BEGIN>"],
            output_end_id=codec.token_to_id["<OUTPUT_END>"],
            max_new_tokens=max_new_tokens,
        )
    expected = codec.decode(example.target_ids)
    predicted = codec.decode(generated[0].tolist())
    metrics = _token_metrics(expected, predicted)
    return {
        "metrics": metrics,
        "first_mismatch_context": _context(
            expected, predicted, metrics["first_mismatch"]
        ),
        "expected_tokens": list(expected),
        "predicted_tokens": list(predicted),
        "unknown_output_tokens": predicted.count(codec.unknown_token),
        "terminated": bool(predicted and predicted[-1] == "<OUTPUT_END>"),
    }


def _world_slot_names(config: StateVectorConfig) -> tuple[str, ...]:
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
        names.extend(f"enemy[{enemy}].{name}" for name in config.enemy_scalar_names)
        names.extend(
            f"enemy[{enemy}].buff.{name}"
            for name in config.monster_buff_names
        )
    names.extend(f"piles.{name}" for name in config.pile_names)
    if len(names) != config.world_dim:
        raise RuntimeError("world slot labels do not match state vector width")
    return tuple(names)


def _delta_metrics(
    expected: Sequence[float] | None,
    predicted: Sequence[float] | None,
    slot_names: Sequence[str],
) -> dict[str, Any] | None:
    if expected is None or predicted is None:
        return None
    if len(expected) != len(predicted) or len(expected) != len(slot_names):
        raise ValueError("delta widths do not match")
    errors = [abs(float(left) - float(right)) for left, right in zip(expected, predicted)]
    active = [index for index, value in enumerate(expected) if abs(float(value)) >= 1e-8]
    return {
        "all_vector_mae": sum(errors) / max(len(errors), 1),
        "active_slot_mae": (
            sum(errors[index] for index in active) / len(active) if active else 0.0
        ),
        "active_slot_count": len(active),
        "active_slots": [
            {
                "index": index,
                "name": slot_names[index],
                "expected": float(expected[index]),
                "predicted": float(predicted[index]),
                "absolute_error": errors[index],
            }
            for index in active
        ],
    }


def _decode_local(
    model: VectorRuleWorldModel,
    example: VectorRuleV7Example,
    codec: V7TokenCodec,
    device: torch.device,
    slot_names: Sequence[str],
) -> dict[str, Any]:
    inputs = _prepare_model_inputs(model, example, codec, device)
    with torch.inference_mode():
        memory, padding = model.encode_world(**inputs)
        expected_ids = list(example.target_ids)
        generated = torch.as_tensor(
            [expected_ids[:example.condition_length]],
            dtype=torch.long,
            device=device,
        )
        predicted_ids = list(expected_ids[:example.condition_length])
        predicted_deltas: dict[int, list[float]] = {}
        model_mask = example.target_model_mask
        for position in range(example.condition_length, len(expected_ids)):
            output = model.decoder(generated, memory, padding)
            if model_mask[position]:
                next_id = int(output.logits[0, -1].argmax().item())
            else:
                next_id = int(expected_ids[position])
            if example.target_delta[position] is not None:
                predicted_deltas[position] = [
                    float(value) for value in output.delta[0, -1].tolist()
                ]
            predicted_ids.append(next_id)
            generated = torch.cat((
                generated,
                torch.as_tensor([[next_id]], dtype=torch.long, device=device),
            ), dim=1)

    expected = codec.decode(expected_ids)
    predicted = codec.decode(predicted_ids)
    scored_positions = tuple(
        index for index in range(example.condition_length, len(expected_ids))
        if model_mask[index]
    )
    metrics = _token_metrics(expected, predicted, positions=scored_positions)
    delta_rows = []
    for position, expected_delta in enumerate(example.target_delta):
        if expected_delta is None:
            continue
        delta_rows.append({
            "position": position,
            **(_delta_metrics(
                expected_delta, predicted_deltas.get(position), slot_names
            ) or {}),
        })
    return {
        "scope_kind": example.scope_kind,
        "condition_length": example.condition_length,
        "metrics": metrics,
        "first_mismatch_context": _context(
            expected, predicted, metrics["first_mismatch"]
        ),
        "expected_tokens": list(expected),
        "predicted_tokens": list(predicted),
        "model_mask": list(model_mask),
        "environment_tokens_injected": sum(
            not model_mask[index]
            for index in range(example.condition_length, len(expected_ids))
        ),
        "unknown_output_tokens": sum(
            predicted[index] == codec.unknown_token for index in scored_positions
        ),
        "delta_slots": delta_rows,
    }


def _expected_value_bindings(tokens: Sequence[str]) -> tuple[VectorValueBinding, ...]:
    """Recover deterministic runtime reads from a teacher local trace."""

    bindings: list[VectorValueBinding] = []
    for index in range(len(tokens) - 5):
        if tokens[index:index + 2] != ["<READ_REQUEST_BEGIN>", "READ_SOURCE"]:
            continue
        source = tokens[index + 2]
        if tokens[index + 3] != "READ_BIND":
            raise ValueError("READ_SOURCE is not followed by READ_BIND")
        binding = tokens[index + 4]
        if tokens[index + 5] != "<READ_REQUEST_END>":
            raise ValueError("READ_BIND is not followed by READ_REQUEST_END")
        value = None
        for operand in range(index + 6, len(tokens) - 2):
            if tokens[operand] != binding or tokens[operand + 1] != "CONST_VALUE":
                continue
            parsed = _parse_calculation_number(tokens, operand + 2)
            if parsed is not None:
                value = parsed[0]
                break
        if value is None:
            raise ValueError(f"runtime binding has no inline value: {binding}")
        bindings.append(VectorValueBinding(
            request_id=f"r{len(bindings)}",
            source_token=source,
            binding_token=binding,
            value=value,
        ))
    return tuple(bindings)


def _decode_local_interactive(
    model: VectorRuleWorldModel,
    example: VectorRuleV7Example,
    codec: V7TokenCodec,
    device: torch.device,
    slot_names: Sequence[str],
) -> dict[str, Any]:
    """Decode with the real contract-10 register store and token injection."""

    expected_ids = list(example.target_ids)
    expected = list(codec.decode(expected_ids))
    binder = VectorValueBinder(_expected_value_bindings(expected))
    inputs = _prepare_model_inputs(model, example, codec, device)
    with torch.inference_mode():
        transcript, decoder_deltas = model.generate_interactive(
            **inputs,
            output_begin_id=codec.token_to_id["<OUTPUT_BEGIN>"],
            output_end_id=codec.token_to_id["<REGION_END>"],
            token_index=codec.token_to_id,
            value_binder=binder,
            prefix_ids=expected_ids[:example.condition_length],
            max_new_tokens=model.decoder.config.max_tokens - example.condition_length,
        )
    predicted = list(transcript.tokens)
    expected_model = [
        token for index, token in enumerate(expected)
        if index >= example.condition_length and example.target_model_mask[index]
    ]
    predicted_model = [
        token for token, owner in zip(transcript.tokens, transcript.owners)
        if owner == TokenOwner.MODEL
    ]
    metrics = _token_metrics(expected_model, predicted_model)
    predicted_delta_rows = transcript.outputs_for_model_token(
        decoder_deltas, "DELTA_SLOT"
    )
    expected_delta_rows = [
        (position, delta) for position, delta in enumerate(example.target_delta)
        if delta is not None
    ]
    delta_rows = []
    for ordinal, (position, expected_delta) in enumerate(expected_delta_rows):
        predicted_delta = (
            predicted_delta_rows[ordinal].tolist()
            if ordinal < len(predicted_delta_rows) else None
        )
        metrics_row = _delta_metrics(expected_delta, predicted_delta, slot_names)
        delta_rows.append({
            "position": position,
            "predicted_ordinal": ordinal if predicted_delta is not None else None,
            **(metrics_row or {}),
        })
    return {
        "scope_kind": example.scope_kind,
        "condition_length": example.condition_length,
        "metrics": metrics,
        "first_mismatch_context": _context(
            expected_model, predicted_model, metrics["first_mismatch"]
        ),
        "expected_tokens": expected,
        "predicted_tokens": predicted,
        "expected_model_tokens": expected_model,
        "predicted_model_tokens": predicted_model,
        "model_mask": list(example.target_model_mask),
        "environment_tokens_injected": sum(
            owner == TokenOwner.ENVIRONMENT
            for owner in transcript.owners[example.condition_length:]
        ),
        "unknown_output_tokens": predicted_model.count(codec.unknown_token),
        "delta_slots": delta_rows,
        "value_bindings_complete": binder.complete,
        "interactive_register_values": True,
    }


def evaluate_case(
    model: VectorRuleWorldModel,
    case: EvaluationCase,
    codec: V7TokenCodec,
    config: VectorRuleModelConfig,
    device: torch.device,
    max_outer_tokens: int,
    *,
    interactive_local: bool = False,
) -> dict[str, Any]:
    examples = examples_from_v7_record(
        case.record,
        codec,
        config.state,
        card_vocab_size=len(CARD_RULE_VOCAB),
        relic_vocab_size=len(RELIC_RULE_VOCAB),
        max_cards=10,
        max_relics=32,
        max_tokens=config.decoder.max_tokens,
    )
    rows = list(case.record["node_local_sequences"])
    if len(rows) != len(examples) - 1:
        raise RuntimeError("local record rows do not align with local examples")
    slot_names = _world_slot_names(config.state)
    outer = _decode_outer(
        model, examples[0], codec, device, max_outer_tokens
    )
    local = []
    for row, example in zip(rows, examples[1:]):
        result = (
            _decode_local_interactive(model, example, codec, device, slot_names)
            if interactive_local
            else _decode_local(model, example, codec, device, slot_names)
        )
        result["path"] = str(row["path"])
        result["kind"] = str(row["kind"])
        local.append(result)
    return {
        "name": case.name,
        "group": case.group,
        "input": _describe_case(case),
        "outer": outer,
        "local": local,
        "record_lengths": dict(case.record.get("lengths", {})),
    }


def _card_name(card_id: str) -> str:
    if card_id in IRONCLAD_CARDS:
        return IRONCLAD_CARDS[card_id].name_zh
    if card_id in STATUS_CARDS:
        return str(STATUS_CARDS[card_id]["name_zh"])
    return card_id


def _card_rule(card_id: str) -> str:
    if card_id in IRONCLAD_CARDS:
        return IRONCLAD_CARDS[card_id].effect_zh
    if card_id in STATUS_CARDS:
        return str(STATUS_CARDS[card_id]["effect_zh"])
    return "规则描述不可用"


def _format_card(card_id: str, upgrade: int) -> str:
    suffix = "+" if int(upgrade) else ""
    return f"{_card_name(card_id)}{suffix}（{card_id}{suffix}）"


def _format_mapping(
    values: Mapping[str, Any], names: Mapping[str, str] | None = None,
    *, include_zero: bool = False,
) -> str:
    names = names or {}
    rows = []
    for key, value in values.items():
        if not include_zero and not float(value or 0):
            continue
        rows.append(f"{names.get(str(key), str(key))}={value}")
    return "、".join(rows) if rows else "无"


def _describe_case(case: EvaluationCase) -> dict[str, Any]:
    canonical = case.canonical
    combat = canonical.get("combat", {})
    global_state = canonical.get("global", {})
    player = combat.get("player", {})
    floor = int(global_state.get("floor", 0))
    act = int(global_state.get("act", 0))
    kind = str(global_state.get("encounter_kind", "fixed"))
    encounter = str(global_state.get("encounter", "固定测试敌人"))
    hand = [_format_card(card_id, upgrade) for card_id, upgrade in case.hand]
    relics = [
        f"{RELIC_NAME_ZH.get(relic_id, relic_id)}（{relic_id}，计数器={counter:g}）"
        for relic_id, counter in case.relics
    ]
    enemies = []
    for index, enemy in enumerate(combat.get("monsters", ())):
        enemies.append({
            "slot": index,
            "id": str(enemy.get("id", f"enemy-{index}")),
            "hp": int(enemy.get("hp", 0)),
            "max_hp": int(enemy.get("max_hp", 0)),
            "block": int(enemy.get("block", 0)),
            "base_damage": int(enemy.get("base_damage", 0)),
            "adjusted_damage": int(enemy.get("adjusted_damage", 0)),
            "hits": int(enemy.get("hits", 0)),
            "buffs": _format_mapping(enemy.get("powers", {}), POWER_NAME_ZH),
        })
    return {
        "note": case.note,
        "floor": floor,
        "act": act,
        "encounter_kind": ENCOUNTER_KIND_ZH.get(kind, "固定回归"),
        "encounter": encounter,
        "turn": int(combat.get("turn", 0)),
        "action": _format_card(case.action_id, case.action_upgrade),
        "action_rule": _card_rule(case.action_id),
        "hand": hand,
        "relics": relics,
        "player": {
            "hp": int(player.get("hp", 0)),
            "max_hp": int(player.get("max_hp", 0)),
            "block": int(player.get("block", 0)),
            "energy": int(player.get("energy", 0)),
            "gold": int(player.get("gold", global_state.get("gold", 0))),
            "buffs": _format_mapping(player.get("powers", {}), POWER_NAME_ZH),
            "counters": _format_mapping(
                player.get("combat_counters", {}), COUNTER_NAME_ZH
            ),
        },
        "enemies": enemies,
        "piles": {
            "draw_count": len(combat.get("draw", ())),
            "discard_count": len(combat.get("discard", ())),
            "exhaust_count": len(combat.get("exhaust", ())),
            "composition_visible_to_model": False,
        },
    }


def _aggregate(
    results: Sequence[Mapping[str, Any]], config: StateVectorConfig,
) -> dict[str, Any]:
    outer_scored = sum(row["outer"]["metrics"]["scored_tokens"] for row in results)
    outer_correct = sum(row["outer"]["metrics"]["correct_tokens"] for row in results)
    local_rows = [node for row in results for node in row["local"]]
    local_scored = sum(node["metrics"]["scored_tokens"] for node in local_rows)
    local_correct = sum(node["metrics"]["correct_tokens"] for node in local_rows)
    delta_rows = [delta for node in local_rows for delta in node["delta_slots"]]
    active_errors = sum(
        delta["active_slot_mae"] * delta["active_slot_count"]
        for delta in delta_rows
    )
    active_count = sum(delta["active_slot_count"] for delta in delta_rows)
    world_width = len(_world_slot_names(config))
    return {
        "cases": len(results),
        "outer_exact_cases": sum(row["outer"]["metrics"]["exact_match"] for row in results),
        "outer_token_accuracy": outer_correct / max(outer_scored, 1),
        "outer_scored_tokens": outer_scored,
        "local_nodes": len(local_rows),
        "local_exact_nodes": sum(node["metrics"]["exact_match"] for node in local_rows),
        "local_token_accuracy": local_correct / max(local_scored, 1),
        "local_scored_tokens": local_scored,
        "delta_slots": len(delta_rows),
        "delta_active_slots": active_count,
        "delta_active_slot_mae": active_errors / max(active_count, 1),
        "delta_all_vector_mae": (
            sum(delta["all_vector_mae"] * world_width for delta in delta_rows)
            / max(len(delta_rows) * world_width, 1)
        ),
        "unknown_output_tokens": sum(
            row["outer"]["unknown_output_tokens"]
            + sum(node["unknown_output_tokens"] for node in row["local"])
            for row in results
        ),
    }


BEGIN_TOKENS = {
    "<OUTPUT_BEGIN>", "<TRANSITION_BEGIN>", "<EFFECT_TREE_BEGIN>",
    "<REGION_BEGIN>", "<CALCULATIONS_BEGIN>", "CALCULATION_BEGIN",
    "<READ_REQUEST_BEGIN>", "<QUERY_BEGIN>", "<OBS_BEGIN>",
}
END_TOKENS = {
    "<OUTPUT_END>", "<TRANSITION_END>", "<EFFECT_TREE_END>",
    "<REGION_END>", "<CALCULATIONS_END>", "CALCULATION_END",
    "<READ_REQUEST_END>", "<QUERY_END>", "<OBS_END>",
}


def _render_tokens(tokens: Sequence[str]) -> str:
    lines: list[str] = []
    depth = 0
    for token in tokens:
        if token in END_TOKENS:
            depth = max(0, depth - 1)
        lines.append("  " * depth + token)
        if token in BEGIN_TOKENS:
            depth += 1
    return "\n".join(lines)


def _fmt_rate(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _first_error_text(metrics: Mapping[str, Any]) -> str:
    if metrics["first_mismatch"] is None:
        return "无"
    return (
        f"位置{metrics['first_mismatch']}："
        f"期望 `{metrics['expected_at_mismatch']}`，"
        f"预测 `{metrics['predicted_at_mismatch']}`"
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# v7 step {report['checkpoint_step']} 固定与随机自回归评测",
        "",
        f"- 参数：`{report['checkpoint']}`",
        f"- checkpoint step：`{report['checkpoint_step']}`",
        f"- schema / executor contract：`{report['schema_version']} / {report['executor_contract_version']}`",
        f"- 随机种子：`{report['random_seed']}`",
        f"- 设备：`{report['device']}`",
        "- outer：从 `<OUTPUT_BEGIN>` 开始完全自由贪心生成 effect-tree header。",
        "- local：真实 NodeSpec 前缀由环境给出；模型 token 自回归，错误 token 保留在上下文；运行时数值通过环境在寄存器操作数后插入被 mask 的 `CONST_VALUE`，随机目标仍通过 `<OBS_*>` 交互。",
        "- 牌堆：模型只接收抽牌堆、弃牌堆、消耗堆数量，不接收各堆具体卡牌组成。",
        "",
        "## 总体结果",
        "",
        f"- outer 完全正确：{summary['outer_exact_cases']}/{summary['cases']}；token 准确率 {_fmt_rate(summary['outer_token_accuracy'])}（{summary['outer_scored_tokens']} 个 token）。",
        f"- local 完全正确节点：{summary['local_exact_nodes']}/{summary['local_nodes']}；模型 token 准确率 {_fmt_rate(summary['local_token_accuracy'])}（{summary['local_scored_tokens']} 个 token）。",
        f"- delta：{summary['delta_slots']} 个向量，活跃位 MAE `{summary['delta_active_slot_mae']:.6f}`，全向量 MAE `{summary['delta_all_vector_mae']:.6f}`。",
        f"- 模型输出 `<UNK>`：{summary['unknown_output_tokens']} 个。",
        "",
        "| 案例 | 打出的牌 | outer | local 节点 | local token | delta 活跃位 MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        local = case["local"]
        scored = sum(node["metrics"]["scored_tokens"] for node in local)
        correct = sum(node["metrics"]["correct_tokens"] for node in local)
        deltas = [delta for node in local for delta in node["delta_slots"]]
        active_count = sum(delta["active_slot_count"] for delta in deltas)
        active_error = sum(
            delta["active_slot_mae"] * delta["active_slot_count"]
            for delta in deltas
        )
        lines.append(
            f"| {case['name']} | {case['input']['action']} | "
            f"{_fmt_rate(case['outer']['metrics']['token_accuracy'])} | "
            f"{sum(node['metrics']['exact_match'] for node in local)}/{len(local)} | "
            f"{_fmt_rate(correct / max(scored, 1))} | "
            f"{active_error / max(active_count, 1):.6f} |"
        )

    for case_index, case in enumerate(report["cases"], 1):
        value = case["input"]
        player = value["player"]
        piles = value["piles"]
        lines.extend([
            "",
            f"## {case_index}. {case['name']}",
            "",
            value["note"],
            "",
            f"- 场景：第 {value['floor']} 层，第 {value['act']} 幕，{value['encounter_kind']}遭遇 `{value['encounter']}`，第 {value['turn']} 回合。",
            f"- 打出：{value['action']}。卡面规则：{value['action_rule']}",
            f"- 手牌（{len(value['hand'])} 张）：{'、'.join(value['hand']) if value['hand'] else '空'}。",
            f"- 遗物：{'、'.join(value['relics']) if value['relics'] else '无'}。",
            f"- 战士：生命 {player['hp']}/{player['max_hp']}，格挡 {player['block']}，能量 {player['energy']}，金币 {player['gold']}；状态：{player['buffs']}；计数器：{player['counters']}。",
            f"- 牌堆计数：抽牌堆 {piles['draw_count']}，弃牌堆 {piles['discard_count']}，消耗堆 {piles['exhaust_count']}；具体组成不提供给模型。",
        ])
        for enemy in value["enemies"]:
            lines.append(
                f"- 敌人 slot {enemy['slot']}（`{enemy['id']}`）：生命 "
                f"{enemy['hp']}/{enemy['max_hp']}，格挡 {enemy['block']}，"
                f"意图伤害 {enemy['base_damage']}→{enemy['adjusted_damage']}，"
                f"段数 {enemy['hits']}，状态：{enemy['buffs']}。"
            )

        outer = case["outer"]
        lines.extend([
            "",
            "### Outer 结果",
            "",
            f"准确率 {_fmt_rate(outer['metrics']['token_accuracy'])}，完全一致={outer['metrics']['exact_match']}，正常结束={outer['terminated']}，首次错误：{_first_error_text(outer['metrics'])}。",
            "",
            "目标 header：",
            "",
            "```text",
            _render_tokens(outer["expected_tokens"]),
            "```",
            "",
            "自回归生成 header：",
            "",
            "```text",
            _render_tokens(outer["predicted_tokens"]),
            "```",
            "",
            "### Local 结果",
            "",
            "| path | 节点 | 完全正确 | token 准确率 | 首次错误 | delta 活跃位 MAE |",
            "| --- | --- | ---: | ---: | --- | ---: |",
        ])
        for node in case["local"]:
            delta_rows = node["delta_slots"]
            active_count = sum(row["active_slot_count"] for row in delta_rows)
            active_error = sum(
                row["active_slot_mae"] * row["active_slot_count"]
                for row in delta_rows
            )
            delta_text = (
                f"{active_error / active_count:.6f}" if active_count else "-"
            )
            lines.append(
                f"| `{node['path']}` | `{node['kind']}` | "
                f"{node['metrics']['exact_match']} | "
                f"{_fmt_rate(node['metrics']['token_accuracy'])} | "
                f"{_first_error_text(node['metrics'])} | {delta_text} |"
            )
        changed = [
            (node, delta)
            for node in case["local"] for delta in node["delta_slots"]
            if delta["active_slot_count"]
        ]
        if changed:
            lines.extend(["", "delta 活跃位明细（数值均为规范化向量空间）：", ""])
            for node, delta in changed:
                values = "；".join(
                    f"`{slot['name']}` 目标={slot['expected']:+.6f}、预测={slot['predicted']:+.6f}、绝对误差={slot['absolute_error']:.6f}"
                    for slot in delta["active_slots"]
                )
                lines.append(f"- `{node['path']}` / `{node['kind']}`：{values}。")
        mismatched_nodes = [
            node for node in case["local"] if not node["metrics"]["exact_match"]
        ]
        if mismatched_nodes:
            lines.extend(["", "local 出错节点的完整自回归对照：", ""])
            for node in mismatched_nodes:
                lines.extend([
                    f"#### `{node['path']}` / `{node['kind']}`",
                    "",
                    "目标：",
                    "",
                    "```text",
                    _render_tokens(node["expected_tokens"]),
                    "```",
                    "",
                    "预测（环境 token 已按真实 transcript 注入）：",
                    "",
                    "```text",
                    _render_tokens(node["predicted_tokens"]),
                    "```",
                    "",
                ])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--random-cases", type=int, default=10)
    parser.add_argument("--max-outer-tokens", type=int, default=512)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.random_cases < 0 or args.max_outer_tokens < 1:
        raise ValueError("random-cases must be non-negative and max-outer-tokens positive")
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

    cases = build_cases(
        config.state,
        random_seed=args.random_seed,
        random_cases=args.random_cases,
    )
    max_outer_tokens = min(
        args.max_outer_tokens, config.decoder.max_tokens - 1
    )
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[v7-eval] case={index}/{len(cases)} {case.name}", flush=True)
        results.append(evaluate_case(
            model, case, codec, config, device, max_outer_tokens
        ))
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "schema_version": int(checkpoint.get("schema_version", -1)),
        "executor_contract_version": int(checkpoint.get("executor_contract_version", -1)),
        "device": str(device),
        "random_seed": args.random_seed,
        "random_cases": args.random_cases,
        "max_outer_tokens": max_outer_tokens,
        "evaluation_protocol": {
            "outer": "free greedy autoregressive generation",
            "local": "ground-truth NodeSpec prefix; model-owned tokens generated autoregressively; READ_BIND and inline CONST_VALUE tokens injected from the realized transcript",
            "model_errors_preserved_in_context": True,
            "pile_composition_visible": False,
        },
        "cases": results,
    }
    report["summary"] = _aggregate(results, config.state)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    summary = report["summary"]
    print(
        f"[v7-eval] step={report['checkpoint_step']} "
        f"outer_acc={summary['outer_token_accuracy']:.6f} "
        f"local_acc={summary['local_token_accuracy']:.6f} "
        f"delta_active_mae={summary['delta_active_slot_mae']:.6f} "
        f"unknown={summary['unknown_output_tokens']}",
        flush=True,
    )
    print(f"json={args.json_output}\nmarkdown={args.markdown_output}")


if __name__ == "__main__":
    main()
