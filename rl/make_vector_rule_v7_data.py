"""Generate schema-v7 effect-tree records and local sequence statistics.

The v7 producer is intentionally small and self-contained.  It reuses the
canonical randomized state/action producer, but compiles the realized flat
events into an anonymous effect tree.  Whirlwind is represented by one static
AOE round under ``REPEAT`` plus ``CALL_SELF``; the runtime repeat count never
duplicates the body in the training record.

Each JSONL record contains:

* ``effect_tree``: nested ``EffectNode`` dictionaries;
* ``tokens``: readable pre-order tree serialization (one logical token/line);
* ``node_local_sequences``: per-node ``NodeSpec`` condition prefixes and
  executor-body targets. Each local body declares runtime reads first, then
  expands all already-valued register operands inline as environment-owned
  constants inside symbolic calculations; and
* ``lengths``: compressed outer/header/local/repeat-body lengths.

These records are a data contract/profiling producer.  The existing v6 trainer
does not consume schema-v7 records.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

from env.core.ironclad_cards import IRONCLAD_CARDS
from rl.make_vector_rule_training_data import CARD_PROGRAM_INDEX, build_record
from rl.rule_vector_model import (
    StateVectorConfig,
    _calculation_atom,
    _calculation_number,
)


V7_DATA_SCHEMA_VERSION = 7
# Contract 10 removes register-value copying from the model. Runtime reads and
# earlier calculation results are inserted inline whenever reused as operands.
V7_EXECUTOR_CONTRACT_VERSION = 10


@dataclass
class EffectNode:
    kind: str
    entity_ref: str | None = None
    runtime_entity_ref: str | None = None
    header_params: dict[str, Any] = field(default_factory=dict)
    children: list["EffectNode"] = field(default_factory=list)
    calculations: list[dict[str, Any]] = field(default_factory=list)
    delta: list[float] | None = None
    # A node may change several independent state fields.  Each sparse vector
    # is emitted at its own DELTA_SLOT; ``delta`` remains the aggregate state
    # transition used only for sibling snapshots.
    delta_slots: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _action_name(record: Mapping[str, Any]) -> str:
    program = tuple(int(value) for value in record["action"]["card_rule"])
    matches = [card_id for (card_id, _upgraded), value in CARD_PROGRAM_INDEX.items()
               if value == program]
    if not matches:
        raise ValueError("action program is not in the canonical card vocabulary")
    return matches[0]


def _event_delta(event: Mapping[str, Any]) -> list[float] | None:
    value = event.get("delta")
    return None if value is None else [float(item) for item in value]


def _event_node(event: Mapping[str, Any], kind: str, *, params=None) -> EffectNode:
    return EffectNode(
        kind=kind,
        entity_ref=str(event.get("entity_ref") or "ENTITY_SELF"),
        header_params=dict(params or {}),
        calculations=[dict(item) for item in event.get("calculations", ())],
        delta=_event_delta(event),
    )


def _damage_base(event: Mapping[str, Any]) -> int | None:
    for calculation in event.get("calculations", ()):
        if calculation.get("type") == "READ_BASE_DAMAGE":
            return int(calculation.get("value", 0))
    return None


def _damage_params(
    event: Mapping[str, Any], *, action_name: str | None = None,
) -> dict[str, Any]:
    calculation_types = {
        str(row.get("type")) for row in event.get("calculations", ())
    }
    if action_name == "Body Slam" and "READ_BASE_DAMAGE" in calculation_types:
        return {"DAMAGE_SOURCE": "PLAYER_BLOCK"}
    if "READ_BASE_DAMAGE" in calculation_types:
        params: dict[str, Any] = {"BASE_DAMAGE": _damage_base(event)}
        if "READ_DAMAGE_MULTIPLIER" in calculation_types:
            params["DAMAGE_MULTIPLIER_SOURCE"] = "PARENT_RELIC_TRIGGER"
        if event.get("damage_source") == "RELIC_FIXED":
            params["DAMAGE_SOURCE_KIND"] = "RELIC_FIXED"
        return params
    raise ValueError("enemy damage event has no supported damage source")


def _call_node(repeat_count: int) -> EffectNode:
    return EffectNode(
        kind="CALL",
        header_params={"CALL_TARGET": "SELF", "ARG_DECREMENT": 1},
        calculations=[{
            "type": "READ_CALL_REMAINING",
            "source": "call_frame.remaining",
            "value": max(0, int(repeat_count) - 1),
        }],
    )


def _repeat_node(
    body: EffectNode, *, repeat_count: int, runtime_source: str | None = None,
) -> EffectNode:
    if runtime_source is None:
        header = {"REPEAT_COUNT": int(repeat_count)}
        calculations: list[dict[str, Any]] = []
    else:
        header = {"REPEAT_SOURCE": runtime_source}
        calculations = [{
            "type": "READ_REPEAT_COUNT",
            "source": "action.paid_energy",
            "value": int(repeat_count),
        }]
    return EffectNode(
        kind="REPEAT", header_params=header,
        children=[body, _call_node(repeat_count)], calculations=calculations,
    )


def _has_dense_delta(delta: Sequence[float] | None) -> bool:
    return delta is not None and any(abs(float(value)) >= 1.0e-8 for value in delta)


def _world_slot_names(config: StateVectorConfig) -> tuple[str, ...]:
    """Return names in exactly the order used by ``VectorizedState.flat``."""

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


def _split_delta_slots(
    delta: Sequence[float] | None, config: StateVectorConfig,
) -> list[list[float]]:
    """Split one dense transition into sparse, independently supervised slots.

    Derived presence/fraction features belong to their source impact: HP carries
    HP fraction and block carries block presence. All other changed fields are
    independent effects and receive their own slot.
    """

    if delta is None:
        return []
    values = [float(value) for value in delta]
    names = _world_slot_names(config)
    if len(values) != len(names):
        raise ValueError("delta width does not match the v7 world layout")
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        if abs(value) < 1.0e-8:
            continue
        name = names[index]
        if name.endswith(".hp_frac"):
            key = name.removesuffix(".hp_frac") + ".hp"
        elif name.endswith(".block_present"):
            key = name.removesuffix(".block_present") + ".block"
        else:
            key = name
        groups.setdefault(key, []).append(index)
    slots: list[list[float]] = []
    for indices in groups.values():
        sparse = [0.0] * len(values)
        for index in indices:
            sparse[index] = values[index]
        slots.append(sparse)
    return slots


def _assign_delta_slots(node: EffectNode, config: StateVectorConfig) -> None:
    node.delta_slots = _split_delta_slots(node.delta, config)
    for child in node.children:
        _assign_delta_slots(child, config)


def _header_param_tokens(key: str, value: Any) -> list[str]:
    """Serialize exact rule constants already present in rule-token input."""

    tokens = [str(key)]
    if isinstance(value, bool):
        tokens.append(f"BOOL_{int(value)}")
    elif isinstance(value, (int, float)):
        tokens.extend(_calculation_number(value))
    else:
        tokens.append(f"VALUE_{_calculation_atom(value)}")
    return tokens


def _header_tokens(node: EffectNode) -> list[str]:
    tokens = ["<REGION_BEGIN>", "REGION_KIND", f"REGION_{node.kind}"]
    if node.entity_ref is not None:
        tokens.extend(("ENTITY_REF", node.entity_ref))
    for key, value in node.header_params.items():
        tokens.extend(_header_param_tokens(key, value))
    return tokens


@dataclass
class _OwnedTokens:
    tokens: list[str] = field(default_factory=list)
    model_mask: list[bool] = field(default_factory=list)
    # Runtime reads are emitted first, while symbolic operations are buffered
    # so every already-valued register can expand inline at operand positions.
    register_reads: list[tuple[str, str, int | float]] = field(default_factory=list)
    register_values: dict[str, int | float] = field(default_factory=dict)
    calculation_tokens: list[str] = field(default_factory=list)
    calculation_model_mask: list[bool] = field(default_factory=list)

    def model(self, *tokens: str) -> None:
        self.tokens.extend(tokens)
        self.model_mask.extend(True for _ in tokens)

    def environment(self, *tokens: str) -> None:
        self.tokens.extend(tokens)
        self.model_mask.extend(False for _ in tokens)

    def _calculation_model(self, *tokens: str) -> None:
        self.calculation_tokens.extend(tokens)
        self.calculation_model_mask.extend(True for _ in tokens)

    def _calculation_environment(self, *tokens: str) -> None:
        self.calculation_tokens.extend(tokens)
        self.calculation_model_mask.extend(False for _ in tokens)

    def request(self, source: str, binding: str, value: int | float) -> None:
        source = str(source)
        binding = str(binding)
        self.register_reads.append((source, binding, value))
        self.register_values[binding] = value
        # The model chooses the field whose value is needed. The environment
        # assigns the stable binding name and closes the request, so variable
        # naming is not another sequence-learning task.
        self.model(
            "<READ_REQUEST_BEGIN>",
            "READ_SOURCE",
            f"VALUE_{_calculation_atom(source)}",
        )
        self.environment(
            "READ_BIND",
            f"VALUE_{_calculation_atom(binding)}",
            "<READ_REQUEST_END>",
        )

    def query_random_target(self, entity_ref: str) -> None:
        self.model("<QUERY_BEGIN>", "QUERY_RANDOM_TARGET", "<QUERY_END>")
        self.environment(
            "<OBS_BEGIN>", "OBS_RANDOM_TARGET", entity_ref, "<OBS_END>",
        )

    def operation(
        self, kind: str, *, numeric_values: Mapping[str, int | float] | None = None,
        **registers: str,
    ) -> None:
        # Operations are buffered so any previously valued register can be
        # expanded to an environment constant immediately after its name.
        # A result is registered only after its model-owned CALC_VALUE is
        # emitted, so the environment never leaks the current answer.
        numeric_values = numeric_values or {}
        self._calculation_model("CALCULATION_BEGIN", f"CALCULATION_{kind}")
        for key, register in registers.items():
            self._calculation_model(
                f"CALC_FIELD_{_calculation_atom(key)}",
                f"VALUE_{_calculation_atom(register)}",
            )
            # Result registers are assigned by the following CALC_VALUE.
            # Previously assigned numeric operands receive an exact constant,
            # whether they came from a runtime read or an earlier operation.
            if (
                isinstance(register, str)
                and register.startswith("reg.")
                and register in self.register_values
                and key not in {"result", "source_result", "destination_result"}
            ):
                self._calculation_environment(
                    "CONST_VALUE", *_calculation_number(self.register_values[register])
                )
            if key in numeric_values:
                self._calculation_model(
                    "CALC_VALUE", *_calculation_number(numeric_values[key]),
                )
                if isinstance(register, str) and register.startswith("reg."):
                    self.register_values[register] = numeric_values[key]
        self._calculation_model("CALCULATION_END")

    def emit_calculations(self) -> None:
        """Emit inline environment constants and model calculations."""

        self.tokens.append("<CALCULATIONS_BEGIN>")
        self.model_mask.append(True)
        self.tokens.extend(self.calculation_tokens)
        self.model_mask.extend(self.calculation_model_mask)


def _calculation(node: EffectNode, kind: str) -> Mapping[str, Any] | None:
    return next(
        (row for row in node.calculations if row.get("type") == kind),
        None,
    )


def _entity_field(node: EffectNode, field_name: str) -> str:
    entity = str(node.runtime_entity_ref or node.entity_ref or "ENTITY_SELF")
    if entity.startswith("ENTITY_ENEMY_SLOT_"):
        slot = int(entity.rsplit("_", 1)[-1])
        return f"enemy[{slot}].{field_name}"
    return f"player.{field_name}"


def _header_registers(node: EffectNode) -> dict[str, str]:
    """Expose parsed NodeSpec constants without an environment read."""

    registers: dict[str, str] = {}
    for key, value in node.header_params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        registers[key] = f"header.{key}"
    return registers


def _symbolic_damage(node: EffectNode, owned: _OwnedTokens, bindings: Mapping[str, str]) -> None:
    if "BASE_DAMAGE" in bindings:
        damage = bindings["BASE_DAMAGE"]
    elif node.header_params.get("DAMAGE_SOURCE") == "PLAYER_BLOCK":
        base = _calculation(node, "READ_BASE_DAMAGE")
        if base is None:
            raise ValueError("PLAYER_BLOCK damage source requires its realized runtime value")
        owned.request("player.block", "reg.damage", int(base.get("value", 0)))
        damage = "reg.damage"
    else:
        raise ValueError("DAMAGE node has neither BASE_DAMAGE nor a runtime source")

    strength = _calculation(node, "READ_STRENGTH")
    if strength is not None and int(strength.get("value", 0)) != 0:
        add_strength = _calculation(node, "ADD_STRENGTH")
        if add_strength is None:
            raise ValueError("Strength read requires ADD_STRENGTH")
        owned.operation(
            "CHECK_PRESENCE", field="player.buff.Strength",
            result="reg.has_strength",
        )
        owned.request(
            "player.buff.Strength", "reg.strength", int(strength["value"])
        )
        owned.operation(
            "ADD", left=damage, right="reg.strength",
            result="reg.damage_after_strength",
            numeric_values={"result": add_strength["result"]},
        )
        damage = "reg.damage_after_strength"

    multiplier = _calculation(node, "READ_DAMAGE_MULTIPLIER")
    if multiplier is not None:
        multiplied_damage = _calculation(node, "MULTIPLY_DAMAGE")
        if multiplied_damage is None:
            raise ValueError("damage multiplier read requires MULTIPLY_DAMAGE")
        owned.request(
            str(multiplier.get("source", "parent.relic_trigger")),
            "reg.damage_multiplier", int(multiplier.get("value", 1)),
        )
        owned.operation(
            "MULTIPLY", left=damage, right="reg.damage_multiplier",
            result="reg.damage_after_multiplier",
            numeric_values={"result": multiplied_damage["result"]},
        )
        damage = "reg.damage_after_multiplier"

    vulnerable = _calculation(node, "READ_VULNERABLE")
    if vulnerable is not None and int(vulnerable.get("value", 0)) != 0:
        multiplied = _calculation(node, "MULTIPLY_VULNERABLE")
        divided = _calculation(node, "FLOOR_DIVIDE_VULNERABLE")
        if multiplied is None or divided is None:
            raise ValueError("Vulnerable requires multiply and floor-divide rows")
        vulnerable_field = _entity_field(node, "buff.Vulnerable")
        owned.operation(
            "CHECK_PRESENCE", field=vulnerable_field,
            result="reg.has_vulnerable",
        )
        owned.operation(
            "MULTIPLY", left=damage,
            result="reg.damage_vulnerable_numerator",
            numeric_values={"result": multiplied["result"]},
        )
        owned.operation(
            "FLOOR_DIVIDE",
            left="reg.damage_vulnerable_numerator",
            result="reg.damage_after_vulnerable",
            numeric_values={"result": divided["result"]},
        )
        damage = "reg.damage_after_vulnerable"

    block = _calculation(node, "READ_TARGET_BLOCK")
    # Always keep the block branch in the local contract, including block=0.
    # This gives the executor one fixed causal program and lets the environment
    # response (rather than a structural guess) select the zero-absorption case.
    if block is not None:
        block_field = _entity_field(node, "block")
        owned.request(block_field, "reg.target_block", int(block["value"]))
        absorbed = _calculation(node, "BLOCK_ABSORB")
        subtract_block = _calculation(node, "SUBTRACT_BLOCK")
        subtract_damage = _calculation(node, "SUBTRACT_BLOCK_DAMAGE")
        if absorbed is None or subtract_block is None or subtract_damage is None:
            raise ValueError("target block read requires the complete block trace")
        owned.operation(
            "CHECK_PRESENCE", field=_entity_field(node, "block_present"),
            result="reg.has_block",
        )
        owned.operation(
            "MIN", left=damage, right="reg.target_block",
            result="reg.absorbed_damage",
            numeric_values={"result": absorbed["absorbed"]},
        )
        owned.operation(
            "SUBTRACT", left="reg.target_block",
            right="reg.absorbed_damage", result="reg.target_block_after",
            numeric_values={"result": subtract_block["result"]},
        )
        owned.operation(
            "SUBTRACT", left=damage,
            right="reg.absorbed_damage", result="reg.unblocked_damage",
            numeric_values={"result": subtract_damage["result"]},
        )
        damage = "reg.unblocked_damage"

    hp = _calculation(node, "READ_TARGET_HP")
    if hp is not None:
        subtract_hp = _calculation(node, "SUBTRACT_HP")
        if subtract_hp is None:
            raise ValueError("target HP read requires SUBTRACT_HP")
        hp_field = _entity_field(node, "hp")
        owned.request(hp_field, "reg.target_hp", int(hp["value"]))
        owned.operation(
            "SUBTRACT", left="reg.target_hp", right=damage,
            result="reg.target_hp_after_subtract",
            numeric_values={
                "result": int(subtract_hp["left"]) - int(subtract_hp["right"]),
            },
        )
        owned.operation(
            "RELU", input="reg.target_hp_after_subtract",
            result="reg.target_hp_after",
            numeric_values={"result": subtract_hp["result"]},
        )

def _symbolic_self_damage(node: EffectNode, owned: _OwnedTokens) -> None:
    trigger = _calculation(node, "READ_TRIGGER_DAMAGE")
    subtract = _calculation(node, "SUBTRACT_PLAYER_HP")
    if trigger is None or subtract is None:
        raise ValueError("SELF_DAMAGE requires trigger damage and player HP")
    owned.request(
        "parent.trigger_damage", "reg.trigger_damage", int(trigger["value"])
    )
    owned.request("player.hp", "reg.player_hp", int(subtract["left"]))
    owned.operation(
        "SUBTRACT", left="reg.player_hp", right="reg.trigger_damage",
        result="reg.player_hp_after_subtract",
        numeric_values={
            "result": int(subtract["left"]) - int(subtract["right"]),
        },
    )
    owned.operation(
        "RELU", input="reg.player_hp_after_subtract",
        result="reg.player_hp_after",
        numeric_values={"result": subtract["result"]},
    )


def _symbolic_pile_operation(
    node: EffectNode, owned: _OwnedTokens, bindings: Mapping[str, str],
) -> None:
    operation = node.header_params.get("PILE_OPERATION")
    draw_row = next(
        row for row in node.calculations
        if row.get("type") == "APPLY_DELTA"
        and row.get("field") == "piles.draw_count"
    )
    owned.request(
        "piles.draw_count", "reg.draw_count", draw_row.get("left", 0),
    )
    if operation == "SHUFFLE_DISCARD_INTO_DRAW":
        discard_row = next(
            row for row in node.calculations
            if row.get("type") == "APPLY_DELTA"
            and row.get("field") == "piles.discard_count"
        )
        owned.request(
            "piles.discard_count", "reg.discard_count", discard_row.get("left", 0)
        )
        owned.operation(
            "TRANSFER_PILE", source="piles.discard",
            destination="piles.draw", source_count="reg.discard_count",
            destination_count="reg.draw_count",
            source_result="reg.discard_count_after",
            destination_result="reg.draw_count_after",
            numeric_values={
                "source_result": discard_row["result"],
                "destination_result": draw_row["result"],
            },
        )
    elif operation == "INSERT_INTO_DRAW":
        owned.operation(
            "ADD", field="piles.draw_count", left="reg.draw_count",
            right=bindings["CARD_COUNT"], result="reg.draw_count_after",
            numeric_values={"result": draw_row["result"]},
        )
    else:
        raise ValueError(f"unsupported PILE_SHUFFLE operation: {operation}")


def _symbolic_relic_counter(
    node: EffectNode, owned: _OwnedTokens, bindings: Mapping[str, str],
) -> None:
    field_name = str(node.header_params["COUNTER_REF"])
    concrete = _calculation(node, "RELIC_COUNTER_UPDATE")
    if concrete is None:
        raise ValueError("RELIC_COUNTER requires RELIC_COUNTER_UPDATE")
    owned.request(field_name, "reg.relic_counter", concrete.get("left", 0))
    owned.operation(
        "ADD", field=field_name, left="reg.relic_counter",
        right=bindings["COUNTER_INCREMENT"], result="reg.counter_advanced",
        numeric_values={
            "result": int(concrete["left"]) + int(concrete["increment"]),
        },
    )
    owned.operation(
        "CHECK_COUNTER_THRESHOLD", left="reg.counter_advanced",
        threshold=bindings["COUNTER_THRESHOLD"], result="reg.counter_triggered",
        numeric_values={"result": int(concrete["triggered"])},
    )
    owned.operation(
        "SELECT_COUNTER_RESULT", advanced="reg.counter_advanced",
        triggered="reg.counter_triggered", reset=bindings["COUNTER_RESET"],
        result="reg.relic_counter_after",
        numeric_values={"result": concrete["result"]},
    )
    owned.operation(
        "WRITE_RELIC_COUNTER", field=field_name,
        value="reg.relic_counter_after",
    )


def _symbolic_control(
    node: EffectNode, owned: _OwnedTokens, bindings: Mapping[str, str],
) -> bool:
    if node.kind in {"CARD_COMMIT", "CARD_PLAY"}:
        owned.operation("EXECUTE_CHILDREN", scope="node.children")
    elif node.kind == "ATTACK":
        owned.operation(
            "EXECUTE_ATTACK_CHILDREN", targets=str(node.entity_ref),
            scope="node.children",
        )
    elif node.kind == "REPEAT":
        if "REPEAT_COUNT" in bindings:
            repeat_count = bindings["REPEAT_COUNT"]
        else:
            repeat = _calculation(node, "READ_REPEAT_COUNT")
            if repeat is None:
                raise ValueError("REPEAT requires a fixed count or runtime count read")
            owned.request(
                str(repeat["source"]), "reg.repeat_count", int(repeat["value"]),
            )
            repeat_count = "reg.repeat_count"
        owned.operation(
            "INITIALIZE_REPEAT", count=repeat_count,
            body="node.children", result="reg.call_frame",
        )
    elif node.kind == "CALL":
        remaining = _calculation(node, "READ_CALL_REMAINING")
        if remaining is None:
            raise ValueError("CALL requires the current call-frame value")
        owned.request(
            str(remaining["source"]), "reg.call_remaining",
            int(remaining["value"]),
        )
        owned.operation(
            "CALL_SELF", remaining="reg.call_remaining",
            decrement=bindings["ARG_DECREMENT"],
            result="reg.call_frame_after",
            numeric_values={
                "result": max(
                    0, int(remaining["value"])
                    - int(node.header_params["ARG_DECREMENT"]),
                ),
            },
        )
    elif node.kind == "AOE_DAMAGE_ROUND":
        owned.operation(
            "ITERATE_ENEMY_MASK", targets=str(node.entity_ref),
            order="header.ENEMY_ORDER",
        )
    elif node.kind == "RANDOM_HIT":
        owned.operation(
            "EXECUTE_RANDOM_HIT_CHILDREN", scope="node.children",
            target="query.response.target",
        )
    elif node.kind == "QUERY":
        response = _calculation(node, "QUERY_RANDOM_TARGET_RESPONSE")
        if response is None:
            raise ValueError("QUERY_RANDOM_TARGET requires an environment response")
        owned.query_random_target(str(response["entity_ref"]))
    elif node.kind == "COMPLETE":
        owned.operation("COMPLETE_ACTION")
    else:
        return False
    return True


def _delta_binding(
    node: EffectNode, field_name: str, row: Mapping[str, Any],
    bindings: Mapping[str, str], owned: _OwnedTokens, *, before: str,
) -> tuple[str, str]:
    if node.kind == "CARD_PAYMENT" and field_name == "player.energy":
        if "PAYMENT_COST" in bindings:
            return bindings["PAYMENT_COST"], "SUBTRACT"
        if node.header_params.get("PAYMENT_SOURCE") == "PLAYER_ENERGY":
            return before, "SUBTRACT"
        raise ValueError("CARD_PAYMENT has neither a fixed cost nor a runtime source")
    if node.kind == "DRAW" and field_name == "piles.draw_count":
        return bindings["DRAW_COUNT"], "SUBTRACT"
    if node.kind == "ENERGY" and field_name == "player.energy":
        operation = "ADD" if node.header_params.get("ENERGY_OP") == "GAIN" else "SUBTRACT"
        return bindings["ENERGY_AMOUNT"], operation
    if node.kind == "CARD_DESTINATION":
        return bindings["CARD_COUNT"], "ADD"
    if node.kind in {"STATE_CHANGE", "RELIC_COUNTER"} and "CHANGE_AMOUNT" in bindings:
        return bindings["CHANGE_AMOUNT"], "ADD"
    if field_name.startswith("player.counter.") and float(row.get("delta", 0)) == 1.0:
        return "constant.one", "ADD"
    raise ValueError(
        f"{node.kind} has no symbolic delta binding for {field_name}; "
        "add an explicit node operation instead of reading an effect delta"
    )


def _symbolic_generic(
    node: EffectNode, owned: _OwnedTokens, bindings: Mapping[str, str],
) -> None:
    for index, row in enumerate(node.calculations):
        if row.get("type") != "APPLY_DELTA":
            continue
        field_name = str(row.get("field", ""))
        # hp_frac is derived exactly from HP/max HP by the runtime and is not
        # an independent read, operation, or prediction target.
        if field_name == "player.hp_frac":
            continue
        before = f"reg.field_before.{index}"
        after = f"reg.field_after.{index}"
        owned.request(field_name, before, row.get("left", 0))
        delta, operation = _delta_binding(
            node, field_name, row, bindings, owned, before=before,
        )
        owned.operation(
            operation, field=field_name, left=before,
            right=delta, result=after,
            numeric_values={"result": row["result"]},
        )


def _local_sequence(node: EffectNode) -> _OwnedTokens:
    """Build one NodeSpec-conditioned, mixed model/environment invocation."""

    owned = _OwnedTokens()
    header = _header_tokens(node)
    owned.environment(*header, "<NODE_EXECUTE>")
    if node.entity_ref == "ENTITY_QUERY_TARGET" and node.runtime_entity_ref:
        owned.environment(
            "<OBS_BEGIN>", "OBS_RANDOM_TARGET", node.runtime_entity_ref,
            "<OBS_END>",
        )
    bindings = _header_registers(node)
    if node.kind == "DAMAGE":
        _symbolic_damage(node, owned, bindings)
    elif node.kind == "SELF_DAMAGE":
        _symbolic_self_damage(node, owned)
    elif node.kind == "PILE_SHUFFLE":
        _symbolic_pile_operation(node, owned, bindings)
    elif node.kind == "RELIC_COUNTER":
        _symbolic_relic_counter(node, owned, bindings)
    elif _symbolic_control(node, owned, bindings):
        pass
    else:
        _symbolic_generic(node, owned, bindings)
    owned.emit_calculations()
    owned.model("<CALCULATIONS_END>")
    for _slot in node.delta_slots:
        owned.model("DELTA_SLOT")
    owned.model("<REGION_END>")
    return owned


def _tree_tokens(node: EffectNode) -> list[str]:
    # The diagnostic full tree includes environment-owned query observations,
    # while
    # training uses the ownership mask from each local invocation.
    local = _local_sequence(node)
    condition_length = len(_header_tokens(node)) + 1
    tokens = _header_tokens(node) + local.tokens[condition_length:-1]
    for child in node.children:
        tokens.extend(_tree_tokens(child))
    tokens.append("<REGION_END>")
    return tokens


def _outer_tree_tokens_for_record(node: EffectNode) -> list[str]:
    tokens = _header_tokens(node)
    for child in node.children:
        tokens.extend(_outer_tree_tokens_for_record(child))
    tokens.append("<REGION_END>")
    return tokens


def _walk(node: EffectNode, path: str = "root") -> Iterable[tuple[str, EffectNode]]:
    yield path, node
    for index, child in enumerate(node.children):
        yield from _walk(child, f"{path}/{index}")


def _static_body_tokens(node: EffectNode) -> list[str]:
    """Return the one-time body used by a REPEAT node."""

    if node.kind != "REPEAT":
        return []
    return [token for child in node.children if child.kind != "CALL"
            for token in _tree_tokens(child)]


def _lengths(root: EffectNode, flat_event_token_length: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    nodes = list(_walk(root))
    outer_headers = [token for _path, node in nodes for token in _header_tokens(node)]
    local_rows = []
    for path, node in nodes:
        local = _local_sequence(node)
        condition_length = len(_header_tokens(node)) + 1
        calc_begin = local.tokens.index("<CALCULATIONS_BEGIN>")
        calc_end = local.tokens.index("<CALCULATIONS_END>", calc_begin)
        calculation_length = calc_end - calc_begin + 1
        constant_expansion_count = local.tokens.count("CONST_VALUE")
        constant_expansion_length = 0
        for index, token in enumerate(local.tokens):
            if token != "CONST_VALUE":
                continue
            constant_expansion_length += 1
            for next_index in range(index + 1, len(local.tokens)):
                constant_expansion_length += 1
                if local.tokens[next_index] == "NUM_END":
                    break
        local_rows.append({
            "path": path,
            "kind": node.kind,
            "tokens": local.tokens,
            "model_mask": local.model_mask,
            "length": len(local.tokens),
            "condition_length": condition_length,
            "target_length": sum(local.model_mask),
            "constant_expansion_length": constant_expansion_length,
            "constant_expansion_count": constant_expansion_count,
            "calculation_length": calculation_length,
        })
    repeat_bodies = [
        _static_body_tokens(node) for _path, node in nodes if node.kind == "REPEAT"
    ]
    lengths = {
        "outer_header_length": len(outer_headers),
        "node_local_length_max": max((row["length"] for row in local_rows), default=0),
        "constant_expansion_length_max": max(
            (row["constant_expansion_length"] for row in local_rows), default=0
        ),
        "constant_expansion_count_max": max(
            (row["constant_expansion_count"] for row in local_rows), default=0
        ),
        "calculation_length_max": max(
            (row["calculation_length"] for row in local_rows), default=0
        ),
        "repeat_static_body_length_max": max(
            (len(body) for body in repeat_bodies), default=0
        ),
        "flat_event_token_length": int(flat_event_token_length),
        "node_count": len(nodes),
    }
    return lengths, local_rows


def _active_slots(record: Mapping[str, Any], config: StateVectorConfig) -> list[int]:
    present = config.enemy_scalar_names.index("present")
    return [
        index for index, row in enumerate(record["state"]["enemies"])
        if float(row[present]) > 0.5
    ]


def _payment_cost(record: Mapping[str, Any], config: StateVectorConfig) -> int:
    """Return the realized payment without depending on a non-zero delta.

    Zero-cost cards legitimately produce no payment delta, and an X-cost card
    consumes all energy before the event is serialized.  The anonymous action
    program still identifies the canonical card definition on the producer
    side, so use that definition and the pre-action state for X-cost cards.
    """
    action_name = _action_name(record)
    info = IRONCLAD_CARDS[action_name]
    upgraded = int(record["action"].get("upgrade", 0) > 0)
    cost = int(info.cost[upgraded])
    if cost < 0:
        energy_index = config.player_scalar_names.index("energy")
        normalized_energy = float(record["state"]["player"][energy_index])
        cost = round(normalized_energy * config.energy_scale)
    return max(0, cost)


def _apply_delta_rows(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row for row in event.get("calculations", ())
        if row.get("type") == "APPLY_DELTA"
    ]


def _play_kind(event: Mapping[str, Any]) -> str:
    fields = {str(row.get("field", "")) for row in _apply_delta_rows(event)}
    if "player.counter.attacks_played" in fields:
        return "ATTACK"
    if "player.counter.skills_played" in fields:
        return "SKILL"
    return "OTHER"


def _single_field_params(event: Mapping[str, Any]) -> dict[str, Any]:
    rows = _apply_delta_rows(event)
    if len(rows) != 1:
        raise ValueError(
            f"{event.get('kind')} requires exactly one APPLY_DELTA row; got {len(rows)}"
        )
    row = rows[0]
    return {
        "FIELD_OR_STATUS": str(row["field"]),
        "CHANGE_AMOUNT": row["delta"],
    }


def _draw_params(event: Mapping[str, Any]) -> dict[str, Any]:
    row = _single_field_params(event)
    if row["FIELD_OR_STATUS"] != "piles.draw_count":
        raise ValueError("EVENT_DRAW must change piles.draw_count")
    return {"DRAW_COUNT": max(0, -int(row["CHANGE_AMOUNT"]))}


def _energy_params(event: Mapping[str, Any]) -> dict[str, Any]:
    row = _single_field_params(event)
    if row["FIELD_OR_STATUS"] != "player.energy":
        raise ValueError("EVENT_ENERGY must change player.energy")
    amount = int(row["CHANGE_AMOUNT"])
    return {
        "ENERGY_OP": "GAIN" if amount >= 0 else "LOSE",
        "ENERGY_AMOUNT": abs(amount),
    }


def _destination_params(event: Mapping[str, Any]) -> dict[str, Any]:
    row = _single_field_params(event)
    field = str(row["FIELD_OR_STATUS"])
    if not field.startswith("piles.") or not field.endswith("_count"):
        raise ValueError(f"card destination has an invalid pile field: {field}")
    return {
        "DESTINATION": field.removeprefix("piles.").removesuffix("_count").upper(),
        "CARD_COUNT": abs(int(row["CHANGE_AMOUNT"])),
    }


def _relic_counter_params(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "COUNTER_REF": str(event["counter_ref"]),
        "COUNTER_EVENT": str(event["counter_event"]),
        "COUNTER_INCREMENT": int(event["counter_increment"]),
        "COUNTER_THRESHOLD": int(event["counter_threshold"]),
        "COUNTER_RESET": int(event["counter_reset"]),
        "TRIGGER_EFFECT": str(event["trigger_effect"]),
        "TRIGGER_AMOUNT": int(event["trigger_amount"]),
    }


def _pile_operation_params(event: Mapping[str, Any]) -> dict[str, Any]:
    rows = _apply_delta_rows(event)
    changes = {str(row["field"]): int(row["delta"]) for row in rows}
    draw_delta = changes.get("piles.draw_count", 0)
    discard_delta = changes.get("piles.discard_count", 0)
    if draw_delta > 0 and discard_delta < 0:
        return {"PILE_OPERATION": "SHUFFLE_DISCARD_INTO_DRAW"}
    if draw_delta > 0 and discard_delta == 0:
        return {
            "PILE_OPERATION": "INSERT_INTO_DRAW",
            "CARD_COUNT": draw_delta,
        }
    raise ValueError(f"unsupported pile operation changes: {changes}")


def _compile_tree(record: Mapping[str, Any], config: StateVectorConfig) -> EffectNode:
    action_name = _action_name(record)
    action_kind = IRONCLAD_CARDS[action_name].kind.upper()
    events = list(record["events"])
    payment_params: dict[str, Any] = {}
    if action_name == "Whirlwind":
        payment_params["PAYMENT_SOURCE"] = "PLAYER_ENERGY"
    else:
        payment_params["PAYMENT_COST"] = _payment_cost(record, config)
    payment = _event_node(events[0], "CARD_PAYMENT", params=payment_params)
    double_tap_event = next(
        (event for event in events if event.get("phase") == "COMMIT_DOUBLE_TAP"),
        None,
    )
    destination_events = [
        event for event in events if event.get("phase") == "COMMIT_DESTINATION"
    ]
    complete_event = next(
        (event for event in events if event.get("phase") == "COMMIT_COMPLETE"),
        None,
    )
    body_events = [
        event for event in events[1:]
        if event is not double_tap_event
        and not any(event is destination for destination in destination_events)
        and event is not complete_event
    ]
    attack_node: EffectNode | None = None
    damage_events = [
        event for event in body_events
        if event.get("kind") == "EVENT_DAMAGE"
        and str(event.get("entity_ref", "")).startswith("ENTITY_ENEMY_SLOT_")
        and event.get("damage_source") != "RELIC_FIXED"
    ]
    damage_by_slot: dict[int, list[Mapping[str, Any]]] = {}
    for event in damage_events:
        slot = int(str(event["entity_ref"]).rsplit("_", 1)[-1])
        damage_by_slot.setdefault(slot, []).append(event)

    if action_name == "Whirlwind":
        repeat_count = _payment_cost(record, config)
        round_children = []
        for slot in _active_slots(record, config):
            event = damage_by_slot.get(slot, [None])[0]
            if event is None:
                continue
            round_children.append(_event_node(
                event, "DAMAGE", params=_damage_params(event, action_name=action_name)
            ))
        round_node = EffectNode(
            kind="AOE_DAMAGE_ROUND",
            entity_ref="ENTITY_ENEMY_MASK",
            header_params={"ENEMY_ORDER": "SLOT_ASCENDING"},
            children=round_children,
        )
        attack_node = EffectNode(
            kind="ATTACK", entity_ref="ENTITY_ENEMY_MASK",
            children=[_repeat_node(
                round_node, repeat_count=repeat_count,
                runtime_source="PAID_ENERGY",
            )],
        )
    elif action_name == "Sword Boomerang" and damage_events:
        repeat_count = 3 + int(record["action"].get("upgrade", 0) > 0)
        event = damage_events[0]
        realized_target = str(event["entity_ref"])
        query = EffectNode(
            kind="QUERY", entity_ref="ENTITY_ENEMY_MASK",
            header_params={"QUERY_KIND": "RANDOM_TARGET"},
            calculations=[{
                "type": "QUERY_RANDOM_TARGET_RESPONSE",
                "entity_ref": realized_target,
            }],
        )
        damage = EffectNode(
            kind="DAMAGE", entity_ref="ENTITY_QUERY_TARGET",
            runtime_entity_ref=realized_target,
            header_params=_damage_params(event, action_name=action_name),
            calculations=[dict(row) for row in event.get("calculations", ())],
            delta=_event_delta(event),
        )
        random_hit = EffectNode(
            kind="RANDOM_HIT", entity_ref="ENTITY_QUERY_TARGET",
            children=[query, damage],
        )
        attack_node = EffectNode(
            kind="ATTACK", entity_ref="ENTITY_ENEMY_MASK",
            children=[_repeat_node(
                random_hit, repeat_count=repeat_count,
            )],
        )
    elif action_name in {"Twin Strike", "Pummel"} and damage_events:
        repeat_count = (
            2 if action_name == "Twin Strike"
            else 4 + int(record["action"].get("upgrade", 0) > 0)
        )
        event = damage_events[0]
        damage = _event_node(
            event, "DAMAGE", params=_damage_params(event, action_name=action_name)
        )
        attack_node = EffectNode(
            kind="ATTACK", entity_ref=str(event["entity_ref"]),
            children=[_repeat_node(damage, repeat_count=repeat_count)],
        )
    elif damage_events:
        damage_children = []
        for event in damage_events:
            damage_children.append(_event_node(
                event, "DAMAGE", params=_damage_params(event, action_name=action_name)
            ))
        attack_node = EffectNode(
            kind="ATTACK", entity_ref="ENTITY_ENEMY_MASK", children=damage_children
        )

    # Preserve the realized causal order.  The attack subtree replaces the
    # contiguous enemy-damage events at their first position; relic counter,
    # shuffle, draw, and status regions remain at their original positions.
    effect_children: list[EffectNode] = []
    inserted_attack = False
    for event in body_events:
        kind = str(event.get("kind"))
        if (
            kind == "EVENT_DAMAGE"
            and str(event.get("entity_ref", "")).startswith("ENTITY_ENEMY_SLOT_")
            and event.get("damage_source") != "RELIC_FIXED"
        ):
            if attack_node is not None and not inserted_attack:
                effect_children.append(attack_node)
                inserted_attack = True
            continue
        if (
            kind == "EVENT_DAMAGE"
            and str(event.get("entity_ref", "")).startswith("ENTITY_ENEMY_SLOT_")
            and event.get("damage_source") == "RELIC_FIXED"
        ):
            effect_children.append(_event_node(
                event, "DAMAGE", params=_damage_params(event, action_name=action_name)
            ))
            continue
        if kind == "EVENT_DAMAGE" and event.get("entity_ref") == "ENTITY_SELF":
            effect_children.append(_event_node(event, "SELF_DAMAGE"))
            continue
        if kind == "EVENT_COUNTER":
            effect_children.append(_event_node(
                event, "RELIC_COUNTER", params=_relic_counter_params(event)
            ))
        elif kind == "EVENT_PILE_SHUFFLE":
            effect_children.append(_event_node(
                event, "PILE_SHUFFLE", params=_pile_operation_params(event)
            ))
        elif kind == "EVENT_DRAW":
            effect_children.append(_event_node(event, "DRAW", params=_draw_params(event)))
        elif kind == "EVENT_ENERGY":
            effect_children.append(_event_node(event, "ENERGY", params=_energy_params(event)))
        elif kind in {"EVENT_EXHAUST", "EVENT_PILE_DELTA"}:
            effect_children.append(_event_node(
                event, "CARD_DESTINATION", params=_destination_params(event)
            ))
        else:
            # The simulator can expose an attempted status application that
            # was fully prevented as an empty event.  It has no executable
            # body and no realized transition, so it is not an EffectNode.
            if not _apply_delta_rows(event):
                continue
            effect_children.append(_event_node(
                event, "STATE_CHANGE", params=_single_field_params(event)
            ))
    if attack_node is not None and not inserted_attack:
        effect_children.append(attack_node)

    card_play = EffectNode(
        kind="CARD_PLAY", entity_ref="ENTITY_SELF",
        header_params={"PLAY_KIND": action_kind},
        children=effect_children,
    )
    commit_children: list[EffectNode] = [payment]
    if double_tap_event is not None:
        commit_children.append(_event_node(
            double_tap_event, "STATE_CHANGE",
            params=_single_field_params(double_tap_event),
        ))
        commit_children.append(_repeat_node(card_play, repeat_count=2))
    else:
        commit_children.append(card_play)
    for event in destination_events:
        commit_children.append(_event_node(
            event, "CARD_DESTINATION", params=_destination_params(event)
        ))
    if complete_event is not None:
        commit_children.append(_event_node(complete_event, "COMPLETE"))
    return EffectNode(
        kind="CARD_COMMIT", entity_ref="ENTITY_SELF",
        header_params={"CARD_ACTION": "ANONYMOUS_RULE_VECTOR"},
        children=commit_children,
    )


def compile_v7_record(
    v6: Mapping[str, Any], config: StateVectorConfig,
) -> dict[str, Any]:
    """Compile an already-realized anonymous record into executor-contract-v7.

    Keeping this conversion separate lets fixed evaluations use exactly the
    same tree and local-sequence serializer as the online training stream.
    """

    root = _compile_tree(v6, config)
    _assign_delta_slots(root, config)
    # The flat length is retained only as a diagnostic baseline; v7 consumers
    # train on the local sequences below.
    from rl.rule_vector_model import serialize_vector_events, VectorEvent

    events = tuple(
        VectorEvent(
            event["kind"], event["entity_ref"],
            delta=tuple(event["delta"]) if event.get("delta") is not None else None,
            calculations=tuple(event.get("calculations", ())),
        )
        for event in v6["events"]
    )
    flat_length = len(serialize_vector_events(events).tokens) - 1
    lengths, local_rows = _lengths(root, flat_length)
    outer_tokens = [
        "<OUTPUT_BEGIN>",
        "<TRANSITION_BEGIN>", "<EFFECT_TREE_BEGIN>",
        *_outer_tree_tokens_for_record(root),
        "<EFFECT_TREE_END>", "<TRANSITION_END>", "<OUTPUT_END>",
    ]
    outer_model_mask = [True for _ in outer_tokens]
    state = dict(v6["state"])
    # The fixed player vector contains the identity-mapped presence bits. Keep
    # the same information explicit for data inspection, but never carry the
    # old active-relic list under the misleading ``relic_counters`` name.
    metadata = v6.get("metadata", {})
    relic_ids = (
        [str(value) for value in metadata.get("relics", ())]
        if isinstance(metadata, Mapping) else []
    )
    state.pop("relic_counters", None)
    state["relic_presence"] = [
        float(relic_name in set(relic_ids))
        for relic_name in config.relic_presence_names
    ]
    relics = [
        {
            "present": float(item.get("present", 1.0)),
            "counter_present": bool(item.get("counter_present", False)),
        }
        for item in v6.get("relics", ())
    ]
    return {
        "schema_version": V7_DATA_SCHEMA_VERSION,
        "executor_contract_version": V7_EXECUTOR_CONTRACT_VERSION,
        "state": state,
        "action": v6["action"],
        "hand": v6["hand"],
        "relics": relics,
        "metadata": v6.get("metadata", {}),
        "effect_tree": root.to_dict(),
        "tokens": [
            "<OUTPUT_BEGIN>",
            "<TRANSITION_BEGIN>", "<EFFECT_TREE_BEGIN>",
            *_tree_tokens(root),
            "<EFFECT_TREE_END>", "<TRANSITION_END>", "<OUTPUT_END>",
        ],
        "outer_tokens": outer_tokens,
        "outer_model_mask": outer_model_mask,
        "node_local_sequences": local_rows,
        "lengths": lengths,
    }


def build_v7_record(index: int, config: StateVectorConfig, rng: random.Random) -> dict[str, Any]:
    return compile_v7_record(build_record(index, config, rng), config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rl/data/vector-rule-v7.jsonl"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--stream", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 1 and not args.stream:
        raise ValueError("samples must be positive")
    config = StateVectorConfig()
    rng = random.Random(args.seed)
    output = sys.stdout if args.stdout or args.stream else None
    if output is not None:
        index = 0
        try:
            while args.stream or index < args.samples:
                print(json.dumps(build_v7_record(index, config, rng), separators=(",", ":")), flush=True)
                index += 1
        except BrokenPipeError:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for index in range(args.samples):
            stream.write(json.dumps(build_v7_record(index, config, rng), separators=(",", ":")) + "\n")
    print(f"wrote {args.samples} schema-v7 records to {args.out}")


if __name__ == "__main__":
    main()
