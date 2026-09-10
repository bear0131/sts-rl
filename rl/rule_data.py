"""Data contract and tokenization for the standalone combat rule model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from env.core.rule_language import (
    RULE_CARD_IDS,
    RULE_LANGUAGE_VERSION,
    card_rule_sentence,
    relic_rule_sentence,
)


RULE_DATA_VERSION = 5
SUPPORTED_RULE_DATA_VERSIONS = (1, 2, 3, 4, RULE_DATA_VERSION)
PAD = "<PAD>"
UNK = "<UNK>"
OUTPUT_BEGIN = "<OUTPUT_BEGIN>"
OUTPUT_END = "<OUTPUT_END>"
EFFECTS_BEGIN = "<EFFECTS_BEGIN>"
EFFECTS_END = "<EFFECTS_END>"
CALCULATIONS_BEGIN = "<CALCULATIONS_BEGIN>"
CALCULATIONS_END = "<CALCULATIONS_END>"
RULE_STEPS_BEGIN = "<RULE_STEPS_BEGIN>"
RULE_STEPS_END = "<RULE_STEPS_END>"
NEXT_STATE_BEGIN = "<NEXT_STATE_BEGIN>"
NEXT_STATE_END = "<NEXT_STATE_END>"
SPECIAL_TOKENS = (PAD, UNK, OUTPUT_BEGIN, OUTPUT_END)
NUMERIC_TOKENS = (
    "SIGN_NEG", *(f"DIGIT_{digit}" for digit in range(10))
)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.ASCII)
_CARD_ID_PIECES = tuple(sorted(
    ((tuple(_WORD_RE.findall(card_id)), card_id) for card_id in RULE_CARD_IDS),
    key=lambda item: len(item[0]),
    reverse=True,
))
def _atom(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_") or "EMPTY"


LEGACY_CARD_ALIAS_TOKENS = frozenset(
    token
    for card_id in RULE_CARD_IDS
    for token in (f"WORD_{_atom(card_id)}", f"VALUE_{_atom(card_id)}")
)

CANONICAL_CARD_TOKENS = frozenset(
    f"CARD_{_atom(card_id)}" for card_id in RULE_CARD_IDS
)


def _number(value: int | float, *, least_significant_first: bool = False) -> list[str]:
    if isinstance(value, float) and not value.is_integer():
        text = format(value, ".8g")
    else:
        text = str(int(value))
    sign = text.startswith("-")
    if sign:
        text = text[1:]
    if least_significant_first:
        if "." in text:
            raise ValueError("least-significant-first numbers must be integers")
        text = text[::-1]
    out = ["NUM_LSD_BEGIN" if least_significant_first else "NUM_BEGIN"]
    if sign:
        out.append("SIGN_NEG")
    for char in text:
        out.append({".": "DECIMAL"}.get(char, f"DIGIT_{char}"))
    out.append("NUM_END")
    return out


def _sentence_tokens(
    sentence: str, *, canonical_card_tokens: bool = True,
    card_ref_map: dict[str, str] | None = None,
) -> list[str]:
    if not canonical_card_tokens:
        return [f"WORD_{_atom(piece)}" for piece in _WORD_RE.findall(sentence)]
    pieces = _WORD_RE.findall(sentence)
    out: list[str] = []
    position = 0
    while position < len(pieces):
        piece = pieces[position]
        out.append(f"WORD_{_atom(piece)}")
        position += 1
        if piece != "CARD":
            continue
        for card_pieces, card_id in _CARD_ID_PIECES:
            stop = position + len(card_pieces)
            if tuple(pieces[position:stop]) == card_pieces:
                out.append(
                    card_ref_map.get(card_id, f"CARD_{_atom(card_id)}")
                    if card_ref_map is not None else f"CARD_{_atom(card_id)}"
                )
                position = stop
                break
    return out


def _card_tokens(
    card: dict[str, Any], *, least_significant_first: bool = False,
    card_ref_map: dict[str, str] | None = None,
) -> list[str]:
    card_id = str(card["id"])
    identity = (
        card_ref_map.get(card_id, f"CARD_{_atom(card_id)}")
        if card_ref_map is not None else f"CARD_{_atom(card_id)}"
    )
    out = [
        "CARD", identity, "UP",
        *_number(card.get("up", 0), least_significant_first=least_significant_first),
    ]
    if "cost" in card:
        out.extend((
            "COST",
            *_number(card["cost"], least_significant_first=least_significant_first),
        ))
    out.append("CARD_END")
    return out


def serialize_state(
    state: dict[str, Any], *, include_rules: bool,
    least_significant_first: bool = False,
    canonical_card_tokens: bool = True,
    card_ref_map: dict[str, str] | None = None,
) -> list[str]:
    """Serialize the canonical combat+run context without free-form JSON order."""
    combat = state["combat"]
    global_state = state["global"]
    player = combat["player"]
    out = ["STATE_BEGIN"]
    for field in ("ascension", "act", "floor"):
        out.extend((
            f"GLOBAL_{field.upper()}",
            *_number(
                global_state[field],
                least_significant_first=least_significant_first,
            ),
        ))
    out.extend((
        "TURN",
        *_number(combat["turn"], least_significant_first=least_significant_first),
        "PLAYER_BEGIN",
    ))
    for field in ("hp", "max_hp", "block", "energy", "gold"):
        out.extend((
            field.upper(),
            *_number(
                player.get(field, 0),
                least_significant_first=least_significant_first,
            ),
        ))
    for name, value in sorted(player.get("powers", {}).items()):
        out.extend((
            "PLAYER_STATUS", f"STATUS_{_atom(name)}",
            *_number(value, least_significant_first=least_significant_first),
        ))
    for name, value in sorted(player.get("combat_counters", {}).items()):
        out.extend((
            "COUNTER", f"COUNTER_{_atom(name)}",
            *_number(value, least_significant_first=least_significant_first),
        ))
    out.append("PLAYER_END")

    for slot, monster in enumerate(combat["monsters"]):
        out.extend(("MONSTER_BEGIN", f"SLOT_{slot}", f"MONSTER_{_atom(monster['id'])}"))
        for field in ("hp", "max_hp", "block", "base_damage", "adjusted_damage", "hits"):
            out.extend((
                field.upper(),
                *_number(
                    monster.get(field, 0),
                    least_significant_first=least_significant_first,
                ),
            ))
        out.extend(("GONE", f"BOOL_{int(bool(monster.get('gone', False)))}"))
        out.extend(("MOVE", f"MOVE_{_atom(monster.get('move', 'INVALID'))}"))
        for name, value in sorted(monster.get("powers", {}).items()):
            out.extend((
                "MONSTER_STATUS", f"STATUS_{_atom(name)}",
                *_number(value, least_significant_first=least_significant_first),
            ))
        out.append("MONSTER_END")

    for zone in ("hand", "draw", "discard", "exhaust"):
        out.append(f"ZONE_{zone.upper()}_BEGIN")
        for card in combat[zone]:
            out.extend(_card_tokens(
                card, least_significant_first=least_significant_first,
                card_ref_map=card_ref_map,
            ))
        out.append(f"ZONE_{zone.upper()}_END")

    relics = list(global_state.get("relics", []))
    counters = list(global_state.get("relic_counters", []))
    for idx, relic_id in enumerate(relics):
        out.extend(("RELIC", f"RELIC_{_atom(relic_id)}", "RELIC_COUNTER"))
        out.extend(_number(
            counters[idx] if idx < len(counters) else 0,
            least_significant_first=least_significant_first,
        ))
        if include_rules:
            out.append("RULE_BEGIN")
            out.extend(_sentence_tokens(
                relic_rule_sentence(relic_id),
                canonical_card_tokens=canonical_card_tokens,
                card_ref_map=card_ref_map,
            ))
            out.append("RULE_END")

    if include_rules:
        versions = {
            (str(card_id), int(upgrades))
            for card_id, upgrades in global_state.get("deck", [])
        }
        for card_id, upgrades in sorted(versions):
            identity = (
                card_ref_map.get(card_id, f"CARD_{_atom(card_id)}")
                if card_ref_map is not None else f"CARD_{_atom(card_id)}"
            )
            out.extend(("CARD_RULE", identity, "RULE_BEGIN"))
            out.extend(_sentence_tokens(
                card_rule_sentence(card_id, upgrades),
                canonical_card_tokens=canonical_card_tokens,
                card_ref_map=card_ref_map,
            ))
            out.append("RULE_END")
    out.append("STATE_END")
    return out


def serialize_action(
    action: dict[str, Any], *, canonical_card_tokens: bool = True,
    card_ref_map: dict[str, str] | None = None,
) -> list[str]:
    out = ["ACTION_BEGIN", f"ACTION_{_atom(action['type'])}"]
    for key in ("source", "target", "selected"):
        if key not in action:
            continue
        value = action[key]
        out.append(f"ACTION_FIELD_{key.upper()}")
        values = value if isinstance(value, list) else [value]
        for item in values:
            out.extend(_number(item))
    card = action.get("card")
    if card is not None:
        out.extend(_card_tokens(card, card_ref_map=card_ref_map))
        out.append("ACTION_RULE_BEGIN")
        out.extend(_sentence_tokens(
            card_rule_sentence(card["id"], card.get("up", 0)),
            canonical_card_tokens=canonical_card_tokens,
            card_ref_map=card_ref_map,
        ))
        out.append("ACTION_RULE_END")
    if "potion" in action:
        out.append(f"POTION_{_atom(action['potion'])}")
    if "select_context" in action:
        out.append(f"SELECT_CONTEXT_{_atom(action['select_context'])}")
    out.append("ACTION_END")
    return out


def _serialize_value(
    value: Any, *, least_significant_first: bool = False,
    card_identity: bool = False,
    card_ref_map: dict[str, str] | None = None,
) -> list[str]:
    if isinstance(value, bool):
        return [f"BOOL_{int(value)}"]
    if isinstance(value, (int, float)):
        return _number(value, least_significant_first=least_significant_first)
    if value is None:
        return ["NONE"]
    if isinstance(value, list):
        out = ["LIST_BEGIN"]
        for item in value:
            out.extend(_serialize_value(
                item, least_significant_first=least_significant_first,
                card_identity=card_identity,
                card_ref_map=card_ref_map,
            ))
        out.append("LIST_END")
        return out
    if card_identity:
        card_id = str(value)
        return [card_ref_map.get(card_id, f"CARD_{_atom(card_id)}")
                if card_ref_map is not None else f"CARD_{_atom(card_id)}"]
    return [f"VALUE_{_atom(value)}"]


def serialize_effects(
    effects: Sequence[dict[str, Any]], *,
    rule_steps: Sequence[dict[str, Any]] | None = None,
    local_calculations: Sequence[Sequence[dict[str, Any]]] | None = None,
    least_significant_first: bool = False,
    canonical_card_tokens: bool = True,
    compact: bool = False,
    card_ref_map: dict[str, str] | None = None,
) -> list[str]:
    """Serialize runtime events with just-in-time arithmetic before outcomes.

    Compact v4 keeps the causal reason and all calculations/outcome values, but
    drops bookkeeping that is recoverable from event order or effect type.
    """
    include_local_process = rule_steps is not None or local_calculations is not None
    if rule_steps is not None and len(rule_steps) != len(effects):
        raise ValueError("rule step count must match effect count")
    if local_calculations is not None and len(local_calculations) != len(effects):
        raise ValueError("local calculation count must match effect count")
    out = [EFFECTS_BEGIN]
    for index, effect in enumerate(effects):
        if compact and effect.get("type") == "ACTION_COMPLETE":
            continue
        out.extend(("EFFECT_BEGIN", f"EFFECT_{_atom(effect['type'])}"))
        if rule_steps is not None:
            step = rule_steps[index]
            if compact:
                out.extend((
                    "STEP_CAUSE", f"VALUE_{_atom(step.get('cause', 'OTHER'))}",
                ))
            else:
                out.append("STEP_INDEX")
                out.extend(_number(
                    step.get("index", index),
                    least_significant_first=least_significant_first,
                ))
                out.extend((
                    "STEP_CAUSE", f"VALUE_{_atom(step.get('cause', 'OTHER'))}",
                    "STEP_RELATION",
                    f"VALUE_{_atom(step.get('relation', 'STATE_TRANSITION'))}",
                    "STEP_CAUSE_STEP",
                ))
                out.extend(_number(
                    step.get("cause_step", -1),
                    least_significant_first=least_significant_first,
                ))
                out.extend((
                    "STEP_EVENT", f"VALUE_{_atom(step.get('event', 'OTHER'))}",
                ))
        for key, value in effect.items():
            if key == "type" or key in ("before", "delta", "after"):
                continue
            if compact and key == "atomic":
                continue
            out.append(f"FIELD_{_atom(key)}")
            out.extend(_serialize_value(
                value, least_significant_first=least_significant_first,
                card_identity=canonical_card_tokens and key == "card",
                card_ref_map=card_ref_map,
            ))
        calculations = (
            local_calculations[index]
            if local_calculations is not None else ()
        )
        if include_local_process and (calculations or not compact):
            out.extend(serialize_calculations(calculations))
        for key in ("before", "delta", "after"):
            if key not in effect:
                continue
            out.append(f"FIELD_{key}")
            out.extend(_serialize_value(
                effect[key], least_significant_first=least_significant_first,
            ))
        out.append("EFFECT_END")
    out.append(EFFECTS_END)
    return out


def serialize_calculations(calculations: Sequence[dict[str, Any]]) -> list[str]:
    """Serialize explicit arithmetic work, with result digits emitted LSD first."""
    out = [CALCULATIONS_BEGIN]
    for calculation in calculations:
        out.extend((
            "CALCULATION_BEGIN",
            f"CALCULATION_{_atom(calculation['type'])}",
        ))
        for key, value in calculation.items():
            if key == "type":
                continue
            out.append(f"CALC_FIELD_{_atom(key)}")
            if isinstance(value, bool):
                out.append(f"BOOL_{int(value)}")
            elif isinstance(value, (int, float)):
                out.extend(_number(value, least_significant_first=True))
            else:
                out.append(f"VALUE_{_atom(value)}")
        out.append("CALCULATION_END")
    out.append(CALCULATIONS_END)
    return out


_STEP_FIELDS = (
    "field", "before", "delta", "after", "slot", "owner", "status",
    "counter", "source", "destination", "card", "up", "atomic_steps",
)


def _status_value(powers: dict[str, Any], name: str) -> int:
    """Read a simulator status without depending on capitalization."""
    wanted = name.casefold()
    return next(
        (int(value) for key, value in powers.items() if str(key).casefold() == wanted),
        0,
    )


def _step_cause(
    effect: dict[str, Any], *, card_step: int | None, attack_step: int | None,
    monster_thorns: bool,
) -> tuple[str, str, int]:
    event_type = str(effect.get("type", "OTHER"))
    field = str(effect.get("field", ""))
    if event_type == "CARD_REMOVE":
        return "CARD_PLAY", "CARD_RULE", -1
    if event_type == "MONSTER_VALUE_CHANGE" and field == "hp":
        return "ATTACK_HIT", "CARD_RULE", -1 if card_step is None else card_step
    if (
        event_type == "PLAYER_VALUE_CHANGE" and field == "hp"
        and int(effect.get("delta", 0)) < 0 and monster_thorns
    ):
        return "THORNS_TRIGGER", "ATTACK_TRIGGER", -1 if attack_step is None else attack_step
    if event_type == "PLAYER_VALUE_CHANGE" and field == "energy":
        return "CARD_COST", "CARD_RULE", -1 if card_step is None else card_step
    if event_type == "COUNTER_CHANGE":
        counter = str(effect.get("counter", ""))
        relic_counters = {"pen_nib", "nunchaku", "sundial", "happy_flower", "incense_burner"}
        if counter in relic_counters:
            return "RELIC_TRIGGER", "RELIC_RULE", -1 if card_step is None else card_step
        return "CARD_COUNTER", "CARD_RULE", -1 if card_step is None else card_step
    if event_type == "STATUS_CHANGE":
        return "STATUS_RULE", "CARD_RULE", -1 if card_step is None else card_step
    if event_type in ("CARD_ADD", "CARD_MOVE", "CARD_REMOVE"):
        return "CARD_ZONE", "CARD_RULE", -1 if card_step is None else card_step
    if event_type == "ACTION_COMPLETE":
        return "ACTION_COMPLETE", "ACTION_SEQUENCE", -1 if attack_step is None else attack_step
    return "STATE_UPDATE", "STATE_TRANSITION", -1 if attack_step is None else attack_step


def derive_rule_steps(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn simulator events into explicit causal, ordered supervision steps."""
    effects = list(sample.get("effects", ()))
    monsters = sample.get("state", {}).get("combat", {}).get("monsters", ())
    monster_thorns = any(
        _status_value(monster.get("powers", {}), "Thorns") > 0
        for monster in monsters
    )
    steps: list[dict[str, Any]] = []
    card_step: int | None = next(
        (index for index, effect in enumerate(effects)
         if effect.get("type") == "CARD_REMOVE"),
        None,
    )
    attack_step: int | None = None
    for index, effect in enumerate(effects):
        cause, relation, cause_step = _step_cause(
            effect, card_step=card_step, attack_step=attack_step,
            monster_thorns=monster_thorns,
        )
        step = {
            "index": index,
            "atomic": int(effect.get("atomic", effect.get("atomic_steps", 0))),
            "cause": cause,
            "relation": relation,
            "cause_step": cause_step,
            "event": str(effect.get("type", "OTHER")),
        }
        for key in _STEP_FIELDS:
            if key in effect:
                step[key] = effect[key]
        steps.append(step)
        if effect.get("type") == "MONSTER_VALUE_CHANGE" and effect.get("field") == "hp":
            attack_step = index
    return steps


def _first_damage_number(sentence: str) -> int | None:
    pieces = sentence.split()
    try:
        position = pieces.index("DEAL_DAMAGE") + 1
    except ValueError:
        return None
    if position >= len(pieces) or pieces[position] != "NUM_BEGIN":
        return None
    digits: list[str] = []
    position += 1
    while position < len(pieces) and pieces[position].startswith("DIGIT_"):
        digits.append(pieces[position][6:])
        position += 1
    if not digits or position >= len(pieces) or pieces[position] != "NUM_END":
        return None
    return int("".join(digits))


def derive_calculations(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize arithmetic for every numeric transition in a simulator trace.

    The generic transition equations cover all effects.  When the action is a
    card with a numeric damage rule, the base-damage and Strength readings are
    added as a compositional hint rather than relying on a card-specific table.
    """
    calculations: list[dict[str, Any]] = []
    action = sample.get("action", {})
    card = action.get("card")
    player = sample.get("state", {}).get("combat", {}).get("player", {})
    if card is not None:
        try:
            rule = card_rule_sentence(card["id"], int(card.get("up", 0)))
        except KeyError:
            rule = ""
        base_damage = _first_damage_number(rule)
        if base_damage is not None:
            strength = _status_value(player.get("powers", {}), "Strength")
            calculations.extend((
                {"type": "READ_BASE_DAMAGE", "value": base_damage},
                {"type": "READ_STRENGTH", "value": strength},
                {"type": "ADD_STRENGTH", "left": base_damage,
                 "right": strength, "result": base_damage + strength},
            ))

    attack_hits = 0
    for effect in sample.get("effects", ()):
        before = effect.get("before")
        delta = effect.get("delta")
        after = effect.get("after")
        if isinstance(before, (int, float)) and isinstance(delta, (int, float)) and isinstance(after, (int, float)):
            calculations.append({
                "type": "APPLY_DELTA", "field": effect.get("field", effect.get("counter", "value")),
                "left": before, "delta": delta, "result": after,
            })
        if effect.get("type") == "MONSTER_VALUE_CHANGE" and effect.get("field") == "hp":
            attack_hits += 1
            damage = max(0, -int(delta or 0))
            calculations.extend((
                {"type": "READ_TARGET_HP", "value": int(before)},
                {"type": "SUBTRACT_HP", "left": int(before),
                 "right": damage, "result": int(after)},
            ))
        if effect.get("type") == "PLAYER_VALUE_CHANGE" and effect.get("field") == "hp" and int(delta or 0) < 0:
            calculations.extend((
                {"type": "READ_TRIGGER_DAMAGE", "value": abs(int(delta))},
                {"type": "SUBTRACT_PLAYER_HP", "left": int(before),
                 "right": abs(int(delta)), "result": int(after)},
            ))
    if attack_hits:
        calculations.append({"type": "COUNT_ATTACK_HITS", "value": attack_hits})
    return calculations


def serialize_rule_steps(
    steps: Sequence[dict[str, Any]], *, least_significant_first: bool = False,
    canonical_card_tokens: bool = True,
    card_ref_map: dict[str, str] | None = None,
) -> list[str]:
    """Serialize ordered rule steps, including their causal parent step."""
    out = [RULE_STEPS_BEGIN]
    for step in steps:
        out.extend(("STEP_BEGIN", "STEP_INDEX"))
        out.extend(_number(step.get("index", 0), least_significant_first=least_significant_first))
        out.extend(("STEP_ATOMIC",))
        out.extend(_number(step.get("atomic", 0), least_significant_first=least_significant_first))
        out.extend(("STEP_CAUSE", f"VALUE_{_atom(step.get('cause', 'OTHER'))}"))
        out.extend(("STEP_RELATION", f"VALUE_{_atom(step.get('relation', 'STATE_TRANSITION'))}"))
        out.extend(("STEP_CAUSE_STEP",))
        out.extend(_number(step.get("cause_step", -1), least_significant_first=least_significant_first))
        out.extend(("STEP_EVENT", f"VALUE_{_atom(step.get('event', 'OTHER'))}"))
        for key in _STEP_FIELDS:
            if key not in step:
                continue
            out.append(f"STEP_{_atom(key).upper()}")
            out.extend(_serialize_value(
                step[key], least_significant_first=least_significant_first,
                card_identity=canonical_card_tokens and key == "card",
                card_ref_map=card_ref_map,
            ))
        out.append("STEP_END")
    out.append(RULE_STEPS_END)
    return out


def controlled_attack_calculations(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize the controlled Strength experiment's complete damage formula."""
    metadata = sample.get("metadata", {})
    if metadata.get("experiment") not in (
        "random_strength_strike_bash",
        "random_strength_enemy_status_strike_bash",
    ):
        return []
    card_id = metadata["card"]
    base_damage = 6 if card_id == "Strike_R" else 8
    strength = int(metadata["strength"])
    strengthened = base_damage + strength
    vulnerable = int(metadata.get("enemy_vulnerable", 0))
    attack_damage = int(
        metadata["calculated_attack_damage"]
        if "calculated_attack_damage" in metadata
        else metadata["expected_damage"]
    )
    block = int(metadata.get("enemy_block", 0))
    block_loss = int(metadata.get("expected_block_loss", 0))
    hp_loss = int(
        metadata["expected_hp_loss"]
        if "expected_hp_loss" in metadata
        else metadata["expected_damage"]
    )
    before_monster = sample["state"]["combat"]["monsters"][0]
    next_monster = sample["next_state"]["combat"]["monsters"][0]
    calculations = [
        {"type": "READ_BASE_DAMAGE", "value": base_damage},
        {"type": "READ_STRENGTH", "value": strength},
        {
            "type": "ADD_STRENGTH",
            "left": base_damage,
            "right": strength,
            "result": strengthened,
        },
    ]
    if vulnerable:
        vulnerable_numerator = strengthened * 3
        calculations.extend((
            {"type": "READ_VULNERABLE", "value": vulnerable},
            {
                "type": "MULTIPLY_VULNERABLE",
                "left": strengthened,
                "right": 3,
                "result": vulnerable_numerator,
            },
            {
                "type": "FLOOR_DIVIDE_VULNERABLE",
                "left": vulnerable_numerator,
                "right": 2,
                "result": attack_damage,
            },
        ))
    else:
        calculations.append({
            "type": "NO_VULNERABLE", "input": strengthened, "result": attack_damage,
        })
    calculations.extend((
        {"type": "READ_TARGET_BLOCK", "value": block},
        {
            "type": "BLOCK_ABSORB",
            "left": attack_damage,
            "right": block,
            "result": block_loss,
        },
        {
            "type": "SUBTRACT_BLOCK_DAMAGE",
            "left": attack_damage,
            "right": block_loss,
            "result": hp_loss,
        },
        {
            "type": "SUBTRACT_BLOCK",
            "left": block,
            "right": block_loss,
            "result": int(next_monster["block"]),
        },
        {"type": "READ_TARGET_HP", "value": int(before_monster["hp"])},
        {
            "type": "SUBTRACT_HP",
            "left": int(before_monster["hp"]),
            "right": hp_loss,
            "result": int(next_monster["hp"]),
        },
    ))
    if card_id == "Bash":
        next_vulnerable = int(
            next_monster.get("powers", {}).get("Vulnerable", 0)
        )
        calculations.append({
            "type": "ADD_VULNERABLE",
            "left": vulnerable,
            "right": next_vulnerable - vulnerable,
            "result": next_vulnerable,
        })
    return calculations


def _numeric_transition_calculation(effect: dict[str, Any]) -> list[dict[str, Any]]:
    before = effect.get("before")
    after = effect.get("after")
    if (
        isinstance(before, (int, float)) and not isinstance(before, bool)
        and isinstance(after, (int, float)) and not isinstance(after, bool)
    ):
        delta = effect.get("delta", after - before)
        if isinstance(delta, (int, float)) and not isinstance(delta, bool):
            return [{
                "type": "APPLY_DELTA",
                "field": effect.get(
                    "field", effect.get("counter", effect.get("status", "value"))
                ),
                "left": before,
                "delta": delta,
                "result": after,
            }]
    return []


def _attack_prefix_calculations(
    sample: dict[str, Any], slot: int,
) -> tuple[int, list[dict[str, Any]]] | None:
    card = sample.get("action", {}).get("card")
    monsters = sample.get("state", {}).get("combat", {}).get("monsters", ())
    if card is None or slot < 0 or slot >= len(monsters):
        return None
    try:
        rule = card_rule_sentence(card["id"], int(card.get("up", 0)))
    except KeyError:
        return None
    base_damage = _first_damage_number(rule)
    if base_damage is None:
        return None
    player = sample["state"]["combat"]["player"]
    strength = _status_value(player.get("powers", {}), "Strength")
    strengthened = base_damage + strength
    calculations = [
        {"type": "READ_BASE_DAMAGE", "value": base_damage},
        {"type": "READ_STRENGTH", "value": strength},
        {
            "type": "ADD_STRENGTH", "left": base_damage,
            "right": strength, "result": strengthened,
        },
    ]
    vulnerable = _status_value(monsters[slot].get("powers", {}), "Vulnerable")
    if vulnerable > 0:
        paper_frog = "Paper Frog" in sample["state"]["global"].get("relics", ())
        numerator, denominator = (7, 4) if paper_frog else (3, 2)
        multiplied = strengthened * numerator
        damage = multiplied // denominator
        calculations.extend((
            {"type": "READ_VULNERABLE", "value": vulnerable},
            {
                "type": "MULTIPLY_VULNERABLE", "left": strengthened,
                "right": numerator, "result": multiplied,
            },
            {
                "type": "FLOOR_DIVIDE_VULNERABLE", "left": multiplied,
                "right": denominator, "result": damage,
            },
        ))
    else:
        damage = strengthened
        calculations.append({
            "type": "NO_VULNERABLE", "input": strengthened, "result": damage,
        })
    return damage, calculations


def _linked_effect_index(
    effects: Sequence[dict[str, Any]], index: int, field: str,
) -> int | None:
    effect = effects[index]
    for candidate_index, candidate in enumerate(effects):
        if candidate_index == index:
            continue
        if (
            candidate.get("type") == "MONSTER_VALUE_CHANGE"
            and candidate.get("field") == field
            and int(candidate.get("slot", -1)) == int(effect.get("slot", -2))
            and candidate.get("atomic") == effect.get("atomic")
        ):
            return candidate_index
    return None


def _detailed_hp_calculations(
    sample: dict[str, Any], effects: Sequence[dict[str, Any]], index: int,
) -> list[dict[str, Any]]:
    effect = effects[index]
    slot = int(effect.get("slot", -1))
    prefix = _attack_prefix_calculations(sample, slot)
    if prefix is None:
        return []
    damage, calculations = prefix
    block_index = _linked_effect_index(effects, index, "block")
    block_effect = effects[block_index] if block_index is not None else None
    block_before = int(block_effect["before"]) if block_effect is not None else 0
    block_loss = max(0, -int(block_effect.get("delta", 0))) \
        if block_effect is not None else 0
    hp_before = int(effect.get("before", 0))
    hp_after = int(effect.get("after", hp_before))
    hp_loss = max(0, -int(effect.get("delta", hp_after - hp_before)))
    if (
        min(damage, block_before) != block_loss
        or damage - block_loss != hp_loss
        or hp_before - hp_loss != hp_after
    ):
        return []
    calculations.extend((
        {"type": "READ_TARGET_BLOCK", "value": block_before},
        {
            "type": "BLOCK_ABSORB", "left": damage,
            "right": block_before, "result": block_loss,
        },
        {
            "type": "SUBTRACT_BLOCK", "left": block_before,
            "right": block_loss, "result": block_before - block_loss,
        },
        {
            "type": "SUBTRACT_BLOCK_DAMAGE", "left": damage,
            "right": block_loss, "result": hp_loss,
        },
        {"type": "READ_TARGET_HP", "value": hp_before},
        {
            "type": "SUBTRACT_HP", "left": hp_before,
            "right": hp_loss, "result": hp_after,
        },
    ))
    return calculations


def _detailed_block_calculations(
    sample: dict[str, Any], effects: Sequence[dict[str, Any]], index: int,
) -> list[dict[str, Any]]:
    hp_index = _linked_effect_index(effects, index, "hp")
    if hp_index is not None:
        calculations = _detailed_hp_calculations(sample, effects, hp_index)
        for position, calculation in enumerate(calculations):
            if calculation["type"] == "SUBTRACT_BLOCK":
                return calculations[:position + 1]
        return []
    effect = effects[index]
    prefix = _attack_prefix_calculations(sample, int(effect.get("slot", -1)))
    if prefix is None:
        return []
    damage, calculations = prefix
    before = int(effect.get("before", 0))
    after = int(effect.get("after", before))
    absorbed = max(0, -int(effect.get("delta", after - before)))
    if min(damage, before) != absorbed or before - absorbed != after:
        return []
    calculations.extend((
        {"type": "READ_TARGET_BLOCK", "value": before},
        {
            "type": "BLOCK_ABSORB", "left": damage,
            "right": before, "result": absorbed,
        },
        {
            "type": "SUBTRACT_BLOCK", "left": before,
            "right": absorbed, "result": after,
        },
    ))
    return calculations


def _controlled_local_calculations(
    sample: dict[str, Any], effect: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = sample.get("metadata", {})
    if metadata.get("experiment") not in (
        "random_strength_strike_bash",
        "random_strength_enemy_status_strike_bash",
    ):
        return []
    calculations = sample.get("calculations") or controlled_attack_calculations(sample)
    terminal = None
    if effect.get("type") == "MONSTER_VALUE_CHANGE" and effect.get("field") == "hp":
        terminal = "SUBTRACT_HP"
    elif effect.get("type") == "MONSTER_VALUE_CHANGE" and effect.get("field") == "block":
        terminal = "SUBTRACT_BLOCK"
    elif effect.get("type") == "STATUS_CHANGE" and effect.get("status") == "Vulnerable":
        terminal = "ADD_VULNERABLE"
    if terminal is None:
        return []
    for position, calculation in enumerate(calculations):
        if calculation.get("type") != terminal:
            continue
        result = calculation.get("result")
        if "after" not in effect or result == effect.get("after"):
            return [dict(item) for item in calculations[:position + 1]]
    return []


def derive_local_calculations(
    sample: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    """Derive one calculation block per event, immediately before its outcome."""
    effects = list(sample.get("effects", ()))
    local: list[list[dict[str, Any]]] = []
    for index, effect in enumerate(effects):
        calculations = _controlled_local_calculations(sample, effect)
        if not calculations and effect.get("type") == "MONSTER_VALUE_CHANGE":
            if effect.get("field") == "hp":
                calculations = _detailed_hp_calculations(sample, effects, index)
            elif effect.get("field") == "block":
                calculations = _detailed_block_calculations(sample, effects, index)
        if not calculations:
            calculations = _numeric_transition_calculation(effect)
        local.append(calculations)
    return local


def _collect_state_card_ids(state: dict[str, Any], output: set[str]) -> None:
    combat = state.get("combat", {})
    for zone in ("hand", "draw", "discard", "exhaust"):
        for card in combat.get(zone, ()):
            if isinstance(card, dict) and "id" in card:
                output.add(str(card["id"]))
    for entry in state.get("global", {}).get("deck", ()):
        if isinstance(entry, dict) and "id" in entry:
            output.add(str(entry["id"]))
        elif isinstance(entry, (list, tuple)) and entry:
            output.add(str(entry[0]))


def anonymous_card_ref_map(sample: dict[str, Any]) -> dict[str, str]:
    """Return a stable per-sample permutation of opaque card references.

    The permutation is deliberately seeded from sample metadata and the complete
    set of cards present in the trace, so aliases cannot become a global lookup
    table while remaining identical in state, rules, action, effects and target.
    """
    card_ids: set[str] = set()
    _collect_state_card_ids(sample.get("state", {}), card_ids)
    _collect_state_card_ids(sample.get("next_state", {}), card_ids)
    action_card = sample.get("action", {}).get("card")
    if isinstance(action_card, dict) and "id" in action_card:
        card_ids.add(str(action_card["id"]))
    for effect in sample.get("effects", ()):
        if not isinstance(effect, dict):
            continue
        value = effect.get("card")
        if isinstance(value, str):
            card_ids.add(value)
        elif isinstance(value, dict) and "id" in value:
            card_ids.add(str(value["id"]))
    # Deck rule text is the authoritative source for versions that are not in
    # the currently visible zones.  Only names following the generic CARD
    # marker are accepted, avoiding relic/monster words in prose.
    for sentence in (
        sample.get("rules", {}).get("deck", ""),
        sample.get("rules", {}).get("relics", ""),
    ):
        pieces = _WORD_RE.findall(str(sentence))
        for position, piece in enumerate(pieces):
            if piece != "CARD":
                continue
            for card_pieces, card_id in _CARD_ID_PIECES:
                if tuple(pieces[position + 1:position + 1 + len(card_pieces)]) == card_pieces:
                    card_ids.add(card_id)
                    break
    if not card_ids:
        return {}
    metadata = sample.get("metadata", {})
    action = sample.get("action", {})
    seed_text = json.dumps(
        {"metadata": metadata, "action": action, "cards": sorted(card_ids)},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    ranked = sorted(
        card_ids,
        key=lambda card_id: hashlib.sha256(
            f"{seed_text}|{card_id}".encode("utf-8")
        ).digest(),
    )
    return {card_id: f"CARD_REF_{index}" for index, card_id in enumerate(ranked)}


def sample_tokens(
    sample: dict[str, Any], *, output_version: int | None = None,
) -> tuple[list[str], list[str]]:
    if output_version is None:
        # Synthetic pre-versioned fixtures historically exercised v4. Collected
        # data carries its explicit version and new data is v5 anonymous.
        output_version = int(sample.get("data_version", 4))
    if output_version not in (1, 2, 3, 4, RULE_DATA_VERSION):
        raise ValueError(f"unsupported output schema version {output_version}")
    anonymous = output_version >= 5
    canonical_card_tokens = output_version >= 3
    card_ref_map = anonymous_card_ref_map(sample) if anonymous else None
    rules = sample.get("rules")
    if rules is None:
        # Compatibility for early v1 samples collected before rule text was
        # materialized in JSONL.
        prefix = serialize_state(
            sample["state"], include_rules=True,
            canonical_card_tokens=canonical_card_tokens,
            card_ref_map=card_ref_map,
        )
    else:
        prefix = serialize_state(
            sample["state"], include_rules=False,
            canonical_card_tokens=canonical_card_tokens,
            card_ref_map=card_ref_map,
        )
        prefix.extend(("RULE_CONTEXT_BEGIN", "DECK_RULE_BEGIN"))
        prefix.extend(_sentence_tokens(
            rules["deck"], canonical_card_tokens=canonical_card_tokens,
            card_ref_map=card_ref_map,
        ))
        prefix.extend(("DECK_RULE_END", "RELIC_RULE_BEGIN"))
        prefix.extend(_sentence_tokens(
            rules["relics"], canonical_card_tokens=canonical_card_tokens,
            card_ref_map=card_ref_map,
        ))
        prefix.extend(("RELIC_RULE_END", "RULE_CONTEXT_END"))
    prefix.extend(serialize_action(
        sample["action"], canonical_card_tokens=canonical_card_tokens,
        card_ref_map=card_ref_map,
    ))
    target = [OUTPUT_BEGIN]
    rule_steps = sample.get("rule_steps")
    if rule_steps is None:
        rule_steps = derive_rule_steps(sample)
    if output_version < 3:
        target.extend(serialize_rule_steps(
            rule_steps, least_significant_first=True,
            canonical_card_tokens=False,
            card_ref_map=card_ref_map,
        ))
        calculations = sample.get("calculations")
        if calculations is None:
            calculations = controlled_attack_calculations(sample)
        if not calculations:
            calculations = derive_calculations(sample)
        target.extend(serialize_calculations(calculations))
        target.extend(serialize_effects(
            sample["effects"], least_significant_first=True,
            canonical_card_tokens=False,
            card_ref_map=card_ref_map,
        ))
        target.append(NEXT_STATE_BEGIN)
        target.extend(serialize_state(
            sample["next_state"], include_rules=False,
            least_significant_first=True,
            canonical_card_tokens=False,
        ))
        target.extend((NEXT_STATE_END, OUTPUT_END))
        return prefix, target
    local_calculations = sample.get("local_calculations")
    if local_calculations is None:
        local_calculations = derive_local_calculations(sample)
    target.extend(serialize_effects(
        sample["effects"], rule_steps=rule_steps,
        local_calculations=local_calculations,
        least_significant_first=True,
            canonical_card_tokens=True,
            compact=output_version >= 4,
            card_ref_map=card_ref_map,
    ))
    target.append(NEXT_STATE_BEGIN)
    target.extend(serialize_state(
        sample["next_state"], include_rules=False,
        least_significant_first=True,
        card_ref_map=card_ref_map,
    ))
    target.extend((NEXT_STATE_END, OUTPUT_END))
    return prefix, target


CRITICAL_NONE = 0
CRITICAL_EFFECT = 1
CRITICAL_NEXT_STATE = 2
CRITICAL_CALCULATION = 3
CRITICAL_STEP = 4
CRITICAL_CAUSAL = 5
CRITICAL_COUNTER = 6
CRITICAL_STRUCTURE = 7


_STRUCTURAL_MARKERS = frozenset({
    OUTPUT_BEGIN, OUTPUT_END,
    EFFECTS_BEGIN, EFFECTS_END,
    CALCULATIONS_BEGIN, CALCULATIONS_END,
    NEXT_STATE_BEGIN, NEXT_STATE_END,
    RULE_STEPS_BEGIN, RULE_STEPS_END,
    "EFFECT_BEGIN", "EFFECT_END",
    "CALCULATION_BEGIN", "CALCULATION_END",
    "STATE_BEGIN", "STATE_END",
    "PLAYER_BEGIN", "PLAYER_END",
    "MONSTER_BEGIN", "MONSTER_END",
    "CARD", "CARD_END", "COUNTER",
})


def _is_structural_token(token: str) -> bool:
    """Identify grammar/schema tokens whose choice changes the output shape."""
    return (
        token in _STRUCTURAL_MARKERS
        or token.startswith((
            "EFFECT_", "CALCULATION_", "CALC_FIELD_", "FIELD_",
            "STEP_", "VALUE_", "ZONE_", "CARD_REF_",
        ))
    )


def critical_target_number_regions(tokens: Sequence[str]) -> list[int]:
    """Classify important numeric symbols by effect or next-state section."""
    regions = [CRITICAL_NONE] * len(tokens)

    def mark_number(position: int, stop: int, region: int) -> None:
        if position >= stop or tokens[position] not in ("NUM_BEGIN", "NUM_LSD_BEGIN"):
            return
        position += 1
        while position < stop and tokens[position] != "NUM_END":
            if tokens[position] == "SIGN_NEG" or tokens[position] == "DECIMAL":
                regions[position] = region
            elif tokens[position].startswith("DIGIT_"):
                regions[position] = region
            position += 1

    try:
        steps_begin = tokens.index(RULE_STEPS_BEGIN)
        steps_end = tokens.index(RULE_STEPS_END, steps_begin)
    except ValueError:
        steps_begin = steps_end = 0
    position = steps_begin
    while position < steps_end:
        if tokens[position] in ("NUM_BEGIN", "NUM_LSD_BEGIN"):
            mark_number(position, steps_end, CRITICAL_STEP)
        position += 1

    position = 0
    while position < len(tokens):
        try:
            calculations_begin = tokens.index(CALCULATIONS_BEGIN, position)
            calculations_end = tokens.index(
                CALCULATIONS_END, calculations_begin
            )
        except ValueError:
            break
        cursor = calculations_begin
        while cursor < calculations_end:
            if tokens[cursor] in ("NUM_BEGIN", "NUM_LSD_BEGIN"):
                mark_number(cursor, calculations_end, CRITICAL_CALCULATION)
            cursor += 1
        position = calculations_end + 1

    try:
        effects_end = tokens.index(EFFECTS_END)
    except ValueError:
        effects_end = 0
    position = 0
    while position < effects_end:
        try:
            event = tokens.index("EFFECT_BEGIN", position, effects_end)
            event_end = tokens.index("EFFECT_END", event, effects_end)
        except ValueError:
            break
        event_tokens = tokens[event:event_end]
        important = (
            "EFFECT_MONSTER_VALUE_CHANGE" in event_tokens
            and (
                "VALUE_hp" in event_tokens
                or "VALUE_block" in event_tokens
            )
        ) or (
            "EFFECT_STATUS_CHANGE" in event_tokens
            and "VALUE_Vulnerable" in event_tokens
        )
        if important:
            for field in ("FIELD_before", "FIELD_delta", "FIELD_after"):
                try:
                    field_position = tokens.index(field, event, event_end)
                except ValueError:
                    continue
                mark_number(field_position + 1, event_end, CRITICAL_EFFECT)
        for marker in ("STEP_INDEX", "STEP_CAUSE_STEP"):
            try:
                marker_position = tokens.index(marker, event, event_end)
            except ValueError:
                continue
            mark_number(marker_position + 1, event_end, CRITICAL_STEP)
        position = event_end + 1

    try:
        state = tokens.index(NEXT_STATE_BEGIN)
        monster = tokens.index("MONSTER_BEGIN", state)
        monster_end = tokens.index("MONSTER_END", monster)
    except ValueError:
        return regions
    for field in ("HP", "BLOCK"):
        try:
            field_position = tokens.index(field, monster, monster_end)
        except ValueError:
            continue
        mark_number(field_position + 1, monster_end, CRITICAL_NEXT_STATE)
    position = monster
    while position < monster_end:
        try:
            status = tokens.index("MONSTER_STATUS", position, monster_end)
        except ValueError:
            break
        if status + 1 < monster_end and tokens[status + 1] == "STATUS_Vulnerable":
            mark_number(status + 2, monster_end, CRITICAL_NEXT_STATE)
        position = status + 1
    return regions


def critical_target_number_mask(tokens: Sequence[str]) -> list[bool]:
    return [region != CRITICAL_NONE for region in critical_target_number_regions(tokens)]


def target_supervision_regions(tokens: Sequence[str]) -> list[int]:
    """Add a separate region for causal labels while retaining number regions."""
    regions = critical_target_number_regions(tokens)
    for position, token in enumerate(tokens):
        if regions[position] == CRITICAL_NONE and _is_structural_token(token):
            regions[position] = CRITICAL_STRUCTURE
    causal_blocks: list[tuple[int, int]] = []
    try:
        begin = tokens.index(RULE_STEPS_BEGIN)
        end = tokens.index(RULE_STEPS_END, begin)
    except ValueError:
        pass
    else:
        position = begin
        while position < end:
            try:
                step = tokens.index("STEP_BEGIN", position, end)
                step_end = tokens.index("STEP_END", step, end)
            except ValueError:
                break
            causal_blocks.append((step, step_end))
            position = step_end + 1
    try:
        effects_end = tokens.index(EFFECTS_END)
    except ValueError:
        effects_end = 0
    position = 0
    while position < effects_end:
        try:
            effect = tokens.index("EFFECT_BEGIN", position, effects_end)
            effect_end = tokens.index("EFFECT_END", effect, effects_end)
        except ValueError:
            break
        causal_blocks.append((effect, effect_end))
        position = effect_end + 1
    for step, step_end in causal_blocks:
        for marker in ("STEP_CAUSE", "STEP_RELATION", "STEP_EVENT"):
            try:
                marker_pos = tokens.index(marker, step, step_end)
            except ValueError:
                continue
            regions[marker_pos] = CRITICAL_CAUSAL
            value_pos = marker_pos + 1
            if value_pos < step_end and tokens[value_pos].startswith("VALUE_"):
                regions[value_pos] = CRITICAL_CAUSAL
        try:
            counter_pos = tokens.index("FIELD_counter", step, step_end) + 1
        except ValueError:
            continue
        if (
            counter_pos < step_end
            and tokens[counter_pos].startswith("VALUE_")
        ):
            regions[counter_pos] = CRITICAL_COUNTER
    return regions


def _token_number(tokens: Sequence[str], position: int, stop: int):
    if position >= stop or tokens[position] not in ("NUM_BEGIN", "NUM_LSD_BEGIN"):
        return None
    least_significant_first = tokens[position] == "NUM_LSD_BEGIN"
    sign = 1
    positions = []
    position += 1
    if position < stop and tokens[position] == "SIGN_NEG":
        sign = -1
        position += 1
    while position < stop and tokens[position].startswith("DIGIT_"):
        positions.append(position)
        position += 1
    if not positions or position >= stop or tokens[position] != "NUM_END":
        return None
    digits = [tokens[idx][6:] for idx in positions]
    if least_significant_first:
        digits.reverse()
    value = sign * int("".join(digits))
    return {
        "positions": tuple(positions), "sign": sign, "value": value,
        "least_significant_first": least_significant_first,
    }


def transition_consistency_specs(
    sample: dict[str, Any], target: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    """Locate output numbers joined by next = before + sum(effect deltas)."""
    deltas: dict[tuple[int, str], list[dict[str, Any]]] = {}
    try:
        effects_end = target.index(EFFECTS_END)
    except ValueError:
        effects_end = 0
    position = 0
    while position < effects_end:
        try:
            event = target.index("EFFECT_BEGIN", position, effects_end)
            event_end = target.index("EFFECT_END", event, effects_end)
        except ValueError:
            break
        event_tokens = target[event:event_end]
        slot = field = None
        if "EFFECT_MONSTER_VALUE_CHANGE" in event_tokens:
            for candidate in ("hp", "block"):
                if f"VALUE_{candidate}" in event_tokens:
                    field = candidate
                    break
            try:
                slot_pos = target.index("FIELD_slot", event, event_end)
                slot_number = _token_number(target, slot_pos + 1, event_end)
                slot = slot_number["value"] if slot_number is not None else None
            except ValueError:
                pass
        elif "EFFECT_STATUS_CHANGE" in event_tokens and "VALUE_Vulnerable" in event_tokens:
            try:
                owner_pos = target.index("FIELD_owner", event, event_end)
                owner = target[owner_pos + 1]
                if owner.startswith("VALUE_monster_"):
                    slot = int(owner.removeprefix("VALUE_monster_"))
                    field = "Vulnerable"
            except (ValueError, IndexError):
                pass
        if slot is not None and field is not None:
            try:
                delta_pos = target.index("FIELD_delta", event, event_end)
                span = _token_number(target, delta_pos + 1, event_end)
            except ValueError:
                span = None
            if span is not None:
                deltas.setdefault((int(slot), field), []).append(span)
        position = event_end + 1

    try:
        state = target.index(NEXT_STATE_BEGIN)
    except ValueError:
        return ()
    specs = []
    monster_position = state
    before_monsters = sample["state"]["combat"]["monsters"]
    while True:
        try:
            monster = target.index("MONSTER_BEGIN", monster_position)
            monster_end = target.index("MONSTER_END", monster)
        except ValueError:
            break
        slot_token = next(
            (token for token in target[monster:monster_end] if token.startswith("SLOT_")),
            None,
        )
        if slot_token is None:
            monster_position = monster_end + 1
            continue
        slot = int(slot_token.removeprefix("SLOT_"))
        if slot >= len(before_monsters):
            monster_position = monster_end + 1
            continue
        before_monster = before_monsters[slot]
        for marker, field in (("HP", "hp"), ("BLOCK", "block")):
            try:
                marker_pos = target.index(marker, monster, monster_end)
                next_span = _token_number(target, marker_pos + 1, monster_end)
            except ValueError:
                next_span = None
            if next_span is not None:
                specs.append({
                    "field": field,
                    "before": int(before_monster.get(field, 0)),
                    "deltas": tuple(deltas.get((slot, field), ())),
                    "next": next_span,
                })
        vulnerable_span = None
        status_position = monster
        while status_position < monster_end:
            try:
                status = target.index("MONSTER_STATUS", status_position, monster_end)
            except ValueError:
                break
            if status + 1 < monster_end and target[status + 1] == "STATUS_Vulnerable":
                vulnerable_span = _token_number(target, status + 2, monster_end)
                break
            status_position = status + 1
        if vulnerable_span is not None:
            specs.append({
                "field": "Vulnerable",
                "before": int(before_monster.get("powers", {}).get("Vulnerable", 0)),
                "deltas": tuple(deltas.get((slot, "Vulnerable"), ())),
                "next": vulnerable_span,
            })
        monster_position = monster_end + 1
    return tuple(specs)


def _card_key(card: dict[str, Any]) -> tuple[str, int]:
    return str(card["id"]), int(card.get("up", 0))


def _status_events(
    owner: str, before: dict[str, int], after: dict[str, int], atomic: int
) -> list[dict[str, Any]]:
    events = []
    for status in sorted(set(before) | set(after)):
        old, new = int(before.get(status, 0)), int(after.get(status, 0))
        if old != new:
            events.append({
                "type": "STATUS_CHANGE", "atomic": atomic, "owner": owner,
                "status": status, "before": old, "delta": new - old, "after": new,
            })
    return events


def _zone_events(before: dict[str, Any], after: dict[str, Any], atomic: int):
    zones = ("hand", "draw", "discard", "exhaust")
    removed: list[tuple[str, tuple[str, int]]] = []
    added: list[tuple[str, tuple[str, int]]] = []
    for zone in zones:
        left = Counter(map(_card_key, before[zone]))
        right = Counter(map(_card_key, after[zone]))
        for card, count in (left - right).items():
            removed.extend((zone, card) for _ in range(count))
        for card, count in (right - left).items():
            added.extend((zone, card) for _ in range(count))

    events = []
    used_added: set[int] = set()
    for source, card in removed:
        destination_idx = next(
            (idx for idx, (zone, other) in enumerate(added)
             if idx not in used_added and other == card),
            None,
        )
        if destination_idx is None:
            events.append({
                "type": "CARD_REMOVE", "atomic": atomic, "source": source,
                "card": card[0], "up": card[1],
            })
            continue
        used_added.add(destination_idx)
        destination = added[destination_idx][0]
        events.append({
            "type": "CARD_MOVE", "atomic": atomic, "source": source,
            "destination": destination, "card": card[0], "up": card[1],
        })
    for idx, (destination, card) in enumerate(added):
        if idx not in used_added:
            events.append({
                "type": "CARD_ADD", "atomic": atomic, "destination": destination,
                "card": card[0], "up": card[1],
            })
    return events


def diff_atomic_states(
    before: dict[str, Any], after: dict[str, Any], atomic: int
) -> list[dict[str, Any]]:
    """Translate one simulator-atomic state transition into ordered changes."""
    events: list[dict[str, Any]] = []
    bp, ap = before["player"], after["player"]
    for field in ("hp", "max_hp", "block", "energy", "gold"):
        old, new = int(bp.get(field, 0)), int(ap.get(field, 0))
        if old != new:
            events.append({
                "type": "PLAYER_VALUE_CHANGE", "atomic": atomic, "field": field,
                "before": old, "delta": new - old, "after": new,
            })
    events.extend(_status_events("player", bp.get("powers", {}), ap.get("powers", {}), atomic))
    for name in sorted(set(bp.get("combat_counters", {})) | set(ap.get("combat_counters", {}))):
        old = bp.get("combat_counters", {}).get(name, 0)
        new = ap.get("combat_counters", {}).get(name, 0)
        if old != new:
            events.append({
                "type": "COUNTER_CHANGE", "atomic": atomic, "counter": name,
                "before": old, "delta": new - old, "after": new,
            })

    for slot, (bm, am) in enumerate(zip(before["monsters"], after["monsters"])):
        for field in ("hp", "max_hp", "block"):
            old, new = int(bm.get(field, 0)), int(am.get(field, 0))
            if old != new:
                events.append({
                    "type": "MONSTER_VALUE_CHANGE", "atomic": atomic, "slot": slot,
                    "field": field, "before": old, "delta": new - old, "after": new,
                })
        events.extend(_status_events(
            f"monster_{slot}", bm.get("powers", {}), am.get("powers", {}), atomic
        ))
        if bm.get("move") != am.get("move"):
            events.append({
                "type": "MONSTER_MOVE_CHANGE", "atomic": atomic, "slot": slot,
                "before": bm.get("move", "INVALID"),
                "after": am.get("move", "INVALID"),
            })
        if bool(bm.get("gone")) != bool(am.get("gone")):
            events.append({
                "type": "MONSTER_GONE_CHANGE", "atomic": atomic, "slot": slot,
                "before": bool(bm.get("gone")), "after": bool(am.get("gone")),
            })
    if len(before["monsters"]) != len(after["monsters"]):
        events.append({
            "type": "MONSTER_COUNT_CHANGE", "atomic": atomic,
            "before": len(before["monsters"]), "after": len(after["monsters"]),
        })
    events.extend(_zone_events(before, after, atomic))
    if before["turn"] != after["turn"]:
        events.append({
            "type": "TURN_CHANGE", "atomic": atomic,
            "before": before["turn"], "after": after["turn"],
        })
    return events


def events_from_snapshots(snapshots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous = snapshots[0]
    atomic = 0
    for current in snapshots[1:]:
        changes = diff_atomic_states(previous, current, atomic)
        if changes:
            events.extend(changes)
            atomic += 1
        previous = current
    events.append({"type": "ACTION_COMPLETE", "atomic_steps": atomic})
    return events


@dataclass(frozen=True)
class RuleTokenizer:
    tokens: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "index", {token: idx for idx, token in enumerate(self.tokens)})

    @classmethod
    def build(cls, samples: Iterable[dict[str, Any]]) -> "RuleTokenizer":
        vocabulary = set(SPECIAL_TOKENS) | set(NUMERIC_TOKENS)
        for sample in samples:
            prefix, target = sample_tokens(sample)
            vocabulary.update(prefix)
            vocabulary.update(target)
        rest = sorted(vocabulary - set(SPECIAL_TOKENS))
        return cls((*SPECIAL_TOKENS, *rest))

    @property
    def pad_id(self) -> int:
        return self.index[PAD]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        unknown = self.index[UNK]
        return [self.index.get(token, unknown) for token in tokens]

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.tokens[int(idx)] for idx in ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"version": RULE_DATA_VERSION, "tokens": self.tokens}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RuleTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["version"] not in (1, 2, 3, 4, RULE_DATA_VERSION):
            raise ValueError(f"unsupported rule tokenizer version {data['version']}")
        return cls(tuple(data["tokens"]))


def canonicalize_training_tokenizer(
    tokenizer: RuleTokenizer, samples: Sequence[dict[str, Any]],
) -> RuleTokenizer:
    """Migrate a shared legacy vocabulary without retaining unused card aliases."""
    required = RuleTokenizer.build(samples)
    required_tokens = set(required.tokens)
    vocabulary = set(tokenizer.tokens) | required_tokens
    vocabulary.difference_update(LEGACY_CARD_ALIAS_TOKENS - required_tokens)
    return RuleTokenizer((
        *SPECIAL_TOKENS,
        *sorted(vocabulary - set(SPECIAL_TOKENS)),
    ))


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"encoding": "utf-8"}
    if path.suffix == ".gz":
        kwargs["mode"] = "rt"
    else:
        kwargs["mode"] = "r"
    with opener(path, **kwargs) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("data_version") not in SUPPORTED_RULE_DATA_VERSIONS:
                raise ValueError(f"{path}:{line_number}: incompatible data version")
            if sample.get("language_version") != RULE_LANGUAGE_VERSION:
                raise ValueError(f"{path}:{line_number}: incompatible language version")
            yield sample


NON_CARD_ACTION_GROUP = "<NON_CARD_ACTION>"


def action_card_group(sample: dict[str, Any]) -> str:
    """Return the canonical action-card group used for balanced training."""
    card = sample.get("action", {}).get("card")
    if card is None:
        return NON_CARD_ACTION_GROUP
    return str(card["id"])


class CardBalancedBatchSampler(Sampler[list[int]]):
    """Cycle uniformly over cards, then sample a length-local batch per card."""

    def __init__(
        self, samples: Sequence[dict[str, Any]], lengths: Sequence[int], *,
        batch_size: int, num_batches: int, bucket_multiplier: int, seed: int,
    ) -> None:
        if not samples:
            raise ValueError("cannot sample an empty dataset")
        if len(samples) != len(lengths):
            raise ValueError("sample and length counts must match")
        if batch_size < 1 or num_batches < 0 or bucket_multiplier < 1:
            raise ValueError("batch size and bucket multiplier must be positive")
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.bucket_size = batch_size * bucket_multiplier
        grouped: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            grouped.setdefault(action_card_group(sample), []).append(index)
        self.group_names = tuple(sorted(grouped))
        self.group_buckets: tuple[tuple[tuple[int, ...], ...], ...] = tuple(
            tuple(
                tuple(ordered[begin:begin + self.bucket_size])
                for begin in range(0, len(ordered), self.bucket_size)
            )
            for name in self.group_names
            for ordered in (
                sorted(grouped[name], key=lambda index: lengths[index]),
            )
        )
        self.group_sizes = tuple(len(grouped[name]) for name in self.group_names)
        self.generator = torch.Generator().manual_seed(seed)
        self._group_order: list[int] = []
        self._group_cursor = 0

    def _next_group(self) -> int:
        if self._group_cursor >= len(self._group_order):
            self._group_order = torch.randperm(
                len(self.group_names), generator=self.generator
            ).tolist()
            self._group_cursor = 0
        group = self._group_order[self._group_cursor]
        self._group_cursor += 1
        return group

    def _next_batch(self) -> list[int]:
        group = self._next_group()
        anchor = int(torch.randint(
            self.group_sizes[group], (1,), generator=self.generator
        ).item())
        bucket = self.group_buckets[group][anchor // self.bucket_size]
        offsets = torch.randint(
            len(bucket), (self.batch_size,), generator=self.generator
        ).tolist()
        return [bucket[offset] for offset in offsets]

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self._next_batch()

    def __len__(self) -> int:
        return self.num_batches

    def skip_batches(self, count: int) -> None:
        """Advance exactly as iteration would for deterministic resume."""
        if count < 0:
            raise ValueError("cannot skip a negative number of batches")
        for _ in range(count):
            self._next_batch()


class RuleDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        tokenizer: RuleTokenizer,
        *,
        include_critical_mask: bool = False,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.include_critical_mask = include_critical_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        prefix, target = sample_tokens(self.samples[index])
        encoded = (
            torch.tensor(self.tokenizer.encode(prefix), dtype=torch.long),
            torch.tensor(self.tokenizer.encode(target), dtype=torch.long),
        )
        if not self.include_critical_mask:
            return encoded
        return (
            *encoded,
            torch.tensor(target_supervision_regions(target), dtype=torch.uint8),
            transition_consistency_specs(self.samples[index], target),
        )


def collate_rule_batch(batch, pad_id: int):
    prefixes, targets = zip(*(row[:2] for row in batch))
    prefix = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True, padding_value=pad_id)
    target = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_id)
    if len(batch[0]) == 2:
        return prefix, target
    critical = torch.nn.utils.rnn.pad_sequence(
        [row[2] for row in batch], batch_first=True, padding_value=False
    )
    return prefix, target, critical, [row[3] for row in batch]
