"""Stream fresh, randomized schema-v5 vector-rule records.

This producer is intentionally independent of the legacy rule-data JSONL.  Each
line is a new sample, so the online trainer can consume it once and terminate
the pipe when its step budget is exhausted.  Card and relic names are used only
on the producer side to select their canonical rule programs; the emitted record
contains rule-token programs and fixed numeric vectors.

The state distribution is deliberately broad: one to five cards in hand (the
played card must be present), zero to five cards in each pile, random Ironclad
combat values and powers, random relic counters, and floor-conditioned encounter
and boss-relic pools.  Event deltas are always recomputed from the before/after
canonical states with ``vector_delta``; every event also carries an explicit
calculation trace and no source-rule token.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Sequence

from env.core.card_language import (
    LANGUAGE_CARDS,
    TOKEN_INDEX as CARD_TOKEN_INDEX,
    card_program,
)
from env.core.encounters import ACT_POOLS
from env.core.ironclad_cards import ATTACK, SKILL, POWER, IRONCLAD_CARDS
from env.core.rule_language import COMBAT_RELIC_RULES
from rl.rules.rule_vector_model import (
    RELIC_RULE_INDEX,
    StateVectorConfig,
    generic_state_calculations,
    vector_delta,
    vectorize_state,
)
from rl.rules.vector_rule_data import VECTOR_DATA_SCHEMA_VERSION


COMMON_RELIC_POOL = (
    "Vajra", "Oddly Smooth Stone", "Anchor", "Bronze Scales", "Akabeko",
    "Bag of Marbles", "Bag of Preparation", "Blood Vial", "Kunai", "Shuriken",
    "Pen Nib", "Ornamental Fan", "Letter Opener", "Lantern", "Happy Flower",
    "Mercury Hourglass", "Red Skull", "Paper Crane", "Paper Frog", "StrikeDummy",
    "Gremlin Horn", "Nunchaku", "Sundial",
)
BOSS_RELIC_POOL = (
    "Snecko Eye", "Runic Pyramid", "Velvet Choker", "Mark of Pain",
    "Coffee Dripper", "Fusion Hammer", "Runic Dome", "Sozu", "Cursed Key",
    "Ice Cream", "Busted Crown", "Philosopher's Stone", "Runic Cube",
    "SlaversCollar", "Gambling Chip",
)

# Exact values live only in the environment ValueStore.  The model receives a
# binary presence feature for every owned relic and must request these values
# from the corresponding RELIC_COUNTER node.
RELIC_COUNTER_SPECS: dict[str, dict[str, Any]] = {
    "Happy Flower": {
        "event": "TURN_START", "minimum": 0, "maximum": 2,
        "threshold": 3, "effect": "GAIN_ENERGY", "effect_amount": 1,
    },
    "Kunai": {
        "event": "ATTACK_PLAY", "minimum": 0, "maximum": 2,
        "threshold": 3, "effect": "GAIN_DEXTERITY", "effect_amount": 1,
    },
    "Shuriken": {
        "event": "ATTACK_PLAY", "minimum": 0, "maximum": 2,
        "threshold": 3, "effect": "GAIN_STRENGTH", "effect_amount": 1,
    },
    "Pen Nib": {
        "event": "ATTACK_PLAY", "minimum": 0, "maximum": 9,
        "threshold": 10, "effect": "NEXT_DAMAGE_X2", "effect_amount": 2,
    },
    "Ornamental Fan": {
        "event": "ATTACK_PLAY", "minimum": 0, "maximum": 2,
        "threshold": 3, "effect": "GAIN_BLOCK", "effect_amount": 4,
    },
    "Letter Opener": {
        "event": "SKILL_PLAY", "minimum": 0, "maximum": 2,
        "threshold": 3, "effect": "DAMAGE_ALL", "effect_amount": 5,
    },
    "Nunchaku": {
        "event": "ATTACK_PLAY", "minimum": 0, "maximum": 9,
        "threshold": 10, "effect": "GAIN_ENERGY", "effect_amount": 1,
    },
    "Sundial": {
        "event": "PILE_SHUFFLE", "minimum": 0, "maximum": 2,
        "threshold": 3, "effect": "GAIN_ENERGY", "effect_amount": 2,
    },
    # A legal action can only be generated below the six-card cap. This counter
    # does not wrap until the environment starts the next turn.
    "Velvet Choker": {
        "event": "CARD_PLAY", "minimum": 0, "maximum": 5,
        "threshold": 6, "reset": False, "effect": "NONE", "effect_amount": 0,
    },
}

CARD_PROGRAM_INDEX = {
    (card_id, bool(upgraded)): tuple(CARD_TOKEN_INDEX[token] for token in card_program(card_id, upgraded))
    for card_id in LANGUAGE_CARDS
    for upgraded in (False, True)
}
CARD_IDS = tuple(sorted(IRONCLAD_CARDS))
HAND_CARD_IDS = tuple(sorted(LANGUAGE_CARDS))

PLAYER_POWER_POOL = (
    "Strength", "Dexterity", "Vulnerable", "Weakened", "Frail", "Intangible",
    "Barricade", "Corruption", "Double Tap", "Brutality", "Combust",
    "Dark Embrace", "Demon Form", "Evolve", "Feel No Pain", "Fire Breathing",
    "Flame Barrier", "Juggernaut", "Metallicize", "Rage", "Rupture", "Vigor",
)
MONSTER_POWER_POOL = (
    "Strength", "Vulnerable", "Weakened", "Artifact", "Metallicize",
    "Plated Armor", "Poison",
)


def _program(card_id: str, upgraded: int) -> list[int]:
    return list(CARD_PROGRAM_INDEX[(card_id, bool(upgraded))])


def _relic_program(relic_id: str) -> list[int]:
    return [RELIC_RULE_INDEX[token] for token in _relic_effect_tokens(relic_id)]


def _relic_effect_tokens(relic_id: str) -> tuple[str, ...]:
    # Keep tokenization exactly aligned with RelicRuleEncoder.relic_vectors.
    import re

    return tuple(re.findall(r"[A-Za-z0-9]+|%|[^\sA-Za-z0-9]", COMBAT_RELIC_RULES[relic_id]))


def _act(floor: int) -> int:
    return 1 if floor <= 16 else 2 if floor <= 33 else 3


def _encounter(rng: random.Random, floor: int) -> tuple[int, str, str]:
    act = _act(floor)
    if floor in (16, 33, 50):
        kind = "boss"
    elif rng.random() < 0.22:
        kind = "elite"
    else:
        kind = "normal"
    pool = ACT_POOLS[act][kind]
    return act, kind, rng.choice(pool)


def _powers(rng: random.Random, pool: tuple[str, ...], maximum: int) -> dict[str, int]:
    names = rng.sample(pool, rng.randint(0, min(maximum, len(pool))))
    return {name: rng.randint(1, 5 if name not in {"Strength", "Dexterity"} else 8) for name in names}


def _card_instance(rng: random.Random, *, action: bool = False) -> dict[str, Any]:
    card_id = rng.choice(CARD_IDS if action else HAND_CARD_IDS)
    upgraded = rng.randint(0, 1)
    return {"id": card_id, "up": upgraded}


def _pile(rng: random.Random, count: int) -> list[dict[str, Any]]:
    return [_card_instance(rng) for _ in range(count)]


def _initial_relics(rng: random.Random, floor: int) -> tuple[list[str], list[float]]:
    act = _act(floor)
    # Burning Blood remains common for Ironclad, while the rest is sampled.
    relics = ["Burning Blood"] if rng.random() < 0.85 else []
    max_common = min(len(COMMON_RELIC_POOL), max(0, floor // 8 + 1))
    if max_common:
        relics.extend(rng.sample(COMMON_RELIC_POOL, rng.randint(0, max_common)))
    # Boss relics are tied to completed acts: none in act 1, one in act 2 and
    # two in act 3.  This avoids impossible early boss relic observations.
    boss_count = max(0, act - 1)
    if boss_count:
        relics.extend(rng.sample(BOSS_RELIC_POOL, boss_count))
    # Defensive de-duplication keeps relic counters slot-aligned.
    relics = list(dict.fromkeys(relics))[:32]
    counters = []
    for relic in relics:
        spec = RELIC_COUNTER_SPECS.get(relic)
        counters.append(
            float(rng.randint(int(spec["minimum"]), int(spec["maximum"])))
            if spec is not None else 0.0
        )
    return relics, counters


def _canonical_state(rng: random.Random, floor: int, encounter: str, kind: str, relics: list[str], relic_counters: list[float], action: dict[str, Any], hand: list[dict[str, Any]]) -> dict[str, Any]:
    act = _act(floor)
    max_hp = rng.randint(70, 110)
    player = {
        "hp": rng.randint(max(1, max_hp // 5), max_hp),
        "max_hp": max_hp,
        "block": rng.randint(0, 60),
        "energy": rng.randint(0, 5),
        "gold": rng.randint(0, 999),
        "powers": _powers(rng, PLAYER_POWER_POOL, 5),
        "combat_counters": {},
    }
    printed_cost = IRONCLAD_CARDS[action["id"]].cost[int(action["up"] > 0)]
    player["energy"] = max(player["energy"], max(0, printed_cost))
    # Whirlwind is the only Ironclad X-cost card in this vocabulary.  A legal
    # played action must have at least one energy available so its X value is
    # well-defined; the actual X value is the complete pre-payment energy.
    if action["id"] == "Whirlwind":
        player["energy"] = max(1, player["energy"])
    monster_count = rng.randint(1, 3)
    monsters = []
    for slot in range(monster_count):
        mmax = rng.randint(35 + 10 * act, 90 + 35 * act)
        mhp = rng.randint(max(1, mmax // 4), mmax)
        monsters.append({
            "id": f"{encounter}#{slot}", "hp": mhp, "max_hp": mmax,
            "block": rng.randint(0, 60), "base_damage": rng.randint(0, 30 + 10 * act),
            "adjusted_damage": rng.randint(0, 35 + 12 * act), "hits": rng.randint(1, 4),
            "gone": False, "powers": _powers(rng, MONSTER_POWER_POOL, 3),
        })
    return {
        "combat": {
            "turn": rng.randint(1, 12), "player": player, "monsters": monsters,
            "hand": copy.deepcopy(hand), "draw": _pile(rng, rng.randint(0, 5)),
            "discard": _pile(rng, rng.randint(0, 5)), "exhaust": _pile(rng, rng.randint(0, 5)),
        },
        "global": {
            "floor": floor, "act": act, "encounter": encounter, "encounter_kind": kind,
            "relics": list(relics), "relic_counters": list(relic_counters),
            "relic_counter_present": [
                relic in RELIC_COUNTER_SPECS for relic in relics
            ],
        },
    }


def _draw(state: dict[str, Any], amount: int, rng: random.Random) -> None:
    combat = state["combat"]
    amount = min(max(0, int(amount)), len(combat["draw"]), max(0, 10 - len(combat["hand"])))
    for _ in range(amount):
        combat["hand"].append(combat["draw"].pop())


def _add_status(mapping: dict[str, int], name: str, amount: int) -> None:
    mapping[name] = max(0, int(mapping.get(name, 0)) + int(amount))


def _damage(enemy: dict[str, Any], amount: int) -> None:
    amount = max(0, int(amount))
    absorbed = min(int(enemy.get("block", 0)), amount)
    enemy["block"] -= absorbed
    enemy["hp"] = max(0, int(enemy.get("hp", 0)) - (amount - absorbed))
    if enemy["hp"] == 0:
        enemy["gone"] = True


def _power_value(mapping: Mapping[str, Any], name: str) -> int:
    wanted = name.casefold()
    for key, value in mapping.items():
        if str(key).casefold() == wanted:
            return int(value or 0)
    return 0


def _damage_calculations(
    state: Mapping[str, Any], slot: int, base_damage: int, amount: int,
    *, damage_multiplier: int = 1,
) -> tuple[dict[str, Any], ...]:
    """Return the complete arithmetic trace for one independent damage call."""

    player = state["combat"]["player"]
    enemy = state["combat"]["monsters"][slot]
    strength = _power_value(player.get("powers", {}), "Strength")
    vulnerable = _power_value(enemy.get("powers", {}), "Vulnerable")
    strengthened = int(base_damage) + strength
    calculations: list[dict[str, Any]] = [
        {"type": "READ_BASE_DAMAGE", "value": int(base_damage)},
        {"type": "READ_STRENGTH", "value": strength},
        {
            "type": "ADD_STRENGTH", "left": int(base_damage),
            "right": strength, "result": strengthened,
        },
    ]
    modified = strengthened
    if damage_multiplier != 1:
        modified = strengthened * int(damage_multiplier)
        calculations.extend((
            {
                "type": "READ_DAMAGE_MULTIPLIER",
                "source": "parent.relic_trigger.damage_multiplier",
                "value": int(damage_multiplier),
            },
            {
                "type": "MULTIPLY_DAMAGE",
                "left": strengthened,
                "right": int(damage_multiplier),
                "result": modified,
            },
        ))
    if vulnerable > 0:
        multiplied = modified * 3
        calculations.extend((
            {"type": "READ_VULNERABLE", "value": vulnerable},
            {
                "type": "MULTIPLY_VULNERABLE", "left": modified,
                "right": 3, "result": multiplied,
            },
            {
                "type": "FLOOR_DIVIDE_VULNERABLE", "left": multiplied,
                "right": 2, "result": int(amount),
            },
        ))
    else:
        calculations.append({
            "type": "NO_VULNERABLE", "input": modified,
            "result": int(amount),
        })
    block_before = int(enemy.get("block", 0))
    absorbed = min(block_before, int(amount))
    unblocked = int(amount) - absorbed
    hp_before = int(enemy.get("hp", 0))
    calculations.extend((
        {"type": "READ_TARGET_BLOCK", "value": block_before},
        {
            "type": "BLOCK_ABSORB", "left": int(amount),
            "right": block_before, "absorbed": absorbed,
        },
        {
            "type": "SUBTRACT_BLOCK", "left": block_before,
            "right": absorbed, "result": block_before - absorbed,
        },
        {
            "type": "SUBTRACT_BLOCK_DAMAGE", "left": int(amount),
            "right": absorbed, "result": unblocked,
        },
        {"type": "READ_TARGET_HP", "value": hp_before},
        {
            "type": "SUBTRACT_HP", "left": hp_before,
            "right": unblocked, "result": max(0, hp_before - unblocked),
        },
    ))
    return tuple(calculations)


def _fixed_damage_calculations(
    state: Mapping[str, Any], slot: int, amount: int,
) -> tuple[dict[str, Any], ...]:
    """Arithmetic trace for relic damage that ignores card modifiers."""

    enemy = state["combat"]["monsters"][slot]
    block_before = int(enemy.get("block", 0))
    absorbed = min(block_before, int(amount))
    unblocked = int(amount) - absorbed
    hp_before = int(enemy.get("hp", 0))
    return (
        {"type": "READ_BASE_DAMAGE", "value": int(amount)},
        {
            "type": "BLOCK_ABSORB", "left": int(amount),
            "right": block_before, "absorbed": absorbed,
        },
        {
            "type": "READ_TARGET_BLOCK", "value": block_before,
        },
        {
            "type": "SUBTRACT_BLOCK", "left": block_before,
            "right": absorbed, "result": block_before - absorbed,
        },
        {
            "type": "SUBTRACT_BLOCK_DAMAGE", "left": int(amount),
            "right": absorbed, "result": unblocked,
        },
        {"type": "READ_TARGET_HP", "value": hp_before},
        {
            "type": "SUBTRACT_HP", "left": hp_before,
            "right": unblocked, "result": max(0, hp_before - unblocked),
        },
    )


def _player_damage_calculations(
    state: Mapping[str, Any], amount: int,
) -> tuple[dict[str, Any], ...]:
    player = state["combat"]["player"]
    hp_before = int(player.get("hp", 0))
    return (
        {"type": "READ_TRIGGER_DAMAGE", "value": int(amount)},
        {
            "type": "SUBTRACT_PLAYER_HP", "left": hp_before,
            "right": int(amount), "result": max(0, hp_before - int(amount)),
        },
    )


def _make_event(
    current: dict[str, Any], events: list[dict[str, Any]], config: StateVectorConfig,
    kind: str, entity: str, mutate: Callable[[dict[str, Any]], None] | None = None,
    *, calculations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    before = current
    after = copy.deepcopy(before)
    if mutate is not None:
        mutate(after)
    dense = [float(x) for x in vector_delta(before, after, config).flat.tolist()]
    event: dict[str, Any] = {
        "kind": kind, "entity_ref": entity,
    }
    # Explicit arithmetic traces (damage and self-damage) already contain the
    # state reads and results for HP/block.  Keep generic calculations for any
    # other changed field (for example a lethal hit's gone/present flags), but
    # do not append duplicate APPLY_DELTA blocks for the arithmetic fields.
    if calculations is None:
        event_calculations = generic_state_calculations(before, after, config)
    else:
        generic = generic_state_calculations(before, after, config)
        if kind == "EVENT_DAMAGE" and entity.startswith("ENTITY_ENEMY_SLOT_"):
            slot = entity.removeprefix("ENTITY_ENEMY_SLOT_")
            covered_suffixes = (f"enemy[{slot}].hp", f"enemy[{slot}].block")
            generic = tuple(
                calculation for calculation in generic
                if calculation.get("field") not in covered_suffixes
            )
        elif kind == "EVENT_DAMAGE" and entity == "ENTITY_SELF":
            generic = tuple(
                calculation for calculation in generic
                if calculation.get("field") != "player.hp"
            )
        event_calculations = tuple(calculations) + generic
    event["calculations"] = [
        dict(calculation)
        for calculation in event_calculations
    ]
    if any(abs(value) >= 1e-8 for value in dense):
        event["delta"] = dense
    events.append(event)
    return after


def _advance_relic_counter(
    current: dict[str, Any],
    events: list[dict[str, Any]],
    config: StateVectorConfig,
    relic_id: str,
    counter_event: str,
) -> tuple[dict[str, Any], bool]:
    """Advance one owned relic counter and emit an explicit counter event."""

    relics = list(current["global"].get("relics", ()))
    if relic_id not in relics:
        return current, False
    spec = RELIC_COUNTER_SPECS[relic_id]
    if spec["event"] != counter_event:
        return current, False
    slot = relics.index(relic_id)
    counters = list(current["global"].get("relic_counters", ()))
    while len(counters) < len(relics):
        counters.append(0.0)
    current["global"]["relic_counters"] = counters
    before = int(counters[slot]) if slot < len(counters) else 0
    advanced = before + 1
    threshold = int(spec["threshold"])
    triggered = advanced >= threshold
    reset = bool(spec.get("reset", True))
    result = 0 if triggered and reset else advanced

    def mutate(state: dict[str, Any]) -> None:
        state["global"]["relic_counters"][slot] = result

    current = _make_event(
        current,
        events,
        config,
        "EVENT_COUNTER",
        f"ENTITY_RELIC_SLOT_{slot}",
        mutate,
        calculations=({
            "type": "RELIC_COUNTER_UPDATE",
            "source": f"relic[{slot}].counter",
            "left": before,
            "increment": 1,
            "threshold": threshold,
            "reset": int(reset),
            "result": result,
            "triggered": int(triggered),
        },),
    )
    events[-1].update({
        "counter_ref": f"relic[{slot}].counter",
        "counter_event": counter_event,
        "counter_increment": 1,
        "counter_threshold": threshold,
        "counter_reset": int(reset),
        "trigger_effect": str(spec.get("effect", "NONE")),
        "trigger_amount": int(spec.get("effect_amount", 0)),
        "triggered": bool(triggered),
    })
    return current, triggered


def _effect_record(before: dict[str, Any], action: dict[str, Any], config: StateVectorConfig, rng: random.Random) -> list[dict[str, Any]]:
    card_id = action["id"]
    upgraded = int(action["up"] > 0)
    info = IRONCLAD_CARDS[card_id]
    current = before
    events: list[dict[str, Any]] = []
    cost = info.cost[upgraded]
    if cost < 0:
        # X-cost is not a random small integer: it consumes all current energy.
        # Keep this value before ``play_card`` mutates the player state because
        # Whirlwind uses the same X as its hit count.
        cost = int(before["combat"]["player"].get("energy", 0))
        if cost < 1:
            raise ValueError("X-cost action requires at least one energy")
    def play_card(s: dict[str, Any]) -> None:
        s["combat"]["hand"].remove(action)
        player = s["combat"]["player"]
        player["energy"] = max(0, int(player["energy"]) - cost)

    # Payment belongs to CARD_COMMIT and is never repeated by Double Tap.
    current = _make_event(current, events, config, "EVENT_CARD_PLAY", "ENTITY_SELF", play_card)
    events[-1]["phase"] = "COMMIT_PAYMENT"
    double_tap = (
        info.kind == ATTACK
        and _power_value(current["combat"]["player"].get("powers", {}), "Double Tap") > 0
    )
    if double_tap:
        current = _make_event(
            current, events, config, "EVENT_STATUS", "ENTITY_SELF",
            lambda s: s["combat"]["player"]["powers"].__setitem__(
                "Double Tap",
                max(
                    0,
                    int(s["combat"]["player"]["powers"].get("Double Tap", 0)) - 1,
                ),
            ),
        )
        events[-1].update({
            "phase": "COMMIT_DOUBLE_TAP",
            "double_tap_repeat": 2,
        })

    pen_nib_multiplier = 1
    counter_events = ("CARD_PLAY",)
    if info.kind == ATTACK:
        counter_events += ("ATTACK_PLAY",)
    elif info.kind == SKILL:
        counter_events += ("SKILL_PLAY",)
    for counter_event in counter_events:
        for relic_id in tuple(current["global"].get("relics", ())):
            spec = RELIC_COUNTER_SPECS.get(relic_id)
            if spec is None or spec["event"] != counter_event:
                continue
            current, triggered = _advance_relic_counter(
                current, events, config, relic_id, counter_event,
            )
            events[-1]["phase"] = "CARD_PLAY"
            if not triggered:
                continue
            effect = str(spec.get("effect", "NONE"))
            amount = int(spec.get("effect_amount", 0))
            if effect == "NEXT_DAMAGE_X2":
                pen_nib_multiplier = amount
            elif effect == "GAIN_ENERGY":
                current = _make_event(
                    current, events, config, "EVENT_ENERGY", "ENTITY_SELF",
                    lambda s, amount=amount: s["combat"]["player"].__setitem__(
                        "energy", int(s["combat"]["player"].get("energy", 0)) + amount,
                    ),
                )
                events[-1].update({"phase": "CARD_PLAY", "trigger_owner": relic_id})
            elif effect in {"GAIN_DEXTERITY", "GAIN_STRENGTH"}:
                status = "Dexterity" if effect == "GAIN_DEXTERITY" else "Strength"
                current = _make_event(
                    current, events, config, "EVENT_STATUS", "ENTITY_SELF",
                    lambda s, status=status, amount=amount: _add_status(
                        s["combat"]["player"]["powers"], status, amount,
                    ),
                )
                events[-1].update({"phase": "CARD_PLAY", "trigger_owner": relic_id})
            elif effect == "GAIN_BLOCK":
                current = _make_event(
                    current, events, config, "EVENT_BLOCK", "ENTITY_SELF",
                    lambda s, amount=amount: s["combat"]["player"].__setitem__(
                        "block", int(s["combat"]["player"].get("block", 0)) + amount,
                    ),
                )
                events[-1].update({"phase": "CARD_PLAY", "trigger_owner": relic_id})
            elif effect == "DAMAGE_ALL":
                for slot in range(len(current["combat"]["monsters"])):
                    calculations = _fixed_damage_calculations(current, slot, amount)
                    current = _make_event(
                        current, events, config, "EVENT_DAMAGE",
                        f"ENTITY_ENEMY_SLOT_{slot}",
                        lambda s, slot=slot, amount=amount: _damage(
                            s["combat"]["monsters"][slot], amount,
                        ),
                        calculations=calculations,
                    )
                    events[-1].update({
                        "phase": "CARD_PLAY", "trigger_owner": relic_id,
                        "damage_source": "RELIC_FIXED",
                    })
    target = rng.randrange(len(current["combat"]["monsters"]))
    base_damage = info.damage[upgraded]
    if card_id == "Body Slam":
        base_damage = int(current["combat"]["player"].get("block", 0))
    hits = {"Twin Strike": 2, "Pummel": 4 + upgraded, "Sword Boomerang": 3 + upgraded}.get(card_id, 1)
    if card_id == "Whirlwind":
        hits = cost
    all_targets = card_id in {"Cleave", "Thunderclap", "Whirlwind", "Immolate", "Reaper"}
    if base_damage > 0:
        for _ in range(hits):
            if card_id == "Sword Boomerang":
                slots = [rng.randrange(len(current["combat"]["monsters"]))]
            else:
                slots = (
                    list(range(len(current["combat"]["monsters"])))
                    if all_targets else [target]
                )
            for slot in slots:
                amount = max(
                    0,
                    base_damage + _power_value(
                        current["combat"]["player"].get("powers", {}), "Strength"
                    ),
                )
                amount *= pen_nib_multiplier
                enemy = current["combat"]["monsters"][slot]
                vulnerable = _power_value(enemy.get("powers", {}), "Vulnerable")
                if vulnerable > 0:
                    amount = (amount * 3) // 2
                calculations = _damage_calculations(
                    current, slot, base_damage, amount,
                    damage_multiplier=pen_nib_multiplier,
                )
                current = _make_event(
                    current, events, config, "EVENT_DAMAGE",
                    f"ENTITY_ENEMY_SLOT_{slot}",
                    lambda s, slot=slot, amount=amount: _damage(
                        s["combat"]["monsters"][slot], amount
                    ),
                    calculations=calculations,
                )
                events[-1]["phase"] = "CARD_PLAY"
    if info.block[upgraded] > 0:
        current = _make_event(current, events, config, "EVENT_BLOCK", "ENTITY_SELF", lambda s: s["combat"]["player"].__setitem__("block", s["combat"]["player"].get("block", 0) + info.block[upgraded]))

    status_amounts = {
        "Bash": ("Vulnerable", 2 + upgraded), "Clothesline": ("Weakened", 2 + upgraded),
        "Thunderclap": ("Vulnerable", 1 + upgraded), "Uppercut": ("Vulnerable", 1 + upgraded),
    }
    if card_id in status_amounts:
        status, amount = status_amounts[card_id]
        slots = range(len(current["combat"]["monsters"])) if card_id == "Thunderclap" else (target,)
        for slot in slots:
            current = _make_event(current, events, config, "EVENT_STATUS", f"ENTITY_ENEMY_SLOT_{slot}", lambda s, slot=slot, status=status, amount=amount: _add_status(s["combat"]["monsters"][slot]["powers"], status, amount))
    if card_id == "Uppercut":
        current = _make_event(current, events, config, "EVENT_STATUS", f"ENTITY_ENEMY_SLOT_{target}", lambda s: _add_status(s["combat"]["monsters"][target]["powers"], "Weakened", 1 + upgraded))
    if card_id in {"Shockwave", "Intimidate"}:
        amount = (3 + 2 * upgraded) if card_id == "Shockwave" else (1 + upgraded)
        for slot in range(len(current["combat"]["monsters"])):
            current = _make_event(current, events, config, "EVENT_STATUS", f"ENTITY_ENEMY_SLOT_{slot}", lambda s, slot=slot, amount=amount: _add_status(s["combat"]["monsters"][slot]["powers"], "Weakened", amount))
            if card_id == "Shockwave":
                current = _make_event(current, events, config, "EVENT_STATUS", f"ENTITY_ENEMY_SLOT_{slot}", lambda s, slot=slot, amount=amount: _add_status(s["combat"]["monsters"][slot]["powers"], "Vulnerable", amount))
    if card_id == "Disarm":
        current = _make_event(current, events, config, "EVENT_STATUS", f"ENTITY_ENEMY_SLOT_{target}", lambda s: s["combat"]["monsters"][target]["powers"].__setitem__("Strength", max(0, s["combat"]["monsters"][target]["powers"].get("Strength", 0) - (2 + upgraded))))
    player_status = {
        "Flex": ("Strength", 2 + 2 * upgraded), "Inflame": ("Strength", 2 + upgraded),
        "Metallicize": ("Metallicize", 3 + upgraded), "Rage": ("Rage", 3 + 2 * upgraded),
        "Demon Form": ("Demon Form", 2 + upgraded), "Feel No Pain": ("Feel No Pain", 3 + upgraded),
        "Fire Breathing": ("Fire Breathing", 6 + 4 * upgraded), "Juggernaut": ("Juggernaut", 5 + 2 * upgraded),
        "Spot Weakness": ("Strength", 3 + upgraded),
    }
    if card_id in player_status:
        status, amount = player_status[card_id]
        current = _make_event(current, events, config, "EVENT_STATUS", "ENTITY_SELF", lambda s, status=status, amount=amount: _add_status(s["combat"]["player"]["powers"], status, amount))

    if card_id in {"Bloodletting", "Hemokinesis", "Offering"}:
        hp_loss = {"Bloodletting": 3, "Hemokinesis": 2, "Offering": 6}[card_id]
        current = _make_event(
            current, events, config, "EVENT_DAMAGE", "ENTITY_SELF",
            lambda s, hp_loss=hp_loss: s["combat"]["player"].__setitem__(
                "hp", max(0, s["combat"]["player"]["hp"] - hp_loss)
            ),
            calculations=_player_damage_calculations(current, hp_loss),
        )
        gain = {"Bloodletting": 2 + upgraded, "Hemokinesis": 0, "Offering": 2}[card_id]
        if gain:
            current = _make_event(current, events, config, "EVENT_ENERGY", "ENTITY_SELF", lambda s, gain=gain: s["combat"]["player"].__setitem__("energy", s["combat"]["player"]["energy"] + gain))
    elif card_id == "Seeing Red":
        current = _make_event(current, events, config, "EVENT_ENERGY", "ENTITY_SELF", lambda s: s["combat"]["player"].__setitem__("energy", s["combat"]["player"]["energy"] + 2))

    draw_amount = {
        "Pommel Strike": 1 + upgraded, "Shrug It Off": 1, "Warcry": 1 + upgraded,
        "Battle Trance": 3 + upgraded, "Burning Pact": 2 + upgraded,
        "Offering": 3 + 2 * upgraded, "Dropkick": 1,
    }.get(card_id, 0)
    if card_id == "Dropkick" and current["combat"]["monsters"][target]["powers"].get("Vulnerable", 0) <= 0:
        draw_amount = 0
    elif card_id == "Dropkick":
        current = _make_event(current, events, config, "EVENT_ENERGY", "ENTITY_SELF", lambda s: s["combat"]["player"].__setitem__("energy", s["combat"]["player"]["energy"] + 1))
    draw_amount = min(draw_amount, max(0, 10 - len(current["combat"]["hand"])))
    if draw_amount:
        first_draw = min(draw_amount, len(current["combat"]["draw"]))
        if first_draw:
            current = _make_event(current, events, config, "EVENT_DRAW", "ENTITY_SELF", lambda s, amount=first_draw: _draw(s, amount, rng))
        remaining = draw_amount - first_draw
        if remaining and current["combat"]["discard"]:
            def shuffle_discard(s: dict[str, Any]) -> None:
                s["combat"]["draw"].extend(s["combat"]["discard"])
                s["combat"]["discard"] = []
                rng.shuffle(s["combat"]["draw"])
            current = _make_event(current, events, config, "EVENT_PILE_SHUFFLE", "ENTITY_PILES", shuffle_discard)
            events[-1]["phase"] = "CARD_PLAY"
            current, sundial_triggered = _advance_relic_counter(
                current, events, config, "Sundial", "PILE_SHUFFLE",
            )
            if events and events[-1].get("kind") == "EVENT_COUNTER":
                events[-1]["phase"] = "CARD_PLAY"
            if sundial_triggered:
                current = _make_event(
                    current, events, config, "EVENT_ENERGY", "ENTITY_SELF",
                    lambda s: s["combat"]["player"].__setitem__(
                        "energy", int(s["combat"]["player"].get("energy", 0)) + 2,
                    ),
                )
                events[-1].update({
                    "phase": "CARD_PLAY", "trigger_owner": "Sundial",
                })
            current = _make_event(current, events, config, "EVENT_DRAW", "ENTITY_SELF", lambda s, amount=remaining: _draw(s, amount, rng))

    if card_id in {"Burning Pact", "True Grit", "Second Wind", "Fiend Fire"} and current["combat"]["hand"]:
        exhaust_count = 1 if card_id in {"Burning Pact", "True Grit"} else len(current["combat"]["hand"])
        current = _make_event(current, events, config, "EVENT_EXHAUST", "ENTITY_PILES", lambda s, amount=exhaust_count, rng=rng: (s["combat"]["exhaust"].extend(s["combat"]["hand"][:amount]), s["combat"].__setitem__("hand", s["combat"]["hand"][amount:])))
    # Read the explicit self-exhaust trait, not another card's exhaust effect
    # (True Grit/Burning Pact) or an end-of-turn ethereal trait (Carnage).
    program = card_program(card_id, bool(upgraded))
    exhaust_self = any(pair == ("TRAIT", "EXHAUST") for pair in zip(program, program[1:]))
    destination = None if info.kind == POWER else ("exhaust" if exhaust_self else "discard")
    if destination is not None:
        current = _make_event(
            current, events, config, "EVENT_EXHAUST" if exhaust_self else "EVENT_PILE_DELTA",
            "ENTITY_PILES", lambda s: s["combat"][destination].append(copy.deepcopy(action)),
        )
        events[-1]["phase"] = "COMMIT_DESTINATION"
    if card_id in {"Anger", "Immolate"}:
        created = action if card_id == "Anger" else {"id": "Burn", "up": 0}
        current = _make_event(current, events, config, "EVENT_CARD_CREATE", "ENTITY_PILES",
                              lambda s: s["combat"]["discard"].append(copy.deepcopy(created)))
    if card_id in {"Wild Strike", "Reckless Charge"}:
        created = {"id": "Wound" if card_id == "Wild Strike" else "Dazed", "up": 0}
        def insert_status(s: dict[str, Any]) -> None:
            draw = s["combat"]["draw"]
            draw.insert(rng.randrange(len(draw) + 1), copy.deepcopy(created))
        current = _make_event(current, events, config, "EVENT_PILE_SHUFFLE", "ENTITY_PILES", insert_status)

    current = _make_event(current, events, config, "EVENT_ACTION_COMPLETE", "ENTITY_SELF", None)
    events[-1]["phase"] = "COMMIT_COMPLETE"
    return events


def build_record_with_canonical(
    index: int,
    config: StateVectorConfig,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one training record and retain its environment-side state.

    The canonical state contains names only for reporting and simulator-side
    execution.  The model-facing record remains anonymous and unchanged.
    """

    floor = rng.randint(1, 50)
    act, kind, encounter = _encounter(rng, floor)
    action = _card_instance(rng, action=True)
    other_count = rng.randint(0, 4)
    hand = [_card_instance(rng) for _ in range(other_count)] + [action]
    rng.shuffle(hand)
    relics, counters = _initial_relics(rng, floor)
    canonical = _canonical_state(rng, floor, encounter, kind, relics, counters, action, hand)
    events = _effect_record(canonical, action, config, rng)
    state_vector = vectorize_state(canonical, config)
    record = {
        "schema_version": VECTOR_DATA_SCHEMA_VERSION,
        "state": {
            "player": [float(x) for x in state_vector.player.tolist()],
            "enemies": [[float(x) for x in row] for row in state_vector.enemies.tolist()],
            "piles": [float(x) for x in state_vector.piles.tolist()],
            "relic_counters": [float(x) for x in state_vector.relic_counters.tolist()],
        },
        "action": {"card_rule": _program(action["id"], action["up"]), "upgrade": action["up"]},
        "hand": [{"card_rule": _program(card["id"], card["up"]), "upgrade": card["up"]} for card in hand],
        "relics": [
            {
                "rule": _relic_program(relic),
                "present": 1.0,
                "counter_present": relic in RELIC_COUNTER_SPECS,
            }
            for relic in relics
        ],
        "events": events,
        "runtime": {
            "relic_counters": [
                {
                    "slot": slot,
                    "present": relic in RELIC_COUNTER_SPECS,
                    "value": counter,
                }
                for slot, (relic, counter) in enumerate(zip(relics, counters))
            ],
        },
        "metadata": {"teacher": "approximate-ironclad-v2-card-identity", "sample_index": index, "floor": floor, "act": act, "encounter_kind": kind, "encounter": encounter, "relics": relics, "hand_size": len(hand)},
    }
    return record, canonical


def build_record(index: int, config: StateVectorConfig, rng: random.Random) -> dict[str, Any]:
    record, _canonical = build_record_with_canonical(index, config, rng)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rl/data/vector-rule-training-v6.jsonl"))
    parser.add_argument("--samples", type=int, default=32)
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
    stream = sys.stdout if args.stdout or args.stream else None
    if stream is not None:
        index = 0
        try:
            while args.stream or index < args.samples:
                print(json.dumps(build_record(index, config, rng), separators=(",", ":")), flush=True)
                index += 1
        except BrokenPipeError:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for index in range(args.samples):
            output.write(json.dumps(build_record(index, config, rng), separators=(",", ":")) + "\n")
    print(f"wrote {args.samples} schema-v5 records to {args.out}")


if __name__ == "__main__":
    main()
