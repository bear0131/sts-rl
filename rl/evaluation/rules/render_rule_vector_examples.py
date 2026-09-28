"""Render count-only vector event examples for the standalone rule model.

The examples are intentionally deterministic.  They use the same canonical
state dictionaries as the simulator adapter, compute every event delta with
``vector_delta``, and print one structural token per line.  Continuous deltas
are rendered sparsely on their ``DELTA_SLOT`` line so the output remains
readable without changing the model's numeric target.

Run with::

    python -m rl.evaluation.rules.render_rule_vector_examples
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from env.core.encode import COMBAT_COUNTER_KEYS
from rl.rules.data.make_vector_rule_training_data import _effect_record
from rl.rules.rule_vector_model import (
    StateVectorConfig,
    VectorEvent,
    SerializedVectorEvents,
    serialize_vector_events,
    vector_delta,
)


def _counter_values(**updates: int) -> dict[str, int]:
    counters = {name: 0 for name in COMBAT_COUNTER_KEYS}
    counters.update(updates)
    return counters


def _state(
    *,
    energy: int,
    cards_played: int,
    attacks_played: int,
    sundial: int,
    enemy_hp: int,
    enemy_block: int,
    draw: int,
    discard: int,
    exhaust: int = 0,
    relics: Sequence[str] = (),
    relic_counters: Sequence[int] = (),
    enemy_powers: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a minimal canonical combat state for the examples."""

    return {
        "combat": {
            "turn": 1,
            "player": {
                "hp": 80,
                "max_hp": 80,
                "block": 0,
                "energy": energy,
                "gold": 99,
                "powers": {},
                "combat_counters": _counter_values(
                    cards_played=cards_played,
                    attacks_played=attacks_played,
                    sundial=sundial,
                ),
            },
            "monsters": [{
                "id": "ExampleEnemy",
                "hp": enemy_hp,
                "max_hp": 60 if enemy_hp <= 60 else enemy_hp,
                "block": enemy_block,
                "base_damage": 0,
                "adjusted_damage": 0,
                "hits": 0,
                "gone": False,
                "powers": dict(enemy_powers or {}),
            }],
            # Card instances are deliberately irrelevant to the pile vector.
            "hand": [{"id": "AnonymousAction", "up": 0}],
            "draw": [{"id": "AnonymousCard", "up": 0}] * draw,
            "discard": [{"id": "AnonymousCard", "up": 0}] * discard,
            "exhaust": [{"id": "AnonymousCard", "up": 0}] * exhaust,
        },
        "global": {
            "relics": list(relics),
            "relic_counters": list(relic_counters),
            "deck": [],
        },
    }


def _with(state: Mapping[str, Any], mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    result = deepcopy(state)
    mutate(result)
    return result


def _labels(config: StateVectorConfig) -> tuple[str, ...]:
    labels = [f"player.{name}" for name in config.player_scalar_names]
    labels.extend(f"player.buff.{name}" for name in config.player_buff_names)
    labels.extend(f"player.counter.{name}" for name in config.counter_names)
    labels.extend(f"player.relic.{name}" for name in config.relic_summary_names)
    labels.extend(
        f"player.relic_counter[{index}]"
        for index in range(len(config.relic_presence_names))
    )
    for enemy in range(config.max_enemies):
        labels.extend(f"enemy[{enemy}].{name}" for name in config.enemy_scalar_names)
        labels.extend(
            f"enemy[{enemy}].buff.{name}" for name in config.monster_buff_names
        )
    labels.extend(f"piles.{name}" for name in config.pile_names)
    return tuple(labels)


def format_delta(
    delta: Sequence[float], config: StateVectorConfig | None = None,
    *, min_abs: float = 1e-8,
) -> str:
    """Render a full delta vector as one sparse, human-readable token line."""

    config = config or StateVectorConfig()
    labels = _labels(config)
    if len(delta) != len(labels):
        raise ValueError(f"delta width {len(delta)} does not match state width {len(labels)}")
    changed = [
        f"{label}={float(value):+.6f}"
        for label, value in zip(labels, delta)
        if abs(float(value)) >= min_abs
    ]
    return "DELTA_SLOT " + ("; ".join(changed) if changed else "zero")


def _serialize_generated(
    state: Mapping[str, Any], action: Mapping[str, Any],
    config: StateVectorConfig, seed: int,
) -> SerializedVectorEvents:
    """Render fixed cases through the exact randomized event producer path."""

    import random

    raw_events = _effect_record(dict(state), dict(action), config, random.Random(seed))
    events = tuple(
        VectorEvent(
            str(event["kind"]),
            str(event["entity_ref"]),
            delta=(
                tuple(float(value) for value in event["delta"])
                if event.get("delta") is not None else None
            ),
            repeat_count=event.get("repeat_count"),
            calculations=tuple(event.get("calculations", ())),
        )
        for event in raw_events
    )
    return serialize_vector_events(events)


def render_serialized(
    serialized: SerializedVectorEvents,
    config: StateVectorConfig | None = None,
    *, min_delta_abs: float = 1e-8,
) -> str:
    """Render a serialized trace with two-space indentation per nesting level."""

    config = config or StateVectorConfig()
    if len(serialized.tokens) != len(serialized.delta_targets):
        raise ValueError("serialized token and delta target lengths differ")
    lines: list[str] = []
    depth = 0
    begin_tokens = {
        "<OUTPUT_BEGIN>", "<TRANSITION_BEGIN>", "<EVENT_BEGIN>",
        "<QUERY_BEGIN>", "<OBS_BEGIN>", "SCOPE_BEGIN", "CALL_BEGIN",
        "<CALCULATIONS_BEGIN>", "CALCULATION_BEGIN",
    }
    end_tokens = {
        "<OUTPUT_END>", "<TRANSITION_END>", "<EVENT_END>",
        "<QUERY_END>", "<OBS_END>", "SCOPE_END", "CALL_END",
        "<CALCULATIONS_END>", "CALCULATION_END",
    }
    for token, delta in zip(serialized.tokens, serialized.delta_targets):
        if token in end_tokens:
            depth = max(0, depth - 1)
        value = (
            format_delta(delta, config, min_abs=min_delta_abs)
            if token == "DELTA_SLOT" and delta is not None else token
        )
        lines.append("  " * depth + value)
        if token in begin_tokens:
            depth += 1
    return "\n".join(lines)


def _fly_kick() -> SerializedVectorEvents:
    state = _state(
        energy=3, cards_played=0, attacks_played=0, sundial=0,
        enemy_hp=50, enemy_block=0, draw=5, discard=1,
        enemy_powers={"Vulnerable": 2},
    )
    state["combat"]["hand"] = [{"id": "Dropkick", "up": 0}]
    return _serialize_generated(state, {"id": "Dropkick", "up": 0}, StateVectorConfig(), 0)


def _pommel_with_sundial() -> SerializedVectorEvents:
    state = _state(
        energy=3, cards_played=0, attacks_played=0, sundial=2,
        enemy_hp=60, enemy_block=0, draw=0, discard=3,
        relics=("Sundial",), relic_counters=(2,),
    )
    state["combat"]["hand"] = [{"id": "Pommel Strike", "up": 0}]
    return _serialize_generated(
        state, {"id": "Pommel Strike", "up": 0}, StateVectorConfig(), 0
    )


def render_examples() -> str:
    config = StateVectorConfig()
    return "\n\n".join((
        "示例一：飞身踢，目标有 2 层易伤（5 点伤害按易伤结算为 7 点）\n\n" + render_serialized(_fly_kick(), config),
        "示例二：剑柄打击，抽牌触发洗牌和日晷\n\n" + render_serialized(_pommel_with_sundial(), config),
    ))


if __name__ == "__main__":
    print(render_examples())
