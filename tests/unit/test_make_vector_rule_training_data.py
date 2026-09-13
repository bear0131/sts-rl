from __future__ import annotations

import json
import random

import pytest

from rl.rules.data.make_vector_rule_training_data import (
    BOSS_RELIC_POOL,
    _initial_relics,
    _effect_record,
    build_record,
    build_record_with_canonical,
)
from rl.rules.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_VOCAB,
    StateVectorConfig,
    VectorEvent,
    serialize_vector_events,
)
from rl.rules.vector_rule_data import example_from_v6_record


def test_random_records_are_valid_v6_and_within_requested_ranges():
    config = StateVectorConfig()
    floor_index = config.player_scalar_names.index("floor")
    for index in range(24):
        record = build_record(index, config, random.Random(1000 + index))
        assert 1 <= record["metadata"]["hand_size"] <= 5
        assert all(0 <= token < len(CARD_RULE_VOCAB) for token in record["action"]["card_rule"])
        assert all(
            0 <= token < len(CARD_RULE_VOCAB)
            for card in record["hand"] for token in card["card_rule"]
        )
        assert all(
            0 <= token < len(RELIC_RULE_VOCAB)
            for relic in record["relics"] for token in relic["rule"]
        )
        example = example_from_v6_record(
            record, config, card_vocab_size=len(CARD_RULE_VOCAB),
            relic_vocab_size=len(RELIC_RULE_VOCAB), max_cards=5, max_relics=32,
        )
        piles = [round(value * config.pile_scale) for value in example.state.piles.tolist()]
        assert all(0 <= value <= 5 for value in piles)
        assert 0.0 < example.state.player[floor_index].item() <= 1.0
        assert any(delta is not None for delta in example.target_delta)


def test_random_stream_record_is_reproducible():
    config = StateVectorConfig()
    left = build_record(7, config, random.Random(17))
    right = build_record(7, config, random.Random(17))
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def test_record_with_canonical_preserves_the_training_record_stream():
    config = StateVectorConfig()
    expected = build_record(9, config, random.Random(31415))
    actual, canonical = build_record_with_canonical(
        9, config, random.Random(31415)
    )
    assert actual == expected
    assert canonical["global"]["floor"] == actual["metadata"]["floor"]
    assert len(canonical["combat"]["hand"]) == actual["metadata"]["hand_size"]


def test_card_play_event_carries_payment_without_generic_play_counters():
    config = StateVectorConfig()
    record = build_record(3, config, random.Random(23))
    first = record["events"][0]
    assert first["kind"] == "EVENT_CARD_PLAY"
    assert first.get("phase") == "COMMIT_PAYMENT"
    assert not any(
        row.get("field", "").endswith(("cards_played", "attacks_played", "skills_played"))
        for event in record["events"]
        for row in event.get("calculations", ())
    )
    assert all(event["kind"] != "EVENT_CARD_COST" for event in record["events"])


def test_relic_counter_updates_have_explicit_metadata_and_stateless_relics_are_zero():
    config = StateVectorConfig()
    for index in range(256):
        record = build_record(index, config, random.Random(8000 + index))
        for relic, runtime in zip(record["relics"], record["runtime"]["relic_counters"]):
            if not relic["counter_present"]:
                assert runtime["present"] is False
                assert runtime["value"] == 0
        for event in record["events"]:
            if event["kind"] == "EVENT_COUNTER":
                assert event["counter_ref"].startswith("relic[")
                assert event["counter_event"]
                assert "counter_threshold" in event


def test_draw_and_relic_trigger_events_use_canonical_owners():
    config = StateVectorConfig()
    for index in range(256):
        record = build_record(index, config, random.Random(7000 + index))
        for event in record["events"]:
            assert "source_rule" not in event
            assert "calculations" in event
            if event["kind"] == "EVENT_DRAW":
                assert event["entity_ref"] == "ENTITY_SELF"


def test_boss_relics_follow_floor_act():
    for floor, expected in ((1, 0), (16, 0), (17, 1), (33, 1), (34, 2), (50, 2)):
        relics, _ = _initial_relics(random.Random(floor), floor)
        assert len(set(relics).intersection(BOSS_RELIC_POOL)) == expected


def test_random_event_stream_fits_default_token_budget():
    config = StateVectorConfig()
    longest = 0
    for index in range(128):
        record = build_record(index, config, random.Random(9000 + index))
        events = tuple(
            VectorEvent(
                event["kind"], event["entity_ref"],
                delta=tuple(event["delta"]) if event.get("delta") is not None else None,
                calculations=tuple(event.get("calculations", ())),
            )
            for event in record["events"]
        )
        longest = max(longest, len(serialize_vector_events(events).tokens))
    assert longest <= 2048


def test_multihit_damage_recomputes_each_independent_repeat_invocation():
    config = StateVectorConfig()
    state = {
        "combat": {
            "turn": 1,
            "player": {
                "hp": 80, "max_hp": 80, "block": 0, "energy": 3,
                "gold": 0, "powers": {"Strength": 3},
                "combat_counters": {},
            },
            "monsters": [{
                "id": f"Target{slot}", "hp": 80, "max_hp": 80, "block": 0,
                "base_damage": 0, "adjusted_damage": 0, "hits": 0,
                "gone": False, "powers": {"Vulnerable": 2},
            } for slot in range(3)],
            "hand": [], "draw": [], "discard": [], "exhaust": [],
        },
        "global": {"relics": [], "relic_counters": []},
    }
    events = _effect_record(
        state, {"id": "Sword Boomerang", "up": 0}, config, random.Random(0)
    )
    damage_events = [event for event in events if event["kind"] == "EVENT_DAMAGE"]
    assert len(damage_events) == 3
    assert len({event["entity_ref"] for event in damage_events}) > 1
    assert all(
        [row["type"] for row in event["calculations"]] == [
            "READ_BASE_DAMAGE", "READ_STRENGTH", "ADD_STRENGTH",
            "READ_VULNERABLE", "MULTIPLY_VULNERABLE",
            "FLOOR_DIVIDE_VULNERABLE",
            "READ_TARGET_BLOCK", "BLOCK_ABSORB", "SUBTRACT_BLOCK",
            "SUBTRACT_BLOCK_DAMAGE", "READ_TARGET_HP", "SUBTRACT_HP",
        ]
        for event in damage_events
    )
    assert all(
        event["calculations"][2]["result"] == 6
        and event["calculations"][4]["result"] == 18
        and event["calculations"][5]["result"] == 9
        for event in damage_events
    )


def test_whirlwind_uses_exact_pre_payment_energy_for_hits():
    config = StateVectorConfig()
    state = {
        "combat": {
            "turn": 1,
            "player": {
                "hp": 80, "max_hp": 80, "block": 0, "energy": 5,
                "gold": 0, "powers": {}, "combat_counters": {},
            },
            "monsters": [
                {
                    "id": f"Target{slot}", "hp": 100, "max_hp": 100,
                    "block": 0, "base_damage": 0, "adjusted_damage": 0,
                    "hits": 0, "gone": False, "powers": {},
                }
                for slot in range(2)
            ],
            "hand": [], "draw": [], "discard": [], "exhaust": [],
        },
        "global": {"relics": [], "relic_counters": []},
    }
    events = _effect_record(
        state, {"id": "Whirlwind", "up": 0}, config, random.Random(0)
    )
    damage_events = [event for event in events if event["kind"] == "EVENT_DAMAGE"]
    assert len(damage_events) == 5 * 2
    # X-cost payment consumes all five energy rather than a capped/random cost.
    energy_delta = events[0]["delta"][config.player_scalar_names.index("energy")]
    assert energy_delta == pytest.approx(-5 / config.energy_scale)
