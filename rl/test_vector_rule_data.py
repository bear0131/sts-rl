from __future__ import annotations

import pytest
import torch

from rl.rule_vector_model import StateVectorConfig, VECTOR_TOKEN_INDEX
from rl.vector_rule_data import (
    VECTOR_DATA_SCHEMA_VERSION,
    VectorRuleV6Dataset,
    collate_vector_rule_v6_batch,
    example_from_v6_record,
)


def _sample():
    config = StateVectorConfig()
    player = [0.0] * config.player_dim
    player[config.player_scalar_names.index("energy")] = 0.3
    enemies = [[0.0] * config.enemy_dim for _ in range(config.max_enemies)]
    enemies[0][config.enemy_scalar_names.index("hp")] = 0.25
    enemies[0][config.enemy_scalar_names.index("block")] = 0.04
    return {
        "schema_version": VECTOR_DATA_SCHEMA_VERSION,
        "state": {
            "player": player,
            "enemies": enemies,
            "piles": [0.01, 0.0, 0.0],
            "relic_counters": [],
        },
        "action": {"card_rule": [1, 2, 3], "upgrade": 0},
        "hand": [{"card_rule": [4, 5], "upgrade": 0}],
        "relics": [],
        "events": [
            {
                "kind": "EVENT_CARD_PLAY",
                "entity_ref": "ENTITY_SELF",
                "delta": [
                    -0.1 if index == config.player_scalar_names.index("energy") else 0.0
                    for index in range(config.world_dim)
                ],
            },
            {
                "kind": "EVENT_ACTION_COMPLETE",
                "entity_ref": "ENTITY_SELF",
            },
        ],
    }


def test_v6_record_is_vector_only_and_validates_shapes():
    example = example_from_v6_record(_sample(), StateVectorConfig(), card_vocab_size=32)
    assert example.state.player.shape == (StateVectorConfig().player_dim,)
    assert example.state.enemies.shape == (StateVectorConfig().max_enemies, StateVectorConfig().enemy_dim)
    assert example.action_program == (1, 2, 3)
    assert example.target_ids[0] == VECTOR_TOKEN_INDEX["<OUTPUT_BEGIN>"]
    assert any(delta is not None for delta in example.target_delta)


def test_v6_collate_pads_programs_and_shifted_targets():
    dataset = VectorRuleV6Dataset([_sample(), _sample()], StateVectorConfig(), card_vocab_size=32)
    assert len(dataset) == 2
    batch = collate_vector_rule_v6_batch([dataset[0], dataset[1]])
    assert batch["target_input_ids"].shape == batch["labels"].shape
    assert batch["action_programs"].shape == (2, 3)
    assert batch["hand_programs"].shape[:2] == (2, 1)
    assert batch["delta_targets"].shape[-1] == StateVectorConfig().world_dim
    assert batch["delta_mask"].dtype == torch.bool
    assert int(batch["delta_mask"].sum()) > 0


def test_legacy_record_is_rejected_instead_of_silently_converted():
    legacy = {"state": {}, "action": {}, "effects": [], "next_state": {}}
    with pytest.raises(ValueError, match="not a v6 vector-rule record"):
        example_from_v6_record(legacy, StateVectorConfig())
