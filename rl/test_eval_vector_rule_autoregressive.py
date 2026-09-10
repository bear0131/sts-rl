from __future__ import annotations

import pytest
import torch

from rl.eval_vector_rule_autoregressive import (
    _delta_metrics,
    _predicted_serialized,
    build_evaluation_cases,
)
from rl.rule_vector_model import StateVectorConfig, SerializedVectorEvents


def test_fly_kick_case_uses_vulnerable_without_block():
    case = build_evaluation_cases()[0]
    monster = case.state["combat"]["monsters"][0]
    assert monster["block"] == 0
    assert monster["powers"] == {"Vulnerable": 2}
    config = StateVectorConfig()
    damage_positions = [
        index for index, token in enumerate(case.expected.tokens)
        if token == "DELTA_SLOT"
    ]
    damage_position = damage_positions[1]
    damage = case.expected.delta_targets[damage_position]
    assert damage is not None
    assert damage[config.player_dim + config.enemy_scalar_names.index("hp")] == pytest.approx(-0.035)


def test_predicted_delta_rows_align_to_delta_tokens():
    expected = SerializedVectorEvents(
        ("<OUTPUT_BEGIN>", "DELTA_SLOT", "<OUTPUT_END>"),
        (None, (1.0, 2.0), None),
    )
    predicted = _predicted_serialized(
        expected.tokens,
        torch.tensor([[1.0, 2.0], [0.0, 0.0]]),
    )
    metrics = _delta_metrics(expected, predicted)
    assert metrics["compared_delta_slots"] == 1
    assert metrics["mean_absolute_error"] == pytest.approx(0.0)


def test_pommel_case_matches_renderer_state_and_expected_stream():
    from rl.render_rule_vector_examples import _pommel_with_sundial

    case = build_evaluation_cases()[1]
    assert len(case.state["combat"]["draw"]) == 0
    assert len(case.state["combat"]["discard"]) == 3
    assert case.state["global"]["relic_counters"] == [2]
    expected = _pommel_with_sundial()
    assert case.expected.tokens == expected.tokens
    assert case.expected.delta_targets == expected.delta_targets
