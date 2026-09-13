from __future__ import annotations

from rl.eval_vector_rule_v7_cases import (
    BEGIN_TOKENS,
    END_TOKENS,
    _expected_value_bindings,
    _render_tokens,
    _token_metrics,
    build_cases,
)
from rl.rule_vector_model import StateVectorConfig


def test_free_outer_metrics_penalize_missing_and_extra_tokens():
    missing = _token_metrics(("A", "B", "C"), ("A", "B"))
    assert not missing["exact_match"]
    assert missing["scored_tokens"] == 3
    assert missing["correct_tokens"] == 2
    assert missing["first_mismatch"] == 2

    extra = _token_metrics(("A", "B"), ("A", "B", "C"))
    assert not extra["exact_match"]
    assert extra["scored_tokens"] == 3
    assert extra["correct_tokens"] == 2
    assert extra["first_mismatch"] == 2


def test_local_metrics_score_only_model_owned_positions():
    metrics = _token_metrics(
        ("PREFIX", "MODEL", "ENV", "MODEL_2"),
        ("PREFIX", "MODEL", "WRONG_ENV", "MODEL_2"),
        positions=(1, 3),
    )
    assert metrics["exact_match"]
    assert metrics["token_accuracy"] == 1.0


def test_interactive_evaluator_recovers_runtime_binding_values():
    bindings = _expected_value_bindings([
        "<READ_REQUEST_BEGIN>", "READ_SOURCE", "VALUE_enemy_0_hp",
        "READ_BIND", "VALUE_reg_target_hp", "<READ_REQUEST_END>",
        "CALC_FIELD_left", "VALUE_reg_target_hp", "CONST_VALUE",
        "NUM_LSD_BEGIN", "DIGIT_0", "DIGIT_0", "DIGIT_1", "NUM_END",
    ])
    assert len(bindings) == 1
    assert bindings[0].request_id == "r0"
    assert bindings[0].source_token == "VALUE_enemy_0_hp"
    assert bindings[0].binding_token == "VALUE_reg_target_hp"
    assert bindings[0].value == 100


def test_v7_renderer_has_no_removed_numeric_response_or_register_context():
    removed = {
        "<READ_RESPONSE_BEGIN>", "<READ_RESPONSE_END>",
        "<REGISTER_CONTEXT_BEGIN>", "<REGISTER_CONTEXT_END>",
        "<REGISTER_BEGIN>", "<REGISTER_END>",
    }
    assert not removed.intersection(BEGIN_TOKENS)
    assert not removed.intersection(END_TOKENS)
    rendered = _render_tokens((
        "<READ_REQUEST_BEGIN>", "READ_SOURCE", "VALUE_enemy_0_hp",
        "READ_BIND", "VALUE_reg_target_hp", "<READ_REQUEST_END>",
        "<CALCULATIONS_BEGIN>", "CALCULATION_BEGIN", "CONST_VALUE",
        "NUM_LSD_BEGIN", "DIGIT_0", "DIGIT_5", "NUM_END",
        "<CALCULATIONS_END>",
    ))
    assert "<READ_RESPONSE" not in rendered
    assert "<REGISTER_CONTEXT" not in rendered


def test_case_builder_keeps_two_fixed_and_unfiltered_random_cases():
    cases = build_cases(
        StateVectorConfig(), random_seed=20260831, random_cases=10
    )
    assert len(cases) == 12
    assert [case.group for case in cases[:2]] == ["fixed", "fixed"]
    assert all(case.group == "random" for case in cases[2:])
    assert [case.name for case in cases[2:]] == [
        f"随机例{index}" for index in range(1, 11)
    ]
    assert all(1 <= len(case.hand) <= 5 for case in cases[2:])
