from __future__ import annotations

from rl.evaluation.rules.eval_vector_rule_v7_simple_cards import build_cases, build_specs
from rl.rules.rule_vector_model import StateVectorConfig


def test_simple_card_suite_contains_the_previous_probes():
    specs = build_specs()
    names = [spec.name for spec in specs]
    assert names == [
        "strike_base",
        "strike_vulnerable",
        "strike_block",
        "strike_vulnerable_block",
        "strike_tampered_6_to_12",
        "perfected_strike_zero_strikes",
        "perfected_strike_three_strikes",
        "perfected_strike_vulnerable",
        "impervious_base",
        "impervious_frail",
        "impervious_tampered_30_to_60",
    ]
    assert specs[4].tamper_from == 6 and specs[4].tamper_to == 12
    assert specs[10].tamper_from == 30 and specs[10].tamper_to == 60


def test_perfected_strike_cases_preserve_hidden_pile_count_without_composition():
    pairs = build_cases(StateVectorConfig())
    by_name = {spec.name: (spec, case) for spec, case in pairs}
    zero_spec, zero_case = by_name["perfected_strike_zero_strikes"]
    three_spec, three_case = by_name["perfected_strike_three_strikes"]
    assert zero_spec.strike_count == 0
    assert three_spec.strike_count == 3
    assert len(zero_case.canonical["combat"]["draw"]) == 0
    assert len(three_case.canonical["combat"]["draw"]) == 3
    assert all(card["id"] == "Strike_R" for card in three_case.canonical["combat"]["draw"])
    assert abs(three_case.record["state"]["piles"][0] - 0.03) < 1e-6
    assert all("STRIKE_COUNT" not in token for row in three_case.record["node_local_sequences"] for token in row["tokens"])
