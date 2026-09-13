import numpy as np

from rl.evaluation.rules.eval_attack_damage_accuracy import (
    _base_damage_checks,
    _decode_number,
    _hp_damage_checks,
    _sample_mixed_deck,
)


def test_decode_lsd_number():
    assert _decode_number([
        "NUM_LSD_BEGIN", "DIGIT_2", "DIGIT_1", "NUM_END",
    ]) == 12
    assert _decode_number([
        "NUM_LSD_BEGIN", "SIGN_NEG", "DIGIT_5", "NUM_END",
    ]) == -5


def test_base_damage_early_end_is_readable_but_incorrect():
    target = [
        "CALCULATION_BEGIN", "CALCULATION_READ_BASE_DAMAGE",
        "CALC_FIELD_value", "NUM_LSD_BEGIN", "DIGIT_4", "DIGIT_1",
        "NUM_END", "CALCULATION_END",
    ]
    predicted = [
        "CALCULATION_BEGIN", "CALCULATION_READ_BASE_DAMAGE",
        "CALC_FIELD_value", "NUM_LSD_BEGIN", "DIGIT_0", "NUM_END",
        "NUM_END", "CALCULATION_END",
    ]
    checks = _base_damage_checks(target, predicted)
    assert checks[0]["expected"] == 14
    assert checks[0]["predicted"] == 0
    assert not checks[0]["correct"]


def test_hp_damage_ignores_non_monster_effects():
    target = [
        "EFFECT_BEGIN", "EFFECT_PLAYER_VALUE_CHANGE", "FIELD_field", "VALUE_hp",
        "FIELD_delta", "NUM_LSD_BEGIN", "SIGN_NEG", "DIGIT_2", "NUM_END",
        "EFFECT_END",
        "EFFECT_BEGIN", "EFFECT_MONSTER_VALUE_CHANGE", "FIELD_field", "VALUE_hp",
        "FIELD_delta", "NUM_LSD_BEGIN", "SIGN_NEG", "DIGIT_5", "NUM_END",
        "EFFECT_END",
    ]
    checks = _hp_damage_checks(target, target)
    assert len(checks) == 1
    assert checks[0]["expected"] == 5
    assert checks[0]["correct"]


def test_sample_mixed_deck_has_exactly_one_requested_card():
    rng = np.random.default_rng(123)
    for upgraded in (False, True):
        deck, floor = _sample_mixed_deck(rng, "Twin Strike", upgraded)
        assert 1 <= floor <= 50
        assert 1 <= len(deck) <= 39
        assert sum(card_id == "Twin Strike" for card_id, _up in deck) == 1
        assert ("Twin Strike", int(upgraded)) in deck
