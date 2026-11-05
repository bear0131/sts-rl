import pytest

from env.search.action_count_budget import ActionCountBudget, calibrate_actions


def test_forced_moves_and_large_pools_get_their_intended_budget():
    for mode in ("linear", "tapered", "fixed"):
        b = ActionCountBudget(mode, 40, 160)
        assert b.simulations(1) == 0
        with pytest.raises(ValueError):
            b.simulations(0)
    linear = ActionCountBudget("linear", 40, 160)
    taper = ActionCountBudget("tapered", 40, 160)
    assert linear.simulations(6) == taper.simulations(6) == 240
    assert linear.simulations(32) == 1280 and taper.simulations(32) == 640
    assert taper.simulations(1024) == 8192
    assert taper.simulations(1024, True) == 32768
    seq = [taper.simulations(n) for n in range(2, 1025)]
    assert seq == sorted(seq)


def test_scales_use_observed_action_frequencies_and_exclude_forced_moves():
    h = {1: 10000, 5: 30, 10: 10}
    assert calibrate_actions(h, "linear", False) == 40
    c = calibrate_actions(h, "tapered", False)
    b = ActionCountBudget("tapered", c, c)
    assert abs(sum(b.simulations(a) * n for a, n in h.items()) - 250 * 40) <= 20
    with pytest.raises(ValueError):
        calibrate_actions({1024: 1}, "tapered", False)


def test_ten_real_card_decisions_do_not_get_a_thousand_action_budgets():
    masks = [(3 << 29) | m for m in range(1024)]
    b = ActionCountBudget("linear", 40, 169)
    assert b.for_root(masks) == 400 and b.for_root(masks, True) == 1690
    assert b.for_root([3 << 29]) == 0
    assert b.for_root([0, 4 << 29]) == 80
