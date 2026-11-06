import pytest

from env.search.action_count_budget import (
    ActionCountBudget, allocate_batch_budgets, calibrate_actions,
    effective_action_width,
)


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


def test_batch_redistribution_preserves_total_and_changes_relative_effort():
    assert allocate_batch_budgets([2, 4, 6], 250, "fixed") == [250] * 3
    assert allocate_batch_budgets([2, 4, 6], 250, "linear") == [125, 250, 375]
    assert allocate_batch_budgets([4, 16], 250, "linear") == [100, 400]
    assert allocate_batch_budgets([4, 16], 250, "tapered") == [131, 369]
    assert allocate_batch_budgets([1, 1, 100], 250, "linear") == [63, 63, 624]
    assert effective_action_width([(3 << 29) | m for m in range(1024)]) == 10


def test_exact_budget_and_bounds_hold_for_small_and_large_tier_batches():
    from itertools import product
    for mode in ("linear", "tapered"):
        for nominal in (1, 250, 1000):
            for size in range(1, 5):
                for widths in product((1, 2, 8, 64), repeat=size):
                    result = allocate_batch_budgets(widths, nominal, mode)
                    assert sum(result) == nominal * size
                    assert all((nominal + 3) // 4 <= b <= 4 * nominal for b in result)
                    assert all(result[i] <= result[j] for i in range(size)
                               for j in range(size) if widths[i] < widths[j])
        result = allocate_batch_budgets([1] * 479 + [1024], 250, mode)
        assert sum(result) == 120000 and result[-1] == 1000


def test_stable_root_keys_preserve_rounding_when_queue_order_changes():
    widths, keys = [2, 2, 3], [14, 7, 9]
    baseline = dict(zip(keys, allocate_batch_budgets(widths, 250, "linear", keys=keys)))
    order = [2, 0, 1]
    reordered = allocate_batch_budgets([widths[i] for i in order], 250, "linear",
                                       keys=[keys[i] for i in order])
    assert dict(zip([keys[i] for i in order], reordered)) == baseline
    assert allocate_batch_budgets([], 250, "linear") == []
    assert allocate_batch_budgets([1024], 250, "tapered") == [250]
    for widths, budget, mode, keys in [([0], 250, "linear", [0]),
                                      ([2], 0, "linear", [0]),
                                      ([2], 250, "unknown", [0]),
                                      ([2, 3], 250, "linear", [1, 1])]:
        with pytest.raises(ValueError):
            allocate_batch_budgets(widths, budget, mode, keys=keys)
