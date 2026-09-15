import pytest

from rl.training.monitoring import HP_ORDER_PREFIX as P, hp_order_fraction


def test_unequal_rank_populations_use_ratio_of_counts():
    # Rank 0: 1/1; rank 1: 0/99. Mean-of-ratios is .5; global is .01.
    assert hp_order_fraction({P + '_count': 50, P + '_matches': .5}) == .01
    assert hp_order_fraction({P + '_count': 100, P + '_matches': 1}) == .01


def test_empty_population_is_not_evidence_of_perfect_calibration():
    assert hp_order_fraction({P + '_count': 0, P + '_matches': 0}) is None


@pytest.mark.parametrize('count,matches', [(1, 2), (-1, 0), (2, -1), (2, float('nan'))])
def test_invalid_counts_are_rejected(count, matches):
    with pytest.raises(ValueError):
        hp_order_fraction({P + '_count': count, P + '_matches': matches})
