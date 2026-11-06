import numpy as np
import pytest

from tools.validation.policy_scores import ScoreTolerance, compare_policy_scores


FP32 = ScoreTolerance(absolute=3e-5, relative=1e-5, probability=2e-5, kl=1e-8)


def test_scaled_fp32_rounding_preserves_decision_distribution():
    reference = np.array([-32.0, 17.8713646, 58.4253387])
    candidate = reference + np.array([0.0, 3.0517578125e-5, 1.9e-5])
    result = compare_policy_scores(candidate, reference, FP32)
    assert result["max_abs_error"] > FP32.absolute
    assert result["max_scaled_error"] < 1
    assert result["max_probability_error"] < 1e-10
    assert result["top1_equal"]


def test_large_common_offset_cannot_hide_probability_error():
    with pytest.raises(AssertionError, match="probability"):
        compare_policy_scores([100.0001, 99.9999], [100, 100], FP32)


def test_kl_guard_is_independent_of_pointwise_probability_guard():
    limits = ScoreTolerance(absolute=3e-5, relative=0, probability=0.1, kl=1e-10)
    with pytest.raises(AssertionError, match="KL"):
        compare_policy_scores([2e-5, -2e-5], [0, 0], limits)


def test_score_guard_catches_large_shift_even_if_softmax_is_identical():
    with pytest.raises(AssertionError, match="logits"):
        compare_policy_scores([101, 101], [100, 100], FP32)


def test_near_tie_reports_top1_without_mistaking_it_for_large_error():
    result = compare_policy_scores([0, 1e-7], [1e-7, 0], FP32)
    assert not result["top1_equal"]
    assert result["total_variation"] < 1e-7


@pytest.mark.parametrize("candidate,reference", [([], []), ([1], [1, 2]), ([[1]], [[1]]), ([np.nan], [0]), ([0], [np.inf])])
def test_invalid_or_masked_inputs_are_rejected(candidate, reference):
    with pytest.raises(ValueError):
        compare_policy_scores(candidate, reference, FP32)


def test_extreme_values_cannot_silently_pass_nan_probability_metrics():
    with pytest.raises(ValueError, match="overflowed"):
        compare_policy_scores([1e308, -1e308], [1e308, -1e308], FP32)


def test_zero_tolerance_accepts_only_identical_scores():
    exact = ScoreTolerance(0, 0, 0, 0)
    assert compare_policy_scores([3, -5], [3, -5], exact)["max_scaled_error"] == 0
    with pytest.raises(AssertionError, match="logits"):
        compare_policy_scores([3.00001, -5], [3, -5], exact)


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_invalid_tolerances_are_rejected(value):
    with pytest.raises(ValueError):
        ScoreTolerance(value, 0, 0, 0)
