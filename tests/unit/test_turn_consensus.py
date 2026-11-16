import numpy as np
import pytest
import torch

from env.search.action_selection import turn_consensus_support
from rl.algorithms.search_auxiliary import improved_prior, search_kl, SearchTrainingConfig


def test_paired_lucky_win_gap_without_turn_evidence_does_not_veto():
    # 21 both win, one only A, seven only B, three both lose: a large .204
    # sample-mean gap can still be uncertain with just 32 paired worlds.
    full = np.array([[1.09, 1.09]]*21 + [[1.09, 0.]] + [[0., 1.09]]*7 + [[0., 0.]]*3)
    short = np.ones_like(full)
    assert full.mean(0)[1]-full.mean(0)[0] > .2
    np.testing.assert_array_equal(turn_consensus_support(full.ravel(), short.ravel(), [0, 1]), [True, True])


def test_clear_inferiority_overrides_prior_and_preserves_remaining_ratios():
    full = np.tile([0., 1., 1.], (16, 1))
    keep = turn_consensus_support(full.ravel(), np.ones_like(full).ravel(), [0, 1, 2])
    np.testing.assert_array_equal(keep, [False, True, True])
    prior = torch.tensor([[.99, .006, .004]], dtype=torch.float64)
    legal = torch.ones_like(prior, dtype=torch.bool)
    q = improved_prior(prior.log(), torch.tensor([[0., 1., 1.]]), legal,
        torch.tensor([16]), .3, objective='turn-consensus-v1', teacher_keep=torch.tensor(keep[None]))
    torch.testing.assert_close(q.exp(), torch.tensor([[0., .6, .4]]))
    current = prior.log().float().requires_grad_()
    value = search_kl(current, q, legal, objective='turn-consensus-v1').sum()
    assert torch.isfinite(value)
    value.backward()
    torch.testing.assert_close(current.grad, prior.float()-q.exp(), atol=2e-7, rtol=1e-6)


def test_undecided_support_keeps_original_distribution_and_illegal_gradient_zero():
    logits = torch.tensor([[3., -1., 999., 2.]], requires_grad=True)
    legal = torch.tensor([[True, True, False, True]])
    q = improved_prior(logits, torch.zeros_like(logits), legal, torch.tensor([8]), .3,
        objective='turn-consensus-v1', teacher_keep=legal)
    torch.testing.assert_close(q, logits.detach().masked_fill(~legal, -torch.inf).log_softmax(-1))
    loss = search_kl(logits, q, legal, objective='turn-consensus-v1').sum()
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=2e-7, rtol=0)
    assert not q.requires_grad and logits.grad[0, 2] == 0


def test_world_pairing_action_permutation_and_partial_tail():
    full = np.tile([1., 0., 1.], (8, 1))
    # Ledger columns correspond to public actions 2, 0, 1.
    expected = [False, True, True]
    np.testing.assert_array_equal(turn_consensus_support(np.r_[full.ravel(), -99.],
        np.r_[full.ravel(), 99.], [2, 0, 1]), expected)
    np.testing.assert_array_equal(turn_consensus_support((full+200).ravel(),
        (full-100).ravel(), [2, 0, 1]), expected)


def test_short_only_preferences_and_too_few_worlds_do_not_supply_a_veto():
    np.testing.assert_array_equal(turn_consensus_support(np.tile([.7, .7], 16),
        np.tile([0., 1.], 16), [0, 1]), [True, True])
    np.testing.assert_array_equal(turn_consensus_support(np.tile([0., 1.], 3), None, [0, 1]), [True, True])
    with pytest.raises(ValueError, match='matching natural-turn'):
        turn_consensus_support(np.ones(8), None, [0, 1])
    with pytest.raises(ValueError, match='explicit legal action permutation'):
        turn_consensus_support(np.ones(8), np.ones(8), [0, 0])


def test_consensus_configuration_and_support_fail_closed():
    with pytest.raises(ValueError, match='full480'):
        SearchTrainingConfig(objective='turn-consensus-v1')
    assert SearchTrainingConfig(objective='turn-consensus-v1', backend='full480',
        action_selection='search-filter-policy-rank-v1').objective == 'turn-consensus-v1'
    with pytest.raises(ValueError, match='aligned nonempty legal support'):
        improved_prior(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([4]), .3, objective='turn-consensus-v1', teacher_keep=torch.zeros(1, 2, dtype=torch.bool))
