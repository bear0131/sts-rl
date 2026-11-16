import numpy as np
import pytest
import torch

from env.search.action_selection import turn_consensus_support
from env.search.action_selection import search_filtered_choice
from rl.algorithms.search_auxiliary import improved_prior, search_kl, SearchTrainingConfig


def test_paired_lucky_win_gap_without_turn_evidence_does_not_veto():
    # 21 both win, one only A, seven only B, three both lose: a large .204
    # sample-mean gap can still be uncertain with just 32 paired worlds.
    full = np.array([[1.09, 1.09]]*21 + [[1.09, 0.]] + [[0., 1.09]]*7 + [[0., 0.]]*3)
    short = np.ones_like(full)
    assert full.mean(0)[1]-full.mean(0)[0] > .2
    np.testing.assert_array_equal(turn_consensus_support(full.ravel(), short.ravel(), [0, 1]), [True, True])


@pytest.mark.parametrize('objective', ['turn-consensus-v1', 'turn-consensus-reverse-v1', 'turn-consensus-logit-v1'])
def test_clear_inferiority_overrides_prior_and_preserves_remaining_ratios(objective):
    full = np.tile([0., 1., 1.], (16, 1))
    keep = turn_consensus_support(full.ravel(), np.ones_like(full).ravel(), [0, 1, 2])
    np.testing.assert_array_equal(keep, [False, True, True])
    prior = torch.tensor([[.99, .006, .004]], dtype=torch.float64)
    legal = torch.ones_like(prior, dtype=torch.bool)
    q = improved_prior(prior.log(), torch.tensor([[0., 1., 1.]]), legal,
        torch.tensor([16]), .3, objective=objective, teacher_keep=torch.tensor(keep[None]))
    torch.testing.assert_close(q.exp(), torch.tensor([[0., .6, .4]]))
    current = prior.log().float().requires_grad_()
    value = search_kl(current, q, legal, objective=objective).sum()
    assert torch.isfinite(value)
    value.backward()
    torch.testing.assert_close(current.grad, prior.float()-q.exp(), atol=2e-7, rtol=1e-6)


@pytest.mark.parametrize('objective', ['turn-consensus-v1', 'turn-consensus-reverse-v1', 'turn-consensus-logit-v1'])
def test_undecided_support_keeps_original_distribution_and_illegal_gradient_zero(objective):
    logits = torch.tensor([[3., -1., 999., 2.]], requires_grad=True)
    legal = torch.tensor([[True, True, False, True]])
    q = improved_prior(logits, torch.zeros_like(logits), legal, torch.tensor([8]), .3,
        objective=objective, teacher_keep=legal)
    torch.testing.assert_close(q, logits.detach().masked_fill(~legal, -torch.inf).log_softmax(-1))
    loss = search_kl(logits, q, legal, objective=objective).sum()
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


def test_consensus_acting_separates_uncertain_order_from_clear_inferiority():
    logits = np.log([.99, .006, .004])
    values, visits = np.array([.75, .95, .95]), np.full(3, 32)
    assert search_filtered_choice(logits, values, visits, 23)[0] == 1
    assert search_filtered_choice(logits, values, visits, 23, support=np.ones(3, bool)) == (0, 0.)
    assert search_filtered_choice(logits, values, visits, 23, support=np.array([False, True, True])) == (1, 0.)
    with pytest.raises(ValueError, match='visited legal action'):
        search_filtered_choice(logits, values, visits, 23, support=np.zeros(3, bool))


def test_consensus_acting_requires_evidence_and_retains_seeded_ties():
    with pytest.raises(ValueError, match='matching teacher evidence'):
        SearchTrainingConfig(backend='full480', action_selection='turn-consensus-policy-rank-v1')
    config = SearchTrainingConfig(backend='full480', objective='turn-consensus-v1',
        action_selection='turn-consensus-policy-rank-v1')
    assert config.action_selection == 'turn-consensus-policy-rank-v1'
    choices = [search_filtered_choice(np.zeros(3), np.ones(3), np.ones(3), nonce,
        support=np.array([False, True, True])) for nonce in range(32)]
    assert {action for action, _ in choices} == {1, 2}
    assert all(logp == -np.log(2) for _, logp in choices)


def test_reverse_conditional_penalizes_spreading_into_previously_rare_actions():
    logits = torch.zeros(1, 2, requires_grad=True)
    target = torch.tensor([[.99, .01]]).log()
    legal = torch.ones(1, 2, dtype=torch.bool)
    value = search_kl(logits, target, legal, objective='turn-consensus-reverse-v1').sum()
    value.backward()
    expected = .25*np.log(99.)
    torch.testing.assert_close(logits.grad, torch.tensor([[-expected, expected]], dtype=torch.float32))
    # Original reverse KL is exactly recovered when no action is vetoed.
    torch.testing.assert_close(value, search_kl(logits, target, legal).sum())


def test_reverse_conditional_handles_extreme_rejected_logits_without_infinity():
    logits = torch.tensor([[1000., -1000., -1001., 999.]], requires_grad=True)
    legal = torch.tensor([[True, True, True, False]])
    target = torch.tensor([[-torch.inf, np.log(.6), np.log(.4), -torch.inf]])
    loss = search_kl(logits, target, legal, objective='turn-consensus-reverse-v1').sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 3] == 0
    assert logits.grad[0, 0] > 0 and logits.grad[0, 1] < 0


def test_logit_conditional_restores_relative_scores_and_ignores_constant_shift():
    target = torch.tensor([[.99, .01, 0.]]).log()
    legal = torch.tensor([[True, True, False]])
    logits = torch.tensor([[0., 0., 999.]], requires_grad=True)
    value = search_kl(logits, target, legal, objective='turn-consensus-logit-v1').sum()
    value.backward()
    d = .5*np.log(99.)
    torch.testing.assert_close(value, torch.tensor(d*d, dtype=torch.float32))
    torch.testing.assert_close(logits.grad, torch.tensor([[-d, d, 0.]], dtype=torch.float32))
    shifted = search_kl(logits.detach()+13., target, legal, objective='turn-consensus-logit-v1')
    torch.testing.assert_close(shifted.sum(), value)


def test_logit_conditional_single_survivor_has_finite_hard_correction():
    logits = torch.tensor([[1000., -1000., 999.]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    target = torch.tensor([[-torch.inf, 0., -torch.inf]])
    loss = search_kl(logits, target, legal, objective='turn-consensus-logit-v1').sum()
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(2000.))
    torch.testing.assert_close(logits.grad, torch.tensor([[1., -1., 0.]]))
