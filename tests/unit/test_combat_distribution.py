import numpy as np
import pytest
import torch

from rl.algorithms.combat_distribution import distribution_loss, outcome_targets, validate_targets
from rl.policies.combat_distribution import CombatDistributionHead


def test_search_utility_is_expected_capped_hp_and_quantiles_are_ordered():
    torch.manual_seed(42)
    head = CombatDistributionHead(8, 32)
    value, output = head(torch.randn(5, 8))
    q = output['hp_quantiles']
    assert ((q >= 0) & (q <= 1)).all() and (q[:, 1:] >= q[:, :-1]).all()
    torch.testing.assert_close(value[:, 0], output['win_logits'][:, 0].sigmoid()*q.mean(-1))


def test_loss_does_not_teach_dead_fights_a_zero_survivor_distribution():
    logits = torch.zeros(2, 1, requires_grad=True)
    q = torch.full((2, 4), .5, requires_grad=True)
    output = dict(win_logits=logits,hp_quantiles=q,transition=torch.zeros(2, 6))
    targets = torch.zeros(2, 9); targets[0, 0] = 1
    losses, metrics = distribution_loss(output,targets,torch.tensor([.8, 0.]),mean_coef=0)
    losses.sum().backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0
    assert (q.grad[0] < 0).all() and (q.grad[1] == 0).all()
    assert metrics['hp_pinball'][1] == 0


def test_microbatching_preserves_distribution_gradients_with_unequal_win_counts():
    torch.manual_seed(42)
    head = CombatDistributionHead(8, 4)
    states = torch.randn(5, 8)
    targets = torch.zeros(5, 9); targets[:2, 0] = 1
    values = torch.tensor([.3, .7, 0., 0., 0.])
    def backward(parts):
        head.zero_grad()
        for part in parts:
            _, output = head(states[part]); output['transition'] = torch.zeros(len(states[part]), 6)
            losses, _ = distribution_loss(output, targets[part], values[part])
            (losses.sum()/5).backward()
        return [p.grad.clone() for p in head.parameters()]
    for whole, micro in zip(backward([slice(None)]), backward([slice(0, 2), slice(2, 5)])):
        torch.testing.assert_close(whole, micro, atol=1e-7, rtol=1e-5)


def test_outcome_is_explicit_and_censored_fights_are_not_deaths():
    samples=[dict(value_target=0.)]
    outcome_targets(samples,'win')
    assert samples[0]['combat_targets'][0] == 1  # never infer outcome from HP
    for outcome in ('escape', 'truncated', 'unknown'):
        with pytest.raises(ValueError,match='observed win or loss'):
            outcome_targets(samples,outcome)
    invalid=np.zeros((1,9),np.float32)
    with pytest.raises(ValueError,match='zero utility'):
        validate_targets(invalid,np.array([.2]))
