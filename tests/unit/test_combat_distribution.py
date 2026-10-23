import numpy as np
import pytest
import torch

from rl.algorithms.combat_distribution import distribution_loss, mean_outcome_loss, outcome_targets, validate_targets
from rl.policies.combat_distribution import CombatDistributionHead, CombatMeanHead


def test_two_mean_loss_has_exactly_bce_and_survivor_mse():
    logits = torch.tensor([[-1.], [1.]], requires_grad=True)
    hp = torch.tensor([[.4], [.7]], requires_grad=True)
    targets = torch.zeros(2, 9)
    targets[0, 0] = 1
    # Obsolete transition labels and boss weights must add no losses.
    targets[:, 2:] = 1
    outputs = dict(win_logits=logits, hp_mean=hp, boss_win_weight=torch.full((2, 1), .9))
    losses, metrics = mean_outcome_loss(outputs, targets, torch.tensor([.8, 0.]))
    torch.testing.assert_close(losses, torch.nn.functional.softplus(torch.ones(2)) + torch.tensor([.16, 0.]))
    assert set(metrics) == {'win_bce', 'hp_mse', 'win_probability', 'win_target', 'hp_given_win'}
    losses.sum().backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0
    torch.testing.assert_close(hp.grad, torch.tensor([[-.8], [0.]]))


def test_two_mean_head_and_microbatch_gradients():
    torch.manual_seed(42)
    head = CombatMeanHead(8)
    states = torch.randn(5, 8)
    targets = torch.zeros(5, 9)
    targets[:2, 0] = 1
    labels = torch.tensor([.3, .7, 0., 0., 0.])
    value, output = head(states)
    assert set(output) == {'win_logits', 'hp_mean'}
    assert output['hp_mean'].shape == (5, 1)
    assert ((output['hp_mean'] > 0) & (output['hp_mean'] < 1)).all()
    torch.testing.assert_close(value, output['win_logits'].sigmoid() * output['hp_mean'])
    def backward(parts):
        head.zero_grad()
        for part in parts:
            _, output = head(states[part])
            loss, _ = mean_outcome_loss(output, targets[part], labels[part])
            (loss.sum()/5).backward()
        return [p.grad.clone() for p in head.parameters()]
    for whole, micro in zip(backward([slice(None)]), backward([slice(0, 2), slice(2, 5)])):
        torch.testing.assert_close(whole, micro, atol=1e-7, rtol=1e-5)


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


def test_disabled_action_effect_loss_ignores_existing_replay_labels():
    logits = torch.zeros(2, 1, requires_grad=True)
    quantiles = torch.full((2, 4), .5, requires_grad=True)
    targets = torch.zeros(2, 9)
    targets[0, 0] = 1
    targets[:, 2] = 1  # Rows collected before disabling may retain these labels.
    targets[:, 3:] = 7
    outputs = dict(win_logits=logits, hp_quantiles=quantiles)
    losses, metrics = distribution_loss(outputs, targets, torch.tensor([.8, 0.]),
                                        auxiliary_coef=0)
    losses.sum().backward()
    assert torch.count_nonzero(metrics['transition_loss']) == 0
    assert logits.grad is not None and quantiles.grad is not None


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


def test_disabled_transition_target_ignores_restored_pending_fight_labels():
    samples = [dict(executed_action=42, transition_target=np.ones(6, np.float32))]
    outcome_targets(samples, 'win', include_transition=False)
    np.testing.assert_array_equal(samples[0]['combat_targets'],
                                  np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.float32))
