import numpy as np
import pytest
import torch

from rl.policies.search_budget import AdaptiveTokenRegretPolicy, BUDGETS, choose_budget
from rl.search.adaptive_budget import audit_targets


def test_independent_audit_keeps_negative_gains_and_identical_action_zero():
    # The search's chosen action can be worse on the independent panel.
    target = audit_targets([[.8, .2], [.6, .4]], [11, 22], [11, 22, 11, 22, 11],
                           [0, 100, 200, 300, 400], 100)
    np.testing.assert_allclose(target[:4], [-4, 0, -4, 0], atol=1e-6)
    np.testing.assert_allclose(target[4:], np.log1p([.1, .2, .3, .4]))


def test_budget_uses_predicted_benefit_and_actual_transition_cost():
    prediction = np.r_[[.2, .3, .4, .5], np.log1p([10, 1000, 2000, 10000])]
    assert BUDGETS[choose_budget(prediction, 2, training=False, round_index=10)] == 32
    assert choose_budget(np.zeros(8), 2, training=False, round_index=10) == 0
    for seed in range(20):
        assert choose_budget(prediction, seed, training=True, round_index=0) == choose_budget(
            prediction, seed, training=True, round_index=0)
    with pytest.raises(ValueError):
        choose_budget(np.full(8, np.nan), 1, training=False, round_index=0)


def test_budget_regression_cannot_change_action_representation():
    torch.manual_seed(93)
    net = AdaptiveTokenRegretPolicy(785)
    # A nonzero final layer makes the gradient-isolation assertion meaningful.
    with torch.no_grad():
        net.budget_head[-1].weight.normal_()
    context = torch.randn(2, 64, requires_grad=True)
    state = torch.randn(2, 32, requires_grad=True)
    scores = torch.randn(2, 3, requires_grad=True)
    prediction = net.predict_budget(context, state, scores, torch.ones(2, 3, dtype=torch.bool), torch.tensor([0., 1.]))
    prediction.square().sum().backward()
    assert context.grad is state.grad is scores.grad is None
    assert all(p.grad is not None for p in net.budget_head.parameters())
    assert not torch.equal(prediction[0], prediction[1])


def test_large_budget_gradient_cannot_shrink_the_action_update():
    from types import SimpleNamespace
    from rl.algorithms.updates import optimizer_step
    policy = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))])
    policy[0].grad, policy[1].grad = torch.ones(1), torch.full((1,), 1000.)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.)
    optimizer_step(SimpleNamespace(policy=policy, max_grad_norm=1), optimizer=optimizer,
                   clip_parameter_groups=[[policy[0]], [policy[1]]])
    torch.testing.assert_close(policy[0], torch.tensor([-1.]), atol=2e-6, rtol=0)
