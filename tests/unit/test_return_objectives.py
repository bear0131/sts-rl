from types import SimpleNamespace
import numpy as np
import torch

from tools.analysis.return_objectives import objective_loss


def test_regret_optimum_combines_cost_and_prior_and_ignores_unobserved_actions():
    bank = SimpleNamespace(
        prior_logp=np.log([0.25, 0.5, 0.25]), action_cost=np.array([0.0, 999.0, 2.0])
    )
    legal = torch.tensor([[True, True, True]])
    observed = torch.tensor([[True, False, True]])
    # On the measured support the old policy is uniform; two utility units cost a factor of four.
    logits = torch.tensor([[np.log(0.8), -np.inf, np.log(0.2)]], requires_grad=True)
    loss = objective_loss(
        "regret", bank, np.array([0]), np.arange(3), logits, legal, observed
    )
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=1e-7)


def test_uncertain_pairs_leave_only_the_prior_and_reliable_pairs_prefer_the_better_move():
    bank = SimpleNamespace(
        prior_logp=np.log([0.3, 0.7]),
        pair_lengths=np.array([0]),
        pair_starts=np.array([0]),
        pair_den=np.array([1]),
        pair_rows=np.zeros((0, 4)),
    )
    obs = torch.tensor([[True, True]])
    logits = torch.tensor([[np.log(0.3), np.log(0.7)]], requires_grad=True)
    loss = objective_loss(
        "preferences", bank, np.array([0]), np.arange(2), logits, obs, obs
    )
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=1e-7)
    logits.grad = None
    bank.pair_lengths[0] = 1
    bank.pair_rows = np.array([[0, 1, 1.0, 2.0]], np.float32)
    objective_loss(
        "preferences", bank, np.array([0]), np.arange(2), logits, obs, obs
    ).backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
