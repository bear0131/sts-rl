import torch

from rl.algorithms.search_auxiliary import improved_prior, search_kl


def test_equal_values_preserve_prior_and_illegal_atoms_have_no_gradient():
    old = torch.tensor([[1., 4., -3.]], requires_grad=True)
    legal = torch.tensor([[True, False, True]])
    q = improved_prior(old, torch.tensor([[.2, float('nan'), .2]]), legal, torch.tensor([2]), .1)
    expected = old.detach().masked_fill(~legal, -torch.inf).log_softmax(-1)
    torch.testing.assert_close(q, expected)
    current = torch.tensor([[.1, 999., -.1]], requires_grad=True)
    loss = search_kl(current, q, legal).sum()
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 1] == 0
    assert old.grad is None and not q.requires_grad


def test_more_worlds_strengthen_bounded_improvement_and_constant_shift_cancels():
    logits = torch.tensor([[.5, 0.]])
    legal = torch.ones(1, 2, dtype=torch.bool)
    values = torch.tensor([[0., 1.]])
    low = improved_prior(logits, values, legal, torch.tensor([1]), .2)
    high = improved_prior(logits, values, legal, torch.tensor([20]), .2)
    shifted = improved_prior(logits, values + 100, legal, torch.tensor([20]), .2)
    assert high[0, 1] > low[0, 1]
    torch.testing.assert_close(high, shifted)
    odds_change = (high[0, 1] - high[0, 0]) - (logits[0, 1] - logits[0, 0])
    assert 0 < odds_change <= 4 * 20 / 24 + 1e-6  # FP32 log-softmax rounding
