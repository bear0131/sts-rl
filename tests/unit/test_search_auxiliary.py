import torch
from types import SimpleNamespace
from dataclasses import asdict

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


def test_auxiliary_normalizes_global_searched_rows_before_physical_chunks():
    from rl.algorithms.search_auxiliary import SearchTrainingConfig, prepare_search_batch, auxiliary_loss
    logits = torch.tensor([[.5, -.2], [.1, .4], [.2, .9], [.8, -.2]], requires_grad=True)
    target = torch.tensor([[.1, .9], [.8, .2], [.5, .5], [.4, .6]]).log()
    valid = torch.tensor([0., 1., 0., 0.])
    obs = dict(_search_target=target, _search_valid=valid)
    model = SimpleNamespace(deck_search_config=asdict(SearchTrainingConfig(coefficient=.1)))
    batch = SimpleNamespace(observations=obs, advantages=torch.ones(4))
    # Across four ranks only this rank has one searched row: the mean count
    # is .25, and must not be rounded/clamped up to one.
    prepare_search_batch(model, batch, lambda x: dict(rows=4., search_count=.25))
    whole = auxiliary_loss(model, obs, logits, torch.ones_like(logits, dtype=torch.bool))
    chunks = sum(auxiliary_loss(model, {k: v[i:i+1] for k, v in obs.items()}, logits[i:i+1],
        torch.ones_like(logits[i:i+1], dtype=torch.bool)) / 4 for i in range(4))
    expected = .1 * search_kl(logits[1:2], target[1:2], torch.ones(1, 2, dtype=torch.bool)).sum()
    torch.testing.assert_close(whole / 4, expected)  # rank gradient average
    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(torch.autograd.grad(chunks, logits, retain_graph=True)[0],
                               torch.autograd.grad(whole, logits)[0])
