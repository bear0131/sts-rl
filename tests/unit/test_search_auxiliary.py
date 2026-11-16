import torch
import pytest
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


@pytest.mark.parametrize('objective', ['prior-advantage-kl-v1', 'filtered-distillation-v1', 'turn-consensus-v1', 'turn-consensus-reverse-v1', 'turn-consensus-logit-v1'])
def test_auxiliary_normalizes_global_searched_rows_before_physical_chunks(objective):
    from rl.algorithms.search_auxiliary import SearchTrainingConfig, prepare_search_batch, auxiliary_loss
    logits = torch.tensor([[.5, -.2], [.1, .4], [.2, .9], [.8, -.2]], requires_grad=True)
    target = torch.tensor([[.1, .9], [.8, .2], [.5, .5], [.4, .6]]).log()
    valid = torch.tensor([0., 1., 0., 0.])
    obs = dict(_search_target=target, _search_valid=valid)
    model = SimpleNamespace(deck_search_config=asdict(SearchTrainingConfig(coefficient=.1,
        backend='full480', action_selection='search-filter-policy-rank-v1', objective=objective)))
    batch = SimpleNamespace(observations=obs, advantages=torch.ones(4))
    # Across four ranks only this rank has one searched row: the mean count
    # is .25, and must not be rounded/clamped up to one.
    prepare_search_batch(model, batch, lambda x: dict(rows=4., search_count=.25))
    whole = auxiliary_loss(model, obs, logits, torch.ones_like(logits, dtype=torch.bool))
    chunks = sum(auxiliary_loss(model, {k: v[i:i+1] for k, v in obs.items()}, logits[i:i+1],
        torch.ones_like(logits[i:i+1], dtype=torch.bool)) / 4 for i in range(4))
    expected = .1 * search_kl(logits[1:2], target[1:2], torch.ones(1, 2, dtype=torch.bool), objective=objective).sum()
    torch.testing.assert_close(whole / 4, expected)  # rank gradient average
    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(torch.autograd.grad(chunks, logits, retain_graph=True)[0],
                               torch.autograd.grad(whole, logits)[0])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_filtered_teacher_preserves_equivalent_choices_and_masks(device):
    if device == 'cuda':
        assert torch.cuda.is_available()
    old = torch.tensor([[1., 999., -3.]], device=device, requires_grad=True)
    legal = torch.tensor([[True, False, True]], device=device)
    q = improved_prior(old, torch.tensor([[.5, float('nan'), .498]], device=device),
        legal, torch.tensor([4], device=device), .3288, objective='filtered-distillation-v1')
    expected = old.detach().masked_fill(~legal, -torch.inf).log_softmax(-1)
    torch.testing.assert_close(q, expected)
    loss = search_kl(old, q, legal, objective='filtered-distillation-v1').sum()
    loss.backward()
    torch.testing.assert_close(old.grad, torch.zeros_like(old), atol=1e-7, rtol=0)
    assert not q.requires_grad and torch.isneginf(q[~legal]).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_filtered_teacher_moves_probability_and_corrects_confident_wrong_policy(device):
    if device == 'cuda':
        assert torch.cuda.is_available()
    logits = torch.tensor([[0., -16., -17., 999.]], device=device, requires_grad=True)
    values = torch.tensor([[.1, .5, .498, float('nan')]], device=device)
    legal = torch.tensor([[True, True, True, False]], device=device)
    target = improved_prior(logits, values, legal, torch.tensor([32], device=device),
        .3288, objective='filtered-distillation-v1')
    probabilities = target.exp()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, device=device))
    assert probabilities[0, 1:3].sum() > .85
    torch.testing.assert_close(target[0, 1]-target[0, 2], torch.tensor(1., device=device))
    loss = search_kl(logits, target, legal, objective='filtered-distillation-v1').sum()
    gradient, = torch.autograd.grad(loss, logits)
    prior = logits.detach().masked_fill(~legal, -torch.inf).softmax(-1)
    torch.testing.assert_close(gradient, prior-probabilities, atol=1e-7, rtol=1e-6)
    assert gradient[0, 1] < -.6 and gradient[0, 0] > .85 and gradient[0, 3] == 0
    # One descent step must increase the probability of the executed best-prior eligible action.
    updated = (logits.detach()-.1*gradient).masked_fill(~legal, -torch.inf).softmax(-1)
    assert updated[0, 1] > prior[0, 1]


def test_filtered_teacher_strength_is_smooth_in_gap_and_increases_with_complete_worlds():
    old = torch.tensor([[.8, .2]]).log()
    legal = torch.ones(1, 2, dtype=torch.bool)
    def probability(gap, worlds):
        return improved_prior(old, torch.tensor([[0., gap]]), legal,
            torch.tensor([worlds]), .3288, objective='filtered-distillation-v1').exp()[0, 1].item()
    assert probability(.004999, 32) == pytest.approx(.2)
    assert abs(probability(.005001, 32)-.2) < .0002
    assert .2 < probability(.008, 4) < probability(.008, 32) < probability(.05, 32) < 1.


def test_filtered_teacher_is_finite_with_extreme_priors_and_rejects_invalid_contracts():
    from rl.algorithms.search_auxiliary import SearchTrainingConfig
    legal = torch.tensor([[True, True, False]])
    old = torch.tensor([[10000., -10000., float('inf')]], requires_grad=True)
    target = improved_prior(old, torch.tensor([[0., 1., float('nan')]]), legal,
        torch.tensor([32]), .3288, objective='filtered-distillation-v1')
    assert torch.isfinite(target[legal]).all() and torch.isneginf(target[~legal]).all()
    search_kl(old, target, legal, objective='filtered-distillation-v1').sum().backward()
    assert torch.isfinite(old.grad).all() and old.grad[0, 2] == 0
    assert SearchTrainingConfig(**{}).objective == 'prior-advantage-kl-v1'
    with pytest.raises(ValueError, match='full480'):
        SearchTrainingConfig(objective='filtered-distillation-v1')
    with pytest.raises(ValueError, match='unknown'):
        SearchTrainingConfig(objective='unknown')
    with pytest.raises(ValueError, match='positive world'):
        improved_prior(old, torch.zeros_like(old), legal, torch.tensor([float('nan')]), .3,
            objective='filtered-distillation-v1')


def test_chain_mixture_only_accepts_original_full480_anchor_supervision():
    from rl.algorithms.search_auxiliary import SearchTrainingConfig
    config = dict(backend='full480', action_selection='search-filter-policy-rank-v1', value_mix=.25)
    assert SearchTrainingConfig(**config).value_mix == .25
    assert SearchTrainingConfig().value_mix == 0.
    for value in (-.1, 1.1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='weight'):
            SearchTrainingConfig(**dict(config, value_mix=value))
    for changes in (dict(backend='native-cpu'), dict(action_selection='shadow'),
                    dict(objective='turn-consensus-logit-v1')):
        with pytest.raises(ValueError, match='original full480'):
            SearchTrainingConfig(**dict(config, **changes))
