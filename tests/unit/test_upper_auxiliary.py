from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.core import vocab as V
from rl.algorithms.upper_auxiliary import (
    UpperAuxiliaryConfig, begin_collection, collect, conditional_loss, target_distribution,
)


def test_unsearched_actions_receive_no_conditional_gradient():
    logits = torch.tensor([[1., 2., 90., -4.]], requires_grad=True)
    loss = conditional_loss(logits, torch.tensor([[.8, .2, 0., 0.]]),
                            torch.tensor([[True, True, False, False]]))
    loss.backward()
    assert logits.grad[0, 2:].tolist() == [0., 0.]
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    assert float(logits.grad.sum()) == pytest.approx(0., abs=1e-7)


def test_soft_target_gap_and_additive_value_offset():
    cfg = UpperAuxiliaryConfig()
    assert target_distribution([.4, .405], cfg) is None
    np.testing.assert_allclose(target_distribution([.4, .5], cfg),
                               target_distribution([1000.4, 1000.5], cfg))
    with pytest.raises(FloatingPointError):
        target_distribution([.4, float('nan')], cfg)


def test_collector_budget_does_not_change_rng_or_rollout_inputs():
    model = SimpleNamespace(policy=SimpleNamespace(upper_auxiliary_config=asdict(UpperAuxiliaryConfig())),
                            num_timesteps=0)
    begin_collection(model, 512)
    obs = {'phase': np.array([[V.PHASE_VOCAB_INDEX['shop']]])}
    masks = np.ones((1, 198), bool)
    logits = np.zeros((1, 198))
    rng = torch.random.get_rng_state().clone()
    calls = []
    def worker(name, request):
        calls.append((name, request))
        return {0: dict(reason='insufficient_actions', transitions=0, seconds=0.)}
    for step in range(512):
        collect(model, obs, masks, logits, step, worker)
    assert len(calls) == 4
    assert model._upper_auxiliary['counts']['shop_attempts'] == 4
    assert torch.equal(rng, torch.random.get_rng_state())
    assert masks.all() and not logits.any()


def test_global_auxiliary_gradient_weights_unequal_and_empty_ranks():
    # Runtime scales local mean by local_count / mean_rank_count before the
    # existing mean gradient collective. It must equal one concatenated mean.
    weights = torch.tensor([[.2, -.3]], requires_grad=True)
    x = torch.tensor([[1.], [2.], [4.]])
    targets = torch.tensor([[.7, .3], [.1, .9], [.5, .5]])
    masks = torch.ones((3, 2), dtype=torch.bool)
    conditional_loss(x @ weights, targets, masks).backward()
    expected = weights.grad.clone()
    gradients = []
    for indices in ([0, 1], [2], []):
        weights.grad = None
        if indices:
            (conditional_loss(x[indices] @ weights, targets[indices], masks[indices])
             * len(indices)).backward()  # mean count is 3 samples / 3 ranks
            gradients.append(weights.grad.clone())
        else:
            gradients.append(torch.zeros_like(weights))
    torch.testing.assert_close(torch.stack(gradients).mean(0), expected)
