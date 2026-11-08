"""Combat supervision belongs to AZ in both local and synchronized PPO."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

from gymnasium import spaces
import pytest
import torch

from env.core import vocab as V
from rl.algorithms.updates import ppo_forward
from rl.training.distributed_sb3 import ppo_microbatch_backward


@pytest.mark.parametrize('only_combat', [False, True])
@pytest.mark.parametrize('phasewise', [False, True])
def test_nosl_combat_excluded_from_local_and_distributed_gradients(only_combat, phasewise):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(2, 3))

        def evaluate_actions(self, obs, actions, action_masks=None):
            scores = self.scores[obs['row'].flatten()]
            return scores[:, :1], scores[:, 1], scores[:, 2]

    def model():
        return SimpleNamespace(policy=Policy(), action_space=spaces.Discrete(2),
            device=torch.device('cpu'), hierarchical_combat_hp=True, nosl_is_az=True,
            normalize_advantage=True, ent_coef=0.01, vf_coef=0.5,
            autocast_context=nullcontext)

    local = model()
    distributed = deepcopy(local)
    combat, overworld = V.PHASE_VOCAB_INDEX['combat'], V.PHASE_VOCAB_INDEX['map']
    batch = SimpleNamespace(observations={
        'phase': torch.tensor([[combat], [combat if only_combat else overworld]]),
        'row': torch.tensor([[0], [1]])},
        actions=torch.zeros(2), action_masks=torch.ones(2, 2, dtype=torch.bool),
        advantages=torch.tensor([1000., 2.]), returns=torch.tensor([1000., 3.]),
        old_values=torch.zeros(2), old_log_prob=torch.zeros(2))
    components, loss = ppo_forward(local, batch, 0.2, None)
    loss.backward()
    result = ppo_microbatch_backward(distributed, batch, 0.2, None,
                                    phasewise_backward=phasewise)
    assert components.combat_samples == 0
    assert components.valid_samples == (0 if only_combat else 1)
    assert result['loss'] == loss.item()
    assert torch.equal(local.policy.scores.grad, distributed.policy.scores.grad)
    assert torch.count_nonzero(local.policy.scores.grad[0]) == 0
    assert bool(torch.count_nonzero(local.policy.scores.grad[1])) == (not only_combat)


def test_fused_ppo_preserves_normalization_groups_gradient_and_adam():
    from rl.algorithms.ppo import ppo_loss_components
    torch.manual_seed(47)
    original = torch.nn.Parameter(torch.randn(16, 3, dtype=torch.float64) * .1)
    fused = torch.nn.Parameter(original.detach().clone())
    # Deliberately different scales/means: whole-batch normalization would
    # materially change this objective despite identical optimizer batch size.
    advantage = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    advantage = (advantage * torch.tensor([1., 10., .1, 100.])[:, None]).flatten()
    def loss(parameter, sl, group=None):
        x = parameter[sl]
        c = ppo_loss_components(hierarchical=False, normalize_advantage=True,
            observations={}, advantages=advantage[sl], returns=advantage[sl] * .001,
            old_values=torch.zeros_like(x[:, 0]), old_log_prob=torch.zeros_like(x[:, 0]),
            values=x[:, 0], log_prob=x[:, 1], entropy=x[:, 2],
            clip_range=.2, clip_range_vf=None, normalization_group_size=group)
        return c.policy_loss + .5*c.value_loss + .02*c.entropy_loss
    expected = sum(loss(original, slice(i, i+4)) for i in range(0, 16, 4)) / 4
    actual = loss(fused, slice(None), 4)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected.backward(); actual.backward()
    torch.testing.assert_close(original.grad, fused.grad, rtol=1e-12, atol=1e-12)
    for parameter in (original, fused):
        torch.optim.Adam([parameter], lr=1e-4).step()
    torch.testing.assert_close(original, fused, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='complete advantage'):
        loss(fused, slice(0, 15), 4)


def test_oom_retry_preserves_prior_and_partial_backward_gradients(monkeypatch):
    from rl.algorithms import updates
    class Policy(torch.nn.Module):
        ppo_advantage_group_size = 4
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.3, -.7], dtype=torch.float64))
    original = SimpleNamespace(policy=Policy(), device=torch.device('cpu'))
    retry = deepcopy(original)
    batch = SimpleNamespace(advantages=torch.arange(16, dtype=torch.float64))
    def forward(model, data, *args):
        x = data.advantages[:, None] * model.policy.weight[None, :]
        return None, x.square().mean()
    monkeypatch.setattr(updates, 'ppo_forward', forward)
    for model in (original, retry):
        model.policy.weight.grad = torch.tensor([2., 3.], dtype=torch.float64)
    expected_loss = updates.ppo_chunked_backward(original, batch, .2, None)
    class PartialFailedBackward:
        def __mul__(self, scale):
            return self
        def backward(self):
            retry.policy.weight.grad = torch.full_like(retry.policy.weight, 999.)
            raise torch.OutOfMemoryError('injected after a partial gradient write')
    def failing_forward(model, data, *args):
        if len(data.advantages) > 8:
            return None, PartialFailedBackward()
        return forward(model, data, *args)
    monkeypatch.setattr(updates, 'ppo_forward', failing_forward)
    actual_loss = updates.ppo_chunked_backward(retry, batch, .2, None)
    assert actual_loss == pytest.approx(expected_loss, rel=1e-12)
    torch.testing.assert_close(original.policy.weight.grad, retry.policy.weight.grad,
                               rtol=1e-12, atol=1e-12)
    assert retry._ppo_microbatch_stats == dict(chunks=2, rows=16, oom_retries=1,
                                              min_rows=8, max_rows=8)
    assert retry._ppo_microbatch_cap == 8
    for model in (original, retry):
        torch.optim.Adam(model.policy.parameters(), lr=1e-4).step()
    torch.testing.assert_close(original.policy.weight, retry.policy.weight,
                               rtol=1e-12, atol=1e-12)
