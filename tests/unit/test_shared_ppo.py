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
