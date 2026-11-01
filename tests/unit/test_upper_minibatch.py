"""Unequal/empty ranks consume real rows once and agree on every Adam step."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch

from env.core import vocab as V
from rl.algorithms.upper_ppo import update_upper
from rl.policies.entity_joint import UpperConfig
from rl.training.distributed_sb3 import pack_gradients, install_flat_gradients


class Policy(torch.nn.Module):
    def __init__(self, *, kl_stop=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.2, -.1]))
        self.core = SimpleNamespace(joint_upper=True, upper_config=UpperConfig(
            epochs=2, minibatch_size=2, micro_batch_size=1, gae_lambda=.97,
            entropy_coef=0., value_coef=1.), upper_parameters=lambda: iter([self.weight]))
        self.upper_optimizer = torch.optim.AdamW([self.weight], lr=1e-4, weight_decay=1e-4)
        self.seen = []
        self.kl_stop = kl_stop

    def set_training_mode(self, flag):
        self.train(flag)

    def evaluate_actions(self, observations, actions, action_masks):
        self.seen.extend(observations['x'][:, 0].tolist())
        value = observations['x'].float() @ self.weight
        zero = value * 0.
        return value, zero + (1. if self.kl_stop else 0.), zero


@pytest.mark.timeout(30)
@pytest.mark.parametrize('kl_stop', [False, True])
def test_unequal_minibatches_include_empty_rank_and_agree(kl_stop):
    sizes = [5, 2, 0, 3]
    barrier = Barrier(4, timeout=15)
    models = []
    values = [None] * 4
    for rank, count in enumerate(sizes):
        rows = [dict(env=i, episode=0, terminal=True, reward=.3 + .01 * i, value=0.,
            observation={'phase': np.array([V.PHASE_VOCAB_INDEX['map']]),
                         'x': np.array([rank * 10 + i + 1., .3], dtype=np.float32)},
            action=0, mask=np.array([True]), log_prob=0.) for i in range(count)]
        models.append(SimpleNamespace(policy=Policy(kl_stop=kl_stop), device=torch.device('cpu'),
            _upper_completed=rows, _upper_pending=[], autocast_context=nullcontext,
            logger=SimpleNamespace(record=lambda *a: None)))
    def mean(rank, value):
        values[rank] = copy.deepcopy(value)
        barrier.wait()
        result = ({k: sum(v[k] for v in values) / 4 for k in value} if isinstance(value, dict)
                  else sum(values) / 4)
        barrier.wait()
        return result
    def run(rank):
        try:
            def synchronize(step):
                model = models[rank]
                install_flat_gradients(model, mean(rank, pack_gradients(model)))
            return update_upper(models[rank], rank=rank, world_size=4,
                average=lambda value: mean(rank, value), synchronize=synchronize)
        except BaseException:
            barrier.abort()
            raise
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, range(4)))
    for rank, (model, out) in enumerate(zip(models, results)):
        assert out['samples'] == 10 and out['requested_updates'] == 6
        assert out['updates'] == (0 if kl_stop else 6)
        assert out['samples_used'] == (0 if kl_stop else 20)
        assert out['kl_stopped'] == int(kl_stop)
        if not kl_stop:
            assert sorted(model.policy.seen) == sorted([rank * 10 + i + 1. for i in range(sizes[rank])] * 2)
        torch.testing.assert_close(model.policy.state_dict(), models[0].policy.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(model.policy.upper_optimizer.state_dict(),
                                   models[0].policy.upper_optimizer.state_dict(), rtol=0, atol=0)

