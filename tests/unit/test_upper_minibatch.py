"""Unequal/empty ranks consume real rows once and agree on every Adam step."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from env.core import vocab as V
from rl.algorithms.upper_ppo import update_upper
from rl.policies.entity_joint import UpperConfig
from rl.training.distributed_sb3 import pack_gradients, install_flat_gradients


class Policy(torch.nn.Module):
    def __init__(self, *, kl_stop=False, target_kl=.02):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.2, -.1]))
        self.core = SimpleNamespace(joint_upper=True, upper_config=UpperConfig(
            epochs=2, minibatch_size=2, micro_batch_size=1, gae_lambda=.97,
            entropy_coef=0., value_coef=1., target_kl=target_kl), upper_parameters=lambda: iter([self.weight]))
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
@pytest.mark.parametrize('target_kl', [.02, None])
def test_unequal_minibatches_include_empty_rank_and_agree(kl_stop, target_kl):
    sizes = [5, 2, 0, 3]
    barrier = Barrier(4, timeout=15)
    models = []
    values = [None] * 4
    for rank, count in enumerate(sizes):
        rows = [dict(env=i, episode=0, terminal=True, reward=.3 + .01 * i, value=0.,
            observation={'phase': np.array([V.PHASE_VOCAB_INDEX['map']]),
                         'x': np.array([rank * 10 + i + 1., .3], dtype=np.float32)},
            action=0, mask=np.array([True]), log_prob=0.) for i in range(count)]
        models.append(SimpleNamespace(policy=Policy(kl_stop=kl_stop, target_kl=target_kl), device=torch.device('cpu'),
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
        stopped = kl_stop and target_kl is not None
        assert out['samples'] == 10 and out['requested_updates'] == 6
        assert out['updates'] == (0 if stopped else 6)
        assert out['samples_used'] == (0 if stopped else 20)
        assert out['kl_stopped'] == int(stopped)
        assert out['kl_early_stop_enabled'] == int(target_kl is not None)
        if kl_stop:
            assert out['kl'] > .03  # Still reported even when the guard is off.
        if not stopped:
            assert sorted(model.policy.seen) == sorted([rank * 10 + i + 1. for i in range(sizes[rank])] * 2)
            assert out['epochs_completed'] == 2
            assert 'kl_epoch_2' in out
        torch.testing.assert_close(model.policy.state_dict(), models[0].policy.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(model.policy.upper_optimizer.state_dict(),
                                   models[0].policy.upper_optimizer.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize('minibatch_size', [0, 2])
@pytest.mark.parametrize('nonfinite', [False, True])
def test_disabled_kl_guard_keeps_statistics_and_rejects_nonfinite(minibatch_size, nonfinite):
    policy = Policy(kl_stop=True, target_kl=None)
    policy.core.upper_config = replace(policy.core.upper_config, minibatch_size=minibatch_size)
    rows = [dict(env=i, episode=0, terminal=True, reward=.3, value=0.,
        observation={'phase': np.array([V.PHASE_VOCAB_INDEX['map']]),
                     'x': np.array([1., .3], dtype=np.float32)},
        action=0, mask=np.array([True]), log_prob=float('nan') if nonfinite else 0.)
        for i in range(4)]
    model = SimpleNamespace(policy=policy, device=torch.device('cpu'),
        _upper_completed=rows, _upper_pending=[], autocast_context=nullcontext,
        logger=SimpleNamespace(record=lambda *a: None))
    if nonfinite:
        with pytest.raises(FloatingPointError, match='nonfinite strategic'):
            update_upper(model)
        assert not policy.upper_optimizer.state
    else:
        out = update_upper(model)
        assert out['updates'] == (4 if minibatch_size else 2)
        assert out['kl_stopped'] == out['kl_early_stop_enabled'] == 0
        assert out['kl_epoch_2'] > .03


@pytest.mark.parametrize('target', [0., -.01, float('nan'), float('inf')])
def test_kl_target_rejects_invalid_values(target):
    with pytest.raises(ValueError, match='invalid strategic PPO'):
        UpperConfig(target_kl=target)


def test_kl_config_default_and_explicit_disabled():
    assert UpperConfig().target_kl == .02
    assert UpperConfig(target_kl=None).target_kl is None

