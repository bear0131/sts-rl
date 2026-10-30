"""Distributed compact CE must match a global batch, including empty ranks."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.algorithms.small_playout import update_online
from rl.policies.small_playout import OnlinePlayoutConfig, SmallPlayoutPolicy, search_target
from rl.training.distributed_sb3 import install_flat_gradients, pack_gradients


def model_for(state, samples):
    policy = torch.nn.Module()
    policy.small_playout = SmallPlayoutPolicy(785)
    policy.small_playout.load_state_dict(state)
    policy.small_playout_config = asdict(OnlinePlayoutConfig(batch_size=16, micro_batch_size=2))
    policy.small_playout_optimizer = torch.optim.Adam(policy.small_playout.parameters(), lr=3e-4)
    policy.core = torch.nn.Linear(1, 1)
    policy.core.joint_upper = True
    policy.set_training_mode = policy.train
    return SimpleNamespace(policy=policy, device=torch.device('cpu'), _small_playout_samples=samples,
        _sync_device=lambda: None, max_grad_norm=1., logger=SimpleNamespace(record=lambda *args: None))


@pytest.mark.parametrize('sizes', [(0, 3, 7), (0, 0, 0)])
def test_unequal_and_empty_ranks_match_global_batch(sizes):
    rng = np.random.default_rng(57)
    samples = [dict(state=rng.normal(size=32).astype(np.float32),
        actions=rng.normal(size=(3, 24)).astype(np.float32),
        keys=np.pad(np.array([[5], [10], [15]]), ((0, 0), (0, 9)), constant_values=-1),
        action_bits=[1, 2, 3],
        visits=[1, 3, 6]) for _ in range(sum(sizes))]
    state = SmallPlayoutPolicy(785).state_dict()
    central = model_for(state, samples)
    expected = update_online(central)
    offsets = np.cumsum([0, *sizes])
    models = [model_for(state, samples[offsets[i]:offsets[i+1]]) for i in range(3)]
    # An actual in-memory collective, not precomputed replacement gradients.
    barrier = Barrier(3, timeout=20)
    slots = [None]*3
    def average(rank, values):
        slots[rank] = values
        barrier.wait()
        result = {k: sum(row[k] for row in slots)/3 for k in values}
        barrier.wait()
        return result
    def train(rank):
        model = models[rank]
        def sync(step):
            flat = pack_gradients(model)
            reduced = average(rank, {'gradient': flat})
            install_flat_gradients(model, reduced['gradient'])
        return update_online(model, rank=rank, world_size=3,
            average=lambda values: average(rank, values), synchronize=sync)
    with ThreadPoolExecutor(3) as executor:
        results = list(executor.map(train, range(3)))
    for model, result in zip(models, results, strict=True):
        assert result['samples'] == sum(sizes)
        assert result['updates'] == expected['updates']
        assert result['loss'] == pytest.approx(expected['loss'], abs=2e-6)
        assert not model._small_playout_samples
        for key, value in central.policy.small_playout.state_dict().items():
            torch.testing.assert_close(model.policy.small_playout.state_dict()[key], value, atol=2e-6, rtol=2e-5)
        assert model.policy.core.weight.requires_grad
        assert model.policy.core.weight.grad is None


@pytest.mark.parametrize('cfg', [dict(epochs=0), dict(batch_size=2.5), dict(learning_rate=float('nan')),
    dict(target_mix=-.1), dict(target_mix=1.1), dict(optimizer='unknown'), dict(weight_decay=-1)])
def test_invalid_online_configuration_is_rejected(cfg):
    with pytest.raises(ValueError, match='invalid online'):
        OnlinePlayoutConfig(**cfg)


def test_recommendation_is_not_the_most_visited_action():
    # Q-band/root guards can recommend a move different from argmax(visits).
    sample = dict(action_bits=[17, 42, 99], visits=[8, 1, 1], recommended_action=42)
    np.testing.assert_allclose(search_target(sample, .2), [.16, .82, .02], atol=1e-7)
    np.testing.assert_allclose(search_target(sample, 1), [.8, .1, .1], atol=1e-7)
    with pytest.raises(ValueError, match='exactly one legal'):
        search_target(dict(sample, recommended_action=3), .2)


def test_aligned_ce_learns_recommended_move_and_exposes_target_entropy():
    torch.manual_seed(57)
    sample = dict(state=np.ones(32, np.float32), actions=np.zeros((2, 24), np.float32),
        keys=np.pad(np.array([[5], [10]]), ((0, 0), (0, 9)), constant_values=-1),
        action_bits=[17, 42], visits=[9, 1], recommended_action=42)
    model = model_for(SmallPlayoutPolicy(785).state_dict(), [sample] * 8)
    model.policy.small_playout_config = asdict(OnlinePlayoutConfig(epochs=8, batch_size=8,
        micro_batch_size=2, learning_rate=.003, target_mix=.2, optimizer='adamw',
        weight_decay=.0001, max_grad_norm=0))
    model.policy.small_playout_optimizer = torch.optim.AdamW(model.policy.small_playout.parameters(),
        lr=.003, weight_decay=.0001)
    from rl.policies.small_playout import collate_features
    inputs = collate_features([sample])
    before = model.policy.small_playout(*inputs).softmax(-1)[0, 1].item()
    report = update_online(model)
    after = model.policy.small_playout(*inputs).softmax(-1)[0, 1].item()
    assert after > before + .1
    assert report['updates'] == 8
    assert report['target_entropy'] == pytest.approx(-(.18*np.log(.18)+.82*np.log(.82)), abs=1e-6)
    assert report['target_kl'] == pytest.approx(report['loss']-report['target_entropy'])
    assert report['uniform_ce'] == pytest.approx(np.log(2))
