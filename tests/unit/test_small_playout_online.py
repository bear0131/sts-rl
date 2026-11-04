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


def model_for(state, samples, device='cpu'):
    policy = torch.nn.Module()
    policy.small_playout = SmallPlayoutPolicy(785).to(device)
    policy.small_playout.load_state_dict(state)
    policy.small_playout_config = asdict(OnlinePlayoutConfig(batch_size=16, micro_batch_size=2))
    policy.small_playout_optimizer = torch.optim.Adam(policy.small_playout.parameters(), lr=3e-4)
    policy.core = torch.nn.Linear(1, 1).to(device)
    policy.core.joint_upper = True
    policy.set_training_mode = policy.train
    return SimpleNamespace(policy=policy, device=torch.device(device), _small_playout_samples=samples,
        _sync_device=lambda: torch.cuda.synchronize() if device == 'cuda' else None,
        max_grad_norm=1., logger=SimpleNamespace(record=lambda *args: None))


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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('batch_size,expected_updates', [(16, 8), (4, 24)])
def test_batch_change_keeps_eight_passes_and_nonempty_adam(
        monkeypatch, device, batch_size, expected_updates):
    """Empty ranks/tails still join every step; batch changes never drop samples."""
    import copy
    from collections import Counter
    from threading import local
    from rl.policies.small_playout import PreparedPlayoutSamples
    if device == 'cuda':
        assert torch.cuda.is_available(), 'explicit lower-batch CUDA qualification requires CUDA'
    torch.manual_seed(57)
    np.random.seed(57)
    rng = np.random.default_rng(57)
    sizes = [0, 3, 7, 9]
    samples = [dict(state=rng.normal(size=32).astype(np.float32),
        actions=rng.normal(size=(3, 24)).astype(np.float32),
        keys=np.pad(np.array([[5], [10], [15]]), ((0, 0), (0, 9)), constant_values=-1),
        action_bits=[1, 2, 3], visits=[1, 3, 6], recommended_action=2)
        for _ in range(sum(sizes))]
    source = model_for(SmallPlayoutPolicy(785).state_dict(), samples, device)
    cfg = asdict(OnlinePlayoutConfig(epochs=8, batch_size=32, micro_batch_size=2,
        learning_rate=.003, target_mix=.2, optimizer='adamw', weight_decay=.0001, max_grad_norm=0))
    source.policy.small_playout_config = cfg
    source.policy.small_playout_optimizer = torch.optim.AdamW(
        source.policy.small_playout.parameters(), lr=.003, weight_decay=.0001)
    assert update_online(source)['updates'] == 8
    source_state = copy.deepcopy(source.policy.small_playout.state_dict())
    source_adam = copy.deepcopy(source.policy.small_playout_optimizer.state_dict())
    assert {int(v['step']) for v in source_adam['state'].values()} == {8}
    offsets = np.cumsum([0, *sizes])
    models = []
    for rank in range(4):
        model = model_for(source_state, samples[offsets[rank]:offsets[rank+1]], device)
        model.policy.small_playout_config = dict(cfg, batch_size=batch_size)
        optimizer = torch.optim.AdamW(model.policy.small_playout.parameters(), lr=.003, weight_decay=.0001)
        optimizer.load_state_dict(copy.deepcopy(source_adam))
        model.policy.small_playout_optimizer = optimizer
        model.small_playout_optimizer_steps = 8
        models.append(model)
    upper_before = [copy.deepcopy(m.policy.core.state_dict()) for m in models]
    seen = [Counter() for _ in models]
    owner = local()
    original_batch = PreparedPlayoutSamples.batch
    def observed_batch(self, indices, target_device):
        seen[owner.rank].update(map(int, indices))
        return original_batch(self, indices, target_device)
    monkeypatch.setattr(PreparedPlayoutSamples, 'batch', observed_batch)
    barrier = Barrier(4, timeout=30)
    slots = [None]*4
    def average(rank, values):
        slots[rank] = values
        barrier.wait()
        result = {k: sum(row[k] for row in slots)/4 for k in values}
        barrier.wait()
        return result
    def train(rank):
        owner.rank = rank
        model = models[rank]
        def sync(step):
            reduced = average(rank, {'gradient': pack_gradients(model)})
            install_flat_gradients(model, reduced['gradient'])
        return update_online(model, rank=rank, world_size=4,
            average=lambda values: average(rank, values), synchronize=sync)
    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(train, range(4)))
    for rank, (model, report) in enumerate(zip(models, results, strict=True)):
        assert seen[rank] == Counter({i: 8 for i in range(sizes[rank])})
        assert report['samples'] == 19 and report['updates'] == expected_updates
        assert report['optimizer_steps_total'] == 8 + expected_updates
        assert {int(v['step']) for v in model.policy.small_playout_optimizer.state.values()} == {8 + expected_updates}
        assert not model._small_playout_samples
        assert np.isfinite(report['loss']) and report['parameter_l2_change'] > 1e-6
        assert model.policy.small_playout_optimizer.param_groups[0]['lr'] == .003
        torch.testing.assert_close(model.policy.core.state_dict(), upper_before[rank], rtol=0, atol=0)
        assert all(p.requires_grad and p.grad is None for p in model.policy.core.parameters())
        torch.testing.assert_close(model.policy.small_playout.state_dict(),
                                   models[0].policy.small_playout.state_dict(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(source.policy.small_playout.state_dict(), source_state, rtol=0, atol=0)
    torch.testing.assert_close(source.policy.small_playout_optimizer.state_dict(), source_adam, rtol=0, atol=0)


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
