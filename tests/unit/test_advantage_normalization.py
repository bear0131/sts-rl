from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.algorithms.ppo import (rollout_advantage_moments, prepare_advantage_normalization,
    model_normalized_advantages, configure_advantage_normalization, normalize_ppo_advantages)


def test_global_moments_weight_samples_not_rank_means():
    ranks = [np.array([-5., 1., 3.]), np.array([7., 13., 21., 40., 51.])]
    all_values = np.concatenate(ranks)
    expected = dict(count=len(all_values), mean=all_values.mean(), std=all_values.std(ddof=1))
    for local in ranks:
        def collective(values):
            if 'total' in values:
                return dict(count=len(all_values) / 2, invalid=0, total=all_values.sum() / 2)
            return dict(centered_sum=np.square(all_values - expected['mean']).sum() / 2)
        assert rollout_advantage_moments(local, average=collective, world_size=2) == expected


def make_model(values):
    return SimpleNamespace(ppo_advantage_normalization='rollout', num_timesteps=8,
        rollout_buffer=SimpleNamespace(advantages=np.array(values), full=True, normalization_generation=1),
        logger=SimpleNamespace(record=lambda *args: None), policy=SimpleNamespace(ppo_advantage_group_size=256))


def test_rollout_normalization_is_invariant_to_partition_and_order_and_preserves_labels():
    model = make_model([1., 2., 4., 8., 16., 32.])
    raw = model.rollout_buffer.advantages.copy()
    prepare_advantage_normalization(model)
    a = torch.tensor(raw, dtype=torch.float32)
    expected = (a - a.mean()) / a.std()
    permutation = torch.tensor([4, 1, 0, 5, 2, 3])
    parts = [model_normalized_advantages(model, part) for part in a[permutation].split(2)]
    torch.testing.assert_close(torch.cat(parts), expected[permutation])
    np.testing.assert_array_equal(raw, model.rollout_buffer.advantages)
    model.rollout_buffer.normalization_generation += 1
    with pytest.raises(RuntimeError, match='this collection'):
        model_normalized_advantages(model, a)


def test_constant_and_invalid_advantages():
    model = make_model([3., 3., 3.])
    prepare_advantage_normalization(model)
    assert torch.equal(model_normalized_advantages(model, torch.full((3,), 3.)), torch.zeros(3))
    for values in ([1.], [], [1., np.inf], [1., np.nan]):
        with pytest.raises(ValueError, match='finite samples'):
            rollout_advantage_moments(values)
    with pytest.raises(ValueError, match='collective'):
        rollout_advantage_moments([1., 2.], world_size=4)
    model.rollout_buffer.full = False
    with pytest.raises(ValueError, match='complete rollout'):
        prepare_advantage_normalization(model)


def test_legacy_groups_unchanged_and_unsupported_modes_rejected():
    torch.manual_seed(20261109)
    a = torch.randn(768)
    model = make_model([1., 2.])
    model.ppo_advantage_normalization = 'group256'
    torch.testing.assert_close(model_normalized_advantages(model, a), normalize_ppo_advantages(a, 256))
    with pytest.raises(ValueError, match='requires normalized'):
        configure_advantage_normalization(model, 'rollout')
    with pytest.raises(ValueError, match='group256 or rollout'):
        configure_advantage_normalization(model, 'invalid')
