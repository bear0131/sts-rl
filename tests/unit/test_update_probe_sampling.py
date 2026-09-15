from types import SimpleNamespace

import numpy as np
import pytest

from tools.analysis.probe_nosl_updates import configure_probe_sampling, audit_indices


def buffer(hps):
    return SimpleNamespace(size=len(hps), observations={
        'player_scalars': np.asarray(hps, dtype=np.float32).reshape(-1, 1)/200},
        sample_indices=object())


def test_uniform_keeps_original_sampler_and_quota_samples_correct_partition():
    replay = buffer([0, 1, 5, 6, 20, 80, 80, 80])
    original = replay.sample_indices
    configure_probe_sampling(replay, 0)
    assert replay.sample_indices is original
    configure_probe_sampling(replay, .25)
    a = replay.sample_indices(8, rng=np.random.default_rng(3))
    b = replay.sample_indices(8, rng=np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert set(a) == set(range(8))
    # Oversampling is required only when the quota exceeds the available pool.
    configure_probe_sampling(replay, .75)
    a = replay.sample_indices(8, rng=np.random.default_rng(3))
    assert np.isin(a, [1, 2]).sum() == 6
    assert len(a) == 8
    assert all(0 <= i < 8 for i in a)
    with pytest.raises(ValueError, match='frozen replay'):
        replay.sample_indices(8, max_old_fraction=.9)


@pytest.mark.parametrize('fraction', [-.1, 1, float('nan')])
def test_invalid_quota_rejected(fraction):
    with pytest.raises(ValueError, match='fraction'):
        configure_probe_sampling(buffer([1, 80]), fraction)


@pytest.mark.parametrize('hps', [[1, 5], [0, 80]])
def test_missing_partition_is_not_silently_substituted(hps):
    with pytest.raises(ValueError, match='both low HP and other'):
        configure_probe_sampling(buffer(hps), .25)


def test_later_audit_filters_source_ids_and_value_only_rows():
    replay = {'observations': {'player_scalars': np.array([[.005], [.4], [.005], [.4]])},
              'arrays': {'policy_weights': np.array([1, 1, 1, 0]),
                         'sample_ids': np.array([10, 11, 12, 13], dtype=np.uint64)}}
    assert audit_indices(replay, minimum_sample_id=12).tolist() == [2]
    assert audit_indices(replay, minimum_sample_id=14).size == 0
