from types import SimpleNamespace

import numpy as np
import pytest

from rl.search.rollout_inference import ExactInputCache


def test_exact_cache_reuses_across_batches_and_owns_outputs():
    cache = ExactInputCache(4096)
    calls = []
    def forward(obs):
        calls.append(obs['x'].copy())
        return obs['x']*2., obs['x'].sum(axis=1)
    obs = dict(x=np.array([[1, 2], [1, 2], [2, 1]], np.float32))
    logits, values = cache.forward(obs, forward)
    np.testing.assert_array_equal(logits, obs['x']*2.)
    np.testing.assert_array_equal(values, [3, 3, 3])
    assert len(calls[0]) == 2
    logits[:] = -100
    values[:] = -100
    again = cache.forward(dict(x=obs['x'][::-1].copy()), forward)
    np.testing.assert_array_equal(again[0], obs['x'][::-1]*2.)
    assert len(calls) == 1
    assert cache.stats['hits'] == 3 and cache.stats['batch_duplicates'] == 1
    cache.forward(obs, forward, namespace=('other-physical-width',))
    assert len(calls) == 2


def test_exact_cache_distinguishes_dtype_shape_field_and_value():
    cache = ExactInputCache(1_000_000)
    calls = []
    def forward(obs):
        calls.append(obs)
        return np.zeros((1, 2), np.float32), np.ones(1, np.float32)
    for obs in (dict(x=np.array([[1, 2]], np.int32)),
                dict(x=np.array([[1, 2]], np.float32)),
                dict(x=np.array([[[1, 2]]], np.int32)),
                dict(y=np.array([[1, 2]], np.int32)),
                dict(x=np.array([[2, 1]], np.int32))):
        cache.forward(obs, forward)
    assert len(calls) == 5


def test_exact_cache_evicts_and_clear_invalidates_new_weights():
    cache = ExactInputCache(600)
    weight = 1
    def forward(obs):
        return obs['x']*weight, obs['x'].sum(axis=1)*weight
    obs = dict(x=np.array([[1], [2], [1]], np.float32))
    np.testing.assert_array_equal(cache.forward(obs, forward)[0], obs['x'])
    assert cache.stats['bytes'] <= 600 and cache.stats['evictions'] == 1
    cache.clear()
    assert not cache.entries and cache.stats['bytes'] == 0
    weight = 2
    np.testing.assert_array_equal(cache.forward(obs, forward)[0], obs['x']*2)


def test_failed_forward_does_not_publish_results():
    cache = ExactInputCache(4096)
    def fail(_):
        raise RuntimeError('model failed')
    with pytest.raises(RuntimeError, match='model failed'):
        cache.forward(dict(x=np.ones((2, 1))), fail)
    assert not cache.entries


def test_zero_capacity_bypasses_packing_and_batch_deduplication():
    cache = ExactInputCache(0)
    obs = dict(x=np.ones((3, 2), np.float32))
    calls = []
    def forward(batch):
        assert batch is obs
        calls.append(len(batch['x']))
        return batch['x']*2, batch['x'].sum(axis=1)
    for _ in range(2):
        np.testing.assert_array_equal(cache.forward(obs, forward)[0], obs['x']*2)
    assert calls == [3, 3]
    assert not cache.entries and not cache.scopes
    assert cache.stats['requests'] == cache.stats['misses'] == 6
    assert cache.stats['key_seconds'] == cache.stats['batch_duplicates'] == 0


def test_schema_eviction_cannot_alias_live_cached_values():
    cache = ExactInputCache(1_000_000)
    obs = dict(x=np.ones((1, 1), np.float32))
    calls = []
    for namespace in [*range(65), 0, 64]:
        def forward(batch):
            calls.append(namespace)
            return batch['x'] * namespace, np.array([namespace], np.float32)
        logits, values = cache.forward(obs, forward, namespace=(namespace,))
        np.testing.assert_array_equal(logits, [[namespace]])
        np.testing.assert_array_equal(values, [namespace])
    assert calls == [*range(65), 0]
    cache.clear()
    logits, _ = cache.forward(obs, lambda batch: (batch['x'] * 99, np.array([99.])))
    np.testing.assert_array_equal(logits, [[99]])
