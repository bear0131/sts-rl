"""Native orchestration must keep Python transport's batches and exact LRU."""
import numpy as np
import pytest
import torch
import slaythespire as sts

from env.core import encode
from env.core.native_observation import upper_value_runtime
from rl.policies.frozen_upper import FrozenUpperTeacher
from rl.search.rollout_inference import ExactInputCache, _upper_graph_inputs

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


@pytest.mark.parametrize('module', [None, object()])
def test_full480_array_cache_retains_optional_extension_compatibility(monkeypatch, module):
    import sys
    from rl.search.rollout_inference import _DeckConsumers
    monkeypatch.setitem(sys.modules, 'slaythespire', module)
    consumer = _DeckConsumers(torch.nn.Linear(1, 1).eval(), 'cpu', 1024)
    assert not consumer.native_anchor_cache
    assert isinstance(consumer.anchor_cache, ExactInputCache)
    consumer.close()


@pytest.mark.parametrize('capacity', [0, 4096, 1024 * 1024])
@pytest.mark.parametrize('threads', [1, 8])
def test_full480_array_values_match_exact_cache(capacity, threads):
    assert getattr(sts, 'array_value_cache_version', 0) == 1, 'rebuild the selected native extension'
    native = sts.UpperValueRuntime(capacity, threads)
    reference = ExactInputCache(capacity)
    traces = [[], []]

    def predict(batch, side):
        traces[side].append(tuple((k, v.shape, v.dtype.str, v.tobytes()) for k, v in batch.items()))
        return (batch['ids'][:, 0] * np.float32(3) + batch['features'][:, 0]).astype(np.float32)

    for width in (1, 7, 65, 1):
        ids = (np.arange(640, dtype=np.int64) % 32)[:, None]
        features = np.repeat(ids.astype(np.float32), width * 2, axis=1)[:, ::2]
        original = dict(ids=ids, features=features, empty=np.empty((640, 0), np.int32))
        for batch in (original, {k: v[::-1] for k, v in original.items()},
                      dict(reversed(list(original.items())))):
            def old_forward(missing):
                return np.empty((len(missing['ids']), 0), np.float32), predict(missing, 0)
            expected = reference.forward(batch, old_forward)[1]
            actual = native.forward_arrays(batch, lambda b: predict(b, 1))
            assert actual.tobytes() == expected.tobytes()
            assert traces[0] == traces[1]
            for key in ('requests', 'misses', 'hits', 'batch_duplicates', 'evictions', 'bytes', 'peak_bytes'):
                assert native.stats[key] == reference.stats[key], key
            actual.fill(-999)
    native.clear()
    reference.clear()
    assert native.stats['bytes'] == 0
    np.testing.assert_array_equal(native.forward_arrays(original, lambda b: predict(b, 1)),
                                  predict(original, 1))


def test_full480_array_values_schema_scope_and_errors():
    assert getattr(sts, 'array_value_cache_version', 0) == 1
    cache = sts.UpperValueRuntime(1024 * 1024, 8)
    calls = []
    def predict(batch):
        calls.append(tuple((k, v.shape, v.dtype.str) for k, v in batch.items()))
        return np.full(len(next(iter(batch.values()))), len(calls), np.float32)
    for width in range(1, 67):
        row = dict(x=np.zeros((1, width), np.float32))
        assert cache.forward_arrays(row, predict).tolist() == [width]
        assert cache.forward_arrays(row, predict).tolist() == [width]
    # The evicted schema cannot reuse an old scope whose entries remain live.
    assert cache.forward_arrays(dict(x=np.zeros((1, 1), np.float32)), predict).tolist() == [67]
    assert cache.forward_arrays(dict(x=np.zeros((1, 1), np.int32)), predict).tolist() == [68]
    assert cache.forward_arrays(dict(y=np.zeros((1, 1), np.int32)), predict).tolist() == [69]
    # Signed zero differs by bytes even though numeric equality says otherwise.
    assert cache.forward_arrays(dict(x=np.full((1, 1), -0., np.float32)), predict).tolist() == [70]
    for bad in ({}, dict(x=np.zeros((0, 1))), dict(x=np.zeros((2, 1)), y=np.zeros((1, 1))),
                dict(x=np.array([object()]))):
        with pytest.raises(ValueError):
            cache.forward_arrays(bad, predict)
    cache.clear()
    def fail(batch):
        with pytest.raises(RuntimeError, match='pending upper inference'):
            cache.clear()
        with pytest.raises(RuntimeError, match='pending upper inference'):
            cache.forward_arrays(batch, predict)
        raise RuntimeError('callback failure')
    batch = dict(x=np.ones((640, 2), np.float32))
    with pytest.raises(RuntimeError, match='callback failure'):
        cache.forward_arrays(batch, fail)
    with pytest.raises(ValueError, match='one float32 value'):
        cache.forward_arrays(batch, lambda b: np.zeros((len(b['x']), 2), np.float32))
    assert cache.stats['bytes'] == 0
    np.testing.assert_array_equal(cache.forward_arrays(batch, lambda b: b['x'][:, 0]), np.ones(640))


@pytest.mark.parametrize('capacity', [0, 4096, 1024 * 1024])
@pytest.mark.parametrize('pad', [False, True])
def test_native_groups_cache_eviction_and_owned_results(frozen_upper_source, capacity, pad):
    assert hasattr(sts, 'UpperValueRuntime'), 'rebuild the explicitly selected native extension'
    core, space, env, _ = frozen_upper_source
    teacher = FrozenUpperTeacher(core, space, source_sha256='1' * 64)
    # Serialized historical checkpoints can append fields after construction.
    # Matching dict contents is insufficient: graph input layout needs order.
    teacher.observation_space.spaces = dict(reversed(list(teacher.observation_space.spaces.items())))
    native = upper_value_runtime(teacher, capacity, pad=pad)
    reference = ExactInputCache(capacity)
    capacities = tuple(space[key].shape[0] for key in ('deck_ids', 'relics'))
    traces = [[], []]

    def predict(batch, signature, side):
        traces[side].append((signature, tuple(batch), {key: (value.shape, value.dtype.str, value.tobytes())
                                       for key, value in batch.items()}))
        return (batch['run_hp'][:, 0] * np.float32(7) +
                batch['deck_ids'].sum(axis=1).astype(np.float32) / np.float32(10))

    def old_forward(rows):
        result = np.empty(len(rows), np.float32)
        for indices, phase, widths, batch in teacher.prepare_search_numpy(rows):
            if pad:
                batch, widths = _upper_graph_inputs(batch, widths, capacities)
            signature = (phase, *widths)
            def compute(missing):
                return np.empty((len(next(iter(missing.values()))), 0), np.float32), predict(missing, signature, 0)
            result[indices] = reference.forward(batch, compute, namespace=signature)[1]
        return result

    original = env._obs()
    batches = []
    # Unpadded widths also exercise the 64-scope FIFO and never-reused IDs.
    for length in ((1, 17, 33, 1) if pad else (*range(1, 67), 1)):
        rows = []
        for i in range(8):
            row = {key: value.copy() for key, value in original.items()}
            row['phase'][0] = encode.V.PHASE_VOCAB_INDEX['map' if i % 3 == 0 else 'combat_reward']
            row['run_hp'][0] += (i % 4) / 100
            row['deck_ids'][:] = 0
            row['deck_ids'][:length] = original['deck_ids'][0]
            row['relics'][:] = 0
            row['relics'][:max(1, length // 4)] = original['relics'][0]
            # Views and unused fields must not change the byte identity.
            row['run_hp'] = np.repeat(row['run_hp'], 2)[::2]
            rows.append(row)
        batches.extend((rows, rows[::-1], rows))
    for rows in batches:
        expected = old_forward(rows)
        actual = native.forward(rows, teacher, lambda batch, signature: predict(batch, signature, 1))
        assert actual.tobytes() == expected.tobytes()
        assert traces[0] == traces[1]
        for key in ('requests', 'misses', 'hits', 'batch_duplicates', 'evictions', 'bytes', 'peak_bytes'):
            assert native.stats[key] == reference.stats[key], key
        actual.fill(-999)  # A caller must not mutate an entry's cached scalar.
    native.clear()
    reference.clear()
    assert native.stats['bytes'] == 0
    assert native.forward(rows, teacher, lambda b, s: predict(b, s, 1)).tobytes() == old_forward(rows).tobytes()


def test_native_validation_sees_registry_changes_and_recovers_after_error(frozen_upper_source):
    core, space, env, _ = frozen_upper_source
    teacher = FrozenUpperTeacher(core, space, source_sha256='1' * 64)
    runtime = upper_value_runtime(teacher, 1024 * 1024, pad=True)
    rows = [env._obs()]
    def compute(batch, signature):
        return np.arange(len(batch['phase']), dtype=np.float32)
    runtime.forward(rows, teacher, compute)
    teacher.overworld.register_parameter('late_parameter', torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(RuntimeError, match='made trainable'):
        runtime.forward(rows, teacher, compute)  # Even on a complete cache hit.
    del teacher.overworld.late_parameter
    teacher.overworld.add_module('late_module', torch.nn.Identity())
    with pytest.raises(RuntimeError, match='made trainable'):
        runtime.forward(rows, teacher, compute)
    del teacher.overworld.late_module
    with pytest.raises(ValueError, match='shape changed: run_hp'):
        runtime.forward([dict(rows[0], run_hp=np.zeros(2, np.float32))], teacher, compute)
    with pytest.raises(ValueError, match='dtype changed: run_hp'):
        runtime.forward([dict(rows[0], run_hp=rows[0]['run_hp'].astype(np.float64))], teacher, compute)
    missing = dict(rows[0]);del missing['run_hp']
    with pytest.raises(KeyError):
        runtime.forward([missing], teacher, compute)
    runtime.clear()
    def fail(batch, signature):
        with pytest.raises(RuntimeError, match='pending upper inference'):
            runtime.clear()
        raise RuntimeError('callback failure')
    with pytest.raises(RuntimeError, match='callback failure'):
        runtime.forward(rows, teacher, fail)
    assert runtime.forward(rows, teacher, compute).tolist() == [0.]
    teacher.assert_integrity()


@pytest.mark.parametrize('capacity', [0, 4096, 1024 * 1024])
def test_packed_reward_and_dict_rows_share_exact_cache(frozen_upper_source, run_combat_factory, capacity):
    from env.core.native_observation import upper_reward_encoder
    core, space, _, _ = frozen_upper_source
    teacher = FrozenUpperTeacher(core, space, source_sha256='1' * 64)
    packed_runtime = upper_value_runtime(teacher, capacity, pad=True)
    dict_runtime = upper_value_runtime(teacher, capacity, pad=True)
    root = run_combat_factory()
    leaf = root.bc.clone()
    leaf.set_player_strength(999)
    leaf.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    branch, _ = root.settle_combat_leaf(leaf)
    codec = upper_reward_encoder(False)
    calls = [[], []]
    def predict(batch, signature, side):
        calls[side].append((signature, tuple(batch),
                           {k: (v.shape, v.dtype.str, v.tobytes()) for k, v in batch.items()}))
        return batch['run_hp'][:, 0] + batch['gold'][:, 0]
    for size in (4, 17, 4):
        branch.gc.set_deck([('STRIKE_RED', 0)] * size)
        packed = codec.encode_packed(branch.gc)
        ordinary = codec.encode(branch.gc)
        rows = [packed, ordinary, packed]
        for repeat in range(3):
            # Mutation through a retained view must invalidate exact identity.
            packed['gold'][0] += np.float32(.1)
            reference = [{key: row[key].copy() for key in row} for row in rows]
            actual = packed_runtime.forward(rows, teacher, lambda b, s: predict(b, s, 0))
            expected = dict_runtime.forward(reference, teacher, lambda b, s: predict(b, s, 1))
            assert actual.tobytes() == expected.tobytes()
            assert calls[0] == calls[1]
            for key in ('requests', 'misses', 'hits', 'batch_duplicates', 'evictions', 'bytes', 'peak_bytes'):
                assert packed_runtime.stats[key] == dict_runtime.stats[key], key
