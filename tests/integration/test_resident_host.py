"""Native owner bookkeeping against the accepted Python contracts."""
import numpy as np
import pytest

pytestmark = [pytest.mark.native, pytest.mark.cuda, pytest.mark.timeout(120)]


def test_actor_packet_keeps_columnar_rows_without_rebuilding_observations(cuda_battle_artifacts):
    from env.search.resident_transport import pack_actor_roots, unpack_actor_roots
    rows = {key: dict(slot=slot, budget=64, observation={
        'phase':np.array([slot],np.int64), 'values':np.array([.1,slot],np.float32),
        'mask':np.array([True,False],bool)}) for key,slot in [('later',3),('earlier',1)]}
    packet = pack_actor_roots(rows,cuda_battle_artifacts[2])
    metadata,columns = unpack_actor_roots(packet,row_observations=False)
    ordinary,reference = unpack_actor_roots(packet)
    assert list(metadata) == list(rows) == list(ordinary)
    assert all('observation' not in row for row in metadata.values())
    for field, column in columns.items():
        assert column.tobytes() == np.stack([r['observation'][field] for r in rows.values()]).tobytes()
        assert column.tobytes() == reference[field].tobytes()
        assert any(np.shares_memory(column,storage) for _,_,storage in packet['columns'])
    _, packed = unpack_actor_roots(packet,row_observations=False,packed=True)
    assert packed.concatenate([]) is packed
    reversed_fields = dict(reversed(list(reference.items())))
    combined = packed.concatenate([packed,reversed_fields])
    for field in reference:
        np.testing.assert_array_equal(combined[field],np.concatenate([reference[field]]*3))
        assert not np.shares_memory(combined[field],packed[field])
    with pytest.raises(ValueError,match='schema changed'):
        packed.concatenate([{'phase':reference['phase']}])


def test_native_root_and_page_ownership_matches_reference(cuda_battle_artifacts):
    from env.search.root_ledger import RootLedger
    from env.search.state_pages import DevicePages
    host = cuda_battle_artifacts[2]
    ledgers = [RootLedger(4), host.RootLedger(4)]
    stale = []
    for ledger in ledgers:
        a, b = ledger.claim(['a', ('b', 1)])
        stale.append(a)
        with pytest.raises(ValueError): ledger.claim(['c', 'c'])
        with pytest.raises(ValueError): ledger.claim(['a'])
        with pytest.raises(OverflowError): ledger.claim(['c', 'd', 'e'])
        ledger.release(a)
        c, = ledger.claim(['c'])
        assert (a.slot, b.slot, c.slot, c.generation) == (0, 1, 0, 2)
        with pytest.raises(ValueError): ledger.release(a)
    native = ledgers[1]
    native.set_deadlines(list(native.pending.values()), np.array([31., 17.]))
    assert native.deadline() == 17.
    assert [h.key for h in native.completed(np.array([0, 1, 0, 0], np.int32))] == ['c']
    native.release(native.pending['c'])
    assert native.deadline() == 31.

    class Block:
        def __init__(self, address): self.address = address
        def data_ptr(self): return self.address
    def allocator():
        blocks = []
        def allocate(n):
            block = Block(0x100000 + len(blocks)*0x10000)
            blocks.append(block)
            return block
        return allocate
    pools = [cls(4, 32, 64, allocator(), growth_pages=2) for cls in (DevicePages, host.DevicePages)]
    owned = [p.acquire([('a', 17), ('b', 3)]) for p in pools]
    for key in ('a', 'b'):
        np.testing.assert_array_equal(owned[0][1][key], owned[1][1][key])
    assert pools[0].statistics() == pools[1].statistics()
    for p, (leases, pointers) in zip(pools, owned, strict=True):
        before = p.statistics()
        with pytest.raises(ValueError): p.release([leases['a'], leases['a']])
        with pytest.raises(ValueError): p.acquire([('a', 1)])
        assert p.statistics() == before
        with pytest.raises(RuntimeError): p.close()
        p.release([leases['a']])
        replacement, new = p.acquire([('c', 16)])
        np.testing.assert_array_equal(new['c'], pointers['a'][:16])
        with pytest.raises(ValueError): p.release([leases['a']])
        p.release([leases['b'], replacement['c']]); p.close(); p.close()
    assert pools[0].statistics() == pools[1].statistics()
    for cls in (DevicePages, host.DevicePages):
        def fail(n): raise MemoryError('allocation denied')
        p = cls(2, 16, 64, fail)
        before = p.statistics()
        with pytest.raises(MemoryError): p.acquire([('a', 1)])
        assert p.statistics() == before
        p.close()


@pytest.mark.parametrize('limit', [0, 1200, 100000])
def test_native_anchor_cache_preserves_full_bytes_order_and_eviction(cuda_battle_artifacts, limit):
    from rl.search.rollout_inference import ExactInputCache
    caches = [ExactInputCache(limit), cuda_battle_artifacts[2].ExactValueCache(limit)]
    rng = np.random.default_rng(20261111)
    values = rng.normal(size=(13, 6)).astype(np.float32)
    # Different NaN payloads and signed zero are distinct exact keys.
    values[:4, 0] = np.array([0x7fc00001, 0x7fc00002, 0, 0x80000000], np.uint32).view(np.float32)
    for namespace in ((), ('weights', 2), (), *range(66), ()):
        for take in ([0, 0, 1, 2, 3, 1], [3, 0, 4, 12, 4], [12, 3, 2]):
            observations = dict(ids=np.asarray(take, np.int64)[:, None], x=values[take, ::2])
            seen, results = [], []
            for cache in caches:
                called = []
                def forward(x):
                    called.append({k: v.copy() for k, v in x.items()})
                    return np.empty((len(x['ids']), 0), np.float32), x['ids'][:, 0].astype(np.float32)*.25
                results.append(cache.forward(observations, forward, namespace=namespace))
                seen.append(called)
            assert len(seen[0]) == len(seen[1])
            if seen[0]:
                for key in observations:
                    assert seen[0][0][key].tobytes() == seen[1][0][key].tobytes()
            np.testing.assert_array_equal(results[0][1], results[1][1])
            for key in ('requests', 'hits', 'misses', 'batch_duplicates', 'evictions', 'bytes', 'peak_bytes'):
                assert caches[0].stats[key] == caches[1].stats[key], key
            results[1][1][:] = 999  # Returned arrays must not alias cached values.
    for cache in caches:
        cache.clear()
        assert cache.stats['bytes'] == 0


def test_native_intern_and_packing_preserve_types_shapes_and_ownership(cuda_battle_artifacts):
    host = cuda_battle_artifacts[2]
    intern = host.ByteIntern(4096)
    data = np.arange(24, dtype=np.int64).reshape(4, 6)
    first = intern.intern([data])
    repeated = intern.intern([data[[2, 0, 2]]])
    np.testing.assert_array_equal(repeated, first[[2, 0, 2]])
    reshaped = intern.intern([data.reshape(4, 2, 3)])
    assert not set(first) & set(reshaped)
    intern.set_limit(0)
    assert intern.bytes == 0
    after = intern.intern([data])
    assert min(after) > max(reshaped)
    intern.clear(); intern.set_limit(4096)
    assert min(intern.intern([data])) > max(after)
    arrays = [data[:, ::2], np.array([.1, -0.], np.float32), np.array([True, False])]
    packed, offsets = host.pack_arrays(arrays)
    for a, offset in zip(arrays, offsets, strict=True):
        assert offset % 8 == 0
        assert packed[offset:offset+a.nbytes].tobytes() == a.tobytes()
    original = packed.copy(); data[:] = 77
    np.testing.assert_array_equal(packed, original)
    obs = [dict(a=np.arange(6, dtype=np.float32).reshape(2, 3), b=np.array([x], np.float32)) for x in (7, 9)]
    np.testing.assert_array_equal(host.stack_fields(obs, ['a', 'b']), [[0, 1, 2, 3, 4, 5, 7], [0, 1, 2, 3, 4, 5, 9]])
    with pytest.raises(ValueError, match='layout'):
        host.stack_fields([obs[0], dict(obs[1], a=obs[1]['a'].reshape(3, 2))], ['a', 'b'])
    snapshots = [dict(portable=True, words=np.arange(n, dtype=np.uint64)) for n in (3, 9)]
    words, lengths = host.pack_snapshots(snapshots, np.array([4, 1], np.int64), 6)
    np.testing.assert_array_equal(lengths, [3, 9])
    np.testing.assert_array_equal(words[0], [0, 1, 2, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError): host.pack_snapshots(snapshots, np.array([1, 1]), 6)


def test_native_job_aggregation_keeps_original_paired_sum_order(cuda_battle_artifacts):
    host = cuda_battle_artifacts[2]
    roots = [dict(actions=np.array([3, 9, 15]), bits=np.array([1, 7, 9], np.uint32), draw_selection=1),
             dict(actions=np.array([8, 12]), bits=np.array([8, 2], np.uint32), draw_selection=0)]
    seeds = [np.array([31, 47]), np.array([19, 23, 29])]
    ledger = host.RootLedger(4); handles = ledger.claim(['one', 'two'])
    s, d, w, b, draw, actions, counts = host.pack_jobs(roots, seeds, np.array([h.slot for h in handles]), 16, 4)
    np.testing.assert_array_equal(s, [0]*6+[1]*6)
    np.testing.assert_array_equal(d, list(range(4, 10))+list(range(20, 26)))
    np.testing.assert_array_equal(w, [31]*3+[47]*3+[19]*2+[23]*2+[29]*2)
    np.testing.assert_array_equal(b, [1, 7, 9]*2+[8, 2]*3)
    np.testing.assert_array_equal(draw, [1]*6+[0]*6)
    jobs = {h.slot: (d[i*6:i*6+6]-4, actions[i*6:i*6+6]) for i, h in enumerate(handles)}
    np.testing.assert_array_equal(host.completed_jobs(handles, jobs), d-4)
    bank = {h.slot: dict(root=roots[i], worlds=seeds[i]) for i, h in enumerate(handles)}
    values = np.array([1e17, 2., 3., -1e17, .01, 7., 1e17, -1e17, 1., 2., -1e17, 1e17])
    results = host.complete_results(handles, bank, jobs, values)
    for i, h in enumerate(handles):
        expected = np.zeros(len(roots[i]['actions']), np.float64)
        np.add.at(expected, actions[i*6:i*6+6], values[i*6:i*6+6])
        expected /= len(seeds[i])
        assert results[h.key]['values'].tobytes() == expected.tobytes()
        np.testing.assert_array_equal(results[h.key]['visits'], [len(seeds[i])]*len(expected))
    with pytest.raises(ValueError): host.complete_results(handles, bank, jobs, values[:-1])
    live = np.array([True, True, False, False, False])
    host.validate_fork(np.array([0, 0, 1]), np.array([2, 3, 4]), live, 5)
    for destinations in ([2, 2, 4], [0, 3, 4], [2, 3, 5]):
        with pytest.raises(ValueError): host.validate_fork(np.array([0, 0, 1]), np.array(destinations), live, 5)


@pytest.mark.parametrize('limit', [0, 1200, 100000])
def test_exact_cache_miss_indices_preserve_original_rows(cuda_battle_artifacts, limit):
    from rl.search.rollout_inference import ExactInputCache
    for cache in (ExactInputCache(limit), cuda_battle_artifacts[2].ExactValueCache(limit)):
        for ids in ([7, 7, 3, 9, 3], [3, 5, 5, 7], [9, 3, 9]):
            obs = dict(ids=np.asarray(ids, np.int64)[:, None])
            def forward(selected, rows):
                assert rows.dtype == np.int64 and np.all(rows[:-1] < rows[1:])
                np.testing.assert_array_equal(selected['ids'], obs['ids'][rows])
                return np.empty((len(rows), 0), np.float32), selected['ids'][:, 0].astype(np.float32)
            _, values = cache.forward(obs, forward, indexed=True)
            np.testing.assert_array_equal(values, ids)
