from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rl.training.ring_allreduce import RingAllReduce, ParallelRingAllReduce


@pytest.mark.parametrize('different', ['extra', 'renamed', 'empty'])
@pytest.mark.parametrize('streams', [1, 8])
def test_scalar_schema_mismatch_rejects_all_ranks_without_corrupting_stop(different, streams):
    from rl.training.distributed_sb3 import average_scalar_dict

    lanes = []
    for _ in range(streams):
        links = [socket.socketpair() for _ in range(4)]
        lanes.append([RingAllReduce(rank=r, world_size=4, outgoing=links[r][0],
            incoming=links[(r-1) % 4][1], timeout=5.) for r in range(4)])
    rings = lanes[0] if streams == 1 else [ParallelRingAllReduce([lane[r] for lane in lanes]) for r in range(4)]
    try:
        def reduce(rank):
            values = {'epochs': 1., 'rows': 196608.}
            if rank == 0:
                if different == 'extra':
                    values['graph_upper_retired'] = 1.
                elif different == 'renamed':
                    values['other_rows'] = values.pop('rows')
                else:
                    values.clear()
            with pytest.raises(ValueError, match='scalar schema differs'):
                average_scalar_dict(values, rings[rank])
            # All headers were drained, and no unvalidated payload was sent.
            assert average_scalar_dict({'stop_requested': 0.}, rings[rank]) == {'stop_requested': 0.}
            fixed = {'epochs': 1., 'rows': 196608., 'graph_upper_retired': float(rank == 0)}
            assert average_scalar_dict(fixed, rings[rank]) == {'epochs': 1., 'rows': 196608., 'graph_upper_retired': .25}
            with pytest.raises(ValueError, match='scalar schema differs'):
                rings[rank].average(torch.zeros(65535 + (rank == 0)), schema=('boundary',))
            assert average_scalar_dict({'stop_requested': 0.}, rings[rank]) == {'stop_requested': 0.}
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(reduce, range(4)))
    finally:
        for ring in rings:
            ring.close()


def test_ring_allreduce_averages_uneven_flat_tensors():
    world_size = 4
    links = [socket.socketpair() for _ in range(world_size)]
    rings = [
        RingAllReduce(
            rank=rank,
            world_size=world_size,
            outgoing=links[rank][0],
            incoming=links[(rank - 1) % world_size][1],
            timeout=5.0,
        )
        for rank in range(world_size)
    ]
    inputs = [
        torch.arange(23, dtype=torch.float32) + rank * 100
        for rank in range(world_size)
    ]
    expected = torch.stack(inputs).mean(dim=0)

    try:
        with ThreadPoolExecutor(max_workers=world_size) as executor:
            results = list(executor.map(
                lambda pair: pair[0].average(pair[1]),
                zip(rings, inputs, strict=True),
            ))
        for averaged, _bytes_sent in results:
            torch.testing.assert_close(averaged, expected)
        assert sum(sent for _tensor, sent in results) == (
            2 * (world_size - 1) * expected.numel() * expected.element_size()
        )
    finally:
        for ring in rings:
            ring.close()


@pytest.mark.parametrize("streams", [2, 4, 8])
def test_parallel_ring_preserves_float32_order_across_large_and_small_collectives(streams):
    def make_rings():
        links = [socket.socketpair() for _ in range(4)]
        return [RingAllReduce(rank=r, world_size=4, outgoing=links[r][0],
            incoming=links[(r-1) % 4][1], timeout=5.) for r in range(4)]
    baseline = make_rings()
    lanes = [make_rings() for _ in range(streams)]
    striped = [ParallelRingAllReduce([lane[r] for lane in lanes]) for r in range(4)]
    generator = torch.Generator().manual_seed(20261109)
    try:
        # Reuse connections across scalar metrics, uneven large gradients and
        # threshold boundaries; no packet may spill into the next collective.
        for size in (1, 23, 65535, 65536, 65537, 262147, 3):
            inputs = [torch.randn(size, generator=generator) *
                      (10. ** ((torch.arange(size) % 6) * 3)) for _ in range(4)]
            results = []
            for rings in (baseline, striped):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results.append(list(pool.map(lambda pair: pair[0].average(pair[1]),
                        zip(rings, [x.clone() for x in inputs], strict=True))))
            for (expected, sent), (actual, actual_sent) in zip(*results, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert actual_sent == sent
            assert sum(n for _, n in results[1]) == 6 * size * 4
    finally:
        for ring in [*baseline, *striped]:
            ring.close()


def test_sparse_buckets_keep_original_fp32_addition_order():
    """Mixed magnitudes make a reassigned reduction owner observable."""
    from types import SimpleNamespace
    from rl.training.ppo_overlap import bucket_partitions

    sizes = [11, 3, 5, 24, 9]
    params = [torch.nn.Parameter(torch.empty(n)) for n in sizes]
    groups = [[params[1], params[3]], [params[0], params[2], params[4]]]
    schema = bucket_partitions(SimpleNamespace(parameters=lambda: iter(params)), groups, 4)
    pids = {id(p):i for i,p in enumerate(params)}
    ranks = range(4)
    links = [socket.socketpair() for _ in ranks]
    rings = [RingAllReduce(rank=r, world_size=4, outgoing=links[r][0],
                          incoming=links[(r-1)%4][1], timeout=5) for r in ranks]
    rng = torch.Generator().manual_seed(123)
    values = [torch.randn(sum(sizes), generator=rng) for _ in ranks]
    values[0].mul_(1e5)
    values[2] = -values[0]
    try:
        with ThreadPoolExecutor(max_workers=4) as worker:
            def reduce(r):
                full, _ = rings[r].average(values[r].clone())
                source = values[r].split(sizes)
                out = torch.empty_like(full)
                dest = out.split(sizes)
                for group, counts in zip(groups, schema, strict=True):
                    ids = [pids[id(p)] for p in group]
                    small = torch.cat([source[i] for i in ids])
                    rings[r].average(small, chunk_sizes=counts)
                    for i,x in zip(ids, small.split([sizes[i] for i in ids]), strict=True):
                        dest[i].copy_(x)
                assert torch.equal(full, out)
            list(worker.map(reduce, ranks))
    finally:
        for ring in rings:
            ring.close()
