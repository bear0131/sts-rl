from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rl.training.ring_allreduce import RingAllReduce, ParallelRingAllReduce


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


@pytest.mark.parametrize("streams", [2, 4])
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
