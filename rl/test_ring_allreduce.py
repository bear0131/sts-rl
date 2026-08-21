from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor

import torch

from rl.ring_allreduce import RingAllReduce


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
