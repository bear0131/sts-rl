"""No-world-line (NOSL) combat determinization helpers.

The simulator owns the actual RNG streams. This module only derives opaque,
experiment-local nonces and asks a cloned BattleContext to redeterminize itself.
It deliberately does not accept or inspect the live run seed/counters.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any


_MASK64 = (1 << 64) - 1


def _pack_u64(value: int) -> bytes:
    return struct.pack("<Q", int(value) & _MASK64)


class NOSLSampler:
    """Generate independent determinization nonces for combat search.

    ``state_nonce`` is an opaque identifier supplied by the caller. It must be
    generated independently of the live environment seed; the sampler never
    derives it from a BattleContext or reads simulator RNG state.
    """

    def __init__(self, master_nonce: int) -> None:
        self.master_nonce = int(master_nonce) & _MASK64

    def nonce(self, *, state_nonce: int, simulation_index: int) -> int:
        if int(simulation_index) < 0:
            raise ValueError("simulation_index must be non-negative")
        digest = hashlib.blake2b(
            _pack_u64(self.master_nonce)
            + _pack_u64(state_nonce)
            + _pack_u64(simulation_index),
            digest_size=8,
            person=b"sts-nosl",
        ).digest()
        return int.from_bytes(digest, "little")

    def clone_and_redeterminize(
        self,
        battle_context: Any,
        *,
        state_nonce: int,
        simulation_index: int,
    ) -> Any:
        """Clone a live BattleContext and sample its hidden future."""
        branch = battle_context.clone()
        branch.nosl_redeterminize(
            self.nonce(
                state_nonce=state_nonce,
                simulation_index=simulation_index,
            )
        )
        return branch

