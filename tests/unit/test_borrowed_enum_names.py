"""Native names must follow live fields even after unrelated name lookups."""

import random

import pytest
import slaythespire as sts

from bridge.schemas.combat_state import _enum_tail
from env.core.spiremap import _room_name


@pytest.mark.native
def test_borrowed_room_names_survive_mutation_and_cache_pressure():
    """OPS08: native field wrappers are not immutable dictionary keys."""
    rng = random.Random(678)
    contexts = [
        sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
        for seed in range(384)
    ]
    rooms = list(sts.Room.__members__.values())
    for _ in range(3000):
        for _ in range(4):
            name = f"FILLER_{rng.randrange(2048)}"
            assert _enum_tail(name) == name
        context = rng.choice(contexts)
        room = rng.choice(rooms)
        context.cur_room = room
        expected = str(room).split(".")[-1]
        assert _enum_tail(context.cur_room) == expected
        assert _room_name(context.cur_room) == expected
