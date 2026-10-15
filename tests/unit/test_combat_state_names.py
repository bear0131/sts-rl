"""Name memoization cannot retain mutable game or reward state."""
from types import SimpleNamespace

import pytest

from bridge.schemas import combat_state as schema


def test_card_name_cache_is_bounded_and_keeps_unknown_names():
    schema._sim_card_id.cache_clear()
    for index in range(1100):
        assert schema._sim_card_id(f'UNKNOWN_CARD_{index}') == f'Unknown Card {index}'
    assert schema._sim_card_id.cache_info().currsize <= 1024
    for name, expected in schema._CARD_OVERRIDE.items():
        assert schema._sim_card_id(name) == expected
    assert schema._sim_card_id('STRIKE_RED') == 'Strike_R'


def test_enum_tail_keeps_mutable_and_unhashable_fallback():
    class Changing:
        __hash__ = None
        tail = 'FIRST'

        def __str__(self):
            return 'Mutable.' + self.tail
    value = Changing()
    assert schema._enum_tail(value) == 'FIRST'
    value.tail = 'SECOND'
    assert schema._enum_tail(value) == 'SECOND'
    assert schema._enum_tail(['A.B']) == "B']"
    assert schema._enum_tail(None) == 'None'

    class ChangingString(str):
        __hash__ = None
        tail = 'FIRST'

        def __str__(self):
            return 'Mutable.' + self.tail
    text = ChangingString('original')
    assert schema._enum_tail(text) == 'FIRST'
    text.tail = 'SECOND'
    assert schema._enum_tail(text) == 'SECOND'


@pytest.mark.native
def test_native_enum_cache_keeps_types_names_and_live_upgrades():
    import slaythespire as sts
    schema._fixed_enum_tail.cache_clear()
    for kind in (sts.CardId, sts.MonsterEncounter, sts.CharacterClass, sts.Room):
        for name, value in kind.__members__.items():
            expected = str(value).split('.')[-1]
            assert schema._enum_tail(value) == expected
            assert schema._enum_tail(kind(int(value))) == expected
    for value in (sts.CardId.STRIKE_RED, sts.CharacterClass.IRONCLAD, sts.Room.BOSS):
        assert schema._enum_tail(value) == str(value).split('.')[-1]
    reward = SimpleNamespace(id=sts.CardId.STRIKE_RED, upgrade_count=0)
    assert schema._card_reward_entry(reward) == ('Strike_R', 0)
    reward.upgrade_count = 3
    assert schema._card_reward_entry(reward) == ('Strike_R', 3)
    assert schema._fixed_enum_tail.cache_info().currsize <= 512
