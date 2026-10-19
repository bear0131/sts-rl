import pytest

from bridge.event_contract import first_two_match_slots


def test_fixed_first_pair_preserves_board_slots():
    assert first_two_match_slots(range(12)) == (0, 1)
    assert first_two_match_slots([8, 5, 2, 11]) == (2, 5)


def test_mismatch_repeats_first_pair_without_memory_or_randomness():
    slots = [0, 1, 4, 8]
    assert [first_two_match_slots(slots) for _ in range(5)] == [(0, 1)] * 5
    assert slots == [0, 1, 4, 8]


def test_removed_pair_is_not_selected_again():
    assert first_two_match_slots([2, 3, 4, 5]) == (2, 3)


@pytest.mark.parametrize('slots', [[], [0], [0, 0], [-1, 1], [0, 12],
                                  [False, 1], ['0', 1], [0.0, 1]])
def test_invalid_public_slot_contract_fails_closed(slots):
    with pytest.raises(ValueError):
        first_two_match_slots(slots)
