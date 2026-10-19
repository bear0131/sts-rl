"""F03: real combat GRID choices use the full existing PICK_CARD segment."""
import numpy as np
import pytest

from bridge.real_game_env import RealGameAdapter
from env.core import action_space as A, game_actions as ga


def _adapter(count):
    adapter = RealGameAdapter()
    adapter.update({'in_game': True, 'available_commands': ['CHOOSE'], 'game_state': {
        'room_phase': 'COMBAT', 'screen_type': 'GRID',
        'combat_state': {'turn': 1},
        'screen_state': {'cards': [{'id': 'Bash', 'upgrades': 0} for _ in range(count)]},
    }})
    return adapter


@pytest.mark.parametrize('count', [2, 12, 96])
def test_real_grid_masks_and_commands_include_every_choice(count):
    adapter = _adapter(count)
    legal = adapter.legal_semantic_actions()
    assert legal == [ga.select(i) for i in range(count)]
    assert A.arg_step_name(None, legal, 'combat') == A.PICK_CARD
    np.testing.assert_array_equal(np.flatnonzero(adapter.action_mask()), np.arange(A._PICK, A._PICK + count))
    for index in (0, count - 1):
        action, pending = A.resolve(A._PICK + index, legal, None, 'combat')
        assert pending is None and action == ga.select(index)
        assert adapter.command_for_semantic(action) == f'CHOOSE {index}'


def test_real_grid_over_capacity_fails_without_map_action_spill():
    with pytest.raises(ValueError, match='96-slot policy contract'):
        _adapter(97).action_mask()
