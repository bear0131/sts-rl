"""BRIDGE41: empty/failed partial edits cannot consume a combat decision."""
from copy import deepcopy

import numpy as np
import pytest
import slaythespire as sts

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _assert_same(actual, expected):
    assert actual.to_dict() == expected.to_dict()
    assert actual.card_state() == expected.card_state()
    def actions(battle):
        return [(int(a.action_type), a.source_idx, a.target_idx) for a in battle.rule_actions()]
    assert actions(actual) == actions(expected)
    for key, value in expected.compact_leaf_arrays().items():
        np.testing.assert_array_equal(actual.compact_leaf_arrays()[key], value)


@pytest.fixture
def edit_battle(combat_factory):
    def make(stage):
        battle = combat_factory(['STRIKE_RED'] * 5, 'Cultist').thick.bc
        if stage in ('selection', 'armed_selection'):
            battle.apply_custom_state({'hand': ['SECRET_TECHNIQUE'],
                'draw': ['DEFEND_RED', 'SHRUG_IT_OFF'], 'discard': [], 'exhaust': []})
            if stage == 'armed_selection':
                battle.execute_to_enemy_boundary(0)
            else:
                battle.execute(sts.Action(sts.ActionType.CARD, 0))
            assert int(battle.input_state) == 2
        elif stage == 'boundary':
            battle.execute_to_enemy_boundary(4 << 29)
            assert battle.to_dict()['post_enemy_turn_boundary']
        elif stage == 'terminal':
            battle.set_player_strength(999)
            battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
            assert battle.is_battle_over
        return battle
    return make


@pytest.mark.parametrize('stage', ['player', 'selection', 'armed_selection', 'boundary', 'terminal'])
@pytest.mark.parametrize('spec', [{}, {'player': {}}, {'monsters': []}, {'monsters': [{}]},
    {'player': {}, 'monsters': [{}]}, {'player': {'counters': {}, 'powers': []}},
    {'monsters': [{'powers': []}]}])
def test_empty_edits_preserve_state_and_next_transition(edit_battle, stage, spec):
    battle = edit_battle(stage)
    control = battle.clone()
    battle.apply_custom_state(deepcopy(spec))
    _assert_same(battle, control)
    # Exercise the original decision, including a queued draw selection. This
    # also detects hidden lifecycle/RNG changes absent from the snapshot.
    if stage in ('player', 'selection', 'armed_selection'):
        action = control.legal_actions()[0]
        battle.execute(action)
        control.execute(action)
        _assert_same(battle, control)
        if stage == 'armed_selection':
            battle.execute(sts.Action(sts.ActionType.END_TURN))
            control.execute(sts.Action(sts.ActionType.END_TURN))
            _assert_same(battle, control)
            assert battle.to_dict()['post_enemy_turn_boundary']


@pytest.mark.parametrize('encounter', ['Gremlin Leader', 'Automaton', 'Collector'])
def test_partial_monster_edits_leave_absent_slots_untouched(combat_factory, encounter):
    battle = combat_factory(['STRIKE_RED'] * 5, encounter).thick.bc
    before = battle.to_dict()
    edits = [{} for _ in before['monsters']]
    control = battle.clone()
    battle.apply_custom_state({'monsters': edits})
    _assert_same(battle, control)
    present = next(i for i, m in enumerate(before['monsters']) if m['id'] != 'INVALID = 0')
    edits[present] = {'block': 7}
    battle.apply_custom_state({'monsters': edits})
    after = battle.to_dict()['monsters']
    assert after[present]['block'] == 7
    for i, monster in enumerate(before['monsters']):
        if i != present:
            assert after[i] == monster
    missing = next(i for i, m in enumerate(before['monsters']) if m['id'] == 'INVALID = 0')
    edits[missing] = {'hp': 1}
    control = battle.clone()
    with pytest.raises(ValueError, match='absent monster'):
        battle.apply_custom_state({'player': {'energy': 0}, 'monsters': edits})
    _assert_same(battle, control)


@pytest.mark.parametrize('stage', ['player', 'selection', 'armed_selection', 'boundary'])
@pytest.mark.parametrize('spec', [
    {'player': {'energy': 0}, 'hand': ['STRIKE_RED', 'NOT_A_CARD']},
    {'hand': ['BASH'], 'draw': ['DEFEND_RED', 'NOT_A_CARD']},
    {'player': {'energy': 0}, 'monsters': [{'hp': 'not-an-integer'}]},
    {'player': {'energy': 0}, 'draw': [object()]},
    {'monsters': [{}, {'block': 1}]},
])
def test_failed_edit_is_atomic(edit_battle, stage, spec):
    battle = edit_battle(stage)
    control = battle.clone()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        battle.apply_custom_state(spec)
    _assert_same(battle, control)
    if stage in ('player', 'selection', 'armed_selection'):
        action = control.legal_actions()[0]
        battle.execute(action)
        control.execute(action)
        _assert_same(battle, control)
        if stage == 'armed_selection':
            battle.execute(sts.Action(sts.ActionType.END_TURN))
            control.execute(sts.Action(sts.ActionType.END_TURN))
            _assert_same(battle, control)
            assert battle.to_dict()['post_enemy_turn_boundary']


def test_empty_card_pile_is_an_actual_edit(edit_battle):
    battle = edit_battle('player')
    assert battle.to_dict()['hand']
    battle.apply_custom_state({'hand': []})
    assert battle.to_dict()['hand'] == []
