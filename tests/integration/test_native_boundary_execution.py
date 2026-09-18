"""Native search horizon survives pending input and card-triggered end of turn."""
import pytest
import slaythespire as sts

from env.core.native_actions import complete_actions

pytestmark = pytest.mark.native
END = 4 << 29
SELECT = 2 << 29


@pytest.mark.parametrize('choice', [0, 3])
def test_codex_pending_then_selection_stops_before_next_player_turn(combat_factory, choice):
    bc = combat_factory(['DEFEND_RED'] * 10, relics=["Nilry's Codex"]).thick.bc
    bc.apply_custom_state({'player': {'energy': 3},
                           'monsters': [{'move': 'JAW_WORM_CHOMP'}]})
    before = bc.to_dict()
    pending = bc.preview_end_turn_result(END)
    assert pending['input_state'] == 2  # CARD_SELECT is legal, not an error.
    assert not pending['post_enemy_turn_boundary'] and pending['outcome'] == 0
    assert bc.to_dict() == before
    assert bc.preview_end_turn_boundary(END) is None  # legacy API is unchanged.

    branch = bc.clone()
    assert branch.execute_to_enemy_boundary(END) == pending
    assert branch.select_info()['context'] == 'CODEX'
    assert SELECT | choice in complete_actions(branch)
    boundary = branch.execute_to_enemy_boundary(SELECT | choice)
    assert boundary == branch.to_dict()
    assert boundary['post_enemy_turn_boundary'] and boundary['outcome'] == 0
    assert boundary['turn'] == before['turn'] + 1
    assert boundary['player']['energy'] == 0
    assert len(boundary['hand']) == 0  # next turn has not drawn a hand.
    assert boundary['player']['hp'] == before['player']['hp'] - 11
    assert len(boundary['draw']) + len(boundary['discard']) == 10 + int(choice != 3)
    assert bc.to_dict() == before
    with pytest.raises(ValueError, match='legal live action before the enemy boundary'):
        branch.execute_to_enemy_boundary(END)
    assert branch.to_dict() == boundary


def test_pending_clone_keeps_armed_stop_when_selection_uses_legacy_execute(combat_factory):
    bc = combat_factory(['DEFEND_RED'] * 10, relics=["Nilry's Codex"]).thick.bc
    bc.execute_to_enemy_boundary(END)
    copied = bc.clone()
    copied.execute(sts.Action(SELECT | 3))
    assert copied.to_dict()['post_enemy_turn_boundary']
    assert not bc.to_dict()['post_enemy_turn_boundary']
    assert bc.select_info()['context'] == 'CODEX'


@pytest.mark.parametrize('codex', [False, True])
def test_time_warp_twelfth_card_stops_at_same_horizon(combat_factory, codex):
    bc = combat_factory(['POMMEL_STRIKE'] * 25, encounter='Time Eater',
                        relics=["Nilry's Codex"] if codex else []).thick.bc
    bc.apply_custom_state({'player': {'energy': 20, 'hp': 80},
                           'hand': ['POMMEL_STRIKE'] * 5,
                           'draw': ['POMMEL_STRIKE'] * 20, 'discard': []})
    turn = bc.to_dict()['turn']
    # Real card plays create Time Warp 11; no debug status setter is used.
    for _ in range(11):
        bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert ('Time Warp', 11) in bc.to_dict()['monsters'][0]['powers']
    before = bc.to_dict()
    old = bc.clone()
    old.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    if codex:
        old.execute(sts.Action(SELECT | 3))
    assert old.to_dict()['turn'] == turn + 1
    assert not old.to_dict()['post_enemy_turn_boundary']
    assert old.to_dict()['player']['energy'] == 3

    result = bc.execute_to_enemy_boundary(0)  # CARD source0,target0.
    if codex:
        assert result['input_state'] == 2 and not result['post_enemy_turn_boundary']
        assert bc.select_info()['context'] == 'CODEX'
        result = bc.execute_to_enemy_boundary(SELECT | 3)
    assert result['post_enemy_turn_boundary'] and result['outcome'] == 0
    assert result['turn'] == turn + 1
    assert result['player']['energy'] == 0 and len(result['hand']) == 0
    assert result['player']['hp'] == old.to_dict()['player']['hp']
    assert ('Strength', 2) in result['monsters'][0]['powers']
    assert result['monsters'][0]['hp'] == before['monsters'][0]['hp'] - 9


def test_boundary_mutator_ordinary_terminal_and_illegal_actions(combat_factory):
    bc = combat_factory(['DEFEND_RED'] * 5).thick.bc
    before = bc.to_dict()
    with pytest.raises(ValueError, match='legal live action'):
        bc.execute_to_enemy_boundary(SELECT)
    assert bc.to_dict() == before
    result = bc.execute_to_enemy_boundary(0)
    assert result['input_state'] == 1 and not result['post_enemy_turn_boundary']
    assert result['turn'] == before['turn']
    assert result['player']['block'] == 5
    # The stop remains armed even if the next call uses the old execute API.
    bc.execute(sts.Action(END))
    assert bc.to_dict()['post_enemy_turn_boundary']

    doomed = combat_factory(['DEFEND_RED'] * 5).thick.bc
    doomed.apply_custom_state({'player': {'hp': 1, 'energy': 0},
                               'monsters': [{'move': 'JAW_WORM_CHOMP'}]})
    terminal = doomed.execute_to_enemy_boundary(END)
    assert terminal['outcome'] == 2 and terminal['player']['hp'] <= 0
    assert not terminal['post_enemy_turn_boundary']
    with pytest.raises(ValueError, match='legal live action'):
        doomed.execute_to_enemy_boundary(END)
    assert doomed.to_dict() == terminal
