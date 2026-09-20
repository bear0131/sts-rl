"""Safe Time Warp reset can preserve enough real next-turn plays to survive."""
import pytest
import slaythespire as sts

from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig

pytestmark = pytest.mark.native
END = 4 << 29


def _battle(combat_factory, *, count=11, move='TIME_EATER_RIPPLE', hp=35):
    env = combat_factory(['DEFEND_RED'] * 10, encounter='Time Eater')
    env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 20)
    env.thick.gc.set_deck([('DEFEND_RED', 1)] * 10)
    env.thick.bc = sts.create_battle(env.thick.gc, 'Time Eater')
    battle = env.thick.bc
    # Native card plays establish Time Warp without an artificial status setter.
    # Finesse avoids the temporary Strength noise from repeatedly playing Flex.
    for n in (min(count, 10), max(0, count - 10)):
        battle.apply_custom_state({'hand': ['FINESSE'] * n})
        for _ in range(n):
            battle.execute(sts.Action(0))
    battle.apply_custom_state({
        'player': {'hp': hp, 'energy': 1, 'block': 0}, 'hand': ['DEFEND_RED'],
        'draw': [('DEFEND_RED', 1)] * 5, 'discard': ['STRIKE_RED'], 'exhaust': [],
        'monsters': [{'hp': 400, 'move': move}],
    })
    assert dict(battle.to_dict()['monsters'][0]['powers']).get('Time Warp', 0) == count
    return battle


def _planner():
    return NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=2048))


def test_clearing_eleven_during_ripple_survives_real_following_turn(combat_factory):
    battle = _battle(combat_factory)
    before = battle.to_dict()
    result = _planner().choose(battle, state_nonce=1, public_relics=())
    assert result.action_bits == 0
    assert result.diagnostics['root_time_warp_penalties']
    assert battle.to_dict() == before

    reset = battle.clone()
    reset.execute(sts.Action(0))
    carried = battle.clone()
    carried.execute(sts.Action(END))
    # Actual first enemy phase is equally harmless, but only reset pays +2Str.
    assert reset.to_dict()['player']['hp'] == carried.to_dict()['player']['hp'] == 35
    assert reset.to_dict()['turn'] == carried.to_dict()['turn'] == 1
    assert dict(reset.to_dict()['monsters'][0]['powers'])['Strength'] == 2
    assert dict(carried.to_dict()['monsters'][0]['powers'])['Time Warp'] == 11
    assert reset.to_dict()['hand'] == carried.to_dict()['hand'] == [('DEFEND_RED', 1, 1)] * 5
    # The same real three defensive plays are available only after resetting.
    for _ in range(3):
        reset.execute(sts.Action(0))
    reset.execute(sts.Action(END))
    carried.execute(sts.Action(0))
    assert reset.to_dict()['turn'] == 2
    assert reset.to_dict()['player']['hp'] == 8
    assert reset.to_dict()['outcome'] == 0
    assert carried.to_dict()['player']['hp'] == 0
    assert carried.to_dict()['outcome'] == 2


def test_attacking_intent_does_not_activate_reset_opportunity(combat_factory):
    battle = _battle(combat_factory, move='TIME_EATER_REVERBERATE', hp=1)
    battle.apply_custom_state({
        'player': {'powers': [('DEXTERITY', 40)]},
        'hand': ['FLEX', 'DEFEND_RED'],
    })
    result = _planner().choose(battle, state_nonce=2, public_relics=())
    assert result.diagnostics['root_time_warp_penalties'] == {}
    assert result.diagnostics['root_outcomes']['0']['deaths'] == 2
    assert result.diagnostics['root_outcomes']['1']['deaths'] == 0
    assert result.action_bits == 1


def test_unsafe_twelfth_card_cannot_outweigh_immediate_death(combat_factory):
    battle = _battle(combat_factory, hp=1)
    battle.apply_custom_state({'hand': ['BLOODLETTING']})
    result = _planner().choose(battle, state_nonce=3, public_relics=())
    assert result.diagnostics['root_time_warp_penalties']
    assert result.diagnostics['root_outcomes']['0']['deaths'] == 2
    assert result.diagnostics['root_outcomes'][str(END)]['deaths'] == 0
    assert result.action_bits == END


def test_no_available_twelfth_play_remains_legal_end(combat_factory):
    battle = _battle(combat_factory)
    battle.apply_custom_state({'hand': ['BASH'], 'player': {'energy': 0}})
    result = _planner().choose(battle, state_nonce=4, public_relics=())
    assert result.action_bits == END
    assert result.diagnostics['root_actions'] == 1


def test_prior_does_not_value_manufacturing_eleven_from_ten(combat_factory):
    battle = _battle(combat_factory, count=10)
    result = _planner().choose(battle, state_nonce=5, public_relics=())
    assert result.diagnostics['root_time_warp_penalties'] == {}
    assert result.action_bits == END


def test_twelfth_card_immediate_kill_remains_a_native_win(combat_factory):
    battle = _battle(combat_factory)
    battle.apply_custom_state({'hand': ['STRIKE_RED', 'FINESSE'],
                               'monsters': [{'hp': 1}]})
    result = _planner().choose(battle, state_nonce=6, public_relics=())
    assert result.diagnostics['root_outcomes']['0']['wins'] == 2
    assert result.action_bits == 0
    battle.execute(sts.Action(0))
    assert battle.to_dict()['outcome'] == 1
