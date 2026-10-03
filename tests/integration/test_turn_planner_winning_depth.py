"""Break completed-horizon ties without changing terminal/HP preferences.

Reduced public mechanism from R4 seed150037/floor24/turn3. This is not a
private-state replay and does not reproduce the original 2375 preceding cards.
"""
import json

import pytest
import slaythespire as sts

from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig
from rl.evaluation.baselines.turn_planner import TurnPlannerStrategy

pytestmark = pytest.mark.native
END = 4 << 29


def _chosen_loop(combat_factory, *, combust=True):
    battle = combat_factory(['STRIKE_RED'] * 10, encounter='Chosen',
                            relics=['Medical Kit']).thick.bc
    # HEX is a real binary native status, unsupported by custom-state injection
    # and omitted from the numeric power list. Obtain it from Chosen's action.
    battle.apply_custom_state({'monsters': [{'move': 'CHOSEN_HEX'}]})
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    battle.apply_custom_state({
        'player': {'hp': 70, 'energy': 3,
                   'powers': [('DARK_EMBRACE', 1), ('COMBUST', 5 if combust else 0)]},
        'hand': ['DAZED'] * 8, 'draw': ['DAZED'] * 2, 'discard': [], 'exhaust': [],
        'monsters': [{'hp': 2, 'block': 0}],
    })
    return battle


def _planner(lookahead_turns):
    return NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=2048,
                                             lookahead_turns=lookahead_turns))


@pytest.mark.parametrize('lookahead_turns', [1, 2])
def test_medical_kit_hex_loop_ends_an_equivalent_proved_win(combat_factory, lookahead_turns):
    battle = _chosen_loop(combat_factory)
    before = battle.to_dict()
    loop = battle.clone()
    loop.execute(sts.Action(sts.ActionType.CARD, 0))
    after = loop.to_dict()
    assert len(after['hand']) == len(before['hand']) == 8
    assert len(after['draw']) == len(before['draw']) == 2
    assert len(after['exhaust']) == len(before['exhaust']) + 1
    assert after['monsters'][0]['hp'] == before['monsters'][0]['hp'] == 2
    direct = battle.clone()
    direct.execute(sts.Action(sts.ActionType.END_TURN))
    assert direct.to_dict()['outcome'] == 1
    assert direct.to_dict()['player']['hp'] == 69

    result = _planner(lookahead_turns).choose(battle, state_nonce=2375, public_relics=('Medical Kit',))
    values = result.diagnostics['root_values']
    assert len(values) == 9 and len(set(values.values())) == 1
    outcomes = result.diagnostics['root_outcomes']
    # This is the old v5 tie rule, applied to actual native search leaves:
    # every root wins twice with exactly the same HP utility, so bits0 won.
    old_choice = max((int(bits) for bits in values), key=lambda bits: (
        -outcomes[str(bits)]['deaths'], outcomes[str(bits)]['wins'],
        values[str(bits)], -bits))
    assert old_choice == 0
    assert result.action_bits == END
    assert outcomes[str(END)]['winning_steps_mean'] == 1
    assert outcomes[str(END)]['winning_steps_total'] == 2
    assert all(outcomes[str(i)]['winning_steps_mean'] == 2 for i in range(8))
    assert all(outcomes[str(i)]['winning_steps_total'] == 4 for i in range(8))
    assert battle.to_dict() == before
    json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize('lookahead_turns', [1, 2])
def test_slower_higher_hp_victory_still_beats_immediate_kill(combat_factory, lookahead_turns):
    battle = combat_factory(['STRIKE_RED'] * 5, encounter='Spheric Guardian').thick.bc
    battle.apply_custom_state({
        'player': {'hp': 71, 'max_hp': 85, 'energy': 4, 'powers': [('STRENGTH', 4)]},
        'hand': [('STRIKE_RED', 1), ('REAPER', 0)],
        'monsters': [{'hp': 13, 'block': 0}],
    })
    result = _planner(lookahead_turns).choose(battle, state_nonce=10025, public_relics=())
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes['0']['winning_steps_mean'] == 1
    assert outcomes['1']['winning_steps_mean'] == 2
    assert outcomes['1']['heuristic_mean'] > outcomes['0']['heuristic_mean']
    assert result.action_bits == 1  # Reaper then Strike+, despite longer win.
    battle.execute(sts.Action(result.action_bits))
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['outcome'] == 1 and battle.to_dict()['player']['hp'] == 79


@pytest.mark.parametrize('lookahead_turns,boundary,model', [
    (1, 'enemy-phase', False),
    (1, 'next-decision', False), (1, 'next-decision', True),
])
def test_live_hex_loop_reaches_equal_boundary_sooner(combat_factory, lookahead_turns, boundary, model):
    battle = _chosen_loop(combat_factory, combust=False)
    before = battle.to_dict()
    planner = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=2048,
        lookahead_turns=lookahead_turns, leaf_boundary=boundary))
    result = planner.choose(battle, state_nonce=2375, public_relics=('Medical Kit',),
        leaf_evaluator=(lambda states: [.5] * len(states)) if model else None)
    outcomes = result.diagnostics['root_outcomes']
    assert all(o['wins'] == 0 for o in outcomes.values())
    assert all(o['winning_steps_total'] == o['winning_steps_mean'] == 0 for o in outcomes.values())
    assert all(o['boundary_live'] == 2 and o['unresolved'] == 0 for o in outcomes.values())
    assert len(set(result.diagnostics['root_values'].values())) == 1
    assert result.action_bits == END
    assert outcomes[str(END)]['live_horizon_steps_mean'] == lookahead_turns
    assert all(outcomes[str(i)]['live_horizon_steps_mean'] == lookahead_turns + 1
               for i in range(8))
    assert result.diagnostics['native_transitions'] <= result.diagnostics['effective_total_budget']
    assert battle.to_dict() == before
    battle.execute(sts.Action(result.action_bits))
    assert battle.turn > before['turn'] and battle.to_dict()['outcome'] == 0


def test_unfinished_two_turn_hex_roots_keep_previous_tie(combat_factory):
    battle = _chosen_loop(combat_factory, combust=False)
    result = _planner(2).choose(battle, state_nonce=2375, public_relics=('Medical Kit',))
    outcomes = result.diagnostics['root_outcomes']
    assert all(outcomes[str(i)]['wins'] == outcomes[str(i)]['boundary_live'] == 0
               and outcomes[str(i)]['unresolved'] > 0 for i in range(8))
    assert outcomes[str(END)]['deaths'] == 2
    assert all(o['live_horizon_steps_total'] == 0 and o['live_horizon_steps_mean'] is None
               for o in outcomes.values())
    assert result.action_bits == 0


@pytest.mark.parametrize('lookahead_turns', [1, 2])
def test_redundant_block_does_not_delay_completed_live_horizon(combat_factory, lookahead_turns):
    battle = combat_factory(['WOUND'] * 15).thick.bc
    battle.apply_custom_state({'hand': ['DEFEND_RED'], 'draw': ['WOUND'] * 15,
        'discard': [], 'player': {'hp': 80, 'energy': 3, 'block': 999},
        'monsters': [{'move': 'JAW_WORM_CHOMP'}]})
    before = battle.to_dict()
    result = _planner(lookahead_turns).choose(battle, state_nonce=42)
    outcomes = result.diagnostics['root_outcomes']
    assert len(set(result.diagnostics['root_values'].values())) == 1
    assert all(o['boundary_live'] == 2 and o['unresolved'] == 0 for o in outcomes.values())
    assert outcomes[str(END)]['live_horizon_steps_mean'] == lookahead_turns
    assert outcomes['0']['live_horizon_steps_mean'] == lookahead_turns + 1
    assert result.action_bits == END
    assert battle.to_dict() == before


def test_manifest_declares_secondary_winning_length_preference():
    manifest = TurnPlannerStrategy().manifest()
    assert manifest['winning_tie_break'] == (
        'fewest-actions-to-equally-valued-sampled-wins')
    assert manifest['live_tie_break'] == 'fewest-actions-to-equally-valued-completed-live-horizons'
    assert manifest['live_tie_scope'] == 'exact-primary-and-winning-length-ties-with-no-fallback'


def test_dazed_is_not_a_legal_active_card_without_medical_kit(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10).thick.bc
    battle.apply_custom_state({'hand': ['DAZED'] * 8,
                               'player': {'powers': [('DARK_EMBRACE', 1)]}})
    legal = battle.rule_actions()
    assert len(legal) == 1 and legal[0].action_type == sts.ActionType.END_TURN
