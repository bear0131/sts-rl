"""Native one-turn planning contracts, using real simulator transitions.

Reaper is a reduced public fixture from development seed 10025/floor22/turn3,
not a replay of its hidden state. Other fixtures deliberately isolate mechanical
contracts; no lucky draw order is selected to make a planner succeed.
"""
import json

import pytest
import slaythespire as sts

from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig


def _snapshot(battle):
    return json.dumps(battle.to_dict(), sort_keys=True)


def _action_identity(action):
    return (int(action.action_type), action.source_idx, action.target_idx,
            tuple(action.selected_indices))


def _assert_legal(battle, result):
    chosen = sts.Action(result.action_bits)
    legal = battle.rule_actions()
    if chosen.action_type == sts.ActionType.POTION:
        assert any(p['slot'] == chosen.source_idx and
                   (not p['requires_target'] or chosen.target_idx in p['targets'])
                   for p in battle.legal_potions())
    else:
        assert _action_identity(chosen) in {_action_identity(a) for a in legal}
    return chosen


def _battle(combat_factory, spec, *, encounter='Jaw Worm'):
    env = combat_factory(['STRIKE_RED'] * 5, encounter=encounter, seed=42)
    env.thick.bc.apply_custom_state(spec)
    return env.thick.bc


def _planner():
    return NativeTurnPlanner(TurnPlannerConfig(
        worlds=2, node_budget=512, beam_width=8, max_depth=16,
        master_nonce=20260918))


def test_reaper_heals_before_an_available_immediate_kill(combat_factory):
    # Native controls: Strike+ kills at71HP; Reaper -> Strike+ kills at79HP.
    battle = _battle(combat_factory, {
        'player': {'hp': 71, 'max_hp': 85, 'energy': 4,
                   'powers': [('STRENGTH', 4), ('FRAIL', 5)]},
        'monsters': [{'hp': 13, 'block': 0}],
        'hand': [('STRIKE_RED', 1), ('REAPER', 0)],
    }, encounter='Spheric Guardian')
    direct = battle.clone()
    direct.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert int(direct.outcome) == 1
    assert direct.to_dict()['player']['hp'] == 71

    planner = NativeTurnPlanner(TurnPlannerConfig(
        worlds=2, node_budget=4096, beam_width=16, max_depth=16,
        master_nonce=20260918))
    result = planner.choose(battle, state_nonce=10025)
    chosen = _assert_legal(battle, result)
    assert chosen.action_type == sts.ActionType.CARD
    assert battle.to_dict()['hand'][chosen.source_idx][0] == 'REAPER'
    played = battle.clone()
    original_turn = played.to_dict()['turn']
    played.execute(chosen)
    next_result = planner.choose(played, state_nonce=10026)
    played.execute(_assert_legal(played, next_result))
    assert int(played.outcome) == 1
    assert played.to_dict()['turn'] == original_turn
    assert played.to_dict()['player']['hp'] == 79


def test_defends_against_visible_lethal_before_spending_last_energy(combat_factory):
    battle = _battle(combat_factory, {
        'player': {'hp': 7, 'energy': 1, 'block': 0},
        'monsters': [{'hp': 40, 'block': 0, 'move': 'JAW_WORM_CHOMP'}],
        'hand': [('STRIKE_RED', 0), ('DEFEND_RED', 0)],
    })
    doomed = battle.clone()
    doomed.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    doomed.execute(sts.Action(sts.ActionType.END_TURN))
    assert doomed.to_dict()['player']['hp'] <= 0

    result = _planner().choose(battle, state_nonce=43)
    chosen = _assert_legal(battle, result)
    assert chosen.action_type == sts.ActionType.CARD
    assert battle.to_dict()['hand'][chosen.source_idx][0] == 'DEFEND_RED'
    safe = battle.clone()
    safe.execute(chosen)
    boundary = safe.preview_end_turn_boundary(4 << 29)
    assert boundary is not None
    assert boundary['player']['hp'] == 1


def test_hidden_draw_permutation_and_future_rng_do_not_change_plan(combat_factory):
    battle = _battle(combat_factory, {
        'player': {'hp': 40, 'energy': 3},
        'hand': [('POMMEL_STRIKE', 0), ('DEFEND_RED', 0), ('STRIKE_RED', 0)],
        'draw': [('BASH', 0), ('ANGER', 0), ('STRIKE_RED', 0), ('DEFEND_RED', 0),
                 ('WOUND', 0), ('CARNAGE', 0), ('SHRUG_IT_OFF', 0)],
        'discard': [],
    })
    worlds, results = [], []
    for hidden_nonce in (11, 22, 33):
        alternate = battle.clone()
        alternate.nosl_redeterminize(hidden_nonce)
        worlds.append(alternate.to_dict())
        result = _planner().choose(alternate, state_nonce=777)
        _assert_legal(alternate, result)
        results.append(result)
    assert len({tuple(raw['draw']) for raw in worlds}) > 1
    public = []
    for raw in worlds:
        raw['draw'] = sorted(raw['draw'])
        public.append(raw)
    assert public[0] == public[1] == public[2]
    assert len({result.action_bits for result in results}) == 1
    assert results[0].diagnostics['root_values'] == results[1].diagnostics['root_values']
    assert results[0].diagnostics['root_values'] == results[2].diagnostics['root_values']


def test_search_preserves_live_state_and_future_native_transitions(combat_factory):
    battle = _battle(combat_factory, {
        'hand': [('POMMEL_STRIKE', 0), ('DEFEND_RED', 0)],
        'draw': [('STRIKE_RED', 0), ('WOUND', 0), ('BASH', 0), ('ANGER', 0),
                 ('DEFEND_RED', 0), ('SHRUG_IT_OFF', 0)],
        'discard': [],
    })
    pristine = battle.clone()
    before = _snapshot(battle)
    result = _planner().choose(battle, state_nonce=444)
    _assert_legal(battle, result)
    assert _snapshot(battle) == before
    # Comparing a future draw AND enemy intent catches RNG mutation that a
    # to_dict snapshot (which does not expose RNG streams) would miss.
    for action in (sts.Action(sts.ActionType.CARD, 0, 0),
                   sts.Action(sts.ActionType.END_TURN)):
        pristine.execute(action)
        battle.execute(action)
        assert _snapshot(battle) == _snapshot(pristine)


@pytest.mark.parametrize('card,extra,context', [
    ('HEADBUTT', {'discard': [('WOUND', 0)] * 12 + [('CARNAGE', 0)]}, 'HEADBUTT'),
    ('PURITY', {}, 'EXHAUST_MANY'),
])
def test_pending_native_selection_is_legal_and_live_selection_is_unchanged(
        combat_factory, card, extra, context):
    battle = _battle(combat_factory, {
        'hand': [(card, 0), ('WOUND', 0), ('STRIKE_RED', 0), ('DEFEND_RED', 0)],
        **extra,
    })
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.select_info()['context'] == context
    before = _snapshot(battle)
    descriptor = battle.select_info()
    result = _planner().choose(battle, state_nonce=81)
    chosen = _assert_legal(battle, result)
    assert chosen.action_type in (sts.ActionType.SINGLE_CARD_SELECT,
                                  sts.ActionType.MULTI_CARD_SELECT)
    assert _snapshot(battle) == before
    assert battle.select_info() == descriptor
    branch = battle.clone()
    branch.execute(chosen)
    assert branch.to_dict()['input_state'] == 1  # PLAYER_NORMAL; context can be stale.
    assert any(action.action_type == sts.ActionType.END_TURN
               for action in branch.rule_actions())
    assert _snapshot(battle) == before


def test_budget_reports_complete_root_coverage_and_native_work(combat_factory):
    battle = _battle(combat_factory, {
        'hand': [('BASH', 0), ('STRIKE_RED', 0), ('DEFEND_RED', 0),
                 ('ANGER', 0), ('SHRUG_IT_OFF', 0)],
        'draw': [('WOUND', 0)] * 5,
    })
    requested = 1
    worlds = 2
    planner = NativeTurnPlanner(TurnPlannerConfig(
        worlds=worlds, node_budget=requested, beam_width=2,
        max_depth=4, master_nonce=20260918))
    result = planner.choose(battle, state_nonce=888)
    _assert_legal(battle, result)
    diagnostics = result.diagnostics
    assert diagnostics['worlds_completed'] == worlds
    assert diagnostics['root_actions'] == len(battle.rule_actions())
    assert len(diagnostics['root_values']) == diagnostics['root_actions']
    assert diagnostics['effective_node_budget'] >= requested
    assert diagnostics['effective_node_budget'] >= diagnostics['root_actions'] * 2
    assert 0 < diagnostics['native_transitions'] <= worlds * diagnostics['effective_node_budget']
    assert 0 < diagnostics['expanded_nodes'] <= diagnostics['native_transitions']
    assert diagnostics['budget_exhausted']


def test_draw_pile_choice_keeps_public_candidate_index_before_sampling(combat_factory):
    # Secret Weapon's current selection is public, including its native pile
    # indices. A hidden-future shuffle must not remap this already-open choice.
    battle = _battle(combat_factory, {
        'player': {'energy': 3, 'block': 0},
        'monsters': [{'hp': 20, 'block': 0}],
        'hand': [('SECRET_WEAPON', 0)],
        'draw': [('WOUND', 0)] * 12 + [('BODY_SLAM', 0), ('BLUDGEON', 0)],
        'discard': [],
    })
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert battle.select_info()['context'] == 'SECRET_WEAPON'
    assert [action.source_idx for action in battle.rule_actions()] == [12, 13]
    before = _snapshot(battle)
    result = _planner().choose(battle, state_nonce=20260918)
    chosen = _assert_legal(battle, result)
    assert chosen.action_type == sts.ActionType.SINGLE_CARD_SELECT
    assert chosen.source_idx == 13
    assert _snapshot(battle) == before
    branch = battle.clone()
    branch.execute(chosen)
    assert branch.to_dict()['hand'] == [('BLUDGEON', 0, 3)]
    branch.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert int(branch.outcome) == 1
