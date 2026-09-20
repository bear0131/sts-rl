"""Public delayed draw survives value-leaf cleanup, never checkpoint exports."""
from copy import deepcopy

import numpy as np
import pytest
import slaythespire as sts

from schemas.combat_state import normalize_sim
from env.search.turn_planner import (
    NativeTurnPlanner, TurnPlannerConfig, public_planner_state, state_value,
)

pytestmark = pytest.mark.native
END = 4 << 29


def _third_card(combat_factory, *, relics=('Pocketwatch',)):
    env = combat_factory(['STRIKE_RED'] * 15, relics=relics)
    env.thick.gc.ascension = 20
    env.thick.bc = sts.create_battle(env.thick.gc, 'Jaw Worm')
    battle = env.thick.bc
    battle.apply_custom_state({
        'player': {'hp': 75, 'max_hp': 75, 'energy': 3},
        'monsters': [{'hp': 45, 'block': 0, 'move': 'JAW_WORM_BELLOW'}],
        'hand': ['DEFEND_RED'] * 3 + ['ANGER'], 'draw': ['ANGER'] * 8,
        'discard': [], 'exhaust': [],
    })
    for _ in range(3):
        battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert battle.to_dict()['player']['combat_counters']['cards_played'] == 3
    return battle


def _end(battle, *, fourth=False, boundary=True):
    branch = battle.clone()
    if fourth:
        branch.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    if boundary:
        branch.execute_to_enemy_boundary(END)
    else:
        branch.execute(sts.Action(END))
    return branch


def _credit(raw):
    without = deepcopy(raw)
    without['player'].pop('boundary_resources')
    return state_value(raw, horizon=3) - state_value(without, horizon=3)


def test_boundary_keeps_public_three_vs_four_eligibility_but_raw_counters_stay_clean(combat_factory):
    battle = _third_card(combat_factory)
    three, four = _end(battle), _end(battle, fourth=True)
    assert three.public_boundary_resources() == {
        'schema_version': 1, 'cards_played': 3, 'base_draw': 5, 'pocketwatch': True}
    assert four.public_boundary_resources()['cards_played'] == 4
    assert three.to_dict()['player']['combat_counters']['cards_played'] == 0
    assert four.to_dict()['player']['combat_counters']['cards_played'] == 0
    assert len(_end(battle, boundary=False).to_dict()['hand']) == 8
    assert len(_end(battle, fourth=True, boundary=False).to_dict()['hand']) == 5
    assert _credit(public_planner_state(three)) == 3
    assert _credit(public_planner_state(four)) == 0


def test_fourth_free_attack_loses_more_next_hand_damage_than_it_deals(combat_factory):
    battle = _third_card(combat_factory)
    before = public_planner_state(battle)
    plan = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=8123, public_relics=['Pocketwatch'])
    assert plan.action_bits == END
    assert plan.diagnostics['root_values'][str(END)] > plan.diagnostics['root_values']['0']
    choices = [_end(battle, fourth=play, boundary=False) for play in (False, True)]
    for branch in choices:
        while branch.to_dict()['hand']:
            assert branch.to_dict()['hand'][0][0] == 'ANGER'
            branch.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    raw = [b.to_dict() for b in choices]
    assert raw[0]['player']['hp'] == raw[1]['player']['hp'] == 75
    assert raw[1]['monsters'][0]['hp'] - raw[0]['monsters'][0]['hp'] == 12
    assert public_planner_state(battle) == before


def test_without_pocketwatch_fourth_free_attack_remains_useful(combat_factory):
    battle = _third_card(combat_factory, relics=())
    assert not battle.public_boundary_resources()['pocketwatch']
    plan = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=8123, public_relics=())
    assert plan.action_bits == 0
    assert _credit(public_planner_state(_end(battle))) == 0


@pytest.mark.parametrize('retained,extra', [(4, 1), (5, 0), (8, 0)])
def test_pyramid_capacity_bounds_pocketwatch_additional_draw(combat_factory, retained, extra):
    battle = _third_card(combat_factory, relics=('Pocketwatch', 'Runic Pyramid'))
    battle.apply_custom_state({'hand': ['ANGER'] * retained, 'draw': ['ANGER'] * 10})
    boundary = _end(battle)
    raw = public_planner_state(boundary)
    assert len(raw['hand']) == retained
    assert _credit(raw) == extra
    assert len(_end(battle, boundary=False).to_dict()['hand']) == 10


def test_empty_future_piles_do_not_invent_bonus_draws(combat_factory):
    battle = _third_card(combat_factory)
    battle.apply_custom_state({'hand': [], 'draw': ['ANGER'] * 2, 'discard': []})
    raw = public_planner_state(_end(battle))
    assert _credit(raw) == 0
    assert len(_end(battle, boundary=False).to_dict()['hand']) == 2


def test_fourth_card_lethal_and_preventing_death_have_priority(combat_factory):
    battle = _third_card(combat_factory)
    battle.apply_custom_state({'monsters': [{'hp': 6}]})
    plan = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=8123, public_relics=['Pocketwatch'])
    assert plan.action_bits == 0
    assert plan.diagnostics['root_outcomes']['0']['wins'] == 2

    danger = _third_card(combat_factory)
    danger.apply_custom_state({'player': {'hp': 6, 'energy': 1, 'block': 0},
        'hand': [('DEFEND_RED', 1)], 'monsters': [{'move': 'JAW_WORM_CHOMP'}]})
    plan = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        danger, state_nonce=8123, public_relics=['Pocketwatch'])
    assert plan.action_bits == 0
    assert plan.diagnostics['root_outcomes']['0']['deaths'] == 0
    assert plan.diagnostics['root_outcomes'][str(END)]['deaths'] == 2


def test_metadata_is_read_only_clones_cleanly_and_never_enters_legacy_exports(combat_factory):
    battle = _third_card(combat_factory)
    raw, packed = battle.to_dict(), battle.compact_leaf_arrays()
    before = battle.public_boundary_resources()
    private = public_planner_state(battle)
    assert 'boundary_resources' not in raw['player']
    assert normalize_sim(private) == normalize_sim(raw)
    assert battle.to_dict() == raw
    for k, value in packed.items():
        np.testing.assert_array_equal(value, battle.compact_leaf_arrays()[k])
    for nonce in (11, 99):
        clone = battle.clone()
        clone.nosl_redeterminize(nonce)
        assert clone.public_boundary_resources() == before
    boundary = _end(battle)
    assert boundary.clone().public_boundary_resources() == boundary.public_boundary_resources()
    # Ordinary next-turn reset and custom state setup cannot leak a previous
    # projected boundary's eligibility into a later playable state.
    normal = _end(battle, boundary=False)
    assert normal.public_boundary_resources()['cards_played'] == 0
    boundary.apply_custom_state({'hand': ['ANGER']})
    assert not boundary.to_dict()['post_enemy_turn_boundary']
    assert boundary.public_boundary_resources()['cards_played'] == 0
    fresh = combat_factory(['STRIKE_RED'] * 10).thick.bc
    assert fresh.public_boundary_resources()['cards_played'] == 0
    assert not fresh.public_boundary_resources()['pocketwatch']


def test_missing_resource_api_fails_only_when_opt_in_planner_constructs(monkeypatch):
    import env.search.turn_planner as module
    class FlagsOnly:
        def public_player_power_flags(self):
            return None
    monkeypatch.setattr(module.sts, 'BattleContext', FlagsOnly)
    with pytest.raises(RuntimeError, match='public_boundary_resources'):
        NativeTurnPlanner()


@pytest.mark.parametrize('field,value', [('schema_version', 2), ('cards_played', -1),
                                       ('base_draw', True), ('pocketwatch', 1)])
def test_unknown_or_ill_typed_resource_metadata_fails_explicitly(combat_factory, field, value):
    battle = _third_card(combat_factory)
    class BadResource:
        def to_dict(self):
            return battle.to_dict()
        def public_player_power_flags(self):
            return battle.public_player_power_flags()
        def public_boundary_resources(self):
            return dict(battle.public_boundary_resources(), **{field: value})
    with pytest.raises(ValueError, match='public_boundary_resources schema'):
        public_planner_state(BadResource())
