"""Opt-in public binary effects never redefine legacy checkpoint observations."""
import json

import numpy as np
import pytest
import slaythespire as sts

from schemas.combat_state import normalize_sim
from env.search.turn_planner import (
    NativeTurnPlanner, TurnPlannerConfig, public_planner_state, state_value,
    no_remaining_damage_source,
)
from rl.evaluation.baselines.turn_planner import TurnPlannerStrategy
from rl.evaluation.strategies import DecisionRequest, strategy_manifest

pytestmark = pytest.mark.native


def corruption(battle):
    battle.apply_custom_state({'player': {'energy': 3},
        'hand': ['CORRUPTION', 'DEFEND_RED'], 'draw': ['STRIKE_RED'] * 5,
        'discard': [], 'exhaust': []})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))


def test_real_corruption_free_skill_and_legacy_exports(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10).thick.bc
    corruption(battle)
    raw = battle.to_dict()
    canonical = normalize_sim(raw)
    packed = battle.compact_leaf_arrays()
    flags = battle.public_player_power_flags()
    assert flags['schema_version'] == 1
    assert flags['flags']['CORRUPTION'] is True
    assert len(flags['flags']) == 6 and all(type(v) is bool for v in flags['flags'].values())
    assert not any(name == 'Corruption' for name, _ in raw['player']['powers'])
    assert raw['player']['energy'] == 0 and raw['hand'][0][2] == 0
    private = public_planner_state(battle)
    assert state_value(private, horizon=2) == state_value(raw, horizon=2) + 12
    assert normalize_sim(private) == canonical == normalize_sim(battle.to_dict())
    assert battle.to_dict() == raw
    for key, before in packed.items():
        np.testing.assert_array_equal(before, battle.compact_leaf_arrays()[key])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert battle.to_dict()['player']['energy'] == 0
    assert ('DEFEND_RED', 0) in battle.to_dict()['exhaust']
    control = combat_factory(['DEFEND_RED'] * 10).thick.bc
    control.apply_custom_state({'player': {'energy': 0}, 'hand': ['DEFEND_RED']})
    assert not any(a.action_type == sts.ActionType.CARD for a in control.rule_actions())


def test_barricade_real_play_preserves_block_through_enemy_phase(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10).thick.bc
    battle.apply_custom_state({'player': {'energy': 3, 'block': 12},
        'hand': ['BARRICADE'], 'monsters': [{'move': 'JAW_WORM_BELLOW'}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert battle.public_player_power_flags()['flags']['BARRICADE'] is True
    battle.execute_to_enemy_boundary(4 << 29)
    assert battle.to_dict()['player']['block'] == 12
    assert battle.public_player_power_flags()['flags']['BARRICADE'] is True


def test_public_binary_flags_do_not_invent_a_damage_source(combat_factory):
    battle = combat_factory(['DEFEND_RED'] * 10).thick.bc
    battle.apply_custom_state({'player': {'energy': 3}, 'hand': ['BARRICADE'],
        'draw': ['DEFEND_RED'] * 5, 'discard': [], 'exhaust': []})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert battle.public_player_power_flags()['flags']['BARRICADE'] is True
    assert no_remaining_damage_source(battle.to_dict(), ())
    assert no_remaining_damage_source(public_planner_state(battle), ())


def test_hex_native_application_and_orange_pellets_removal(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10, encounter='Chosen',
                            relics=['OrangePellets']).thick.bc
    battle.apply_custom_state({'monsters': [{'move': 'CHOSEN_HEX'}]})
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert battle.public_player_power_flags()['flags']['HEX'] is True
    battle.apply_custom_state({'player': {'energy': 4},
                               'hand': ['INFLAME', 'STRIKE_RED', 'DEFEND_RED']})
    for _ in range(3):
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.public_player_power_flags()['flags']['HEX'] is False


def test_pen_nib_public_flag_tracks_actual_attack_consumption(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10, relics=['Pen Nib'],
                            encounter='Lagavulin').thick.bc
    battle.apply_custom_state({'player': {'energy': 30}, 'hand': ['STRIKE_RED'] * 10})
    for _ in range(9):
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.public_player_power_flags()['flags']['PEN_NIB'] is True
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.public_player_power_flags()['flags']['PEN_NIB'] is False


def test_flags_survive_clone_and_nosl_without_mutating_root(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10).thick.bc
    corruption(battle)
    before, flags = battle.to_dict(), battle.public_player_power_flags()
    for nonce in (11, 99):
        sample = battle.clone()
        sample.nosl_redeterminize(nonce)
        assert sample.public_player_power_flags() == flags
    assert battle.to_dict() == before and battle.public_player_power_flags() == flags
    fresh = combat_factory(['STRIKE_RED'] * 10).thick.bc
    assert not any(fresh.public_player_power_flags()['flags'].values())


@pytest.mark.parametrize('turns', [1, 2])
def test_private_flags_are_traced_and_manifested(combat_factory, turns):
    env = combat_factory(['STRIKE_RED'] * 10, combat_action_contract='native-v1')
    corruption(env.thick.bc)
    traces = []
    strategy = TurnPlannerStrategy(TurnPlannerConfig(node_budget=32, lookahead_turns=turns), trace=traces.append)
    before = env.thick.bc.to_dict()
    strategy.decide([DecisionRequest(env, env._obs(), env._info(), 42, 0)])
    assert traces[0]['state']['player']['public_power_flags']['flags']['CORRUPTION'] is True
    manifest = strategy_manifest(strategy)
    assert manifest['version'] == 'ironclad-turn-v13-icecream-v1'
    assert manifest['public_power_flags'] == 'native-public-player-power-flags-v1-private-planner-state'
    assert env.thick.bc.to_dict() == before
    json.dumps(traces, allow_nan=False)


def test_old_native_capability_check_is_on_construction(monkeypatch):
    from rl.evaluation.baselines.combat_rules import CombatRuleStrategy
    import env.search.turn_planner as module
    monkeypatch.setattr(module.sts, 'BattleContext', type('OldNative', (), {}))
    assert CombatRuleStrategy().version == 'ironclad-v3'
    with pytest.raises(RuntimeError, match='public_player_power_flags'):
        NativeTurnPlanner()


def test_unknown_flag_schema_is_rejected():
    class Invalid:
        def public_player_power_flags(self):
            return {'schema_version': 2, 'flags': {}}
    with pytest.raises(ValueError, match='schema'):
        public_planner_state(Invalid())
