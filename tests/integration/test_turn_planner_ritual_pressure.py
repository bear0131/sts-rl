"""Public Ritual creates a root-anchored urgency to finish the scaling enemy."""
import json

import pytest
import slaythespire as sts

import env.search.turn_planner as planner_module
from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig, scaling_damage_premiums

pytestmark = pytest.mark.native
END = 4 << 29


def _cultist(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, encounter='Cultist', relics=['Pocketwatch'])
    env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 20)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 5 + [('DEFEND_RED', 0)] * 4
                         + [('BASH', 0), ('ASCENDERS_BANE', 0)])
    env.thick.gc.set_relics(['Pocketwatch'])
    env.thick.gc.cur_hp = 68
    env.thick.bc = sts.create_battle(env.thick.gc, 'Cultist')
    return env.thick.bc


def _planner():
    return NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=2048))


def _rollout(battle):
    planner = _planner()
    for step in range(120):
        result = planner.choose(battle, state_nonce=step, public_relics=('Pocketwatch',))
        battle.execute(sts.Action(result.action_bits))
        if battle.is_battle_over:
            return battle.to_dict()
    pytest.fail('bounded Cultist regression did not terminate within120 decisions')


def test_only_visible_ritual_activates_pressure_and_existing_larger_weight_wins(combat_factory):
    battle = _cultist(combat_factory)
    assert scaling_damage_premiums(battle.to_dict()) == {}
    battle.execute(sts.Action(END))
    raw = battle.to_dict()
    assert dict(raw['monsters'][0]['powers'])['Ritual'] == 5
    premiums = scaling_damage_premiums(raw)
    assert list(premiums) == [(0, 'CULTIST')]
    assert premiums[(0, 'CULTIST')] > 0
    assert scaling_damage_premiums(raw, base_damage_weight=2.) == {}


def test_every_scored_leaf_uses_same_public_root_pressure(combat_factory, monkeypatch):
    battle = _cultist(combat_factory)
    battle.execute(sts.Action(END))
    root = battle.to_dict()
    expected = scaling_damage_premiums(root)
    seen = []
    original = planner_module.state_value
    def capture(raw, **kwargs):
        seen.append((raw, kwargs.get('damage_premiums')))
        return original(raw, **kwargs)
    monkeypatch.setattr(planner_module, 'state_value', capture)
    result = _planner().choose(battle, state_nonce=9, public_relics=('Pocketwatch',))
    assert seen and all(premium == expected for _, premium in seen)
    assert any(raw['monsters'][0]['hp'] < root['monsters'][0]['hp'] for raw, _ in seen)
    assert battle.to_dict() == root
    json.dumps(result.diagnostics, allow_nan=False)


def test_pocketwatch_starter_deck_finishes_cultist_before_runaway_scaling(combat_factory, monkeypatch):
    battle = _cultist(combat_factory)
    baseline = battle.clone()
    original = planner_module.scaling_damage_premiums
    monkeypatch.setattr(planner_module, 'scaling_damage_premiums', lambda raw, *args, **kwargs: {})
    before = _rollout(baseline)
    monkeypatch.setattr(planner_module, 'scaling_damage_premiums', original)
    after = _rollout(battle)
    assert before['outcome'] == 2  # reproduced deferred-damage failure
    assert after['outcome'] == 1 and after['player']['hp'] > 0
    assert after['turn'] < before['turn']


def test_urgency_never_overrides_immediate_survival(combat_factory):
    battle = _cultist(combat_factory)
    battle.execute(sts.Action(END))
    battle.apply_custom_state({
        'player': {'hp': 1, 'energy': 3},
        'hand': ['STRIKE_RED', 'STRIKE_RED', 'DEFEND_RED', 'DEFEND_RED', 'DEFEND_RED'],
        'monsters': [{'hp': 40, 'powers': [('STRENGTH', 9)]}],
    })
    result = _planner().choose(battle, state_nonce=9, public_relics=('Pocketwatch',))
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes['0']['deaths'] == outcomes['1']['deaths'] == 2
    assert outcomes[str(result.action_bits)]['deaths'] == 0
    assert result.action_bits in (2, 3, 4)
