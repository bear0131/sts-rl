"""Root-anchored payoff duration, from R2 public development counterexamples."""
import pytest
import slaythespire as sts

import env.search.turn_planner as planner_module
from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig, power_horizon, state_value


def _hexaghost(combat_factory, *, enemy_strength=2):
    # Reduced actual R2 seed150005/floor16/turn11 state. No favorable draw or RNG
    # is assumed: the comparison stops before the next player's draw.
    battle = combat_factory(['STRIKE_RED'] * 5, encounter='Hexaghost').thick.bc
    battle.apply_custom_state({
        'turn': 11,
        'player': {'hp': 19, 'max_hp': 80, 'block': 6, 'energy': 3,
                   'powers': [('STRENGTH', 2), ('METALLICIZE', 3), ('FEEL_NO_PAIN', 3)]},
        'hand': ['STRIKE_RED', 'DEFEND_RED'], 'draw': [], 'discard': [], 'exhaust': [],
        'monsters': [{'hp': 120, 'block': 0, 'move': 'HEXAGHOST_SEAR',
                      'powers': [('STRENGTH', max(0, enemy_strength)), ('VULNERABLE', 2)]}],
    })
    if enemy_strength < 0:
        # The public fixture setter clamps status amounts nonnegative. Obtain
        # negative Strength through the actual debuff, not a test-only field.
        battle.apply_custom_state({'player': {'energy': 4},
                                   'hand': [('DISARM', 1), 'STRIKE_RED', 'DEFEND_RED']})
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert dict(battle.to_dict()['monsters'][0]['powers']).get('Strength', 0) == enemy_strength
    return battle


@pytest.mark.parametrize('enemy_strength', [-3, 0, 2])
def test_native_nonlethal_damage_does_not_erase_power_value(combat_factory, enemy_strength):
    battle = _hexaghost(combat_factory, enemy_strength=enemy_strength)
    horizon = power_horizon(battle.to_dict())
    waiting = battle.clone().execute_to_enemy_boundary(4 << 29)
    attack = battle.clone()
    attack.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    damaged = attack.execute_to_enemy_boundary(4 << 29)
    assert waiting['player']['hp'] == damaged['player']['hp'] == 19
    assert waiting['monsters'][0]['hp'] - damaged['monsters'][0]['hp'] == 12
    # Native debuffs, generated Burn, and HP cost are equal. Keeping the same
    # horizon must reward this actual damage for either sign of enemy Strength.
    assert state_value(damaged, horizon=horizon) > state_value(waiting, horizon=horizon)
    if enemy_strength == 2:
        assert state_value(damaged) < state_value(waiting)  # Old failure isolated.


def test_hexaghost_planner_spends_available_energy_on_real_damage(combat_factory):
    battle = _hexaghost(combat_factory)
    planner = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512))
    selected = []
    for step in range(3):
        result = planner.choose(battle, state_nonce=step, public_relics=())
        action = sts.Action(result.action_bits)
        if action.action_type == sts.ActionType.CARD:
            selected.append(battle.to_dict()['hand'][action.source_idx][0])
        raw = battle.execute_to_enemy_boundary(result.action_bits)
        if raw['post_enemy_turn_boundary'] or raw['outcome']:
            break
    assert 'STRIKE_RED' in selected
    assert battle.to_dict()['monsters'][0]['hp'] == 108
    assert battle.to_dict()['player']['hp'] == 19


def test_every_scored_native_branch_uses_one_public_root_horizon(combat_factory, monkeypatch):
    battle = _hexaghost(combat_factory)
    battle.apply_custom_state({'hand': ['FLEX', 'STRIKE_RED', 'DEFEND_RED']})
    original = planner_module.state_value
    seen = []
    def capture(raw, **kwargs):
        seen.append((kwargs.get('horizon'), kwargs.get('partial', False),
                     raw['monsters'][0]['hp'], dict(raw['player']['powers']).get('Strength', 0)))
        return original(raw, **kwargs)
    monkeypatch.setattr(planner_module, 'state_value', capture)
    expected = power_horizon(battle.to_dict())
    NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=77, public_relics=())
    assert seen and {entry[0] for entry in seen} == {expected}
    assert {entry[1] for entry in seen} == {False, True}
    assert len({entry[2] for entry in seen}) > 1
    assert len({entry[3] for entry in seen}) > 1


def test_metallicize_and_disarm_do_not_reward_retaining_slime_hp(combat_factory):
    # Reduced150016/floor7: preserve its Metallicize/negative Strength engine,
    # while keeping this Strike above the split threshold to isolate horizon.
    battle = combat_factory(['STRIKE_RED'] * 5, encounter='Large Slime').thick.bc
    battle.apply_custom_state({
        'player': {'hp': 80, 'max_hp': 80, 'energy': 3, 'block': 0,
                   'powers': [('METALLICIZE', 11)]},
        'hand': [('DISARM', 1), 'STRIKE_RED', 'DEFEND_RED'], 'draw': [], 'discard': [], 'exhaust': [],
        'monsters': [{'hp': 50, 'block': 0, 'move': 'ACID_SLIME_L_TACKLE'}],
    })
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['id'] == 'ACID_SLIME_L'
    assert dict(battle.to_dict()['monsters'][0]['powers'])['Strength'] == -3
    horizon = power_horizon(battle.to_dict())
    waiting = battle.clone().execute_to_enemy_boundary(4 << 29)
    attack = battle.clone()
    attack.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    damaged = attack.execute_to_enemy_boundary(4 << 29)
    assert len(damaged['monsters']) == 1
    assert waiting['player']['hp'] == damaged['player']['hp']
    assert waiting['monsters'][0]['hp'] - damaged['monsters'][0]['hp'] == 6
    assert state_value(damaged) < state_value(waiting)
    assert state_value(damaged, horizon=horizon) > state_value(waiting, horizon=horizon)


def test_native_ordinary_victory_and_death_keep_terminal_order(combat_factory):
    battle = _hexaghost(combat_factory)
    battle.apply_custom_state({'monsters': [{'hp': 12}]})
    horizon = power_horizon(battle.to_dict())
    waiting = battle.clone().execute_to_enemy_boundary(4 << 29)
    win = battle.clone()
    win.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert int(win.outcome) == 1
    assert state_value(win.to_dict(), horizon=horizon) > state_value(waiting, horizon=horizon)
    doomed = _hexaghost(combat_factory, enemy_strength=30)
    dead = doomed.execute_to_enemy_boundary(4 << 29)
    assert dead['outcome'] == 2
    assert state_value(dead, horizon=horizon) < state_value(waiting, horizon=horizon)
