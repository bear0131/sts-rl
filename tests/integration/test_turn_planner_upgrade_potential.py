"""Permanent upgrades survive the one-turn value boundary without leaking order."""
from copy import deepcopy

import pytest
import slaythespire as sts

from env.search.turn_planner import (
    NativeTurnPlanner, TurnPlannerConfig, power_horizon, upgrade_potential,
    upgrade_reference,
)

pytestmark = pytest.mark.native
END = 4 << 29


def _battle(combat_factory, *, encounter='Jaw Worm', relics=()):
    env = combat_factory(['STRIKE_RED'] * 5, encounter=encounter, relics=relics)
    env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 20)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 5)
    env.thick.gc.set_relics(list(relics))
    env.thick.bc = sts.create_battle(env.thick.gc, encounter)
    return env.thick.bc


def _planner():
    return NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512))


def _apotheosis(battle, upgraded=0):
    battle.apply_custom_state({
        'player': {'energy': 1}, 'hand': [('APOTHEOSIS', 1)],
        'draw': [('STRIKE_RED', upgraded)] * 5, 'discard': [], 'exhaust': [],
        'monsters': [{'move': 'JAW_WORM_BELLOW'}],
    })


def test_apotheosis_upgrades_future_draw_instead_of_discarding_it(combat_factory):
    battle = _battle(combat_factory)
    _apotheosis(battle)
    before = battle.to_dict()
    result = _planner().choose(battle, state_nonce=1, public_relics=())
    assert result.action_bits == 0
    assert result.diagnostics['root_values']['0'] > result.diagnostics['root_values'][str(END)]
    assert battle.to_dict() == before
    battle.execute(sts.Action(0))
    assert battle.to_dict()['draw'] == [('STRIKE_RED', 1)] * 5
    assert battle.to_dict()['exhaust'] == [('APOTHEOSIS', 1)]


def test_already_upgraded_deck_does_not_reward_useless_apotheosis(combat_factory):
    battle = _battle(combat_factory)
    _apotheosis(battle, upgraded=1)
    result = _planner().choose(battle, state_nonce=1, public_relics=())
    assert result.action_bits == END
    assert result.diagnostics['root_values']['0'] == result.diagnostics['root_values'][str(END)]


def test_armaments_retains_upgrade_value_after_unused_block_expires(combat_factory):
    battle = _battle(combat_factory)
    battle.apply_custom_state({
        'player': {'energy': 1}, 'hand': [('ARMAMENTS', 1), 'BASH'],
        'draw': ['STRIKE_RED'] * 5, 'discard': [], 'exhaust': [],
        'monsters': [{'move': 'JAW_WORM_BELLOW'}],
    })
    result = _planner().choose(battle, state_nonce=1, public_relics=())
    assert result.action_bits == 0
    upgraded = battle.clone()
    upgraded.execute(sts.Action(0))
    leaf = upgraded.execute_to_enemy_boundary(END)
    assert leaf['player']['block'] == 0
    assert ('BASH', 1) in leaf['discard']
    assert result.diagnostics['root_values']['0'] > result.diagnostics['root_values'][str(END)]


def test_multiset_potential_ignores_exhaust_only_and_draw_order(combat_factory):
    battle = _battle(combat_factory)
    _apotheosis(battle)
    before = battle.to_dict()
    reference = upgrade_reference(before, power_horizon(before))
    after = deepcopy(before)
    # Exhausting a plain card cannot be mistaken for upgrading it.
    after['exhaust'].append(after['draw'].pop())
    assert upgrade_potential(after, reference) == 0
    # Upgrading an exhausted card has no future-play payoff by itself.
    after['exhaust'][-1] = ('STRIKE_RED', 1)
    assert upgrade_potential(after, reference) == 0
    # One newly upgraded reusable copy counts once, independent of pile order.
    after['draw'][0] = ('STRIKE_RED', 1)
    one = upgrade_potential(after, reference)
    assert one > 0
    after['draw'].reverse()
    assert upgrade_potential(after, reference) == one


@pytest.mark.parametrize('encounter,relics,plays,move', [
    ('Jaw Worm', ('Velvet Choker',), 5, 'JAW_WORM_CHOMP'),
    ('Time Eater', (), 11, 'TIME_EATER_REVERBERATE'),
])
def test_upgrade_does_not_consume_last_safe_play(combat_factory, encounter, relics, plays, move):
    battle = _battle(combat_factory, encounter=encounter, relics=relics)
    # Apply in batches so the eleven-card Time Warp setup fits the ten-card hand.
    for count in (min(plays, 10), max(0, plays - 10)):
        battle.apply_custom_state({'hand': ['FLEX'] * count})
        for _ in range(count):
            battle.execute(sts.Action(0))
    before = battle.to_dict()
    if encounter == 'Time Eater':
        assert ('Time Warp', 11) in before['monsters'][0]['powers']
    else:
        assert before['player']['combat_counters']['cards_played'] == 5
    battle.apply_custom_state({
        'player': {'hp': 1, 'energy': 1, 'powers': [('DEXTERITY', 40)]},
        'hand': [('APOTHEOSIS', 1), 'DEFEND_RED'],
        'draw': ['STRIKE_RED'] * 15, 'discard': [], 'exhaust': [],
        'monsters': [{'move': move}],
    })
    result = _planner().choose(battle, state_nonce=2, public_relics=relics)
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes['0']['deaths'] == 2
    assert outcomes['1']['deaths'] == 0
    assert result.action_bits == 1


def test_immediate_win_stays_above_future_upgrades(combat_factory):
    battle = _battle(combat_factory, relics=('Velvet Choker',))
    battle.apply_custom_state({'hand': ['FLEX'] * 5})
    for _ in range(5):
        battle.execute(sts.Action(0))
    battle.apply_custom_state({
        'player': {'energy': 1}, 'hand': [('APOTHEOSIS', 1), 'STRIKE_RED'],
        'draw': ['STRIKE_RED'] * 20, 'discard': [], 'exhaust': [],
        'monsters': [{'hp': 1, 'move': 'JAW_WORM_BELLOW'}],
    })
    result = _planner().choose(battle, state_nonce=2, public_relics=('Velvet Choker',))
    assert result.diagnostics['root_outcomes']['1']['wins'] == 2
    assert result.action_bits == 1


def test_second_apotheosis_does_not_count_the_same_upgrades_twice(combat_factory):
    battle = _battle(combat_factory)
    _apotheosis(battle)
    battle.apply_custom_state({'player': {'energy': 2},
                               'hand': [('APOTHEOSIS', 1)] * 2})
    before = battle.to_dict()
    reference = upgrade_reference(before, power_horizon(before))
    battle.execute(sts.Action(0))
    first = upgrade_potential(battle.to_dict(), reference)
    assert first > 0
    battle.execute(sts.Action(0))
    assert upgrade_potential(battle.to_dict(), reference) == first


@pytest.mark.parametrize('relics,expected', [((), True), (('Snecko Eye',), False)])
def test_cost_only_upgrade_has_no_future_value_under_snecko(combat_factory, relics, expected):
    battle = _battle(combat_factory, relics=relics)
    _apotheosis(battle)
    battle.apply_custom_state({'draw': ['CORRUPTION', 'BODY_SLAM', 'DARK_EMBRACE']})
    before = battle.to_dict()
    reference = upgrade_reference(before, power_horizon(before), relics)
    battle.execute(sts.Action(0))
    assert (upgrade_potential(battle.to_dict(), reference) > 0) == expected
