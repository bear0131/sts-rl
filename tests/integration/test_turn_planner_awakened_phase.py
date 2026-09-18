"""Awakened phase progress uses existing public powers and half-dead state."""
import pytest
import slaythespire as sts

from env.search.turn_planner import (
    NativeTurnPlanner, TurnPlannerConfig, enemy_burden, leaf_value, state_value,
)

pytestmark = pytest.mark.native
END = 4 << 29


def _awakened(combat_factory, ascension=0):
    env = combat_factory(['STRIKE_RED'] * 10, encounter='Awakened One')
    if ascension:
        env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, ascension)
        env.thick.gc.set_deck([('STRIKE_RED', 0)] * 10)
        env.thick.bc = sts.create_battle(env.thick.gc, 'Awakened One')
    battle = env.thick.bc
    battle.apply_custom_state({
        'player': {'hp': 80, 'max_hp': 80, 'energy': 3, 'block': 0},
        'hand': ['STRIKE_RED'], 'draw': ['DEFEND_RED'] * 5, 'discard': [], 'exhaust': [],
        'monsters': [{'hp': 0}, {'hp': 0}, {'hp': 6, 'block': 0}],
    })
    return battle


@pytest.mark.parametrize('ascension,second_hp', [(0, 300), (9, 320), (20, 320)])
def test_real_rebirth_preserves_remaining_life_and_never_looks_terminal(
        combat_factory, ascension, second_hp):
    battle = _awakened(combat_factory, ascension)
    before = battle.to_dict()
    assert dict(before['monsters'][2]['powers'])['Curiosity'] > 0
    visible_max = before['monsters'][2]['max_hp']
    assert enemy_burden(before['monsters']) == (6 + visible_max, 1)
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 2))
    pending = battle.to_dict()
    boss = pending['monsters'][2]
    assert boss['hp'] == 0 and boss['half_dead'] and boss['gone']
    assert boss['move'] == 'AWAKENED_ONE_REBIRTH'
    assert 'Curiosity' not in dict(boss['powers'])
    assert pending['outcome'] == 0 and leaf_value(pending)[0] == 0
    assert enemy_burden(pending['monsters']) == (visible_max, 1)
    # Death rank stays live, and zero visible HP still represents one real life.
    assert state_value(pending, horizon=3) < state_value(
        {**pending, 'monsters': []}, horizon=3) - .26 * visible_max
    # Removing the last six first-phase HP is progress; no HP/debuff loss here.
    # Pending boss Strength is temporarily ignored by the ordinary threat loop;
    # high ascension can therefore gain more than the HP contribution alone.
    assert state_value(pending, horizon=3) >= state_value(before, horizon=3) + .26 * 6 - 1e-9
    after = battle.execute_to_enemy_boundary(END)
    assert after['outcome'] == 0
    assert enemy_burden(after['monsters']) == (second_hp, 1)
    # No ascension parameter is added: high native ascensions can reveal
    # up to20 additional HP at rebirth versus the public first-life max.
    assert 0 <= second_hp - visible_max <= 20
    assert state_value(after, horizon=3) >= state_value(before, horizon=3) - .26 * 20
    boss = after['monsters'][2]
    assert boss['hp'] == second_hp and not boss['half_dead']
    assert dict(boss['powers'])['Minion Leader'] == 1
    assert 'Curiosity' not in dict(boss['powers'])
    if ascension == 0:
        assert state_value(after, horizon=3) == pytest.approx(state_value(pending, horizon=3))


def test_second_life_death_does_not_invent_another_rebirth(combat_factory):
    battle = _awakened(combat_factory)
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 2))
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    battle.apply_custom_state({'hand': ['STRIKE_RED'], 'monsters': [{}, {}, {'hp': 6}]})
    before = battle.to_dict()
    assert dict(before['monsters'][2]['powers'])['Minion Leader'] == 1
    assert enemy_burden(before['monsters']) == (6, 1)
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 2))
    after = battle.to_dict()
    assert after['outcome'] == 1
    assert not after['monsters'][2]['half_dead']
    assert enemy_burden(after['monsters']) == (0, 0)
    assert leaf_value(after)[0] == 1


def test_reduced_r6_awakened_state_starts_second_phase_without_delaying(combat_factory):
    # R6 seed150023/floor50/turn4 line15545:70HP,Awakened3HP with long debuffs,
    # END295.94 vs phase-switch218.32. Reduced mechanism, not a private replay:
    # no Snecko/relics and a fixed public five-Strike next-draw multiset.
    battle = _awakened(combat_factory)
    battle.apply_custom_state({
        'player': {'hp': 70, 'energy': 2, 'block': 33,
                   'powers': [('STRENGTH', 10), ('FEEL_NO_PAIN', 4)]},
        'monsters': [{}, {}, {'hp': 3, 'move': 'AWAKENED_ONE_SLASH',
                             'powers': [('VULNERABLE', 13), ('WEAK', 18)]}],
        'hand': [('STRIKE_RED', 1), ('CLOTHESLINE', 1)],
        'draw': ['STRIKE_RED'] * 5, 'discard': [], 'exhaust': [],
    })
    before = battle.to_dict()
    result = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=2048,
        lookahead_turns=1)).choose(battle, state_nonce=20260918, public_relics=())
    assert result.action_bits != END
    assert result.diagnostics['fallback_leaves'] == 0
    assert battle.to_dict() == before
    battle.execute(sts.Action(result.action_bits))
    assert battle.to_dict()['monsters'][2]['half_dead']
    assert battle.to_dict()['outcome'] == 0
