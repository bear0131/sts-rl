"""SIM01: actual power uses retaliate only while Curiosity is active."""
import numpy as np
import pytest
import slaythespire as sts

pytestmark = pytest.mark.native


def _battle(combat_factory, ascension=0, relics=()):
    env = combat_factory(['INFLAME'] * 10, encounter='Awakened One', relics=relics)
    if ascension:
        env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, ascension)
        env.thick.gc.set_deck([('INFLAME', 0)] * 10)
        env.thick.gc.set_relics(list(relics))
        env.thick.bc = sts.create_battle(env.thick.gc, 'Awakened One')
    battle = env.thick.bc
    battle.apply_custom_state({
        'player': {'hp': 80, 'max_hp': 80, 'energy': 3, 'block': 0},
        'hand': ['INFLAME'], 'draw': ['DEFEND_RED'] * 5,
        'discard': [], 'exhaust': [],
    })
    return battle


def _strength(battle, slot=2):
    return dict(battle.to_dict()['monsters'][slot]['powers']).get('Strength', 0)


@pytest.mark.parametrize('ascension,gain', [(0, 1), (20, 2)])
@pytest.mark.parametrize('card,player_strength', [
    ('INFLAME', 2), (('INFLAME', 1), 3), ('DEMON_FORM', 0),
])
@pytest.mark.parametrize('clone', [False, True])
def test_power_use_triggers_curiosity(combat_factory, ascension, gain,
                                     card, player_strength, clone):
    source = _battle(combat_factory, ascension)
    source.apply_custom_state({'hand': [card]})
    before = source.to_dict()
    battle = source.clone() if clone else source
    strength = _strength(battle)
    assert dict(before['monsters'][2]['powers'])['Curiosity'] == gain
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert _strength(battle) == strength + gain
    assert dict(battle.to_dict()['player']['powers']).get('Strength', 0) == player_strength
    if clone:
        assert source.to_dict() == before


@pytest.mark.parametrize('ascension,gain', [(0, 1), (20, 2)])
def test_free_power_still_triggers(combat_factory, ascension, gain):
    battle = _battle(combat_factory, ascension)
    battle.apply_custom_state({'hand': ['PANACHE'], 'player': {'energy': 0}})
    before = _strength(battle)
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert battle.to_dict()['player']['energy'] == 0
    assert dict(battle.to_dict()['player']['powers'])['Panache'] == 10
    assert _strength(battle) == before + gain


@pytest.mark.parametrize('ascension,gain', [(0, 1), (20, 2)])
def test_duplicated_power_triggers_twice(combat_factory, ascension, gain):
    battle = _battle(combat_factory, ascension)
    battle.apply_custom_state({'potions': ['Duplication Potion']})
    battle.execute(sts.Action(sts.ActionType.POTION, 0))
    before = _strength(battle)
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert dict(battle.to_dict()['player']['powers'])['Strength'] == 4
    assert _strength(battle) == before + 2 * gain


@pytest.mark.parametrize('card', ['STRIKE_RED', 'DEFEND_RED', 'RAGE'])
def test_non_power_does_not_trigger(combat_factory, card):
    battle = _battle(combat_factory)
    battle.apply_custom_state({'hand': [card]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 2))
    assert _strength(battle) == 0


@pytest.mark.parametrize('ascension', [0, 20])
def test_first_death_and_second_life_disable_curiosity(combat_factory, ascension):
    battle = _battle(combat_factory, ascension)
    battle.apply_custom_state({'hand': ['STRIKE_RED', 'INFLAME'],
                              'monsters': [{'hp': 0}, {'hp': 0}, {'hp': 6}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 2))
    boss = battle.to_dict()['monsters'][2]
    assert boss['half_dead'] and 'Curiosity' not in dict(boss['powers'])
    before = _strength(battle)
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert _strength(battle) == before
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert not battle.to_dict()['monsters'][2]['half_dead']
    battle.apply_custom_state({'hand': ['INFLAME']})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert _strength(battle) == before


def test_dead_curiosity_holder_does_not_trigger(combat_factory):
    battle = _battle(combat_factory)
    battle.apply_custom_state({'monsters': [{}, {}, {'hp': 0}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert [_strength(battle, slot) for slot in range(3)] == [0, 0, 0]


def test_search_leaf_contains_retaliation_without_mutating_root(combat_factory):
    battle = _battle(combat_factory)
    before = battle.to_dict()
    logits = np.full(181, -100.0)
    logits[1] = 100.0
    session = battle.nosl_neural_search_begin(
        logits, simulations=1, depth=1, nonce=42, root_search_mode=2,
        paper_full_gumbel=True, gumbel_scale=0, fixed_worldline=True)
    leaf = session.next_batch(1)[0]
    assert dict(leaf['state']['monsters'][2]['powers'])['Strength'] == 1
    assert battle.to_dict() == before
