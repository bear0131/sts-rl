"""BRIDGE29: combat upgrades preserve the game's current cost adjustments."""
import pytest
import slaythespire as sts

from tests.bridge_fixtures import native_card, native_card_state

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def test_natural_damage_then_armaments_does_not_restore_blood_for_blood_to_three(combat_factory):
    env = combat_factory(['BLOODLETTING'] * 3 + ['ARMAMENTS', 'BLOOD_FOR_BLOOD'], 'Cultist')
    battle = env.thick.bc
    for _ in range(3):
        index = next(i for i, row in enumerate(battle.to_dict()['hand']) if row[0] == 'BLOODLETTING')
        battle.execute(sts.Action(sts.ActionType.CARD, index))
    before = next(row for row in battle.to_dict()['hand'] if row[0] == 'BLOOD_FOR_BLOOD')
    assert before[2] == 1, 'three real HP-loss actions must reduce the original four-cost card'
    armaments = next(i for i, row in enumerate(battle.to_dict()['hand']) if row[0] == 'ARMAMENTS')
    battle.execute(sts.Action(sts.ActionType.CARD, armaments))
    assert battle.to_dict()['hand'] == [('BLOOD_FOR_BLOOD', 1, 0)]
    enemy_hp = battle.to_dict()['monsters'][0]['hp']
    energy = battle.to_dict()['player']['energy']
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == enemy_hp - 22
    assert battle.to_dict()['player']['energy'] == energy


# Expectations from the installed game's upgrade methods and upgradeBaseCost,
# including BloodForBlood's independent clamp of its new base cost.
@pytest.mark.parametrize('card_id,base,current,expected_base,expected_current', [
    ('BLOOD_FOR_BLOOD', 4, 4, 3, 3),
    ('BLOOD_FOR_BLOOD', 4, 0, 3, 0),
    ('BLOOD_FOR_BLOOD', 3, 3, 2, 2),
    ('BLOOD_FOR_BLOOD', 2, 1, 1, 0),
    ('BLOOD_FOR_BLOOD', 1, 1, 0, 0),
    ('BLOOD_FOR_BLOOD', 0, 0, 0, 0),
    ('BLOOD_FOR_BLOOD', 0, 2, 0, 1),
    ('BARRICADE', 3, 0, 2, 0),
    ('BARRICADE', 3, 1, 2, 0),
    ('BARRICADE', 3, 5, 2, 4),
    ('BARRICADE', 0, 0, 2, 0),
    ('BODY_SLAM', 1, 1, 0, 0),
    ('BASH', 0, 0, 0, 0),
])
@pytest.mark.parametrize('pile', ['hand', 'draw', 'discard', 'exhaust'])
def test_apotheosis_preserves_current_cost_adjustments_in_every_pile(
        combat_factory, card_id, base, current, expected_base, expected_current, pile):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    state = native_card_state(hand=[native_card('APOTHEOSIS', 0, base_cost=2, cost=2)])
    state[pile].append(native_card(card_id, 4, base_cost=base, cost=current, free=True))
    env.thick.bc.set_card_state(state)
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    actual = next(card for card in env.thick.bc.card_state()[pile] if card['uid'] == 4)
    assert (actual['base_cost'], actual['cost']) == (expected_base, expected_current)
    assert actual['upgrades'] == 1
    assert actual['free_to_play_once'] is True


@pytest.mark.parametrize('upgrades,base,cost', [(0, 1, 1), (0, 0, 0), (1, 0, 0)])
def test_armaments_upgrade_keeps_blood_for_blood_playable_at_zero_energy(
        combat_factory, upgrades, base, cost):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    env.thick.bc.set_card_state(native_card_state(hand=[
        native_card('ARMAMENTS', 0, upgrades=1),
        native_card('BLOOD_FOR_BLOOD', 4, upgrades=upgrades, base_cost=base, cost=cost)]))
    env.thick.bc.apply_custom_state({'player': {'energy': 1},
                                   'monsters': [{'hp': 100, 'max_hp': 100, 'block': 0}]})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    assert env.thick.bc.to_dict()['player']['energy'] == 0
    assert env.thick.bc.legal_combat_mask()[0], 'upgraded Blood for Blood should still be playable'
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert env.thick.bc.to_dict()['monsters'][0]['hp'] == 78


def test_second_upgrade_keeps_modified_cost_and_does_not_upgrade_twice(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    env.thick.bc.set_card_state(native_card_state(hand=[
        native_card('ARMAMENTS', 0, upgrades=1),
        native_card('BARRICADE', 4, upgrades=1, base_cost=0, cost=0, free=True)]))
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    card = env.thick.bc.card_state()['hand'][0]
    assert (card['upgrades'], card['base_cost'], card['cost'], card['free_to_play_once']) == (1, 0, 0, True)
