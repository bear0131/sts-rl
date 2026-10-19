"""BRIDGE28: persistent card values and physical bottle slots are not fresh IDs."""
import pytest
import slaythespire as sts

from tests.bridge_fixtures import real_card
from bridge.schemas.card_state import real_master_deck_state
from bridge.schemas.combat_state import real_to_sim_card_enum

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _native(records):
    return [dict(card, id=real_to_sim_card_enum(card['id'])) for card in real_master_deck_state(records)]


def test_current_ritual_growth_survives_real_to_master_snapshot_and_actual_damage(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    snapshot = _native([real_card('RitualDagger', misc=39, bottle='ATTACK', damage=39, exhaust=True)])
    env.thick.gc.set_deck_state(snapshot)
    assert env.thick.gc.deck_state() == snapshot
    battle = sts.create_battle(env.thick.gc, 'Cultist')
    battle.apply_custom_state({'monsters': [{'hp': 50, 'block': 0}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == 11, 'current39 must not be reset to fresh15'


def test_searing_java_misc_zero_translates_to_native_full_upgrade_count(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    state = _native([real_card('Searing Blow', upgrades=3, misc=0, bottle='ATTACK', base_cost=2)])
    assert state[0]['misc'] == 3
    env.thick.gc.set_deck_state(state)
    assert env.thick.gc.deck_state() == state
    assert env.thick.gc.deck_ids() == [('SEARING_BLOW', 3)]


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5, 6, 7, 8])
def test_all_three_physical_bottle_kinds_are_guaranteed_in_opening_hand(seed):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    state = [dict(id='STRIKE_RED', upgrades=0, misc=0, bottle='NONE') for _ in range(25)]
    state += [dict(id='BASH', upgrades=0, misc=0, bottle='ATTACK'),
              dict(id='BATTLE_TRANCE', upgrades=0, misc=0, bottle='SKILL'),
              dict(id='DEMON_FORM', upgrades=0, misc=0, bottle='POWER')]
    gc.set_deck_state(state)
    assert gc.deck_state() == state
    hand = sts.create_battle(gc, 'Cultist').to_dict()['hand']
    assert {'BASH', 'BATTLE_TRANCE', 'DEMON_FORM'} <= {card[0] for card in hand}


def test_fresh_deck_replacement_clears_old_bottled_indices(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    env.thick.gc.set_deck_state([dict(id='BASH', upgrades=0, misc=0, bottle='ATTACK')])
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 3)
    assert [card['bottle'] for card in env.thick.gc.deck_state()] == ['NONE'] * 3


@pytest.mark.parametrize('bad', [
    {'id': 'RITUAL_DAGGER', 'upgrades': 0, 'bottle': 'NONE'},
    {'id': 'RITUAL_DAGGER', 'upgrades': 0, 'misc': True, 'bottle': 'NONE'},
    {'id': 'RITUAL_DAGGER', 'upgrades': 0, 'misc': -1, 'bottle': 'NONE'},
    {'id': 'RITUAL_DAGGER', 'upgrades': 0, 'misc': 32768, 'bottle': 'NONE'},
    {'id': 'SEARING_BLOW', 'upgrades': 3, 'misc': 0, 'bottle': 'NONE'},
    {'id': 'BASH', 'upgrades': 0, 'misc': 0, 'bottle': 'SKILL'},
    {'id': 'BASH', 'upgrades': True, 'misc': 0, 'bottle': 'NONE'},
    {'id': 'BASH', 'upgrades': 2, 'misc': 0, 'bottle': 'NONE'},
    {'id': 'INVALID', 'upgrades': 0, 'misc': 0, 'bottle': 'NONE'},
    {'id': 'BASH', 'upgrades': 0, 'misc': 0, 'bottle': 'PAD'},
    {'id': 'BASH', 'upgrades': 0, 'misc': 0, 'bottle': 'NONE', 'lost_state': 42},
])
def test_invalid_master_snapshot_preserves_whole_previous_deck(combat_factory, bad):
    env = combat_factory(['STRIKE_RED'] * 5)
    previous = [dict(id='BASH', upgrades=0, misc=0, bottle='ATTACK')]
    env.thick.gc.set_deck_state(previous)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        env.thick.gc.set_deck_state([dict(id='STRIKE_RED', upgrades=0, misc=0, bottle='NONE'), bad])
    assert env.thick.gc.deck_state() == previous


def test_duplicate_bottle_slots_fail_atomically(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    before = env.thick.gc.deck_state()
    with pytest.raises(ValueError, match='multiple physical slots'):
        env.thick.gc.set_deck_state([dict(id='BASH', upgrades=0, misc=0, bottle='ATTACK')] * 2)
    assert env.thick.gc.deck_state() == before


@pytest.mark.parametrize('records', [[('BASH', 0), ('UNKNOWN', 0)], [('BASH', -1)], [('BASH', True)], [('STRIKE_RED', 0)] * 97])
def test_invalid_fresh_deck_replacement_is_atomic(combat_factory, records):
    env = combat_factory(['STRIKE_RED'] * 5)
    before = env.thick.gc.deck_state()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        env.thick.gc.set_deck(records)
    assert env.thick.gc.deck_state() == before
