"""DECK03: native master-deck restoration preserves persistent values and slots."""
from copy import deepcopy

import pytest
import slaythespire as sts

from tests.bridge_fixtures import native_master_card as card

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def test_grown_ritual_dagger_snapshot_controls_actual_damage(combat_factory):
    gc = combat_factory(['STRIKE_RED'] * 5, 'Cultist').thick.gc
    state = [card('RITUAL_DAGGER', misc=39, bottle='ATTACK')]
    gc.set_deck_state(state)
    assert gc.deck_state() == state
    battle = sts.create_battle(gc, 'Cultist')
    battle.apply_custom_state({'monsters': [{'hp': 50, 'block': 0}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == 11


@pytest.mark.parametrize('upgrades,damage', [(0, 12), (1, 16), (3, 27)])
def test_searing_snapshot_preserves_upgrade_count_and_actual_damage(combat_factory, upgrades, damage):
    gc = combat_factory(['STRIKE_RED'] * 5, 'Cultist').thick.gc
    state = [card('SEARING_BLOW', upgrades=upgrades, misc=upgrades)]
    gc.set_deck_state(state)
    assert gc.deck_state() == state
    assert gc.deck_ids() == [('SEARING_BLOW', upgrades)]
    battle = sts.create_battle(gc, 'Cultist')
    before = battle.to_dict()['monsters'][0]['hp']
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == before - damage


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5, 6, 7, 8])
def test_physical_bottle_slots_guarantee_exact_instances_in_opening_hand(seed):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    # Duplicates make checking only card names insufficient.
    state = [card('BASH'), card('BATTLE_TRANCE'), card('DEMON_FORM')]
    state += [card('STRIKE_RED') for _ in range(22)]
    state += [card('BASH', upgrades=1, bottle='ATTACK'),
              card('BATTLE_TRANCE', upgrades=1, bottle='SKILL'),
              card('DEMON_FORM', upgrades=1, bottle='POWER')]
    gc.set_deck_state(state)
    assert gc.deck_state() == state
    hand = sts.create_battle(gc, 'Cultist').card_state()['hand']
    assert {25, 26, 27} <= {c['uid'] for c in hand}
    assert {c['id'] for c in hand if c['uid'] in (25, 26, 27)} == {'BASH', 'BATTLE_TRANCE', 'DEMON_FORM'}


@pytest.mark.parametrize('field,value', [
    ('id', 'UNKNOWN'), ('id', 'INVALID'), ('id', 'BURN'), ('id', None),
    ('upgrades', True), ('upgrades', -1), ('upgrades', 2), ('upgrades', 32768),
    ('misc', True), ('misc', -1), ('misc', 32768), ('misc', '0'),
    ('bottle', 'SKILL'), ('bottle', 'PAD'), ('bottle', None), ('unmapped', 1),
])
def test_invalid_record_rejects_entire_replacement(combat_factory, field, value):
    gc = combat_factory(['STRIKE_RED'] * 5).thick.gc
    previous = [card('RITUAL_DAGGER', misc=39, bottle='ATTACK'), card('DEFEND_RED')]
    gc.set_deck_state(previous)
    bad = card('BASH')
    bad[field] = value
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        gc.set_deck_state([card('ANGER'), bad])
    assert gc.deck_state() == previous


@pytest.mark.parametrize('problem', ['missing', 'duplicate_bottle', 'searing_misc', 'nonrecord', 'capacity'])
def test_invalid_master_snapshot_preserves_original_state(combat_factory, problem):
    gc = combat_factory(['BASH', 'DEFEND_RED']).thick.gc
    before = gc.deck_state()
    state = [card('BASH')]
    if problem == 'missing':
        del state[0]['misc']
    elif problem == 'duplicate_bottle':
        state = [card('BASH', bottle='ATTACK')] * 2
    elif problem == 'searing_misc':
        state = [card('SEARING_BLOW', upgrades=3)]
    elif problem == 'nonrecord':
        state = [('BASH', 0)]
    else:
        state *= 97
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        gc.set_deck_state(state)
    assert gc.deck_state() == before


@pytest.mark.parametrize('records', [
    [('ANGER', 0), ('UNKNOWN', 0)], [('BASH', -1)], [('BASH', True)],
    [('BASH', 2)], [('SEARING_BLOW', 32768)], [('BURN', 0)], [('INVALID', 0)],
    [('BASH', 0, 1)], ['BASH'], [('STRIKE_RED', 0)] * 97,
])
def test_invalid_fresh_deck_preserves_values_and_bottles(combat_factory, records):
    gc = combat_factory(['STRIKE_RED'] * 5).thick.gc
    before = [card('RITUAL_DAGGER', misc=39, bottle='ATTACK')]
    gc.set_deck_state(before)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        gc.set_deck(records)
    assert gc.deck_state() == before


@pytest.mark.parametrize('size', [0, 96])
def test_master_snapshot_capacity_boundaries_and_independent_export(combat_factory, size):
    gc = combat_factory(['BASH']).thick.gc
    state = [card('DEFEND_RED') for _ in range(size)]
    gc.set_deck_state(state)
    assert gc.deck_state() == state
    if size:
        exported = gc.deck_state()
        exported[0]['misc'] = 123
        assert gc.deck_state() == state
    gc.set_deck([('STRIKE_RED', 0)] * size)
    assert gc.deck_state() == [card('STRIKE_RED') for _ in range(size)]


@pytest.mark.parametrize('container', [list, tuple])
def test_fresh_deck_keeps_sequence_pair_compatibility(combat_factory, container):
    gc = combat_factory(['STRIKE_RED']).thick.gc
    gc.set_deck(container([container(['RITUAL_DAGGER', 1]), container(['SEARING_BLOW', 3])]))
    assert gc.deck_state() == [card('RITUAL_DAGGER', upgrades=1, misc=15),
                               card('SEARING_BLOW', upgrades=3, misc=3)]


def test_restoring_deck_does_not_reapply_on_obtain_effects_or_consume_card_rng():
    restored = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    control = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    for gc in (restored, control):
        gc.set_relics(['Molten Egg 2', 'Darkstone Periapt', 'CeramicFish'])
    before = (restored.cur_hp, restored.max_hp, restored.gold)
    state = [card('BASH'), card('PARASITE')]
    restored.set_deck_state(state)
    assert restored.deck_state() == state
    assert (restored.cur_hp, restored.max_hp, restored.gold) == before
    assert restored._dbg_card_reward() == control._dbg_card_reward()


@pytest.mark.parametrize('upgrade,growth', [(0, 3), (1, 5)])
def test_restored_growth_commits_to_correct_duplicate_master_card(short_combat_env_type, upgrade, growth):
    env = short_combat_env_type()
    try:
        env.reset(seed=901)
        gc = env.thick.gc
        state = [card('RITUAL_DAGGER', misc=15),
                 card('RITUAL_DAGGER', upgrades=upgrade, misc=39, bottle='ATTACK')]
        gc.set_deck_state(state)
        battle = sts.create_battle(gc, 'Cultist')
        hand = battle.card_state()['hand']
        index = next(i for i, c in enumerate(hand) if c['uid'] == 1)
        battle.apply_custom_state({'monsters': [{'hp': 1, 'block': 0}]})
        battle.execute(sts.Action(sts.ActionType.CARD, index, 0))
        battle.exit_battle(gc)
        expected = deepcopy(state)
        expected[1]['misc'] += growth
        assert gc.deck_state() == expected
    finally:
        env.close()


def test_restored_counts_and_bottles_control_actual_designer_choices(designer_factory):
    gc = designer_factory(0, 3)
    state = [card('BASH', bottle='ATTACK'), card('DEFEND_RED'), card('ASCENDERS_BANE')]
    gc.set_deck_state(state)
    assert 3 not in gc.legal_event_options(), 'only one unbottled transformable card remains'
    state[0]['bottle'] = 'NONE'
    gc.set_deck_state(state)
    assert 3 in gc.legal_event_options()
    gc.choose_event_option(3)
    assert gc.get_select_cards() == [('BASH', 0), ('DEFEND_RED', 0)]
