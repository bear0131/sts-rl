"""BRIDGE30: explicit per-card transport and stable Ironclad/colorless IDs."""
from copy import deepcopy

import pytest

from bridge.schemas.card_state import real_card_state, real_master_deck_state
from bridge.schemas.combat_state import _card, real_to_sim_card_enum, normalize_real_overworld
from bridge.real_game_env import RealGameAdapter
from env.core.cards_vocab import CARD_INDEX
from tests.bridge_fixtures import real_card

ALIASES = [
    ('Ghostly', 'Apparition', 'APPARITION'),
    ('Flash of Steel', 'Flash Of Steel', 'FLASH_OF_STEEL'),
    ('HandOfGreed', 'Hand Of Greed', 'HAND_OF_GREED'),
    ('J.A.X.', 'Jax', 'JAX'),
    ('Master of Strategy', 'Master Of Strategy', 'MASTER_OF_STRATEGY'),
    ('PanicButton', 'Panic Button', 'PANIC_BUTTON'),
    ('RitualDagger', 'Ritual Dagger', 'RITUAL_DAGGER'),
    ('Blood for Blood', 'Blood For Blood', 'BLOOD_FOR_BLOOD'),
    ('Strike_R', 'Strike_R', 'STRIKE_RED'),
    ('Defend_R', 'Defend_R', 'DEFEND_RED'),
]


@pytest.mark.parametrize('wire,canonical,native', ALIASES)
def test_wire_id_matches_frozen_token_and_native_enum(wire, canonical, native):
    assert _card(wire, 1) == dict(id=canonical, up=1)
    assert real_to_sim_card_enum(wire) == native
    adapter = RealGameAdapter()
    gs = dict(screen_type='REST', current_hp=70, max_hp=80, gold=99,
              relics=[], deck=[dict(id=wire, upgrades=1)], screen_state={})
    adapter.update(dict(in_game=True, available_commands=[], game_state=gs))
    assert adapter.observation()['deck_ids'][0] == CARD_INDEX[canonical]
    assert normalize_real_overworld(gs)['deck'] == [(canonical, 1)]


def test_complete_record_preserves_values_without_mutating_wire_input():
    source = real_card('Bash', base_cost=2, cost=0)
    source['simulator_card_state'].update(free_to_play_once=True, retain=True)
    before = deepcopy(source)
    parsed = real_card_state(source)
    assert (parsed['base_cost'], parsed['cost'], parsed['free_to_play_once'], parsed['retain']) == (2, 0, True, True)
    assert parsed['uuid'] == source['uuid']
    assert source == before


@pytest.mark.parametrize('field,value', [
    ('id', None), ('id', ''), ('upgrades', True), ('upgrades', -1), ('upgrades', 32768),
    ('uuid', None), ('uuid', 'not-a-uuid'), ('uuid', 'ABCDEFAB-1234-1234-1234-123456789012'),
    ('misc', True), ('misc', -1), ('misc', 32768), ('cost', None), ('cost', -3),
    ('exhausts', 1), ('ethereal', 'false'), ('unmapped', 1),
])
def test_invalid_wire_record_is_rejected(field, value):
    record = real_card()
    record[field] = value
    with pytest.raises(ValueError):
        real_card_state(record)


@pytest.mark.parametrize('field,value', [
    ('version', True), ('version', 2), ('base_cost', -3), ('base_cost', 128),
    ('base_damage', -2), ('base_block', None), ('free_to_play_once', 0),
    ('retain', 'false'), ('self_retain', 1), ('innate', None),
    ('bottled_flame', 1), ('bottled_lightning', None), ('bottled_tornado', 'false'),
    ('unmapped', 1),
])
def test_invalid_instance_sidecar_is_rejected(field, value):
    record = real_card()
    record['simulator_card_state'][field] = value
    with pytest.raises(ValueError):
        real_card_state(record)


@pytest.mark.parametrize('field', ['version', 'base_cost', 'bottled_flame'])
def test_incomplete_sidecar_is_not_filled_from_printed_defaults(field):
    record = real_card()
    del record['simulator_card_state'][field]
    with pytest.raises(ValueError):
        real_card_state(record)


def test_master_mapping_preserves_order_misc_and_bottles():
    records = [real_card('Searing Blow', upgrades=3, index=0, bottle='ATTACK'),
               real_card('RitualDagger', misc=39, index=1)]
    assert real_master_deck_state(records) == [
        dict(id='Searing Blow', upgrades=3, misc=3, bottle='ATTACK'),
        dict(id='Ritual Dagger', upgrades=0, misc=39, bottle='NONE')]


@pytest.mark.parametrize('problem', ['multiple_bottles', 'searing_misc', 'duplicate_uuid', 'capacity', 'nonlist'])
def test_invalid_master_deck_frame_is_rejected(problem):
    records = [real_card('Bash')]
    if problem == 'multiple_bottles':
        records[0]['simulator_card_state'].update(bottled_flame=True, bottled_tornado=True)
    elif problem == 'searing_misc':
        records = [real_card('Searing Blow', misc=7, upgrades=3)]
    elif problem == 'duplicate_uuid':
        records *= 2
    elif problem == 'capacity':
        records = [real_card(index=i) for i in range(97)]
    else:
        records = tuple(records)
    with pytest.raises(ValueError):
        real_master_deck_state(records)
