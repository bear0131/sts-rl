"""Typed real card state must never infer missing fields from a fresh card."""
import pytest

from tests.bridge_fixtures import real_card
from bridge.schemas.card_state import real_card_state, real_master_deck_state


def test_real_snapshot_preserves_instance_and_all_typed_runtime_fields():
    record = real_card('RitualDagger', upgrades=1, misc=39, cost=0, damage=39, exhaust=True)
    record['simulator_card_state']['free_to_play_once'] = True
    actual = real_card_state(record)
    assert (actual['id'], actual['up'], actual['uuid'], actual['misc']) == ('Ritual Dagger', 1, record['uuid'], 39)
    assert (actual['base_cost'], actual['cost'], actual['free_to_play_once'], actual['base_damage']) == (1, 0, True, 39)


def test_real_typed_card_field_inventory_matches_complete_word_table():
    from bridge.schemas.codec_contract import complete_word_tables
    table = complete_word_tables()
    assert set(real_card()['simulator_card_state']) == set(table['card_state_transport_fields'])
    assert set(table['native_master_deck_snapshot_fields']) == {'id', 'upgrades', 'misc', 'bottle'}
    assert table['card_state_field_contracts']['bottled_flame']['status'] == 'master_restored'


@pytest.mark.parametrize('field', ['uuid', 'misc', 'cost', 'exhausts', 'ethereal', 'simulator_card_state'])
def test_missing_card_state_cannot_be_filled_with_defaults(field):
    record = real_card()
    del record[field]
    with pytest.raises(ValueError, match='missing'):
        real_card_state(record)


@pytest.mark.parametrize('field,value', [('base_cost', True), ('base_cost', '1'), ('version', 2), ('retain', 1), ('bottled_flame', None)])
def test_malformed_typed_runtime_fields_are_rejected(field, value):
    record = real_card()
    record['simulator_card_state'][field] = value
    with pytest.raises(ValueError):
        real_card_state(record)


def test_unknown_card_state_and_multiple_bottles_are_not_discarded():
    record = real_card()
    record['simulator_card_state']['unknown_state'] = 3
    with pytest.raises(ValueError, match='inventory'):
        real_card_state(record)
    record = real_card(bottle='ATTACK')
    record['simulator_card_state']['bottled_lightning'] = True
    with pytest.raises(ValueError, match='multiple bottle'):
        real_master_deck_state([record])
