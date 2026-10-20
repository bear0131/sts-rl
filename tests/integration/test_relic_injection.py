"""Fresh relics and complete native snapshots are different input contracts."""
import pytest

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('relic,count', [
    ('Lizard Tail', 1), ('MawBank', 1), ('NlothsMask', 1), ('Circlet', 1),
    ('Matryoshka', 2), ('Omamori', 2), ('NeowsBlessing', 3), ('WingedGreaves', 3),
])
def test_fresh_relic_injection_uses_declared_initial_charges(combat_factory, relic, count):
    env = combat_factory(['STRIKE_RED'], relics=[relic])
    assert env.thick.gc.relic_state() == [{'id': relic, 'counter': count}]


@pytest.mark.parametrize('records', [
    ['Lizard Tail'], [{'id': 'Lizard Tail'}], [{'counter': 1}],
    [{'id': 'Lizard Tail', 'counter': True}], [{'id': 'Lizard Tail', 'counter': '1'}],
    [{'id': 'Lizard Tail', 'counter': 1, 'unmapped_state': 42}],
    [{'id': 'Lizard Tail', 'counter': 1}, {'id': 'Lizard Tail', 'counter': 0}],
    [{'id': 'Lizard Tail', 'counter': 1}, {'id': 'UNKNOWN', 'counter': 0}],
    [{'id': 'Lizard Tail', 'counter': -2}], [{'id': 'INVALID', 'counter': 0}],
    [{'id': 'Du-Vu Doll', 'counter': -1}],
])
def test_invalid_native_relic_snapshot_never_defaults_or_partially_replaces(combat_factory, records):
    env = combat_factory(['STRIKE_RED'], relics=['Burning Blood'])
    before = env.thick.gc.relic_state()
    with pytest.raises((ValueError, TypeError, RuntimeError, KeyError)):
        env.thick.gc.set_relic_state(records)
    assert env.thick.gc.relic_state() == before


def test_invalid_fresh_relic_list_does_not_erase_existing_state(combat_factory):
    env = combat_factory(['STRIKE_RED'], relics=['Burning Blood'])
    before = env.thick.gc.relic_state()
    with pytest.raises((ValueError, RuntimeError)):
        env.thick.gc.set_relics(['Lizard Tail', 'UNKNOWN'])
    assert env.thick.gc.relic_state() == before


def test_valid_relic_snapshot_preserves_explicit_consumed_and_accumulated_values(combat_factory):
    env = combat_factory(['STRIKE_RED'])
    state = [{'id': 'Lizard Tail', 'counter': 0}, {'id': 'Circlet', 'counter': 4},
             {'id': 'Du-Vu Doll', 'counter': 4}]
    env.thick.gc.set_relic_state(state)
    assert env.thick.gc.relic_state() == state


def test_complete_fresh_relic_word_table_matches_native_constructor(combat_factory):
    from bridge.schemas.codec_contract import complete_word_tables
    env = combat_factory(['STRIKE_RED'])
    for relic, contract in complete_word_tables()['relic_counter_contracts'].items():
        env.thick.gc.set_relics([relic])
        assert env.thick.gc.relic_state()[0]['counter'] == contract['fresh_native_initial_data']


def test_replay_restore_keeps_real_relic_counters_instead_of_injecting_fresh_ones():
    from bridge.replay import _sync_gc
    from tests.bridge_fixtures import real_card
    from bridge.schemas.entity_ids import SIMULATOR_PARITY_EXPORT_VERSION
    gc = _sync_gc(42, 0, {
        'simulator_parity_export': SIMULATOR_PARITY_EXPORT_VERSION, 'class': 'IRONCLAD',
        'ascension_level': 0, 'act': 1, 'room_type': 'MonsterRoom', 'act_boss': 'Slime Boss',
        'has_ruby_key': False, 'has_emerald_key': False, 'has_sapphire_key': False,
        'potions': [{'id': 'Potion Slot'}] * 3,
        'floor': 1, 'current_hp': 70, 'max_hp': 80, 'gold': 99,
        'deck': [real_card()],
        'relics': [{'id': 'Lizard Tail', 'counter': -2},
                   {'id': 'Happy Flower', 'counter': 2}, {'id': 'Circlet', 'counter': 4}],
    })
    assert gc.relic_state() == [
        {'id': 'Lizard Tail', 'counter': 0}, {'id': 'Happy Flower', 'counter': 2},
        {'id': 'Circlet', 'counter': 4}]
