"""BRIDGE26: public global snapshots must not lose or invent state at entry."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import slaythespire as sts

from bridge.native_state import make_native_combat_root, native_combat_room
from bridge.schemas.combat_state import normalize_sim, normalize_sim_global
from bridge.schemas.entity_ids import NATIVE_GLOBAL_FIELDS, REAL_ROOM_CLASS_IDS

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _global(ascension=0, belt=False):
    return {
        'character': 'IRONCLAD', 'ascension': ascension, 'act': 2, 'floor': 20,
        'hp': 53, 'max_hp': 87, 'gold': 421, 'boss': 'COLLECTOR',
        'combat_is_boss': False, 'boss_relics': [],
        'deck': [('Strike_R', 0), ('Bash', 1)],
        'deck_state': [{'id': 'Strike_R', 'upgrades': 0, 'misc': 0, 'bottle': 'NONE'},
                       {'id': 'Bash', 'upgrades': 1, 'misc': 0, 'bottle': 'NONE'}],
        'relics': ['Potion Belt'] if belt else ['Lizard Tail'],
        'relic_counters': [0],
        'potions': ['Fire Potion'] + ['PAD'] * ((3 if ascension < 11 else 2) + (2 if belt else 0) - 1),
        'keys': {'ruby': True, 'emerald': False, 'sapphire': True},
    }


@pytest.mark.parametrize('ascension,belt,capacity', [(0, False, 3), (0, True, 5), (10, True, 5), (11, False, 2), (11, True, 4), (20, True, 4)])
def test_complete_global_root_preserves_keys_and_actual_potion_capacity(ascension, belt, capacity):
    state = _global(ascension, belt)
    gc = make_native_combat_root(state, seed=42, room_type='EventRoom')
    assert gc.potion_capacity == capacity
    actual = normalize_sim_global(gc, combat=True)
    assert set(actual) - {'post_enemy_turn_boundary'} == set(NATIVE_GLOBAL_FIELDS)
    assert actual == dict(state, post_enemy_turn_boundary=False)
    battle = sts.create_battle(gc, 'Cultist')
    assert len(battle.to_dict()['potions']) == capacity
    assert battle.to_dict()['player']['gold'] == state['gold']


def test_shared_search_root_restores_current_growth_and_bottled_slot():
    state = _global()
    state['deck'] = [('Ritual Dagger', 1)]
    state['deck_state'] = [{'id': 'Ritual Dagger', 'upgrades': 1, 'misc': 39, 'bottle': 'ATTACK'}]
    gc = make_native_combat_root(state, seed=42, room_type='MonsterRoom')
    assert gc.deck[0].misc == 39
    assert gc.deck_state() == [{'id': 'RITUAL_DAGGER', 'upgrades': 1, 'misc': 39, 'bottle': 'ATTACK'}]


@pytest.mark.parametrize('key,value', [
    ('ascension', -1), ('ascension', 21), ('ascension', True), ('act', 0), ('act', 5),
    ('floor', -1), ('max_hp', 0), ('hp', -1), ('hp', 88), ('gold', -1),
    ('gold', 32768), ('gold', '421'), ('keys', {'ruby': True}),
    ('keys', {'ruby': 1, 'emerald': False, 'sapphire': True}),
    ('potions', ['PAD']), ('relic_counters', []), ('relic_counters', [True]),
    ('boss_relics', ['Circlet']), ('combat_is_boss', True),
    ('boss', 'CULTIST'), ('deck', [('Strike_R', 2)]), ('unmapped_state', 7),
])
def test_invalid_root_values_are_rejected_instead_of_clamped_or_defaulted(key, value):
    state = _global()
    state[key] = value
    with pytest.raises((ValueError, TypeError)):
        make_native_combat_root(state, seed=42, room_type='MonsterRoom')


@pytest.mark.parametrize('key', list(NATIVE_GLOBAL_FIELDS))
def test_every_root_field_is_required(key):
    state = _global()
    del state[key]
    with pytest.raises(ValueError, match='missing global fields'):
        make_native_combat_root(state, seed=42, room_type='MonsterRoom')


@pytest.mark.parametrize('name', list(REAL_ROOM_CLASS_IDS) + ['FakeMonsterRoomBoss', 'MONSTER'])
def test_room_identity_is_exact_not_a_substring(name):
    if name in {'MonsterRoom', 'MonsterRoomElite', 'MonsterRoomBoss', 'EventRoom'}:
        assert native_combat_room(name).name == REAL_ROOM_CLASS_IDS[name]
    else:
        with pytest.raises(ValueError, match='room class'):
            native_combat_room(name)


def test_potion_replacement_failure_is_atomic_and_capacity_cannot_delete_held_slots():
    gc = make_native_combat_root(_global(0, True), seed=42, room_type='MonsterRoom')
    gc.set_potion_state(['Fire Potion', 'EMPTY_POTION_SLOT', 'EMPTY_POTION_SLOT', 'EMPTY_POTION_SLOT', 'Block Potion'])
    before = list(gc.overworld_potions)
    with pytest.raises((ValueError, RuntimeError)):
        gc.set_potion_state(['Block Potion', 'Unknown Potion'])
    assert list(gc.overworld_potions) == before
    with pytest.raises(ValueError, match='held potion'):
        gc.potion_capacity = 3
    assert gc.potion_capacity == 5 and list(gc.overworld_potions) == before


def test_search_attach_uses_real_ascension_keys_and_event_room(combat_factory):
    from bridge.search_bridge import RealCombatSearch
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    state = _global(15, True)
    search = RealCombatSearch.__new__(RealCombatSearch)
    search.adapter = SimpleNamespace(ascension=0, gs={'room_type': 'EventRoom'}, obs_canonical=lambda: {'global': deepcopy(state)})
    search.env, search._combat_branch_bound, search.run_encounters = env, True, 0
    search._attach_battle(normalize_sim(env.thick.bc.to_dict()), (42, state['floor']))
    assert search.env.thick.gc.ascension == 15
    assert search.env.thick.gc.cur_room == sts.Room.EVENT
    assert search.env.thick.gc.red_key and search.env.thick.gc.blue_key
    assert not search.env.thick.gc.green_key
    assert search.env.thick.gc.potion_capacity == 4
