"""Strict, model-free construction of native combat roots from canonical globals.

This restores public state, not the hidden Java RNG or a pre-entry snapshot.
No on-equip callbacks run. Runtime battle fields are injected separately.
"""
from bridge.schemas.combat_state import real_to_sim_card_enum
from bridge.schemas.entity_ids import (
    BOSS_CANONICAL_IDS, NATIVE_COMBAT_CARD_ENUM_IDS, NATIVE_COMBAT_ROOM_IDS, NATIVE_GLOBAL_FIELDS,
    POTION_CAPACITY_RULES, REAL_ROOM_CLASS_IDS, RUN_KEY_FIELDS,
)
import slaythespire as sts


def native_combat_room(room_type):
    if room_type not in REAL_ROOM_CLASS_IDS:
        raise ValueError(f"unknown real room class: {room_type!r}")
    name = REAL_ROOM_CLASS_IDS[room_type]
    if name not in NATIVE_COMBAT_ROOM_IDS:
        raise ValueError(f"real room class has no supported combat root: {room_type!r}")
    return getattr(sts.Room, name)


def make_native_combat_root(global_state, *, seed, room_type, seed_nonce=0):
    missing = set(NATIVE_GLOBAL_FIELDS) - set(global_state)
    if missing:
        raise ValueError(f"native combat root is missing global fields: {sorted(missing)}")
    unknown = set(global_state) - set(NATIVE_GLOBAL_FIELDS)
    if unknown:
        raise ValueError(f"native combat root has unmapped global fields: {sorted(unknown)}")
    if global_state['character'] != 'IRONCLAD':
        raise ValueError("native combat bridge supports IRONCLAD only")
    room = native_combat_room(room_type)
    if type(global_state['combat_is_boss']) is not bool or global_state['combat_is_boss'] != (room == sts.Room.BOSS):
        raise ValueError("combat root boss flag differs from the exact room class")
    if global_state['boss_relics'] != []:
        raise ValueError("combat root cannot carry boss-reward screen candidates")

    def integer(key, low, high):
        value = global_state[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"native combat root {key} outside {low}..{high}: {value!r}")
        return value

    ascension = integer('ascension', 0, 20)
    act = integer('act', 1, 4)
    floor = integer('floor', 0, 2147483647)
    max_hp = integer('max_hp', 1, 2147483647)
    hp = integer('hp', 0, max_hp)
    gold = integer('gold', 0, 32767)
    if type(seed) is not int or not -(1 << 63) <= seed < (1 << 64):
        raise ValueError("native combat seed must be a Java signed-long or uint64 bit pattern")
    if type(seed_nonce) is not int or seed_nonce < 0:
        raise ValueError("shadow seed nonce must be a nonnegative integer")
    keys = global_state['keys']
    if not isinstance(keys, dict) or set(keys) != set(RUN_KEY_FIELDS) or any(type(v) is not bool for v in keys.values()):
        raise ValueError("native combat root requires all three explicit boolean keys")
    relics, counters = global_state['relics'], global_state['relic_counters']
    if len(relics) != len(counters):
        raise ValueError("native root relic identities/counters are not slot-aligned")
    if 'PrismaticShard' in relics:
        raise ValueError("PrismaticShard foreign-card combat transitions are not implemented")
    deck = []
    for cid, upgrades in global_state['deck']:
        native_id = real_to_sim_card_enum(cid)
        if native_id not in NATIVE_COMBAT_CARD_ENUM_IDS:
            raise ValueError(f"native deck card transition is not implemented: {cid!r}")
        if type(upgrades) is not int or not 0 <= upgrades <= 32767:
            raise ValueError("native deck upgrades must be an explicit nonnegative int16")
        if native_id != 'SEARING_BLOW' and upgrades > 1:
            raise ValueError("only Searing Blow can preserve multiple upgrades in a native master deck")
        deck.append((native_id, upgrades))
    capacity = POTION_CAPACITY_RULES['ascension_0_to_10' if ascension < 11 else 'ascension_11_to_20']
    if 'Potion Belt' in relics:
        capacity += POTION_CAPACITY_RULES['Potion Belt']
    potions = global_state['potions']
    if not isinstance(potions, list) or len(potions) != capacity:
        raise ValueError(f"native root potion slots must equal actual capacity {capacity}")
    if any(p == 'EMPTY_POTION_SLOT' for p in potions):
        raise ValueError("canonical potions cannot use a native empty sentinel")
    boss = global_state['boss']
    if not isinstance(boss, str) or boss not in BOSS_CANONICAL_IDS or not hasattr(sts.MonsterEncounter, boss):
        raise ValueError(f"real act boss is absent from native enum: {boss!r}")
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, (seed + seed_nonce) & ((1 << 64) - 1), ascension)
    gc.act, gc.floor_num = act, floor
    gc.max_hp, gc.cur_hp, gc.gold = max_hp, hp, gold
    gc.set_deck(deck)
    gc.set_relic_state([{'id': rid, 'counter': counter} for rid, counter in zip(relics, counters)])
    gc.potion_capacity = capacity
    gc.set_potion_state(['EMPTY_POTION_SLOT' if p == 'PAD' else p for p in potions])
    for semantic, fields in RUN_KEY_FIELDS.items():
        setattr(gc, fields['sim'], keys[semantic])
    gc.cur_room, gc.boss = room, getattr(sts.MonsterEncounter, boss)
    return gc
