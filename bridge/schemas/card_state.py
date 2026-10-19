"""Typed card instance transport; fresh constructor values are never snapshots."""
from uuid import UUID

from .entity_ids import CARD_STATE_TRANSPORT_FIELDS, REAL_CARD_WIRE_FIELDS


def real_card_state(record):
    from .combat_state import _card
    if not isinstance(record, dict):
        raise ValueError('real card state must be a record')
    unknown = set(record) - (set(REAL_CARD_WIRE_FIELDS) | {'simulator_card_state'})
    if unknown:
        raise ValueError(f'real card has unmapped fields: {sorted(unknown)}')
    required = {'id', 'upgrades', 'uuid', 'misc', 'cost', 'exhausts', 'ethereal', 'simulator_card_state'}
    missing = required - set(record)
    if missing:
        raise ValueError(f'real card snapshot is missing fields: {sorted(missing)}')
    extra = record['simulator_card_state']
    if not isinstance(extra, dict) or set(extra) != set(CARD_STATE_TRANSPORT_FIELDS):
        raise ValueError('real card simulator_card_state field inventory is incomplete or unknown')
    if type(extra['version']) is not int or extra['version'] != 1:
        raise ValueError('unsupported real card state version')
    card = _card(record['id'], record['upgrades'], record['cost'])
    identity = record['uuid']
    if not isinstance(identity, str):
        raise ValueError('real card uuid must be an explicit UUID string')
    try:
        if str(UUID(identity)) != identity:
            raise ValueError('real card uuid must use canonical Java UUID spelling')
    except (ValueError, AttributeError) as exc:
        raise ValueError('invalid real card uuid') from exc
    card['uuid'] = identity

    def integer(source, key, low, high):
        value = source[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'real card {key} must be an integer in {low}..{high}')
        return value

    card['misc'] = integer(record, 'misc', 0, 32767)
    card['base_cost'] = integer(extra, 'base_cost', -2, 127)
    card['cost'] = integer(record, 'cost', -2, 127)
    card['base_damage'] = integer(extra, 'base_damage', -1, 2147483647)
    card['base_block'] = integer(extra, 'base_block', -1, 2147483647)
    for name in ('free_to_play_once', 'retain', 'self_retain', 'innate',
                 'bottled_flame', 'bottled_lightning', 'bottled_tornado', 'exhausts', 'ethereal'):
        value = record[name] if name in ('exhausts', 'ethereal') else extra[name]
        if type(value) is not bool:
            raise ValueError(f'real card {name} must be an explicit boolean')
        card[name] = value
    return card


def real_selection_card_states(records, selected_records):
    """Preserve physical GRID identities, including identical printed cards."""
    if not isinstance(records, list) or not isinstance(selected_records, list):
        raise ValueError('GRID cards and selected_cards must be explicit lists')
    cards = [real_card_state(record) for record in records]
    selected = [real_card_state(record) for record in selected_records]
    by_uuid = {card['uuid']: card for card in cards}
    chosen = [card['uuid'] for card in selected]
    if (len(by_uuid) != len(cards) or len(set(chosen)) != len(chosen)
            or any(by_uuid.get(card['uuid']) != card for card in selected)):
        raise ValueError('GRID selected instances do not match their unique source cards')
    return cards, selected


def real_master_deck_state(records):
    if not isinstance(records, list):
        raise ValueError('real master deck must be a list')
    result = []
    for record in records:
        card = real_card_state(record)
        bottles = [name for field, name in (
            ('bottled_flame', 'ATTACK'), ('bottled_lightning', 'SKILL'), ('bottled_tornado', 'POWER')
        ) if card[field]]
        if len(bottles) > 1:
            raise ValueError('one master card cannot have multiple bottle kinds')
        misc = card['misc']
        if card['id'] == 'Searing Blow':
            if misc != 0:
                raise ValueError('Searing Blow has unrepresentable nonzero Java misc independent of upgrades')
            misc = card['up']  # Native persistence stores the upgrade count here.
        result.append({'id': card['id'], 'upgrades': card['up'], 'misc': misc,
                       'bottle': bottles[0] if bottles else 'NONE'})
    return result


def native_combat_card_state(card, uid):
    """Convert explicit instance data; uid=None denotes a generated template."""
    from .combat_state import real_to_sim_card_enum
    from .entity_ids import NATIVE_COMBAT_CARD_ENUM_IDS
    import slaythespire as sts

    native_id = real_to_sim_card_enum(card['id'])
    if native_id not in NATIVE_COMBAT_CARD_ENUM_IDS:
        raise ValueError(f'card combat transition is not implemented: {native_id}')
    traits = sts.card_instance_traits(native_id, card['up'])
    for field in ('ethereal', 'self_retain'):
        if card[field] != traits[field]:
            raise ValueError(f'{native_id} dynamic {field} is not representable in native state')
    special = card['misc']
    if native_id == 'SEARING_BLOW':
        if special != 0:
            raise ValueError('Searing Blow has independent nonzero misc')
        special = card['up']
    elif native_id == 'RAMPAGE':
        if special != 0:
            raise ValueError('Rampage has independent nonzero misc')
        special = card['base_damage'] - 8
    elif native_id != 'RITUAL_DAGGER' and special != 0:
        raise ValueError(f'{native_id} has unrepresentable nonzero misc')
    if not 0 <= special <= 32767:
        raise ValueError('combat card growth exceeds native int16')

    def cost(field):
        value = card[field]
        # Explicit sentinel translation, never construction as fallback.
        return -3 if value == -2 and traits['base_cost'] == -3 else value

    return dict(id=native_id, upgrades=card['up'], uid=uid, special_data=special,
                base_cost=cost('base_cost'), cost=cost('cost'),
                free_to_play_once=card['free_to_play_once'], retain=card['retain'],
                exhausts=card['exhausts'])


class CombatCardIdentityMap:
    """Physical master slots plus persistent, disjoint temporary-card identities.

    Master slots are read afresh because Java can change the master deck during
    combat. All live cards are restored together at a settled decision boundary.
    Temporary UUIDs survive pile moves and disappearance; their IDs are never
    recycled within a combat. Native speculative copies do not update this map.
    """
    MASTER_CAPACITY = 96

    def __init__(self):
        self._temporary = {}

    def _master(self, master_records):
        master = [real_card_state(row) for row in master_records]
        if len(master) > self.MASTER_CAPACITY:
            raise ValueError('master deck exceeds native identity capacity')
        master_ids = {card['uuid']: index for index, card in enumerate(master)}
        if len(master_ids) != len(master):
            raise ValueError('duplicate master card UUID')
        return master, master_ids

    @staticmethod
    def _convert(card, master, master_ids, temporary):
        identity = card['uuid']
        if identity in master_ids:
            uid = master_ids[identity]
            if master[uid]['id'] != card['id']:
                raise ValueError('combat UUID refers to a different master card identity')
        elif identity in temporary:
            uid = temporary[identity]
        else:
            # A pending action may have created a card, but its speculative UID
            # is not evidence of the real UUID. Never invent that association.
            raise ValueError('selection has an unbound temporary card UUID')
        return native_combat_card_state(card, uid)

    def selection_records(self, master_records, cards):
        """Validate known physical identities without allocating during an action."""
        master, master_ids = self._master(master_records)
        identities = [card['uuid'] for card in cards]
        if len(set(identities)) != len(identities):
            raise ValueError('duplicate physical selection card UUID')
        return [self._convert(card, master, master_ids, self._temporary) for card in cards]

    def snapshot(self, master_records, piles, stasis):
        if set(piles) != {'hand', 'draw', 'discard', 'exhaust'} or len(stasis) != 2:
            raise ValueError('combat snapshot requires four piles and two stasis slots')
        master, master_ids = self._master(master_records)
        active = [card for pile in piles.values() for card in pile]
        active += [card for card in stasis if card is not None]
        identities = [card['uuid'] for card in active]
        if len(set(identities)) != len(identities):
            raise ValueError('duplicate physical combat card UUID')
        temporary = dict(self._temporary)
        for identity in sorted(set(identities) - set(master_ids) - set(temporary)):
            temporary[identity] = self.MASTER_CAPACITY + len(temporary)
        next_uid = self.MASTER_CAPACITY + len(temporary)
        if next_uid > 32767:
            raise ValueError('combat card identity cursor exceeds native int16')

        def convert(card):
            if card is None:
                return None  # Only a real empty stasis slot calls this branch.
            return self._convert(card, master, master_ids, temporary)

        result = dict(version=2, next_uid=next_uid,
                      **{name: [convert(card) for card in pile] for name, pile in piles.items()},
                      stasis=[convert(card) for card in stasis])
        self._temporary = temporary  # Failed conversion never consumes an identity.
        return result
