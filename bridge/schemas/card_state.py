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
