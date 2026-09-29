"""Native-v1 complete combat actions, independent of legacy policy slot ids.

This contract includes plays, single picks, all supported multi subsets, end
turn, and drinkable potions (not potion discard). It never renumbers candidates.
Only choices on the current select screen are exposed; native indices are
execution handles, not features describing future draw order.
"""
from dataclasses import dataclass


CONTRACT = 'native-v1'


def action_bits(action):
    """Lossless native Action packing, including MULTI's low-bit selection set."""
    return int(action.action_type) << 29 | int(action.target_idx) << 16 | int(action.source_idx)


def complete_actions(battle):
    """Use native legality, including the existing exhaustive multi enumerator."""
    enumerate_actions = getattr(battle, 'rule_actions', None)
    if enumerate_actions is None:
        raise RuntimeError('native-v1 requires a native build with rule_actions')
    result = [action_bits(action) for action in enumerate_actions()]
    for potion in battle.legal_potions():
        targets = potion['targets'] if potion['requires_target'] else (0,)
        result.extend(1 << 29 | int(target) << 16 | int(potion['slot']) for target in targets)
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class SelectionCandidate:
    contract: str
    context: str
    zone: str
    native_index: int
    action_bits: int
    card: tuple | None  # native card id, upgrade count; None is an explicit skip


def selection_candidates(battle):
    """Map legal SINGLE_CARD_SELECT actions to their actual source entities."""
    if int(battle.input_state) != 2:
        return ()
    desc = battle.select_info()
    if desc['multi']:
        return ()
    if 'candidates' in desc:
        return tuple(SelectionCandidate(
            CONTRACT, desc['context'], desc['zone'], int(item['native_index']),
            2 << 29 | int(item['native_index']),
            None if item['skip'] else tuple(item['card'][:2]),
        ) for item in desc['candidates'])
    context = desc['context']
    zones = {'SECRET_TECHNIQUE': 'draw', 'SECRET_WEAPON': 'draw',
             'HEADBUTT': 'discard', 'LIQUID_MEMORIES_POTION': 'discard',
             'EXHUME': 'exhaust', 'CODEX': 'generated', 'DISCOVERY': 'generated',
             'ARMAMENTS': 'hand', 'DUAL_WIELD': 'hand', 'EXHAUST_ONE': 'hand',
             'FORETHOUGHT': 'hand', 'WARCRY': 'hand'}
    if context not in zones:
        raise ValueError(f'native-v1 unsupported selection context: {context}')
    zone = zones[context]
    cards = desc['generated_cards'] if zone == 'generated' else battle.to_dict()[zone]
    result = []
    for bits in complete_actions(battle):
        if bits >> 29 != 2:
            continue
        index = bits & 0xffff
        card = None if context == 'CODEX' and index == 3 else tuple(cards[index][:2])
        result.append(SelectionCandidate(CONTRACT, context, zone, index, bits, card))
    return tuple(result)
