"""Native-v1 complete combat actions, independent of legacy policy slot ids.

This contract includes plays, single picks, all supported multi subsets, end
turn, and drinkable potions (not potion discard). It never renumbers candidates.
Only choices on the current select screen are exposed; native indices are
execution handles, not features describing future draw order.
"""
from dataclasses import dataclass
import numpy as np
from . import action_space as A


CONTRACT = 'native-v1'
JOINT_CONTRACT = 'joint-v2'
# Existing overworld controls keep their IDs. Combat IDs are complete actions;
# NONE is the sixth target, distinct from native enemy slot zero.
JOINT_END = A.N_ACTIONS
JOINT_PLAY = JOINT_END + 1
JOINT_POTION = JOINT_PLAY + 10 * 6
JOINT_PICK = JOINT_POTION + 5 * 6
JOINT_MULTI = JOINT_PICK + A.G_MAX
JOINT_SIZE = JOINT_MULTI + (1 << A.HAND_SLOTS)
JOINT_WIDTH = JOINT_SIZE - JOINT_END
AUTOREGRESSIVE_CONTRACT = 'autoregressive-v1'
AUTOREGRESSIVE_STOP = JOINT_MULTI + A.HAND_SLOTS
AUTOREGRESSIVE_SIZE = AUTOREGRESSIVE_STOP + 1
AUTOREGRESSIVE_WIDTH = AUTOREGRESSIVE_SIZE - JOINT_END
NEURAL_CONTRACTS = (JOINT_CONTRACT, AUTOREGRESSIVE_CONTRACT)


def policy_contract(space):
    """Recover the versioned action contract from the checkpoint's space."""
    spaces = getattr(space, 'spaces', {})
    if 'combat_legal' not in spaces:
        return 'legacy'
    width = spaces['combat_legal'].shape[-1]
    if width == JOINT_WIDTH:
        return JOINT_CONTRACT
    if width == AUTOREGRESSIVE_WIDTH:
        return AUTOREGRESSIVE_CONTRACT
    raise ValueError(f'unknown combat policy width: {width}')


def policy_size(contract):
    if contract == AUTOREGRESSIVE_CONTRACT:
        return AUTOREGRESSIVE_SIZE
    if contract == JOINT_CONTRACT:
        return JOINT_SIZE
    if contract == 'legacy':
        return A.N_ACTIONS
    raise ValueError(f'unknown policy contract: {contract}')


def autoregressive_action_id(bits, **kwargs):
    kind, source = int(bits) >> 29, int(bits) & 0xffff
    if kind == 5 and 0 <= source < A.HAND_SLOTS:
        return JOINT_MULTI + source
    if kind == 6 and source == 0:
        return AUTOREGRESSIVE_STOP
    if kind == 3:
        raise ValueError('complete subsets are not autoregressive policy actions')
    return joint_action_id(bits, **kwargs)


def autoregressive_native_bits(action_id, **kwargs):
    if JOINT_MULTI <= action_id < AUTOREGRESSIVE_STOP:
        return 5 << 29 | (action_id-JOINT_MULTI)
    if action_id == AUTOREGRESSIVE_STOP:
        return 6 << 29
    if action_id >= JOINT_MULTI:
        raise ValueError('invalid autoregressive action')
    return joint_native_bits(action_id, **kwargs)


def joint_action_id(bits, *, targeted=False, candidate_indices=()):
    """Complete native action -> policy ID, preserving targets and subsets."""
    bits = int(bits)
    kind, source, target = bits >> 29, bits & 0xffff, (bits >> 16) & 0x1fff
    if kind == 4:
        return JOINT_END
    if kind in (0, 1):
        limit, base = (10, JOINT_PLAY) if kind == 0 else (5, JOINT_POTION)
        if not 0 <= source < limit or (targeted and not 0 <= target < 5) or target > 5:
            raise ValueError('invalid play/potion source or target (discard is not a drink)')
        return base + source * 6 + (target if targeted else 5)
    if kind == 2:
        indices = tuple(int(i) for i in candidate_indices)
        if source not in indices or len(indices) > A.G_MAX:
            raise ValueError('selection handle is absent or exceeds public capacity')
        return JOINT_PICK + indices.index(source)
    if kind == 3 and 0 <= source < (1 << A.HAND_SLOTS):
        return JOINT_MULTI + source
    raise ValueError('invalid complete native action')


def joint_native_bits(action_id, *, candidate_indices=()):
    """Inverse policy mapping. Native no-target actions conventionally use zero."""
    action_id = int(action_id)
    if action_id == JOINT_END:
        return 4 << 29
    if JOINT_PLAY <= action_id < JOINT_PICK:
        kind = int(action_id >= JOINT_POTION)
        offset = action_id - (JOINT_POTION if kind else JOINT_PLAY)
        source, target = divmod(offset, 6)
        return kind << 29 | source | (0 if target == 5 else target) << 16
    if JOINT_PICK <= action_id < JOINT_MULTI:
        index = action_id - JOINT_PICK
        indices = tuple(int(i) for i in candidate_indices)
        if index >= len(indices):
            raise ValueError('selection policy ID has no current entity')
        return 2 << 29 | indices[index]
    if JOINT_MULTI <= action_id < JOINT_SIZE:
        return 3 << 29 | (action_id - JOINT_MULTI)
    raise ValueError('not a complete combat policy ID')


def joint_mask(policy_ids, *, contract=JOINT_CONTRACT):
    size = policy_size(contract)
    mask = np.zeros(size, dtype=bool)
    ids = np.asarray(policy_ids, dtype=np.int64)
    if np.any((ids < JOINT_END) | (ids >= size)):
        raise ValueError('invalid complete combat policy ID')
    mask[ids] = True
    return mask


def joint_semantic_action(action_id):
    """Public complete policy ID -> shared semantic action (no simulator needed)."""
    from . import actions, game_actions as ga
    aid = int(action_id)
    if aid == JOINT_END:
        return ga.combat(actions.END_TURN)
    if JOINT_PLAY <= aid < JOINT_PICK:
        potion = aid >= JOINT_POTION
        slot, target = divmod(aid - (JOINT_POTION if potion else JOINT_PLAY), 6)
        target = None if target == 5 else target
        return ga.use_potion(slot, target) if potion else ga.combat(actions.encode_play(slot, target))
    if JOINT_PICK <= aid < JOINT_MULTI:
        return ga.select(aid - JOINT_PICK)
    if JOINT_MULTI <= aid < JOINT_SIZE:
        bits = aid - JOINT_MULTI
        return ga.multi_select(i for i in range(A.HAND_SLOTS) if bits >> i & 1)
    raise ValueError("not a complete combat policy ID")


def legacy_action_id(joint_id):
    """Project a leaf's own public ID to an old source-only policy slot."""
    if joint_id == JOINT_END:
        return A.END_TURN
    if JOINT_PLAY <= joint_id < JOINT_POTION:
        return A._PLAY + (joint_id - JOINT_PLAY) // 6
    if JOINT_POTION <= joint_id < JOINT_PICK:
        return A._POTION + (joint_id - JOINT_POTION) // 6
    if JOINT_PICK <= joint_id < JOINT_MULTI:
        return A._PICK + joint_id - JOINT_PICK
    if JOINT_MULTI <= joint_id < JOINT_SIZE:
        return A._CONFIRM
    raise ValueError('invalid joint policy ID')


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
