"""Explicit real-card transport fixtures shared by bridge unit/integration tests."""
from uuid import UUID


def native_card(card_id, uid, *, upgrades=0, special=0, base_cost=1, cost=1,
                free=False, retain=False, exhaust=None, cached_damage=0):
    if exhaust is None:
        import slaythespire as sts
        exhaust = sts.card_instance_traits(card_id, upgrades)['exhausts']
    return dict(id=card_id, uid=uid, upgrades=upgrades, special_data=special,
                cached_base_damage=cached_damage if card_id == 'MIND_BLAST' else None,
                base_cost=base_cost, cost=cost, free_to_play_once=free, retain=retain,
                exhausts=exhaust)


def native_card_state(*, hand=(), draw=(), discard=(), exhaust=(),
                      stasis=(None, None), next_uid=100):
    return dict(version=3, next_uid=next_uid, hand=list(hand), draw=list(draw),
                discard=list(discard), exhaust=list(exhaust), stasis=list(stasis))


def real_card(card_id='Strike_R', *, upgrades=0, misc=0, bottle='NONE', index=0,
              base_cost=1, cost=None, damage=-1, block=-1, exhaust=False, ethereal=False):
    return {
        'id': card_id, 'upgrades': upgrades, 'misc': misc,
        'uuid': str(UUID(int=index + 1)),
        'cost': base_cost if cost is None else cost, 'exhausts': exhaust, 'ethereal': ethereal,
        'simulator_card_state': {
            'version': 1, 'base_cost': base_cost, 'base_damage': damage, 'base_block': block,
            'free_to_play_once': False, 'retain': False, 'self_retain': False, 'innate': False,
            'bottled_flame': bottle == 'ATTACK', 'bottled_lightning': bottle == 'SKILL',
            'bottled_tornado': bottle == 'POWER',
        },
    }


def real_combat_state(*, deck=(), hand=(), draw=(), discard=(), exhaust=(), monsters=None):
    """Settled, complete parity frame; explicit neutral values are test data."""
    from bridge.schemas.combat_state import _REAL_COMBAT_COUNTER_KEYS
    from bridge.schemas.entity_ids import SIMULATOR_PARITY_EXPORT_VERSION
    if monsters is None:
        monsters = [{
            'id': 'Cultist', 'current_hp': 100, 'max_hp': 100, 'block': 0,
            'is_gone': False, 'half_dead': False, 'powers': [], 'intent': 'BUFF',
            'is_dying': False, 'is_dead': False, 'is_escaping': False,
            'escape_next': False, 'unawakened': False,
            'move_base_damage': -1, 'move_adjusted_damage': -1, 'move_hits': 0,
            'move_id': 3, 'last_move_id': -1, 'second_last_move_id': -1,
            'current_intent_export': SIMULATOR_PARITY_EXPORT_VERSION,
            'simulator_stasis_card': None,
        }]
    return {
        'gold': 99, 'relics': [], 'deck': list(deck),
        'combat_state': {
            'simulator_parity_export': SIMULATOR_PARITY_EXPORT_VERSION,
            'simulator_energy_per_turn': 3,
            'simulator_combat_counters': {key: 0 for key in _REAL_COMBAT_COUNTER_KEYS},
            'simulator_player_power_overrides': {}, 'turn': 1,
            'simulator_bombs': [],
            'player': {'current_hp': 80, 'max_hp': 80, 'block': 0, 'energy': 3, 'powers': []},
            'monsters': monsters, 'hand': list(hand), 'draw_pile': list(draw),
            'discard_pile': list(discard), 'exhaust_pile': list(exhaust),
        },
    }
