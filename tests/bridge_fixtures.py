"""Explicit native and real instance snapshots shared by bridge tests."""
from uuid import UUID

def native_card(card_id, uid, *, upgrades=0, special=0, base_cost=1, cost=1,
         free=False, retain=False, exhaust=False):
    return dict(id=card_id, uid=uid, upgrades=upgrades, special_data=special,
                base_cost=base_cost, cost=cost, free_to_play_once=free, retain=retain, exhausts=exhaust)


def native_card_state(*, hand=(), draw=(), discard=(), exhaust=(), stasis=(None, None), next_uid=100):
    return dict(version=2, next_uid=next_uid, hand=list(hand), draw=list(draw),
                discard=list(discard), exhaust=list(exhaust), stasis=list(stasis))


def native_master_card(card_id, *, upgrades=0, misc=0, bottle='NONE'):
    return dict(id=card_id, upgrades=upgrades, misc=misc, bottle=bottle)


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
