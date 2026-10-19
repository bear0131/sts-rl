"""Explicit native instance snapshots shared by bridge integration tests."""

def native_card(card_id, uid, *, upgrades=0, special=0, base_cost=1, cost=1,
         free=False, retain=False, exhaust=False):
    return dict(id=card_id, uid=uid, upgrades=upgrades, special_data=special,
                base_cost=base_cost, cost=cost, free_to_play_once=free, retain=retain, exhausts=exhaust)


def native_card_state(*, hand=(), draw=(), discard=(), exhaust=(), stasis=(None, None), next_uid=100):
    return dict(version=2, next_uid=next_uid, hand=list(hand), draw=list(draw),
                discard=list(discard), exhaust=list(exhaust), stasis=list(stasis))


def native_master_card(card_id, *, upgrades=0, misc=0, bottle='NONE'):
    return dict(id=card_id, upgrades=upgrades, misc=misc, bottle=bottle)
