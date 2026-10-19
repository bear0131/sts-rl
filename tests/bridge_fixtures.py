"""Explicit real-card transport fixtures shared by bridge unit/integration tests."""
from uuid import UUID


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
