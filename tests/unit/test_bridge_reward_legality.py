"""REWARD01: reward commands must respect Sozu, physical slots and wire indices."""
import pytest

from bridge.real_game_env import RealGameAdapter
from env.core import game_actions as ga


@pytest.mark.parametrize('slots', [2, 3, 5])
@pytest.mark.parametrize('blocked', ['Sozu', 'full', 'available'])
def test_reward_potion_legality_and_command_guard(slots, blocked):
    adapter = RealGameAdapter()
    adapter.update({'in_game': True, 'available_commands': ['CHOOSE', 'PROCEED'],
        'game_state': {'screen_type': 'COMBAT_REWARD', 'choice_list': ['gold', 'potion', 'relic', 'potion'],
            'screen_state': {'rewards': [
                {'reward_type': 'GOLD', 'gold': 10},
                {'reward_type': 'POTION', 'potion': {'id': 'Block Potion'}},
                {'reward_type': 'RELIC', 'relic': {'id': 'Anchor'}},
                {'reward_type': 'POTION', 'potion': {'id': 'Fire Potion'}},
            ]}, 'relics': [{'id': 'Sozu'}] if blocked == 'Sozu' else [],
            'potions': [{'id': 'Block Potion' if blocked == 'full' else 'Potion Slot'}] * slots}})
    for i, wire in [(0, 1), (1, 3)]:
        action = ga.claim(ga.CLAIM_POTION, i)
        assert (action in adapter.legal_semantic_actions()) is (blocked == 'available')
        if blocked == 'available':
            assert adapter.command_for_semantic(action) == f'CHOOSE {wire}'
        else:
            with pytest.raises(ValueError):
                adapter.command_for_semantic(action)
    assert adapter.command_for_semantic(ga.claim(ga.CLAIM_GOLD, 0)) == 'CHOOSE 0'


@pytest.mark.parametrize('screen', ['SHOP_SCREEN', 'EVENT', 'CARD_REWARD'])
def test_claim_command_rejects_wrong_screen_even_with_stale_labels(screen):
    adapter = RealGameAdapter()
    adapter.update({'in_game': True, 'game_state': {'screen_type': screen,
        'choice_list': ['gold'], 'screen_state': {}}})
    with pytest.raises(ValueError):
        adapter.command_for_semantic(ga.claim(ga.CLAIM_GOLD, 0))
