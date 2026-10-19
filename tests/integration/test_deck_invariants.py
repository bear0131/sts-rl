"""BRIDGE17: observable choices must preserve counts and physical card identity."""

import pytest
import slaythespire as sts

from env.core import game_actions as ga
from env.core.game_env import SimGameEnv


@pytest.mark.parametrize('cards,can_transform_two', [
    ([], False),
    ([('STRIKE_RED', 0)], False),
    ([('REGRET', 0)], False),
    ([('ASCENDERS_BANE', 0), ('CURSE_OF_THE_BELL', 0), ('NECRONOMICURSE', 0)], False),
    ([('STRIKE_RED', 0), ('ASCENDERS_BANE', 0)], False),
    ([('STRIKE_RED', 0), ('DEFEND_RED', 0)], True),
    ([('REGRET', 0), ('INFLAME', 0)], True),
])
def test_imported_deck_cannot_offer_more_picks_than_eligible_cards(
        designer_factory, cards, can_transform_two):
    gc = designer_factory(0, 3)
    gc.set_deck(cards)
    assert (3 in gc.legal_event_options()) is can_transform_two
    if can_transform_two:
        gc.choose_event_option(3)
        assert len(gc.get_select_cards()) == 2
        gc.choose_select_card_option(0)
        gc.choose_select_card_option(0)
        assert gc.screen_state_name.endswith('MAP_SCREEN')


@pytest.fixture
def bottled_deck_factory():
    def make(prefix, bottles):
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 63, 0)
        gc.set_deck([(prefix, 0), ('ANGER', 0), ('SHRUG_IT_OFF', 0),
                     ('INFLAME', 0), ('STRIKE_RED', 0)])
        targets = {'Bottled Flame': 'ANGER', 'Bottled Lightning': 'SHRUG_IT_OFF',
                   'Bottled Tornado': 'INFLAME'}
        # Seven bounded shop entries in a fixed map. Buy only requested bottles;
        # use the actual native bottle selection, not injected bottle indices.
        acquired = set()
        for _ in range(7):
            gc.cur_map_node_y, gc.gold = 1, 9999
            gc.transition_to_map_node(5)
            for slot, item in enumerate(gc.shop_contents()['relics']):
                if item['id'] not in bottles:
                    continue
                gc.shop_buy_relic(slot)
                assert gc.select_screen_info()['context'] == 'BOTTLE'
                cards = gc.get_select_cards()
                gc.choose_select_card_option(cards.index((targets[item['id']], 0)))
                acquired.add(item['id'])
        assert acquired == set(bottles)
        return gc
    return make


@pytest.mark.parametrize('bottles', [
    ('Bottled Flame',),
    ('Bottled Lightning',),
    ('Bottled Tornado',),
    ('Bottled Flame', 'Bottled Lightning', 'Bottled Tornado'),
])
@pytest.mark.parametrize('prefix', ['REGRET', 'PARASITE', 'DEFEND_RED'])
def test_removal_shifts_bottles_with_their_cards(bottled_deck_factory, prefix, bottles):
    gc = bottled_deck_factory(prefix, bottles)
    protected = {'Bottled Flame': 'ANGER', 'Bottled Lightning': 'SHRUG_IT_OFF',
                 'Bottled Tornado': 'INFLAME'}
    max_hp = gc.max_hp
    gc.shop_remove()
    gc.choose_select_card_option(gc.get_select_cards().index((prefix, 0)))
    assert gc.deck_ids() == [('ANGER', 0), ('SHRUG_IT_OFF', 0),
                             ('INFLAME', 0), ('STRIKE_RED', 0)]
    assert gc.max_hp == max_hp - (3 if prefix == 'PARASITE' else 0)
    # A new shop has a fresh removal service. Bottled identities must still be
    # excluded, while ordinary neighbors must remain removable.
    gc.cur_map_node_y, gc.gold = 1, 9999
    gc.transition_to_map_node(5)
    gc.shop_remove()
    expected = [card for card in gc.deck_ids()
                if card[0] not in {protected[bottle] for bottle in bottles}]
    assert gc.get_select_cards() == expected


@pytest.mark.parametrize('cards', [
    [('STRIKE_RED', 0), ('DEFEND_RED', 0), ('STRIKE_RED', 0)],
    [('DEFEND_RED', 0), ('STRIKE_RED', 0)],
])
def test_vampires_does_not_remove_a_stale_strike_outside_deck(cards):
    env = SimGameEnv()
    env.reset(seed=21)
    env.gc.skip_battles = True
    for _ in range(80):
        if env.phase() == 'event' and env.gc.cur_event_name == 'Vampires':
            break
        assert not env.is_over
        actions = env.legal_semantic_actions()
        assert actions
        env.step(ga.proceed() if ga.proceed() in actions else actions[0])
    else:
        pytest.fail('fixed bounded path did not reach Vampires')
    gc = env.gc
    gc.set_relics([])
    gc.set_deck(cards)
    # Removing the tail leaves a physical Strike beyond the logical deck end.
    gc.remove_card(len(cards) - 1)
    assert gc.deck_ids() == cards[:-1]
    assert 1 in gc.legal_event_options()
    gc.choose_event_option(1)
    assert gc.deck_ids() == [('DEFEND_RED', 0)] + [('BITE', 0)] * 5
    assert gc.screen_state_name.endswith('MAP_SCREEN')


@pytest.mark.parametrize('bottle', ['Bottled Flame', 'Bottled Lightning', 'Bottled Tornado'])
def test_fresh_deck_replacement_does_not_bottle_unrelated_cards(bottled_deck_factory, bottle):
    gc = bottled_deck_factory('REGRET', (bottle,))
    replacement = [('BASH', 0), ('ANGER', 0), ('DEFEND_RED', 0), ('INFLAME', 0), ('STRIKE_RED', 0)]
    gc.set_deck(replacement)
    gc.cur_map_node_y, gc.gold = 1, 9999
    gc.transition_to_map_node(5)
    gc.shop_remove()
    assert gc.get_select_cards() == replacement


def test_invalid_fresh_deck_does_not_partially_replace_original_cards(combat_factory):
    gc = combat_factory(['BASH', 'DEFEND_RED'], 'Cultist').thick.gc
    before = gc.deck_ids()
    with pytest.raises(RuntimeError, match='unknown card enum'):
        gc.set_deck([('ANGER', 0), ('UNKNOWN', 0)])
    assert gc.deck_ids() == before
