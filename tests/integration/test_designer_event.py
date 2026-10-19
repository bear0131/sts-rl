"""F20: actual Designer choices must match their displayed costs and effects."""

import pytest
import slaythespire as sts

from bridge.schemas.combat_state import _sim_screen_options


_DECK = [('STRIKE_RED', 0), ('DEFEND_RED', 0), ('BASH', 0),
         ('ANGER', 0), ('INFLAME', 0)]
# Public map entry fixtures, not a claim of naturally reaching Act 2.
_ENTRIES = {
    (0, False): (165, 5, 1), (0, True): (284, 4, 1),
    (15, False): (164, 3, 2), (15, True): (235, 4, 1),
}


@pytest.fixture
def designer_factory():
    def make(ascension, option):
        seed, x, y = _ENTRIES[ascension, option in (1, 3)]
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.act, gc.floor_num, gc.gold, gc.cur_hp = 2, 20, 500, 60
        gc.monster_chance = gc.shop_chance = gc.treasure_chance = 0
        gc.cur_map_node_y = y - 1
        gc.transition_to_map_node(x)
        gc.set_deck(_DECK)
        assert gc.cur_event_name == 'Designer'
        assert option in gc.legal_event_options()
        return gc
    return make


@pytest.mark.parametrize('ascension', [0, 15])
@pytest.mark.parametrize('option', range(6))
def test_designer_charges_only_selected_service(designer_factory, ascension, option):
    gc = designer_factory(ascension, option)
    before = gc.deck_ids()
    # Costs are the event's public contracts, not copied from its mutator.
    costs = (40, 40, 60, 60, 90, 0) if ascension == 0 else (50, 50, 75, 75, 110, 0)
    hp_loss = (3 if ascension == 0 else 5) if option == 5 else 0
    detail = next(d for d in gc.event_option_details() if d['index'] == option)
    assert detail.get('gold', 0) == costs[option]
    gc.choose_event_option(option)
    assert gc.gold == 500 - costs[option]
    assert gc.cur_hp == 60 - hp_loss

    if option in (0, 2, 3, 4):
        assert gc.screen_state_name.endswith('CARD_SELECT')
        assert gc.deck_ids() == before
        context = {0: 'UPGRADE', 2: 'REMOVE', 3: 'TRANSFORM', 4: 'REMOVE'}[option]
        assert gc.select_screen_info()['context'] == context
        assert gc.select_screen_info()['pick_count'] == (2 if option == 3 else 1)
        gc.choose_select_card_option(2)  # Bash; preserve identity across the grid.
        if option == 3:
            assert gc.deck_ids() == before  # Both selections precede the transform.
            assert gc.select_screen_info()['have_selected'] == 1
            assert gc.screen_state_name.endswith('CARD_SELECT')
            gc.choose_select_card_option(0)  # Strike, now a second distinct card.
            after = gc.deck_ids()
            assert len(after) == len(before)
            assert after[:3] == [before[1], before[3], before[4]]
            assert after[3][0] != 'BASH'
            assert after[4][0] != 'STRIKE_RED'
        elif option == 0:
            assert gc.deck_ids() == before[:2] + [('BASH', 1)] + before[3:]
        else:
            after = gc.deck_ids()
            assert [cid for cid, _ in after] == [cid for i, (cid, _) in enumerate(before) if i != 2]
            assert sum(up for _, up in after) == (1 if option == 4 else 0)
    elif option == 1:
        assert [cid for cid, _ in gc.deck_ids()] == [cid for cid, _ in before]
        assert sum(up for _, up in gc.deck_ids()) == 2
    else:
        assert gc.deck_ids() == before
    assert gc.screen_state_name.endswith('MAP_SCREEN')
    assert gc.gold == 500 - costs[option]
    assert gc.cur_hp == 60 - hp_loss


@pytest.mark.parametrize('ascension', [0, 15])
def test_designer_transform_metadata_preserves_player_selection(designer_factory, ascension):
    gc = designer_factory(ascension, 3)
    details = _sim_screen_options(gc, 'event')['event_choices']
    assert next(d for d in details if d['index'] == 3) == {
        'index': 3, 'kind': 'transform_cards',
        'gold': 60 if ascension == 0 else 75, 'count': 2,
    }


@pytest.mark.parametrize('ascension', [0, 15])
@pytest.mark.parametrize('first,second', [(i, j) for i in range(5) for j in range(5) if i != j])
def test_designer_transforms_exact_selected_cards_in_any_order(designer_factory, ascension, first, second):
    gc = designer_factory(ascension, 3)
    before = gc.deck_ids()
    gc.choose_event_option(3)
    gc.choose_select_card_option(first)
    assert gc.select_screen_info()['selected'] == [before[first]]
    assert gc.deck_ids() == before
    # The public candidate list removes the first pick; master-deck indices do not move yet.
    gc.choose_select_card_option(second - (second > first))
    after = gc.deck_ids()
    assert after[:-2] == [card for i, card in enumerate(before) if i not in (first, second)]
    assert len(after) == len(before)
    assert after[-2][0] != before[first][0]
    assert after[-1][0] != before[second][0]
    assert gc.screen_state_name.endswith('MAP_SCREEN')
