"""Real shop purchase and SHOP01 regression contracts; no whole-run evaluation."""
from copy import deepcopy

import pytest
import slaythespire as sts

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv


@pytest.fixture
def shop_factory():
    created = []
    def make(relics=(), *, ascension=0):
        env = SlayTheSpireEnv(combat_action_contract='native-v1', ascension=ascension)
        env.reset(seed=42)
        gc = env.thick.gc
        # Controlled public map entry creates native stock and callbacks; this
        # fixture makes no claim to replay the preceding whole-run decisions.
        x,y = next((x,y) for y in range(15) for x in range(7)
                   if env.thick.rmap.room_type(x,y) == 'SHOP')
        gc.set_relics(list(relics))
        gc.act, gc.floor_num, gc.cur_map_node_y, gc.gold = 2,18,y-1,400
        gc.transition_to_map_node(x)
        env._invalidate_backend_cache()
        assert env.thick.phase() == 'shop'
        created.append(env)
        return env
    yield make
    for env in created:
        env.close()


@pytest.mark.parametrize('ascension', [15, 16, 20])
@pytest.mark.parametrize('relics,low_price,high_price', [
    ([], 50, 55), (['The Courier'], 40, 44),
    (['Membership Card'], 25, 28), (['The Courier', 'Membership Card'], 20, 22),
])
def test_shop_price_reaches_observation_mask_and_purchase(shop_factory, ascension, relics,
                                                        low_price, high_price):
    env = shop_factory(relics, ascension=ascension)
    price = high_price if ascension >= 16 else low_price
    before = env.thick.obs_canonical()
    assert before['options']['shop_potions'][2] == ('Strength Potion', price)
    action = A._SHOP_POTION + 2
    env.thick.gc.gold = price - 1
    env._invalidate_backend_cache()
    assert not env.action_masks()[action]
    env.thick.gc.gold = price
    env._invalidate_backend_cache()
    assert env.action_masks()[action]
    env.step(action)
    after = env.thick.obs_canonical()
    assert after['global']['gold'] == 0
    assert after['global']['potions'][0] == 'Strength Potion'


@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('blocked_by', ['Sozu', 'full'])
def test_shop_never_charges_without_obtaining_a_potion(shop_factory, blocked_by, ascension):
    env = shop_factory(['Sozu'] if blocked_by == 'Sozu' else ['The Courier'],
                       ascension=ascension)
    gc = env.thick.gc
    if blocked_by == 'full':
        for _ in range(2 if ascension >= 11 else 3):
            env.step(A._SHOP_POTION + 2)
    before = deepcopy(env.thick.obs_canonical())
    shelf_before = gc.shop_contents()
    # A stale command bypassing enumeration must not spend gold or restock.
    for slot in range(3):
        gc.shop_buy_potion(slot)
        env._invalidate_backend_cache()
        assert env.thick.obs_canonical() == before
        assert gc.shop_contents() == shelf_before
    # Both native enumeration and the policy-facing mask must reject the buy.
    assert not any(kind == 'potion' for kind, _ in gc.legal_shop_options())
    assert not any(env.action_masks()[A._SHOP_POTION:A._SHOP_POTION + 3])


def _shop_snapshot(gc):
    return (gc.gold, gc.cur_hp, gc.max_hp, gc.deck_ids(), gc.relic_state(),
            list(gc.overworld_potions), gc.shop_contents(), gc.shop_remove_count,
            gc.screen_state)


@pytest.mark.parametrize('kind', ['card', 'relic', 'potion', 'remove'])
def test_shop_rejects_unaffordable_transaction(shop_factory, kind):
    gc = shop_factory().thick.gc
    gc.gold = 0
    before = _shop_snapshot(gc)
    method = gc.shop_remove if kind == 'remove' else getattr(gc, 'shop_buy_' + kind)
    with pytest.raises(ValueError, match='shop'):
        method(*(() if kind == 'remove' else (0,)))
    assert _shop_snapshot(gc) == before


@pytest.mark.parametrize('kind', ['card', 'relic', 'potion'])
def test_shop_rejects_sold_slot(shop_factory, kind):
    gc = shop_factory().thick.gc
    gc.gold = 1000
    method = getattr(gc, 'shop_buy_' + kind)
    price = gc.shop_contents()[kind + 's'][0]['price']
    method(0)
    assert gc.gold == 1000 - price
    assert gc.shop_contents()[kind + 's'][0]['price'] == -1
    before = _shop_snapshot(gc)
    with pytest.raises(ValueError, match='shop'):
        method(0)
    assert _shop_snapshot(gc) == before


@pytest.mark.parametrize('kind,limit', [('card', 7), ('relic', 3), ('potion', 3)])
@pytest.mark.parametrize('index', [-1, 'limit', 255, 256, 257])
@pytest.mark.parametrize('relics', [[], ['Sozu']])
def test_shop_rejects_raw_invalid_indices(shop_factory, kind, limit, index, relics):
    gc = shop_factory(relics).thick.gc
    before = _shop_snapshot(gc)
    with pytest.raises(ValueError, match='shop'):
        getattr(gc, 'shop_buy_' + kind)(limit if index == 'limit' else index)
    assert _shop_snapshot(gc) == before


@pytest.mark.parametrize('screen', ['REWARDS', 'EVENT_SCREEN', 'MAP_SCREEN', 'CARD_SELECT'])
@pytest.mark.parametrize('kind', ['card', 'relic', 'potion', 'remove', 'proceed'])
def test_shop_rejects_other_screens(shop_factory, screen, kind):
    import slaythespire as sts
    gc = shop_factory().thick.gc
    gc.screen_state = getattr(sts.ScreenState, screen)
    before = _shop_snapshot(gc)
    assert gc.legal_shop_options() == []
    method = getattr(gc, 'shop_' + (kind if kind in ('remove', 'proceed') else 'buy_' + kind))
    with pytest.raises(ValueError, match='shop'):
        method(*(() if kind in ('remove', 'proceed') else (0,)))
    assert _shop_snapshot(gc) == before


def test_shop_remove_is_single_use_and_returns_to_shop(shop_factory):
    gc = shop_factory().thick.gc
    gold = gc.gold
    cost = gc.shop_contents()['remove_cost']
    gc.shop_remove()
    assert gc.gold == gold - cost
    gc.choose_select_card_option(0)
    assert ('remove', 0) not in gc.legal_shop_options()
    assert gc.shop_contents()['remove_cost'] == -1
    before = _shop_snapshot(gc)
    with pytest.raises(ValueError, match='shop'):
        gc.shop_remove()
    assert _shop_snapshot(gc) == before
    gc.shop_proceed()
    assert str(gc.screen_state) == 'ScreenState.MAP_SCREEN'


@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('egg,upgraded_types', [
    ('Molten Egg 2', [sts.CardType.ATTACK]),
    ('Toxic Egg 2', [sts.CardType.SKILL]),
    ('Frozen Egg 2', [sts.CardType.POWER]),
    ('all', [sts.CardType.ATTACK, sts.CardType.SKILL, sts.CardType.POWER]),
])
def test_shop_egg_preview_matches_purchased_card(shop_factory, ascension, egg, upgraded_types):
    relics = ['Molten Egg 2', 'Toxic Egg 2', 'Frozen Egg 2'] if egg == 'all' else [egg]
    env = shop_factory(relics, ascension=ascension)
    gc = env.thick.gc
    gc.gold = 10000
    shelf = gc.shop_contents()['cards']
    before = env.thick.obs_canonical()
    for i, item in enumerate(shelf):
        card = sts.Card(getattr(sts.CardId, item['id']))
        expected = int(card.type in upgraded_types)
        assert item['upgrades'] == expected
        assert before['options']['shop_cards'][i][1] == expected
        gold = gc.gold
        gc.shop_buy_card(i)
        assert gc.deck_ids()[-1] == (item['id'], expected)
        assert gc.gold == gold - item['price']


@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('slot,type_', [(0, sts.CardType.ATTACK), (2, sts.CardType.SKILL),
                                      (4, sts.CardType.POWER)])
def test_courier_repeated_public_buys_keep_shelf_type(shop_factory, ascension, slot, type_):
    env = shop_factory(['The Courier', 'Molten Egg 2', 'Toxic Egg 2', 'Frozen Egg 2'],
                       ascension=ascension)
    gc = env.thick.gc
    gc.gold = 10000
    for _ in range(8):
        before = gc.shop_contents()['cards'][slot]
        gold = gc.gold
        gc.shop_buy_card(slot)
        assert gc.gold == gold - before['price']
        assert gc.deck_ids()[-1] == (before['id'], before['upgrades'])
        after = gc.shop_contents()['cards'][slot]
        card = sts.Card(getattr(sts.CardId, after['id']))
        assert card.type == type_
        assert after['upgrades'] == 1
        env._invalidate_backend_cache()
        assert env.action_masks()[A._SHOP_CARD + slot]
        assert env.thick.obs_canonical()['options']['shop_cards'][slot][1] == 1
