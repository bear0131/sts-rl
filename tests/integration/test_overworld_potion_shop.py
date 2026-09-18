"""Real shop purchase and known SHOP01 boundaries; no whole-run evaluation."""
from copy import deepcopy

import pytest
import slaythespire as sts

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.baselines.overworld_potion_shop import rank_potion_shop


@pytest.fixture
def shop_factory():
    created = []
    def make(relics=()):
        env = SlayTheSpireEnv(combat_action_contract='native-v1')
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


def test_native_purchase_changes_gold_inventory_and_sold_slot_then_is_usable(shop_factory):
    env = shop_factory()
    before = deepcopy(env.thick.obs_canonical())
    assert before['options']['shop_potions'][2] == ('Strength Potion',50)
    assert before['global']['potions'] == ['EMPTY_POTION_SLOT'] * 3
    # Isolate the post-preferred-purchase potion/proceed surface. Every chosen
    # action below is also checked against the full real environment mask.
    action = rank_potion_shop(before,[A._SHOP_POTION+2,A._PROCEED])[0][0]['action']
    assert action == A._SHOP_POTION+2 and env.action_masks()[action]
    env.step(action)
    after = env.thick.obs_canonical()
    assert after['global']['gold'] == before['global']['gold']-50
    assert after['global']['potions'] == ['Strength Potion','EMPTY_POTION_SLOT','EMPTY_POTION_SLOT']
    assert after['options']['shop_potions'][2][1] == -1
    assert not env.action_masks()[action]
    assert rank_potion_shop(after,[A._SHOP_POTION,A._PROCEED]) is None
    env.thick.bc = sts.create_battle(env.thick.gc,'Jaw Worm')
    env._invalidate_backend_cache()
    before_strength = dict(env.thick.bc.to_dict()['player']['powers']).get('Strength',0)
    potion = next(bits for bits in env.native_combat_actions()
                  if sts.Action(bits).action_type == sts.ActionType.POTION
                  and sts.Action(bits).source_idx == 0)
    env.step_native_combat(potion)
    after_combat = env.thick.obs_canonical()
    assert dict(env.thick.bc.to_dict()['player']['powers']).get('Strength',0) == before_strength+2
    assert after_combat['global']['potions'][0] == 'EMPTY_POTION_SLOT'


def test_native_courier_restock_does_not_create_second_reserve(shop_factory):
    env=shop_factory(['The Courier'])
    before=env.thick.obs_canonical()
    # Courier has already discounted the actual native price.
    assert before['options']['shop_potions'][2] == ('Strength Potion',40)
    action=rank_potion_shop(before,[A._SHOP_POTION+2,A._PROCEED])[0][0]['action']
    env.step(action)
    after=env.thick.obs_canonical()
    assert after['global']['gold']==360
    assert after['options']['shop_potions'][2][1] >= 0
    assert env.action_masks()[action]
    assert rank_potion_shop(after,[action,A._PROCEED]) is None


@pytest.mark.parametrize('blocked_by',['Sozu','full'])
def test_supplement_explicitly_rejects_native_legal_but_unobtainable_potion(shop_factory,blocked_by):
    env=shop_factory(['Sozu'] if blocked_by=='Sozu' else ['The Courier'])
    if blocked_by=='full':
        for _ in range(3):
            env.step(A._SHOP_POTION+2)
    state=env.thick.obs_canonical()
    assert env.action_masks()[A._SHOP_POTION+2]
    assert rank_potion_shop(state,[A._SHOP_POTION+2,A._PROCEED]) is None


@pytest.mark.parametrize('blocked_by',['Sozu','full'])
@pytest.mark.xfail(strict=True,raises=AssertionError,reason='SHOP01: native shop charges for unobtainable Sozu/full-belt potion')
def test_shop_never_charges_without_obtaining_a_potion(shop_factory,blocked_by):
    env=shop_factory(['Sozu'] if blocked_by=='Sozu' else ['The Courier'])
    if blocked_by=='full':
        for _ in range(3):
            env.step(A._SHOP_POTION+2)
    before=deepcopy(env.thick.obs_canonical()['global'])
    action=A._SHOP_POTION+2
    if not env.action_masks()[action]:
        return  # Rejecting the purchase is an acceptable SHOP01 fix.
    env.step(action)
    after=env.thick.obs_canonical()['global']
    assert after['potions'] == before['potions']  # Native obtain is genuinely blocked.
    assert after['gold'] == before['gold'], 'SHOP01: gold deducted with no possible potion obtain'
