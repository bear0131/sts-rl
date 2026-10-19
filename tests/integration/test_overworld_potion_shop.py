"""Real shop purchase and SHOP01 regression contracts; no whole-run evaluation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import slaythespire as sts

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.baselines.overworld_demand import EnemyDemandUpperStrategy
from rl.evaluation.baselines.overworld_energy import EnergyAwareUpperStrategy
from rl.evaluation.baselines.overworld_engine import EngineUpperStrategy, FuelUpperStrategy
from rl.evaluation.baselines.overworld_potion_shop import PotionShopUpperStrategy, rank_potion_shop
from rl.evaluation.baselines.overworld_rules import OverworldRuleStrategy
from rl.evaluation.baselines.overworld_survival import (
    GrowthUpperStrategy, ReadinessUpperStrategy, SurvivalUpperStrategy,
)
from rl.evaluation.strategies import DecisionRequest, strategy_manifest


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
def test_native_and_supplement_reject_unobtainable_potion(shop_factory,blocked_by):
    env=shop_factory(['Sozu'] if blocked_by=='Sozu' else ['The Courier'])
    if blocked_by=='full':
        for _ in range(3):
            env.step(A._SHOP_POTION+2)
    state=env.thick.obs_canonical()
    assert not env.action_masks()[A._SHOP_POTION+2]
    assert rank_potion_shop(state,[A._SHOP_POTION+2,A._PROCEED]) is None


@pytest.mark.parametrize('strategy_type', [OverworldRuleStrategy, EngineUpperStrategy,
    FuelUpperStrategy, SurvivalUpperStrategy, GrowthUpperStrategy,
    ReadinessUpperStrategy, PotionShopUpperStrategy, EnemyDemandUpperStrategy,
    EnergyAwareUpperStrategy])
@pytest.mark.parametrize('belt', ['empty', 'Sozu', 'full'])
def test_a20_upper_decisions_only_buy_obtainable_potions(shop_factory, strategy_type, belt):
    env = shop_factory(['Sozu'] if belt == 'Sozu' else ['The Courier'], ascension=20)
    action = A._SHOP_POTION + 2
    if belt == 'full':
        for _ in range(2):
            env.step(action)
    env.thick.gc.cur_hp = 30  # Base upper wants an affordable emergency potion.
    env._invalidate_backend_cache()
    before = deepcopy(env.thick.obs_canonical())
    assert len(before['global']['potions']) == 2
    assert bool(env.action_masks()[action]) is (belt == 'empty')
    assert env.action_masks()[A._PROCEED]
    # Compare potion/proceed after other purchases, while preserving the real
    # canonical screen and checking both choices against the actual native mask.
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[[action, A._PROCEED]] = True
    view = SimpleNamespace(unwrapped=env, action_masks=lambda: mask)
    request = DecisionRequest(view, {}, {'phase': 'shop'}, 42, 0)
    decision = strategy_type().decide([request])[0]
    assert decision.action == (action if belt == 'empty' else A._PROCEED)
    assert env.thick.obs_canonical() == before
    if belt == 'empty':
        price = before['options']['shop_potions'][2][1]
        potion = before['options']['shop_potions'][2][0]
        env.step(decision.action)
        after = env.thick.obs_canonical()['global']
        assert after['gold'] == before['global']['gold'] - price
        assert potion in after['potions']
        assert after['potions'].count('EMPTY_POTION_SLOT') == 1
    # The tests below also bypass the legal mask to verify the native mutator.


def test_affected_upper_manifests_identify_the_shared_potion_guard():
    base_version = 'ironclad-overworld-v3.1-potion-contracts'
    assert strategy_manifest(OverworldRuleStrategy())['version'] == base_version
    for strategy, expected in (
        (EngineUpperStrategy(), 'ironclad-overworld-engine-v2.2-potion-contracts'),
        (FuelUpperStrategy(), 'ironclad-overworld-fuel-v1.1-potion-contracts'),
    ):
        manifest = strategy_manifest(strategy)
        assert manifest['version'] == expected
        assert manifest['base_version'] == base_version
    assert strategy_manifest(PotionShopUpperStrategy())['base_version'] == base_version
    for strategy_type in (SurvivalUpperStrategy, GrowthUpperStrategy, ReadinessUpperStrategy):
        assert strategy_manifest(strategy_type())['base_events'] == base_version


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
