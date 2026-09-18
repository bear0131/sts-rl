"""Supplemental shop preparation, with explicit opportunity cost and delegation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from env.core import action_space as A
from rl.evaluation.baselines.overworld_potion_shop import (
    PotionShopUpperStrategy, potion_preference, rank_potion_shop,
)
from rl.evaluation.baselines.overworld_rules import rank_overworld
from rl.evaluation.strategies import Decision, DecisionRequest, UPPER_STRATEGIES, strategy_manifest


@pytest.fixture
def shop():
    return dict(phase='shop', options=dict(
        shop_potions=[('Fire Potion', 50), ('Explosive Potion', 50), ('Weak Potion', 50)],
        shop_cards=[('Immolate', 0, 60)], shop_relics=[], shop_remove_cost=75),
        **{'global': dict(act=2, hp=80, max_hp=80, gold=300, floor=20, relics=[],
            deck=[('Strike_R', 0)] * 5 + [('Defend_R', 0)] * 4 + [('Bash', 0)],
            potions=['EMPTY_POTION_SLOT'] * 3)})


LEGAL = [A._SHOP_POTION, A._SHOP_POTION + 1, A._SHOP_POTION + 2, A._PROCEED]


def test_healthy_empty_belt_prepares_missing_area_damage(shop):
    before = deepcopy(shop)
    assert rank_overworld(shop, LEGAL)[0]['action'] == A._PROCEED
    rows, original = rank_potion_shop(shop, LEGAL)
    assert original == A._PROCEED and rows[0]['action'] == A._SHOP_POTION + 1
    assert rows[0]['preparation']['role'] == 'area-damage'
    assert shop == before
    shop['global']['deck'].append(('Cleave', 0))
    assert rank_potion_shop(shop, LEGAL)[0][0]['action'] == A._SHOP_POTION


@pytest.mark.parametrize('gold,price,eligible', [(125,50,True),(124,50,False),(150,75,True),(160,81,False)])
def test_reserves_gold_and_rejects_majority_spending(shop, gold, price, eligible):
    shop['global']['gold'] = gold
    shop['options']['shop_potions'] = [('Heart Of Iron',price)]
    assert (rank_potion_shop(shop, [A._SHOP_POTION,A._PROCEED]) is not None) == eligible


@pytest.mark.parametrize('belt', [[], ['INVALID'], ['Fire Potion'], ['EMPTY_POTION_SLOT','Weak Potion']])
def test_actual_empty_belt_is_required(shop, belt):
    shop['global']['potions'] = belt
    assert rank_potion_shop(shop, LEGAL) is None


@pytest.mark.parametrize('belt', [['EMPTY_POTION_SLOT'] * 2, ['EMPTY_POTION_SLOT'] * 5, [''] * 3])
def test_uses_observed_capacity_and_bridge_empty_representation(shop, belt):
    shop['global']['potions'] = belt
    assert rank_potion_shop(shop, LEGAL) is not None


def test_sozu_unsupported_effect_and_non_act2_abstain(shop):
    shop['global']['relics'] = ['Sozu']
    assert rank_potion_shop(shop, LEGAL) is None
    shop['global']['relics'] = []
    shop['options']['shop_potions'] = [('Skill Potion', 50)]
    assert rank_potion_shop(shop, [A._SHOP_POTION,A._PROCEED]) is None
    shop['options']['shop_potions'] = [('Fire Potion', 50)]
    shop['global']['act'] = 3
    assert rank_potion_shop(shop, [A._SHOP_POTION,A._PROCEED]) is None
    shop['phase'] = 'rest'
    assert rank_potion_shop(shop, [A._CAMP]) is None


def test_v1_preferred_card_or_removal_is_preserved(shop):
    for action in (A._SHOP_CARD, A._SHOP_REMOVE):
        legal = [*LEGAL, action]
        assert rank_overworld(shop, legal)[0]['action'] == action
        assert rank_potion_shop(shop, legal) is None


def test_masks_live_prices_and_current_inventory_not_cached(shop):
    shop['global']['relics'] = ['Membership Card', 'The Courier']
    shop['options']['shop_potions'] = [('Fire Potion', 50), ('Fire Potion', 40), ('Fire Potion', -1)]
    rows, _ = rank_potion_shop(shop, [A._SHOP_POTION,A._SHOP_POTION+1,A._PROCEED])
    assert rows[0]['action'] == A._SHOP_POTION + 1
    assert rows[0]['preparation']['price'] == 40
    assert rows[0]['preparation']['gold_after'] == 260
    assert rank_potion_shop(shop, [A._PROCEED]) is None
    shop['global']['potions'][0] = 'Fire Potion'
    shop['options']['shop_potions'][1] = ('Heart Of Iron',30)  # Courier restock
    assert rank_potion_shop(shop, LEGAL) is None


def test_strength_fear_dexterity_require_existing_cards(shop):
    shop['global']['deck'] = [('Inflame',0)] * 5
    for name in ('Strength Potion','Fear Potion','Dexterity Potion'):
        assert potion_preference(name, shop) is None
    shop['global']['deck'] = [('Twin Strike',0)] * 3 + [('Defend_R',0)] * 3
    assert potion_preference('Strength Potion',shop)[1] == 'attack-scaling'
    assert potion_preference('Dexterity Potion',shop)[1] == 'block-scaling'


def request(state, legal, seed):
    mask = np.zeros(A.N_ACTIONS,dtype=bool);mask[legal] = True
    env = SimpleNamespace(thick=SimpleNamespace(obs_canonical=lambda: state, rmap=None,
        gc=SimpleNamespace(cur_map_node_y=0)),action_masks=lambda: mask)
    env.unwrapped = env
    return DecisionRequest(env,{}, {'phase':state['phase']},seed,0)


def test_batch_fallback_trace_manifest_and_default_identity(shop):
    class Sentinel:
        name = 'sentinel'
        def decide(self, requests, *, check):
            assert [r.seed for r in requests] == [2]
            return [Decision(A._EVENT+1,self.name)]
    unknown = deepcopy(shop);unknown.update(phase='event',options={'event':'UNKNOWN'})
    trace=[];strategy=PotionShopUpperStrategy(Sentinel(),trace=trace.append)
    decisions=strategy.decide([request(shop,LEGAL,1),request(unknown,[A._EVENT+1],2)])
    assert decisions[0].action == A._SHOP_POTION+1
    assert decisions[1].diagnostics['overworld_fallback_decisions'] == 1
    assert {r['seed'] for r in trace} == {1,2}
    assert all(r['kind']=='overworld_decision' and 'action' in r for r in trace)
    manifest=strategy_manifest(strategy)
    assert manifest['fallback']['name']=='sentinel'
    assert manifest['version']=='ironclad-overworld-potion-shop-v1'
    assert strategy.decide([])==[]
    args=SimpleNamespace(stochastic=False)
    model=object()
    assert UPPER_STRATEGIES['model'](model,args).model is model
    assert UPPER_STRATEGIES['potion-shop'](model,args).fallback.model is model
