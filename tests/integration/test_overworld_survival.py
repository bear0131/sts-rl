"""Native A20 public-state/legal-action smoke for the survival upper policy."""
from copy import deepcopy

import pytest

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.baselines.combat_rules import CombatRuleStrategy
from rl.evaluation.baselines.overworld_survival import (
    GrowthUpperStrategy, ReadinessUpperStrategy, SurvivalUpperStrategy, rank_survival,
)
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate
from rl.evaluation.strategies import DecisionRequest, LeftmostStrategy, PhaseStrategy


@pytest.mark.parametrize('strategy_type', [SurvivalUpperStrategy, GrowthUpperStrategy, ReadinessUpperStrategy])
def test_native_a20_neow_and_map_decisions_are_public_legal_and_nonmutating(strategy_type):
    env = SlayTheSpireEnv(ascension=20)
    try:
        obs, info = env.reset(seed=20200920)
        strategy = strategy_type(LeftmostStrategy())
        phases = set()
        for step in range(20):
            before = deepcopy(env.thick.obs_canonical())
            assert before['global']['ascension'] == 20
            if before['phase'] == 'combat':
                break
            phases.add(before['phase'])
            request = DecisionRequest(env, obs, info, 20200920, step)
            decision = strategy.decide([request])[0]
            assert env.action_masks()[decision.action]
            assert env.thick.obs_canonical() == before
            obs, _, _, _, info = env.step(decision.action)
        assert 'map' in phases
        assert before['phase'] == 'combat'
    finally:
        env.close()


def test_native_a20_shop_potion_purchase_has_real_inventory_and_gold_effect():
    env = SlayTheSpireEnv(ascension=20)
    try:
        env.reset(seed=42)
        gc = env.thick.gc
        x, y = next((x, y) for y in range(15) for x in range(7)
                    if env.thick.rmap.room_type(x, y) == 'SHOP')
        gc.act, gc.floor_num, gc.cur_map_node_y, gc.gold = 2, 18, y - 1, 200
        gc.transition_to_map_node(x)
        env._invalidate_backend_cache()
        state = deepcopy(env.thick.obs_canonical())
        assert state['phase'] == 'shop'
        actions = [A._SHOP_POTION + i for i in range(len(state['options']['shop_potions']))
                   if env.action_masks()[A._SHOP_POTION + i]] + [A._PROCEED]
        action = rank_survival(state, actions)[0]['action']
        kind, index = A.segment_of(action)
        assert kind == 'SHOP_POTION'
        potion, price = state['options']['shop_potions'][index]
        assert env.action_masks()[action]
        env.step(action)
        after = env.thick.obs_canonical()
        assert after['global']['gold'] == state['global']['gold'] - price
        assert potion in after['global']['potions']
    finally:
        env.close()


@pytest.mark.parametrize('strategy_type', [SurvivalUpperStrategy, GrowthUpperStrategy, ReadinessUpperStrategy])
def test_bounded_a20_joint_rule_smoke_uses_existing_runner(strategy_type):
    strategy = PhaseStrategy(strategy_type(LeftmostStrategy()),
                             {'combat': CombatRuleStrategy()})
    scenario = Scenario('wholerun', lambda: SlayTheSpireEnv(ascension=20))
    rows = list(evaluate(strategy, scenario, EvaluationConfig(1, 20200920, 1, 3000, 20)))
    assert rows[0]['result'] in ('win', 'death')
    assert rows[0][f'overworld_{strategy_type.profile.name}_decisions'] > 5
