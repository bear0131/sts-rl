"""Native public identifiers and bounded legal joint play, no checkpoint needed."""
import copy

from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.baselines.combat_rules import CombatRuleStrategy
from rl.evaluation.baselines.overworld_rules import OverworldRuleStrategy
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate
from rl.evaluation.strategies import DecisionRequest, LeftmostStrategy, PhaseStrategy


def test_native_neow_is_public_deterministic_and_does_not_mutate_state():
    env = SlayTheSpireEnv()
    try:
        obs, info = env.reset(seed=10000)
        before = copy.deepcopy(env.thick.obs_canonical())
        assert before['global']['potions'] == ['EMPTY_POTION_SLOT'] * 3
        request = DecisionRequest(env, obs, info, 10000, 0)
        strategy = OverworldRuleStrategy()
        first = strategy.decide([request])[0]
        assert env.overworld_rule_decision() == first.action
        assert first == strategy.decide([request])[0]
        assert env.action_masks()[first.action]
        assert env.thick.obs_canonical() == before
        env.step(first.action)
    finally:
        env.close()


def test_bounded_joint_rules_finish_development_seed_with_legal_actions():
    upper = OverworldRuleStrategy(LeftmostStrategy())
    strategy = PhaseStrategy(upper, {'combat': CombatRuleStrategy()})
    rows = list(evaluate(strategy, Scenario('wholerun', SlayTheSpireEnv),
                        EvaluationConfig(1, 10000, 1, 6000, 30)))
    assert rows[0]['result'] in ('win', 'death')
    assert rows[0]['overworld_rule_decisions'] > 20
    assert rows[0]['rule_decisions'] > 20
