"""Native public identifiers and bounded legal joint play, no checkpoint needed."""
import copy
from collections import Counter

import pytest

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


@pytest.mark.timeout(180)
def test_dynamic_event_public_details_cover_legal_buttons_without_fallback():
    target_events = {
        'Falling', 'Nloth', 'WeMeetAgain', 'Note For Yourself', 'Designer',
    }
    seen, fallbacks, misaligned = Counter(), Counter(), []

    def trace(row):
        if row['phase'] != 'event':
            return
        opts = row['state']['options']
        event = opts.get('event')
        if event not in target_events:
            return
        seen[event] += 1
        if 'fallback:' in row['source']:
            fallbacks[event] += 1
        legal = set(opts.get('event_options', ()))
        choices = opts.get('event_choices', ())
        detailed = {int(choice['index']) for choice in choices}
        if legal != detailed:
            misaligned.append((row['seed'], event, legal, detailed))
        global_state = row['state']['global']
        for choice in choices:
            if choice['kind'] == 'lose_card' and tuple(choice['card']) not in global_state['deck']:
                misaligned.append((row['seed'], event, 'card_not_in_deck', choice))
            elif choice['kind'] == 'lose_relic' and choice['relic'] not in global_state['relics']:
                misaligned.append((row['seed'], event, 'relic_not_held', choice))
            elif choice['kind'] == 'lose_potion' and choice['potion'] not in global_state['potions']:
                misaligned.append((row['seed'], event, 'potion_not_held', choice))
            elif choice['kind'] == 'lose_gold' and not 0 <= choice['gold'] <= global_state['gold']:
                misaligned.append((row['seed'], event, 'invalid_gold_cost', choice))

    upper = OverworldRuleStrategy(LeftmostStrategy(), trace=trace)
    strategy = PhaseStrategy(upper, {'combat': CombatRuleStrategy()})
    rows = list(evaluate(strategy, Scenario('wholerun', SlayTheSpireEnv),
                         EvaluationConfig(64, 10000, 8, 6000, 120)))
    assert len(rows) == 64
    assert set(seen) == target_events
    assert not fallbacks
    assert not misaligned
