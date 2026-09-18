"""Native public-state identifiers, without whole-run evaluation or training."""
from copy import deepcopy

import pytest
import slaythespire as sts

from env.core.run_env import SlayTheSpireEnv
from env.core import action_space as A
from rl.evaluation.baselines.combat_rules import CombatRuleStrategy
from rl.evaluation.baselines.overworld_demand import (
    EnemyDemandUpperStrategy, demand_profile, hand_block_coverage,
)
from rl.evaluation.baselines.overworld_rules import OverworldRuleStrategy
from rl.evaluation.strategies import DecisionRequest


@pytest.mark.parametrize('boss,reference', [('AUTOMATON', 14), ('CHAMP', 16), ('COLLECTOR', 18)])
def test_native_boss_deck_and_relics_feed_public_profile(boss, reference):
    env = SlayTheSpireEnv()
    try:
        env.reset(seed=10000)
        gc = env.thick.gc
        gc.act = 2
        gc.boss = getattr(sts.MonsterEncounter, boss)
        gc.set_deck([('SHRUG_IT_OFF', 0)] + [('STRIKE_RED', 0)] * 5)
        gc.set_relics(['SlaversCollar', 'Oddly Smooth Stone'])
        env._invalidate_backend_cache()
        public = env.thick.obs_canonical()
        snapshot = deepcopy(public)
        profile = demand_profile(public)
        assert profile['boss'] == boss
        assert profile['reference'] == reference
        assert profile['energy_budget'] == 3
        assert profile['dexterity'] == 1
        assert hand_block_coverage(tuple(public['global']['deck']), 3, reference, 1) == pytest.approx(9 * 5 / 6)
        assert env.thick.obs_canonical() == snapshot
    finally:
        env.close()


def test_real_neow_decision_delegates_without_changing_default_or_native_state():
    env = SlayTheSpireEnv()
    try:
        observation, info = env.reset(seed=10000)
        before = deepcopy(env.thick.obs_canonical())
        request = DecisionRequest(env, observation, info, 10000, 0)
        baseline = OverworldRuleStrategy().decide([request])[0]
        candidate = EnemyDemandUpperStrategy().decide([request])[0]
        assert candidate == baseline
        assert env.action_masks()[candidate.action]
        assert env.thick.obs_canonical() == before
        env.step(candidate.action)
    finally:
        env.close()


def test_native_reward_choice_is_legal_and_changes_the_real_deck(short_combat_env_type):
    # Only the first bounded fight and reward, not a whole-run experiment.
    env = short_combat_env_type()
    try:
        observation, info = env.reset(seed=10000)
        combat = CombatRuleStrategy()
        for step in range(16):
            if env.thick.phase() != 'combat':
                break
            action = combat.decide([DecisionRequest(env, observation, info, 10000, step)])[0].action
            observation, _, _, _, info = env.step(action)
        assert env.thick.phase() == 'combat_reward'
        env.thick.gc.act = 2
        env.thick.gc.boss = sts.MonsterEncounter.AUTOMATON
        env._invalidate_backend_cache()
        trace = []
        strategy = EnemyDemandUpperStrategy(trace=trace.append)
        for step in range(12):
            before = deepcopy(env.thick.obs_canonical())
            request = DecisionRequest(env, env._obs(), env._info(), 10000, step)
            decision = strategy.decide([request])[0]
            assert env.action_masks()[decision.action]
            assert env.thick.obs_canonical() == before
            kind, index = A.segment_of(decision.action)
            env.step(decision.action)
            if kind in ('REWARD_CARD', 'REWARD_SKIP'):
                after_deck = env.thick.obs_canonical()['global']['deck']
                if kind == 'REWARD_CARD':
                    assert len(after_deck) == len(before['global']['deck']) + 1
                    assert after_deck[-1] == before['options']['cards'][index]
                else:
                    assert after_deck == before['global']['deck']
                assert any(t['version'] == strategy.version for t in trace)
                break
        else:
            pytest.fail('bounded native reward fixture did not resolve a card choice')
    finally:
        env.close()


def test_native_true_grit_consumption_explains_why_it_is_not_flat_block(combat_factory):
    env = combat_factory(['TRUE_GRIT'] * 5)
    env.thick.bc.set_player_energy(4)
    env._invalidate_backend_cache()
    for remaining in (3, 1, 0):
        assert env.action_masks()[A._PLAY]
        env.step(A._PLAY)
        if env._info()['pending']:
            env.step(next(i for i, legal in enumerate(env.action_masks()) if legal))
        assert len(env.thick.obs_canonical()['hand']) == remaining
    native = env.thick.bc.to_dict()
    assert native['player']['block'] == 21
    assert native['player']['energy'] == 1
    assert not env.action_masks()[A._PLAY]
