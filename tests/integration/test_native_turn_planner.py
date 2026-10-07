"""UPPER03: preserve the reference rule search when batching native workers."""
import numpy as np
import pytest
import slaythespire as sts

from env.search.native_batch import NativeSearchBatch
from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


def assert_plan_equal(actual, expected):
    assert actual.action_bits == expected.action_bits
    a = {k: v for k, v in actual.diagnostics.items() if k != 'search_seconds'}
    b = {k: v for k, v in expected.diagnostics.items() if k != 'search_seconds'}
    assert a == b


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('encounter', ['Jaw Worm', 'Three Sentries', 'Champ', 'Awakened One', 'Time Eater'])
@pytest.mark.parametrize('turns,profile', [(1, 'legacy-v8'), (2, 'time-clock-v9')])
def test_native_rule_search_matches_reference(combat_factory, threads, encounter, turns, profile):
    envs = [combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH', 'INFLAME', 'SHRUG_IT_OFF'] * 2,
                          encounter=encounter, seed=seed, combat_action_contract='autoregressive-v2')
            for seed in (42, 71)]
    planner = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=128,
        beam_width=8, max_depth=8, lookahead_turns=turns, leaf_profile=profile))
    battles = [e.thick.bc for e in envs]
    before = [b.to_dict() for b in battles]
    expected = [planner.choose(b, state_nonce=i, public_relics=()) for i, b in enumerate(battles)]
    pool = sts.NeuralSearchBatch(threads)
    for _ in range(2):
        actual = planner.choose_batch(battles, state_nonces=[0, 1], public_relics=[(), ()], pool=pool)
        for a, b in zip(actual, expected, strict=True):
            assert_plan_equal(a, b)
    assert [b.to_dict() for b in battles] == before


@pytest.mark.parametrize('card', ['PURITY', 'SECRET_TECHNIQUE'])
def test_native_rule_batch_keeps_selection_handles(combat_factory, card):
    env = combat_factory([card] + ['DEFEND_RED'] * 8, combat_action_contract='autoregressive-v2')
    env.thick.bc.apply_custom_state({'hand': [card] + ['DEFEND_RED'] * 3,
        'draw': ['DEFEND_RED'] * 14 + ['SHRUG_IT_OFF'], 'discard': []})
    env.thick.bc.execute(sts.Action(0))
    env._invalidate_backend_cache()
    planner = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=128, leaf_profile='legacy-v8'))
    expected = planner.choose(env.thick.bc, state_nonce=123, public_relics=())
    actual = planner.choose_batch([env.thick.bc], state_nonces=[123], public_relics=[()],
                                 pool=sts.NeuralSearchBatch(3))[0]
    assert_plan_equal(actual, expected)


def test_native_rule_worker_preserves_dome_and_action_execution(combat_factory):
    envs = [combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH'] * 3,
                relics=relics, combat_action_contract='autoregressive-v2')
            for relics in ((), ('Runic Dome',))]
    references = [e.fork_combat() for e in envs]
    calls = {i: ((), dict(state_nonce=100+i, worlds=2, node_budget=128,
                         leaf_profile='legacy-v8')) for i in range(2)}
    pool = NativeSearchBatch(envs, 4)
    try:
        for _ in range(3):
            expected = {i: e.rulebase_turn_decision(**calls[i][1]) for i, e in enumerate(references)}
            actual = pool.rulebase_turn_decisions(calls)
            for i in calls:
                assert actual[i]['action'] == expected[i]['action']
                assert actual[i]['native_action_bits'] == expected[i]['native_action_bits']
                envs[i].step(actual[i]['action'])
                references[i].step(expected[i]['action'])
                np.testing.assert_equal(envs[i]._obs(), references[i]._obs())
    finally:
        pool.close()
        for e in references:
            e.close()


@pytest.mark.parametrize('relics,powers,hand', [
    ((), [], ['FIEND_FIRE', 'STRIKE_RED', 'WOUND']),
    (('Bronze Scales',), [], ['DEFEND_RED', 'WOUND']),
    ((), [('COMBUST', 5), ('CORRUPTION', 1), ('BARRICADE', 1)], ['DEFEND_RED', 'WOUND']),
    ((), [('DEMON_FORM', 2), ('STRENGTH', -2), ('FEEL_NO_PAIN', 3)], ['DEFEND_RED', 'BURN']),
])
def test_native_rule_resource_and_power_scores(combat_factory, relics, powers, hand):
    env = combat_factory(['STRIKE_RED'] * 5, relics=relics)
    env.thick.bc.apply_custom_state({'hand': hand, 'draw': ['DEFEND_RED'],
        'discard': [], 'player': {'hp': 37, 'max_hp': 80, 'block': 17, 'powers': powers}})
    planner = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512, leaf_profile='legacy-v8'))
    expected = planner.choose(env.thick.bc, state_nonce=77, public_relics=relics)
    actual = planner.choose_batch([env.thick.bc], state_nonces=[77], public_relics=[relics],
                                 pool=sts.NeuralSearchBatch(4))[0]
    assert_plan_equal(actual, expected)


def test_native_rule_pool_recovers_after_worker_error(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    bad = env.thick.bc.clone()
    bad.execute_to_enemy_boundary(4 << 29)
    planner = NativeTurnPlanner(TurnPlannerConfig(worlds=1, node_budget=64))
    pool = sts.NeuralSearchBatch(4)
    with pytest.raises(ValueError, match='live combat decision'):
        planner.choose_batch([env.thick.bc, bad], state_nonces=[1, 2], public_relics=[(), ()], pool=pool)
    expected = planner.choose(env.thick.bc, state_nonce=3, public_relics=())
    actual = planner.choose_batch([env.thick.bc], state_nonces=[3], public_relics=[()], pool=pool)[0]
    assert_plan_equal(actual, expected)
