"""Contracts for arbitrary future strategies, batching and run ownership."""
import numpy as np
import pytest

from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate, EvaluationCancelled, summarize
from rl.evaluation.strategies import Decision, PhaseStrategy
from rl.search.native import decode_leaves


class Game:
    def __init__(self):
        self.unwrapped = self
        self.closed = False
        self.resets = []

    def reset(self, seed):
        self.resets.append(seed)
        self.seed, self.steps = seed, 0
        return {'seed': np.array([seed])}, {'phase': 'custom'}

    def action_masks(self):
        return np.array([True, True])

    def step(self, action):
        assert action == self.seed % 2
        self.steps += 1
        return {'seed': np.array([self.seed])}, 1, self.steps >= 3, False, {'phase': 'custom'}

    def close(self):
        self.closed = True


class FutureStrategy:
    name = 'new-strategy-with-no-SB3-or-native-dependency'

    def __init__(self):
        self.batch_sizes = []

    def decide(self, requests, *, check):
        check()
        self.batch_sizes.append(len(requests))
        return [Decision(r.seed % 2, self.name) for r in requests]


def scenario(created):
    def make():
        env = Game()
        created.append(env)
        return env
    return Scenario('custom-game', make,
                    lambda env, info, terminated, truncated: dict(result='finished', floor=3, act=1))


def test_new_strategy_batches_reuses_environments_and_preserves_seeds():
    envs = []
    strategy = PhaseStrategy(FutureStrategy(), {'custom': FutureStrategy()})
    rows = list(evaluate(strategy, scenario(envs), EvaluationConfig(episodes=5, seed=10, num_envs=2)))
    assert len(envs) == 2 and all(e.closed for e in envs)
    assert sorted(seed for e in envs for seed in e.resets) == list(range(10, 15))
    assert [r['seed'] for r in sorted(rows, key=lambda r: r['episode'])] == list(range(10, 15))
    assert all(r['steps'] == 3 and r['raw_return'] == 3 for r in rows)
    assert strategy.routes['custom'].batch_sizes == [2] * 6 + [1] * 3


def test_step_limits_are_not_deaths():
    envs = []
    rows = list(evaluate(FutureStrategy(), scenario(envs), EvaluationConfig(episodes=1, max_steps=1)))
    assert rows[0]['result'] == 'step_limit'
    assert summarize(rows)['death_floors'] == {}
    assert summarize(rows)['win_rate'] == 0.0
    assert summarize(rows)['act1_rate'] == 1.0


def test_complete_native_decisions_do_not_use_thin_masks():
    class NativeGame(Game):
        def action_masks(self):
            raise AssertionError('complete actions must not query a thin mask')

        def step_native_combat(self, bits):
            assert bits==1073741835  # single select, native index11
            return {'seed':np.array([self.seed])},0,True,False,{'combat_action_contract':'native-v1'}

    class NativeFuture(FutureStrategy):
        def decide(self,requests,*,check):
            return [Decision(None,self.name,native_action=1073741835) for _ in requests]

    rows=list(evaluate(NativeFuture(),Scenario('native',NativeGame,
        lambda *args:dict(result='finished',floor=1,act=1)),EvaluationConfig(episodes=1)))
    assert rows[0]['combat_action_contract']=='native-v1'


@pytest.mark.parametrize('action,native',[(None,None),(1,1)])
def test_decision_action_contract_is_unambiguous(action,native):
    with pytest.raises(ValueError,match='exactly one'):
        Decision(action,'custom',native_action=native)


def test_complete_native_decisions_do_not_use_thin_masks():
    class NativeGame(Game):
        def action_masks(self):
            raise AssertionError('complete actions must not query a thin mask')

        def step_native_combat(self, bits):
            assert bits==1073741835  # single select, native index11
            return {'seed':np.array([self.seed])},0,True,False,{'combat_action_contract':'native-v1'}

    class NativeFuture(FutureStrategy):
        def decide(self,requests,*,check):
            return [Decision(None,self.name,native_action=1073741835) for _ in requests]

    rows=list(evaluate(NativeFuture(),Scenario('native',NativeGame,
        lambda *args:dict(result='finished',floor=1,act=1)),EvaluationConfig(episodes=1)))
    assert rows[0]['combat_action_contract']=='native-v1'


@pytest.mark.parametrize('action,native',[(None,None),(1,1)])
def test_decision_action_contract_is_unambiguous(action,native):
    with pytest.raises(ValueError,match='exactly one'):
        Decision(action,'custom',native_action=native)


@pytest.mark.parametrize('reason', ['cancel', 'strategy', 'invalid-action'])
def test_failures_close_every_owned_environment(reason):
    envs = []
    class Failing(FutureStrategy):
        def decide(self, requests, *, check):
            if reason == 'strategy':
                raise RuntimeError('broken policy')
            if reason == 'invalid-action':
                return [Decision(99, self.name) for _ in requests]
            return super().decide(requests, check=check)
    with pytest.raises((RuntimeError, EvaluationCancelled, ValueError)):
        list(evaluate(Failing(), scenario(envs), EvaluationConfig(episodes=2, num_envs=2),
                      cancelled=lambda: reason == 'cancel' and len(envs) == 2))
    assert envs and all(e.closed for e in envs)


def test_mixed_native_transports_keep_all_workers_and_request_order():
    records, observations = decode_leaves({
        2: {'leaves': [{'request_id': 22, 'observation': {'x': np.array([20])}}]},
        0: {'leaf_meta': [{'request_id': 1}, {'request_id': 2}],
            'observation_batch': {'x': np.array([[1], [2]])}},
    })
    assert [(i, meta['request_id']) for i, meta in records] == [(0, 1), (0, 2), (2, 22)]
    assert observations['x'].flatten().tolist() == [1, 2, 20]


def test_native_transport_rejects_missing_batch_rows():
    with pytest.raises(ValueError, match='batch size'):
        decode_leaves({0: {'leaf_meta': [{'request_id': 1}], 'observation_batch': {'x': np.zeros((2, 3))}}})
