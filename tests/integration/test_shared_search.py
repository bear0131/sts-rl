from collections import defaultdict, deque
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.search.native import (
    LeafInferenceSchedule,
    NativeSearchConfig,
    complete_searches,
    complete_searches_streaming,
    forward_root,
    search,
    supports_compact,
)
from rl.evaluation.strategies import DecisionRequest, NativeStrategy

pytestmark = pytest.mark.native


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph search requires a GPU runner')
def test_eval_graph_preserves_strict_native_complete_actions(combat_factory, request):
    from rl.evaluation.eval import build_parser
    from rl.evaluation.strategies import native_strategy, strategy_manifest

    previous_precision = torch.get_float32_matmul_precision()
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(previous_precision))
    torch.set_float32_matmul_precision('highest')
    envs = [combat_factory(['BASH'] + ['STRIKE_RED'] * 4, seed=seed) for seed in (42, 43)]
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, envs[0], device='cuda', seed=7,
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4))
    model.policy.set_training_mode(False)
    weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
    parsed = build_parser().parse_args(['unused.zip', '--solver', 'native', '--simulations', '128',
                                      '--depth', '0', '--leaf-batch-size', '1'])
    requests = [DecisionRequest(env, env._obs(), env._info(), seed, 0)
                for env, seed in zip(envs, (42, 43))]
    eager = native_strategy(model, parsed).decide(requests)
    eager_actions = [env._nosl_cpp_pending for env in envs]
    # Complete native actions may be staged on the wrapper; abort restores the
    # untouched game roots before comparing the second backend.
    for env in envs:
        env.nosl_is_az_cpp_abort()
    parsed.leaf_inference = 'cuda-graph'
    strategy = native_strategy(model, parsed)
    graphed = strategy.decide(requests)
    assert [d.action for d in graphed] == [d.action for d in eager]
    assert [env._nosl_cpp_pending for env in envs] == eager_actions
    for left, right in zip(eager, graphed):
        assert {k:v for k,v in left.diagnostics.items() if k != 'search_seconds'} == {
            k:v for k,v in right.diagnostics.items() if k != 'search_seconds'}
    assert all(torch.equal(value, weights[key]) for key, value in model.policy.state_dict().items())
    assert strategy_manifest(strategy)['phases']['combat']['leaf_inference']['statistics']['captures'] > 0
    graph_cache = strategy.routes['combat'].leaf_backend
    snapshot = graph_cache.statistics()
    parsed.leaf_inference = 'eager'
    eager_after_graph = native_strategy(model, parsed)
    parsed.leaf_inference = 'cuda-graph'
    second_graph = native_strategy(model, parsed)
    assert second_graph.routes['combat'].leaf_backend is not graph_cache
    for alternate in (eager_after_graph, second_graph, eager_after_graph):
        for env in envs:
            env.nosl_is_az_cpp_abort()
        decisions = alternate.decide(requests)
        assert [d.action for d in decisions] == [d.action for d in eager]
        assert [env._nosl_cpp_pending for env in envs] == eager_actions
    assert graph_cache.statistics() == snapshot
    assert not hasattr(model, '_native_leaf_inference')
    assert strategy_manifest(eager_after_graph)['phases']['combat']['leaf_inference'] == {'backend': 'eager'}


class _LocalGroupedMethodStream:
    """Exercise the streamed scheduler against real env methods in one process."""

    def __init__(self, envs, envs_per_group=2):
        self.envs = envs
        self.envs_per_group = envs_per_group
        self._pending = {}
        self._ready = deque()

    @property
    def pending(self):
        return bool(self._pending)

    def submit(self, method_name, calls):
        grouped = defaultdict(dict)
        for index, call in calls.items():
            grouped[int(index) // self.envs_per_group][int(index)] = call
        for group, group_calls in grouped.items():
            assert group not in self._pending
            responses = {
                index: getattr(self.envs[index], method_name)(*args, **kwargs)
                for index, (args, kwargs) in group_calls.items()
            }
            self._pending[group] = SimpleNamespace(
                group_index=group,
                method_name=method_name,
                responses=responses,
            )
            self._ready.append(group)

    def receive(self, timeout=None):
        assert self._ready
        group = self._ready.popleft()
        return [self._pending.pop(group)]


@pytest.mark.parametrize('mode,strict', [(0, False), (2, True)])
def test_resident_batch_and_single_tree_use_same_search_contract(combat_factory, mode, strict):
    envs = [combat_factory(['STRIKE_RED'] * 5, seed=seed) for seed in (42, 43)]
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, envs[0], device='cpu', seed=7,
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4))
    model.policy.set_training_mode(False)
    weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
    config = NativeSearchConfig(simulations=4, root_search_mode=mode, paper_full_gumbel=strict)
    expected = [search(model, env, env._obs(), config, nonce=seed)['action']
                for env, seed in zip(envs, (42, 43))]
    # Search may stage a complete native action. Abort that staging before
    # testing the second driver against the same untouched game roots.
    for env in envs:
        env.nosl_is_az_cpp_abort()
    strategy = NativeStrategy(model, config)
    decisions = strategy.decide([DecisionRequest(env, env._obs(), env._info(), seed, 0)
                                 for env, seed in zip(envs, (42, 43))])
    assert [d.action for d in decisions] == expected
    assert all(d.diagnostics['search_leaf_evaluations'] > 0 for d in decisions)
    assert all(torch.equal(value, weights[key]) for key, value in model.policy.state_dict().items())


def test_cancelled_native_search_discards_pending_session(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, device='cpu', seed=7,
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4))
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError('budget')
    with pytest.raises(TimeoutError, match='budget'):
        search(model, env, env._obs(), NativeSearchConfig(simulations=4), nonce=42, check=cancelled)
    assert env._nosl_cpp_az_session is None
    assert env._nosl_cpp_pending is None


def test_streamed_groups_match_synchronous_native_search(combat_factory):
    synchronous_envs = [
        combat_factory(['STRIKE_RED'] * 5, 'Two Louse', seed=seed)
        for seed in range(70, 74)
    ]
    streamed_envs = [
        combat_factory(['STRIKE_RED'] * 5, 'Two Louse', seed=seed)
        for seed in range(70, 74)
    ]
    model = AccumulatingMaskablePPO(
        SplitStSPointerPolicy,
        synchronous_envs[0],
        device='cpu',
        seed=11,
        n_steps=8,
        batch_size=4,
        policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4),
    )
    model.policy.set_training_mode(False)
    compact = supports_compact(model)
    observations = {
        key: np.stack([env._obs()[key] for env in synchronous_envs])
        for key in synchronous_envs[0]._obs()
    }
    logits, values = forward_root(model, observations)
    config = NativeSearchConfig(
        simulations=16,
        depth=0,
        root_search_mode=2,
        paper_full_gumbel=True,
    )

    def calls_for(envs):
        return {
            index: ((logits[index],), {
                **asdict(config),
                'root_value': float(values[index]),
                'state_nonce': 900 + index,
                'compact_observation': compact,
            })
            for index, _env in enumerate(envs)
        }

    sync_calls = calls_for(synchronous_envs)
    sync_responses = {
        index: env.nosl_is_az_cpp_begin(*args, **kwargs)
        for index, (env, (args, kwargs)) in enumerate(
            zip(synchronous_envs, sync_calls.values(), strict=True)
        )
    }
    sync_results, sync_stats = complete_searches(
        model,
        lambda method, calls: {
            index: getattr(synchronous_envs[index], method)(*args, **kwargs)
            for index, (args, kwargs) in calls.items()
        },
        sync_responses,
        leaf_batch_size=1,
        compact=compact,
    )
    streamed_results, streamed_stats = complete_searches_streaming(
        model,
        _LocalGroupedMethodStream(streamed_envs),
        calls_for(streamed_envs),
        leaf_batch_size=1,
        schedule=LeafInferenceSchedule(
            target_batch_size=2,
            max_wait_seconds=0.0,
        ),
        compact=compact,
    )

    assert sync_stats['leaf_evaluations'] == streamed_stats['leaf_evaluations']
    for index in range(4):
        assert (
            streamed_results[index]['native_action_bits']
            == sync_results[index]['native_action_bits']
        )
        np.testing.assert_array_equal(
            streamed_results[index]['root_visits'],
            sync_results[index]['root_visits'],
        )
        np.testing.assert_allclose(
            streamed_results[index]['visit_policy'],
            sync_results[index]['visit_policy'],
            rtol=0,
            # Dynamic batch shapes select different CPU GEMM kernels. The
            # discrete search path above must match exactly; completed-policy
            # floats only need to cover the observed accumulation rounding.
            atol=5e-4,
        )
        assert streamed_results[index]['root_value'] == pytest.approx(
            sync_results[index]['root_value'], rel=0, abs=5e-4
        )
