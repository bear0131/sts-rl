import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.search.native import NativeSearchConfig, search
from rl.evaluation.strategies import DecisionRequest, NativeStrategy

pytestmark = pytest.mark.native


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
