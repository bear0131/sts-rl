"""The virtual benchmark must remove inference without changing the tree."""
import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.benchmarks.virtual_inference import VirtualInference
from rl.policies.entity_combat import EntityCombatPolicy
from rl.search.native import NativeSearchConfig, search

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


def test_virtual_and_shadow_search_have_identical_work(combat_factory):
    results = []
    for shadow in (False, True):
        env = combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5,
            combat_action_contract='autoregressive-v2', card_language=True)
        model = AccumulatingMaskablePPO(EntityCombatPolicy, env, device='cpu', seed=1729,
            n_steps=8, batch_size=4,
            policy_kwargs=dict(entity_config=dict(width=32, layers=1, heads=4)),
            nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
            nosl_is_az_upper_strategy='rules')
        model.policy.set_training_mode(False)
        if not shadow:
            def forbidden(*args, **kwargs):
                raise AssertionError('virtual workload executed the model')
            model.policy.core.combat_actor_critic = forbidden
        runtime = VirtualInference(model, shadow=shadow).install()
        result = search(model, env, env._obs(), NativeSearchConfig(
            simulations=128, depth=0, leaf_batch_size=1, root_search_mode=2,
            paper_full_gumbel=True, fixed_worldline=True, gumbel_rescale_q=False,
            gumbel_value_scale=1.), nonce=1729)
        assert runtime.leaf_rows > 0 and runtime.root_calls == 1
        assert result['simulations'] == 128
        results.append((result, runtime.statistics()))
    first, second = results
    for field in ('action', 'native_action_bits', 'root_value', 'simulations',
                  'visit_policy', 'root_visit_policy', 'root_visits'):
        np.testing.assert_array_equal(first[0][field], second[0][field])
    assert first[1]['leaf_rows'] == second[1]['leaf_rows']
    assert first[1]['leaf_calls'] == second[1]['leaf_calls']
    assert first[1]['model_seconds'] == 0
    assert second[1]['model_seconds'] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='virtual CUDA transport requires a GPU runner')
def test_virtual_graph_preserves_transfers_without_model(combat_factory):
    from rl.search.inference import CombatLeafGraphs
    from rl.search.native import forward_leaves
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='autoregressive-v2', card_language=True)
    model = AccumulatingMaskablePPO(EntityCombatPolicy, env, device='cuda', seed=1729,
        n_steps=8, batch_size=4, policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4)),
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='rules')
    model.policy.set_training_mode(False)
    def forbidden(*args, **kwargs):
        raise AssertionError('virtual graph executed the real model')
    model.policy.core.combat_actor_critic = forbidden
    model.policy.core.graph_actor_critic = forbidden
    model._native_leaf_inference = CombatLeafGraphs(model.policy.core, model.device,
        amp='bf16', batch_multiple=32, profile_interval=1)
    runtime = VirtualInference(model).install()
    batch = {k: np.stack([v]*17) for k,v in env._obs().items()
             if k in model.policy.core.combat_observation_keys}
    for _ in range(2):
        logits, values = forward_leaves(model, batch)
        np.testing.assert_array_equal(logits, np.zeros((17, model.action_space.n)))
        np.testing.assert_array_equal(values, np.full(17, .5))
    stats = runtime.statistics()
    assert stats['model_seconds'] == 0
    assert stats['graphs']['captures'] == 1 and stats['graphs']['hits'] == 1
    assert stats['graphs']['profile_samples'] == 2
    assert stats['graphs']['h2d_ms'] > 0 and stats['graphs']['d2h_ms'] > 0
