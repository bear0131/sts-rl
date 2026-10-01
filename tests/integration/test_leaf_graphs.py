"""CUDA-specific graph contracts; CPU CI still checks the role lookup algebra."""
import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.search.inference import CombatLeafGraphs
from rl.search.native import forward_leaves

pytestmark = pytest.mark.native


@pytest.mark.skipif(not torch.cuda.is_available(), reason='BF16 reduction validation requires a GPU runner')
@pytest.mark.parametrize('backend', ['eager', 'graph'])
def test_bf16_leaf_inference_keeps_full_precision_reductions(combat_factory, backend):
    """A batch must match FP32-accumulating BF16, not truncated partial sums."""
    env = combat_factory(['STRIKE_RED'] * 5)
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, device='cuda', seed=1729,
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=256, n_layers=2, n_heads=8))
    model.amp_mode = 'bf16'
    model.policy.set_training_mode(False)
    obs = env._obs()
    batch = {k: np.stack([v] * 64) for k, v in obs.items()
             if k in model.policy.core.combat_observation_keys}
    original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        reference = forward_leaves(model, batch, autocast=model.autocast_context,
                                   leaf_backend=None)
        # Simulate the process default before entering the model-owned runtime.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        cache = (CombatLeafGraphs(model.policy.core, model.device, amp='bf16', batch_multiple=32)
                 if backend == 'graph' else None)
        actual = forward_leaves(model, batch, autocast=model.autocast_context,
                                leaf_backend=cache)
        for expected, received in zip(reference, actual):
            np.testing.assert_array_equal(expected, received)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original
        env.close()


def test_constant_role_lookup_preserves_values_and_gradients(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, device='cpu',
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4))
    encoder = model.policy.core.combat.encoder
    token = torch.randn(2, 3, 32)
    for role in encoder.role_index:
        actual = encoder._role(token, role)
        expected = token + encoder.role_embedding(torch.tensor(encoder.role_index[role])).view(1, 1, -1)
        assert torch.equal(actual, expected)
        a, = torch.autograd.grad(actual.square().sum(), encoder.role_embedding.weight)
        b, = torch.autograd.grad(expected.square().sum(), encoder.role_embedding.weight)
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph validation requires a GPU runner')
@pytest.mark.parametrize('amp', ['off', 'bf16'])
@pytest.mark.parametrize('batch_multiple', [0, 4])
@pytest.mark.parametrize('combat_arch', ['transformer', 'mlp'])
def test_graph_outputs_follow_inputs_weights_and_shapes(tmp_path, combat_factory, amp, batch_multiple, combat_arch):
    env = combat_factory(['STRIKE_RED'] * 5)
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, device='cuda', seed=7,
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4, combat_arch=combat_arch))
    model.amp_mode = amp
    model.policy.set_training_mode(False)
    cache = CombatLeafGraphs(model.policy.core, model.device, amp=amp, capacity=2,
                             batch_multiple=batch_multiple)
    obs = env._obs()
    def compare(count):
        batch = {k: np.stack([v] * count) for k, v in obs.items()
                 if k in model.policy.core.combat_observation_keys}
        model._native_leaf_inference = None
        eager = forward_leaves(model, batch, autocast=model.autocast_context)
        if batch_multiple:
            padded_count = (count + batch_multiple - 1) // batch_multiple * batch_multiple
            padded = {k:np.concatenate([v, np.repeat(v[:1], padded_count-count, axis=0)])
                      for k,v in batch.items()}
            reference = forward_leaves(model, padded, autocast=model.autocast_context)
            # GEMM selection can change with batch shape, even for independent
            # rows. Keep graph-vs-eager bitwise at the SAME shape; separately
            # bound cross-shape floating-point drift for this small fixture.
            for a,b in zip(eager, reference):
                np.testing.assert_allclose(a, b[:count], rtol=0.02 if amp=='bf16' else 1e-5,
                                           atol=2e-4 if amp=='bf16' else 1e-7)
            eager = tuple(v[:count] for v in reference)
        model._native_leaf_inference = cache
        graphed = forward_leaves(model, batch, autocast=model.autocast_context)
        for a, b in zip(eager, graphed):
            np.testing.assert_array_equal(a, b)
    compare(2)
    obs['player_scalars'] = obs['player_scalars'].copy() + 0.01
    compare(2)
    with torch.no_grad():
        if combat_arch == 'mlp':
            model.policy.core.combat.policy_head.bias.add_(0.125)
            model.policy.core.combat.trunk[0].weight.mul_(1.1)
        else:
            model.policy.core.combat.fixed_head.bias.add_(0.125)
            # Catch stale BF16 weight casts as well as stale inputs.
            model.policy.core.combat.encoder.global_projection.weight.mul_(1.1)
    compare(2)
    for size in (3, 4, 9, 5, 2):
        compare(size)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    compare(2)
    assert len(cache.entries) == 2
    assert cache.hits >= 2
    checkpoint = tmp_path / 'graph-runtime.zip'
    model.save(checkpoint)
    restored = AccumulatingMaskablePPO.load(checkpoint, device='cuda')
    assert getattr(restored, '_native_leaf_inference', None) is None
    model.policy.set_training_mode(True)
    with pytest.raises(RuntimeError, match='eval mode'):
        cache({}, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph validation requires a GPU runner')
def test_graph_cache_survives_many_capture_streams_and_evictions(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, device='cuda', seed=9,
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=256, n_layers=8, n_heads=8))
    model.amp_mode = 'bf16'
    model.policy.set_training_mode(False)
    cache = CombatLeafGraphs(model.policy.core, model.device, amp='bf16', capacity=32)
    obs = env._obs()
    for index in range(100):
        count = 1 + index % 24
        width = 1 + (index // 24) % min(4, obs['relics'].shape[0])
        batch = {k: np.stack([v] * count) for k,v in obs.items()
                 if k in model.policy.core.combat_observation_keys}
        batch['relics'].fill(0)
        batch['relics'][:, width-1] = 1
        model._native_leaf_inference = None
        expected = forward_leaves(model, batch, autocast=model.autocast_context)
        model._native_leaf_inference = cache
        actual = forward_leaves(model, batch, autocast=model.autocast_context)
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(a,b,err_msg=f'capture {index}, batch {count}, width {width}')
        # Keep one old graph alive while other captures cycle through CUDA's
        # stream pool; only replaying newly captured graphs misses this lifetime.
        anchor = {k:v[:1].copy() for k,v in batch.items()}
        anchor['relics'].fill(0)
        model._native_leaf_inference = None
        expected = forward_leaves(model, anchor, autocast=model.autocast_context)
        model._native_leaf_inference = cache
        actual = forward_leaves(model, anchor, autocast=model.autocast_context)
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(a,b,err_msg=f'old graph after capture {index}')
    assert cache.captures >= 96
    assert len(cache.entries) == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph validation requires a GPU runner')
def test_graph_owns_workspace_after_blas_cache_is_released(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, device='cuda', seed=9,
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=256, n_layers=8, n_heads=8))
    model.amp_mode = 'bf16'
    model.policy.set_training_mode(False)
    obs = env._obs()
    batch = {k:np.stack([obs[k]] * 17) for k in sorted(obs)
             if k in model.policy.core.combat_observation_keys}
    for key in ('monster_move', 'monster_intent_cat'):
        batch[key] = batch[key].astype(np.int32)
    batch['relics'].fill(0)
    batch['relics'][:, 4] = 1
    expected = forward_leaves(model, batch, autocast=model.autocast_context)
    model._native_leaf_inference = CombatLeafGraphs(model.policy.core, model.device, amp='bf16')
    forward_leaves(model, batch, autocast=model.autocast_context)
    # CUDA graph destruction clears stream-local cuBLAS workspaces. Releasing
    # eager workspaces must not invalidate any other cached graph's pointers.
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    actual = forward_leaves(model, batch, autocast=model.autocast_context)
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)
