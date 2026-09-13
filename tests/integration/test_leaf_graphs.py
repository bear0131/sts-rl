"""CUDA-specific graph contracts; CPU CI still checks the role lookup algebra."""
import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.search.inference import CombatLeafGraphs
from rl.search.native import forward_leaves

pytestmark = pytest.mark.native


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
def test_graph_outputs_follow_inputs_weights_and_shapes(tmp_path, combat_factory, amp):
    env = combat_factory(['STRIKE_RED'] * 5)
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, device='cuda', seed=7,
        n_steps=8, batch_size=4, policy_kwargs=dict(d_model=32, n_layers=2, n_heads=4))
    model.amp_mode = amp
    model.policy.set_training_mode(False)
    cache = CombatLeafGraphs(model.policy.core, model.device, amp=amp, capacity=2)
    obs = env._obs()
    def compare(count):
        batch = {k: np.stack([v] * count) for k, v in obs.items()
                 if k in model.policy.core.combat_observation_keys}
        model._native_leaf_inference = None
        eager = forward_leaves(model, batch, autocast=model.autocast_context)
        model._native_leaf_inference = cache
        graphed = forward_leaves(model, batch, autocast=model.autocast_context)
        for a, b in zip(eager, graphed):
            np.testing.assert_array_equal(a, b)
    compare(2)
    obs['player_scalars'] = obs['player_scalars'].copy() + 0.01
    compare(2)
    with torch.no_grad():
        model.policy.core.combat.fixed_head.bias.add_(0.125)
        # Catch stale BF16 weight casts as well as stale inputs.
        model.policy.core.combat.encoder.global_projection.weight.mul_(1.1)
    compare(2)
    for size in (3, 4, 2):
        compare(size)
    assert len(cache.entries) == 2
    assert cache.hits >= 2
    checkpoint = tmp_path / 'graph-runtime.zip'
    model.save(checkpoint)
    restored = AccumulatingMaskablePPO.load(checkpoint, device='cuda')
    assert getattr(restored, '_native_leaf_inference', None) is None
    model.policy.set_training_mode(True)
    with pytest.raises(RuntimeError, match='eval mode'):
        cache({}, 1)
