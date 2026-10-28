"""FP16 leaves have their own precision and retain graph/cache lifecycle rules."""
import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.entity_combat import EntityCombatPolicy
from rl.search.inference import CombatLeafGraphs

pytestmark = [pytest.mark.native, pytest.mark.timeout(240),
              pytest.mark.usefixtures('isolated_torch_compile_cache')]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires explicit GPU validation')
@pytest.mark.parametrize('compiled', [False, True])
def test_fp16_entity_inputs_weights_eviction_release_and_checkpoint(combat_factory, tmp_path, compiled):
    env = combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5,
                         combat_action_contract='autoregressive-v2', card_language=True)
    model = AccumulatingMaskablePPO(EntityCombatPolicy, env, device='cuda', seed=42,
        amp_mode='bf16', n_steps=8, batch_size=4, nosl_is_az=True,
        nosl_is_az_cpp=True, nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='rules',
        policy_kwargs=dict(entity_config=dict(width=32, layers=1, heads=4)))
    model.policy.set_training_mode(False)
    cache = CombatLeafGraphs(model.policy.core, 'cuda', amp='fp16', capacity=2,
                             batch_multiple=4, compile_forward=compiled)
    obs = env._obs()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')

    def compare(count):
        batch = {key: np.stack([value] * count) for key, value in obs.items()
                 if key in model.policy.core.combat_observation_keys}
        actual = cache(batch, 1)
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            logits, values = model.policy.core.combat_actor_critic(
                {key: torch.as_tensor(value, device='cuda') for key, value in batch.items()},
                active_relic_width=1, evaluate_subsets=False)
        for a, b in zip(actual, (logits.float().cpu().numpy(), values.float().flatten().cpu().numpy())):
            assert np.isfinite(a).all()
            np.testing.assert_allclose(a, b, atol=.001, rtol=.005)
        assert model.amp_mode == 'bf16'
        return actual

    try:
        before = compare(2)
        with torch.no_grad():
            model.policy.core.combat.end.bias.add_(.25)
            model.policy.core.combat.card.weight.add_(.1)
        after = compare(2)
        assert not np.array_equal(before[0], after[0])
        obs['player_scalars'] = obs['player_scalars'].copy() + .01
        for count in (3, 7, 9, 2):
            compare(count)
        assert cache.evictions > 0
        cache.clear()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        compare(2)
        model._native_leaf_inference = cache
        path = tmp_path / 'fp16-leaves.zip'
        model.save(path)
        restored = AccumulatingMaskablePPO.load(path, device='cuda')
        assert restored.amp_mode == 'bf16'
        assert getattr(restored, '_native_leaf_inference', None) is None
        model.policy.set_training_mode(True)
        with pytest.raises(RuntimeError, match='eval mode'):
            cache({key: np.stack([value]) for key, value in obs.items()}, 1)
    finally:
        cache.clear()
        torch.set_float32_matmul_precision(previous_precision)


def test_leaf_precision_override_is_independent_parser_option():
    from rl.training.distributed_sb3 import build_parser
    args = build_parser().parse_args(['--rank', '0', '--world-size', '1',
                                     '--leaf-inference', 'cuda-graph', '--amp-mode', 'bf16',
                                     '--leaf-amp-mode', 'fp16'])
    assert args.amp_mode == 'bf16' and args.leaf_amp_mode == 'fp16'
    assert build_parser().parse_args(['--rank', '0', '--world-size', '1']).leaf_amp_mode is None


def test_fp16_rejects_unsupported_core():
    with pytest.raises(ValueError, match='autoregressive entity'):
        CombatLeafGraphs(object(), 'cuda', amp='fp16')
