"""Frozen evaluation dispatch and paired resource response contracts."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from rl.evaluation.resource_response import audit
from rl.evaluation.strategies import DecisionRequest, policy_strategy, compose_strategy
from rl.search.native import observation_as_tensor

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


def model_for(core, env):
    return SimpleNamespace(policy=SimpleNamespace(core=core,
        obs_to_tensor=lambda o: (observation_as_tensor(o, 'cpu'), True),
        set_training_mode=core.train),
        device='cpu', observation_space=env.observation_space, action_space=env.action_space,
        nosl_is_az_fixed_worldline=True, nosl_is_az_gumbel_scale=0.)


def test_frozen_upper_evaluation_ignores_lower_stochastic_flag(outer_return_core):
    core, _, env = outer_return_core
    model = model_for(core, env)
    strategy = policy_strategy(model, SimpleNamespace(stochastic=True, leaf_amp='off'))
    obs, info = env.reset(seed=42101)
    expected, _ = core.teacher.decide(observation_as_tensor({k:v[None] for k,v in obs.items()}, 'cpu'),
                                     env.action_masks()[None])
    requests = [DecisionRequest(env, obs, info, 42101, 0)]
    assert all(strategy.decide(requests)[0].action == int(expected[0]) for _ in range(3))


def test_response_audit_reports_axes_without_forcing_monotonicity(outer_return_core):
    core, _, env = outer_return_core
    result = audit(model_for(core, env), env, seed=42101, max_steps=256, timeout=90, simulations=4)
    assert {r['axis'] for r in result['combat']} == {'hp', 'potions'}
    assert {r['axis'] for r in result['upper']} == {'hp', 'potions', 'wallet_gold', 'reward_gold'}
    assert all(np.isfinite(r['upper_value']) for r in result['upper'])
    assert all(np.isfinite(r['lower_value']) and np.isfinite(r['search_value']) for r in result['combat'])
    assert result['search_simulations'] == 4
    assert result['teacher_sha256'] == core.teacher.assert_integrity()


@pytest.mark.parametrize('override', [dict(upper_strategy='rules'), dict(route='min-elite-max-rest'),
                                  dict(solver='value'), dict(solver='leftmost'),
                                  dict(solver='native', native_search_threads=0)])
def test_frozen_evaluation_rejects_incomplete_or_changed_contract(outer_return_core, override):
    core, _, env = outer_return_core
    args = SimpleNamespace(solver='policy', stochastic=False, upper_strategy='model', route='model')
    vars(args).update(override)
    with pytest.raises(ValueError, match='frozen'):
        compose_strategy(model_for(core, env), args)


def test_outer_checkpoint_relocation_validates_material_and_context(outer_material, tmp_path, monkeypatch):
    import shutil
    from env.core.run_env import SlayTheSpireEnv
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.updates import optimizer_step
    from rl.policies.outer_return import OuterReturnPolicy
    from rl.policies.checkpoint import load_checkpoint
    from rl.training.native_vec_env import NativeSearchVecEnv
    env = SlayTheSpireEnv(global_map_view=True, combat_action_contract='autoregressive-v2', outer_return_context=True)
    vector = NativeSearchVecEnv([lambda: env], threads=1)
    model = AccumulatingMaskablePPO(OuterReturnPolicy, vector, policy_kwargs=outer_material,
        n_steps=2, batch_size=2, device='cpu', gamma=1., nosl_is_az=True, nosl_is_az_cpp=True,
        nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='model', nosl_is_az_fixed_worldline=True,
        nosl_is_az_leaf_batch_size=1)
    path = tmp_path/'model.zip'
    model.save(path)
    relocated = tmp_path/'moved'
    shutil.copytree(outer_material['material_directory'], relocated)
    monkeypatch.setenv('STS_OUTER_MATERIAL_DIR', str(relocated))
    restored = load_checkpoint(path, device='cpu', inference_only=True)
    assert restored._outer_return
    assert restored.policy.core.teacher.assert_integrity() == model.policy.core.teacher.assert_integrity()
    obs, _ = env.reset(seed=42101)
    restored.predict(obs, deterministic=True, action_masks=env.action_masks())
    obs.pop('outer_deck_misc')
    with pytest.raises(ValueError, match='context'):
        restored.predict(obs, deterministic=True, action_masks=env.action_masks())
    with pytest.raises(ValueError, match='entire lower'):
        optimizer_step(restored, value_only=True)
    (relocated/'upper.pt').write_bytes(b'invalid')
    with pytest.raises(ValueError, match='hash|SHA|digest'):
        load_checkpoint(path, device='cpu', inference_only=True)
    vector.close()


@pytest.mark.timeout(600)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU-only graph contract; deployment separately requires CUDA')
@pytest.mark.parametrize('control', [False, True, 'finite', 'means', 'conditional', 'matched', 'unscaled'])
def test_outer_graph_lifecycle(outer_return_core, finite_curve_bank, six_curve_bank, control):
    """Explicit GPU gate: missing CUDA is a failure when this test is selected."""
    import gc
    import os
    import torch
    from rl.search.inference import CombatLeafGraphs
    from rl.policies.precision import bf16_autocast
    assert torch.cuda.is_available(), 'select this acceptance test on the GPU host'
    core, source, env = outer_return_core
    if control:
        from rl.policies.outer_return import OuterReturnCore
        from env.core.finite_curves import OBJECTIVE, FACTORED_OBJECTIVE, MATCHED_OBJECTIVE, UNSCALED_OBJECTIVE, SIX_RAW_PAYOFF
        if control == 'unscaled':
            import copy
            finite_curve_bank = copy.deepcopy(six_curve_bank)
            finite_curve_bank['payoff'] = SIX_RAW_PAYOFF
            finite_curve_bank['curves'] = {k: [v / 1.6 for v in values]
                                           for k, values in finite_curve_bank['curves'].items()}
        core = OuterReturnCore(source, core.teacher.observation_space, env.observation_space,
            source_sha256='1'*64,
            objective=UNSCALED_OBJECTIVE if control == 'unscaled' else MATCHED_OBJECTIVE if control == 'matched' else FACTORED_OBJECTIVE if control == 'conditional' else OBJECTIVE if control == 'finite' else (
                'combat-outcome-means-v1' if control == 'means' else 'combat-outcome-quantiles-v1'),
            finite_curve_bank=finite_curve_bank if control in ('finite', 'conditional', 'matched', 'unscaled') else None)
    for _ in range(64):
        if env._phase() == 'combat':
            break
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert env._phase() == 'combat'
    core.to('cuda').eval()
    frozen = core.teacher.assert_integrity()
    cache = CombatLeafGraphs(core, 'cuda', amp='bf16', capacity=2, batch_multiple=32,
                             compile_forward=os.name != 'nt')
    try:
        for count in (1, 33, 65, 1, 1):
            obs = {k: np.stack([v]*count) for k, v in env._obs().items()
                   if k in core.combat_observation_keys}
            tensor = observation_as_tensor(obs, 'cuda')
            with torch.inference_mode(), bf16_autocast():
                reference = core.combat_actor_critic(tensor, active_relic_width=1)
            logits, values = cache(obs, 1)
            # BF16 batch padding / compiled fusion can round at 1/128 scale.
            np.testing.assert_allclose(values, reference[1].float().cpu().numpy().flatten(), atol=.02, rtol=.02)
            assert np.isfinite(logits).all()
            reference_logits = reference[0].float().cpu().numpy()
            finite_actions = np.abs(reference_logits) < 1e6
            np.testing.assert_allclose(logits[finite_actions], reference_logits[finite_actions], atol=.02, rtol=.02)
            del tensor, reference, obs
            gc.collect()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        assert cache.statistics()['evictions'] >= 2 and cache.statistics()['hits'] >= 1
        before = values.copy()
        with torch.no_grad():
            if control is True:
                core.combat.distribution.win.bias.add_(1.)
            elif control in ('means', 'matched'):
                core.combat.means.win.bias.add_(1.)
            elif control in ('conditional', 'unscaled'):
                core.combat.conditional.value[-1].bias.add_(1.)
            else:
                core.combat.scalar[-1].bias.add_(1.)
        obs = {k: v[None] for k, v in env._obs().items() if k in core.combat_observation_keys}
        _, changed = cache(obs, 1)
        if control is True or control in ('means', 'conditional', 'matched', 'unscaled'):
            assert np.all(changed > before)
            with torch.inference_mode(), bf16_autocast():
                expected = core.combat_actor_critic(observation_as_tensor(obs, 'cuda'), active_relic_width=1)[1]
            np.testing.assert_allclose(changed, expected.float().cpu().numpy().flatten(), atol=.02, rtol=.02)
        else:
            np.testing.assert_allclose(changed-before, 1., atol=.002, rtol=0)
        assert core.teacher.assert_integrity() == frozen
    finally:
        cache.clear()
        core.to('cpu')
        torch.cuda.empty_cache()


@pytest.mark.parametrize('matmul_precision', ['highest', 'high'])
def test_upper_rollout_graph_eviction_release_and_exact_cache(outer_return_core, matmul_precision):
    import gc
    from types import SimpleNamespace
    import torch
    from rl.search.rollout_inference import RolloutInference
    assert torch.cuda.is_available(), 'select this acceptance test on the GPU host'
    core, _, env = outer_return_core
    assert env._phase() != 'combat'
    core.to('cuda').eval()
    frozen = core.teacher.assert_integrity()
    model = SimpleNamespace(policy=SimpleNamespace(core=core), device='cuda',
                            _native_leaf_inference=object())
    runtime = RolloutInference(model, upper_bytes=0)
    runtime.upper_graphs.capacity = 2
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(matmul_precision)
    try:
        for count, deck_width, relic_width in ((1, 1, 1), (33, 17, 5), (65, 33, 9),
                                              (1, 65, 17), (1, 1, 1), (1, 1, 1)):
            obs = {k: np.stack([v]*count) for k, v in env._obs().items()}
            card, relic = obs['deck_ids'][0, 0], obs['relics'][0, 0]
            assert card and relic
            for key in ('deck_ids', 'deck_upgrades', 'relics', 'relic_counters'):
                obs[key][:] = 0
            obs['deck_ids'][:, :deck_width] = card
            obs['relics'][:, :relic_width] = relic
            # Distinct actual inputs prevent batch dedup hiding shape changes.
            obs['run_hp'][:, 0] += np.arange(count, dtype=np.float32)*.001
            reference = core.teacher(observation_as_tensor(obs, 'cuda'))
            logits, values = runtime.forward_upper(obs)
            # Explicit full-FP32 gate is independent of earlier trainer tests.
            # Production's 'high' also permits TF32 (10-bit input mantissa);
            # padding may select a different kernel. Bound it separately, not
            # by silently applying the full-FP32 tolerance to mixed settings.
            atol, rtol = (2e-5, 2e-5) if matmul_precision == 'highest' else (2e-4, 2e-3)
            np.testing.assert_allclose(values, reference[1].cpu().numpy().flatten(), atol=atol, rtol=rtol)
            expected = reference[0].cpu().numpy()
            valid = np.abs(expected) < 1e6
            np.testing.assert_allclose(logits[valid], expected[valid], atol=atol, rtol=rtol)
            del reference
            gc.collect()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        stats = runtime.upper_graphs.statistics()
        assert stats['evictions'] >= 2 and stats['hits'] >= 1
        assert core.teacher.assert_integrity() == frozen
    finally:
        runtime.clear()
        core.to('cpu')
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision(previous_precision)
