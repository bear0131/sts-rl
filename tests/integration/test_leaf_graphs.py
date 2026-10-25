"""CUDA-specific graph contracts; CPU CI still checks the role lookup algebra."""
import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.search.inference import CombatLeafGraphs
from rl.search.native import forward_leaves

pytestmark = pytest.mark.native


@pytest.mark.skipif(not torch.cuda.is_available(), reason='direct native graph inputs require CUDA')
@pytest.mark.parametrize('contract', ['legacy', 'joint-v2', 'autoregressive-v2', 'boss'])
def test_native_writer_matches_materialized_model_inputs(combat_factory, contract):
    import slaythespire as sts
    from env.search.native_batch import NativeSearchBatch
    from rl.policies.entity_combat import EntityCombatPolicy
    boss_weight = .9 if contract == 'boss' else 0.
    if boss_weight:
        contract = 'autoregressive-v2'
    entity = contract != 'legacy'
    env = combat_factory(['PURITY'] * 5, combat_action_contract=contract, card_language=entity)
    if boss_weight:
        env.thick.gc.cur_room = sts.Room.BOSS
    env.thick.bc.apply_custom_state({'hand': ['PURITY'] + ['STRIKE_RED'] * 5})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    model = AccumulatingMaskablePPO(EntityCombatPolicy if entity else SplitStSPointerPolicy,
        env, device='cuda', seed=1729, n_steps=8, batch_size=4,
        nosl_is_az=entity, nosl_is_az_cpp=entity, nosl_is_az_inner_only=entity,
        nosl_is_az_upper_strategy='rules' if entity else 'model',
        policy_kwargs=(dict(entity_config=dict(width=32,layers=1,heads=4,boss_win_weight=boss_weight)) if entity
                       else dict(d_model=32,n_layers=2,n_heads=4)))
    model.policy.set_training_mode(False)
    cache = CombatLeafGraphs(model.policy.core, 'cuda', amp='bf16', batch_multiple=4)
    materialized = CombatLeafGraphs(model.policy.core, 'cuda', amp='bf16', batch_multiple=4)
    pool = NativeSearchBatch([env], 2)
    try:
        pool.begin({0: ((np.zeros(env.action_space.n),), dict(simulations=16, state_nonce=1729,
            compact_observation=True, leaf_batch_size=1, fixed_worldline=True,
            boss_win_weight=boss_weight))})
        calls = 0
        while not pool.done:
            direct = cache.forward_native(pool)
            expected = materialized(pool.observations(), pool.active_relic_width)
            for actual, reference in zip(direct, expected):
                assert actual.tobytes() == reference.tobytes()
            pool.advance(*direct)
            calls += 1
        assert calls > 1
    finally:
        pool.close()
        cache.clear()
        materialized.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='captured transport requires a GPU runner')
@pytest.mark.parametrize('profile_interval', [0, 2])
def test_captured_transfers_own_buffers_and_return_independent_outputs(profile_interval):
    import gc

    class Core(torch.nn.Module):
        def combat_actor_critic(self, obs, **kwargs):
            value = sum(v.float() for v in obs.values())
            return torch.cat((value, -value), dim=1), value

    cache = CombatLeafGraphs(Core().eval(), 'cuda', capacity=2, batch_multiple=32,
                             profile_interval=profile_interval)
    saved = []
    for iteration, count in enumerate((17, 31, 4, 65, 33, 17)):
        observations = {str(dtype): (np.arange(count)[:, None] + iteration).astype(dtype)
                        for dtype in (np.int64, np.int32, np.float32, np.bool_)}
        expected = sum(v.astype(np.float32) for v in observations.values())
        logits, values = cache(observations, 1)
        np.testing.assert_array_equal(logits, np.concatenate((expected, -expected), axis=1))
        np.testing.assert_array_equal(values, expected[:, 0])
        saved.append((logits, values, expected))
        del observations
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
    cache.clear()
    for logits, values, expected in saved:
        np.testing.assert_array_equal(logits, np.concatenate((expected, -expected), axis=1))
        np.testing.assert_array_equal(values, expected[:, 0])
    stats = cache.statistics()
    assert stats['evictions'] > 0 and stats['hits'] >= 2
    assert stats['profile_samples'] == (3 if profile_interval else 0)
    if profile_interval:
        assert all(stats[key] > 0 for key in ('h2d_ms', 'replay_ms', 'd2h_ms'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='BF16 reduction validation requires a GPU runner')
@pytest.mark.parametrize('backend', ['eager', 'graph'])
def test_bf16_leaf_runtime_disables_partial_sum_truncation(combat_factory, backend):
    from rl.policies.entity_combat import EntityCombatPolicy
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='autoregressive-v2', card_language=True)
    model = AccumulatingMaskablePPO(EntityCombatPolicy, env, device='cuda', seed=1729,
        n_steps=8, batch_size=4, policy_kwargs=dict(entity_config=dict(width=256, layers=4, heads=8)),
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='rules')
    model.amp_mode = 'bf16'
    model.policy.set_training_mode(False)
    obs = env._obs()
    batch = {k: np.stack([v] * 64) for k, v in obs.items() if k in model.policy.core.combat_observation_keys}
    old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        cache = CombatLeafGraphs(model.policy.core, model.device, amp='bf16') if backend == 'graph' else None
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        expected = forward_leaves(model, batch, autocast=model.autocast_context, leaf_backend=cache)
        if cache is not None:
            cache.clear()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        actual = forward_leaves(model, batch, autocast=model.autocast_context, leaf_backend=cache)
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old


@pytest.fixture(autouse=True)
def explicit_fp32_precision():
    # Production lifecycle tests select TF32 globally. Cross-shape FP32 error
    # bounds below are for full FP32, independent of test ordering and hardware.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph validation requires a GPU runner')
@pytest.mark.parametrize('amp', ['off', 'bf16'])
@pytest.mark.parametrize('entity', [False, True, 'autoregressive'])
def test_joint_multi_rows_keep_order_outside_graphs(combat_factory, amp, entity):
    import slaythespire as sts
    from rl.policies.entity_combat import EntityCombatPolicy
    env = combat_factory(['PURITY'] * 5, combat_action_contract='autoregressive-v2' if entity == 'autoregressive' else 'joint-v2',card_language=entity)
    kwargs = dict(entity_config=dict(width=32,layers=1,heads=4)) if entity else dict(d_model=32,n_layers=2,n_heads=4)
    model = AccumulatingMaskablePPO(EntityCombatPolicy if entity else SplitStSPointerPolicy, env, device='cuda', seed=42,
        n_steps=8, batch_size=4, policy_kwargs=kwargs,
        nosl_is_az=entity,nosl_is_az_cpp=entity,nosl_is_az_inner_only=entity,
        nosl_is_az_upper_strategy='rules' if entity else 'model')
    model.amp_mode = amp
    model.policy.set_training_mode(False)
    cache = CombatLeafGraphs(model.policy.core, model.device, amp=amp, capacity=2)
    ordinary = env._obs()
    env.thick.bc.apply_custom_state({'hand':['PURITY'] + ['STRIKE_RED'] * 9})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    multi = env._obs()
    for choices in ((ordinary, multi, ordinary), (multi,), (multi, ordinary, multi, ordinary)):
        batch = {k:np.stack([obs[k] for obs in choices]) for k in ordinary
                 if k in model.policy.core.combat_observation_keys}
        model._native_leaf_inference = None
        expected = forward_leaves(model, batch, autocast=model.autocast_context)
        model._native_leaf_inference = cache
        actual = forward_leaves(model, batch, autocast=model.autocast_context)
        for a,b in zip(actual, expected):
            # Splitting subset rows changes BF16 GEMM batch shapes. The entity
            # encoder adds several such projections; allow 0.003 absolute
            # logits drift near zero, while same-shape graph tests stay exact.
            np.testing.assert_allclose(a,b,rtol=.03 if amp=='bf16' else 1e-5,
                                       atol=(.003 if entity else 3e-4) if amp=='bf16' else 2e-7)
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
    assert cache.subset_eager_rows == (0 if entity == 'autoregressive' else 4)
    if entity == 'autoregressive':
        from env.core import native_actions as C
        env.step(C.AUTOREGRESSIVE_MULTI + 1)
        selected = env._obs()
        batch = {k: np.stack([selected[k]] * 4) for k in batch}
        model._native_leaf_inference = None
        expected = forward_leaves(model, batch, autocast=model.autocast_context)
        model._native_leaf_inference = cache
        actual = forward_leaves(model, batch, autocast=model.autocast_context)
        for a, b in zip(actual, expected):
            # Compiled entity leaves fuse reductions and round BF16 at fewer
            # boundaries than eager. Other graph backends remain bitwise.
            np.testing.assert_allclose(a, b, rtol=.03 if amp == 'bf16' else 1e-5,
                                       atol=.003 if amp == 'bf16' else 2e-7)
    assert cache.hits >= 1


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
@pytest.mark.timeout(180)
def test_entity_mature_first_capture_keeps_reference_compiled_layout(combat_factory):
    import copy
    from rl.policies.entity_combat import EntityCombatPolicy
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='autoregressive-v2',
                         card_language=True)
    model = AccumulatingMaskablePPO(EntityCombatPolicy, env, device='cuda', seed=23,
        n_steps=8, batch_size=4, nosl_is_az=True, nosl_is_az_cpp=True,
        nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='rules',
        policy_kwargs=dict(entity_config=dict(width=256, layers=4, heads=8)))
    model.policy.set_training_mode(False)
    reference = copy.deepcopy(model.policy.core)
    state = reference.combat.transformer.state_dict()
    layer = torch.nn.TransformerEncoderLayer(256, 8, 1024, dropout=0., activation='gelu',
        batch_first=True, norm_first=True, device='cuda')
    reference.combat.transformer = torch.nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
    reference.combat.transformer.load_state_dict(state)
    reference.eval()
    caches = [CombatLeafGraphs(core, model.device, amp='bf16', batch_multiple=32, capacity=2)
              for core in (reference, model.policy.core)]
    raw = env._obs()
    # Starting at width=14 exposed a symbolic-shape compiler failure which
    # fresh games (first width=1) and small toy networks did not reproduce.
    for width, count in ((14, 46), (14, 64), (21, 128), (13, 32), (14, 46)):
        observations = {key: np.repeat(value[None], count, axis=0)
                        for key, value in raw.items()
                        if key in model.policy.core.combat_observation_keys}
        expected = caches[0](observations, width)
        actual = caches[1](observations, width)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        for a, b in zip(caches[1](observations, width), actual):
            np.testing.assert_array_equal(a, b)
    assert caches[1].evictions > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph validation requires a GPU runner')
@pytest.mark.parametrize('amp', ['off', 'bf16'])
@pytest.mark.parametrize('batch_multiple', [0, 4])
@pytest.mark.parametrize('combat_arch,contract', [('transformer','legacy'), ('mlp','legacy'), ('transformer','joint-v2'), ('entity','joint-v2'), ('entity','autoregressive-v2')])
def test_graph_outputs_follow_inputs_weights_and_shapes(tmp_path, combat_factory, amp, batch_multiple, combat_arch, contract, monkeypatch):
    from rl.policies.entity_combat import EntityCombatPolicy
    entity=combat_arch=='entity'
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract=contract,card_language=entity)
    kwargs = dict(entity_config=dict(width=32,layers=1,heads=4)) if entity else dict(d_model=32,n_layers=2,n_heads=4,combat_arch=combat_arch)
    model = AccumulatingMaskablePPO(EntityCombatPolicy if entity else SplitStSPointerPolicy, env, device='cuda', seed=7,
        n_steps=8, batch_size=4, policy_kwargs=kwargs,
        nosl_is_az=entity,nosl_is_az_cpp=entity,nosl_is_az_inner_only=entity,
        nosl_is_az_upper_strategy='rules' if entity else 'model')
    model.amp_mode = amp
    model.policy.set_training_mode(False)
    cache = CombatLeafGraphs(model.policy.core, model.device, amp=amp, capacity=2,
                             batch_multiple=batch_multiple)
    obs = env._obs()
    def compare(count):
        batch = {k: np.stack([v] * count) for k, v in obs.items()
                 if k in model.policy.core.combat_observation_keys}
        model._native_leaf_inference = None
        with monkeypatch.context() as patch:
            if entity:
                patch.setattr(model.policy.core.combat, 'inference_card_table', model.policy.core.combat._encode_card_table)
            eager = forward_leaves(model, batch, autocast=model.autocast_context)
        if batch_multiple:
            padded_count = (count + batch_multiple - 1) // batch_multiple * batch_multiple
            padded = {k:np.concatenate([v, np.repeat(v[:1], padded_count-count, axis=0)])
                      for k,v in batch.items()}
            with monkeypatch.context() as patch:
                if entity:
                    patch.setattr(model.policy.core.combat, 'inference_card_table', model.policy.core.combat._encode_card_table)
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
            if cache.compiled_forward is not None:
                np.testing.assert_allclose(a, b, rtol=.03 if amp == 'bf16' else 1e-5,
                                           atol=.003 if amp == 'bf16' else 2e-7)
            else:
                np.testing.assert_array_equal(a, b)
    compare(2)
    obs['player_scalars'] = obs['player_scalars'].copy() + 0.01
    compare(2)
    with torch.no_grad():
        if entity:
            model.policy.core.combat.end.bias.add_(0.125)
            model.policy.core.combat.player_projection.weight.mul_(1.1)
            model.policy.core.combat.card.local_encoder.weight.mul_(1.5)
            model.policy.core.combat.card.weight.add_(0.2)
        elif combat_arch == 'mlp':
            model.policy.core.combat.policy_head.bias.add_(0.125)
            model.policy.core.combat.trunk[0].weight.mul_(1.1)
        else:
            model.policy.core.combat.fixed_head.bias.add_(0.125)
            # Catch stale BF16 weight casts as well as stale inputs.
            model.policy.core.combat.encoder.global_projection.weight.mul_(1.1)
    compare(2)
    for size in (3, 4, 9, 5, 2):
        compare(size)
    if entity:
        state={k:v.clone() for k,v in model.policy.core.combat.state_dict().items()}
        state['card.local_encoder.bias'].add_(0.25)
        model.policy.core.combat.load_state_dict(state)
        compare(2)
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
    model._native_leaf_inference.clear()
    torch.cuda.empty_cache()
    assert not model._native_leaf_inference.entries
    recaptured = forward_leaves(model, batch, autocast=model.autocast_context)
    for a,b in zip(recaptured,expected):
        np.testing.assert_array_equal(a,b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires production CUDA geometry')
@pytest.mark.parametrize('amp', ['off', 'bf16'])
def test_compiled_entity_production_geometry(combat_factory, amp):
    """Fusion tolerances, dynamic shapes and live weights at 256/4/8."""
    from rl.policies.entity_combat import EntityCombatPolicy
    torch.compiler.reset()
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='autoregressive-v2', card_language=True)
    model = AccumulatingMaskablePPO(EntityCombatPolicy, env, device='cuda', seed=23,
        n_steps=8, batch_size=4, policy_kwargs=dict(entity_config=dict(width=256,layers=4,heads=8)),
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='rules')
    model.amp_mode = amp
    model.policy.set_training_mode(False)
    core = model.policy.core
    cache = CombatLeafGraphs(core, model.device, amp=amp, capacity=2, batch_multiple=32)
    assert (cache.compiled_forward is not None) == (amp == 'bf16')
    initial_keys = tuple(core.state_dict())
    obs = env._obs()
    previous = None
    for index, (count, width) in enumerate(((32,4), (64,4), (32,8), (32,4), (32,4), (7,4), (31,4))):
        batch = {k:np.stack([v]*count) for k,v in obs.items() if k in core.combat_observation_keys}
        batch['relics'].fill(0)
        batch['relics'][:,width-1] = 1
        if index == 4:
            with torch.no_grad():
                core.combat.end.bias.add_(.25)
                core.combat.card.weight.add_(.1)
        model._native_leaf_inference = None
        eager = forward_leaves(model, batch, autocast=model.autocast_context)
        model._native_leaf_inference = cache
        actual = forward_leaves(model, batch, autocast=model.autocast_context)
        # BF16 fusion changes intermediate rounding. Bound logits, probability
        # and utility separately; FP32 receives a much tighter error budget.
        np.testing.assert_allclose(actual[0], eager[0], rtol=.03 if amp=='bf16' else 2e-5,
                                   atol=.006 if amp=='bf16' else 2e-6)
        np.testing.assert_allclose(actual[1], eager[1], rtol=0,
                                   atol=3e-4 if amp=='bf16' else 2e-6)
        legal = batch['combat_legal'].astype(bool)
        probabilities = [torch.as_tensor(x[0]).masked_fill(~torch.as_tensor(legal), -1e9).softmax(-1)
                         for x in (actual, eager)]
        torch.testing.assert_close(*probabilities, rtol=0, atol=.002 if amp=='bf16' else 2e-6)
        if index == 3:
            previous = actual[0].copy()
        if index == 4:
            assert np.max(np.abs(previous-actual[0])) > .05
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
    assert cache.evictions >= 1 and cache.hits >= 1
    assert tuple(core.state_dict()) == initial_keys
