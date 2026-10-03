"""Bounded checkpoint inference on native branches, independent of execution."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from env.core import encode, native_actions as C  # bootstrap before sts
import slaythespire as sts
from schemas.combat_state import normalize_sim_global
from rl.evaluation.baselines import planner_value as P
from rl.policies.entity_combat import EntityCombatConfig, EntityCombatCore
from rl.search.native import forward_leaves


@pytest.fixture
def value_fixture(combat_factory):
    torch.manual_seed(20261003)
    native, ar = [combat_factory(['STRIKE_RED'] * 5, 'Three Sentries',
        combat_action_contract=contract, card_language=True)
        for contract in ('native-v1', C.AUTOREGRESSIVE_CONTRACT)]
    for env in (native, ar):
        env.thick.bc.apply_custom_state({'player': {'hp': 40, 'max_hp': 80},
            'hand': ['STRIKE_RED', 'DEFEND_RED', 'PURITY'],
            'draw': [('STRIKE_RED', 0), ('DEFEND_RED', 1), ('STRIKE_RED', 2), ('BASH', 0)],
            'monsters': [{'hp': 0}, {'hp': 40}, {'hp': 0}]})
        env._invalidate_backend_cache()
    core = EntityCombatCore(ar.observation_space, EntityCombatConfig(width=32, layers=1, heads=4))
    core.eval()
    model = SimpleNamespace(policy=SimpleNamespace(core=core, training=False),
        device=torch.device('cpu'), observation_space=ar.observation_space, action_space=ar.action_space,
        nosl_value_contract='combat-end-hp-v1', nosl_supervision_contract='combat-outcome-quantiles-v1',
        nosl_is_az=True, nosl_is_az_inner_only=True, hierarchical_combat_hp=True,
        nosl_is_az_end_turn_as_leaf=False, nosl_is_az_boss_transition_target=False)
    return native, ar, model


def test_real_entity_value_matches_live_ar_input_without_changing_native_env(value_fixture):
    native, ar, model = value_fixture
    backend = object()
    model._native_leaf_inference = backend  # Explicit eager must bypass this unsupported backend.
    adapter = P.CheckpointLeafValue(model)
    before = native.thick.bc.to_dict()
    global_state = normalize_sim_global(native.thick.gc)
    metadata = copy.deepcopy(global_state)
    weights = {key: value.clone() for key, value in model.policy.core.state_dict().items()}
    expected_obs = ar._obs()
    _, expected = forward_leaves(model, {k: v[None] for k, v in expected_obs.items()},
                                compact=True, leaf_backend=None)
    actual = adapter.evaluate([native.thick.bc], root_global=global_state)
    np.testing.assert_array_equal(actual, expected)
    assert 0 < actual[0] < 1
    assert native.combat_action_contract == 'native-v1'
    assert native.thick.bc.to_dict() == before and global_state == metadata
    assert model._native_leaf_inference is backend
    assert not model.policy.training and not model.policy.core.training
    for key, value in weights.items():
        assert torch.equal(value, model.policy.core.state_dict()[key])
    assert adapter.manifest()['batch_cap'] == 64
    json.dumps(adapter.manifest())


def test_chunk_cap_order_and_empty_input(value_fixture, monkeypatch):
    native, _, model = value_fixture
    adapter = P.CheckpointLeafValue(model)
    battles = []
    expected = []
    for i in range(130):
        branch = native.thick.bc.clone()
        branch.apply_custom_state({'player': {'hp': 1 + i % 79}})
        battles.append(branch)
        expected.append((1 + i % 79) / encode.HP_SCALE)
    calls = []
    def forward(model_arg, observations, *, compact, leaf_backend):
        assert model_arg is model and compact and leaf_backend is None
        calls.append(len(observations['player_scalars']))
        return None, observations['player_scalars'][:, 0]
    monkeypatch.setattr(P, 'forward_leaves', forward)
    result = adapter.evaluate(iter(battles), root_global=normalize_sim_global(native.thick.gc))
    assert calls == [64, 64, 2]
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)
    assert adapter.evaluate([], root_global={}) == []
    assert calls == [64, 64, 2]


@pytest.mark.parametrize('name,value', [
    ('nosl_value_contract', 'legacy-boss-transition'),
    ('nosl_supervision_contract', 'scalar-hp-v1'), ('nosl_is_az', False),
    ('nosl_is_az_inner_only', False), ('hierarchical_combat_hp', False),
    ('nosl_is_az_end_turn_as_leaf', True), ('nosl_is_az_boss_transition_target', True),
])
def test_rejects_unsupported_target_metadata(value_fixture, name, value):
    _, _, model = value_fixture
    setattr(model, name, value)
    with pytest.raises(ValueError, match=name):
        P.CheckpointLeafValue(model)


def test_missing_metadata_is_not_a_compatible_default(value_fixture):
    _, _, model = value_fixture
    del model.nosl_is_az_end_turn_as_leaf
    with pytest.raises(ValueError, match='end_turn_as_leaf'):
        P.CheckpointLeafValue(model)


@pytest.mark.parametrize('change', ['core', 'contract', 'action', 'schema', 'training'])
def test_rejects_unsupported_model_and_schema(value_fixture, change):
    _, _, model = value_fixture
    if change == 'core':
        model.policy.core = SimpleNamespace(combat_only=True)
    elif change == 'contract':
        model.observation_space.spaces['combat_legal'] = spaces.Box(0, 1, (C.JOINT_WIDTH,), dtype=bool)
    elif change == 'action':
        model.action_space = spaces.Discrete(71)
    elif change == 'schema':
        model.observation_space.spaces['monster_scalars'] = spaces.Box(-1, 1, (5, 7), dtype=np.float32)
    else:
        model.policy.core.train()
    with pytest.raises(ValueError):
        P.CheckpointLeafValue(model)


def test_eval_mode_is_checked_again_without_mutating_it(value_fixture):
    native, _, model = value_fixture
    adapter = P.CheckpointLeafValue(model)
    model.policy.core.train()
    with pytest.raises(ValueError, match='eval mode'):
        adapter.evaluate([native.thick.bc], root_global=normalize_sim_global(native.thick.gc))
    assert model.policy.core.training


@pytest.mark.parametrize('kind', ['projected', 'terminal', 'no-legal'])
def test_rejects_invalid_leaf_before_forward(value_fixture, monkeypatch, kind):
    native, _, model = value_fixture
    adapter = P.CheckpointLeafValue(model)
    branch = native.thick.bc.clone()
    if kind == 'projected':
        branch.execute_to_enemy_boundary(4 << 29)
    elif kind == 'terminal':
        branch.apply_custom_state({'monsters': [{'hp': 0}, {'hp': 1}, {'hp': 0}]})
        branch.execute(sts.Action(sts.ActionType.CARD, 0, 1))
        assert int(branch.to_dict()['outcome']) == 1
    else:
        raw = branch.to_dict()
        branch = SimpleNamespace(to_dict=lambda: raw, autoregressive_policy_actions=lambda prefix: [])
    before = branch.to_dict()
    monkeypatch.setattr(P, 'forward_leaves', lambda *a, **kw: pytest.fail('invalid leaf reached model'))
    with pytest.raises(ValueError):
        adapter.evaluate([branch], root_global=normalize_sim_global(native.thick.gc))
    assert branch.to_dict() == before


@pytest.mark.parametrize('bad', [[np.nan], [np.inf], [-.1], [1.1], [.2, .3], [[.2]]])
def test_rejects_nonfinite_wrong_shape_and_wrong_scale(value_fixture, monkeypatch, bad):
    native, _, model = value_fixture
    adapter = P.CheckpointLeafValue(model)
    monkeypatch.setattr(P, 'forward_leaves', lambda *a, **kw: (None, np.asarray(bad)))
    with pytest.raises(ValueError):
        adapter.evaluate([native.thick.bc], root_global=normalize_sim_global(native.thick.gc))


def test_actual_pending_native_selection_uses_ar_prefix_zero(value_fixture):
    native, ar, model = value_fixture
    for env in (native, ar):
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 2))
        env._invalidate_backend_cache()
    before = native.thick.bc.to_dict()
    expected_obs = ar._obs()
    assert expected_obs['select_context'].any() and not expected_obs['select_hand_mask'].any()
    _, expected = forward_leaves(model, {k: v[None] for k, v in expected_obs.items()},
                                compact=True, leaf_backend=None)
    actual = P.CheckpointLeafValue(model).evaluate([native.thick.bc],
        root_global=normalize_sim_global(native.thick.gc))
    np.testing.assert_array_equal(actual, expected)
    assert native.thick.bc.to_dict() == before


def test_aligned_pile_group_permutations_preserve_real_entity_predictions(value_fixture):
    _, ar, model = value_fixture
    obs = ar._obs()
    rows = [obs]
    rng = np.random.default_rng(20261003)
    for _ in range(8):
        permuted = {k: v.copy() for k, v in obs.items()}
        for pile in ('draw', 'discard'):
            keys = [pile + suffix for suffix in (
                '_card_ids', '_card_counts', '_card_upgraded_counts', '_card_upgrades')]
            indices = rng.permutation(len(obs[keys[0]]))
            for key in keys:
                permuted[key] = permuted[key][indices]
        rows.append(permuted)
    logits, values = forward_leaves(model, {k: np.stack([r[k] for r in rows]) for k in obs},
                                   compact=True, leaf_backend=None)
    # FP32 attention reductions can reorder; this is a same-device numerical
    # contract, not a cross-platform or bitwise invariance claim.
    np.testing.assert_allclose(values, np.full_like(values, values[0]), rtol=1e-6, atol=1e-7)
    legal = obs['combat_legal']
    np.testing.assert_allclose(logits[:, legal], np.broadcast_to(logits[0, legal], logits[:, legal].shape),
                               rtol=1e-6, atol=1e-6)
