"""POL01: choose and search card effects, then execute a concrete hand slot."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import slaythespire as sts

from env.core import native_actions as C
from rl.evaluation.strategies import DecisionRequest, PolicyStrategy
from rl.policies.entity_combat import EntityCombatPolicy

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


def _env(combat_factory, contract='autoregressive-v2'):
    env = combat_factory(['ANGER'] * 10, 'Jaw Worm',
                         combat_action_contract=contract, card_language=True)
    env.thick.bc.apply_custom_state({
        'hand': ['ANGER'] * 4, 'draw': ['DEFEND_RED'] * 5,
        'discard': [], 'exhaust': [], 'player': {'energy': 3}})
    env._invalidate_backend_cache()
    return env


@pytest.mark.parametrize('contract', C.NEURAL_CONTRACTS)
def test_group_mass_beats_end_and_executes_real_card(combat_factory, contract):
    env = _env(combat_factory, contract)
    offset = C.policy_offset(contract)
    ids = [offset + 1 + 6 * i for i in range(4)]
    probs = np.zeros(env.action_space.n, np.float32)
    probs[offset] = .261
    probs[ids] = .739 / 4
    assert probs.argmax() == offset
    policy = EntityCombatPolicy(env.observation_space, env.action_space, lambda _: 0.0,
        entity_config=dict(width=32, layers=1, heads=4, quantiles=8))
    policy.core.action_logits = lambda obs: torch.as_tensor(
        np.log(np.maximum(probs, 1e-30))[None], device=policy.device)
    model = SimpleNamespace(policy=policy, predict=policy.predict,
                            observation_space=env.observation_space,
                            action_space=env.action_space)
    groups = env.equivalent_action_groups()
    assert groups[ids].tolist() == [ids[0]] * 4
    request = DecisionRequest(env, env._obs(), {'phase': 'combat'}, 42, 0)
    decision = PolicyStrategy(model).decide([request])[0]
    assert decision.action == ids[0]
    assert decision.diagnostics == {'equivalent_play_mass_decisions': 1}
    assert all(isinstance(value, (int, float)) for value in decision.diagnostics.values())
    before = env.thick.bc.to_dict()['monsters'][0]['hp']
    env.step(decision.action)
    assert env.thick.bc.to_dict()['monsters'][0]['hp'] == before - 6


@pytest.mark.parametrize('contract', C.NEURAL_CONTRACTS)
@pytest.mark.parametrize('mode', [0, 2])
def test_search_uses_one_edge_and_preserves_alias_target_mass(combat_factory, contract, mode):
    env = _env(combat_factory, contract)
    offset = C.policy_offset(contract)
    ids = [offset + 1 + 6 * i for i in range(4)]
    logits = np.full(env.action_space.n, -100.0)
    logits[offset] = np.log(.261)
    logits[ids] = np.log(.739 / 4)
    before = env.thick.bc.to_dict()
    response = env.nosl_is_az_cpp_begin(logits, simulations=32, depth=1, state_nonce=42,
        root_search_mode=mode, paper_full_gumbel=mode == 2,
        gumbel_scale=0, fixed_worldline=True, leaf_batch_size=1,
        compact_observation=True)
    for _ in range(64):
        if response['done']:
            break
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            [(leaf['request_id'], .5, logits) for leaf in response['leaf_meta']], max_leaves=1)
    assert response['done']
    children = env.nosl_is_az_cpp_tree_snapshot((), 64)['children']
    plays = [c for c in children if c['action_bits'] >> 29 == 0]
    assert len(plays) == 1 and len(children) == 2
    assert plays[0]['prior'] == pytest.approx(.739)
    assert sum(c['visits'] for c in children) == 32
    targets = response['result']['visit_policy']
    np.testing.assert_array_equal(response['result']['action_groups'], env.equivalent_action_groups())
    assert targets.sum() == pytest.approx(1)
    assert np.all(targets[ids] > 0)
    np.testing.assert_array_equal(targets[ids], np.repeat(targets[ids[0]], 4))
    assert response['result']['action'] == ids[0]
    assert env.thick.bc.to_dict() == before


def test_upgrades_targets_and_permanent_instance_effects_stay_distinct(combat_factory):
    env = _env(combat_factory)
    env.thick.bc.apply_custom_state({'hand': [
        'ANGER', 'ANGER', ('ANGER', 1), 'RITUAL_DAGGER', 'RITUAL_DAGGER'],
        'monsters': [{'hp': 40}]})
    assert env.thick.bc.equivalent_play_slots()[:5] == [0, 0, 2, 3, 4]
    other = combat_factory(['ANGER'] * 10, 'Two Louse',
        combat_action_contract='autoregressive-v2', card_language=True)
    other.thick.bc.apply_custom_state({'hand': ['ANGER', 'ANGER']})
    other._invalidate_backend_cache()
    groups = other.equivalent_action_groups()
    assert groups[7] == 1 and groups[8] == 2 and groups[1] != groups[2]


def test_selection_prefixes_are_not_merged(combat_factory):
    env = combat_factory(['PURITY'] * 10,
        combat_action_contract='autoregressive-v2', card_language=True)
    env.thick.bc.apply_custom_state({'hand': ['PURITY', 'ANGER', 'ANGER']})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    np.testing.assert_array_equal(env.equivalent_action_groups(), np.arange(C.AUTOREGRESSIVE_SIZE))


def test_snecko_current_costs_control_groups(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 10, relics=['Snecko Eye'],
        combat_action_contract='autoregressive-v2', card_language=True)
    obs = env._obs()
    slots = env.thick.bc.equivalent_play_slots()
    occupied = np.flatnonzero(obs['hand_ids'])
    costs = obs['hand_costs'][occupied]
    assert len(np.unique(costs)) > 1
    for i in occupied:
        for j in occupied:
            assert (slots[i] == slots[j]) == (obs['hand_costs'][i] == obs['hand_costs'][j])


def test_future_identity_ignores_rng_and_interchangeable_slots(combat_factory):
    env = _env(combat_factory)
    env.thick.bc.apply_custom_state({'hand': ['ANGER','DEFEND_RED','ANGER']})
    a, b = env.thick.bc.clone(), env.thick.bc.clone()
    a.execute(sts.Action(sts.ActionType.CARD,0,0))
    b.execute(sts.Action(sts.ActionType.CARD,2,0))
    assert a.to_dict()['hand'] != b.to_dict()['hand']
    assert a.nosl_future_key() == b.nosl_future_key()
    b.nosl_redeterminize(987654)
    assert a.nosl_future_key() == b.nosl_future_key()
    b.apply_custom_state({'player': {'energy': 0}})
    assert a.nosl_future_key() != b.nosl_future_key()


@pytest.mark.parametrize('mode', [0, 2])
def test_blue_search_keeps_alias_edges_and_slot_targets(combat_factory, mode):
    env = _env(combat_factory)
    ids = [1 + 6 * i for i in range(4)]
    logits = np.full(198, -100.0)
    logits[0] = np.log(.261)
    logits[ids] = np.log(.739 / 4)
    response = env.nosl_is_az_cpp_begin(logits, simulations=32, depth=1,
        state_nonce=42, root_search_mode=mode, paper_full_gumbel=mode == 2,
        gumbel_scale=0, fixed_worldline=True, leaf_batch_size=1,
        merge_equivalent_actions=False, compact_observation=True)
    for _ in range(64):
        if response['done']:
            break
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            [(leaf['request_id'], .5, logits) for leaf in response['leaf_meta']], max_leaves=1)
    assert response['done']
    children = env.nosl_is_az_cpp_tree_snapshot((), 64)['children']
    assert len(children) == 5
    plays = [c for c in children if c['action_bits'] >> 29 == 0]
    assert len(plays) == 4
    assert [c['prior'] for c in plays] == pytest.approx([.739 / 4] * 4)
    np.testing.assert_array_equal(response['result']['action_groups'], np.arange(198))
    assert response['result']['visit_policy'].sum() == pytest.approx(1)
    assert sum(c['visits'] for c in children) == 32


def test_individual_policy_does_not_regroup_same_cards(combat_factory):
    env = _env(combat_factory)
    policy = EntityCombatPolicy(env.observation_space, env.action_space, lambda _: 0.0,
        entity_config=dict(width=32, layers=1, heads=4, quantiles=8))
    logits = np.full(198, -100.0)
    logits[0] = np.log(.261)
    logits[[1, 7, 13, 19]] = np.log(.739 / 4)
    policy.core.action_logits = lambda obs: torch.as_tensor(logits[None], device=policy.device)
    model = SimpleNamespace(policy=policy, predict=policy.predict,
        observation_space=env.observation_space, action_space=env.action_space,
        nosl_action_group_contract='individual-actions')
    request = DecisionRequest(env, env._obs(), {'phase': 'combat'}, 42, 0)
    assert PolicyStrategy(model).decide([request])[0].action == 0
