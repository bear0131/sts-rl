"""Sequential selection paths against the independent complete-action oracle."""
import numpy as np
import pytest
import slaythespire as sts
import torch
from sb3_contrib import MaskablePPO

from env.core import native_actions as C
from rl.policies.entity_combat import EntityCombatPolicy

pytestmark = pytest.mark.native


def selection_env(combat_factory, card='PURITY', count=6, *, card_language=True):
    env = combat_factory(['PURITY'] * 5, 'Two Louse',
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, card_language=card_language)
    if card == 'GAMBLERS_BREW':
        env.thick.bc.apply_custom_state({'hand': ['STRIKE_RED'] * count, 'potions': ['Gamblers Brew']})
        env.thick.bc.execute(sts.Action(sts.ActionType.POTION, 0))
    else:
        env.thick.bc.apply_custom_state({'hand': [card] + ['STRIKE_RED'] * count})
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    return env


@pytest.mark.parametrize('card', ['PURITY', 'GAMBLERS_BREW'])
@pytest.mark.parametrize('count', [1, 3, 6, 9])
def test_ordered_paths_cover_each_legal_set_once(combat_factory, card, count):
    env = selection_env(combat_factory, card, count)
    bc = env.thick.bc
    before = bc.to_dict()
    oracle = {int(a.source_idx) for a in bc.rule_actions() if int(a.action_type) == 3}
    completed = []
    def walk(prefix):
        actions = bc.autoregressive_policy_actions(prefix)
        assert 1 <= len(actions) <= 11
        for aid, bits in actions:
            assert C.autoregressive_action_id(bits) == aid
            assert C.autoregressive_native_bits(aid) == bits
            if aid == C.AUTOREGRESSIVE_STOP:
                completed.append(prefix)
            else:
                slot = aid - C.JOINT_MULTI
                assert not prefix >> slot
                walk(prefix | (1 << slot))
    walk(0)
    assert len(completed) == len(set(completed))
    assert set(completed) == oracle
    assert bc.to_dict() == before
    with pytest.raises(ValueError):
        C.autoregressive_action_id(3 << 29)


@pytest.mark.parametrize('card', ['PURITY', 'GAMBLERS_BREW'])
def test_prefix_steps_do_not_execute_until_stop(combat_factory, card):
    env = selection_env(combat_factory, card)
    bc = env.thick.bc
    before = bc.to_dict()
    oracle = bc.clone()
    assert env.action_space.n == 379
    for slot in (1, 4):
        obs, reward, terminated, truncated, info = env.step(C.JOINT_MULTI + slot)
        assert reward == 0 and not terminated and not truncated
        assert not info['simulator_transition']
        assert bc.to_dict() == before
        assert obs['select_hand_mask'][slot] == 1
        assert obs['combat_legal'].shape == (198,)
        assert not env.action_masks()[C.JOINT_MULTI:C.JOINT_MULTI + slot + 1].any()
    oracle.execute(sts.Action(sts.ActionType.MULTI_CARD_SELECT, (1 << 1) | (1 << 4)))
    result = env.step(C.AUTOREGRESSIVE_STOP)
    assert result[4]['simulator_transition']
    assert bc.to_dict() == oracle.to_dict()
    # Check a subsequent random transition too, not just the immediate cards.
    bc.execute(sts.Action(sts.ActionType.END_TURN))
    oracle.execute(sts.Action(sts.ActionType.END_TURN))
    assert bc.to_dict() == oracle.to_dict()


@pytest.mark.parametrize('compact', [False, True])
def test_native_search_exposes_prefix_leaves_without_subsets(combat_factory, compact):
    env = selection_env(combat_factory, card_language=not compact)
    before = env.thick.bc.to_dict()
    logits = np.zeros(C.AUTOREGRESSIVE_SIZE)
    logits[C.JOINT_MULTI] = 30
    response = env.nosl_is_az_cpp_begin(logits, state_nonce=42, simulations=8, depth=1,
        leaf_batch_size=4, compact_observation=compact, root_search_mode=2,
        paper_full_gumbel=True, gumbel_scale=0, gumbel_rescale_q=False,
        gumbel_value_scale=1)
    selected_seen = False
    for _ in range(64):
        if response['done']:
            break
        obs = response['observation_batch']
        selected_seen |= bool(obs['select_hand_mask'].any())
        assert obs['combat_legal'].shape[1] == 198
        for row, leaf in enumerate(response['leaf_meta']):
            np.testing.assert_array_equal(leaf['action_mask'][C.JOINT_END:], obs['combat_legal'][row])
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            [(leaf['request_id'], .3, logits) for leaf in response['leaf_meta']], max_leaves=4)
    assert response['done'] and selected_seen
    result = response['result']
    assert result['visit_policy'].shape == (379,)
    assert result['action'] == C.JOINT_MULTI
    assert result['visit_policy'].sum() == pytest.approx(1)
    assert env.thick.bc.to_dict() == before
    env.step(result['action'])
    assert env._multi == {0}


def test_sequential_model_predicts_updates_and_reloads(combat_factory, tmp_path):
    env = selection_env(combat_factory)
    model = MaskablePPO(EntityCombatPolicy, env, seed=42, device='cpu', n_steps=2, batch_size=2,
        policy_kwargs=dict(entity_config=dict(width=32, layers=1, heads=4)))
    network = model.policy.core.combat
    assert network.autoregressive_actions
    assert not hasattr(network.complete_head.subset, 'path_edges')
    assert not hasattr(network.complete_head.subset, 'membership')
    model.policy.set_training_mode(False)
    mask = env.action_masks()
    action, _ = model.predict(env._obs(), action_masks=mask, deterministic=True)
    assert mask[int(action)]
    env.step(C.JOINT_MULTI + 1)
    obs = {k: torch.as_tensor(v[None]) for k, v in env._obs().items()}
    model.policy.set_training_mode(True)
    logits, value, outputs = network(obs, auxiliary_actions=torch.tensor([C.AUTOREGRESSIVE_STOP]))
    assert logits.shape == (1, 379) and outputs['transition'].shape == (1, 6)
    legal = torch.as_tensor(env.action_masks()[None])
    loss = -logits.masked_fill(~legal, -1e9).log_softmax(-1)[0, C.AUTOREGRESSIVE_STOP] + value.sum()
    model.policy.optimizer.zero_grad()
    loss.backward()
    assert network.complete_head.subset.choice_key.weight.grad.abs().sum() > 0
    model.policy.optimizer.step()
    model.policy.set_training_mode(False)
    expected, _ = model.predict(env._obs(), action_masks=env.action_masks(), deterministic=True)
    path = tmp_path / 'sequential.zip'
    model.save(path)
    restored = MaskablePPO.load(path, device='cpu')
    actual, _ = restored.predict(env._obs(), action_masks=env.action_masks(), deterministic=True)
    assert int(actual) == int(expected)
    assert C.policy_contract(restored.observation_space) == C.AUTOREGRESSIVE_CONTRACT


def test_initial_conditional_prior_is_uniform_over_valid_sets(combat_factory):
    env = selection_env(combat_factory, count=5)
    model = MaskablePPO(EntityCombatPolicy, env, seed=42, device='cpu', n_steps=2, batch_size=2,
        policy_kwargs=dict(entity_config=dict(width=32, layers=1, heads=4)))
    decoder = model.policy.core.combat.complete_head.subset
    # Fixed contextual entities, changing only the autoregressive prefix.
    state, hand = torch.randn(1,32), torch.randn(1,10,32)
    probabilities = []
    def walk(prefix, mass):
        selected = torch.tensor([[(prefix >> i) & 1 for i in range(10)]], dtype=torch.float32)
        legal = torch.zeros(1,11,dtype=torch.bool)
        pairs = env.thick.bc.autoregressive_policy_actions(prefix)
        for aid, _ in pairs:
            legal[0,aid-C.JOINT_MULTI] = True
        with torch.no_grad():
            scores = decoder.step(state, hand, selected, torch.tensor([3-prefix.bit_count()]), legal)
        for aid, _ in pairs:
            probability = mass * scores[0,aid-C.JOINT_MULTI].exp().item()
            if aid == C.AUTOREGRESSIVE_STOP:
                probabilities.append(probability)
            else:
                walk(prefix | (1 << (aid-C.JOINT_MULTI)), probability)
    walk(0, 1.)
    assert len(probabilities) == 26
    np.testing.assert_allclose(probabilities, np.full(26, 1/26), rtol=1e-6)
