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
        env.thick.bc.apply_custom_state({'hand': [card] + ['STRIKE_RED'] * count,
                                        'potions': ['Fairy Potion']})
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
                slot = aid - C.AUTOREGRESSIVE_MULTI
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
    assert env.action_space.n == 198
    assert 'hand #1' in env.describe_action(C.AUTOREGRESSIVE_MULTI + 1)
    for slot in (1, 4):
        obs, reward, terminated, truncated, info = env.step(C.AUTOREGRESSIVE_MULTI + slot)
        assert reward == 0 and not terminated and not truncated
        assert not info['simulator_transition']
        assert bc.to_dict() == before
        assert obs['select_hand_mask'][slot] == 1
        assert obs['combat_legal'].shape == (198,)
        assert not env.action_masks()[C.AUTOREGRESSIVE_MULTI:C.AUTOREGRESSIVE_MULTI + slot + 1].any()
    oracle.execute(sts.Action(sts.ActionType.MULTI_CARD_SELECT, (1 << 1) | (1 << 4)))
    assert env.describe_action(C.AUTOREGRESSIVE_STOP) == 'confirm multi-select [1, 4]'
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
    logits[C.AUTOREGRESSIVE_MULTI] = 30
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
            np.testing.assert_array_equal(leaf['action_mask'][C.AUTOREGRESSIVE_END:], obs['combat_legal'][row])
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            [(leaf['request_id'], .3, logits) for leaf in response['leaf_meta']], max_leaves=4)
    assert response['done'] and selected_seen
    result = response['result']
    assert result['visit_policy'].shape == (198,)
    assert result['action'] == C.AUTOREGRESSIVE_MULTI
    assert result['visit_policy'].sum() == pytest.approx(1)
    assert env.thick.bc.to_dict() == before
    env.step(result['action'])
    assert env._multi == {0}


@pytest.mark.parametrize('compact', [False, True])
def test_non_drinkable_potion_never_leaks_into_search_leaves(combat_factory, compact):
    env = combat_factory(['DEFEND_RED'] * 5, 'Jaw Worm',
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, card_language=not compact)
    env.thick.bc.apply_custom_state({'potions': ['Fairy Potion']})
    env._invalidate_backend_cache()
    assert not env.action_masks()[C.AUTOREGRESSIVE_POTION:C.AUTOREGRESSIVE_PICK].any()
    logits = np.zeros(198)
    response = env.nosl_is_az_cpp_begin(logits, state_nonce=42, simulations=16,
        depth=0, leaf_batch_size=4, compact_observation=compact)
    leaves = policy_leaves = 0
    for _ in range(32):
        if response['done']:
            break
        leaves += len(response['leaf_meta'])
        obs = response['observation_batch']
        policy_leaves += int(obs['combat_legal'].any(axis=1).sum())
        assert obs['combat_legal'].shape[-1] == 198
        assert not obs['combat_legal'][:, C.AUTOREGRESSIVE_POTION:C.AUTOREGRESSIVE_PICK].any()
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            [(leaf['request_id'], .3, logits) for leaf in response['leaf_meta']], max_leaves=4)
    assert leaves > 0 and policy_leaves > 0 and response['done']


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
    env.step(C.AUTOREGRESSIVE_MULTI + 1)
    obs = {k: torch.as_tensor(v[None]) for k, v in env._obs().items()}
    model.policy.set_training_mode(True)
    logits, value, outputs = network(obs, auxiliary_actions=torch.tensor([C.AUTOREGRESSIVE_STOP]))
    assert logits.shape == (1, 198) and 'transition' not in outputs
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
            legal[0,aid-C.AUTOREGRESSIVE_MULTI] = True
        with torch.no_grad():
            scores = decoder.step(state, hand, selected, torch.tensor([3-prefix.bit_count()]), legal)
        for aid, _ in pairs:
            probability = mass * scores[0,aid-C.AUTOREGRESSIVE_MULTI].exp().item()
            if aid == C.AUTOREGRESSIVE_STOP:
                probabilities.append(probability)
            else:
                walk(prefix | (1 << (aid-C.AUTOREGRESSIVE_MULTI)), probability)
    walk(0, 1.)
    assert len(probabilities) == 26
    np.testing.assert_allclose(probabilities, np.full(26, 1/26), rtol=1e-6)


def test_prefix_checkpoint_and_real_transition_replay(replayable_selection_env):
    import copy
    from stable_baselines3.common.logger import configure
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.combat_distribution import outcome_targets, transition_features
    env = replayable_selection_env
    env.step(C.AUTOREGRESSIVE_PLAY + 5)  # Purity, no target.
    samples = []
    for action in (C.AUTOREGRESSIVE_MULTI, C.AUTOREGRESSIVE_STOP, C.AUTOREGRESSIVE_PLAY):
        obs, mask = env._obs(), env.action_masks()
        assert mask[action]
        target = np.zeros(198, np.float32); target[action] = 1
        sample = dict(observation=obs, action_mask=mask, policy_target=target,
                      value_target=.4, policy_weight=1.)
        after, _, _, _, info = env.step(action)
        if info['simulator_transition']:
            sample.update(executed_action=action,
                          transition_target=transition_features(after)-transition_features(obs))
        else:
            state = env.export_training_state()
            before = env.thick.bc.to_dict()
            env.restore_training_state(state)
            assert env.thick.bc.to_dict() == before and env._multi == {0}
            assert env.export_training_state()['observation_sha256'] == state['observation_sha256']
        samples.append(sample)
    assert env.thick.bc is None or env.thick.bc.is_battle_over
    outcome_targets(samples, 'win')
    assert [s['combat_targets'][2] for s in samples] == [0, 1, 1]
    model = AccumulatingMaskablePPO(EntityCombatPolicy, env, seed=42, device='cpu',
        n_steps=4, batch_size=4, nosl_is_az=True, nosl_is_az_cpp=True,
        nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='rules',
        nosl_is_az_updates=2, nosl_is_az_batch_size=3, nosl_is_az_micro_batch_size=2,
        nosl_is_az_replay_capacity=8,
        policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4)))
    model.set_logger(configure(format_strings=[]))
    initial_parameters = {name: parameter.detach().clone()
                          for name, parameter in model.policy.named_parameters()}
    assert model.reset_nosl_is_az_shared_embeddings(42) == 0
    frozen, trainable = model.configure_nosl_is_az_inner_only(train_shared_embeddings=True)
    assert frozen == sum(p.numel() for p in model.policy.core.combat.transition.parameters())
    assert trainable > 0
    for name, parameter in model.policy.named_parameters():
        assert parameter.requires_grad == (not name.startswith('core.combat.transition.'))
        assert torch.equal(parameter, initial_parameters[name])
    model._nosl_is_az_samples = samples
    stats = model._train_nosl_is_az()
    assert stats['updates'] == 2 and stats['finite_fraction'] == 1
    replay = model._nosl_is_az_replay
    saved = replay.state_dict()
    assert saved['policy_contract'] == C.AUTOREGRESSIVE_CONTRACT
    replay.load_state_dict(saved)
    incompatible = copy.deepcopy(saved); incompatible['policy_contract'] = C.JOINT_CONTRACT
    with pytest.raises(ValueError, match='contract'):
        replay.load_state_dict(incompatible)
    head = model.policy.core.combat.complete_head.subset
    assert head.choice_key.weight.abs().sum() > 0


def test_rules_only_collection_never_calls_combat_network(monkeypatch):
    from env.core.run_env import SlayTheSpireEnv
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: SlayTheSpireEnv(combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
                                               card_language=True)] * 2)
    try:
        model = AccumulatingMaskablePPO(EntityCombatPolicy, env, seed=42, device='cpu',
            n_steps=1, batch_size=2, nosl_is_az=True, nosl_is_az_cpp=True,
            nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='rules',
            policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4)))
        _, callback = model._setup_learn(2, callback=lambda *_: True)
        def forbidden(*args, **kwargs):
            raise AssertionError('rules-only rows must never enter the combat network')
        monkeypatch.setattr(model.policy.core.combat, 'forward', forbidden)
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 1)
        assert model.num_timesteps == 2
    finally:
        env.close()


def test_compact_normal_actions_preserve_logits_without_legacy_slots(combat_factory):
    old_env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract=C.JOINT_CONTRACT, card_language=True)
    new_env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, card_language=True)
    kwargs = dict(seed=42, device='cpu', n_steps=2, batch_size=2,
                  policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4)))
    old, new = (MaskablePPO(EntityCombatPolicy, env, **kwargs) for env in (old_env, new_env))
    new.policy.load_state_dict(old.policy.state_dict())
    widths = []
    hook = new.policy.core.combat.complete_head.register_forward_pre_hook(
        lambda _module, args: widths.append(args[0].shape[-1]))
    results = []
    for model, env in ((old, old_env), (new, new_env)):
        model.policy.set_training_mode(False)
        obs = {k: torch.as_tensor(v[None]) for k, v in env._obs().items()}
        with torch.no_grad():
            results.append(model.policy.core.combat_actor_critic(obs))
    hook.remove()
    assert widths == [112]  # End, 15 action sources, 96 single choices; no legacy layout.
    torch.testing.assert_close(results[0][0][:, C.JOINT_END:C.JOINT_MULTI],
                               results[1][0][:, :C.AUTOREGRESSIVE_MULTI], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
