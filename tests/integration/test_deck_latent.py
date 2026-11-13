from copy import deepcopy

import numpy as np
import pytest
import torch
from stable_baselines3.common.utils import obs_as_tensor

from env.core import vocab
from env.core.run_env import SlayTheSpireEnv
from rl.policies.deck_latent import DeckLatentPolicy


pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


@pytest.fixture
def latent_env():
    env = SlayTheSpireEnv(card_language=True, global_map_view=True, upper_public_context=True,
        event_option_observation=True, combat_action_contract='autoregressive-v2',
        combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True)
    env.reset(seed=20261111)
    yield env
    env.close()


def policy_for(env, encoder='pooled-mlp'):
    torch.manual_seed(20261111)
    return DeckLatentPolicy(env.observation_space, env.action_space, lambda _: 5e-5,
        d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1',
        deck_combat_encoder=encoder)


def tensor_obs(env, repeats=1):
    return obs_as_tensor({k: np.repeat(v[None], repeats, 0) for k, v in env._obs().items()}, 'cpu')


def enter_combat(env):
    for _ in range(128):
        if env.thick.bc is not None:
            return
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    raise AssertionError('could not enter combat with bounded legal actions')


@pytest.mark.cuda
def test_resident_frozen_pile_projection_preserves_large_counts(latent_env):
    from rl.search.resident_deck import ResidentDeckScorer
    from rl.policies.resident_ragged import frozen_pile_summary
    policy = policy_for(latent_env, 'transformer64x2').cuda().eval()
    scorer = ResidentDeckScorer(policy)
    try:
        torch.manual_seed(20261111)
        counts = torch.randint(0, 334, (17,3,len(scorer.card_base)),device='cuda').float()
        counts *= torch.rand_like(counts) > .9
        counts[0].zero_()
        upgraded = counts * (torch.rand_like(counts) > .5)
        magnitudes = upgraded * torch.randint(0, 16, upgraded.shape,device='cuda') / 4
        with torch.no_grad():
            want = counts@scorer.card_base + upgraded@scorer.card_delta
            want += magnitudes.sum(2,keepdim=True)*policy.core.combat.summaries.upgrade_projection
            got,sizes = frozen_pile_summary(counts,upgraded,magnitudes,scorer.card_base,
                scorer.card_delta,policy.core.combat.summaries.upgrade_projection)
            # Large counts amplify sums; compare normalized learned summaries
            # in the units consumed by the network, not relative to near zero.
            normalizer=counts[:,:,1:].sum(2,keepdim=True).clamp_min(1)
            torch.testing.assert_close(got/normalizer,want/normalizer,rtol=3e-5,atol=3e-6)
            torch.testing.assert_close(sizes,counts[:,:,1:].sum(2,keepdim=True),rtol=0,atol=0)
    finally:
        scorer.close()


@pytest.mark.cuda
def test_resident_frozen_entity_projection_preserves_unbounded_upgrades(latent_env):
    from rl.search.resident_deck import ResidentDeckScorer
    policy = policy_for(latent_env, 'transformer64x2').cuda()
    policy.set_training_mode(False)
    scorer = ResidentDeckScorer(policy)
    obs = {k: v.cuda() for k, v in tensor_obs(latent_env, 4).items()}
    for prefix in ('hand', 'select_card'):
        ids, upgrades = obs[prefix+'_ids'], obs[prefix+'_upgrades']
        ids.copy_(torch.arange(ids.numel(), device='cuda').reshape_as(ids) % policy.core.combat.n_cards)
        upgrades.copy_(torch.arange(upgrades.numel(), device='cuda').reshape_as(upgrades) / 4)
    for key, count in (('monster_id', policy.core.combat.monster_id.num_embeddings),
                       ('monster_move', policy.core.combat.monster_move.num_embeddings),
                       ('monster_intent_cat', policy.core.combat.monster_intent.num_embeddings)):
        obs[key].copy_(torch.arange(obs[key].numel(), device='cuda').reshape_as(obs[key]) % count)
    obs['monster_scalars'].normal_()
    obs['monster_buffs'].normal_()
    try:
        with torch.no_grad():
            expected = policy.core.combat._dynamic_inputs(obs, scorer.card_table)
            frozen = dict(obs, _resident_card_projection=(scorer.projected_cards, scorer.card_upgrade),
                          _resident_monster_projection=scorer.monster_tables)
            actual = policy.core.combat._dynamic_inputs(frozen, scorer.card_table)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for key in expected[1]:
            torch.testing.assert_close(actual[1][key], expected[1][key], rtol=3e-5, atol=3e-6)
        with pytest.raises(ValueError, match='cannot be used for training'):
            policy.core.combat._dynamic_inputs(frozen, scorer.card_table)
    finally:
        scorer.close()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_repeated_status_cards_keep_three_pile_tokens_and_exact_counts(latent_env, device):
    from tests.bridge_fixtures import native_card, native_card_state
    from env.core import encode

    enter_combat(latent_env)
    policy = policy_for(latent_env, 'transformer64x2').to(device)
    policy.set_training_mode(False)
    net = policy.core.combat
    widths = []
    hook = net.blocks[0].register_forward_pre_hook(lambda _, args: widths.append(args[0].shape[1]))
    try:
        for count in (1, 207):
            piles = {zone: [native_card('DAZED', 1 + i * count + j, base_cost=-2, cost=-2)
                           for j in range(count)]
                     for i, zone in enumerate(('draw', 'discard', 'exhaust'))}
            latent_env.thick.bc.set_card_state(native_card_state(
                hand=[native_card('STRIKE_RED', 0)], next_uid=1 + 3 * count, **piles))
            latent_env._invalidate_backend_cache()
            obs = {k: v.to(device) for k, v in tensor_obs(latent_env).items()}
            status = encode._card_id_idx('Dazed')
            for zone in piles:
                assert obs[zone + '_counts'][0, status].item() == count
            with torch.no_grad():
                summary = net.summaries.rebuild(obs)
                assert summary.piles.shape == (1, 3, 64)
                assert summary.pile_counts.flatten().tolist() == [count] * 3
                context = policy.core.prepare_search_context(obs)
                _, entities = net.encode(dict(obs, _combat_context=context.combat))
                for i in range(3):
                    assert entities[f'pile{i}'].shape == (1, 1, 64)
        assert widths[0] == widths[1] <= 32
    finally:
        hook.remove()


def test_public_context_present_in_combat_and_checkpoint_replay(latent_env):
    env = latent_env
    enter_combat(env)
    obs = env._obs()
    assert obs['deck_ids'].any()
    assert obs['map_full_rooms'].any()
    assert obs['map_full_features'][..., 2].any()  # current node
    assert obs['deck_latent_context'].item() == 1
    state = env.export_training_state()
    env.step(int(np.flatnonzero(env.action_masks())[0]))
    env.restore_training_state(state)
    for key, value in obs.items():
        np.testing.assert_array_equal(env._obs()[key], value)
    incompatible = deepcopy(state); incompatible['deck_latent_context'] = False
    with pytest.raises(ValueError, match='deck latent'):
        env.restore_training_state(incompatible)


@pytest.mark.parametrize('encoder', ['pooled-mlp', 'transformer64x2'])
def test_all_phases_gradients_cache_invalidation_and_direct_search_interface(latent_env, encoder):
    env = latent_env
    policy = policy_for(env, encoder)
    for combat in (False, True):
        if combat:
            enter_combat(env)
        obs = tensor_obs(env, 2)
        policy.set_training_mode(True)
        logits, values = policy.core.actor_critic(obs)
        assert logits.shape == (2, env.action_space.n)
        assert values.shape == (2, 1)
        (values.square().sum() + logits[logits > -1e8].square().mean()).backward()
        assert policy.core.context.card_projection.weight.grad.abs().sum() > 0
        policy.zero_grad(set_to_none=True)
        policy.set_training_mode(False)
        with torch.no_grad():
            reference = policy.core.actor_critic(obs)
            cached = policy.core.actor_critic(obs)
            for a, b in zip(reference, cached, strict=True):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            obs['relic_counters'] = obs['relic_counters'] + .1
            z = policy.core.run_context(obs)
            torch.testing.assert_close(z, policy.core._context_cache[2], rtol=0, atol=0)
            obs['outer_deck_misc'][:, 0] += .2
            changed = policy.core.run_context(obs)
            assert not torch.allclose(z, changed)
            policy.core.context.roles.add_(.03)
            refreshed = policy.core.run_context(obs)
            torch.testing.assert_close(refreshed, policy.core.context(obs), atol=2e-6, rtol=2e-5)
            if combat:
                direct = policy.core.combat_with_context(obs, refreshed)
                routed = policy.core.actor_critic(obs)
                for a, b in zip(direct, routed, strict=True):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_anchor_uses_branch_hp_and_ignores_unrevealed_rewards(latent_env):
    env = latent_env
    enter_combat(env)
    policy = policy_for(env)
    policy.set_training_mode(False)
    obs = tensor_obs(env)
    obs['combat_exit_anchor'][0, 0] = 1.
    with torch.no_grad():
        z = policy.core.run_context(obs)
        projected = policy.core.anchor.context_projection(z)
        baseline = policy.core.anchor_with_context(obs, projected)
        obs['run_hp'] += .5
        obs['reward_gold'] += 1
        obs['offer_card_ids'] += 1
        torch.testing.assert_close(policy.core.anchor_with_context(obs, projected), baseline, rtol=0, atol=0)
        obs['player_scalars'][:, 0] += .3
        assert not torch.allclose(policy.core.anchor_with_context(obs, projected), baseline)
        logits, value = policy.core.actor_critic(obs)
        torch.testing.assert_close(value, policy.core.anchor_with_context(obs, projected))
        assert not logits.any()


@pytest.mark.parametrize('scenario', ['ordinary', 'lethal', 'ARMAMENTS', 'SECRET_TECHNIQUE', 'HEADBUTT'])
def test_native_search_frontiers_match_reference_observations_and_results(latent_env, scenario):
    import slaythespire as sts
    from env.core.native_observation import NativeRootObservationBatch
    from env.search.anchored_chains import paired_anchor_search
    from rl.search.deck_latent import DeckLatentScorer

    env = latent_env
    enter_combat(env)
    if scenario == 'lethal':
        env.thick.bc.set_player_strength(999)
    elif scenario != 'ordinary':
        env.thick.bc.apply_custom_state(dict(hand=[scenario, 'DEFEND_RED', 'STRIKE_RED'],
            draw=['DEFEND_RED', 'SHRUG_IT_OFF', 'STRIKE_RED'], discard=['BASH', 'ANGER']))
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._invalidate_backend_cache()
    original = env._array_mapping_digest(env._obs())
    journal = deepcopy(env._episode_action_journal)
    policy = policy_for(env, 'transformer64x2'); policy.set_training_mode(False)
    encoder = NativeRootObservationBatch(env)
    encoded_rows = []
    native_encode = encoder.encode
    def checked_encode(branches):
        expected = [branch._obs() for branch in branches]
        actual = native_encode(branches)
        for row, reference in enumerate(expected):
            assert actual.keys() == reference.keys()
            for key in reference:
                assert actual[key][row].tobytes() == reference[key].tobytes(), key
        encoded_rows.append(len(branches))
        return actual
    encoder.encode = checked_encode
    runs = []
    for batch in (None, encoder):
        trace = []
        scorer = DeckLatentScorer(policy, [env._obs()])
        def score(requests):
            trace.extend((r.root, r.kind, env._array_mapping_digest(r.observation),
                          r.action_mask.tobytes()) for r in requests)
            return scorer(requests)
        result, stats = paired_anchor_search([env], [64], score, seed=20261111,
            max_depth=12, max_active=16, timeout_seconds=30., observation_batch=batch)
        runs.append((result, {k: v for k, v in stats.items() if k != 'seconds'}, trace))
        assert env._array_mapping_digest(env._obs()) == original
        assert env._episode_action_journal == journal
        assert not env._defer_combat_observation and not env._combat_observation_pending
    assert encoded_rows
    assert runs[0][1:] == runs[1][1:]
    for key in ('actions', 'values', 'visits'):
        np.testing.assert_array_equal(runs[0][0][0][key], runs[1][0][0][key])


@pytest.mark.parametrize('encoder', ['pooled-mlp', 'transformer64x2'])
def test_search_head_projection_refreshes_branch_context_and_preserves_siblings(latent_env, encoder):
    from env.search.anchored_chains import ChainRequest
    from rl.search.deck_latent import DeckLatentScorer
    enter_combat(latent_env)
    policy = policy_for(latent_env, encoder); policy.set_training_mode(False)
    root = latent_env._obs()
    scorer = DeckLatentScorer(policy, [root])
    rows = [ChainRequest(0, {k: v.copy() for k, v in root.items()},
                        latent_env.action_masks(), kind) for kind in ('combat', 'anchor', 'cutoff')]
    rows[1].observation['combat_exit_anchor'][0] = 1
    # A permanent card change belongs to one branch; later resource changes
    # retain that branch's refreshed context and leave its sibling untouched.
    rows[0].observation['outer_deck_misc'][0] += .2
    rows[1].observation['deck_upgrades'][0] += .2
    for iteration in range(2):
        if iteration:
            for row in rows:
                row.observation['player_scalars'][0] *= .8
                row.observation['relic_counters'][0] += .1
        actual = scorer(rows)
        assert scorer.context_refreshes == 2
        with torch.no_grad():
            for request, (logits, value) in zip(rows, actual, strict=True):
                full = obs_as_tensor({k: v[None] for k, v in request.observation.items()}, 'cpu')
                context = policy.core.prepare_search_context(full)
                if request.kind == 'anchor':
                    expected_value = policy.core.anchor_with_context(full, context)
                else:
                    expected_logits, expected_value = policy.core.combat_with_context(full, context)
                    np.testing.assert_allclose(logits, expected_logits[0], rtol=3e-5, atol=3e-6)
                np.testing.assert_allclose(value, expected_value.item(), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('pattern', ['holes', 'global_only', 'all_present'])
def test_context_padding_compaction_preserves_output_and_gradients(latent_env, device, pattern):
    if device == 'cuda':
        assert torch.cuda.is_available(), 'explicit CUDA context qualification required'
    torch.manual_seed(20261111)
    policy = DeckLatentPolicy(latent_env.observation_space, latent_env.action_space, lambda _: 5e-5,
        d_model=32, n_layers=2, n_heads=4, room_lambda=1., reward_flow='sequential-v1').to(device)
    actual = policy.core.context.train()
    reference = deepcopy(actual).train()
    mask = torch.zeros(3, 24, dtype=torch.bool, device=device)
    if pattern == 'holes':
        mask[:, 16:] = True
        mask[0, 1:15:2] = True
        mask[1, 3:16:3] = True
        mask[2, 1:] = True
    elif pattern == 'global_only':
        mask[:, 1:] = True
    # Masked positions deliberately contain nonzero values; PAD identities
    # alone must not be used to decide which queries can be removed.
    x = torch.randn(3, 24, 32, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    widths = []
    hook = actual.transformer.layers[0].register_forward_pre_hook(
        lambda _, args: widths.append(args[0].shape[1]))
    try:
        expected = reference.norm(reference.transformer(y, src_key_padding_mask=mask))[:, 0]
        result = actual._pool_tokens(x, mask)
        assert widths[-1] == int((~mask).sum(1).max())
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
        weight = torch.randn_like(result)
        (result * weight).mean().backward()
        (expected * weight).mean().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=3e-4, atol=2e-6)
        for (name, parameter), (other_name, other) in zip(
                actual.named_parameters(), reference.named_parameters(), strict=True):
            assert name == other_name and (parameter.grad is None) == (other.grad is None)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-4, atol=2e-6, msg=name)
        actual.eval()
        with torch.no_grad():
            result = actual._pool_tokens(x.detach(), mask)
        assert widths[-1] == 24  # fixed inference and collection-graph geometry
    finally:
        hook.remove()


def test_duplicate_context_gradients_equal_full_recomputation(latent_env):
    core = policy_for(latent_env).core
    obs = tensor_obs(latent_env, 3)
    obs['outer_deck_misc'][1, 0] = .4
    params = tuple(core.context.parameters())
    a = core.run_context(obs)
    ga = torch.autograd.grad(a.square().sum(), params, allow_unused=True)
    b = core.context(obs)
    gb = torch.autograd.grad(b.square().sum(), params, allow_unused=True)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    for x, y in zip(ga, gb, strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, atol=3e-5, rtol=1e-3)


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('transport', ['dummy', 'native-subproc'])
@pytest.mark.parametrize('context_mode', ['off', 'replay', 'grouped', 'screens', 'context-limit', 'direct-mask', 'prefetch'])
def test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path, search, transport, context_mode,
                                                       encoder='pooled-mlp', device='cpu', search_backend='native-cpu',
                                                       action_selection='shadow', capture_result=False):
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.packed_ppo import configure_packed_updates
    from rl.training.checkpoints import save_bundle, restore_learner
    from rl.training.environments import TrainingStateWrapper
    from rl.training.reward import StrategicRewardWrapper
    from rl.policies.checkpoint import load_checkpoint

    def factory():
        return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(SlayTheSpireEnv(
            card_language=True, global_map_view=True, upper_public_context=True,
            event_option_observation=True, combat_action_contract='autoregressive-v2',
            combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True)), 6000))
    def vector():
        if transport == 'dummy':
            return DummyVecEnv([factory, factory])
        from rl.training.native_subproc_vec_env import NativeSubprocVecEnv
        return NativeSubprocVecEnv([factory, factory], envs_per_worker=1, start_method='spawn')
    env = vector()
    executed_actions=[]
    original_step=env.step_async
    def record_actions(actions):
        executed_actions.append(np.array(actions,copy=True))
        return original_step(actions)
    env.step_async=record_actions
    slot_actions=[[],[]]
    if hasattr(env, 'step_subset'):
        original_subset=env.step_subset
        def record_subset(actions):
            for i,action in actions.items():
                slot_actions[i].append(action)
            return original_subset(actions)
        env.step_subset=record_subset
    model = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
        n_steps=32, batch_size=32, n_epochs=1, device=device, gamma=1., gae_lambda=1.,
        learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1',
                           deck_combat_encoder=encoder))
    configure_packed_updates(model, dict(token_budget=1024, max_rows=8, precision='fp32',
                                        direct_masked_categorical=context_mode == 'direct-mask',
                                        prefetch_groups=context_mode == 'prefetch',
                                        context_gradient_replay=context_mode == 'replay',
                                        context_grouping=context_mode in ('grouped', 'screens', 'context-limit'),
                                        upper_screen_max_rows=4 if context_mode in ('screens', 'context-limit') else None,
                                        max_contexts_per_chunk=2 if context_mode == 'context-limit' else None))
    if search:
        from dataclasses import asdict
        from rl.algorithms.search_auxiliary import SearchTrainingConfig
        model.policy.combat_value_target = 'pre-reward-anchor'
        model.policy_kwargs['combat_value_target'] = 'pre-reward-anchor'
        model.deck_search_config = asdict(SearchTrainingConfig(fraction=1., normal_budget=2,
            action_selection=action_selection,
            boss_budget=2, max_depth=2, max_active=16, timeout_seconds=30., backend=search_backend,
            cuda_artifacts=__import__('os').environ.get('STS_CUDA_BATTLE_ARTIFACTS', '') if search_backend == 'cuda-resident' else '',
            cuda_wave_capacity=64,
            inference_cache_bytes=64 * 1024**2))
        model.deck_search_snapshot_dir = str(tmp_path / 'weights')
    bootstrap_checks = []
    def check_bootstrap_memory(learner):
        original = learner.policy.predict_values
        def predict_values(obs):
            if learner.rollout_buffer.full:
                assert learner._resident_search_service.parked, 'tail bootstrap needs the search pool released'
                bootstrap_checks.append(learner.num_timesteps)
            return original(obs)
        learner.policy.predict_values = predict_values
    if search_backend == 'cuda-resident':
        check_bootstrap_memory(model)
    from stable_baselines3.common.callbacks import BaseCallback
    class SearchMetrics(BaseCallback):
        def _on_step(self): return True
        def _on_rollout_end(self):
            if search_backend == 'cuda-resident' and action_selection != 'shadow':
                assert self.model.logger.name_to_value['pure_rl/search_simulations'] == self.model.deck_search_last_stats['chains'] > 0
    model.learn(128, callback=SearchMetrics())
    assert model.num_timesteps == 128
    if search:
        assert model.deck_search_last_stats['searched_roots'] > 0
        assert len(model.rollout_buffer.search_targets) > 0
        assert model._deck_search_loss_normalization > 0
        assert model.deck_search_last_stats['atomic_steps'] > model.deck_search_last_stats['searched_roots']
        if action_selection != 'shadow':
            assert model.deck_search_last_stats['executed_search_actions'] == len(model.rollout_buffer.search_actions)>0
            actual=(np.asarray([row[-32:] for row in slot_actions]).T
                    if any(slot_actions) else np.asarray(executed_actions[-32:]))
            np.testing.assert_array_equal(actual,model.rollout_buffer.actions[:,:,0])
            for (position,index),(action,logp) in model.rollout_buffer.search_actions.items():
                assert model.rollout_buffer.actions[position,index,0]==action
                assert model.rollout_buffer.action_masks[position,index,action]
                assert model.rollout_buffer.log_probs[position,index]==pytest.approx(logp,abs=1e-6)
        if search_backend == 'cuda-resident':
            assert model.deck_search_last_stats['backend_cuda_simulation'] == 1
            service=model._resident_search_service
            assert service.parked and service.status['generation']==2
            assert service.status['park']['reserved'] < 256*1024**2
        if search_backend == 'cuda-batched-native':
            assert model.deck_search_last_stats['backend_cuda_inference'] == 1
            assert model.deck_search_last_stats['backend_cuda_simulation'] == 0
            assert model.deck_search_last_stats['frontier_rows'] > 0
            assert model.deck_search_last_stats['scoring_seconds'] > 0
            assert not (tmp_path / 'weights' / 'frozen.pt').exists()
    if context_mode == 'grouped':
        model._ppo_microbatch_caps = {'upper': 2}
    source = save_bundle(model, tmp_path/'first', rank=0, world_size=1, run_id='deck-latent-test',
                         wandb_id=None, config={'ascension': 0})
    loaded = load_checkpoint(source/'model.zip', device=device)
    assert not hasattr(loaded,'_resident_search_service')
    if search_backend == 'cuda-resident':
        from rl.search.resident_service import close_collection_service
        close_collection_service(model)
        assert not service.process.is_alive()
    assert isinstance(loaded.policy, DeckLatentPolicy)
    assert loaded.policy.deck_combat_encoder == encoder
    assert loaded.policy.core.combat_encoder == encoder
    assert loaded.ppo_update_config['context_gradient_replay'] == (context_mode == 'replay')
    assert loaded.ppo_update_config['context_grouping'] == (context_mode in ('grouped', 'screens', 'context-limit'))
    assert loaded.ppo_update_config.get('upper_screen_max_rows') == (4 if context_mode in ('screens', 'context-limit') else None)
    assert loaded.ppo_update_config.get('max_contexts_per_chunk') == (2 if context_mode == 'context-limit' else None)
    if context_mode == 'grouped':
        assert loaded._ppo_microbatch_caps == {'upper': 2}
    if search:
        assert loaded.deck_search_config == model.deck_search_config
        assert loaded.deck_search_rng == model.deck_search_rng
    target = vector()
    restored = AccumulatingMaskablePPO.load(source/'model.zip', env=target, device=device)
    restored._setup_learn(64, reset_num_timesteps=False)
    restore_learner(restored, source, rank=0, world_size=1)
    assert restored._recovery_environment_restored
    for a, b in zip(model.policy.parameters(), restored.policy.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if context_mode == 'grouped' and not search:
        from rl.algorithms.deck_schedule import configure, apply_stage
        schedule = configure(restored, dict(anchor_start_round=2))
        apply_stage(restored, schedule, 2)
        assert restored.policy.combat_value_target == 'pre-reward-anchor'
    if search_backend == 'cuda-resident':
        assert set(bootstrap_checks) == {64, 128}
        check_bootstrap_memory(restored)
    restored.learn(128, reset_num_timesteps=False)
    assert restored.num_timesteps == 256
    resumed = save_bundle(restored, tmp_path/'resumed', rank=0, world_size=1,
        run_id='deck-latent-test', wandb_id=None, config={'ascension': 0})
    assert resumed.is_dir()
    if search_backend == 'cuda-resident':
        assert set(bootstrap_checks) == {64, 128, 192, 256}
        assert restored._resident_search_service.parked
        assert restored._resident_search_service.status['generation']==2
        close_collection_service(restored)
    if context_mode == 'grouped' and not search:
        again = load_checkpoint(resumed/'model.zip', device='cpu')
        assert again.policy.combat_value_target == 'pre-reward-anchor'
        assert again.deck_training_state['stage'] == 'anchor'
        assert again.deck_training_state['stage_start'] == 2
        assert again.deck_training_state['good_checks'] == 0
    result = None
    if capture_result:
        result = dict(observations={k:v.copy() for k,v in restored.rollout_buffer.observations.items()},
            actions=restored.rollout_buffer.actions.copy(),rewards=restored.rollout_buffer.rewards.copy(),
            episode_starts=restored.rollout_buffer.episode_starts.copy(),
            targets={k:v.copy() for k,v in restored.rollout_buffer.search_targets.items()},
            rng=restored.deck_search_rng, search_stats=dict(restored.deck_search_last_stats))
    env.close(); target.close()
    return result


def test_cpu_shadow_search_native_rounds_and_resume(tmp_path):
    test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path, True, 'native-subproc',
        'grouped', encoder='transformer64x2', device='cpu')


def test_cuda_shadow_search_native_rounds_and_resume(tmp_path):
    assert torch.cuda.is_available(), 'explicit CUDA search qualification required'
    test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path, True, 'native-subproc',
        'grouped', encoder='transformer64x2', device='cuda')


def test_cuda_batched_search_native_rounds_and_resume(tmp_path):
    assert torch.cuda.is_available(), 'explicit CUDA search qualification required'
    test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path, True, 'native-subproc',
        'grouped', encoder='transformer64x2', device='cuda', search_backend='cuda-batched-native')


def test_cuda_resident_search_native_rounds_and_resume(tmp_path, cuda_battle_artifacts):
    test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path, True, 'native-subproc',
        'grouped', encoder='transformer64x2', device='cuda', search_backend='cuda-resident')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_external_frontiers_preserve_teacher_and_root(latent_env, device):
    from env.search.anchored_chains import paired_anchor_search, ChainRequest
    from env.search.frontier_transport import FrontierSession
    from env.core.native_observation import NativeRootObservationBatch
    from rl.search.deck_latent import DeckLatentScorer
    if device == 'cuda':
        assert torch.cuda.is_available()
    env = latent_env
    enter_combat(env)
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    obs = env._obs()
    policy = policy_for(env, 'transformer64x2').to(device)
    policy.set_training_mode(False)
    root = env.thick.bc.nosl_future_key()
    journal = deepcopy(env._episode_action_journal)
    options = dict(seed=20261111, max_depth=12, max_active=16, timeout_seconds=30.)
    budget = int(np.count_nonzero(env.action_masks())) * 2
    reference = DeckLatentScorer(policy, [obs])
    expected, expected_stats = paired_anchor_search([env], [budget], reference,
        observation_batch=NativeRootObservationBatch(env), **options)
    scorer = DeckLatentScorer(policy, [obs])
    session = FrontierSession([env])
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    packet = session.start(dict(indices=[0], budgets=[budget], generation=7,
        seed=options['seed'], options={k: v for k, v in options.items() if k != 'seed'},
        observation_keys=scorer.observation_keys, static_keys=scorer.context_keys))
    caches, seen = {}, []
    try:
        with pytest.raises(RuntimeError, match='stale or out-of-order'):
            session.advance(dict(generation=8, sequence=packet['sequence'], outputs=[]))
        while not packet['done']:
            requests = []
            for row, lane, kind, mask, delta in packet['rows']:
                # Static maps/deck bytes do not cross the pipe when unchanged.
                assert 'map_full_features' not in delta
                requests.append(ChainRequest(row, dict(obs, **delta), mask, kind,
                    caches.setdefault(lane, {}), lane))
                seen.append(kind)
            packet = session.advance(dict(generation=7, sequence=packet['sequence'], outputs=scorer(requests)))
        actual, stats = packet['result']
    finally:
        session.close()
    for key in ('actions', 'visits', 'values'):
        np.testing.assert_array_equal(actual[0][key], expected[0][key])
    assert {k: v for k, v in stats.items() if k != 'seconds'} == {
        k: v for k, v in expected_stats.items() if k != 'seconds'}
    assert 'anchor' in seen
    assert env.thick.bc.nosl_future_key() == root
    assert env._episode_action_journal == journal
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


def test_frontier_cancel_closes_forks_without_advancing_root(latent_env, monkeypatch):
    from env.search.anchored_chains import paired_anchor_frontiers
    env = latent_env
    enter_combat(env)
    before = env.thick.bc.nosl_future_key()
    fork = env.fork_run
    closed = []
    def tracked(**kwargs):
        child = fork(**kwargs)
        close = child.close
        def record():
            closed.append(child)
            close()
        child.close = record
        return child
    monkeypatch.setattr(env, 'fork_run', tracked)
    session = paired_anchor_frontiers([env], [64], seed=20261111, max_active=4)
    next(session)
    session.close()
    assert len(closed) == 4 and len({id(c) for c in closed}) == 4
    assert env.thick.bc.nosl_future_key() == before


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('encoder', ['pooled-mlp', 'transformer64x2'])
def test_search_result_cache_tracks_consumed_inputs_and_weights(latent_env, device, encoder):
    from env.search.anchored_chains import ChainRequest
    from rl.search.deck_latent import DeckLatentScorer, DeckSearchInferenceCache
    if device == 'cuda':
        assert torch.cuda.is_available()
    env = latent_env
    enter_combat(env)
    policy = policy_for(env, encoder).to(device)
    policy.set_training_mode(False)
    obs = env._obs()
    cached = DeckLatentScorer(policy, [obs], inference_cache=DeckSearchInferenceCache(policy.core))
    plain = DeckLatentScorer(policy, [obs], inference_cache=DeckSearchInferenceCache(policy.core, 0))
    mask = env.action_masks().copy()
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None

    def compare(observation, kind):
        requests = [ChainRequest(0, observation, mask, kind) for _ in range(2)]
        expected, actual = plain(requests), cached(requests)
        for (a, av), (b, bv) in zip(expected, actual, strict=True):
            if a is not None:
                np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)
            np.testing.assert_allclose(av, bv, rtol=3e-5, atol=3e-6)
        return actual

    for kind in ('combat', 'anchor'):
        first = compare(obs, kind)
        side = getattr(cached.inference_cache, kind)
        assert side.stats['misses'] == 1 and side.stats['batch_duplicates'] == 1
        compare(obs, kind)
        assert side.stats['hits'] == 2
        # Exact cache outputs cannot be mutated by their consumer.
        if first[0][0] is not None:
            first[0][0][:] = 12345
        compare(obs, kind)
        for field, position, delta in [('player_scalars', 0, -.01), ('potions', 0, 1),
                                      ('gold', 0, .01), ('relic_counters', 0, .01),
                                      ('outer_deck_misc', 0, .01)]:
            changed = deepcopy(obs)
            changed[field].flat[position] += delta
            misses = side.stats['misses']
            compare(changed, kind)
            assert side.stats['misses'] == misses + 1, field
        if kind == 'anchor':
            changed = deepcopy(obs)
            changed['player_scalars'][2:] += .125
            changed['combat_counters'] += .125
            misses = side.stats['misses']
            compare(changed, kind)
            assert side.stats['misses'] == misses  # Not consumed by this head.
        else:
            for field in ('select_native_indices', 'select_hand_mask', 'combat_legal'):
                changed = deepcopy(obs)
                changed[field].flat[0] = 1 - changed[field].flat[0]
                misses = side.stats['misses']
                compare(changed, kind)
                assert side.stats['misses'] == misses + 1, field
        # Reuse across searches in the same frozen collection.
        other = DeckLatentScorer(policy, [obs], context=cached.context,
                                inference_cache=cached.inference_cache)
        before = side.stats['hits']
        other([ChainRequest(0, obs, mask, kind)])
        assert side.stats['hits'] == before + 1
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    with torch.no_grad():
        next(policy.core.parameters()).add_(.001)
    with pytest.raises(RuntimeError, match='weights changed'):
        cached([ChainRequest(0, obs, mask, 'combat')])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_search_result_cache_eviction_and_full_input_reference(latent_env, device):
    import gc
    from env.search.anchored_chains import ChainRequest
    from rl.policies.combat_summary import CombatSummary
    from rl.search.deck_latent import DeckLatentScorer, DeckSearchInferenceCache
    if device == 'cuda':
        assert torch.cuda.is_available()
    env = latent_env
    enter_combat(env)
    policy = policy_for(env, 'transformer64x2').to(device)
    policy.set_training_mode(False)
    obs = env._obs()
    cache = DeckSearchInferenceCache(policy.core, 32768)
    scorer = DeckLatentScorer(policy, [obs], inference_cache=cache)
    for i in range(18):
        changed = deepcopy(obs)
        changed['player_scalars'][0] -= .001 * i
        # Changing active pending width cannot reuse a different execution width.
        if i % 2:
            changed['select_native_indices'][20] = 1
        request = ChainRequest(0, changed, env.action_masks(), 'combat')
        with torch.no_grad():
            full = {k: torch.as_tensor(v[None], device=device) for k, v in changed.items()}
            state = scorer.tables.new_state(changed)
            summary = CombatSummary(*(torch.as_tensor(getattr(state, k)[None], device=device)
                                      for k in ('piles', 'counts', 'relics')))
            logits, value = policy.core.combat_with_context(full, scorer.context, summary)
            actual, av = scorer([request])[0]
            np.testing.assert_allclose(actual, logits[0].cpu().numpy(), rtol=3e-5, atol=3e-6)
            np.testing.assert_allclose(av, value.item(), rtol=3e-5, atol=3e-6)
    assert cache.combat.stats['evictions'] > 0
    assert cache.combat.stats['bytes'] <= cache.combat.max_bytes
    cache.clear()
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    assert cache.combat.stats['bytes'] == cache.anchor.stats['bytes'] == 0
    misses = cache.combat.stats['misses']
    scorer([ChainRequest(0, obs, env.action_masks(), 'combat')])
    assert cache.combat.stats['misses'] == misses + 1


def test_cuda_actor_search_context_survives_graph_changes(latent_env):
    from rl.policies.collection_graphs import SharedCollectionGraphs
    assert torch.cuda.is_available(), 'explicit CUDA search qualification required'
    enter_combat(latent_env)
    policy = policy_for(latent_env, 'transformer64x2').to('cuda')
    policy.set_training_mode(False)
    core = policy.core
    runtime = SharedCollectionGraphs(core, capacity=1)
    core._collection_graphs = runtime
    try:
        with torch.no_grad():
            for iteration, count in enumerate((3, 17, 3, 1)):
                host = {k: np.repeat(v[None], count, 0) for k, v in latent_env._obs().items()}
                host['outer_deck_misc'][:, 0] += iteration * .1
                tensors = obs_as_tensor(host, 'cuda')
                if iteration == 2:
                    core.context.roles.add_(.03)
                    torch.cuda.synchronize()
                    torch._C._cuda_clearCublasWorkspaces()
                    torch.cuda.empty_cache()
                # A previous eager cache may exist but must never supply a
                # different observation's context to the graph search hook.
                core.run_context(obs_as_tensor({k: v[:1] for k, v in host.items()}, 'cuda'))
                runtime.bind_host_observations(tensors, host)
                core.actor_critic(tensors)
                rows = torch.arange(count-1, -1, -1, device='cuda')
                actual = core.actor_search_context(rows)
                expected = core.prepare_search_context({k: v[rows] for k, v in tensors.items()})
                for a, b in zip((actual.combat, actual.anchor), (expected.combat, expected.anchor)):
                    torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
            assert runtime.stats['evictions'] > 0
            # The same runtime also supports its eager fallback (no host bind).
            tensors['outer_deck_misc'][:, 0] += .5
            core.actor_critic(tensors)
            actual = core.actor_search_context(rows)
            expected = core.prepare_search_context(tensors)
            torch.testing.assert_close(actual.combat, expected.combat, rtol=3e-5, atol=3e-6)
    finally:
        core.__dict__.pop('_collection_graphs', None)
        runtime.close()


@pytest.mark.parametrize('encoder', ['pooled-mlp', 'transformer64x2'])
def test_paired_shadow_search_batches_real_anchors_without_mutating_root(latent_env, encoder):
    from env.search.anchored_chains import paired_anchor_search
    from rl.search.deck_latent import DeckLatentScorer
    env = latent_env
    enter_combat(env)
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    obs = env._obs()
    before = env._array_mapping_digest(obs)
    root_key = env.thick.bc.nosl_future_key()
    journal = deepcopy(env._episode_action_journal)
    policy = policy_for(env, encoder); policy.set_training_mode(False)
    scorer = DeckLatentScorer(policy, [obs])
    seen = []
    def score(requests):
        seen.extend(r.kind for r in requests)
        result = scorer(requests)
        with torch.no_grad():
            for request, (logit, value) in zip(requests, result, strict=True):
                full = obs_as_tensor({k: v[None] for k, v in request.observation.items()}, 'cpu')
                expected_logit, expected_value = policy.core.actor_critic(full)
                np.testing.assert_allclose(value, expected_value.item(), rtol=3e-5, atol=3e-6)
                if logit is not None:
                    np.testing.assert_allclose(logit, expected_logit[0], rtol=3e-5, atol=3e-6)
        return result
    count = np.count_nonzero(env.action_masks())
    result, stats = paired_anchor_search([env], [count * 2], score, seed=20261111,
        max_depth=12, max_active=16, timeout_seconds=30)
    assert stats['chains'] == count * 2
    assert stats['max_active'] > 1
    assert (result[0]['visits'] == 2).all()
    assert np.isfinite(result[0]['values']).all()
    assert 'anchor' in seen
    assert env._array_mapping_digest(env._obs()) == before
    assert env.thick.bc.nosl_future_key() == root_key
    assert env._episode_action_journal == journal
    # Changing the frozen policy must invalidate the entire teacher.
    with torch.no_grad():
        policy.core.combat.end.bias.add_(.1)
    with pytest.raises(RuntimeError, match='weights changed'):
        scorer([])


@pytest.mark.parametrize('device,backend', [('cpu','native-cpu'),('cuda','native-cpu'),
                                             ('cuda','cuda-resident')])
def test_shadow_teaching_preserves_real_actions_rewards_and_sampling_rng(device,backend,tmp_path):
    from dataclasses import asdict
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.packed_ppo import configure_packed_updates
    from rl.algorithms.search_auxiliary import SearchTrainingConfig
    from rl.training.reward import StrategicRewardWrapper

    if device == 'cuda':
        assert torch.cuda.is_available(), 'explicit CUDA search qualification required'

    class Capture(BaseCallback):
        def _on_step(self):
            return True
        def _on_rollout_end(self):
            self.actions = self.model.rollout_buffer.actions.copy()
            self.rewards = self.model.rollout_buffer.rewards.copy()
            self.rng = torch.get_rng_state().clone()
            self.cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None

    captures = []
    for searched in (False, True):
        factory = lambda: StrategicRewardWrapper(SlayTheSpireEnv(
            card_language=True, global_map_view=True, upper_public_context=True,
            event_option_observation=True, combat_action_contract='autoregressive-v2',
            combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True))
        if backend == 'cuda-resident':
            from rl.training.native_subproc_vec_env import NativeSubprocVecEnv
            env = NativeSubprocVecEnv([factory],envs_per_worker=1,start_method='spawn')
        else:
            env = DummyVecEnv([factory])
        model = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
            n_steps=24, batch_size=24, n_epochs=1, device=device, gamma=1., gae_lambda=1.,
            learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
            policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1.,
                reward_flow='sequential-v1', combat_value_target='pre-reward-anchor',
                deck_combat_encoder='transformer64x2'))
        configure_packed_updates(model, dict(token_budget=1024, max_rows=8, precision='fp32'))
        if searched:
            model.deck_search_config = asdict(SearchTrainingConfig(fraction=1., normal_budget=2,
                boss_budget=2, max_depth=2, max_active=16, timeout_seconds=30.,backend=backend,
                cuda_artifacts=__import__('os').environ.get('STS_CUDA_BATTLE_ARTIFACTS','')
                    if backend == 'cuda-resident' else '',cuda_wave_capacity=64))
            model.deck_search_snapshot_dir = str(tmp_path / 'weights')
        callback = Capture()
        model.learn(24, callback=callback)
        if searched:
            assert model.deck_search_last_stats['searched_roots'] > 0
        captures.append(callback)
        env.close()
    np.testing.assert_array_equal(captures[0].actions, captures[1].actions)
    np.testing.assert_array_equal(captures[0].rewards, captures[1].rewards)
    torch.testing.assert_close(captures[0].rng, captures[1].rng, rtol=0, atol=0)
    if device == 'cuda':
        torch.testing.assert_close(captures[0].cuda_rng, captures[1].cuda_rng, rtol=0, atol=0)


@pytest.mark.parametrize('card,zone', [('ARMAMENTS', 1), ('SECRET_TECHNIQUE', 2), ('HEADBUTT', 3)])
def test_root_selection_identity_depends_on_candidate_zone(latent_env, card, zone):
    import slaythespire as sts
    from env.search.anchored_chains import paired_anchor_search
    env = latent_env
    enter_combat(env)
    env.thick.bc.apply_custom_state(dict(hand=[card, 'DEFEND_RED', 'STRIKE_RED'],
        draw=['DEFEND_RED', 'SHRUG_IT_OFF', 'STRIKE_RED'], discard=['BASH', 'ANGER']))
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._invalidate_backend_cache()
    assert env._obs()['select_zone'].item() == zone
    before = env._array_mapping_digest(env._obs())
    legal = np.flatnonzero(env.action_masks())
    result, stats = paired_anchor_search([env], [2 * len(legal)],
        lambda requests: [(None, 0.) for r in requests], seed=20261111, max_depth=1,
        max_active=16, timeout_seconds=10)
    assert (result[0]['visits'] == 2).all()
    assert env._array_mapping_digest(env._obs()) == before


def test_anchor_validation_uses_completed_natural_games_and_marks_limits(latent_env):
    from types import SimpleNamespace
    from rl.evaluation.anchor_readiness import anchor_readiness
    from rl.evaluation.runner import EvaluationConfig, Scenario
    policy = policy_for(latent_env)
    model = SimpleNamespace(policy=policy, observation_space=latent_env.observation_space,
        action_space=latent_env.action_space, device=torch.device('cpu'),
        nosl_action_group_contract='individual-actions')
    model.predict = policy.predict
    def factory():
        return SlayTheSpireEnv(card_language=True, global_map_view=True, upper_public_context=True,
            event_option_observation=True, combat_action_contract='autoregressive-v2',
            combat_exit_anchor=True, reward_flow='sequential-v1', deck_latent_context=True)
    # Real environment lifecycle with a strict small cap: no incomplete future
    # may become an anchor's "actual" return or a natural win/loss.
    result = anchor_readiness(model, Scenario('latent-readiness', factory),
                             EvaluationConfig(episodes=2, seed=20261111, num_envs=2, max_steps=8, timeout=10.))
    assert result['natural'] == 0
    assert result['censored'] == 2
    assert result['anchors'] == 0
    rng = torch.get_rng_state()
    natural = anchor_readiness(model, Scenario('latent-readiness', factory),
        EvaluationConfig(episodes=2, seed=20261111, num_envs=2, max_steps=6000, timeout=30.))
    assert natural['natural'] == 2 and natural['censored'] == 0
    assert natural['anchors'] > 0
    torch.set_rng_state(rng)
    response = anchor_readiness(model, Scenario('latent-readiness', factory),
        EvaluationConfig(episodes=2, seed=20261111, num_envs=2, max_steps=6000, timeout=30.), hp_response=True)
    for key in natural:
        assert response[key] == pytest.approx(natural[key], rel=1e-5, abs=1e-6), key
    rows = response['hp_response']['rows']
    assert len(rows) == response['anchors']
    assert all(row['outcome'] in ('death', 'win') for row in rows)
    assert all(row['hp_levels'][0] == 1 and row['hp_levels'][-1] == row['max_hp'] for row in rows)


@pytest.mark.parametrize('failure', [None, 'encode', 'encoder_backward'])
def test_context_replay_preserves_all_gradients_adam_and_failure_cleanup(latent_env, failure):
    from types import SimpleNamespace
    from rl.algorithms.deck_context import context_gradient_batch
    from rl.algorithms.packed_ppo import PackedPPOConfig
    env = latent_env
    outer = env._obs()
    enter_combat(env)
    combat = env._obs()
    anchor = deepcopy(combat); anchor['combat_exit_anchor'][0] = 1
    # Repeated static contexts span physical chunks and all three heads.
    observations = {k: torch.as_tensor(np.stack([x[k] for x in
        [outer, combat, anchor, combat, outer, combat, combat, anchor, outer, combat, anchor, combat]]))
        for k in outer}
    observations['outer_deck_misc'][:8, 0] += torch.arange(8) / 10
    observations['player_scalars'][3, 0] += .1
    reference = policy_for(env); reference.set_training_mode(True)
    candidate = deepcopy(reference)
    flags = []
    handle = None
    if failure == 'encode':
        def fail_forward(module, args):
            if not torch.is_grad_enabled() and not flags:
                flags.append(failure)
                raise torch.OutOfMemoryError('injected context forward failure')
        handle = candidate.core.context.register_forward_pre_hook(fail_forward)
    elif failure == 'encoder_backward':
        def fail_backward(gradient):
            if not flags:
                flags.append(failure)
                raise torch.OutOfMemoryError('injected partial context backward failure')
            return gradient
        handle = candidate.core.context.card_projection.weight.register_hook(fail_backward)
    config = PackedPPOConfig(1024, 8, context_gradient_replay=True)
    def loss(policy, obs, start):
        logits, values = policy.core.actor_critic(obs)
        # Nonconstant readouts exercise shared card, actor, anchor and outer heads.
        safe = logits.clamp(-20, 20)
        coefficients = torch.linspace(-.3, .4, safe.shape[1])
        return (safe * coefficients).sum() / 1200 + (
            values.flatten() * torch.arange(start + 1, start + len(values) + 1)).sum() / 12
    try:
        for update in range(2):
            for policy in (reference, candidate):
                policy.optimizer.zero_grad()
            for start in range(0, 12, 4):
                loss(reference, {k: v[start:start+4] for k, v in observations.items()}, start).backward()
            model = SimpleNamespace(policy=candidate)
            with context_gradient_batch(model, SimpleNamespace(observations=observations), config) as (batch, replay):
                assert replay.unique_rows < replay.input_rows
                for start in range(0, 12, 4):
                    loss(candidate, {k: v[start:start+4] for k, v in batch.observations.items()}, start).backward()
                replay.backward()
            assert candidate.core._context_gradient_batch is None
            for (name, a), (other, b) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
                assert name == other and (a.grad is None) == (b.grad is None), name
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-5, msg=name)
            reference.optimizer.step(); candidate.optimizer.step()
            for (name, a), (_, b) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6, msg=name)
                sa, sb = reference.optimizer.state.get(a, {}), candidate.optimizer.state.get(b, {})
                assert sa.keys() == sb.keys()
                for key in sa:
                    torch.testing.assert_close(sa[key], sb[key], rtol=4e-4, atol=4e-6, msg=name + '/' + key)
        assert flags == ([] if failure is None else [failure])
        with pytest.raises(RuntimeError, match='abort'):
            with context_gradient_batch(model, SimpleNamespace(observations=observations), config):
                raise RuntimeError('abort')
        assert candidate.core._context_gradient_batch is None
    finally:
        if handle is not None:
            handle.remove()


def test_ppo_context_adjoint_retry_does_not_double_count_samples(latent_env, tmp_path, monkeypatch):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.logger import configure
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.packed_ppo import configure_packed_updates, packed_ppo_epoch
    from rl.algorithms.ppo import configure_advantage_normalization, prepare_advantage_normalization
    from rl.algorithms.deck_context import ContextGradientBatch
    from rl.training.reward import StrategicRewardWrapper

    env = DummyVecEnv([lambda: StrategicRewardWrapper(latent_env)])
    source = AccumulatingMaskablePPO(DeckLatentPolicy, env, seed=20261111,
        n_steps=32, batch_size=16, n_epochs=1, device='cpu', gamma=1., gae_lambda=1.,
        learning_rate=5e-5, reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4, room_lambda=1., reward_flow='sequential-v1'))
    _, callback = source._setup_learn(32, callback=lambda *_: True)
    assert source.collect_rollouts(env, callback, source.rollout_buffer, 32)
    source.save(tmp_path/'source.zip')
    source_rng = np.random.get_state()
    outputs = []
    injected = []
    original_init = ContextGradientBatch.__init__
    for mode in ('reference', 'replay', 'grouped', 'retry'):
        model = AccumulatingMaskablePPO.load(tmp_path/'source.zip', device='cpu')
        model.rollout_buffer = deepcopy(source.rollout_buffer)
        model.set_logger(configure(folder=None, format_strings=[]))
        model.policy.set_training_mode(True)
        configure_advantage_normalization(model, 'rollout'); prepare_advantage_normalization(model)
        configure_packed_updates(model, dict(token_budget=1024, max_rows=8,
                                            context_gradient_replay=mode in ('replay', 'retry'),
                                            context_grouping=mode == 'grouped'))
        model._ppo_update_metric_sums = None; model._ppo_update_metric_count = 0
        np.random.set_state(source_rng)
        if mode == 'retry':
            def initialize(self, *args):
                original_init(self, *args)
                def fail(gradient):
                    if not injected:
                        injected.append(True)
                        raise torch.OutOfMemoryError('injected adjoint accumulation failure')
                    return gradient
                self.latents.register_hook(fail)
            monkeypatch.setattr(ContextGradientBatch, '__init__', initialize)
        result = packed_ppo_epoch(model, .2, None)
        assert result['optimizer_steps'] == 2
        assert model._ppo_update_metric_count == 32
        assert getattr(model.policy.core, '_context_gradient_batch', None) is None
        if mode == 'retry':
            assert injected == [True] and model._ppo_microbatch_stats['oom_retries'] == 1
        outputs.append(model)
    for candidate in outputs[1:]:
        for (name, a), (_, b) in zip(outputs[0].policy.named_parameters(), candidate.policy.named_parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6, msg=name)
            sa, sb = outputs[0].policy.optimizer.state.get(a, {}), candidate.policy.optimizer.state.get(b, {})
            assert sa.keys() == sb.keys()
            for key in sa:
                torch.testing.assert_close(sa[key], sb[key], rtol=4e-4, atol=4e-6, msg=name + '/' + key)

@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('transport', ['dummy', 'native-subproc'])
def test_transformer_two_round_lifecycle(tmp_path, search, transport):
    test_two_real_ppo_rounds_checkpoint_and_packed_resume(
        tmp_path, search, transport, 'grouped', encoder='transformer64x2')


def test_transformer_padding_candidates_and_card_interaction(latent_env, tmp_path):
    env = latent_env
    enter_combat(env)
    policy = policy_for(env, 'transformer64x2')
    policy.set_training_mode(False)
    obs = tensor_obs(env)
    # Same first row, but a second row with a far-end handle-only candidate
    # expands the entire attention batch. The first row must be unchanged.
    other = {k: v.clone() for k, v in obs.items()}
    other['select_native_indices'][0, -1] = 123
    other['select_card_ids'][0, -1] = 0
    other['select_card_costs'][0, -1] = .3
    mixed = {k: torch.cat((obs[k], other[k])) for k in obs}
    with torch.no_grad():
        reference = policy.core.actor_critic(obs)
        expanded = policy.core.actor_critic(mixed)
        for a, b in zip(reference, expanded):
            torch.testing.assert_close(a, b[:1], atol=3e-6, rtol=3e-5)
        z = policy.core.run_context(other)
        _, blocks = policy.core.combat.encode(dict(other, _run_context=z))
        assert blocks['select'].shape[1] == other['select_native_indices'].shape[1]
        assert blocks['select'][:, -1].abs().sum() > 0
        # Independent serialization must retain geometry, not silently load MLP.
        policy.save(tmp_path/'policy.pt')
        loaded = DeckLatentPolicy.load(tmp_path/'policy.pt', device='cpu')
        loaded.set_training_mode(False)
        for a, b in zip(policy.core.actor_critic(mixed), loaded.core.actor_critic(mixed)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    # A card query must interact with another card through attention, rather
    # than only its own lookup. Test the derivative on actual populated slots.
    policy.set_training_mode(True)
    obs['hand_damage'] = obs['hand_damage'].clone().requires_grad_(True)
    z = policy.core.run_context(obs)
    _, blocks = policy.core.combat.encode(dict(obs, _run_context=z))
    present = (obs['hand_ids'][0] != 0).nonzero().flatten()
    assert len(present) >= 2
    grad, = torch.autograd.grad(blocks['hand'][0, present[0], 0], obs['hand_damage'])
    assert grad[0, present[1]].abs() > 1e-9


@pytest.mark.parametrize('encoder', ['pooled-mlp', 'transformer64x2'])
def test_cuda_collection_graphs_route_deck_context_and_replay_after_eviction(latent_env, encoder):
    from rl.policies.collection_graphs import SharedCollectionGraphs
    assert torch.cuda.is_available(), 'explicit CUDA graph qualification required'
    torch.set_float32_matmul_precision('highest')
    core = policy_for(latent_env, encoder).core.cuda().eval()
    upper = latent_env._obs()
    enter_combat(latent_env)
    combat = latent_env._obs()
    anchor = deepcopy(combat)
    anchor['combat_exit_anchor'][0] = 1
    long_choice = deepcopy(combat)
    long_choice['select_native_indices'][47] = 48
    graphs = SharedCollectionGraphs(core, capacity=2)
    retained = []
    try:
        with torch.no_grad():
            for index, rows in enumerate(((upper, combat, anchor), (long_choice,), (upper,),
                                          (combat, anchor, upper), (combat, anchor, upper))):
                if index == 3:
                    core.context.roles.add_(.1)
                    core.combat.value_head[-1].bias.add_(.11)
                    torch._C._cuda_clearCublasWorkspaces()
                    torch.cuda.empty_cache()
                host = {k: np.stack([r[k] for r in rows]) for k in combat}
                batch = obs_as_tensor(host, 'cuda')
                expected = core.actor_critic(batch)
                graphs.bind_host_observations(batch, host)
                actual = graphs(batch)
                for x, y in zip(actual, expected, strict=True):
                    torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
                retained.append((actual, tuple(x.clone() for x in actual)))
            assert graphs.stats['evictions'] and graphs.stats['hits']
            for actual, original in retained:
                for x, y in zip(actual, original, strict=True):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
    finally:
        graphs.close()


@pytest.mark.parametrize('context_limit', [None, 1, 2])
def test_upper_screen_chunks_keep_original_rows_advantages_and_weights(latent_env, context_limit):
    from types import SimpleNamespace
    from rl.algorithms.packed_ppo import PackedPPOConfig, packed_chunks
    names = ['shop', 'combat', 'event', 'map', 'shop', 'combat', 'card_select', 'shop', 'map']
    obs = tensor_obs(latent_env, len(names))
    obs['phase'][:, 0] = torch.tensor([vocab.PHASE_VOCAB_INDEX[name] for name in names])
    obs['_deck_context_index'] = torch.arange(len(names)) % 3
    rows = torch.arange(len(names))
    adv = rows.float() - 3.25
    batch = SimpleNamespace(observations=obs, actions=rows, advantages=adv)
    model = SimpleNamespace(policy=policy_for(latent_env), normalize_advantage=False, device='cpu')
    config = PackedPPOConfig(4096, 8, context_grouping=True, upper_screen_max_rows=2,
                             max_contexts_per_chunk=context_limit)
    seen, groups = [], {}
    for chunk, advantages, weight, group in packed_chunks(model, batch, config):
        indices = chunk.actions
        if context_limit is not None:
            assert len(chunk.observations['_deck_context_index'].unique()) <= context_limit
        screen, = {names[i] for i in indices.tolist()}
        groups.setdefault(screen, set()).add(group)
        if screen != 'combat':
            assert len(indices) <= 2
        assert weight == pytest.approx(len(indices) / len(rows))
        torch.testing.assert_close(advantages, adv[indices], rtol=0, atol=0)
        seen.extend(indices.tolist())
    assert sorted(seen) == rows.tolist()
    assert all(len(ids) == 1 for ids in groups.values())
    assert len(set.union(*groups.values())) == len(set(names))


def test_context_length_buckets_count_feature_only_map_nodes(latent_env):
    from types import SimpleNamespace
    from rl.algorithms.packed_ppo import PackedPPOConfig, packed_chunks
    obs = tensor_obs(latent_env, 6)
    obs['phase'].fill_(vocab.PHASE_VOCAB_INDEX['shop'])
    obs['deck_ids'].zero_()
    obs['relics'].zero_()
    obs['map_full_rooms'].zero_()
    obs['map_full_features'].zero_()
    indices = [0, 1, 2, 3, 2, 0]
    obs['_deck_context_index'] = torch.tensor(indices)
    for row, width in enumerate([18, 1, 19, 2, 19, 18]):
        obs['map_full_features'][row, :width, 0] = 1
    batch = SimpleNamespace(observations=obs, actions=torch.arange(6), advantages=torch.ones(6))
    model = SimpleNamespace(policy=policy_for(latent_env), normalize_advantage=False, device='cpu')
    config = PackedPPOConfig(4096, 6, context_grouping=True, upper_screen_max_rows=6,
                             max_contexts_per_chunk=2)
    groups = [set(c.actions.tolist()) for c, _, _, _ in packed_chunks(model, batch, config)]
    assert groups == [{1, 3}, {0, 2, 4, 5}]


def test_combat_cannot_contribute_to_early_upper_ring_bucket(latent_env):
    """If the model starts sharing a head, staging it early would lose gradients."""
    from rl.training.ppo_overlap import gradient_buckets
    policy = policy_for(latent_env, 'transformer64x2')
    early, tail = gradient_buckets(policy)
    policy.set_training_mode(True)
    def backward():
        logits, values = policy.core.actor_critic(tensor_obs(latent_env, 2))
        (values.square().sum() + logits[logits > -1e8].square().mean()).backward()
    backward()
    assert any(p.grad is not None and p.grad.any() for p in early)
    policy.zero_grad(set_to_none=True)
    enter_combat(latent_env)
    backward()
    assert all(p.grad is None for p in early)
    assert any(p.grad is not None and p.grad.any() for p in tail)
    assert {id(p) for p in early + tail} == {id(p) for p in policy.parameters() if p.requires_grad}
    assert len(early) + len(tail) == len({id(p) for p in early + tail})


@pytest.mark.cuda
def test_planned_anchor_search_loss_gradients_and_adam_after_graph_release(latent_env):
    """Mixed anchor routes and search KL use the same original optimizer group."""
    import gc
    from contextlib import nullcontext
    from dataclasses import asdict
    from types import SimpleNamespace
    from rl.algorithms.deck_update_plan import DeckUpdatePlan
    from rl.algorithms.search_auxiliary import SearchTrainingConfig, prepare_search_batch
    from rl.algorithms.updates import ppo_forward, optimizer_step
    from rl.training.distributed_sb3 import pack_gradients, average_peer_gradients
    assert torch.cuda.is_available(), 'explicit CUDA planned search qualification required'
    torch.set_float32_matmul_precision('highest')
    enter_combat(latent_env)
    reference = policy_for(latent_env, 'transformer64x2').cuda().train()
    reference.combat_value_target = 'pre-reward-anchor'
    reference.core.combat.compact_training_tokens = True
    planned = deepcopy(reference)
    def learner(policy):
        return SimpleNamespace(policy=policy, device=torch.device('cuda'),
            action_space=latent_env.action_space, autocast_context=nullcontext,
            hierarchical_combat_hp=False, normalize_advantage=False, ent_coef=.02, vf_coef=.5,
            max_grad_norm=.5, deck_search_config=asdict(SearchTrainingConfig(coefficient=.1)))
    models = [learner(reference), learner(planned)]
    plan = DeckUpdatePlan(models[1])
    previous = torch.cuda.current_stream()
    stream = torch.cuda.Stream(); stream.wait_stream(previous)
    try:
        with torch.cuda.stream(stream):
            for iteration, size in enumerate((896, 768, 32)):
                obs = {k: v.cuda() for k, v in tensor_obs(latent_env, size).items()}
                obs['_deck_context_index'] = torch.zeros(size, dtype=torch.long, device='cuda')
                obs['combat_exit_anchor'][::8, 0] = 1.
                legal = obs['combat_legal'].bool()
                target = torch.randn(legal.shape, device='cuda').masked_fill(~legal, -torch.inf).log_softmax(-1)
                obs['_search_target'] = target
                obs['_search_valid'] = torch.arange(size, device='cuda').remainder(3).eq(1).float()
                batch = SimpleNamespace(observations=obs, advantages=torch.randn(size, device='cuda'),
                    actions=legal.long().argmax(-1), action_masks=legal,
                    old_values=torch.zeros(size, device='cuda'), returns=torch.rand(size, device='cuda'),
                    old_log_prob=torch.full((size,), -1., device='cuda'))
                losses = []
                for model in models:
                    prepare_search_batch(model, batch, None)
                    model.policy.optimizer.zero_grad(set_to_none=True)
                    direct = model is models[1]
                    chunks = [(batch, batch.advantages, 1., 'combat')]
                    for chunk, _, _, _ in (plan.schedule(chunks) if direct else chunks):
                        _, loss = ppo_forward(model, chunk, .2, None,
                            direct_masked_categorical=direct, update_plan=plan if direct else None)
                        loss.backward()
                        losses.append(loss.detach())
                torch.testing.assert_close(*losses, rtol=3e-4, atol=3e-6)
                for (name, left), (_, right) in zip(reference.named_parameters(), planned.named_parameters(), strict=True):
                    assert (left.grad is None) == (right.grad is None), name
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=3e-6, msg=name)
                for model in models:
                    average_peer_gradients(model, pack_gradients(model), [], rank=0, world_size=1)
                    optimizer_step(model)
                for left, right in zip(reference.parameters(), planned.parameters(), strict=True):
                    # FP32 padding changes reduction order; compare parameters
                    # and Adam moments, not a cross-kernel bitwise claim.
                    torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-6)
                    for key, value in reference.optimizer.state[left].items():
                        torch.testing.assert_close(value, planned.optimizer.state[right][key], rtol=6e-4, atol=3e-6)
                torch.cuda.synchronize(); torch._C._cuda_clearCublasWorkspaces()
                gc.collect(); torch.cuda.empty_cache()
            assert plan.graphs.stats['combat_captures'] == 1
            assert plan.graphs.stats['combat_replays'] == 2
    finally:
        plan.graphs.entries.clear()
        previous.wait_stream(stream)

@pytest.mark.parametrize('transport', ['dummy', 'native-subproc'])
@pytest.mark.timeout(240)
def test_search_acts_and_restores_cpu(tmp_path, transport):
    test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path, True, transport, 'screens',
        action_selection='search-filter-policy-rank-v1')


@pytest.mark.cuda
@pytest.mark.timeout(300)
def test_search_acts_and_restores_cuda(tmp_path, cuda_battle_artifacts):
    test_two_real_ppo_rounds_checkpoint_and_packed_resume(tmp_path, True, 'native-subproc', 'screens',
        encoder='transformer64x2', device='cuda', search_backend='cuda-resident',
        action_selection='search-filter-policy-rank-v1')


@pytest.mark.cuda
@pytest.mark.timeout(600)
def test_search_stream_completion_order_keeps_slot_trajectories(tmp_path, cuda_battle_artifacts, monkeypatch):
    from rl.algorithms import deck_stream_rollout
    original = deck_stream_rollout.collect
    results=[]
    for barrier in (True, False):
        def collect(*args, **kwargs):
            return original(*args, **kwargs, barrier=barrier)
        monkeypatch.setattr(deck_stream_rollout, 'collect', collect)
        results.append(test_two_real_ppo_rounds_checkpoint_and_packed_resume(
            tmp_path/str(barrier), True, 'native-subproc', 'screens',
            encoder='transformer64x2', device='cuda', search_backend='cuda-resident',
            action_selection='search-filter-policy-rank-v1', capture_result=True))
    a,b=results
    assert a['rng']==b['rng']
    for k in ('actions','rewards','episode_starts'):
        np.testing.assert_array_equal(a[k],b[k])
    for k in a['observations']:
        np.testing.assert_array_equal(a['observations'][k],b['observations'][k])
    assert a['targets'].keys()==b['targets'].keys()
    for k in a['targets']:
        np.testing.assert_allclose(a['targets'][k],b['targets'][k],atol=2e-5,rtol=2e-5)


@pytest.mark.cuda
@pytest.mark.timeout(600)
def test_search_stream_native_fallback_and_resume(tmp_path, cuda_battle_artifacts, monkeypatch):
    from rl.training.native_subproc_vec_env import NativeSubprocVecEnv
    original = NativeSubprocVecEnv.deck_resident_roots
    forced = []
    def roots(self, indices, artifacts, generation=None):
        rows, groups = original(self, indices, artifacts, generation)
        # Keep one worker on native frontiers while its sibling uses CUDA.
        for i, row in zip(indices, rows):
            if i == 0 and not row.get('skipped'):
                row['native_fallback'] = True
                forced.append((generation, i))
        return rows, groups
    monkeypatch.setattr(NativeSubprocVecEnv, 'deck_resident_roots', roots)
    result = test_two_real_ppo_rounds_checkpoint_and_packed_resume(
        tmp_path, True, 'native-subproc', 'screens', encoder='transformer64x2',
        device='cuda', search_backend='cuda-resident',
        action_selection='search-filter-policy-rank-v1', capture_result=True)
    assert forced and result['search_stats']['native_fallback_roots'] > 0
    assert result['search_stats']['stream_submitted_roots'] > 0


@pytest.mark.cuda
def test_resident_actor_caches_original_slots_across_reordering_and_release(latent_env, cuda_battle_artifacts):
    import gc
    from rl.search.resident_inference import ResidentInference
    enter_combat(latent_env)
    policy = policy_for(latent_env, 'transformer64x2').cuda().eval()
    inference = ResidentInference(policy.core, 1024**2)
    inference.graphs.capacity = 1
    host = {k: np.repeat(v[None], 3, axis=0) for k,v in latent_env._obs().items()}
    masks = np.repeat(latent_env.action_masks()[None], 3, axis=0)
    def check(ids):
        obs = {k:v[ids] for k,v in host.items()}
        with torch.no_grad():
            packet = inference.actor((obs, masks[ids], np.asarray(ids), 3))
            _, values, _ = policy(obs_as_tensor(obs, 'cuda'), deterministic=True, action_masks=masks[ids])
            np.testing.assert_allclose(packet['values'], values.flatten().cpu().numpy(), atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(packet['probabilities'], policy.action_dist.distribution.probs.cpu().numpy(), atol=2e-5, rtol=2e-5)
    try:
        check([0,1,2]); initial=inference.graphs.stats['context_rows']
        check([2]); check([1,0]); check([2,1,0])
        assert inference.graphs.stats['context_rows']==initial
        host['floor'][1] += 1
        check([2,1]); assert inference.graphs.stats['context_rows']==initial+1
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        check([0,1,2])
        assert inference.graphs.stats['context_rows']==initial+1
        with torch.no_grad(): next(policy.core.context.parameters()).add_(.0001)
        check([2,1,0])
        assert inference.graphs.stats['context_rows']>initial+1
        assert inference.graphs.stats['evictions']>0
    finally: inference.close()
