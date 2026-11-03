"""Unified strategic model: real observation ABI, targets and save lifecycle."""
import copy
import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from env.core import action_space as A, vocab as V
from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.upper_ppo import macro_advantages, update_upper
from rl.algorithms.unified_upper import complete_returns
from rl.policies.entity_joint import EntityJointPolicy

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(120)]


@pytest.fixture
def unified_env():
    env = SlayTheSpireEnv(card_language=True, upper_public_context=True, global_map_view=True,
        combat_action_contract='autoregressive-v2', deck_size_observation=True,
        neow_observation=True, event_option_observation=True)
    yield env
    env.close()


def make_model(env, device='cpu'):
    return AccumulatingMaskablePPO(EntityJointPolicy, env, seed=31, n_steps=8,
        batch_size=4, n_epochs=1, device=device, nosl_is_az=True, nosl_is_az_cpp=True,
        nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='model', policy_kwargs=dict(
            entity_config=dict(width=32, layers=1, heads=4, quantiles=8, auxiliary_loss_coef=0),
            upper_config=dict(architecture='upper-latent-actions-v1', width=32, layers=1,
                heads=4, latents=4, epochs=2, critic_epochs=3, micro_batch_size=2,
                effective_batch_size=4, minibatch_size=4, optimizer='adamw', target_kl=None,
                gamma=1., gae_lambda=.97, learning_rate=6e-5, value_learning_rate=1e-4)))


def tensor(obs):
    return {k: torch.as_tensor(v[None]) for k, v in obs.items()}


def test_independent_parameters_mixed_screen_and_entity_permutation(unified_env):
    model = make_model(unified_env)
    core = model.policy.core
    obs, _ = unified_env.reset(seed=9)
    actor, critic = core.upper_actor, core.upper_critic
    assert not {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()}
    reward = copy.deepcopy(obs)
    reward['phase'][0] = V.PHASE_VOCAB_INDEX['combat_reward']
    reward['offer_card_ids'][:3] = [1, 2, 3]
    reward['offer_card_upgrades'][:3] = [0, 1, 0]
    batch = {k: torch.stack((torch.as_tensor(obs[k]), torch.as_tensor(reward[k]))) for k in obs}
    actor.eval(); critic.eval()
    with torch.no_grad():
        mixed = core.actor_critic(batch)
        singles = [core.actor_critic(tensor(row)) for row in (obs, reward)]
        for k in (0, 1):
            torch.testing.assert_close(mixed[k], torch.cat([s[k] for s in singles]), rtol=1e-5, atol=2e-6)
        permuted = copy.deepcopy(reward)
        for key in ('offer_card_ids', 'offer_card_upgrades'):
            permuted[key][[0, 2]] = permuted[key][[2, 0]]
        for key in ('deck_ids', 'deck_upgrades'):
            permuted[key] = permuted[key][::-1].copy()
        logits, value = core.actor_critic(tensor(permuted))
        expected = singles[1][0].clone()
        expected[:, [A._RCARD, A._RCARD+2]] = expected[:, [A._RCARD+2, A._RCARD]]
        # Only legal/relevant reward action slots matter; deck permutation is a set.
        torch.testing.assert_close(logits[:, A._RGOLD:A._RSKIP+1], expected[:, A._RGOLD:A._RSKIP+1], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(value, singles[1][1], rtol=1e-5, atol=2e-6)
    before = copy.deepcopy(actor.state_dict())
    model.policy.upper_optimizer.zero_grad()
    critic(tensor(obs)).square().mean().backward()
    model.policy.upper_optimizer.step()
    for k, v in actor.state_dict().items():
        assert torch.equal(v, before[k])


def test_mc_return_is_separate_from_frozen_gae():
    rows = [dict(env=0, episode=0, terminal=False, reward=.2, value=10.),
            dict(env=1, episode=0, terminal=True, reward=.5, value=3.),
            dict(env=0, episode=0, terminal=True, reward=1., value=-2.)]
    np.testing.assert_allclose(complete_returns(rows), [1.2, .5, 1.])
    advantages, lambda_return = macro_advantages(rows, np.zeros(3), 1., .97)
    assert lambda_return[0] != pytest.approx(1.2)
    assert advantages[0] == pytest.approx(.2 - 2 - 10 + .97 * 3)
    with pytest.raises(ValueError, match='complete'):
        complete_returns(rows[:1])


@pytest.mark.parametrize('critic_batch', [0, 8])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_two_updates_save_load_and_old_value_freezing(unified_env, tmp_path, device, critic_batch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.fail('selected CUDA qualification requires an available device')
    from dataclasses import asdict, replace
    model = make_model(unified_env, device)
    cfg = replace(model.policy.core.upper_config, critic_effective_batch_size=critic_batch)
    model.policy.core.upper_config = cfg
    model.policy.upper_config = model.policy_kwargs['upper_config'] = asdict(cfg)
    critic_steps_per_round = 3 if critic_batch == 8 else 6
    model.set_logger(configure(str(tmp_path), []))
    obs, _ = unified_env.reset(seed=41)
    model._upper_pending = [None]
    for iteration in range(2):
        inputs = {k: v.to(device) for k, v in tensor(obs).items()}
        mask = unified_env.action_masks()
        with torch.no_grad():
            action, value, logp = model.policy(inputs, action_masks=mask[None])
        rows = [dict(observation=copy.deepcopy(obs), action=int(action), mask=mask.copy(),
            value=float(value), log_prob=float(logp), reward=float(i % 2), terminal=True,
            env=i, episode=0) for i in range(7)]
        model._upper_completed = rows
        frozen = [(r['value'], r['log_prob']) for r in rows]
        result = update_upper(model)
        assert result['updates'] == 4 and result['critic_updates'] == critic_steps_per_round
        assert result['requested_critic_updates'] == critic_steps_per_round
        assert result['requested_critic_effective_batch_size'] == (critic_batch or 4)
        assert result['critic_effective_batch_size_global'] == (7 if critic_batch == 8 else 3.5)
        assert result['samples_used'] == 14 and result['critic_samples_used'] == 21
        assert frozen == [(r['value'], r['log_prob']) for r in rows]
        path = tmp_path / f'round{iteration}.zip'
        model.save(path)
        loaded = AccumulatingMaskablePPO.load(path, device=device)
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, loaded.policy.state_dict()[key], rtol=0, atol=0)
        actor_steps = {int(s['step']) for p, s in loaded.policy.upper_optimizer.state.items()
                       if any(p is q for q in loaded.policy.core.upper_actor.parameters())}
        critic_steps = {int(s['step']) for p, s in loaded.policy.upper_optimizer.state.items()
                        if any(p is q for q in loaded.policy.core.upper_critic.parameters())}
        assert actor_steps == {4 * (iteration+1)} and critic_steps == {critic_steps_per_round * (iteration+1)}
        assert loaded.policy.core.upper_config.critic_effective_batch_size == critic_batch
        model = loaded
        model.set_logger(configure(str(tmp_path), []))


@pytest.mark.parametrize('critic_batch', [0, 4])
@pytest.mark.parametrize('measure', [False, True])
def test_unequal_empty_ranks_match_global_weighted_update(unified_env, tmp_path, measure, critic_batch, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from dataclasses import replace
    from rl.training.distributed_sb3 import pack_gradients, install_flat_gradients
    if critic_batch:
        # All samples on one rank retain the reference partition boundaries;
        # deterministic order isolates weighted reduction from thread RNG order.
        monkeypatch.setattr(np.random, 'permutation', lambda n: np.arange(n))
    obs, _ = unified_env.reset(seed=53)
    reference = make_model(unified_env)
    state = copy.deepcopy(reference.policy.state_dict())
    with torch.no_grad():
        action, value, logp = reference.policy(tensor(obs), action_masks=unified_env.action_masks()[None])
    rows = [dict(observation=copy.deepcopy(obs), action=int(action), mask=unified_env.action_masks(),
        value=float(value), log_prob=float(logp), reward=i/7, terminal=True, env=i, episode=0) for i in range(7)]
    legal = np.flatnonzero(unified_env.action_masks())
    for i, row in enumerate(rows):
        row['action'] = int(legal[i % len(legal)])
        row['observation']['gold'] *= i + 1
        with torch.no_grad():
            value, logp, _ = reference.policy.evaluate_actions(tensor(row['observation']),
                torch.tensor([row['action']]), action_masks=row['mask'][None])
        row.update(value=float(value), log_prob=float(logp))
    def ready(model, samples):
        model.policy.load_state_dict(state)
        model.policy.core.upper_config = replace(model.policy.core.upper_config,
            effective_batch_size=64, critic_effective_batch_size=critic_batch, measure_value_loss=measure)
        model.set_logger(configure(str(tmp_path), []))
        model._upper_completed, model._upper_pending = copy.deepcopy(samples), []
        return model
    ready(reference, rows)
    reference_result = update_upper(reference)
    partitions = ([], rows, []) if critic_batch else ([], rows[:3], rows[3:])
    models = [ready(make_model(unified_env), samples) for samples in partitions]
    barrier, slots = Barrier(3, timeout=40), [None]*3
    def average(rank, values):
        slots[rank] = values
        barrier.wait()
        result = {k: sum(x[k] for x in slots)/3 for k in values}
        barrier.wait()
        return result
    def train(rank):
        def sync(step):
            install_flat_gradients(models[rank], average(rank, {'g': pack_gradients(models[rank])})['g'])
        return update_upper(models[rank], rank=rank, world_size=3,
            average=lambda x: average(rank, x), synchronize=sync)
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(train, range(3)))
    for model, result in zip(models, results):
        assert result['samples_used'] == 14 and result['critic_samples_used'] == 21
        assert result['updates'] == 2 and result['critic_updates'] == (6 if critic_batch else 3)
        if measure:
            assert result['value_measurement_samples'] == 7
            for when in ('pre', 'post'):
                assert result[f'value_mc_mse_{when}_update'] == pytest.approx(
                    reference_result[f'value_mc_mse_{when}_update'], rel=2e-5, abs=3e-6)
        for p, q in zip(model.policy.parameters(), reference.policy.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('calibration_games', [0, 2])
def test_game_holdout_candidate_probe_restores_adam_and_repeats_control(unified_env, tmp_path, device, monkeypatch, calibration_games):
    from dataclasses import asdict, replace
    from unittest.mock import patch
    from rl.algorithms import unified_upper as U
    from rl.algorithms.upper_probe import run_probe, state_digest, split_games, categorical_kl
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware unavailable; CPU result is not GPU qualification')
    model = make_model(unified_env, device=device)
    cfg = replace(model.policy.core.upper_config, measure_value_loss=True)
    model.policy.core.upper_config = cfg
    model.policy.upper_config = model.policy_kwargs['upper_config'] = asdict(cfg)
    model.set_logger(configure(str(tmp_path), []))
    obs, _ = unified_env.reset(seed=41)
    masks = unified_env.action_masks()
    with torch.no_grad():
        action, value, logp = model.policy({k: v.to(device) for k, v in tensor(obs).items()}, action_masks=masks[None])
    rows = [dict(observation=copy.deepcopy(obs), action=int(action), mask=masks.copy(),
        value=float(value), log_prob=float(logp), reward=i/7, terminal=True,
        env=i, episode=0) for i in range(7)]
    # Seed actual nonempty Adam state before testing its exact restoration.
    model._upper_completed = copy.deepcopy(rows)
    update_upper(model)
    model._upper_completed = rows
    before = state_digest(dict(policy=model.policy.state_dict(),
        optimizer=model.policy.upper_optimizer.state_dict()))
    protocol = dict(seed=31, diagnostic=dict(world_size=1, games_per_rank=7,
        holdout_games_per_rank=2, max_probe_decisions_per_split_per_rank=10, full_critic_mse=True, profile_updates=True),
        arms=[dict(id='control', changes={}),
              dict(id='lambda1', changes={}, critic_target='frozen_lambda_return', critic_lambda=1.),
              dict(id='room_half', changes={}, critic_target='frozen_lambda_return', critic_lambda=.5, lambda_scope='room'),
              dict(id='batch3', changes={'effective_batch_size': 3}),
              dict(id='critic_lambda', changes={}, critic_target='frozen_lambda_return'),
              dict(id='control_repeat', changes={})])
    protocol['diagnostic']['calibration_games_per_rank'] = calibration_games
    if calibration_games:
        protocol['adaptive_lambda'] = dict(base=.9, span=.07, shrinkage_games=32, noise_floor=.05)
        protocol['arms'].insert(-1, dict(id='adaptive', changes={},
            critic_target='frozen_lambda_return', critic_lambda=.9, lambda_scope='calibrated_room'))
    measurement_counts = []
    original_measurement = U.measure_critic_mse
    def count_measurements(current, observations, targets, **kwargs):
        measurement_counts.append(len(next(iter(targets.values()))))
        return original_measurement(current, observations, targets, **kwargs)
    monkeypatch.setattr(U, 'measure_critic_mse', count_measurements)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    result = run_probe(model, protocol, output_dir=tmp_path/'probe')
    # All records on this rank are terminal, so MC/lambda labels coincide.
    # Other ranks can have longer games: collective evaluation counts must
    # still follow the shared target contract, not local label equality.
    changed_targets = sum(a.get('critic_target') == 'frozen_lambda_return' for a in protocol['arms'])
    assert measurement_counts.count(2) == 1 + len(protocol['arms']) + changed_targets
    assert torch.are_deterministic_algorithms_enabled() == deterministic_before
    assert result['control_repeat_exact'] and result['source_state_restored']
    assert result['rng_restored'] and result['lower_optimizer_unchanged']
    assert result['training_games'] == 5-calibration_games and result['holdout_games'] == 2
    assert result['calibration_games'] == calibration_games
    assert all(a['resources']['interval_seconds'] == 1. for a in result['arms'])
    assert all((a['torch_max_allocated_bytes'] is not None) == (device == 'cuda') for a in result['arms'])
    for split, count in [('train', 5-calibration_games), ('holdout', 2)]:
        pre = result['full_critic_pre_update'][split]
        assert pre['samples'] == count
        for arm in result['arms']:
            post = arm['full_critic_post_update'][split]
            assert post['samples'] == count
            assert post['mc_mse'] == pytest.approx(arm['probes'][split]['critic_mc_mse'], rel=1e-5, abs=2e-6)
            assert pre['mc_mse'] == pytest.approx(arm['probes'][split]['critic_old_mc_mse'], rel=1e-5, abs=2e-6)
    if calibration_games:
        assert result['frozen_scene_calibration']['calibration_games'] == calibration_games
    assert all(r['stats']['samples'] == 5-calibration_games and r['stats']['samples_used'] == 2*(5-calibration_games)
               for r in result['arms'])
    assert result['arms'][0]['probes'] == result['arms'][-1]['probes']
    assert result['arms'][0]['final_digest'] == result['arms'][1]['final_digest']
    assert 'value_mc_mse_pre_update' in result['arms'][0]['stats']
    assert 'value_mc_mse_pre_update' not in result['arms'][1]['stats']
    assert model.policy.core.upper_config.measure_value_loss
    assert result['room_transition_audit']['terminals'] == 7
    assert model._upper_completed is rows
    assert state_digest(dict(policy=model.policy.state_dict(),
        optimizer=model.policy.upper_optimizer.state_dict())) == before
    with patch.object(U, 'update_unified_upper', side_effect=RuntimeError('probe failed')):
        with pytest.raises(RuntimeError, match='probe failed'):
            run_probe(model, protocol)
    assert torch.are_deterministic_algorithms_enabled() == deterministic_before
    assert state_digest(dict(policy=model.policy.state_dict(),
        optimizer=model.policy.upper_optimizer.state_dict())) == before
    # A trajectory with two decisions must stay wholly on one side.
    trajectories = [dict(r, terminal=False, reward=0.) for r in rows] + rows
    train, hold = split_games(trajectories, 2, 31)
    assert {trajectories[i]['env'] for i in train}.isdisjoint(
        {trajectories[i]['env'] for i in hold})
    old = np.log([[.5, .5], [1., 1e-30]])
    new = np.log([[.75, .25], [1., 1e-30]])
    kl = categorical_kl(old, new, np.array([[True, True], [True, False]]))
    np.testing.assert_allclose(kl, [.5*np.log(4/3), 0.], rtol=1e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_probe_reports_fitted_lambda_targets_and_whole_game_validation(unified_env, tmp_path, device, monkeypatch):
    from collections import Counter
    from dataclasses import asdict, replace
    from rl.algorithms import unified_upper as U
    from rl.algorithms.upper_probe import run_probe, state_digest
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.fail('selected CUDA qualification requires an available device')
    model = make_model(unified_env, device=device)
    cfg = replace(model.policy.core.upper_config, critic_target='frozen-lambda',
                  critic_effective_batch_size=8, measure_value_loss=True)
    model.policy.core.upper_config = cfg
    model.policy.upper_config = model.policy_kwargs['upper_config'] = asdict(cfg)
    model.set_logger(configure(str(tmp_path), []))
    obs, _ = unified_env.reset(seed=41)
    mask = unified_env.action_masks()
    with torch.no_grad():
        action, value, logp = model.policy({k: v.to(device) for k, v in tensor(obs).items()},
                                         action_masks=mask[None])
    rows = []
    for game in range(7):
        for terminal in (False, True):
            rows.append(dict(observation=copy.deepcopy(obs), action=int(action), mask=mask.copy(),
                value=float(value), log_prob=float(logp), reward=(game+1)/7 if terminal else .25,
                terminal=terminal, env=game, episode=0))
    # Exercise restoration of real, nonempty optimizer moments as well as weights.
    model._upper_completed = copy.deepcopy(rows)
    update_upper(model)
    with torch.no_grad():
        logits, value = model.policy.core.actor_critic({k: v.to(device) for k, v in tensor(obs).items()})
        logp = logits.float().masked_fill(~torch.as_tensor(mask[None], device=device), -1e9).log_softmax(-1)[0, int(action)]
    for row in rows:
        row.update(value=float(value), log_prob=float(logp))
    model._upper_completed = rows
    before = state_digest(dict(policy=model.policy.state_dict(), optimizer=model.policy.upper_optimizer.state_dict()))
    fitted_games = []
    original_update = U.update_unified_upper
    def observe_update(current, **kwargs):
        counts = Counter(r['env'] for r in current._upper_completed)
        assert set(counts.values()) == {2}
        fitted_games.append(set(counts))
        return original_update(current, **kwargs)
    monkeypatch.setattr(U, 'update_unified_upper', observe_update)
    protocol = dict(seed=31, diagnostic=dict(world_size=1, games_per_rank=7,
        holdout_games_per_rank=2, max_probe_decisions_per_split_per_rank=100, full_critic_mse=True),
        arms=[dict(id='control', changes={}), dict(id='critic6', changes={'critic_epochs': 6}),
              dict(id='control_repeat', changes={})])
    result = run_probe(model, protocol, output_dir=tmp_path/'probe')
    assert result['source_state_restored'] and result['rng_restored'] and result['lower_optimizer_unchanged']
    assert result['control_repeat_numerical']
    for arm, epochs in zip(result['arms'], (3, 6, 3)):
        assert arm['stats']['samples'] == 10 and arm['stats']['samples_used'] == 20
        assert arm['stats']['critic_samples_used'] == 10*epochs
        assert arm['stats']['critic_updates'] == 2*epochs
        for split, games in [('train', 5), ('holdout', 2)]:
            with np.load(tmp_path/'probe'/f'{arm["id"]}-{split}.npz') as saved:
                ids = saved['indices']
                expected_mc = np.array([rows[i]['reward'] if rows[i]['terminal']
                    else .25+rows[i+1]['reward'] for i in ids])
                expected_target = np.array([rows[i]['reward'] if rows[i]['terminal']
                    else .25+.03*rows[i+1]['value']+.97*rows[i+1]['reward'] for i in ids])
                np.testing.assert_allclose(saved['mc_return'], expected_mc, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(saved['critic_target'], expected_target, rtol=1e-12, atol=1e-12)
                assert np.max(np.abs(saved['critic_target']-saved['mc_return'])) > 1e-3
                selected_games = set(saved['env'])
                assert len(selected_games) == games
                if split == 'train':
                    assert all(group == selected_games for group in fitted_games)
                else:
                    assert all(group.isdisjoint(selected_games) for group in fitted_games)
                for when, values in [('pre', saved['old_value']), ('post', saved['value'])]:
                    measured = arm[f'full_critic_{when}_update'][split]
                    assert measured['samples'] == 2*games
                    for name, targets in [('mc', expected_mc), ('target', expected_target)]:
                        expected_mse = np.square(values-targets.astype(np.float32)).mean()
                        assert measured[f'{name}_mse'] == pytest.approx(expected_mse, rel=2e-5, abs=3e-6)
    assert model._upper_completed is rows
    assert state_digest(dict(policy=model.policy.state_dict(), optimizer=model.policy.upper_optimizer.state_dict())) == before
    invalid = copy.deepcopy(protocol)
    invalid['arms'] = [dict(id='invalid', changes={}, critic_target='frozen_lambda_return')]
    with pytest.raises(ValueError, match='require the MC update path'):
        run_probe(model, invalid)
    assert state_digest(dict(policy=model.policy.state_dict(), optimizer=model.policy.upper_optimizer.state_dict())) == before


def test_probe_float_tolerance_distinguishes_roundoff_from_policy_change():
    from rl.algorithms.upper_probe import compare_probe_arrays
    old = {'value': np.array([.5, 1.])}
    limits = dict(value=1e-4, relative_l2=3e-5)
    assert compare_probe_arrays(old, {'value': old['value']+1e-6}, limits)['value']['passed']
    assert not compare_probe_arrays(old, {'value': old['value']+.001}, limits)['value']['passed']
    assert not compare_probe_arrays(old, {'value': np.array([np.nan, 1.])}, limits)['value']['passed']


def test_frozen_lambda_critic_preserves_actor_and_adam_resume(unified_env, tmp_path):
    from rl.algorithms.upper_probe import state_digest
    from rl.policies.entity_joint import UpperConfig
    model = make_model(unified_env)
    model.set_logger(configure(str(tmp_path), []))
    obs, _ = unified_env.reset(seed=41)
    mask = unified_env.action_masks()
    with torch.no_grad():
        action, value, logp = model.policy(tensor(obs), action_masks=mask[None])
    rows = [dict(observation=copy.deepcopy(obs), action=int(action), mask=mask.copy(),
                 value=float(value), log_prob=float(logp), reward=i/7, terminal=True,
                 env=i, episode=0) for i in range(7)]
    model._upper_completed = copy.deepcopy(rows)
    update_upper(model)  # actual nonempty actor and critic Adam state
    source = tmp_path/'source.zip'
    model.save(source)
    original_adam = state_digest(model.policy.upper_optimizer.state_dict())
    kwargs = copy.deepcopy(model.policy_kwargs)
    kwargs['upper_config']['critic_target'] = 'frozen-lambda'
    candidate = AccumulatingMaskablePPO.load(source, device='cpu', custom_objects={'policy_kwargs': kwargs})
    assert state_digest(candidate.policy.upper_optimizer.state_dict()) == original_adam
    assert state_digest(candidate.policy.state_dict()) == state_digest(model.policy.state_dict())
    rows = [dict(r, terminal=False, reward=.1, value=float(r['value'])+.3) for r in rows] + rows
    mc = complete_returns(rows)
    _, frozen = macro_advantages(rows, np.zeros(len(rows)), 1., .97)
    assert not np.array_equal(mc, frozen)
    old_labels = [(r['value'], r['log_prob']) for r in rows]
    for learner in (model, candidate):
        learner.set_logger(configure(str(tmp_path), []))
        learner._upper_completed = copy.deepcopy(rows)
        np.random.seed(71)
        stats = update_upper(learner)
        assert stats['samples_used'] == 28 and stats['critic_samples_used'] == 42
    assert old_labels == [(r['value'], r['log_prob']) for r in rows]
    assert stats['critic_target_version'] == 2
    assert state_digest(candidate.policy.core.upper_actor.state_dict()) == state_digest(model.policy.core.upper_actor.state_dict())
    assert state_digest(candidate.policy.core.upper_critic.state_dict()) != state_digest(model.policy.core.upper_critic.state_dict())
    saved = tmp_path/'candidate.zip'; candidate.save(saved)
    restored = AccumulatingMaskablePPO.load(saved, device='cpu')
    assert restored.policy.core.upper_config.critic_target == 'frozen-lambda'
    assert state_digest(restored.policy.upper_optimizer.state_dict()) == state_digest(candidate.policy.upper_optimizer.state_dict())
    restored.set_logger(configure(str(tmp_path), []))
    restored._upper_completed = copy.deepcopy(rows)
    assert update_upper(restored)['critic_target_version'] == 2
    with pytest.raises(ValueError, match='unified'):
        UpperConfig(critic_target='frozen-lambda')


@pytest.mark.parametrize('critic_batch', [None, 4])
def test_fewer_updates_small_lr_preserves_adam_and_survives_resume(unified_env, tmp_path, critic_batch):
    from rl.algorithms.upper_probe import state_digest
    from rl.training.distributed_sb3 import build_parser, apply_unified_upper_update_overrides
    model = make_model(unified_env)
    model.set_logger(configure(str(tmp_path), []))
    obs, _ = unified_env.reset(seed=41)
    mask = unified_env.action_masks()
    with torch.no_grad():
        action, value, logp = model.policy(tensor(obs), action_masks=mask[None])
    rows = [dict(observation=copy.deepcopy(obs), action=int(action), mask=mask.copy(),
                 value=float(value), log_prob=float(logp), reward=i/7, terminal=True,
                 env=i, episode=0) for i in range(7)]
    model._upper_completed = copy.deepcopy(rows)
    update_upper(model)
    source = tmp_path/'source.zip'; model.save(source)
    candidate = AccumulatingMaskablePPO.load(source, device='cpu')
    policy_digest = state_digest(candidate.policy.state_dict())
    adam_digest = state_digest(candidate.policy.upper_optimizer.state_dict()['state'])
    args = build_parser().parse_args(['--rank', '0', '--world-size', '1', '--upper-learning-rate', '0.00003',
        '--upper-value-learning-rate', '0.00005', '--upper-effective-batch-size', '8',
        '--upper-critic-epochs', '2'])
    if critic_batch is not None:
        args.upper_critic_effective_batch_size = critic_batch
    critic_steps_per_round = 4 if critic_batch == 4 else 2
    apply_unified_upper_update_overrides(candidate, args)
    assert state_digest(candidate.policy.state_dict()) == policy_digest
    assert state_digest(candidate.policy.upper_optimizer.state_dict()['state']) == adam_digest
    assert [g['lr'] for g in candidate.policy.upper_optimizer.param_groups] == [3e-5, 5e-5]
    for iteration in range(2):
        candidate.set_logger(configure(str(tmp_path), []))
        candidate._upper_completed = copy.deepcopy(rows)
        result = update_upper(candidate)
        assert result['updates'] == 2 and result['critic_updates'] == critic_steps_per_round
        assert result['samples_used'] == result['critic_samples_used'] == 14
        assert result['critic_target_version'] == 1
        assert result['actor_learning_rate'] == 3e-5 and result['critic_learning_rate'] == 5e-5
        for network, expected in [(candidate.policy.core.upper_actor, 4+2*(iteration+1)),
                                  (candidate.policy.core.upper_critic, 6+critic_steps_per_round*(iteration+1))]:
            assert {int(candidate.policy.upper_optimizer.state[p]['step']) for p in network.parameters()} == {expected}
        saved = tmp_path/f'candidate{iteration}.zip'; candidate.save(saved)
        restored = AccumulatingMaskablePPO.load(saved, device='cpu')
        assert state_digest(restored.policy.upper_optimizer.state_dict()) == state_digest(candidate.policy.upper_optimizer.state_dict())
        apply_unified_upper_update_overrides(restored, build_parser().parse_args(['--rank', '0', '--world-size', '1']))
        cfg = restored.policy.core.upper_config
        assert cfg.learning_rate == 3e-5 and cfg.value_learning_rate == 5e-5
        assert cfg.effective_batch_size == 8 and cfg.critic_epochs == 2
        assert cfg.critic_effective_batch_size == (critic_batch or 0)
        candidate = restored
    args.upper_learning_rate = float('nan')
    with pytest.raises(ValueError, match='finite and positive'):
        apply_unified_upper_update_overrides(candidate, args)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('target', ['mc', 'frozen-lambda'])
def test_frozen_value_measurement_preserves_updates_rng_and_resume(unified_env, tmp_path, target, device):
    import random
    from dataclasses import asdict, replace
    from rl.algorithms.upper_probe import state_digest
    from rl.training.distributed_sb3 import build_parser, apply_unified_upper_update_overrides
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware unavailable; CPU result is not GPU qualification')
    model = make_model(unified_env, device=device)
    model.set_logger(configure(str(tmp_path), []))
    obs, _ = unified_env.reset(seed=41)
    mask = unified_env.action_masks()
    with torch.no_grad():
        action, value, logp = model.policy({k: v.to(device) for k, v in tensor(obs).items()}, action_masks=mask[None])
    terminal = [dict(observation=copy.deepcopy(obs), action=int(action), mask=mask.copy(),
        value=float(value), log_prob=float(logp), reward=i/7, terminal=True, env=i, episode=0)
        for i in range(7)]
    rows = [dict(r, terminal=False, reward=.1, value=float(r['value'])+.3) for r in terminal] + terminal
    model._upper_completed = copy.deepcopy(rows)
    update_upper(model)  # Nonempty Adam before enabling instrumentation.
    cfg = replace(model.policy.core.upper_config, critic_target=target)
    model.policy.core.upper_config = cfg
    model.policy.upper_config = model.policy_kwargs['upper_config'] = asdict(cfg)
    source = tmp_path/'measurement-source.zip'; model.save(source)
    control = AccumulatingMaskablePPO.load(source, device=device)
    measured = AccumulatingMaskablePPO.load(source, device=device)
    original = state_digest((measured.policy.state_dict(), measured.policy.upper_optimizer.state_dict()))
    parser = build_parser()
    apply_unified_upper_update_overrides(measured, parser.parse_args(
        ['--rank', '0', '--world-size', '1', '--upper-measure-value-loss']))
    assert original == state_digest((measured.policy.state_dict(), measured.policy.upper_optimizer.state_dict()))
    mc = complete_returns(rows).astype(np.float32)
    labels = (mc if target == 'mc' else macro_advantages(rows, np.zeros(len(rows)), 1., .97)[1].astype(np.float32))
    batch = {k: torch.as_tensor(np.stack([r['observation'][k] for r in rows]), device=device) for k in obs}
    def oracle(learner):
        network = learner.policy.core.upper_critic
        mode = network.training
        network.eval()
        with torch.no_grad():
            predictions = network(batch).flatten().double().cpu().numpy()
        network.train(mode)
        return {key: float(np.square(predictions-value.astype(np.float64)).mean())
                for key, value in [('mc', mc), ('target', labels)]}
    for iteration in range(2):
        expected_pre = oracle(measured)
        results, rngs = [], []
        for learner in (control, measured):
            learner.set_logger(configure(str(tmp_path), []))
            learner._upper_completed = copy.deepcopy(rows)
            random.seed(700+iteration); np.random.seed(700+iteration); torch.manual_seed(700+iteration)
            results.append(update_upper(learner))
            rngs.append(state_digest((random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if device == 'cuda' else None)))
        expected_post = oracle(measured)
        assert rngs[0] == rngs[1]
        if device == 'cpu':
            assert state_digest(control.policy.state_dict()) == state_digest(measured.policy.state_dict())
            assert state_digest(control.policy.upper_optimizer.state_dict()) == state_digest(measured.policy.upper_optimizer.state_dict())
        else:
            # Independent CUDA reductions may differ in roundoff. Actual Adam
            # counts and every RNG stream still have to match exactly.
            torch.testing.assert_close(control.policy.state_dict(), measured.policy.state_dict(), rtol=1e-5, atol=2e-6)
            torch.testing.assert_close(control.policy.upper_optimizer.state_dict(),
                                       measured.policy.upper_optimizer.state_dict(), rtol=1e-5, atol=2e-6)
        assert 'value_mc_mse_pre_update' not in results[0]
        stats = results[1]
        for key in ('updates', 'critic_updates', 'samples_used', 'critic_samples_used'):
            assert stats[key] == results[0][key]
        assert stats['value_loss'] == pytest.approx(results[0]['value_loss'], rel=1e-5, abs=2e-6)
        assert stats['value_measurement_samples'] == len(rows)
        for name in ('mc', 'target'):
            assert stats[f'value_{name}_mse_pre_update'] == pytest.approx(expected_pre[name], rel=1e-5, abs=2e-6)
            assert stats[f'value_{name}_mse_post_update'] == pytest.approx(expected_post[name], rel=1e-5, abs=2e-6)
        assert stats['value_measurement_seconds_rank'] > 0
        assert stats['value_measurement_seconds_rank'] == (
            stats['value_measurement_pre_seconds_rank'] + stats['value_measurement_post_seconds_rank'])
        for name, learner in [('control', control), ('measured', measured)]:
            learner.save(tmp_path/f'{name}.zip')
        control = AccumulatingMaskablePPO.load(tmp_path/'control.zip', device=device)
        measured = AccumulatingMaskablePPO.load(tmp_path/'measured.zip', device=device)
        apply_unified_upper_update_overrides(measured, parser.parse_args(['--rank', '0', '--world-size', '1']))
        assert measured.policy.core.upper_config.measure_value_loss
    apply_unified_upper_update_overrides(measured, parser.parse_args(
        ['--rank', '0', '--world-size', '1', '--no-upper-measure-value-loss']))
    assert not measured.policy.core.upper_config.measure_value_loss


def test_frozen_measurement_failure_restores_modes_and_rng(unified_env, monkeypatch):
    import random
    from rl.algorithms.unified_upper import measure_critic_mse
    from rl.algorithms.upper_probe import state_digest
    model = make_model(unified_env)
    obs, _ = unified_env.reset(seed=41)
    network = model.policy.core.upper_critic
    network.train()
    modules = list(network.modules())
    modules[-1].training = False
    modes = [m.training for m in modules]
    before = state_digest((random.getstate(), np.random.get_state(), torch.get_rng_state()))
    policy = state_digest(model.policy.state_dict())
    def fail(*args, **kwargs):
        random.random(); np.random.random(); torch.rand(1)
        raise RuntimeError('measurement failed')
    monkeypatch.setattr(network, 'forward', fail)
    with pytest.raises(RuntimeError, match='measurement failed'):
        measure_critic_mse(model, {k: v[None] for k, v in obs.items()}, {'mc': np.array([1.])})
    assert [m.training for m in modules] == modes
    assert before == state_digest((random.getstate(), np.random.get_state(), torch.get_rng_state()))
    assert policy == state_digest(model.policy.state_dict())


@pytest.mark.parametrize('value', [-1, 1.5, True])
def test_critic_batch_rejects_invalid_configuration(value):
    from rl.policies.entity_joint import UpperConfig
    with pytest.raises(ValueError, match='critic effective batch'):
        UpperConfig(architecture='upper-latent-actions-v1', optimizer='adamw',
                    target_kl=None, critic_effective_batch_size=value)


def test_critic_batch_requires_unified_and_cli_positive():
    from rl.policies.entity_joint import UpperConfig
    from rl.training.distributed_sb3 import build_parser, apply_unified_upper_update_overrides
    assert UpperConfig().critic_effective_batch_size == 0
    with pytest.raises(ValueError, match='unified'):
        UpperConfig(critic_effective_batch_size=8)
    for value in ('0', '-1'):
        args = build_parser().parse_args(['--rank', '0', '--world-size', '1',
                                         '--upper-critic-effective-batch-size', value])
        with pytest.raises(ValueError, match='finite and positive'):
            apply_unified_upper_update_overrides(None, args)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_microbatch_partition_preserves_weighted_update(unified_env, tmp_path, device):
    """Changing execution chunks keeps sample/order/Adam semantics, within FP32 rounding."""
    from dataclasses import asdict, replace
    if device == 'cuda':
        assert torch.cuda.is_available(), 'explicit microbatch CUDA qualification requires CUDA'
    model = make_model(unified_env, device)
    cfg = replace(model.policy.core.upper_config, critic_effective_batch_size=8)
    model.policy.core.upper_config = cfg
    model.policy.upper_config = model.policy_kwargs['upper_config'] = asdict(cfg)
    model.set_logger(configure(str(tmp_path), []))
    obs, _ = unified_env.reset(seed=41)
    mask = unified_env.action_masks()
    with torch.no_grad():
        action, value, logp = model.policy({k:v.to(device) for k,v in tensor(obs).items()}, action_masks=mask[None])
    rows = [dict(observation=copy.deepcopy(obs), action=int(action), mask=mask.copy(),
                 value=float(value), log_prob=float(logp), reward=i/7, terminal=True,
                 env=i, episode=0) for i in range(7)]
    # Start from nonempty Adam so chunk equivalence includes saved moments.
    model._upper_completed = copy.deepcopy(rows)
    update_upper(model)
    source = tmp_path/'micro-source.zip'; model.save(source)
    states, counters, rngs = [], [], []
    for micro in (1, 2, 4):
        candidate = AccumulatingMaskablePPO.load(source, device=device)
        candidate.policy.core.upper_config = replace(cfg, micro_batch_size=micro)
        candidate.set_logger(configure(str(tmp_path), []))
        candidate._upper_completed = copy.deepcopy(rows)
        np.random.seed(914)
        result = update_upper(candidate)
        states.append((copy.deepcopy(candidate.policy.state_dict()),
                       copy.deepcopy(candidate.policy.upper_optimizer.state_dict())))
        counters.append({k: result[k] for k in ('updates','critic_updates','samples_used','critic_samples_used')})
        rngs.append(np.random.get_state())
    assert counters == [dict(updates=4., critic_updates=3., samples_used=14., critic_samples_used=21.)]*3
    for state, rng in zip(states[1:], rngs[1:]):
        torch.testing.assert_close(state, states[0], rtol=2e-5, atol=3e-6)
        np.testing.assert_equal(rng, rngs[0])
