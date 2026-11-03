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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_two_updates_save_load_and_old_value_freezing(unified_env, tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.fail('selected CUDA qualification requires an available device')
    model = make_model(unified_env, device)
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
        assert result['updates'] == 4 and result['critic_updates'] == 6
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
        assert actor_steps == {4 * (iteration+1)} and critic_steps == {6 * (iteration+1)}
        model = loaded
        model.set_logger(configure(str(tmp_path), []))


def test_unequal_empty_ranks_match_global_weighted_update(unified_env, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from dataclasses import replace
    from rl.training.distributed_sb3 import pack_gradients, install_flat_gradients
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
        model.policy.core.upper_config = replace(model.policy.core.upper_config, effective_batch_size=64)
        model.set_logger(configure(str(tmp_path), []))
        model._upper_completed, model._upper_pending = copy.deepcopy(samples), []
        return model
    ready(reference, rows)
    update_upper(reference)
    models = [ready(make_model(unified_env), samples) for samples in ([], rows[:3], rows[3:])]
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
        for p, q in zip(model.policy.parameters(), reference.policy.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('calibration_games', [0, 2])
def test_game_holdout_candidate_probe_restores_adam_and_repeats_control(unified_env, tmp_path, device, monkeypatch, calibration_games):
    from unittest.mock import patch
    from rl.algorithms import unified_upper as U
    from rl.algorithms.upper_probe import run_probe, state_digest, split_games, categorical_kl
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware unavailable; CPU result is not GPU qualification')
    model = make_model(unified_env, device=device)
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
        holdout_games_per_rank=2, max_probe_decisions_per_split_per_rank=10),
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
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    result = run_probe(model, protocol, output_dir=tmp_path/'probe')
    assert torch.are_deterministic_algorithms_enabled() == deterministic_before
    assert result['control_repeat_exact'] and result['source_state_restored']
    assert result['rng_restored'] and result['lower_optimizer_unchanged']
    assert result['training_games'] == 5-calibration_games and result['holdout_games'] == 2
    assert result['calibration_games'] == calibration_games
    if calibration_games:
        assert result['frozen_scene_calibration']['calibration_games'] == calibration_games
    assert all(r['stats']['samples'] == 5-calibration_games and r['stats']['samples_used'] == 2*(5-calibration_games)
               for r in result['arms'])
    assert result['arms'][0]['probes'] == result['arms'][-1]['probes']
    assert result['arms'][0]['final_digest'] == result['arms'][1]['final_digest']
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
