"""Bounded, reversible update diagnostics; never selects or saves a policy.

Game-level holdout and exact old-to-final categorical KL supplement the
production updater's changing-minibatch sampled-action statistics.
"""
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from rl.algorithms import unified_upper as U
from env.core import encode, vocab


def room_key(observation):
    """Public native room visit: floor advances on entry, including boss treasure.

    Event subpages, embedded combat, shop and reward menus retain this key.
    Act disambiguates transitions between maps; phase is deliberately excluded.
    """
    values = []
    for key, scale in (('act', encode.ACT_SCALE), ('floor', encode.FLOOR_SCALE)):
        value = float(observation[key][0]) * scale
        if not np.isfinite(value) or abs(value - round(value)) > 1e-4:
            raise ValueError('room identity must contain integer public act/floor')
        values.append(round(value))
    return tuple(values)


def fit_room_calibration(records, settings, *, average=lambda x: x, world_size=1):
    """Freeze scene bias estimates using separate complete calibration games.

    One contribution per game/scene prevents long menu sequences dominating.
    Residual spread contains outcome noise; it is not epistemic uncertainty.
    """
    returns = U.complete_returns(records)
    groups, games = {}, {}
    for row, target in zip(records, returns):
        game = (row['env'], row['episode'])
        scene = (room_key(row['observation'])[0], int(row['observation']['phase'][0]))
        residual = float(row['value']) - target
        groups.setdefault((game, scene), []).append(residual)
        games.setdefault(game, []).append(residual)
    counts = dict(global_n=float(len(games)), global_sum=0., global_square=0.)
    for values in games.values():
        mean = float(np.mean(values))
        counts['global_sum'] += mean
        counts['global_square'] += mean*mean
    for act in range(5):
        for phase in range(len(vocab.PHASE_VOCAB)):
            for field in ('n', 'sum', 'square'):
                counts[f'{act}:{phase}:{field}'] = 0.
    for (_, (act, phase)), values in groups.items():
        mean = float(np.mean(values))
        counts[f'{act}:{phase}:n'] += 1
        counts[f'{act}:{phase}:sum'] += mean
        counts[f'{act}:{phase}:square'] += mean*mean
    reduced = {k: v*world_size for k, v in average(counts).items()}
    if reduced['global_n'] <= 0:
        raise ValueError('adaptive lambda requires distinct calibration games')
    global_mean = reduced['global_sum']/reduced['global_n']
    global_second = reduced['global_square']/reduced['global_n']
    base, span = settings['base'], settings['span']
    prior, floor = settings['shrinkage_games'], settings['noise_floor']
    if not (0 <= base <= base+span <= 1 and prior >= 0 and floor > 0):
        raise ValueError('invalid scene calibration settings')
    scenes = {}
    for act in range(5):
        for phase in range(len(vocab.PHASE_VOCAB)):
            key = f'{act}:{phase}'
            n = reduced[f'{key}:n']
            if n+prior:
                mean = (reduced[f'{key}:sum']+prior*global_mean)/(n+prior)
                second = (reduced[f'{key}:square']+prior*global_second)/(n+prior)
            else:
                mean, second = global_mean, global_second
            spread = max(0., second-mean*mean)**.5
            lam = base+span*abs(mean)/(abs(mean)+spread+floor)
            scenes[key] = dict(games=n, bias=mean, residual_spread=spread, coefficient=lam)
    return dict(settings=settings, calibration_games=reduced['global_n'],
                global_bias=global_mean, scenes=scenes)


def critic_lambda_returns(records, coefficient, scope='decision', calibration=None):
    """Frozen-old-V targets, gamma=1, on complete natural upper trajectories.

    Room scope has lambda=1 on every within-room transition. Only crossing
    an act/floor visit boundary applies coefficient; terminal never bootstraps.
    Actor advantages and the original records are untouched.
    """
    if not np.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError('critic lambda must be between zero and one')
    if scope not in ('decision', 'room', 'calibrated_room'):
        raise ValueError('unknown critic lambda scope')
    if scope == 'calibrated_room' and calibration is None:
        raise ValueError('adaptive lambda requires frozen calibration')
    if U.bootstrap_indices(records):
        raise ValueError('critic lambda requires complete natural episodes')
    targets = np.zeros(len(records), dtype=np.float64)
    tails = {}
    for i in range(len(records)-1, -1, -1):
        row = records[i]
        key = (row['env'], row['episode'])
        if row['terminal']:
            targets[i] = row['reward']
        else:
            next_target, next_value, next_room, next_scene = tails[key]
            lam = coefficient
            if scope != 'decision':
                if room_key(row['observation']) == next_room:
                    lam = 1.
                elif scope == 'calibrated_room':
                    lam = calibration['scenes'][next_scene]['coefficient']
            targets[i] = row['reward'] + (1-lam)*next_value + lam*next_target
        current_room = room_key(row['observation']) if scope != 'decision' else None
        scene = (f"{current_room[0]}:{int(row['observation']['phase'][0])}"
                 if scope == 'calibrated_room' else None)
        tails[key] = (targets[i], float(row['value']),
                      current_room, scene)
    return targets


def room_transition_audit(records):
    """Check visit continuity against the collector's actual successor frames."""
    counts = dict(within_room=0, across_room=0, across_act=0, terminals=0)
    successors = {}
    for row in reversed(records):
        key = (row['env'], row['episode'])
        current = room_key(row['observation'])
        if row['terminal']:
            counts['terminals'] += 1
        else:
            following = successors[key]
            if following < current:
                raise ValueError('room visits moved backwards within a game')
            if row.get('next_observation') is not None:
                if room_key(row['next_observation']) != following:
                    raise ValueError('recorded room successor differs from next decision')
            counts['within_room' if current == following else 'across_room'] += 1
            counts['across_act'] += int(current[0] != following[0])
        successors[key] = current
    return counts


def state_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def split_games(records, heldout_games, seed):
    keys = sorted({(r['env'], r['episode']) for r in records})
    if not 0 < heldout_games < len(keys):
        raise ValueError('holdout must retain distinct complete training games')
    rng = np.random.default_rng(seed)
    selected = {keys[i] for i in rng.choice(len(keys), heldout_games, replace=False)}
    train = [i for i, r in enumerate(records) if (r['env'], r['episode']) not in selected]
    hold = [i for i, r in enumerate(records) if (r['env'], r['episode']) in selected]
    for indices in (train, hold):
        U.complete_returns([records[i] for i in indices])
    return train, hold


def categorical_kl(old_logp, new_logp, masks):
    """KL(old||new) over all legal actions, not only sampled actions."""
    old_logp, new_logp = np.asarray(old_logp), np.asarray(new_logp)
    if old_logp.shape != new_logp.shape or old_logp.shape != masks.shape:
        raise ValueError('categorical probe shapes differ')
    return np.where(masks, np.exp(old_logp) * (old_logp - new_logp), 0.).sum(-1)


def evaluate(model, records, indices):
    logps, values = [], []
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(indices), model.policy.core.upper_config.micro_batch_size):
            batch = [records[i] for i in indices[start:start + model.policy.core.upper_config.micro_batch_size]]
            keys = set().union(*(r['observation'] for r in batch))
            obs = {k: torch.as_tensor(np.stack([r['observation'].get(k,
                np.zeros(model.observation_space[k].shape)) for r in batch]), device=model.device) for k in keys}
            masks = torch.as_tensor(np.stack([r['mask'] for r in batch]), device=model.device)
            logits = model.policy.core.upper_actor(obs).double().masked_fill(~masks, -1e9)
            logps.append(logits.log_softmax(-1).cpu().numpy())
            values.append(model.policy.core.upper_critic(obs).flatten().double().cpu().numpy())
    return np.concatenate(logps), np.concatenate(values)


def _distribution(values):
    if not len(values):
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, maximum=None)
    return dict(count=len(values), mean=float(np.mean(values)),
        p50=float(np.quantile(values, .5)), p95=float(np.quantile(values, .95)),
        p99=float(np.quantile(values, .99)), maximum=float(np.max(values)))


def run_probe(model, protocol, *, rank=0, world_size=1, average=lambda x: x,
              synchronize=None, output_dir=None):
    """Compare candidates on identical data/Adam/RNG; restore even on failure.

    The caller owns collection and its wall timeout. This function reuses the
    production updater and synchronization, then restores all policy/Adam/RNG.
    No training checkpoint is written. Probe arrays stay in the output folder.
    """
    cfg = model.policy.core.upper_config
    if cfg.architecture != 'upper-latent-actions-v1':
        raise ValueError('candidate screen requires the unified upper model')
    records = model._upper_completed
    limits = protocol['diagnostic']
    if world_size != limits['world_size']:
        raise ValueError('diagnostic world size differs from protocol')
    games = {(r['env'], r['episode']) for r in records}
    if len(games) != limits['games_per_rank']:
        raise ValueError('diagnostic requires every complete game')
    returns = U.complete_returns(records)
    advantages, lambda_returns = U.macro_advantages(records, np.zeros(len(records)), 1., cfg.gae_lambda)
    train, hold = split_games(records, limits['holdout_games_per_rank'], protocol['seed'] + rank)
    calibration_count = limits.get('calibration_games_per_rank', 0)
    calibration = None
    if calibration_count:
        remaining, selected = split_games([records[i] for i in train], calibration_count,
                                          protocol['seed'] + rank + 30000)
        calibration_rows = [records[train[i]] for i in selected]
        train = [train[i] for i in remaining]
        calibration = fit_room_calibration(calibration_rows, protocol['adaptive_lambda'],
                                            average=average, world_size=world_size)
    train_rows = [records[i] for i in train]
    normalization = average(dict(n=float(len(train)), total=float(advantages[train].sum()),
        square=float(np.square(advantages[train]).sum())))
    mean = normalization['total'] / normalization['n']
    std = max(0., normalization['square'] / normalization['n'] - mean * mean) ** .5
    normalized = (advantages - mean) / (std + 1e-8)
    random_probe = np.random.default_rng(protocol['seed'] + rank + 10000)
    probes = {name: sorted(random_probe.choice(indices, min(len(indices),
        limits['max_probe_decisions_per_split_per_rank']), replace=False).tolist())
        for name, indices in [('train', train), ('holdout', hold)]}
    policy = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.policy.upper_optimizer.state_dict())
    configs = (model.policy.upper_config, model.policy_kwargs['upper_config'])
    training = model.policy.training
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if model.device.type == 'cuda' else None)
    lower_optimizer = getattr(model.policy, 'small_playout_optimizer', None)
    lower_digest = state_digest(lower_optimizer.state_dict()) if lower_optimizer is not None else None
    initial_digest = state_digest(dict(policy=policy, optimizer=optimizer))
    output = Path(output_dir) if output_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    results = dict(rank=rank, world_size=world_size, games=len(games),
        training_games=len(games) - limits['holdout_games_per_rank'] - calibration_count,
        holdout_games=limits['holdout_games_per_rank'], training_samples=len(train),
        holdout_samples=len(hold), initial_digest=initial_digest, arms=[],
        calibration_games=calibration_count, frozen_scene_calibration=calibration)
    if any(arm.get('lambda_scope') in ('room','calibrated_room') for arm in protocol['arms']):
        results['room_transition_audit'] = room_transition_audit(records)
    def reset():
        model.policy.load_state_dict(policy)
        model.policy.upper_optimizer.load_state_dict(copy.deepcopy(optimizer))
        model.policy.upper_optimizer.zero_grad(set_to_none=True)
        model.policy.core.upper_config = cfg
        model.policy.upper_config, model.policy_kwargs['upper_config'] = configs
        model._upper_completed = records
        model.policy.set_training_mode(training)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    # RNG restoration alone does not order CUDA embedding/index reductions.
    # Exact control replay is a diagnostic contract, not a production setting.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    results['deterministic_algorithms'] = True
    try:
        old = {name: evaluate(model, records, indices) for name, indices in probes.items()}
        for arm in protocol['arms']:
            reset()
            candidate = replace(cfg, **arm['changes'])
            model.policy.core.upper_config = candidate
            model.policy.upper_config = asdict(candidate)
            model.policy_kwargs['upper_config'] = asdict(candidate)
            actor_ids = {id(p) for p in model.policy.core.upper_actor.parameters()}
            for group in model.policy.upper_optimizer.param_groups:
                if any(id(p) in actor_ids for p in group['params']):
                    group['lr'] = candidate.learning_rate
            np.random.seed(protocol['seed'] + rank)
            model._upper_completed = train_rows
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            started = time.perf_counter()
            target = U.complete_returns
            if arm.get('critic_target') == 'frozen_lambda_return':
                coefficient = arm.get('critic_lambda', cfg.gae_lambda)
                scope = arm.get('lambda_scope', 'decision')
                target = lambda rows: critic_lambda_returns(rows, coefficient, scope, calibration)
            arm_targets = target(records)
            with patch.object(U, 'complete_returns', target):
                stats = U.update_unified_upper(model, rank=rank, world_size=world_size,
                    average=average, synchronize=synchronize)
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            result = dict(id=arm['id'], config=asdict(candidate), stats=stats,
                critic_target_spec={k: arm[k] for k in ('critic_target', 'critic_lambda', 'lambda_scope') if k in arm},
                update_seconds=time.perf_counter() - started,
                final_digest=state_digest(dict(policy=model.policy.state_dict(),
                    optimizer=model.policy.upper_optimizer.state_dict())), probes={})
            for name, indices in probes.items():
                old_logp, old_values = old[name]
                logp, values = evaluate(model, records, indices)
                rows = [records[i] for i in indices]
                masks = np.stack([r['mask'] for r in rows]).astype(bool)
                kl = categorical_kl(old_logp, logp, masks)
                legal = masks.sum(-1)
                phases = np.array([int(r['observation']['phase'][0]) for r in rows])
                actions = np.array([r['action'] for r in rows])
                sampled_old = np.array([r['log_prob'] for r in rows])
                ratio = np.exp(logp[np.arange(len(rows)), actions] - sampled_old)
                reference_ratio = np.exp(old_logp[np.arange(len(rows)), actions] - sampled_old)
                adv = normalized[indices]
                surrogate_gain = np.minimum(ratio * adv, np.clip(ratio, 1-cfg.clip, 1+cfg.clip) * adv) - reference_ratio * adv
                target_values = returns[indices]
                residual = target_values - values
                variance = float(np.var(target_values))
                summary = dict(kl_all=_distribution(kl), kl_single=_distribution(kl[legal == 1]),
                    kl_multiple=_distribution(kl[legal > 1]),
                    phase_multiple={str(p): _distribution(kl[(phases == p) & (legal > 1)]) for p in np.unique(phases)},
                    surrogate_gain_mean=float(surrogate_gain.mean()),
                    critic_mc_mse=float(np.square(residual).mean()),
                    critic_old_mc_mse=float(np.square(target_values-old_values).mean()),
                    critic_bias=float((values-target_values).mean()),
                    critic_mc_mse_by_phase={str(p): float(np.square(residual[phases == p]).mean())
                        for p in np.unique(phases)},
                    critic_target_mc_mse=float(np.square(arm_targets[indices]-target_values).mean()),
                    critic_explained_variance=1-float(np.var(residual))/variance if variance > 0 else None,
                    old_logp_max_disagreement=float(np.max(np.abs(old_logp[np.arange(len(rows)), actions]-sampled_old))))
                if not all(np.isfinite(v).all() for v in (kl, values, surrogate_gain)):
                    raise FloatingPointError('nonfinite candidate probes')
                result['probes'][name] = summary
                if output:
                    np.savez_compressed(output/f'{arm["id"]}-{name}.npz', indices=indices,
                        kl=kl, legal=legal, phase=phases, surrogate_gain=surrogate_gain,
                        value=values, old_value=old_values, mc_return=target_values,
                        critic_target=arm_targets[indices],
                        lambda_return=lambda_returns[indices], env=[r['env'] for r in rows],
                        episode=[r['episode'] for r in rows])
            results['arms'].append(result)
            if output:
                (output/'probe.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        if results['arms'][0]['id'] == 'control' and results['arms'][-1]['id'] == 'control_repeat':
            results['control_repeat_exact'] = results['arms'][0]['final_digest'] == results['arms'][-1]['final_digest']
            if not results['control_repeat_exact']:
                raise AssertionError('candidate screen changed repeated control update')
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        reset()
        results['source_state_restored'] = state_digest(dict(policy=model.policy.state_dict(),
            optimizer=model.policy.upper_optimizer.state_dict())) == initial_digest
        results['rng_restored'] = state_digest(rng) == state_digest((random.getstate(),
            np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if model.device.type == 'cuda' else None))
        results['lower_optimizer_unchanged'] = (lower_optimizer is None or
            state_digest(lower_optimizer.state_dict()) == lower_digest)
        if output:
            (output/'probe.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        if not all(results[k] for k in ('source_state_restored', 'rng_restored', 'lower_optimizer_unchanged')):
            raise AssertionError('candidate screen failed to restore source model/Adam')
    return results
