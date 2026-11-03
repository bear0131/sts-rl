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


def compare_probe_arrays(reference, current, tolerances):
    """FP32 output repeatability, distinct from exact source-state restore."""
    report = {}
    for key, old in reference.items():
        new = current[key]
        if new.shape != old.shape:
            raise ValueError('repeat probe shapes differ')
        difference = np.abs(new - old)
        relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(old), 1e-20))
        maximum = float(difference.max(initial=0.))
        report[key] = dict(max_abs=maximum, relative_l2=relative_l2,
            passed=bool(np.isfinite(new).all() and maximum <= tolerances[key]
                        and relative_l2 <= tolerances['relative_l2']))
    return report


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
        training_games=len(games) - limits['holdout_games_per_rank'],
        holdout_games=limits['holdout_games_per_rank'], training_samples=len(train),
        holdout_samples=len(hold), initial_digest=initial_digest, arms=[])
    control_outputs = {}
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
                target = lambda rows: U.macro_advantages(rows, np.zeros(len(rows)), 1., cfg.gae_lambda)[1]
            with patch.object(U, 'complete_returns', target):
                stats = U.update_unified_upper(model, rank=rank, world_size=world_size,
                    average=average, synchronize=synchronize)
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            result = dict(id=arm['id'], config=asdict(candidate), stats=stats,
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
                    critic_explained_variance=1-float(np.var(residual))/variance if variance > 0 else None,
                    old_logp_max_disagreement=float(np.max(np.abs(old_logp[np.arange(len(rows)), actions]-sampled_old))))
                if not all(np.isfinite(v).all() for v in (kl, values, surrogate_gain)):
                    raise FloatingPointError('nonfinite candidate probes')
                result['probes'][name] = summary
                current_outputs = dict(kl=kl, value=values, surrogate_gain=surrogate_gain)
                if arm['id'] == 'control':
                    control_outputs[name] = current_outputs
                elif arm['id'] == 'control_repeat' and 'repeat_tolerances' in limits:
                    results.setdefault('control_repeat_differences', {})[name] = compare_probe_arrays(
                        control_outputs[name], current_outputs, limits['repeat_tolerances'])
                if output:
                    np.savez_compressed(output/f'{arm["id"]}-{name}.npz', indices=indices,
                        kl=kl, legal=legal, phase=phases, surrogate_gain=surrogate_gain,
                        value=values, old_value=old_values, mc_return=target_values,
                        lambda_return=lambda_returns[indices], env=[r['env'] for r in rows],
                        episode=[r['episode'] for r in rows])
            results['arms'].append(result)
            if output:
                (output/'probe.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        if results['arms'][0]['id'] == 'control' and results['arms'][-1]['id'] == 'control_repeat':
            results['control_repeat_exact'] = results['arms'][0]['final_digest'] == results['arms'][-1]['final_digest']
            results['control_repeat_numerical'] = (all(item['passed']
                for split in results['control_repeat_differences'].values() for item in split.values())
                if 'control_repeat_differences' in results else None)
            passed = (results['control_repeat_numerical'] if 'repeat_tolerances' in limits
                      else results['control_repeat_exact'])
            if not passed:
                raise AssertionError('candidate screen changed repeated control update')
    finally:
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
