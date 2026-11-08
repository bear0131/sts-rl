"""Whole-game heldout diagnosis using the unchanged online lower updater."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from rl.algorithms.small_playout import update_online
from rl.algorithms.upper_probe import split_games, state_digest
from rl.policies.small_playout import OnlinePlayoutConfig
from rl.training.resource_monitor import ResourceMonitor


def split_lower_games(samples, records, heldout_games, seed):
    """A complete cohort has one episode per environment; never split roots."""
    episodes = {}
    for row in records:
        episodes.setdefault(int(row['env']), set()).add(int(row['episode']))
    if any(len(v) != 1 for v in episodes.values()):
        raise ValueError('lower probe requires one complete episode per environment')
    train, hold = split_games(records, heldout_games, seed)
    envs = [{int(records[i]['env']) for i in ids} for ids in (train, hold)]
    if envs[0] & envs[1] or any(int(row['env']) not in episodes for row in samples):
        raise ValueError('lower sample has ambiguous or unknown game identity')
    indices = [[i for i, row in enumerate(samples) if int(row['env']) in group] for group in envs]
    if not all(indices):
        raise ValueError('lower probe requires samples in both complete-game splits')
    return indices, [sorted(group) for group in envs]


@torch.no_grad()
def evaluate_lower(net, prepared, indices, *, device, micro_batch_size):
    """Measure the actual reverse-KL target and both moving/frozen guide teachers."""
    values = {k: [] for k in ('reverse_kl', 'forward_kl', 'teacher_top1',
        'prior_kl', 'guide_current_root_kl', 'guide_frozen_root_kl', 'guide_teacher_top1')}
    for offset in range(0, len(indices), micro_batch_size):
        ids = indices[offset:offset + micro_batch_size]
        inputs, target, teacher = prepared.batch(ids, device)
        full, guide = net.scores_and_guide(*inputs)
        legal = inputs[-1]
        observed = torch.isfinite(target)
        logp = full.masked_fill(~observed, -torch.inf).log_softmax(-1).masked_fill(~observed, 0.)
        logt = target.masked_fill(~observed, 0.)
        p = logp.exp().masked_fill(~observed, 0.)
        root = full.masked_fill(~legal, -torch.inf).log_softmax(-1).masked_fill(~legal, 0.)
        glp = guide.masked_fill(~legal, -torch.inf).log_softmax(-1).masked_fill(~legal, 0.)
        prior = torch.full_like(full, -torch.inf)
        for j, i in enumerate(ids):
            row = prepared.rows[int(i)]
            prior[j, :len(row['prior_logits'])] = torch.as_tensor(row['prior_logits'], device=device)
        prior = prior.log_softmax(-1).masked_fill(~legal, 0.)
        prior_p = prior.exp().masked_fill(~legal, 0.)
        root_p = root.exp().masked_fill(~legal, 0.)
        batch = dict(reverse_kl=(p * (logp - logt)).sum(-1),
            forward_kl=(target.exp() * (logt - logp)).sum(-1),
            teacher_top1=(full.masked_fill(~legal, -torch.inf).argmax(-1) == teacher).float(),
            prior_kl=(prior_p * (prior - root)).sum(-1),
            guide_current_root_kl=(root_p * (root - glp)).sum(-1),
            guide_frozen_root_kl=(prior_p * (prior - glp)).sum(-1),
            guide_teacher_top1=(guide.masked_fill(~legal, -torch.inf).argmax(-1) == teacher).float())
        for key, value in batch.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError('nonfinite lower probe metric: ' + key)
            values[key].append(value.double().cpu().numpy())
    return {k: np.concatenate(v) for k, v in values.items()}


def run_lower_probe(model, protocol, *, rank=0, world_size=1, average=lambda x: x,
                    synchronize=None, output_dir=None):
    """Restore policy, Adam, RNG, counters and collected samples after every arm."""
    limits = protocol['diagnostic']
    cfg = OnlinePlayoutConfig(**model.policy.small_playout_config)
    if cfg.context_mode != 'token-root-64x2' or world_size != limits['world_size']:
        raise ValueError('lower diagnostic supports the current token-root cohort only')
    samples = model._small_playout_samples
    records = model._upper_completed
    if len({(r['env'], r['episode']) for r in records}) != limits['games_per_rank']:
        raise ValueError('lower diagnostic requires the complete cohort')
    splits, games = split_lower_games(samples, records, limits['holdout_games_per_rank'], protocol['seed'] + rank)
    net = model.policy.small_playout
    prepared = net.prepare_samples(samples, cfg.target_mix)
    policy = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.policy.small_playout_optimizer.state_dict())
    upper_digest = state_digest(model.policy.upper_optimizer.state_dict())
    kwargs = copy.deepcopy(model.policy_kwargs)
    original_config = copy.deepcopy(model.policy.small_playout_config)
    original_steps = getattr(model, 'small_playout_optimizer_steps', None)
    training = model.policy.training
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if model.device.type == 'cuda' else None)
    digest = state_digest(dict(policy=policy, optimizer=optimizer))
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    output = Path(output_dir) if output_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    report = dict(rank=rank, world_size=world_size, games_per_split=games,
        samples_per_split=[len(x) for x in splits], source_digest=digest, arms=[])
    control = {}

    def reset():
        model.policy.load_state_dict(policy)
        model.policy.small_playout_optimizer.load_state_dict(copy.deepcopy(optimizer))
        model.policy.small_playout_optimizer.zero_grad(set_to_none=True)
        model.policy.small_playout_config = copy.deepcopy(original_config)
        model.policy_kwargs = copy.deepcopy(kwargs)
        model._small_playout_samples = samples
        model._upper_completed = records
        if original_steps is None:
            model.__dict__.pop('small_playout_optimizer_steps', None)
        else:
            model.small_playout_optimizer_steps = original_steps
        model.policy.set_training_mode(training)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])

    def measure(label):
        model.policy.set_training_mode(False)
        result = {}
        for name, ids in zip(('train', 'holdout'), splits, strict=True):
            arrays = evaluate_lower(net, prepared, ids, device=model.device,
                micro_batch_size=cfg.micro_batch_size)
            totals = average(dict(n=float(len(ids)), **{k: float(v.sum()) for k, v in arrays.items()}))
            result[name] = {k: v / totals['n'] for k, v in totals.items() if k != 'n'}
            result[name]['samples_global'] = int(round(totals['n'] * world_size))
            if output:
                np.savez_compressed(output/f'{label}-{name}.npz', **arrays,
                    env=np.array([int(samples[i]['env']) for i in ids]), indices=ids)
            if label == 'control':
                control[name] = arrays
            if label == 'control_repeat':
                differences = {k: float(np.max(np.abs(v - control[name][k]))) for k, v in arrays.items()}
                if any(not np.allclose(v, control[name][k], atol=1e-5, rtol=3e-5) for k, v in arrays.items()):
                    raise AssertionError('lower repeated control changed heldout results')
                report.setdefault('control_repeat_differences', {})[name] = differences
        return result

    try:
        torch.use_deterministic_algorithms(True)
        report['pre'] = measure('pre')
        for arm in protocol['lower_arms']:
            reset()
            if set(arm['changes']) - {'learning_rate', 'epochs'}:
                raise ValueError('lower screen varies only learning rate and epochs')
            candidate = replace(cfg, **arm['changes'])
            model.policy.small_playout_config = asdict(candidate)
            model.policy_kwargs['small_playout_config'] = asdict(candidate)
            for group in model.policy.small_playout_optimizer.param_groups:
                group['lr'] = candidate.learning_rate
            model._small_playout_samples = [samples[i] for i in splits[0]]
            np.random.seed(protocol['seed'] + rank)
            model._sync_device()
            started = time.perf_counter()
            with ResourceMonitor(interval=1., gpu=model.device.type == 'cuda', torch_device=model.device) as monitor:
                stats = update_online(model, rank=rank, world_size=world_size,
                    average=average, synchronize=synchronize)
                model._sync_device()
            row = dict(id=arm['id'], config=asdict(candidate), stats=stats,
                actual_lrs=[g['lr'] for g in model.policy.small_playout_optimizer.param_groups],
                update_seconds=time.perf_counter() - started, resources=monitor.summary,
                final_digest=state_digest(dict(policy=model.policy.state_dict(),
                    optimizer=model.policy.small_playout_optimizer.state_dict())))
            row['post'] = measure(arm['id'])
            report['arms'].append(row)
            if output:
                (output/'lower-probe.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        if report['arms'][0]['id'] == 'control' and report['arms'][-1]['id'] == 'control_repeat':
            report['control_repeat_exact'] = report['arms'][0]['final_digest'] == report['arms'][-1]['final_digest']
    finally:
        reset()
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        report['source_state_restored'] = state_digest(dict(policy=model.policy.state_dict(),
            optimizer=model.policy.small_playout_optimizer.state_dict())) == digest
        report['upper_optimizer_unchanged'] = state_digest(model.policy.upper_optimizer.state_dict()) == upper_digest
        report['rng_restored'] = state_digest(rng) == state_digest((random.getstate(), np.random.get_state(),
            torch.get_rng_state(), torch.cuda.get_rng_state_all() if model.device.type == 'cuda' else None))
        report['completed'] = len(report['arms']) == len(protocol['lower_arms']) and all(report[k]
            for k in ('source_state_restored', 'upper_optimizer_unchanged', 'rng_restored'))
        if output:
            (output/'lower-probe.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        if not all(report[k] for k in ('source_state_restored', 'upper_optimizer_unchanged', 'rng_restored')):
            raise AssertionError('lower probe failed to restore source state')
    return report
