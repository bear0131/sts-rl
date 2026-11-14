"""Strategic macro transitions and PPO; combat remains the existing NOSL learner."""
from contextlib import contextmanager
from functools import wraps
import time

import numpy as np
import torch

from env.core import encode, vocab as V
from env.core.outer_return import strategic_reward
from rl.algorithms.ppo import ppo_loss_components
from rl.algorithms.updates import optimizer_step
from rl.policies.split_policy import OVERWORLD_OBSERVATION_KEYS, OVERWORLD_PHASE_OBSERVATION_KEYS
from rl.policies.precision import full_precision
from rl.search.native import observation_as_tensor
from rl.training import reward_monitor, campfire_monitor
from rl.algorithms import upper_auxiliary

from rl.policies.deck_latent import RunContextEncoder
UPPER_KEYS = (OVERWORLD_OBSERVATION_KEYS | frozenset().union(*OVERWORLD_PHASE_OBSERVATION_KEYS.values())
              | frozenset(RunContextEncoder.keys) | {'combat_exit_anchor'})
COMBAT_PHASE = V.PHASE_VOCAB_INDEX['combat']


def enabled(model):
    return bool(getattr(getattr(model.policy, 'core', None), 'joint_upper', False))


@contextmanager
def train_phase(model, upper=False, *, small_playout=False):
    if not enabled(model):
        yield
        return
    flags = [(p, p.requires_grad) for p in model.policy.parameters()]
    active = {id(p) for p in (model.policy.small_playout.parameters() if small_playout else
                             model.policy.core.upper_parameters() if upper else
                             model.policy.core.combat.parameters())}
    try:
        for p, flag in flags:
            p.grad = None
            p.requires_grad_(flag and id(p) in active)
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def combat_update(function):
    @wraps(function)
    def call(model, *args, **kwargs):
        with train_phase(model):
            return function(model, *args, **kwargs)
    return call


def capture_step(model, obs, actions, masks, values, log_probs, new_obs, dones, infos, *, indices=None):
    """Hold the decision entering a fight across collections, until its real outcome."""
    count = len(dones)
    slots = list(range(count)) if indices is None else list(indices)
    if indices is not None:
        if (len(slots) != count or len(set(slots)) != count
                or not getattr(model, '_upper_pending', None)
                or any(type(i) is not int or not 0 <= i < len(model._upper_pending) for i in slots)):
            raise ValueError('invalid strategic subset slot mapping')
        count = len(model._upper_pending)
    if not getattr(model, '_upper_pending', None):
        model._upper_pending = [None] * count
    if not getattr(model, '_upper_episode_ids', None):
        model._upper_episode_ids = [0] * count
    if len(model._upper_pending) != count:
        raise ValueError('strategic pending environment count changed')
    def row(source, index):
        return {k: np.array(v[index], copy=True) for k, v in source.items() if k in UPPER_KEYS}
    for offset, index in enumerate(slots):
        if infos[offset].get('episode_cohort_hold'):
            continue
        if int(obs['phase'][offset, 0]) != COMBAT_PHASE:
            if model._upper_pending[index] is not None:
                raise RuntimeError('unresolved strategic action before next strategic decision')
            model._upper_pending[index] = dict(
                observation=row(obs, offset), action=int(actions[offset]),
                mask=np.array(masks[offset], copy=True), value=float(values[offset]),
                log_prob=float(log_probs[offset]), reward=0., env=index,
                episode=model._upper_episode_ids[index],
            )
        pending = model._upper_pending[index]
        censored = bool(infos[offset].get('TimeLimit.truncated') or infos[offset].get('dead_screen'))
        if pending is not None:
            stats = infos[offset].get('run_stats')
            if stats is None:
                raise ValueError('joint training requires whole-run outcome statistics')
            pending['reward'] += strategic_reward(
                round(float(obs['act'][offset, 0]) * encode.ACT_SCALE),
                round(float(obs['floor'][offset, 0]) * encode.FLOOR_SCALE),
                stats, done=bool(dones[offset]), censored=censored)
            if dones[offset] or int(new_obs['phase'][offset, 0]) != COMBAT_PHASE:
                if not censored:
                    pending['terminal'] = bool(dones[offset])
                    pending['next_observation'] = None if dones[offset] else row(new_obs, offset)
                    model._upper_completed.append(pending)
                else:
                    model._upper_censored += 1
                model._upper_pending[index] = None
        if dones[offset]:
            model._upper_episode_ids[index] += 1
            if (getattr(model, 'nosl_is_az_combat_strategy', 'model') == 'small-playout'
                    and not censored and not infos[offset].get('startup_stagger_hold')
                    and not infos[offset].get('startup_occupancy_hold')):
                model.small_playout_games_local = getattr(model, 'small_playout_games_local', 0) + 1


def macro_advantages(records, bootstraps, gamma, gae_lambda):
    """Discount once per strategic action, never once per combat click."""
    advantages = np.zeros(len(records), dtype=np.float64)
    continuation = {}
    for i in range(len(records) - 1, -1, -1):
        record = records[i]
        key = (record['env'], record['episode'])
        if record['terminal']:
            tail, future = 0., 0.
        else:
            # The same recorded V must appear with opposite signs in adjacent
            # deltas. Re-estimating every next state breaks lambda=1 cancellation
            # through mixed precision, changed batch shapes or pending records.
            tail, future = continuation[key] if key in continuation else (0., float(bootstraps[i]))
        advantages[i] = record['reward'] + gamma * future - record['value'] + gamma * gae_lambda * tail
        continuation[key] = (float(advantages[i]), float(record['value']))
    returns = advantages + np.array([r['value'] for r in records], dtype=np.float64)
    return advantages, returns


def bootstrap_indices(records):
    """Only the unfinished tail of each environment/episode needs a new V."""
    seen, indices = set(), []
    for i in range(len(records) - 1, -1, -1):
        record = records[i]
        key = (record['env'], record['episode'])
        if key not in seen and not record['terminal']:
            indices.append(i)
        seen.add(key)
    return indices[::-1]


def _training_batches(records, micro_batch_size):
    """Own immutable CPU inputs once; keep device memory bounded by one batch."""
    for start in range(0, len(records), micro_batch_size):
        batch = records[start:start + micro_batch_size]
        yield dict(
            observations={k: np.stack([r['observation'][k] for r in batch])
                          for k in batch[0]['observation']},
            actions=np.asarray([r['action'] for r in batch], dtype=np.int64),
            masks=np.stack([r['mask'] for r in batch]),
            old_values=np.asarray([r['value'] for r in batch], dtype=np.float32),
            old_log_prob=np.asarray([r['log_prob'] for r in batch], dtype=np.float32),
        )


def finish_frozen_collection(model, *, world_size=1, average=lambda values: values):
    """Report real decisions and release finished records, retaining unfinished macros.

    A frozen learner has no loss/gradient measurement. Do not fabricate zeros for
    those metrics or call the upper optimizer, even with a zero learning rate.
    """
    stats = average(dict(count=float(len(model._upper_completed)),
        censored=float(getattr(model, '_upper_censored', 0)),
        pending=float(sum(x is not None for x in model._upper_pending)),
        **reward_monitor.counts(model), **campfire_monitor.counts(model)))
    model._upper_completed = []
    return dict(frozen=1., updates=0., samples=stats['count'] * world_size,
        censored=stats['censored'] * world_size, pending=stats['pending'] * world_size,
        **reward_monitor.metrics(stats, world_size), **campfire_monitor.metrics(stats, world_size))


def update_upper(model, *, rank=0, world_size=1, average=lambda values: values, synchronize=None):
    """Configured full-batch PPO epochs, weighted by actual global row count.

    Microbatches only bound memory. Unequal/empty ranks use the same collective
    sequence, global advantage normalization and a single clip/Adam per epoch.
    """
    if not enabled(model):
        return {}
    if model.policy.core.upper_config.architecture != 'legacy':
        from rl.algorithms.unified_upper import update_unified_upper
        return update_unified_upper(model, rank=rank, world_size=world_size,
                                    average=average, synchronize=synchronize)
    records = model._upper_completed
    cfg = model.policy.core.upper_config
    count = len(records)
    bootstrap = np.zeros(count, dtype=np.float32)
    bootstrap_started = time.perf_counter()
    indices = bootstrap_indices(records)
    with torch.no_grad(), full_precision(model.device):
        for start in range(0, len(indices), cfg.micro_batch_size):
            batch = indices[start:start + cfg.micro_batch_size]
            observations = observation_as_tensor({k: np.stack([records[i]['next_observation'][k] for i in batch])
                for k in records[batch[0]]['next_observation']}, model.device)
            bootstrap[batch] = model.policy.predict_values(observations).flatten().float().cpu().numpy()
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    advantages, returns = macro_advantages(records, bootstrap, cfg.gamma, cfg.gae_lambda)
    auxiliary, aux_stats = upper_auxiliary.prepare(model)
    stats = average(dict(count=float(count), advantage_sum=float(advantages.sum(dtype=np.float64)),
        advantage_square_sum=float(np.square(advantages.astype(np.float64)).sum()),
        censored=float(getattr(model, '_upper_censored', 0)),
        pending=float(sum(x is not None for x in model._upper_pending)),
        bootstrap_samples=float(len(indices)), bootstrap_seconds=bootstrap_seconds,
        **reward_monitor.counts(model), **campfire_monitor.counts(model), **aux_stats))
    mean_count = stats['count']
    out = dict(samples=mean_count * world_size, updates=0., kl=0., policy_loss=0.,
        requested_epochs=float(cfg.epochs), requested_updates=0., samples_used=0., epochs_completed=0.,
        kl_stopped=0., kl_early_stop_enabled=float(cfg.target_kl is not None), effective_batch_size_global=0.,
        value_loss=0., entropy=0., censored=stats['censored'] * world_size,
        pending=stats['pending'] * world_size,
        bootstrap_samples=stats['bootstrap_samples'] * world_size,
        bootstrap_seconds=stats['bootstrap_seconds'], bootstrap_precision_bits=32.,
        gae_precision_bits=64., target_precision_version=1.,
        **reward_monitor.metrics(stats, world_size), **campfire_monitor.metrics(stats, world_size))
    mean_aux_count = stats.get('aux_count', 0.)
    if aux_stats:
        out.update({k: stats[k] * (1. if k.endswith('_seconds') else world_size)
                    for k in aux_stats if not k.endswith('_sum')})
        if mean_aux_count:
            out.update(aux_q_spread=stats['aux_spread_sum'] / mean_aux_count,
                       aux_target_entropy=stats['aux_entropy_sum'] / mean_aux_count,
                       aux_policy_disagreement=stats['aux_disagreement_sum'] / mean_aux_count)
    if not mean_count:
        return out
    mean = stats['advantage_sum'] / mean_count
    std = max(0., stats['advantage_square_sum'] / mean_count - mean * mean) ** .5
    out.update(advantage_mean=mean, advantage_std=std, grad_norm=0., parameter_l2_change=0.,
               clip_fraction=0., requested_updates=float(cfg.epochs), kl_stopped=0.,
               effective_batch_size_global=mean_count * world_size)
    advantages = (advantages - mean) / (std + 1e-8)
    if cfg.minibatch_size:
        if auxiliary or aux_stats:
            raise ValueError('strategic minibatch PPO currently requires auxiliary search off')
        return _update_minibatches(model, records, advantages, returns, out,
                                  rank, world_size, average, synchronize)
    batches = list(_training_batches(records, cfg.micro_batch_size))
    parameters = list(model.policy.core.upper_parameters())
    before = [p.detach().clone() for p in parameters]
    norms = []
    model.policy.set_training_mode(True)
    with train_phase(model, upper=True):
        for epoch in range(cfg.epochs):
            model.policy.upper_optimizer.zero_grad(set_to_none=True)
            totals = dict(kl=0., policy_loss=0., value_loss=0., entropy=0., clip_fraction=0.)
            for batch_index, batch in enumerate(batches):
                start = batch_index * cfg.micro_batch_size
                n = len(batch['actions'])
                observations = observation_as_tensor(batch['observations'], model.device)
                tensor = lambda x: torch.as_tensor(x, device=model.device, dtype=torch.float32)
                with model.autocast_context():
                    values, log_prob, entropy = model.policy.evaluate_actions(observations,
                        torch.as_tensor(batch['actions'], device=model.device),
                        action_masks=batch['masks'])
                components = ppo_loss_components(hierarchical=True, normalize_advantage=False,
                    observations=observations, advantages=tensor(advantages[start:start+n]),
                    returns=tensor(returns[start:start+n]), old_values=tensor(batch['old_values']),
                    old_log_prob=tensor(batch['old_log_prob']), values=values.flatten(),
                    log_prob=log_prob, entropy=entropy, clip_range=cfg.clip, clip_range_vf=None,
                    skip_combat=True)
                loss = components.policy_loss + cfg.value_coef * components.value_loss + cfg.entropy_coef * components.entropy_loss
                (loss * (n / mean_count)).backward()
                for name, value in (('kl', components.approx_kl), ('policy_loss', components.policy_loss),
                                    ('value_loss', components.value_loss), ('entropy', -components.entropy_loss),
                                    ('clip_fraction', components.clip_fraction)):
                    totals[name] += float(value.detach()) * n
            auxiliary_started = time.perf_counter()
            if aux_stats:
                totals['aux_loss'] = upper_auxiliary.backward(model, auxiliary, mean_aux_count)
            reduced = average(totals)
            out.update({k: v / mean_count for k, v in reduced.items() if k != 'aux_loss'})
            if aux_stats:
                out['aux_loss'] = reduced['aux_loss'] / max(mean_aux_count, 1.)
                out['aux_update_seconds'] = out.get('aux_update_seconds', 0.) + time.perf_counter() - auxiliary_started
            out[f'policy_loss_epoch_{epoch + 1}'] = out['policy_loss']
            out[f'kl_epoch_{epoch + 1}'] = out['kl']
            out[f'value_loss_epoch_{epoch + 1}'] = out['value_loss']
            out[f'clip_fraction_epoch_{epoch + 1}'] = out['clip_fraction']
            if not all(np.isfinite(v) for v in out.values()):
                raise FloatingPointError('nonfinite strategic PPO statistics')
            if cfg.target_kl is not None and out['kl'] > 1.5 * cfg.target_kl:
                out['kl_stopped'] = 1.
                break
            if synchronize is not None:
                synchronize(epoch)
            norms.append(optimizer_step(model, optimizer=model.policy.upper_optimizer, max_grad_norm=cfg.max_grad_norm))
            out['updates'] += 1
    out['grad_norm'] = float(torch.stack(norms).mean()) if norms else 0.
    out['parameter_l2_change'] = float(torch.stack([
        (p.detach() - old).float().square().sum() for p, old in zip(parameters, before)]).sum().sqrt())
    model._upper_completed = []
    for name, value in out.items():
        model.logger.record('upper/' + name, value)
    return out


def _update_minibatches(model, records, advantages, returns, out, rank, world_size, average, synchronize):
    """Shuffle once per pass, consume each real row once, pad only collectives.

    Each rank contributes at most minibatch_size rows to an Adam update. Empty
    ranks send zero gradients, never duplicate observations. Weighting before
    the mean gradient reduction makes the result a global *sample* mean.
    """
    cfg = model.policy.core.upper_config
    counts = average({f'rank_{i}': float(len(records)) if i == rank else 0.
                      for i in range(world_size)})
    sizes = [int(round(counts[f'rank_{i}'] * world_size)) for i in range(world_size)]
    batches_per_epoch = max((n + cfg.minibatch_size - 1) // cfg.minibatch_size for n in sizes)
    out.update(requested_updates=float(cfg.epochs * batches_per_epoch),
               requested_epochs=float(cfg.epochs), epochs_completed=0., samples_used=0.,
               effective_batch_size_global=float(sum(min(n, cfg.minibatch_size) for n in sizes)))
    parameters = list(model.policy.core.upper_parameters())
    before = [p.detach().clone() for p in parameters]
    norms = []
    model.policy.set_training_mode(True)
    with train_phase(model, upper=True):
        for epoch in range(cfg.epochs):
            order = np.random.permutation(len(records))
            epoch_totals = dict(count=0., kl=0., policy_loss=0., value_loss=0., entropy=0., clip_fraction=0.)
            for batch_index in range(batches_per_epoch):
                indices = order[batch_index * cfg.minibatch_size:(batch_index + 1) * cfg.minibatch_size]
                mean_count = sum(max(0, min(cfg.minibatch_size, n - batch_index * cfg.minibatch_size))
                                 for n in sizes) / world_size
                model.policy.upper_optimizer.zero_grad(set_to_none=True)
                totals = {key: 0. for key in epoch_totals}
                for start in range(0, len(indices), cfg.micro_batch_size):
                    selected = indices[start:start + cfg.micro_batch_size]
                    batch = next(_training_batches([records[i] for i in selected], cfg.micro_batch_size))
                    observations = observation_as_tensor(batch['observations'], model.device)
                    tensor = lambda x: torch.as_tensor(x, device=model.device, dtype=torch.float32)
                    with model.autocast_context():
                        values, log_prob, entropy = model.policy.evaluate_actions(observations,
                            torch.as_tensor(batch['actions'], device=model.device), action_masks=batch['masks'])
                    c = ppo_loss_components(hierarchical=True, normalize_advantage=False,
                        observations=observations, advantages=tensor(advantages[selected]),
                        returns=tensor(returns[selected]), old_values=tensor(batch['old_values']),
                        old_log_prob=tensor(batch['old_log_prob']), values=values.flatten(),
                        log_prob=log_prob, entropy=entropy, clip_range=cfg.clip, clip_range_vf=None,
                        skip_combat=True)
                    loss = c.policy_loss + cfg.value_coef * c.value_loss + cfg.entropy_coef * c.entropy_loss
                    (loss * (len(selected) / mean_count)).backward()
                    totals['count'] += len(selected)
                    for name, value in (('kl', c.approx_kl), ('policy_loss', c.policy_loss),
                                        ('value_loss', c.value_loss), ('entropy', -c.entropy_loss),
                                        ('clip_fraction', c.clip_fraction)):
                        totals[name] += float(value.detach()) * len(selected)
                reduced = average(totals)
                if not all(np.isfinite(v) for v in reduced.values()):
                    raise FloatingPointError('nonfinite strategic minibatch PPO statistics')
                if not np.isclose(reduced['count'], mean_count):
                    raise RuntimeError('strategic minibatch collective row count mismatch')
                out.update({k: v / mean_count for k, v in reduced.items() if k != 'count'})
                if cfg.target_kl is not None and out['kl'] > 1.5 * cfg.target_kl:
                    out['kl_stopped'] = 1.
                    break
                if synchronize is not None:
                    synchronize(epoch * batches_per_epoch + batch_index)
                norms.append(optimizer_step(model, optimizer=model.policy.upper_optimizer,
                                            max_grad_norm=cfg.max_grad_norm))
                out['updates'] += 1
                out['samples_used'] += reduced['count'] * world_size
                for k, v in reduced.items():
                    epoch_totals[k] += v
            if epoch_totals['count']:
                for name in ('policy_loss', 'value_loss', 'kl', 'clip_fraction'):
                    out[f'{name}_epoch_{epoch + 1}'] = epoch_totals[name] / epoch_totals['count']
            if out['kl_stopped']:
                break
            out['epochs_completed'] += 1
    out['grad_norm'] = float(torch.stack(norms).mean()) if norms else 0.
    out['parameter_l2_change'] = float(torch.stack([
        (p.detach() - old).float().square().sum() for p, old in zip(parameters, before)]).sum().sqrt())
    model._upper_completed = []
    for name, value in out.items():
        model.logger.record('upper/' + name, value)
    return out
