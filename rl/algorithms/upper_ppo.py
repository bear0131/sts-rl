"""Strategic macro transitions and PPO; combat remains the existing NOSL learner."""
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch

from env.core import encode, vocab as V
from env.core.outer_return import strategic_reward
from rl.algorithms.ppo import ppo_loss_components
from rl.algorithms.updates import optimizer_step
from rl.policies.split_policy import OVERWORLD_OBSERVATION_KEYS, OVERWORLD_PHASE_OBSERVATION_KEYS
from rl.search.native import observation_as_tensor
from rl.training import reward_monitor

UPPER_KEYS = OVERWORLD_OBSERVATION_KEYS | frozenset().union(*OVERWORLD_PHASE_OBSERVATION_KEYS.values())
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
    encoder = getattr(getattr(model.policy.core, 'overworld', None), 'encoder', None)
    layers = getattr(getattr(encoder, 'transformer', None), 'layers', ())
    checkpoint_flags = [(layer, layer.checkpoint_training) for layer in layers
                        if hasattr(layer, 'checkpoint_training')]
    try:
        for layer, _ in checkpoint_flags:
            layer.checkpoint_training = upper and not small_playout
        for p, flag in flags:
            p.grad = None
            p.requires_grad_(flag and id(p) in active)
        yield
    finally:
        for layer, flag in checkpoint_flags:
            layer.checkpoint_training = flag
        for p, flag in flags:
            p.requires_grad_(flag)


def combat_update(function):
    @wraps(function)
    def call(model, *args, **kwargs):
        with train_phase(model):
            return function(model, *args, **kwargs)
    return call


def capture_step(model, obs, actions, masks, values, log_probs, new_obs, dones, infos):
    """Hold the decision entering a fight across collections, until its real outcome."""
    count = len(dones)
    if not getattr(model, '_upper_pending', None):
        model._upper_pending = [None] * count
    if not getattr(model, '_upper_episode_ids', None):
        model._upper_episode_ids = [0] * count
    if len(model._upper_pending) != count:
        raise ValueError('strategic pending environment count changed')
    def row(source, index):
        return {k: np.array(v[index], copy=True) for k, v in source.items() if k in UPPER_KEYS}
    for index in range(count):
        if int(obs['phase'][index, 0]) != COMBAT_PHASE:
            if model._upper_pending[index] is not None:
                raise RuntimeError('unresolved strategic action before next strategic decision')
            model._upper_pending[index] = dict(
                observation=row(obs, index), action=int(actions[index]),
                mask=np.array(masks[index], copy=True), value=float(values[index]),
                log_prob=float(log_probs[index]), reward=0., env=index,
                episode=model._upper_episode_ids[index],
            )
        pending = model._upper_pending[index]
        censored = bool(infos[index].get('TimeLimit.truncated') or infos[index].get('dead_screen'))
        if pending is not None:
            stats = infos[index].get('run_stats')
            if stats is None:
                raise ValueError('joint training requires whole-run outcome statistics')
            pending['reward'] += strategic_reward(
                round(float(obs['act'][index, 0]) * encode.ACT_SCALE),
                round(float(obs['floor'][index, 0]) * encode.FLOOR_SCALE),
                stats, done=bool(dones[index]), censored=censored)
            if dones[index] or int(new_obs['phase'][index, 0]) != COMBAT_PHASE:
                if not censored:
                    pending['terminal'] = bool(dones[index])
                    pending['next_observation'] = None if dones[index] else row(new_obs, index)
                    model._upper_completed.append(pending)
                else:
                    model._upper_censored += 1
                model._upper_pending[index] = None
        if dones[index]:
            model._upper_episode_ids[index] += 1
            if (getattr(model, 'nosl_is_az_combat_strategy', 'model') == 'small-playout'
                    and not censored and not infos[index].get('startup_stagger_hold')
                    and not infos[index].get('startup_occupancy_hold')):
                model.small_playout_games_local = getattr(model, 'small_playout_games_local', 0) + 1


def macro_advantages(records, bootstraps, gamma, gae_lambda):
    """Discount once per strategic action, never once per combat click."""
    advantages = np.zeros(len(records), dtype=np.float32)
    continuation = {}
    for i in range(len(records) - 1, -1, -1):
        record = records[i]
        key = (record['env'], record['episode'])
        future = 0. if record['terminal'] else float(bootstraps[i])
        tail = 0. if record['terminal'] else continuation.get(key, 0.)
        advantages[i] = record['reward'] + gamma * future - record['value'] + gamma * gae_lambda * tail
        continuation[key] = float(advantages[i])
    returns = advantages + np.array([r['value'] for r in records], dtype=np.float32)
    return advantages, returns


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


def update_upper(model, *, world_size=1, average=lambda values: values, synchronize=None):
    """Four full-batch PPO epochs by default, weighted by actual global row count.

    Microbatches only bound memory. Unequal/empty ranks use the same collective
    sequence, global advantage normalization and a single clip/Adam per epoch.
    """
    if not enabled(model):
        return {}
    records = model._upper_completed
    cfg = model.policy.core.upper_config
    count = len(records)
    bootstrap = np.zeros(count, dtype=np.float32)
    with torch.no_grad(), model.autocast_context():
        indices = [i for i, r in enumerate(records) if not r['terminal']]
        for start in range(0, len(indices), cfg.micro_batch_size):
            batch = indices[start:start + cfg.micro_batch_size]
            observations = observation_as_tensor({k: np.stack([records[i]['next_observation'][k] for i in batch])
                for k in records[batch[0]]['next_observation']}, model.device)
            bootstrap[batch] = model.policy.predict_values(observations).flatten().float().cpu().numpy()
    advantages, returns = macro_advantages(records, bootstrap, cfg.gamma, cfg.gae_lambda)
    stats = average(dict(count=float(count), advantage_sum=float(advantages.sum(dtype=np.float64)),
        advantage_square_sum=float(np.square(advantages.astype(np.float64)).sum()),
        censored=float(getattr(model, '_upper_censored', 0)),
        pending=float(sum(x is not None for x in model._upper_pending)),
        **reward_monitor.counts(model)))
    mean_count = stats['count']
    out = dict(samples=mean_count * world_size, updates=0., kl=0., policy_loss=0.,
        value_loss=0., entropy=0., censored=stats['censored'] * world_size,
        pending=stats['pending'] * world_size, **reward_monitor.metrics(stats, world_size))
    if not mean_count:
        return out
    mean = stats['advantage_sum'] / mean_count
    std = max(0., stats['advantage_square_sum'] / mean_count - mean * mean) ** .5
    out.update(advantage_mean=mean, advantage_std=std, grad_norm=0., parameter_l2_change=0.,
               clip_fraction=0., requested_updates=float(cfg.epochs), kl_stopped=0.)
    advantages = (advantages - mean) / (std + 1e-8)
    # Advantages must be computed in trajectory order. Only the execution of
    # the full-batch update is grouped: every sample keeps its target and weight.
    # This avoids repeatedly launching all phase-specific encoders per chunk.
    order = np.argsort([int(r['observation']['phase'][0]) for r in records], kind='stable')
    training_records = [records[int(i)] for i in order]
    advantages, returns = advantages[order], returns[order]
    batches = list(_training_batches(training_records, cfg.micro_batch_size))
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
            reduced = average(totals)
            out.update({k: v / mean_count for k, v in reduced.items()})
            out[f'policy_loss_epoch_{epoch + 1}'] = out['policy_loss']
            out[f'kl_epoch_{epoch + 1}'] = out['kl']
            if not all(np.isfinite(v) for v in out.values()):
                raise FloatingPointError('nonfinite strategic PPO statistics')
            if out['kl'] > 1.5 * cfg.target_kl:
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
