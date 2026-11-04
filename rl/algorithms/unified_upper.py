"""Complete-game critic and frozen-GAE PPO actor with independent updates."""
import math
import random
import time
import numpy as np
import torch

from rl.algorithms.upper_ppo import macro_advantages, bootstrap_indices, train_phase
from rl.algorithms.updates import optimizer_step
from rl.training import reward_monitor, campfire_monitor


def complete_returns(records):
    """Never silently bootstrap or train on an unfinished/censored episode."""
    if bootstrap_indices(records):
        raise ValueError('MC upper requires complete natural episodes')
    returns = np.zeros(len(records), dtype=np.float64)
    tails = {}
    for i in range(len(records) - 1, -1, -1):
        r = records[i]
        key = (r['env'], r['episode'])
        returns[i] = r['reward'] + (0. if r['terminal'] else tails[key])
        tails[key] = returns[i]
    return returns


def measure_critic_mse(model, observations, targets, *, average=lambda x: x):
    """Frozen full-cohort forward passes; never train on or mutate the inputs.

    Targets use the learner's float32 labels, with float64 squared-error sums.
    All ranks participate, including empty ranks. This measures training data,
    not held-out generalization. The returned time includes reduction and CUDA
    synchronization; callers retain it in their end-to-end round time.
    """
    network, cfg = model.policy.core.upper_critic, model.policy.core.upper_config
    if not targets:
        raise ValueError('critic measurement requires named targets')
    count = len(next(iter(targets.values())))
    if any(len(value) != count for value in targets.values()):
        raise ValueError('critic measurement targets have different sample counts')
    device = torch.device(model.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    if devices:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    modes = [(module, module.training) for module in network.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            sums = torch.zeros(len(targets), dtype=torch.float64, device=device)
            for start in range(0, count, cfg.micro_batch_size):
                stop = min(start + cfg.micro_batch_size, count)
                obs = {key: torch.as_tensor(value[start:stop], device=device)
                       for key, value in observations.items()}
                prediction = network(obs).flatten().double()
                for index, value in enumerate(targets.values()):
                    label = torch.as_tensor(value[start:stop], dtype=torch.float32, device=device).double()
                    sums[index] += (prediction - label).square().sum()
            reduced = average(dict(count=float(count), **{
                key: float(value) for key, value in zip(targets, sums.cpu().tolist())}))
            if not all(np.isfinite(value) for value in reduced.values()):
                raise FloatingPointError('nonfinite frozen critic measurement')
            result = {key: reduced[key] / reduced['count'] if reduced['count'] else None for key in targets}
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if devices:
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def measure_actor_surrogate(model, observations, actions, masks, old_logp, advantages, *, average=lambda x: x):
    """Full-cohort PPO surrogate before updates, using collection-time logp.

    This is a relative objective, not a prediction error: normalized advantages
    and an unchanged policy make its mean approximately zero. Preserve RNG,
    module modes, gradients and optimizer state; empty ranks still reduce sums.
    """
    network, cfg = model.policy.core.upper_actor, model.policy.core.upper_config
    count = len(actions)
    if any(len(v) != count for v in (masks, old_logp, advantages)):
        raise ValueError('actor measurement inputs have different sample counts')
    device = torch.device(model.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    if devices:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    modes = [(module, module.training) for module in network.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    names = ('loss', 'kl', 'entropy', 'nll', 'ratio_abs_error')
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            sums = torch.zeros(len(names), dtype=torch.float64, device=device)
            for start in range(0, count, cfg.micro_batch_size):
                stop = min(start + cfg.micro_batch_size, count)
                obs = {k: torch.as_tensor(v[start:stop], device=device) for k, v in observations.items()}
                logits = network(obs).masked_fill(~torch.as_tensor(masks[start:stop], device=device), -1e9)
                logp_all = logits.log_softmax(-1)
                logp = logp_all.gather(1, torch.as_tensor(actions[start:stop, None], device=device)).flatten()
                delta = logp - torch.as_tensor(old_logp[start:stop], dtype=torch.float32, device=device)
                ratio = delta.exp()
                adv = torch.as_tensor(advantages[start:stop], dtype=torch.float32, device=device)
                metrics = (-torch.minimum(adv * ratio, adv * ratio.clamp(1-cfg.clip, 1+cfg.clip)),
                           (ratio-1)-delta, -(logp_all.exp()*logp_all).sum(-1), -logp, (ratio-1).abs())
                sums += torch.stack([v.double().sum() for v in metrics])
            reduced = average(dict(count=float(count), **dict(zip(names, sums.cpu().tolist()))))
            if not all(np.isfinite(v) for v in reduced.values()):
                raise FloatingPointError('nonfinite frozen actor measurement')
            result = {k: reduced[k] / reduced['count'] if reduced['count'] else None for k in names}
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if devices:
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def update_unified_upper(model, *, rank=0, world_size=1, average=lambda x: x, synchronize=None):
    records, core = model._upper_completed, model.policy.core
    cfg, count = core.upper_config, len(records)
    returns = complete_returns(records)
    mc_returns = returns
    advantages, lambda_returns = macro_advantages(records, np.zeros(count), 1., cfg.gae_lambda)
    # Both targets require natural complete games. Lambda labels use only
    # collection-time frozen values, never this batch's newly fitted critic.
    if cfg.critic_target == 'frozen-lambda':
        returns = lambda_returns
    stats = average(dict(count=float(count), total=float(advantages.sum()),
        square=float(np.square(advantages).sum()),
        **reward_monitor.counts(model), **campfire_monitor.counts(model)))
    total = int(round(stats['count'] * world_size))
    out = dict(samples=float(total), updates=0., critic_updates=0., samples_used=0., critic_samples_used=0.,
        requested_epochs=float(cfg.epochs), epochs_completed=0., critic_epochs_completed=0.,
        kl_stopped=0., kl_early_stop_enabled=0., bootstrap_samples=0., pending=0., censored=0.,
        gae_precision_bits=64., target_precision_version=2.,
        critic_target_version=2. if cfg.critic_target == 'frozen-lambda' else 1.,
        **reward_monitor.metrics(stats, world_size), **campfire_monitor.metrics(stats, world_size))
    if not total:
        return out
    mean = stats['total'] / stats['count']
    std = max(0., stats['square'] / stats['count'] - mean * mean) ** .5
    advantages = (advantages - mean) / (std + 1e-8)
    out.update(advantage_mean=mean, advantage_std=std,
               actor_learning_rate=float(model.policy.upper_optimizer.param_groups[0]['lr']),
               critic_learning_rate=float(model.policy.upper_optimizer.param_groups[1]['lr']),
               requested_effective_batch_size=float(cfg.effective_batch_size),
               requested_critic_effective_batch_size=float(cfg.critic_effective_batch_size or cfg.effective_batch_size))
    # Pack once on CPU. Missing inactive-screen fields are zeros, never old state.
    keys = set().union(*(r['observation'] for r in records)) if records else set()
    observations = {k: np.stack([r['observation'].get(k, np.zeros(model.observation_space[k].shape))
                                for r in records]) for k in keys}
    actions = np.asarray([r['action'] for r in records], dtype=np.int64)
    old_logp = np.asarray([r['log_prob'] for r in records], dtype=np.float32)
    masks = np.stack([r['mask'] for r in records]) if records else np.zeros((0, model.action_space.n), bool)
    actor_batches = max(1, math.ceil(total / cfg.effective_batch_size))
    critic_batches = max(1, math.ceil(total / (cfg.critic_effective_batch_size or cfg.effective_batch_size)))
    out.update(requested_updates=float(cfg.epochs * actor_batches),
               requested_critic_updates=float(cfg.critic_epochs * critic_batches),
               effective_batch_size_global=total / actor_batches,
               critic_effective_batch_size_global=total / critic_batches)
    measurement_seconds = 0.
    actor_measured, actor_measurement_seconds = measure_actor_surrogate(
        model, observations, actions, masks, old_logp, advantages, average=average)
    out.update({f'policy_{key}_pre_update': value for key, value in actor_measured.items()})
    out.update(policy_measurement_pre_seconds_rank=actor_measurement_seconds,
               policy_measurement_samples=float(total), policy_measurement_version=1.)
    if cfg.measure_value_loss:
        measured, measurement_seconds = measure_critic_mse(model, observations,
            dict(mc=mc_returns, target=returns), average=average)
        out.update(value_mc_mse_pre_update=measured['mc'],
                   value_target_mse_pre_update=measured['target'],
                   value_measurement_pre_seconds_rank=measurement_seconds,
                   value_measurement_samples=float(total), value_measurement_version=1.)
    model.policy.set_training_mode(True)
    serial = 0
    with train_phase(model, upper=True):
        for name, network, epochs, batches in [
                ('actor', core.upper_actor, cfg.epochs, actor_batches),
                ('critic', core.upper_critic, cfg.critic_epochs, critic_batches)]:
            active = {id(p) for p in network.parameters()}
            for p in core.upper_parameters():
                p.requires_grad_(id(p) in active)
                p.grad = None
            norms = []
            before = [p.detach().clone() for p in network.parameters()]
            started = time.perf_counter()
            for epoch in range(epochs):
                totals = np.zeros(5, dtype=np.float64)
                # Balanced partitions use every sample exactly once, including
                # unequal/empty ranks; all ranks execute identical collectives.
                for selected in np.array_split(np.random.permutation(count), batches):
                    size = average({'n': float(len(selected))})['n']
                    if not size:
                        continue
                    model.policy.upper_optimizer.zero_grad(set_to_none=True)
                    local = torch.zeros(4, device=model.device)
                    for start in range(0, len(selected), cfg.micro_batch_size):
                        ix = selected[start:start + cfg.micro_batch_size]
                        obs = {k: torch.as_tensor(v[ix], device=model.device) for k, v in observations.items()}
                        if name == 'actor':
                            logits = network(obs).masked_fill(~torch.as_tensor(masks[ix], device=model.device), -1e9)
                            logp_all = logits.log_softmax(-1)
                            logp = logp_all.gather(1, torch.as_tensor(actions[ix, None], device=model.device)).flatten()
                            delta = logp - torch.as_tensor(old_logp[ix], device=model.device)
                            ratio = delta.exp()
                            adv = torch.as_tensor(advantages[ix], device=model.device, dtype=torch.float32)
                            policy_loss = -torch.minimum(adv * ratio, adv * ratio.clamp(1-cfg.clip, 1+cfg.clip)).mean()
                            entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                            loss = policy_loss - cfg.entropy_coef * entropy
                            metrics = torch.stack((policy_loss, ((ratio-1)-delta).mean(), entropy,
                                                   ((ratio-1).abs() > cfg.clip).float().mean()))
                        else:
                            prediction = network(obs).flatten()
                            loss = (prediction - torch.as_tensor(returns[ix], device=model.device, dtype=torch.float32)).square().mean()
                            metrics = torch.stack((loss, loss.detach()*0, loss.detach()*0, loss.detach()*0))
                        (loss * (len(ix) / size)).backward()
                        local += metrics.detach() * len(ix)
                    if synchronize is not None:
                        synchronize(serial)
                    serial += 1
                    norms.append(optimizer_step(model, optimizer=model.policy.upper_optimizer, max_grad_norm=cfg.max_grad_norm))
                    reduced = average({str(i): float(v) for i, v in enumerate(local.cpu().tolist())})
                    if not all(np.isfinite(v) for v in reduced.values()):
                        raise FloatingPointError('nonfinite unified upper statistics')
                    totals += np.array([*reduced.values(), size])
                    out['updates' if name == 'actor' else 'critic_updates'] += 1
                    out['samples_used' if name == 'actor' else 'critic_samples_used'] += size * world_size
                if totals[-1]:
                    prefix = 'policy_loss' if name == 'actor' else 'value_loss'
                    out[f'{prefix}_epoch_{epoch+1}'] = totals[0] / totals[-1]
                    out[prefix] = out[f'{prefix}_epoch_{epoch+1}']
                    if name == 'actor':
                        for i, key in enumerate(('kl', 'entropy', 'clip_fraction'), 1):
                            out[key] = totals[i] / totals[-1]
                            out[f'{key}_epoch_{epoch+1}'] = out[key]
                out['epochs_completed' if name == 'actor' else 'critic_epochs_completed'] += 1
            out[name+'_seconds'] = time.perf_counter() - started
            out[name+'_grad_norm'] = float(torch.stack(norms).mean()) if norms else 0.
            out[name+'_parameter_l2_change'] = float(torch.stack([(p.detach()-q).square().sum()
                for p, q in zip(network.parameters(), before)]).sum().sqrt())
    if cfg.measure_value_loss:
        measured, seconds = measure_critic_mse(model, observations,
            dict(mc=mc_returns, target=returns), average=average)
        out.update(value_mc_mse_post_update=measured['mc'],
                   value_target_mse_post_update=measured['target'],
                   value_measurement_post_seconds_rank=seconds,
                   value_measurement_seconds_rank=measurement_seconds + seconds)
    out['grad_norm'] = out['actor_grad_norm']
    out['parameter_l2_change'] = out['actor_parameter_l2_change']
    model._upper_completed = []
    for key, value in out.items():
        model.logger.record('upper/'+key, value)
    return out
