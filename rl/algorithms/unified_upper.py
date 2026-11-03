"""Complete-game critic and frozen-GAE PPO actor with independent updates."""
import math
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


def update_unified_upper(model, *, rank=0, world_size=1, average=lambda x: x, synchronize=None):
    records, core = model._upper_completed, model.policy.core
    cfg, count = core.upper_config, len(records)
    returns = complete_returns(records)
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
               requested_effective_batch_size=float(cfg.effective_batch_size))
    # Pack once on CPU. Missing inactive-screen fields are zeros, never old state.
    keys = set().union(*(r['observation'] for r in records)) if records else set()
    observations = {k: np.stack([r['observation'].get(k, np.zeros(model.observation_space[k].shape))
                                for r in records]) for k in keys}
    actions = np.asarray([r['action'] for r in records], dtype=np.int64)
    old_logp = np.asarray([r['log_prob'] for r in records], dtype=np.float32)
    masks = np.stack([r['mask'] for r in records]) if records else np.zeros((0, model.action_space.n), bool)
    batches = max(1, math.ceil(total / cfg.effective_batch_size))
    out.update(requested_updates=float(cfg.epochs * batches),
               effective_batch_size_global=total / batches)
    model.policy.set_training_mode(True)
    serial = 0
    with train_phase(model, upper=True):
        for name, network, epochs in [('actor', core.upper_actor, cfg.epochs), ('critic', core.upper_critic, cfg.critic_epochs)]:
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
    out['grad_norm'] = out['actor_grad_norm']
    out['parameter_l2_change'] = out['actor_parameter_l2_change']
    model._upper_completed = []
    for key, value in out.items():
        model.logger.record('upper/'+key, value)
    return out
