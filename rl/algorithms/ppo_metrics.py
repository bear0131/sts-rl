"""Sample-weighted diagnostics; never change PPO targets, gradients or RNG."""
import numpy as np
import torch

from env.core import vocab


UPDATE_METRICS = ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')
ENTROPY_LAYERS = ('upper', 'lower')


def count_search_action_overrides(logits, legal, executed):
    """Count real search actions outside the frozen policy's tied best set."""
    executed = np.asarray(executed, dtype=np.int64).reshape(-1)
    rows = np.arange(len(executed))
    if not np.asarray(legal)[rows, executed].all():
        raise ValueError('search execution diagnostic requires legal recorded actions')
    scores = np.where(legal, logits, -np.inf)
    return int((scores[rows, executed] < scores.max(axis=1)).sum())


def policy_entropy_moments(entropy, observations):
    """Actual masked policy entropy, including searched rows, excluding value anchors.

    These are update-time diagnostics, separate from the entropy regularizer.
    Forced actions remain real decisions with zero entropy. No policy forward,
    RNG operation or device-to-host synchronization is needed here.
    """
    if entropy is None or 'phase' not in observations:
        return None
    entropy = entropy.detach().reshape(-1).double()
    combat = observations['phase'].reshape(-1) == vocab.PHASE_VOCAB_INDEX['combat']
    anchor = observations.get('combat_exit_anchor')
    valid = (torch.ones_like(combat) if anchor is None else
             ~anchor.bool().reshape(len(entropy), -1).any(1))
    moments = []
    for layer in (~combat, combat):
        mask = valid & layer
        moments.extend((torch.where(mask, entropy, 0.).sum(), mask.sum()))
    return torch.stack(moments)


def value_moments(values, returns, phase, *, anchors=None):
    """Behavior V against the actual frozen lambda-return targets, before updates.

    Return additive moments, not rank-local means: combat/upper populations can
    differ across workers. Both sum and mean reductions preserve their ratios.
    """
    values, returns = (np.asarray(x, dtype=np.float64) for x in (values, returns))
    phase = np.asarray(phase)
    if values.shape != returns.shape or phase.shape != values.shape:
        raise ValueError('value diagnostic arrays must have the same shape')
    if not (np.isfinite(values).all() and np.isfinite(returns).all()):
        raise ValueError('value diagnostics require finite predictions and targets')
    combat = phase == vocab.PHASE_VOCAB_INDEX['combat']
    groups = [('all', np.ones_like(combat)), ('combat', combat), ('upper', ~combat)]
    if anchors is not None:
        anchors = np.asarray(anchors, dtype=bool)
        if anchors.shape != values.shape:
            raise ValueError('anchor diagnostic mask has the wrong shape')
        groups[1] = ('combat', combat & ~anchors)
        groups.append(('anchor', anchors))
    result = {}
    for name, mask in groups:
        target, error = returns[mask], (returns - values)[mask]
        prefix = f'pure_rl/value_pre/{name}/'
        result.update({prefix + key: float(value) for key, value in dict(
            count=target.size, target_sum=target.sum(), target_sq_sum=np.square(target).sum(),
            error_sum=error.sum(), error_sq_sum=np.square(error).sum()).items()})
    return result


def accumulate_update_metrics(model, components, count):
    """Only call after successful backward; failed OOM attempts do not count."""
    metrics = torch.stack([components.policy_loss, components.value_loss,
        -components.entropy_loss, components.approx_kl, components.clip_fraction]).detach().double()
    weighted = metrics * count
    entropy = getattr(components, 'policy_entropy_moments', None)
    weighted = torch.cat((weighted, weighted.new_zeros(4) if entropy is None else entropy))
    if getattr(model, '_ppo_update_metric_sums', None) is None:
        model._ppo_update_metric_sums = weighted
        model._ppo_update_metric_count = count
    else:
        model._ppo_update_metric_sums.add_(weighted)
        model._ppo_update_metric_count += count


def update_moments(model):
    sums = getattr(model, '_ppo_update_metric_sums', None)
    values = np.zeros(len(UPDATE_METRICS) + 4) if sums is None else sums.cpu().numpy()
    result = {'pure_rl/update/' + key + '_sum': float(value)
              for key, value in zip(UPDATE_METRICS, values[:len(UPDATE_METRICS)], strict=True)}
    result['pure_rl/update/count'] = float(getattr(model, '_ppo_update_metric_count', 0))
    for i, layer in enumerate(ENTROPY_LAYERS):
        prefix = 'pure_rl/update/entropy_' + layer
        result[prefix + '_sum'] = float(values[len(UPDATE_METRICS) + 2*i])
        result[prefix + '_count'] = float(values[len(UPDATE_METRICS) + 2*i + 1])
    return result


def finalize_metrics(moments, *, world_size=1):
    """Derive ratios after cross-rank reduction. Missing data stays missing."""
    result = {}
    for name in ('all', 'combat', 'upper', 'anchor'):
        p = f'pure_rl/value_pre/{name}/'
        count = moments.get(p + 'count', 0)
        if count <= 0:
            continue
        mse = moments[p + 'error_sq_sum'] / count
        error_mean = moments[p + 'error_sum'] / count
        variance = moments[p + 'target_sq_sum'] / count - (moments[p + 'target_sum'] / count) ** 2
        result[p + 'mse'] = mse
        result[p + 'bias'] = -error_mean
        if variance > 1e-12:
            result[p + 'explained_variance'] = 1 - max(0., mse - error_mean ** 2) / variance
    count = moments.get('pure_rl/update/count', 0)
    if count > 0:
        for key in UPDATE_METRICS:
            result['pure_rl/' + key] = moments['pure_rl/update/' + key + '_sum'] / count
    for layer in ENTROPY_LAYERS:
        prefix = 'pure_rl/update/entropy_' + layer
        if prefix + '_count' not in moments:
            continue
        count = moments[prefix + '_count']
        result['pure_rl/entropy_' + layer + '_samples_global'] = int(round(count * world_size))
        if count > 0:
            result['pure_rl/entropy_' + layer] = moments[prefix + '_sum'] / count
    prefix = 'deck_search/teacher_top1'
    if prefix + '_count' in moments:
        count = moments[prefix + '_count']
        result[prefix + '_samples_global'] = int(round(count * world_size))
        if count > 0:
            result[prefix] = moments[prefix + '_matches'] / count
    prefix = 'deck_search/action_override'
    if prefix + '_roots' in moments:
        count = moments[prefix + '_count']
        roots = moments[prefix + '_roots']
        result[prefix + '_count_global'] = int(round(count * world_size))
        result[prefix + '_roots_global'] = int(round(roots * world_size))
        if roots > 0:
            result[prefix + '_rate'] = count / roots
    return result
