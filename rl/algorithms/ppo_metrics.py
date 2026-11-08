"""Sample-weighted diagnostics; never change PPO targets, gradients or RNG."""
import numpy as np
import torch

from env.core import vocab


UPDATE_METRICS = ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')


def value_moments(values, returns, phase):
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
    result = {}
    for name, mask in [('all', np.ones_like(combat)), ('combat', combat), ('upper', ~combat)]:
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
    if getattr(model, '_ppo_update_metric_sums', None) is None:
        model._ppo_update_metric_sums = weighted
        model._ppo_update_metric_count = count
    else:
        model._ppo_update_metric_sums.add_(weighted)
        model._ppo_update_metric_count += count


def update_moments(model):
    sums = getattr(model, '_ppo_update_metric_sums', None)
    values = np.zeros(len(UPDATE_METRICS)) if sums is None else sums.cpu().numpy()
    result = {'pure_rl/update/' + key + '_sum': float(value)
              for key, value in zip(UPDATE_METRICS, values, strict=True)}
    result['pure_rl/update/count'] = float(getattr(model, '_ppo_update_metric_count', 0))
    return result


def finalize_metrics(moments):
    """Derive ratios after cross-rank reduction. Missing data stays missing."""
    result = {}
    for name in ('all', 'combat', 'upper'):
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
    return result
