"""PPO statistical objective, shared by local and synchronized updates."""
from dataclasses import dataclass
import numpy as np
import torch as th
import torch.nn.functional as F
from env.core import vocab as V


def masked_categorical_statistics(logits, actions, masks, *, return_logits=False):
    """SB3 masked categorical log probability and entropy on the learner device."""
    original = logits - logits.logsumexp(dim=-1, keepdim=True)
    allowed = masks.bool().reshape_as(logits)
    masked = th.where(allowed, original, -1e8)
    log_probs = masked - masked.logsumexp(dim=-1, keepdim=True)
    indices = actions.long().reshape(-1, 1)
    selected = log_probs.gather(1, indices.clamp(0, logits.shape[1] - 1)).flatten()
    entropy = -th.where(allowed, log_probs * log_probs.softmax(dim=-1), 0.).sum(dim=-1)
    # SB3 validates the original distribution before applying the mask. Carry
    # invalid rows to the packed learner's collective finite vote before Adam.
    valid = (~original.isnan().any() & ~log_probs.isnan().any()
             & (indices >= 0).all() & (indices < logits.shape[1]).all())
    result = selected, entropy, valid
    return (*result, log_probs) if return_logits else result

@dataclass(frozen=True)
class PPOLossComponents:
    policy_loss: th.Tensor
    value_loss: th.Tensor
    entropy_loss: th.Tensor
    approx_kl: th.Tensor
    clip_fraction: th.Tensor
    valid_samples: int
    combat_samples: int
    overworld_samples: int
    combat_value_loss: th.Tensor | None
    overworld_value_loss: th.Tensor | None


def _phase_balanced_mean(
    values: th.Tensor,
    *,
    valid: th.Tensor,
    combat: th.Tensor,
) -> th.Tensor:
    means = []
    for group in (valid & combat, valid & ~combat):
        if bool(group.any()):
            means.append(values[group].mean())
    if not means:
        raise RuntimeError("PPO minibatch has no valid hierarchical samples")
    return th.stack(means).mean()


def normalize_ppo_advantages(advantages, group_size=None):
    """Normalize the original statistical groups before any model routing."""
    if group_size is not None and advantages.numel() > group_size:
        if group_size < 2 or advantages.numel() % group_size:
            raise ValueError('PPO batch must contain complete advantage normalization groups')
        groups = advantages.reshape(-1, group_size)
        return ((groups - groups.mean(dim=1, keepdim=True)) /
                (groups.std(dim=1, keepdim=True) + 1e-8)).reshape_as(advantages)
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def configure_advantage_normalization(model, mode=None):
    """Keep legacy groups by default; whole-rollout statistics need pure PPO."""
    mode = mode or getattr(model, 'ppo_advantage_normalization', 'group256')
    if mode not in ('group256', 'rollout'):
        raise ValueError('PPO advantage normalization must be group256 or rollout')
    if mode == 'rollout' and (
            getattr(model, 'nosl_supervision_contract', None) != 'shared-wholerun-ppo-room-v1'
            or model.hierarchical_combat_hp or model.nosl_is_az
            or not model.normalize_advantage):
        raise ValueError('rollout advantage normalization requires normalized shared whole-run PPO')
    model.ppo_advantage_normalization = mode
    model._ppo_advantage_stats = None
    return mode


def rollout_advantage_moments(advantages, *, average=None, world_size=1):
    """Two-pass, count-weighted global sample variance; no trajectory reweighting.

    Local accumulation is float64. The supplied collective may use float32
    transport, so center before the second pass instead of subtracting E[x]^2.
    Every rank participates even if a rank contains invalid samples.
    """
    if world_size < 1 or (world_size > 1 and average is None):
        raise ValueError('global rollout normalization requires a collective')
    average = average or (lambda values: values)
    values = np.asarray(advantages, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    first = average(dict(count=float(values.size), invalid=float((~finite).sum()),
                         total=float(np.where(finite, values, 0).sum())))
    count = first['count'] * world_size
    if first['invalid'] or count < 2:
        raise ValueError('rollout advantages must contain at least two finite samples')
    mean = first['total'] / first['count']
    second = average(dict(centered_sum=float(np.square(values - mean).sum())))
    variance = second['centered_sum'] / (first['count'] - 1 / world_size)
    if not np.isfinite(mean) or not np.isfinite(variance) or variance < 0:
        raise ValueError('invalid global advantage moments')
    return dict(count=int(round(count)), mean=float(mean), std=float(np.sqrt(variance)))


def prepare_advantage_normalization(model, *, average=None, world_size=1):
    """Freeze one statistic set for all epochs/microbatches of this collection."""
    if getattr(model, 'ppo_advantage_normalization', 'group256') != 'rollout':
        return
    buffer = model.rollout_buffer
    if not buffer.full:
        raise ValueError('advantage statistics require a complete rollout')
    stats = rollout_advantage_moments(buffer.advantages, average=average, world_size=world_size)
    model._ppo_advantage_stats = dict(stats, buffer_id=id(buffer),
        generation=buffer.normalization_generation, timesteps=model.num_timesteps)
    for key, value in stats.items():
        model.logger.record('pure_rl/advantage_normalization/' + key, value)


def model_normalized_advantages(model, advantages):
    if getattr(model, 'ppo_advantage_normalization', 'group256') == 'rollout':
        stats = getattr(model, '_ppo_advantage_stats', None)
        buffer = model.rollout_buffer
        if (stats is None or stats['buffer_id'] != id(buffer)
                or stats['generation'] != buffer.normalization_generation
                or stats['timesteps'] != model.num_timesteps or not buffer.full):
            raise RuntimeError('prepare global advantage statistics for this collection first')
        return (advantages - stats['mean']) / (stats['std'] + 1e-8)
    return normalize_ppo_advantages(advantages, getattr(model.policy, 'ppo_advantage_group_size', None))


def ppo_loss_components(
    *,
    hierarchical: bool,
    normalize_advantage: bool,
    observations,
    advantages: th.Tensor,
    returns: th.Tensor,
    old_values: th.Tensor,
    old_log_prob: th.Tensor,
    values: th.Tensor,
    log_prob: th.Tensor,
    entropy: th.Tensor | None,
    clip_range: float,
    clip_range_vf: float | None,
    skip_combat: bool = False,
    normalization_group_size: int | None = None,
) -> PPOLossComponents:
    """Compute the shared native/distributed PPO objective.

    Hierarchical rollouts use NaN to mark an overworld macro transition whose
    combat option did not finish before the rollout boundary. Those samples are
    excluded. Combat and overworld means are then weighted equally so the much
    larger number of combat clicks cannot drown out strategic decisions.
    """
    if clip_range_vf is None:
        values_pred = values
    else:
        values_pred = old_values + th.clamp(
            values - old_values,
            -clip_range_vf,
            clip_range_vf,
        )

    ratio = th.exp(log_prob - old_log_prob)
    clipped_ratio = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    entropy_per_sample = log_prob if entropy is None else -entropy
    kl_per_sample = (th.exp(log_prob - old_log_prob) - 1) - (
        log_prob - old_log_prob
    )
    clipped_per_sample = (th.abs(ratio - 1) > clip_range).float()

    if not hierarchical:
        normalized_advantages = advantages
        if normalize_advantage:
            normalized_advantages = normalize_ppo_advantages(advantages, normalization_group_size)
        policy_loss = -th.min(
            normalized_advantages * ratio,
            normalized_advantages * clipped_ratio,
        ).mean()
        value_loss = F.mse_loss(returns, values_pred)
        return PPOLossComponents(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_per_sample.mean(),
            approx_kl=kl_per_sample.mean(),
            clip_fraction=clipped_per_sample.mean(),
            valid_samples=int(advantages.numel()),
            combat_samples=0,
            overworld_samples=0,
            combat_value_loss=None,
            overworld_value_loss=None,
        )

    phase = observations["phase"].long().reshape(-1)
    combat = phase == V.PHASE_VOCAB_INDEX["combat"]
    valid = th.isfinite(advantages) & th.isfinite(returns)
    if skip_combat:
        valid = valid & ~combat
    if not bool(valid.any()):
        zero = values.sum() * 0.0
        return PPOLossComponents(
            policy_loss=zero,
            value_loss=zero,
            entropy_loss=zero,
            approx_kl=zero.detach(),
            clip_fraction=zero.detach(),
            valid_samples=0,
            combat_samples=0,
            overworld_samples=0,
            combat_value_loss=None,
            overworld_value_loss=None,
        )
    normalized_advantages = th.where(valid, advantages, th.zeros_like(advantages))
    if normalize_advantage:
        normalized_advantages = normalized_advantages.clone()
        for group in (valid & combat, valid & ~combat):
            count = int(group.sum().item())
            if count > 1:
                selected = normalized_advantages[group]
                normalized_advantages[group] = (
                    selected - selected.mean()
                ) / (selected.std(unbiased=False) + 1e-8)

    policy_per_sample = -th.min(
        normalized_advantages * ratio,
        normalized_advantages * clipped_ratio,
    )
    safe_returns = th.where(valid, returns, values_pred.detach())
    value_per_sample = (safe_returns - values_pred).square()
    policy_loss = _phase_balanced_mean(
        policy_per_sample, valid=valid, combat=combat
    )
    value_loss = _phase_balanced_mean(
        value_per_sample, valid=valid, combat=combat
    )
    entropy_loss = _phase_balanced_mean(
        entropy_per_sample, valid=valid, combat=combat
    )

    combat_valid = valid & combat
    overworld_valid = valid & ~combat
    return PPOLossComponents(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        approx_kl=kl_per_sample[valid].mean(),
        clip_fraction=clipped_per_sample[valid].mean(),
        valid_samples=int(valid.sum().item()),
        combat_samples=int(combat_valid.sum().item()),
        overworld_samples=int(overworld_valid.sum().item()),
        combat_value_loss=(
            value_per_sample[combat_valid].mean()
            if bool(combat_valid.any()) else None
        ),
        overworld_value_loss=(
            value_per_sample[overworld_valid].mean()
            if bool(overworld_valid.any()) else None
        ),
    )
