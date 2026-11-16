"""Conservative paired-search policy target; real PPO returns stay unchanged."""
import math
from dataclasses import dataclass
import numpy as np
import torch

from env.search.action_selection import SEARCH_FILTER_VALUE_TOLERANCE
from rl.algorithms.shared_buffer import SharedRolloutBuffer

SEARCH_OBJECTIVES = ('prior-advantage-kl-v1', 'filtered-distillation-v1', 'turn-consensus-v1', 'turn-consensus-reverse-v1', 'turn-consensus-logit-v1')


@dataclass(frozen=True)
class SearchTrainingConfig:
    """One explicit supported hybrid configuration, separate from NOSL."""
    fraction: float = .25
    coefficient: float = .1
    normal_budget: int = 64
    boss_budget: int = 256
    max_depth: int = 200
    max_active: int = 256
    timeout_seconds: float = 120.
    target_std: float = .25
    seed: int = 20261111
    # Retained for reading saved configurations. Retired GPU execution is
    # rejected at collection entry; it must never silently become CPU/shadow.
    backend: str = 'native-cpu'
    inference_cache_bytes: int = 0
    cuda_artifacts: str = ''
    cuda_wave_capacity: int = 4096
    action_selection: str = 'shadow'
    objective: str = 'prior-advantage-kl-v1'
    value_mix: float = 0.

    def __post_init__(self):
        if not math.isfinite(self.value_mix) or not 0 <= self.value_mix <= 1:
            raise ValueError('chain value mixture weight must be in [0, 1]')
        if self.value_mix and (self.backend != 'full480'
                or self.objective != 'prior-advantage-kl-v1'
                or self.action_selection != 'search-filter-policy-rank-v1'):
            raise ValueError('chain value mixture requires original full480 search-filter supervision')
        if self.objective not in SEARCH_OBJECTIVES:
            raise ValueError('unknown search supervision objective')
        if self.objective in ('filtered-distillation-v1', 'turn-consensus-v1', 'turn-consensus-reverse-v1', 'turn-consensus-logit-v1') and (
                self.backend != 'full480' or self.action_selection not in ('search-filter-policy-rank-v1', 'turn-consensus-policy-rank-v1')):
            raise ValueError('filtered distillation requires full480 search-filter acting')
        if self.action_selection not in ('shadow', 'search-filter-policy-rank-v1', 'turn-consensus-policy-rank-v1'):
            raise ValueError('unknown search action selection contract')
        if self.action_selection == 'turn-consensus-policy-rank-v1' and self.objective not in ('turn-consensus-v1', 'turn-consensus-reverse-v1', 'turn-consensus-logit-v1'):
            raise ValueError('turn consensus acting requires the matching teacher evidence')
        if self.inference_cache_bytes < 0:
            raise ValueError('search inference cache capacity must be nonnegative')
        if self.backend not in ('native-cpu', 'cuda-batched-native', 'cuda-resident', 'full480'):
            raise ValueError('unknown shadow search backend')
        if self.backend == 'cuda-resident' and (not self.cuda_artifacts or self.cuda_wave_capacity < 1):
            raise ValueError('resident search requires explicit artifacts and a positive wave capacity')
        if not (0 < self.fraction <= 1 and 0 <= self.coefficient <= 1):
            raise ValueError('invalid shadow search fraction/coefficient')
        if min(self.normal_budget, self.boss_budget, self.max_depth, self.max_active) < 1:
            raise ValueError('shadow search limits must be positive')
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('shadow search needs a finite positive timeout')
        if not math.isfinite(self.target_std) or self.target_std < 0:
            raise ValueError('invalid frozen search target scale')


class SearchRolloutBuffer(SharedRolloutBuffer):
    """Search targets are side data, never observations or PPO trajectories."""
    def reset(self):
        super().reset()
        self.search_targets = {}
        self.search_actions = {}


    def add(self, obs, action, reward, episode_start, value, log_prob, *, action_masks=None):
        if self.search_actions:
            log_prob = log_prob.detach().cpu()
        for env_index in range(self.n_envs):
            chosen = self.search_actions.get((self.pos, env_index))
            if chosen is not None:
                actual = int(np.asarray(action).reshape(-1)[env_index])
                if actual != chosen[0] or not np.isclose(float(log_prob[env_index]), chosen[1], atol=1e-6):
                    raise ValueError('rollout must record the executed search action and behavior probability')
        return super().add(obs, action, reward, episode_start, value, log_prob, action_masks=action_masks)

    def add_search_target(self, env_index, log_target, *, position=None):
        if ((position is None and self.full) or not 0 <= env_index < self.n_envs):
            raise ValueError('search target must refer to the current real pre-action row')
        position = self.pos if position is None else int(position)
        if not (0 <= position < self.buffer_size and (self.full or position <= self.pos)):
            raise ValueError('search target must refer to a captured real pre-action row')
        if np.shape(log_target) != (self.mask_dims,):
            raise ValueError('search target action contract mismatch')
        if (position, env_index) in self.search_targets:
            raise ValueError('search target already supplied for this real row')
        self.search_targets[(position, env_index)] = np.array(log_target, dtype=np.float32, copy=True)

    def _get_samples(self, batch_inds, env=None, device_storage=None):
        result = super()._get_samples(batch_inds, env, device_storage=device_storage)
        if not self.search_targets:
            return result
        target = np.zeros((len(batch_inds), self.mask_dims), np.float32)
        valid = np.zeros(len(batch_inds), np.float32)
        executed = np.zeros(len(batch_inds), np.float32)
        for row, index in enumerate(batch_inds):
            key = (int(index % self.buffer_size), int(index // self.buffer_size))
            if key in self.search_targets:
                target[row] = self.search_targets[key]
                valid[row] = 1
            executed[row] = float(key in self.search_actions)
        result.observations['_search_target'] = self.to_torch(target, copy=False)
        result.observations['_search_valid'] = self.to_torch(valid, copy=False)
        result.observations['_search_executed'] = self.to_torch(executed, copy=False)
        return result


def prepare_search_batch(model, batch, average):
    """Normalize over all searched rows in this ORIGINAL optimizer batch.

    Gradient synchronization averages ranks, so N / mean_rank(searched)
    converts the ordinary per-row loss mean to the global searched-row mean.
    Physical packing and differing search coverage cannot change the objective.
    """
    config = getattr(model, 'deck_search_config', None)
    if config is None:
        return
    valid = batch.observations.get('_search_valid')
    count = 0. if valid is None else float(valid.sum())
    moments = dict(search_count=count, rows=float(len(batch.advantages)))
    if average is not None:
        moments = average(moments)
    model._deck_search_loss_normalization = moments['rows'] / (moments['search_count'] or 1.)


def auxiliary_loss(model, observations, logits, legal):
    target = observations.get('_search_target')
    valid = observations.get('_search_valid')
    if target is None or valid is None:
        return logits.sum() * 0
    config = SearchTrainingConfig(**model.deck_search_config)
    selected = valid.bool()
    # An unsearched row's zero target is not a probability distribution.
    if not selected.any():
        return logits.sum() * 0
    loss = search_kl(logits[selected], target[selected], legal[selected], objective=config.objective)
    return (config.coefficient * model._deck_search_loss_normalization
            * loss.sum() / len(valid))


def improved_prior(logits, values, legal, worlds, target_std, *, objective='prior-advantage-kl-v1', teacher_keep=None):
    """Frozen legal-action target; the legacy objective remains the default."""
    if objective not in SEARCH_OBJECTIVES:
        raise ValueError('unknown search supervision objective')
    if not math.isfinite(target_std) or target_std < 0:
        raise ValueError('target scale must be a finite nonnegative standard deviation')
    if not legal.any(-1).all() or not torch.isfinite(worlds).all() or (worlds <= 0).any():
        raise ValueError('teacher requires legal actions and complete positive world counts')
    with torch.no_grad():
        old = logits.float().masked_fill(~legal, -torch.inf).log_softmax(-1)
        q = values.float().masked_fill(~legal, 0)
        if not torch.isfinite(q).all() or not torch.isfinite(old[legal]).all():
            raise FloatingPointError('nonfinite search target')
        if objective in ('turn-consensus-v1', 'turn-consensus-reverse-v1', 'turn-consensus-logit-v1'):
            if (teacher_keep is None or teacher_keep.dtype != torch.bool or teacher_keep.shape != legal.shape
                    or (teacher_keep & ~legal).any() or not teacher_keep.any(-1).all()):
                raise ValueError('turn consensus requires aligned nonempty legal support')
            return old.masked_fill(~teacher_keep, -torch.inf).log_softmax(-1)
        baseline = (old.exp() * q).sum(-1, keepdim=True)
        shrink = worlds.float().reshape(-1, 1) / (worlds.float().reshape(-1, 1) + 4)
        if objective == 'filtered-distillation-v1':
            tolerance = SEARCH_FILTER_VALUE_TOLERANCE
            best = q.masked_fill(~legal, -torch.inf).max(-1, keepdim=True).values
            eligible = legal & (q >= best - tolerance)
            # Preserve preference within the same set used by real acting.
            filtered = old.masked_fill(~eligible, -torch.inf).log_softmax(-1)
            rejected_mass = old.exp().masked_fill(eligible | ~legal, 0)
            excess = (best - q - tolerance).clamp_min(0)
            gap = (rejected_mass * excess).sum(-1, keepdim=True) / rejected_mass.sum(
                -1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
            # Smooth evidence weight, not a statistical confidence interval.
            weight = (shrink * gap / (gap + tolerance)).clamp_max(1 - torch.finfo(torch.float32).eps)
            return torch.logaddexp(old + torch.log1p(-weight), filtered + weight.log())
        delta = shrink * ((q - baseline) / max(.02, .25 * target_std)).clamp(-2, 2)
        return (old + delta).log_softmax(-1)


def search_kl(logits, log_target, legal, *, objective='prior-advantage-kl-v1'):
    if objective not in SEARCH_OBJECTIVES:
        raise ValueError('unknown search supervision objective')
    current = logits.float().masked_fill(~legal, -torch.inf).log_softmax(-1)
    if objective in ('turn-consensus-v1', 'turn-consensus-reverse-v1', 'turn-consensus-logit-v1'):
        support = legal & torch.isfinite(log_target)
        if not support.any(-1).all():
            raise ValueError('turn consensus target must have nonempty support')
        if objective in ('turn-consensus-reverse-v1', 'turn-consensus-logit-v1'):
            # Enforce total probability on supported actions, then retain the
            # legacy reverse-KL geometry within the unresolved group. A hard
            # rejected action never creates an infinite reverse-KL target.
            log_mass = current.masked_fill(~support, -torch.inf).logsumexp(-1, keepdim=True)
            conditional = current - log_mass
            difference = conditional.masked_fill(~support, 0) - log_target.detach().masked_fill(~support, 0)
            if objective == 'turn-consensus-logit-v1':
                # Equal weight per unresolved relative logit resists a global
                # temperature change even when the old probabilities are tiny.
                count = support.sum(-1, keepdim=True)
                centered = (difference - difference.sum(-1, keepdim=True) / count).masked_fill(~support, 0)
                return -log_mass.squeeze(-1) + (centered.square().sum(-1, keepdim=True) / count).squeeze(-1)
            probability = conditional.masked_fill(~support, -torch.inf).exp()
            return -log_mass.squeeze(-1) + (probability * difference).sum(-1)
        difference = log_target.detach().masked_fill(~support, 0) - current.masked_fill(~support, 0)
        return (log_target.detach().exp() * difference).sum(-1)
    # Mask before subtraction: -inf - -inf must not poison gradients.
    difference = current.masked_fill(~legal, 0) - log_target.detach().masked_fill(~legal, 0)
    if objective == 'filtered-distillation-v1':
        return (-log_target.detach().exp() * difference).sum(-1)
    return (current.exp() * difference).sum(-1)
