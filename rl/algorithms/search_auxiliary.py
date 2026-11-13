"""Conservative paired-search policy target; real PPO returns stay unchanged."""
import math
from dataclasses import dataclass
import numpy as np
import torch

from rl.algorithms.shared_buffer import SharedRolloutBuffer


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
    backend: str = 'native-cpu'
    inference_cache_bytes: int = 0
    cuda_artifacts: str = ''
    cuda_wave_capacity: int = 4096

    def __post_init__(self):
        if self.inference_cache_bytes < 0:
            raise ValueError('search inference cache capacity must be nonnegative')
        if self.backend not in ('native-cpu', 'cuda-batched-native', 'cuda-resident'):
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
        self._target_table = None

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
        self._target_table = None

    def _prepare_target_table(self):
        if self._target_table is None and self.search_targets:
            # Build once after the collector drains, before prefetch starts.
            # Row zero represents missing teachers; targets remain side data.
            lookup = np.zeros(self.buffer_size * self.n_envs, dtype=np.int64)
            indices = np.fromiter((env * self.buffer_size + pos
                                   for pos, env in self.search_targets), dtype=np.int64)
            lookup[indices] = np.arange(1, len(indices) + 1)
            targets = np.empty((len(indices) + 1, self.mask_dims), dtype=np.float32)
            targets[0] = 0
            np.stack(tuple(self.search_targets.values()), out=targets[1:])
            self._target_table = lookup, targets

    def get(self, batch_size=None, *, prefetch=False):
        self._prepare_target_table()
        yield from super().get(batch_size, prefetch=prefetch)

    def _get_samples(self, batch_inds, env=None, device_storage=None):
        result = super()._get_samples(batch_inds, env, device_storage=device_storage)
        if not self.search_targets:
            return result
        self._prepare_target_table()
        lookup, targets = self._target_table
        rows = lookup[batch_inds]
        target = targets[rows]
        valid = (rows != 0).astype(np.float32)
        result.observations['_search_target'] = self.to_torch(target, copy=False)
        result.observations['_search_valid'] = self.to_torch(valid, copy=False)
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


def auxiliary_loss(model, observations, logits, legal, *, selected_rows=None):
    target = observations.get('_search_target')
    valid = observations.get('_search_valid')
    if target is None or valid is None:
        return logits.sum() * 0
    config = SearchTrainingConfig(**model.deck_search_config)
    # An unsearched row's zero target is not a probability distribution.
    if selected_rows is None:
        selected_rows = valid.bool().nonzero().flatten()
    if not len(selected_rows):
        return logits.sum() * 0
    loss = search_kl(logits[selected_rows], target[selected_rows], legal[selected_rows])
    return (config.coefficient * model._deck_search_loss_normalization
            * loss.sum() / len(valid))


def improved_prior(logits, values, legal, worlds, target_std):
    """Frozen q ∝ old_pi exp(shrunk clipped advantage), on legal atoms only."""
    if not math.isfinite(target_std) or target_std < 0:
        raise ValueError('target scale must be a finite nonnegative standard deviation')
    if not legal.any(-1).all() or (worlds <= 0).any():
        raise ValueError('teacher requires legal actions and complete positive world counts')
    with torch.no_grad():
        old = logits.float().masked_fill(~legal, -torch.inf).log_softmax(-1)
        q = values.float().masked_fill(~legal, 0)
        if not torch.isfinite(q).all() or not torch.isfinite(old[legal]).all():
            raise FloatingPointError('nonfinite search target')
        baseline = (old.exp() * q).sum(-1, keepdim=True)
        shrink = worlds.float().reshape(-1, 1) / (worlds.float().reshape(-1, 1) + 4)
        delta = shrink * ((q - baseline) / max(.02, .25 * target_std)).clamp(-2, 2)
        return (old + delta).log_softmax(-1)


def search_kl(logits, log_target, legal):
    current = logits.float().masked_fill(~legal, -torch.inf).log_softmax(-1)
    # Mask before subtraction: -inf - -inf must not poison gradients.
    difference = current.masked_fill(~legal, 0) - log_target.detach().masked_fill(~legal, 0)
    return (current.exp() * difference).sum(-1)
