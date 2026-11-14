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
    # Retained for reading saved configurations. Retired GPU execution is
    # rejected at collection entry; it must never silently become CPU/shadow.
    backend: str = 'native-cpu'
    inference_cache_bytes: int = 0
    cuda_artifacts: str = ''
    cuda_wave_capacity: int = 4096
    action_selection: str = 'shadow'

    def __post_init__(self):
        if self.action_selection not in ('shadow', 'search-filter-policy-rank-v1'):
            raise ValueError('unknown search action selection contract')
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
        self.search_actions = {}
        self._row_positions = None


    def add(self, obs, action, reward, episode_start, value, log_prob, *, action_masks=None):
        if self._row_positions is not None:
            raise ValueError('cannot mix independent rows and full-vector rollout writes')
        if self.search_actions:
            log_prob = log_prob.detach().cpu()
        for env_index in range(self.n_envs):
            chosen = self.search_actions.get((self.pos, env_index))
            if chosen is not None:
                actual = int(np.asarray(action).reshape(-1)[env_index])
                if actual != chosen[0] or not np.isclose(float(log_prob[env_index]), chosen[1], atol=1e-6):
                    raise ValueError('rollout must record the executed search action and behavior probability')
        return super().add(obs, action, reward, episode_start, value, log_prob, action_masks=action_masks)

    @property
    def row_positions(self):
        """Next real timestep per environment, independent of completion order."""
        return (np.full(self.n_envs, self.pos, dtype=np.int64) if self._row_positions is None
                else self._row_positions.copy())

    def prefill_observation(self, obs):
        if self._row_positions is not None:
            raise ValueError('cannot prefill a full vector during independent row collection')
        return super().prefill_observation(obs)

    def add_rows(self, indices, obs, action, reward, episode_start, value, log_prob,
                 *, action_masks, search_targets=None, search_actions=None):
        """Consume ready rows from full480's existing subset step boundary.

        This is storage only: it cannot schedule, step, reset or search an env.
        The original (time, env) layout and learner's env-major shuffle remain
        unchanged. A failed packet is rejected before modifying any row.
        """
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                or len(np.unique(indices)) != len(indices)
                or (indices < 0).any() or (indices >= self.n_envs).any()):
            raise ValueError('independent rows require distinct valid environment indices')
        if self.action_dim != 1 or self.full or self._prefilled is not None:
            raise ValueError('independent rows require unfilled discrete rollout storage')
        if self._row_positions is None and self.pos:
            raise ValueError('cannot mix full-vector and independent rollout writes')
        positions = self.row_positions[indices]
        if (positions >= self.buffer_size).any():
            raise ValueError('environment already completed its fixed rollout length')
        count = len(indices)
        if set(obs) != set(self.observations):
            raise ValueError('independent rows must preserve every observation field')
        observed = {}
        for key, destination in self.observations.items():
            row = np.asarray(obs[key], dtype=destination.dtype)
            shape = (count, *destination.shape[2:])
            if row.shape == (count,) and shape == (count, 1):
                row = row[:, None]
            if row.shape != shape:
                raise ValueError('independent observation shape mismatch: ' + key)
            observed[key] = row
        actions = np.asarray(action).reshape(-1)
        masks = np.asarray(action_masks, dtype=bool)
        if (actions.shape != (count,) or actions.dtype.kind not in 'iu'
                or masks.shape != (count, self.mask_dims)
                or (actions < 0).any() or (actions >= self.mask_dims).any()
                or not masks[np.arange(count), actions].all()):
            raise ValueError('independent rows require the actual legal actions and masks')
        def host(data):
            array = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
            return np.asarray(array, dtype=np.float32).reshape(-1)
        rewards, starts, values, logps = map(host, (reward, episode_start, value, log_prob))
        if any(x.shape != (count,) or not np.isfinite(x).all() for x in (rewards, starts, values, logps)):
            raise ValueError('independent rows require finite aligned PPO fields')
        if not np.isin(starts, (0, 1)).all():
            raise ValueError('episode start must be a terminal/reset boundary flag')
        teachers, executed = search_targets or {}, search_actions or {}
        if not set(executed) <= set(teachers) <= set(indices):
            raise ValueError('search actions require teachers for the same real rows')
        pending_targets, pending_actions = {}, {}
        for offset, env_index in enumerate(indices):
            key = (int(positions[offset]), int(env_index))
            if key in self.search_targets or key in self.search_actions:
                raise ValueError('independent row search metadata was already supplied')
            if env_index in teachers:
                target = np.array(teachers[env_index], dtype=np.float32, copy=True)
                if target.shape != (self.mask_dims,) or not np.isfinite(target[masks[offset]]).all():
                    raise ValueError('invalid teacher for an independent real row')
                pending_targets[key] = target
            if env_index in executed:
                chosen, behavior_logp = executed[env_index]
                if (chosen != actions[offset] or not np.isfinite(behavior_logp)
                        or not np.isclose(behavior_logp, logps[offset], rtol=0., atol=1e-6)):
                    raise ValueError('independent row must record the executed search behavior')
                pending_actions[key] = (int(chosen), float(behavior_logp))
        if self._row_positions is None:
            self._row_positions = np.zeros(self.n_envs, dtype=np.int64)
        for key, rows in observed.items():
            self.observations[key][positions, indices] = rows
        self.actions[positions, indices, 0] = actions
        for name, rows in (('rewards', rewards), ('episode_starts', starts),
                           ('values', values), ('log_probs', logps), ('action_masks', masks)):
            getattr(self, name)[positions, indices] = rows
        self.search_targets.update(pending_targets)
        self.search_actions.update(pending_actions)
        self._row_positions[indices] += 1
        self.pos = int(self._row_positions.min())
        self.full = self.pos == self.buffer_size

    def add_search_target(self, env_index, log_target, *, position=None):
        if self._row_positions is not None:
            raise ValueError('independent search metadata must accompany add_rows')
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
    loss = search_kl(logits[selected], target[selected], legal[selected])
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
