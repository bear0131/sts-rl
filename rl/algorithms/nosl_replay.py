"""Compact FIFO replay storage for NOSL AlphaZero combat targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import prod

import numpy as np
from gymnasium import spaces


def _compact_integer_dtype(space: spaces.Box) -> np.dtype:
    low = int(np.min(space.low))
    high = int(np.max(space.high))
    if low >= 0:
        if high <= np.iinfo(np.uint8).max:
            return np.dtype(np.uint8)
        if high <= np.iinfo(np.uint16).max:
            return np.dtype(np.uint16)
        if high <= np.iinfo(np.uint32).max:
            return np.dtype(np.uint32)
        return np.dtype(np.uint64)
    if np.iinfo(np.int8).min <= low and high <= np.iinfo(np.int8).max:
        return np.dtype(np.int8)
    if np.iinfo(np.int16).min <= low and high <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    if np.iinfo(np.int32).min <= low and high <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def _storage_dtype(space: spaces.Box) -> np.dtype:
    if np.issubdtype(space.dtype, np.integer):
        return _compact_integer_dtype(space)
    if np.issubdtype(space.dtype, np.bool_):
        return np.dtype(bool)
    # Combat observations are normalized features. Float16 cuts the replay
    # footprint roughly in half and is expanded by the policy before use.
    return np.dtype(np.float16)


class NOSLISAZReplayBuffer:
    """Fixed-capacity ring buffer containing labelled combat search roots.

    Arrays are allocated lazily on first insertion. This matters for inference
    and checkpoint-inspection processes, which load the model but never train.
    """

    def __init__(
        self,
        *,
        capacity: int,
        observation_space: spaces.Dict,
        observation_keys: Iterable[str],
        action_dim: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        if action_dim < 1:
            raise ValueError("replay action dimension must be positive")
        keys = tuple(sorted(set(observation_keys) & set(observation_space.spaces)))
        if not keys:
            raise ValueError("replay requires at least one observation field")

        self.capacity = int(capacity)
        self.action_dim = int(action_dim)
        self.observation_keys = keys
        self._observation_spaces = {
            key: observation_space[key] for key in self.observation_keys
        }
        self.observations: dict[str, np.ndarray] | None = None
        self.action_masks: np.ndarray | None = None
        self.policy_targets: np.ndarray | None = None
        self.policy_weights: np.ndarray | None = None
        self.value_targets: np.ndarray | None = None
        self.sample_ids: np.ndarray | None = None
        self.position = 0
        self.size = 0
        self.total_added = 0

    @property
    def allocated_bytes(self) -> int:
        arrays = [] if self.observations is None else list(self.observations.values())
        arrays.extend(
            array
            for array in (
                self.action_masks,
                self.policy_targets,
                self.policy_weights,
                self.value_targets,
                self.sample_ids,
            )
            if array is not None
        )
        return sum(array.nbytes for array in arrays)

    def _allocate(self) -> None:
        if self.observations is not None:
            return
        self.observations = {
            key: np.empty(
                (self.capacity, *space.shape), dtype=_storage_dtype(space)
            )
            for key, space in self._observation_spaces.items()
        }
        self.action_masks = np.empty(
            (self.capacity, self.action_dim), dtype=bool
        )
        self.policy_targets = np.empty(
            (self.capacity, self.action_dim), dtype=np.float16
        )
        self.policy_weights = np.empty(self.capacity, dtype=np.float16)
        self.value_targets = np.empty(self.capacity, dtype=np.float32)
        self.sample_ids = np.empty(self.capacity, dtype=np.uint64)

    def _write(self, destination: slice, source: slice, batch: Mapping) -> None:
        assert self.observations is not None
        assert self.action_masks is not None
        assert self.policy_targets is not None
        assert self.policy_weights is not None
        assert self.value_targets is not None
        assert self.sample_ids is not None
        for key in self.observation_keys:
            self.observations[key][destination] = batch["observations"][key][source]
        self.action_masks[destination] = batch["action_masks"][source]
        self.policy_targets[destination] = batch["policy_targets"][source]
        self.policy_weights[destination] = batch["policy_weights"][source]
        self.value_targets[destination] = batch["value_targets"][source]
        self.sample_ids[destination] = batch["sample_ids"][source]

    def add_samples(self, samples: Sequence[Mapping]) -> int:
        """Append labelled samples and return the number accepted."""
        if not samples:
            return 0
        labelled = [
            sample
            for sample in samples
            if np.isfinite(float(sample.get("value_target", np.nan)))
        ]
        if not labelled:
            return 0

        observations = {
            key: np.stack([sample["observation"][key] for sample in labelled])
            for key in self.observation_keys
        }
        action_masks = np.stack([sample["action_mask"] for sample in labelled])
        policy_targets = np.stack(
            [sample["policy_target"] for sample in labelled]
        )
        policy_weights = np.asarray(
            [sample.get("policy_weight", 1.0) for sample in labelled],
            dtype=np.float32,
        )
        value_targets = np.asarray(
            [sample["value_target"] for sample in labelled], dtype=np.float32
        )
        count = len(labelled)
        if action_masks.shape != (count, self.action_dim):
            raise ValueError(
                f"invalid replay mask shape {action_masks.shape}; "
                f"expected {(count, self.action_dim)}"
            )
        if policy_targets.shape != (count, self.action_dim):
            raise ValueError(
                f"invalid replay policy shape {policy_targets.shape}; "
                f"expected {(count, self.action_dim)}"
            )
        if policy_weights.shape != (count,) or not np.all(
            np.isfinite(policy_weights) & (policy_weights >= 0.0)
        ):
            raise ValueError(
                "replay policy weights must be finite, non-negative scalars"
            )
        for key, space in self._observation_spaces.items():
            expected = (count, *space.shape)
            if observations[key].shape != expected:
                raise ValueError(
                    f"invalid replay observation {key!r} shape "
                    f"{observations[key].shape}; expected {expected}"
                )

        start_id = self.total_added
        self.total_added += count
        sample_ids = np.arange(start_id, start_id + count, dtype=np.uint64)
        if count > self.capacity:
            offset = count - self.capacity
            observations = {key: value[offset:] for key, value in observations.items()}
            action_masks = action_masks[offset:]
            policy_targets = policy_targets[offset:]
            policy_weights = policy_weights[offset:]
            value_targets = value_targets[offset:]
            sample_ids = sample_ids[offset:]
            count = self.capacity

        self._allocate()
        batch = {
            "observations": observations,
            "action_masks": action_masks,
            "policy_targets": policy_targets,
            "policy_weights": policy_weights,
            "value_targets": value_targets,
            "sample_ids": sample_ids,
        }
        first = min(count, self.capacity - self.position)
        self._write(slice(self.position, self.position + first), slice(0, first), batch)
        remaining = count - first
        if remaining:
            self._write(slice(0, remaining), slice(first, count), batch)
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        return len(labelled)

    def sample_indices(
        self,
        count: int,
        rng: np.random.Generator | None = None,
        *,
        fresh_start_id: int | None = None,
        max_old_fraction: float = 1.0,
    ) -> np.ndarray:
        """Sample replay slots, optionally reserving space for fresh roots.

        The ring buffer is already a FIFO sliding window.  Uniform sampling
        from that window can nevertheless make the newest collection almost
        disappear once the buffer is full.  ``max_old_fraction`` stratifies
        the batch by the monotonically increasing sample id.  When the fresh
        partition is smaller than its required quota, every fresh root is
        included once before sampling additional fresh entries with
        replacement.  This preserves the requested batch size and guarantees
        the configured current-collection fraction.
        """
        if count < 1:
            raise ValueError("replay sample count must be positive")
        if self.size < 1:
            raise ValueError("cannot sample an empty replay buffer")
        if not np.isfinite(max_old_fraction) or not 0.0 <= max_old_fraction <= 1.0:
            raise ValueError("max_old_fraction must be finite and in [0, 1]")
        count = min(int(count), self.size)
        chooser = np.random if rng is None else rng
        if max_old_fraction >= 1.0:
            return chooser.choice(self.size, size=count, replace=False)
        if fresh_start_id is None:
            raise ValueError(
                "fresh_start_id is required when replay freshness is protected"
            )
        assert self.sample_ids is not None
        slots = np.arange(self.size, dtype=np.int64)
        fresh_mask = self.sample_ids[:self.size] >= int(fresh_start_id)
        fresh_slots = slots[fresh_mask]
        old_slots = slots[~fresh_mask]
        if fresh_slots.size == 0:
            raise ValueError(
                "cannot satisfy replay freshness protection without fresh roots"
            )

        max_old_count = min(
            int(old_slots.size),
            int(np.floor(count * float(max_old_fraction) + 1.0e-12)),
        )
        fresh_count = count - max_old_count

        old_selected = chooser.choice(
            old_slots, size=max_old_count, replace=False
        ) if max_old_count else np.empty(0, dtype=np.int64)
        if fresh_count <= fresh_slots.size:
            fresh_selected = chooser.choice(
                fresh_slots, size=fresh_count, replace=False
            )
        else:
            # Include the complete newest collection once.  Only the quota
            # remainder is oversampled, avoiding accidental omission of a
            # fresh root when that partition is small.
            fresh_selected = np.concatenate((
                chooser.permutation(fresh_slots),
                chooser.choice(
                    fresh_slots,
                    size=fresh_count - fresh_slots.size,
                    replace=True,
                ),
            ))
        selected = np.concatenate((old_selected, fresh_selected)).astype(
            np.int64, copy=False
        )
        chooser.shuffle(selected)
        return selected

    def get_batch(self, indices: np.ndarray) -> dict[str, object]:
        if self.observations is None:
            raise ValueError("cannot read an unallocated replay buffer")
        assert self.action_masks is not None
        assert self.policy_targets is not None
        assert self.policy_weights is not None
        assert self.value_targets is not None
        assert self.sample_ids is not None
        indices = np.asarray(indices, dtype=np.int64)
        return {
            "observations": {
                key: values[indices] for key, values in self.observations.items()
            },
            "action_masks": self.action_masks[indices],
            "policy_targets": self.policy_targets[indices],
            "policy_weights": self.policy_weights[indices],
            "value_targets": self.value_targets[indices],
            "sample_ids": self.sample_ids[indices],
        }

    def estimated_capacity_bytes(self) -> int:
        obs_bytes = sum(
            np.dtype(_storage_dtype(space)).itemsize * prod(space.shape)
            for space in self._observation_spaces.values()
        )
        per_sample = (
            obs_bytes
            + self.action_dim * np.dtype(bool).itemsize
            + self.action_dim * np.dtype(np.float16).itemsize
            + np.dtype(np.float16).itemsize
            + np.dtype(np.float32).itemsize
            + np.dtype(np.uint64).itemsize
        )
        return self.capacity * per_sample
