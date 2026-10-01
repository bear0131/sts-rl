"""Compact FIFO replay storage for NOSL AlphaZero combat targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from math import prod

import numpy as np
from gymnasium import spaces
from env.core.combat_value import VALUE_CONTRACT
from env.core import native_actions as C


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


def _storage_dtype(space: spaces.Box, *, key: str | None = None) -> np.dtype:
    if np.issubdtype(space.dtype, np.integer):
        return _compact_integer_dtype(space)
    if np.issubdtype(space.dtype, np.bool_):
        return np.dtype(bool)
    # Player scalars and buffs can exceed float16 while a battle is running.
    if key in ("player_scalars", "player_buffs"):
        return np.dtype(np.float32)
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
        value_contract: str = VALUE_CONTRACT,
        supervision_contract: str = 'scalar-hp-v1',
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
        self.policy_contract = ({C.JOINT_SIZE: C.JOINT_CONTRACT, C.AUTOREGRESSIVE_SIZE: C.AUTOREGRESSIVE_CONTRACT}.get(action_dim, 'legacy-thin'))
        if self.policy_contract in C.NEURAL_CONTRACTS and C.policy_contract(observation_space) != self.policy_contract:
            raise ValueError('replay action/observation contracts differ')
        self.group_offset = C.policy_offset(self.policy_contract) + 1 if self.policy_contract in C.NEURAL_CONTRACTS else 0
        self.group_width = 60 if self.policy_contract in C.NEURAL_CONTRACTS else self.action_dim
        self.joint_rows: list | None = None
        # Rebuilt after collection append; never persisted in checkpoints.
        self._joint_dense_masks: np.ndarray | None = None
        self._joint_dense_targets: np.ndarray | None = None
        self.value_contract = str(value_contract)
        from rl.algorithms.combat_distribution import CONTRACT
        if supervision_contract not in ('scalar-hp-v1', CONTRACT):
            raise ValueError('unsupported replay supervision contract')
        self.supervision_contract = supervision_contract
        self.combat_targets = None
        self.observation_keys = keys
        self._observation_spaces = {
            key: observation_space[key] for key in self.observation_keys
        }
        self.observations: dict[str, np.ndarray] | None = None
        self.action_masks: np.ndarray | None = None
        self.policy_targets: np.ndarray | None = None
        self.action_groups: np.ndarray | None = None
        self.policy_weights: np.ndarray | None = None
        self.value_targets: np.ndarray | None = None
        self.sample_ids: np.ndarray | None = None
        self.position = 0
        self.size = 0
        self.total_added = 0

    @property
    def allocated_bytes(self) -> int:
        arrays = [] if self.observations is None else list(self.observations.values())
        if self.combat_targets is not None:
            arrays.append(self.combat_targets)
        arrays.extend(
            array
            for array in (
                self.action_masks,
                self.policy_targets,
                self._joint_dense_masks,
                self._joint_dense_targets,
                self.action_groups,
                self.policy_weights,
                self.value_targets,
                self.sample_ids,
            )
            if array is not None
        )
        sparse_bytes = sum(ids.nbytes + probs.nbytes for row in (self.joint_rows or [])
                           if row is not None for ids, probs in [row])
        return sum(array.nbytes for array in arrays) + sparse_bytes

    def state_dict(self):
        """Persist occupied slots only; never serialize uninitialized capacity."""
        names = ('action_masks', 'policy_targets', 'action_groups', 'policy_weights', 'value_targets', 'sample_ids')
        if self.supervision_contract != 'scalar-hp-v1':
            names += ('combat_targets',)
        joint = None
        if self.joint_rows is not None:
            rows = self.joint_rows[:self.size]
            joint = dict(offsets=np.cumsum([0] + [len(row[0]) for row in rows], dtype=np.int64),
                         ids=np.concatenate([row[0] for row in rows]) if rows else np.empty(0, np.uint16),
                         probabilities=np.concatenate([row[1] for row in rows]) if rows else np.empty(0, np.float32))
        from rl.algorithms.action_groups import CONTRACT
        return dict(version=4, action_group_contract=CONTRACT,
            supervision_contract=self.supervision_contract,
            value_contract=self.value_contract, policy_contract=self.policy_contract,
            joint=joint,
            capacity=self.capacity, action_dim=self.action_dim,
            observation_keys=self.observation_keys, position=self.position, size=self.size,
            total_added=self.total_added,
            observations=None if self.observations is None else
                {k:v[:self.size].copy() for k,v in self.observations.items()},
            arrays={k:None if getattr(self,k) is None else getattr(self,k)[:self.size].copy() for k in names})

    def load_state_dict(self, state):
        from rl.algorithms.action_groups import CONTRACT, validate_groups
        if state.get('version') in (2,3) and self.policy_contract == 'legacy-thin':
            # Legacy actions never merged; their historical grouping is known.
            state = dict(state, version=4, action_group_contract=CONTRACT,
                arrays=dict(state['arrays'], action_groups=None if state['observations'] is None else
                    np.tile(np.arange(self.action_dim,dtype=np.int16),(state['size'],1))))
        if state.get("action_group_contract") != CONTRACT:
            raise ValueError("replay action group contract mismatch; collect fresh grouped targets")
        if state.get('supervision_contract','scalar-hp-v1') != self.supervision_contract:
            raise ValueError('replay supervision contract mismatch; collect fresh outcome labels')
        if state.get('policy_contract', 'legacy-thin') != self.policy_contract:
            raise ValueError('replay policy contract mismatch; collect fresh complete-action targets')
        if state.get('value_contract') != self.value_contract:
            raise ValueError(
                'replay value contract mismatch; migrate weights and collect '
                'new replay instead of mixing pre/post-combat HP targets')
        if (state['version'] != 4 or state['capacity'] != self.capacity
                or state['action_dim'] != self.action_dim
                or tuple(state['observation_keys']) != self.observation_keys):
            raise ValueError('replay checkpoint geometry does not match runtime')
        size, position = int(state['size']), int(state['position'])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError('invalid replay cursor')
        if size < self.capacity and position != size:
            raise ValueError('invalid partially filled replay cursor')
        if state['observations'] is not None:
            if self.supervision_contract != 'scalar-hp-v1':
                from rl.algorithms.combat_distribution import validate_targets
                if state['arrays'].get('combat_targets') is None:
                    raise ValueError('missing combat distribution targets')
                validate_targets(state['arrays']['combat_targets'],state['arrays']['value_targets'], action_dim=self.action_dim)
            if state['arrays'].get('action_groups') is None:
                raise ValueError('missing action group metadata')
            self._allocate()
            for key, target in self.observations.items():
                source = state['observations'][key]
                if source.shape != target[:size].shape:
                    raise ValueError('replay observation schema mismatch: ' + key)
                if not np.can_cast(source.dtype, target.dtype, casting='safe'):
                    raise ValueError('replay observation schema mismatch: ' + key)
                if np.issubdtype(source.dtype, np.floating) and not np.isfinite(source).all():
                    raise ValueError('replay observation has nonfinite values: ' + key)
                target[:size] = source
            for key, source in state['arrays'].items():
                target = getattr(self, key)
                if target is None and source is None:
                    continue
                if target is None or source is None:
                    raise ValueError('replay target schema mismatch: ' + key)
                if source.shape != target[:size].shape or source.dtype != target.dtype:
                    raise ValueError('replay target schema mismatch: ' + key)
                target[:size] = source
            if self.joint_rows is not None:
                joint = state.get('joint')
                if joint is None:
                    raise ValueError('joint replay has no complete-action targets')
                offsets, ids, probs = (joint[k] for k in ('offsets', 'ids', 'probabilities'))
                if (offsets.dtype != np.int64 or offsets.shape != (size + 1,)
                        or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                        or ids.ndim != 1 or ids.dtype != np.uint16
                        or probs.shape != ids.shape or probs.dtype != np.float32
                        or offsets[-1] != len(ids) or np.any(ids >= self.action_dim)
                        or not np.all(np.isfinite(probs) & (probs >= 0))):
                    raise ValueError('invalid sparse joint replay schema')
                for row in range(size):
                    lo, hi = offsets[row:row+2]
                    row_ids = ids[lo:hi]
                    if len(np.unique(row_ids)) != len(row_ids):
                        raise ValueError('duplicate complete-action id in replay')
                    self.joint_rows[row] = (row_ids.copy(), probs[lo:hi].copy())
                masks = np.zeros((size,self.action_dim),dtype=bool)
                for row,(ids,_) in enumerate(self.joint_rows[:size]):
                    masks[row,ids] = True
            else:
                masks = self.action_masks[:size]
            validate_groups(self._expand_groups(self.action_groups[:size]), masks)
        elif size:
            raise ValueError('nonempty replay checkpoint has no observations')
        self.size, self.position, self.total_added = size, position, int(state['total_added'])
        self._joint_dense_masks = None
        self._joint_dense_targets = None

    def _allocate(self) -> None:
        if self.observations is not None:
            return
        self.observations = {
            key: np.empty(
                (self.capacity, *space.shape), dtype=_storage_dtype(space, key=key)
            )
            for key, space in self._observation_spaces.items()
            if not (self.policy_contract in C.NEURAL_CONTRACTS and key == 'combat_legal')
        }
        if self.policy_contract in C.NEURAL_CONTRACTS:
            self.joint_rows = [None] * self.capacity
        else:
            self.action_masks = np.empty((self.capacity, self.action_dim), dtype=bool)
            self.policy_targets = np.empty((self.capacity, self.action_dim), dtype=np.float16)
        self.action_groups = np.empty((self.capacity,self.group_width),dtype=np.int16)
        self.policy_weights = np.empty(self.capacity, dtype=np.float16)
        self.value_targets = np.empty(self.capacity, dtype=np.float32)
        if self.supervision_contract != 'scalar-hp-v1':
            from rl.algorithms.combat_distribution import TARGET_WIDTH
            self.combat_targets = np.empty((self.capacity,TARGET_WIDTH),dtype=np.float32)
        self.sample_ids = np.empty(self.capacity, dtype=np.uint64)

    def _write(self, destination: slice, source: slice, batch: Mapping) -> None:
        assert self.observations is not None
        assert self.policy_weights is not None
        assert self.value_targets is not None
        assert self.sample_ids is not None
        for key in self.observations:
            self.observations[key][destination] = batch["observations"][key][source]
        if self.joint_rows is not None:
            for dest, src in zip(range(destination.start, destination.stop), range(source.start, source.stop)):
                ids = np.flatnonzero(batch['action_masks'][src]).astype(np.uint16)
                self.joint_rows[dest] = (ids, batch['policy_targets'][src, ids].astype(np.float32))
        else:
            self.action_masks[destination] = batch["action_masks"][source]
            self.policy_targets[destination] = batch["policy_targets"][source]
        self.action_groups[destination] = batch["action_groups"][source, self.group_offset:self.group_offset+self.group_width]
        self.policy_weights[destination] = batch["policy_weights"][source]
        self.value_targets[destination] = batch["value_targets"][source]
        if self.combat_targets is not None:
            self.combat_targets[destination] = batch['combat_targets'][source]
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
        combat_targets = None
        if self.supervision_contract != 'scalar-hp-v1':
            from rl.algorithms.combat_distribution import validate_targets
            combat_targets = np.stack([sample['combat_targets'] for sample in labelled]).astype(np.float32)
            validate_targets(combat_targets,value_targets, action_dim=self.action_dim)
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
        if self.policy_contract in C.NEURAL_CONTRACTS:
            if not np.all(np.isfinite(policy_targets) & (policy_targets >= 0)):
                raise ValueError('joint policy probabilities must be finite and non-negative')
            if np.any(policy_targets[~action_masks.astype(bool)] != 0):
                raise ValueError('joint policy has probability on an illegal action')
            if 'combat_legal' in observations and not np.array_equal(
                    observations['combat_legal'][policy_weights > 0],
                    action_masks[policy_weights > 0, C.policy_offset(self.policy_contract):]):
                raise ValueError('joint replay observation and action masks disagree')
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
            value = observations[key]
            storage_dtype = _storage_dtype(space, key=key)
            if np.issubdtype(value.dtype, np.floating):
                if not np.isfinite(value).all():
                    raise ValueError(f"replay observation {key!r} contains nonfinite values")
                if storage_dtype == np.dtype(np.float16) and np.any(np.abs(value) > np.finfo(np.float16).max):
                    raise ValueError(f"replay observation {key!r} exceeds float16 storage range")

        from rl.algorithms.action_groups import sample_groups
        action_groups = sample_groups(labelled, self.action_dim)
        outside = action_groups.copy()
        outside[:,self.group_offset:self.group_offset+self.group_width] = np.arange(self.group_offset,self.group_offset+self.group_width)
        if np.any(outside != np.arange(self.action_dim)):
            raise ValueError('equivalent-plays-v1 only merges complete card plays')
        start_id = self.total_added
        self.total_added += count
        sample_ids = np.arange(start_id, start_id + count, dtype=np.uint64)
        if count > self.capacity:
            offset = count - self.capacity
            observations = {key: value[offset:] for key, value in observations.items()}
            action_masks = action_masks[offset:]
            action_groups = action_groups[offset:]
            policy_targets = policy_targets[offset:]
            policy_weights = policy_weights[offset:]
            value_targets = value_targets[offset:]
            if combat_targets is not None:
                combat_targets = combat_targets[offset:]
            sample_ids = sample_ids[offset:]
            count = self.capacity

        self._allocate()
        batch = {
            "observations": observations,
            "action_masks": action_masks,
            "action_groups": action_groups,
            "policy_targets": policy_targets,
            "policy_weights": policy_weights,
            "value_targets": value_targets,
            "sample_ids": sample_ids,
        }
        if combat_targets is not None:
            batch['combat_targets'] = combat_targets
        first = min(count, self.capacity - self.position)
        self._write(slice(self.position, self.position + first), slice(0, first), batch)
        remaining = count - first
        if remaining:
            self._write(slice(0, remaining), slice(first, count), batch)
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self._joint_dense_masks = None
        self._joint_dense_targets = None
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

    def get_batch(self, indices: np.ndarray, *, workers: int = 1) -> dict[str, object]:
        """Gather identical ordered rows; bounded threads copy independent fields.

        The caller must not append to replay until this synchronous call returns.
        Sampling and its RNG stay with the caller. Small/CPU batches use one worker.
        """
        if not 1 <= workers <= 4:
            raise ValueError("replay gather workers must be in [1, 4]")
        if self.observations is None:
            raise ValueError("cannot read an unallocated replay buffer")
        assert self.policy_weights is not None
        assert self.value_targets is not None
        assert self.sample_ids is not None
        indices = np.asarray(indices, dtype=np.int64)
        if self.joint_rows is not None and self._joint_dense_masks is None:
            masks = np.zeros((self.size, self.action_dim), dtype=bool)
            targets = np.zeros((self.size, self.action_dim), dtype=np.float32)
            for row, (ids, probabilities) in enumerate(self.joint_rows[:self.size]):
                masks[row, ids] = True
                targets[row, ids] = probabilities
            self._joint_dense_masks = masks
            self._joint_dense_targets = targets
        fields = list(self.observations.items()) + [
            (name, getattr(self, name)) for name in (
                "action_masks", "action_groups", "policy_targets", "policy_weights", "value_targets", "sample_ids", "combat_targets"
            ) if getattr(self, name) is not None
        ]
        if self.joint_rows is not None:
            fields.extend((
                ('action_masks', self._joint_dense_masks),
                ('policy_targets', self._joint_dense_targets),
            ))
        def gather(field):
            return field[1][indices]
        if workers == 1:
            values = list(map(gather, fields))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                values = list(pool.map(gather, fields))
        count = len(self.observations)
        batch = dict(
            observations=dict(zip((key for key, _ in fields[:count]), values[:count])),
            **dict(zip((key for key, _ in fields[count:]), values[count:])),
        )
        if self.joint_rows is not None:
            if 'combat_legal' in self.observation_keys:
                batch['observations']['combat_legal'] = batch['action_masks'][:, C.policy_offset(self.policy_contract):]
        batch['action_groups'] = self._expand_groups(batch['action_groups'])
        return batch

    def _expand_groups(self, compact):
        groups = np.tile(np.arange(self.action_dim,dtype=np.int16),(len(compact),1))
        groups[:,self.group_offset:self.group_offset+self.group_width] = compact
        return groups

    def estimated_capacity_bytes(self) -> int:
        obs_bytes = sum(
            np.dtype(_storage_dtype(space, key=key)).itemsize * prod(space.shape)
            for key, space in self._observation_spaces.items()
            if not (self.policy_contract in C.NEURAL_CONTRACTS and key == 'combat_legal')
        )
        # Joint storage is ragged; report a conservative all-actions bound.
        policy_bytes = self.action_dim * (6 if self.policy_contract in C.NEURAL_CONTRACTS else 3)
        per_sample = (
            obs_bytes
            + policy_bytes + 2*self.group_width
            + np.dtype(np.float16).itemsize
            + np.dtype(np.float32).itemsize
            + np.dtype(np.uint64).itemsize
        )
        if self.supervision_contract != 'scalar-hp-v1':
            from rl.algorithms.combat_distribution import TARGET_WIDTH
            per_sample += 4*TARGET_WIDTH
        return self.capacity * per_sample
