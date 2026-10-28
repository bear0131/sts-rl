"""Independent, FP32 inference-only copy of an entity-joint upper policy."""
from copy import deepcopy
import hashlib

import numpy as np
import torch
from torch import nn

from env.core import encode, vocab as V
from rl.policies.model import (
    action_mask_for_policy, observation_for_policy, validate_reward_policy_contract,
)
from rl.policies.split_policy import (
    OVERWORLD_OBSERVATION_KEYS, OVERWORLD_PHASE_OBSERVATION_KEYS,
)


def _tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}\n'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenUpperTeacher(nn.Module):
    """Own every dependency; a parent's train()/autocast cannot train this copy.

    The caller owns the lower optimizer. This object never constructs or restores
    an optimizer. Check integrity at collection/save boundaries as well as after
    restore; those checks include buffers, not only trainable parameters.
    """

    def __init__(self, core, observation_space, *, source_sha256,
                 reward_policy_contract='checkpoint-v7-act3'):
        super().__init__()
        if not getattr(core, 'joint_upper', False):
            raise TypeError('teacher extraction requires an entity-joint core')
        if len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
            raise ValueError('teacher source must have a lowercase SHA256 identity')
        self.source_sha256 = source_sha256
        self.observation_space = deepcopy(observation_space)
        self.reward_policy_contract = reward_policy_contract
        validate_reward_policy_contract(self, reward_policy_contract)
        # Copy together to preserve any internal aliases, without aliases to lower.
        self.upper_tokenizer, self.overworld = deepcopy((core.upper_tokenizer, core.overworld))
        self._assert_fp32()
        self.train(False)
        self._expected_state_digest = _tensor_digest(self._persistent_tensors())
        self._expected_all_digest = _tensor_digest(self._all_tensors())

    def _persistent_tensors(self):
        return {name: tensor for name, tensor in self.state_dict().items()
                if isinstance(tensor, torch.Tensor)}

    def _all_tensors(self):
        return dict(list(self.named_parameters()) + list(self.named_buffers()))

    def _assert_fp32(self):
        for name, tensor in self._all_tensors().items():
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise ValueError(f'frozen upper requires FP32: {name}')

    def train(self, mode=True):
        # Module.train recurses into children even when invoked on the parent.
        super().train(False)
        super().requires_grad_(False)
        return self

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError('frozen upper cannot enable gradients')
        return super().requires_grad_(False)

    def _apply(self, fn, recurse=True):
        # Reject casts before they round any teacher weight; device moves are OK.
        probe = next(self.parameters()).new_zeros(1)
        if fn(probe).dtype != torch.float32:
            raise ValueError('move frozen upper by device only; keep its FP32 weights')
        return super()._apply(fn, recurse=recurse)

    def get_extra_state(self):
        return dict(schema='frozen-upper-actor-critic-v1', source_sha256=self.source_sha256,
                    reward_policy_contract=self.reward_policy_contract,
                    tensor_sha256=getattr(self, '_expected_state_digest', None))

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('frozen upper checkpoint identity mismatch')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
        # Parent.load_state_dict does not invoke the child's load_state_dict.
        if local_metadata.get('assign_to_params_buffers', False):
            raise ValueError('frozen upper restore must copy, not alias checkpoint storage')
        expected = self._persistent_tensors()
        incoming = {name: state_dict[prefix + name] for name in expected
                    if prefix + name in state_dict}
        if (incoming.keys() != expected.keys()
                or _tensor_digest(incoming) != self._expected_state_digest
                or state_dict.get(prefix + '_extra_state') != self.get_extra_state()):
            raise ValueError('frozen upper checkpoint tensors or identity changed')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)
        self.train(False)

    def assert_integrity(self):
        self._assert_fp32()
        if any(m.training for m in self.modules()) or any(p.requires_grad or p.grad is not None
                                                        for p in self.parameters()):
            raise RuntimeError('frozen upper training/gradient invariant violated')
        if _tensor_digest(self._all_tensors()) != self._expected_all_digest:
            raise RuntimeError('frozen upper parameter or buffer changed')
        return self._expected_state_digest

    def _validated_observations(self, observations):
        self._validate_inference_mode()
        observations = observation_for_policy(self, observations)
        batch_size = len(observations['phase'])
        for name, space in self.observation_space.spaces.items():
            if tuple(observations[name].shape) != (batch_size, *space.shape):
                raise ValueError(f'frozen upper observation shape changed: {name}')
        return observations

    def _validate_inference_mode(self):
        # Inspect the current registries in one pass. named_parameters() and
        # modules() separately walk the whole tree and build unused names on
        # every small search batch. Do not cache references: replacing a child
        # or registering a parameter must be noticed on the very next call.
        pending, seen = [self], set()
        while pending:
            module = pending.pop()
            if id(module) in seen:
                continue
            seen.add(id(module))
            if module.training:
                raise RuntimeError('frozen upper was made trainable')
            for parameter in module._parameters.values():
                if parameter is not None and parameter.requires_grad:
                    raise RuntimeError('frozen upper was made trainable')
            pending.extend(child for child in module._modules.values() if child is not None)

    @staticmethod
    def _trim_numpy_padding(batch):
        widths = []
        for key in ('deck_ids', 'relics'):
            active = np.flatnonzero(np.any(batch[key] != 0, axis=0))
            widths.append(int(active[-1])+1 if len(active) else 1)
        for key in ('deck_ids', 'deck_upgrades'):
            batch[key] = batch[key][:, :widths[0]]
        for key in ('relics', 'relic_counters'):
            batch[key] = batch[key][:, :widths[1]]
        return tuple(widths)

    def prepare_search_numpy(self, observations):
        """Stack only this phase's consumed fields from settled search rows.

        This internal transport validates every consumed field against the
        checkpoint. The ordinary forward/prepare_numpy full-schema contract
        remains strict; unrelated combat arrays never enter this transport.
        """
        self._validate_inference_mode()
        if not observations or any(np.asarray(row['phase']).shape != (1,) for row in observations):
            raise ValueError('upper search requires nonempty rows with scalar phase')
        phase = np.asarray([row['phase'][0] for row in observations], dtype=np.int64)
        if (phase == V.PHASE_VOCAB_INDEX['combat']).any():
            raise ValueError('upper teacher requires nonempty upper decision rows')
        groups = []
        for phase_id in np.unique(phase):
            selected = np.flatnonzero(phase == phase_id)
            keys = OVERWORLD_OBSERVATION_KEYS | OVERWORLD_PHASE_OBSERVATION_KEYS.get(int(phase_id), frozenset())
            batch = {}
            for key, space in self.observation_space.spaces.items():
                if key not in keys:
                    continue
                # These rows already own equally shaped encoded arrays. The
                # sequence constructor adds the batch axis directly, without
                # stack's per-row expanded views and concatenate dispatch.
                value = np.asarray([observations[index][key] for index in selected])
                if value.shape != (len(selected), *space.shape):
                    raise ValueError(f'frozen upper observation shape changed: {key}')
                batch[key] = value.astype(np.float32, copy=False) if np.issubdtype(value.dtype, np.floating) else value
            groups.append((selected, int(phase_id), self._trim_numpy_padding(batch), batch))
        return groups

    def prepare_numpy(self, observations):
        """Validate full checkpoint schema, then project/route on the CPU.

        The returned widths are computed before deduplication so cache hits or
        graph padding cannot silently change the encoder's physical token width.
        """
        observations = self._validated_observations(observations)
        phase = np.asarray(observations['phase']).astype(np.int64).reshape(-1)
        if not len(phase) or (phase == V.PHASE_VOCAB_INDEX['combat']).any():
            raise ValueError('upper teacher requires nonempty upper decision rows')
        groups = []
        for phase_id in np.unique(phase):
            selected = np.flatnonzero(phase == phase_id)
            keys = OVERWORLD_OBSERVATION_KEYS | OVERWORLD_PHASE_OBSERVATION_KEYS.get(int(phase_id), frozenset())
            batch = {key: np.asarray(value)[selected].astype(np.float32, copy=False)
                     if np.issubdtype(value.dtype, np.floating) else np.asarray(value)[selected]
                     for key, value in observations.items() if key in keys}
            # The encoder already discards these padding tails. Do that before
            # transport too, without changing token positions or model weights.
            groups.append((selected, int(phase_id), self._trim_numpy_padding(batch), batch))
        return groups

    @torch.no_grad()
    def forward_phase_values(self, observations, phase_id, active_widths=None):
        """Search-only critic; identical encoder/readout, no unused actor heads.

        Like forward_phase, inputs are validated by the preparation boundary.
        Ordinary action selection always retains the complete frozen actor.
        """
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            token, _ = self.overworld.encoder(observations, self.upper_tokenizer,
                                             phase_id, active_widths=active_widths)
            return self.overworld.values_from_token(token).float()

    @torch.no_grad()
    def forward_phase(self, observations, phase_id, active_widths=None):
        """Internal phase entry for already validated/projected inputs."""
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            logits, values = self.overworld(observations, self.upper_tokenizer, phase_id,
                                           upper_active_widths=active_widths)
        return logits.float(), values.float()

    @torch.no_grad()
    def forward(self, observations):
        observations = self._validated_observations(observations)
        batch_size = len(observations['phase'])
        phase = observations['phase'].long().flatten()
        if not batch_size or (phase == V.PHASE_VOCAB_INDEX['combat']).any():
            raise ValueError('upper teacher requires nonempty upper decision rows')
        indices, logits, values = [], [], []
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            for phase_id in torch.unique(phase).tolist():
                selected = (phase == phase_id).nonzero().flatten()
                keys = OVERWORLD_OBSERVATION_KEYS | OVERWORLD_PHASE_OBSERVATION_KEYS.get(phase_id, frozenset())
                batch = {key: value[selected].float() if value.is_floating_point() else value[selected]
                         for key, value in observations.items() if key in keys}
                out, value = self.forward_phase(batch, phase_id)
                indices.append(selected)
                logits.append(out.float())
                values.append(value.float())
        order = torch.argsort(torch.cat(indices))
        return torch.cat(logits)[order], torch.cat(values)[order]

    @torch.no_grad()
    def decide(self, observations, masks):
        logits, values = self(observations)
        return self.actions_from_logits(observations, masks, logits), values

    @torch.no_grad()
    def actions_from_logits(self, observations, masks, logits):
        """Apply the frozen actor's legal-action contract to an existing forward."""
        self._validate_inference_mode()
        phase_names = {value: key for key, value in V.PHASE_VOCAB_INDEX.items()}
        masks = np.asarray(masks, dtype=bool)
        if masks.shape != tuple(logits.shape):
            raise ValueError('upper masks must match the policy batch/action shape')
        def host_column(key):
            value = observations[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.dtype == torch.bfloat16:
                    value = value.float()
                value = value.numpy()
            value = np.asarray(value).reshape(-1)
            if len(value) != len(masks):
                raise ValueError('upper phase/act rows must match the policy batch')
            return value
        phases, acts = host_column('phase'), host_column('act')
        legal = np.stack([action_mask_for_policy(self, masks[i],
            phase=phase_names[int(phases[i])],
            act=round(float(acts[i]) * encode.ACT_SCALE),
            reward_policy_contract=self.reward_policy_contract) for i in range(len(masks))])
        legal = torch.as_tensor(legal, device=logits.device)
        return logits.masked_fill(~legal, -torch.inf).argmax(-1)
