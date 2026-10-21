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

    @torch.no_grad()
    def forward(self, observations):
        if any(p.requires_grad for p in self.parameters()) or any(m.training for m in self.modules()):
            raise RuntimeError('frozen upper was made trainable')
        observations = observation_for_policy(self, observations)
        batch_size = len(observations['phase'])
        for name, space in self.observation_space.spaces.items():
            if tuple(observations[name].shape) != (batch_size, *space.shape):
                raise ValueError(f'frozen upper observation shape changed: {name}')
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
                out, value = self.overworld(batch, self.upper_tokenizer, phase_id)
                indices.append(selected)
                logits.append(out.float())
                values.append(value.float())
        order = torch.argsort(torch.cat(indices))
        return torch.cat(logits)[order], torch.cat(values)[order]

    @torch.no_grad()
    def decide(self, observations, masks):
        logits, values = self(observations)
        phase_names = {value: key for key, value in V.PHASE_VOCAB_INDEX.items()}
        masks = np.asarray(masks, dtype=bool)
        if masks.shape != tuple(logits.shape):
            raise ValueError('upper masks must match the policy batch/action shape')
        legal = np.stack([action_mask_for_policy(self, masks[i],
            phase=phase_names[int(observations['phase'][i].item())],
            act=round(float(observations['act'][i].item()) * encode.ACT_SCALE),
            reward_policy_contract=self.reward_policy_contract) for i in range(len(masks))])
        legal = torch.as_tensor(legal, device=logits.device)
        actions = logits.masked_fill(~legal, -torch.inf).argmax(-1)
        return actions, values
