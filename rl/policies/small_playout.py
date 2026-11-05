"""Compact StS1 policy and the versioned, contiguous native weight contract.

No critic: MCTS uses policy playouts and explicit terminal outcome targets.
The key table contains real StS1 card/upgrade and potion IDs, not StS2 padding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import numpy as np
import torch
from torch import nn

CONTRACT = 'sts1-small-playout-v1'


def export_online_search_pair(policy, *, collect_samples):
    """Snapshot the online lower policy with its shared collection search budget."""
    import slaythespire as sts
    from env.search.small_playout import SEARCH_CONTRACT
    if policy.key_count != sts.small_playout_key_count():
        raise ValueError('native/Python action vocabulary mismatch')
    weights = policy.export()
    context = getattr(policy, 'context_mode', 'none')
    contract = CONTRACT if context == 'none' else 'sts1-small-playout-public-v1'
    digest = hashlib.sha256(weights.tobytes())
    if context != 'none':
        digest.update(context.encode())
    return dict(sha256=digest.hexdigest(), context_mode=context,
                contract=contract, search_contract=SEARCH_CONTRACT,
                online=bool(collect_samples), policy=weights, prior=None,
                common_simulations=250, elite_simulations=250, boss_simulations=1000,
                playout_depth=200, tree_depth=1024)


@dataclass(frozen=True)
class OnlinePlayoutConfig:
    key_count: int = 785
    learning_rate: float = 3e-4
    epochs: int = 2
    batch_size: int = 2048
    micro_batch_size: int = 256
    max_grad_norm: float = 1.
    target_mix: float = 1.
    weight_decay: float = 0.
    optimizer: str = 'adam'
    context_mode: str = 'none'

    def __post_init__(self):
        if (not all(math.isfinite(v) for k, v in asdict(self).items() if k not in ('optimizer','context_mode'))
                or self.context_mode not in ('none','root','turn','state')
                or self.optimizer not in ('adam', 'adamw')
                or not 0 <= self.target_mix <= 1 or self.weight_decay < 0
                or self.key_count < 3 or self.learning_rate <= 0 or self.max_grad_norm < 0
                or min(self.epochs, self.batch_size, self.micro_batch_size) < 1
                or any(int(v) != v for v in (self.key_count, self.epochs, self.batch_size, self.micro_batch_size))):
            raise ValueError('invalid online compact policy configuration')


def search_target(sample, target_mix):
    """Reference teacher: recommended move mass plus the root visit distribution."""
    visits = np.asarray(sample['visits'], np.float32)
    if not np.isfinite(visits).all() or np.any(visits < 0) or visits.sum() <= 0:
        raise ValueError('online compact supervision needs finite positive visits')
    target = visits / visits.sum()
    if target_mix < 1:
        bits = np.asarray(sample['action_bits'])
        chosen = np.flatnonzero(bits == sample['recommended_action'])
        if len(chosen) != 1:
            raise ValueError('search recommendation must name exactly one legal action')
        target *= target_mix
        target[chosen[0]] += 1 - target_mix
    return target


class SmallPlayoutPolicy(nn.Module):
    def __init__(self, key_count: int):
        super().__init__()
        if key_count < 3:
            raise ValueError('invalid StS1 action vocabulary')
        self.key_count = key_count
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.keys = nn.Embedding(key_count, 32)
        self.bias = nn.Embedding(key_count, 1)
        self.projection = nn.Linear(24, 32, bias=False)
        self.linear = nn.Linear(24, 1, bias=False)
        nn.init.normal_(self.keys.weight, std=.02)
        nn.init.zeros_(self.bias.weight)

    def forward(self, state, actions, keys, mask):
        z = self.fc2(torch.relu(self.fc1(state)))
        logits = self._score_actions(z.unsqueeze(1), actions, keys)
        return logits.masked_fill(~mask, -torch.inf)

    def _score_actions(self, z, actions, keys):
        embedding, bias = self._key_features(keys)
        logits = ((embedding + self.projection(actions)) * z).sum(-1)
        logits = logits + bias + self.linear(actions).squeeze(-1)
        return logits

    def _key_features(self, keys):
        valid = keys >= 0
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        indices = keys.clamp_min(0)
        embedding = (self.keys(indices) * valid.unsqueeze(-1)).sum(-2) / count
        bias = (self.bias(indices).squeeze(-1) * valid).sum(-1) / count.squeeze(-1)
        return embedding, bias

    def forward_packed(self, state, actions, keys, slots, mask):
        """Embed real candidates; preserve dense projection/gradient reductions.

        Packing the projection GEMM also changes near-zero gradient cancellation
        in action-constant features, which Adam can amplify. Keep that reduction
        layout and remove the much larger candidate-by-key embedding padding.
        """
        z = self.fc2(torch.relu(self.fc1(state)))
        return self._score_packed(z, actions, keys, slots, mask)

    def _score_packed(self, z, actions, keys, slots, mask):
        embedding, bias = self._key_features(keys)
        dense_embedding = embedding.new_zeros((mask.numel(), embedding.shape[-1]))
        dense_embedding = dense_embedding.index_copy(0, slots, embedding).view(*mask.shape, -1)
        dense_bias = bias.new_zeros(mask.numel()).index_copy(0, slots, bias).view_as(mask)
        dense_actions = actions.new_zeros((mask.numel(), actions.shape[-1]))
        dense_actions = dense_actions.index_copy(0, slots, actions).view(*mask.shape, -1)
        logits = ((dense_embedding + self.projection(dense_actions)) * z.unsqueeze(1)).sum(-1)
        logits = logits + dense_bias + self.linear(dense_actions).squeeze(-1)
        return logits.masked_fill(~mask, -torch.inf)

    def prepare_samples(self,samples,target_mix):
        return PreparedPlayoutSamples(samples,target_mix)

    def export(self):
        # Keep order explicit: module registration order is not an export ABI.
        parameters = (self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias,
                      self.keys.weight, self.bias.weight, self.projection.weight, self.linear.weight)
        return torch.cat([p.detach().flatten().cpu() for p in parameters]).float().numpy()


def collate_features(samples, device='cpu'):
    if not samples:
        raise ValueError('empty compact policy batch')
    width = max(len(s['action_bits']) for s in samples)
    if width == 0:
        raise ValueError('compact policy requires a live decision')
    state = np.stack([s['state'] for s in samples])
    actions = np.zeros((len(samples), width, 24), np.float32)
    keys = np.full((len(samples), width, 10), -1, np.int64)
    mask = np.zeros((len(samples), width), bool)
    for i, sample in enumerate(samples):
        n = len(sample['action_bits'])
        actions[i, :n], keys[i, :n], mask[i, :n] = sample['actions'], sample['keys'], True
    return tuple(torch.as_tensor(x, device=device) for x in (state, actions, keys, mask))


class PreparedPlayoutSamples:
    """Immutable CPU feature/target arrays reused across online CE epochs.

    Inputs contain real candidates only; the forward pass retains the projection
    reduction layout. Candidate/key order, shuffle, microbatch boundaries and the
    effective optimizer batch remain unchanged.
    This is collection-local data and is never part of a model checkpoint.
    """

    def __init__(self, samples, target_mix):
        if not samples:
            raise ValueError('empty compact policy samples')
        self.lengths = np.asarray([len(s['action_bits']) for s in samples], np.int64)
        if np.any(self.lengths == 0):
            raise ValueError('compact policy requires a live decision')
        self.starts = np.cumsum(self.lengths) - self.lengths
        self.states = np.stack([s['state'] for s in samples])
        self.actions = np.concatenate([s['actions'] for s in samples]).astype(np.float32, copy=False)
        self.keys = np.concatenate([s['keys'] for s in samples]).astype(np.int64, copy=False)
        targets = [search_target(s, target_mix) for s in samples]
        self.targets = np.concatenate(targets)
        self.teacher = np.asarray([row.argmax() for row in targets], np.int64)
        self.target_entropy = sum(-float(np.sum(row[row > 0] * np.log(row[row > 0])))
                                  for row in targets)
        self.uniform_ce = float(np.log(self.lengths).sum())

    def batch(self, indices, device):
        lengths = self.lengths[indices]
        width = int(lengths.max())
        ends = np.cumsum(lengths)
        owners = np.repeat(np.arange(len(indices)), lengths)
        within = np.arange(ends[-1]) - np.repeat(ends - lengths, lengths)
        source = np.repeat(self.starts[indices], lengths) + within
        slots = owners * width + within
        mask = np.zeros(len(indices) * width, bool)
        mask[slots] = True
        target = np.zeros(len(indices) * width, np.float32)
        target[slots] = self.targets[source]
        inputs = (self.states[indices], self.actions[source], self.keys[source],
                  slots, mask.reshape(len(indices), width))
        return (tuple(torch.as_tensor(x, device=device) for x in inputs),
                torch.as_tensor(target.reshape(len(indices), width), device=device),
                torch.as_tensor(self.teacher[indices], device=device))
