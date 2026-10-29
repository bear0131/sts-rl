"""Compact StS1 policy and the versioned, contiguous native weight contract.

No critic: MCTS leaves use a policy playout and a hand-written battle score.
The key table contains real StS1 card/upgrade and potion IDs, not StS2 padding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch import nn

CONTRACT = 'sts1-small-playout-v1'


@dataclass(frozen=True)
class OnlinePlayoutConfig:
    key_count: int = 785
    learning_rate: float = 3e-4
    epochs: int = 2
    batch_size: int = 2048
    micro_batch_size: int = 256
    max_grad_norm: float = 1.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in asdict(self).values())
                or self.key_count < 3 or self.learning_rate <= 0 or self.max_grad_norm <= 0
                or min(self.epochs, self.batch_size, self.micro_batch_size) < 1
                or any(int(v) != v for v in (self.key_count, self.epochs, self.batch_size, self.micro_batch_size))):
            raise ValueError('invalid online compact policy configuration')


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
        valid = keys >= 0
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        indices = keys.clamp_min(0)
        embedding = (self.keys(indices) * valid.unsqueeze(-1)).sum(-2) / count
        bias = (self.bias(indices).squeeze(-1) * valid).sum(-1) / count.squeeze(-1)
        logits = ((embedding + self.projection(actions)) * z.unsqueeze(1)).sum(-1)
        logits = logits + bias + self.linear(actions).squeeze(-1)
        return logits.masked_fill(~mask, -torch.inf)

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
