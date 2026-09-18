"""Small fixed-shape combat actor-critic with no shared tokenizer or attention."""

from math import prod

import torch
from torch import nn

from env.core import action_space as A


class CombatMLPActorCritic(nn.Module):
    """Slot-preserving numeric/embedded inputs and two independent MLP heads.

    Pile and relic IDs are unordered multisets; language token streams use masked
    mean embeddings rather than their former sequence encoder. Numeric counters,
    full card-count vectors, hand order and monster slots remain explicit inputs.
    """

    def __init__(self, observation_space, observation_keys):
        super().__init__()
        keys = sorted(key for key in observation_keys if key in observation_space.spaces)
        self.scalar_keys = tuple(
            key for key in keys if observation_space[key].dtype.kind == "f"
        )
        self.slot_keys = tuple(
            key for key in keys if observation_space[key].dtype.kind in "iu"
            and key not in {"draw_card_ids", "discard_card_ids", "relics"}
            and not key.endswith("_language")
        )
        self.pool_keys = tuple(
            key for key in keys if key in {"draw_card_ids", "discard_card_ids", "relics"}
            or key.endswith("_language")
        )
        self.embeddings = nn.ModuleDict()
        width = sum(prod(observation_space[key].shape) for key in self.scalar_keys)
        for key in (*self.slot_keys, *self.pool_keys):
            dim = 16 if "card_ids" in key or key == "hand_ids" else 8
            self.embeddings[key] = nn.Embedding(
                int(observation_space[key].high.max()) + 1, dim, padding_idx=0,
            )
            if key in self.slot_keys:
                width += prod(observation_space[key].shape) * dim
            else:
                rows = prod(observation_space[key].shape[:-1])
                width += rows * dim * (2 if key in {"draw_card_ids", "discard_card_ids"} else 1)
        self.input_width = width
        self.trunk = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.policy_mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.policy_head = nn.Linear(128, A.N_ACTIONS)
        self.value_mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.value_head = nn.Linear(128, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    @staticmethod
    def _pool(vectors, weights):
        return (vectors * weights.unsqueeze(-1)).sum(dim=-2) / weights.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1)

    def forward(self, obs, shared=None, phase_id=None, *, active_relic_width=None):
        batch = obs["player_scalars"].shape[0]
        parts = [obs[key].float().reshape(batch, -1) for key in self.scalar_keys]
        for key in self.slot_keys:
            parts.append(self.embeddings[key](obs[key].long()).reshape(batch, -1))
        for key in self.pool_keys:
            ids = obs[key].long()
            vectors = self.embeddings[key](ids)
            weights = ids.ne(0).float()
            if key in {"draw_card_ids", "discard_card_ids"}:
                pile = key.removesuffix("_card_ids")
                parts.append(self._pool(vectors, weights * obs[f"{pile}_card_counts"].float()))
                parts.append(self._pool(vectors, weights * obs[f"{pile}_card_upgraded_counts"].float()))
            else:
                parts.append(self._pool(vectors, weights).reshape(batch, -1))
        features = self.trunk(torch.cat(parts, dim=-1))
        return self.policy_head(self.policy_mlp(features)), self.value_head(self.value_mlp(features))
