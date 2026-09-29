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
        self.public_choices = "select_native_indices" in observation_space.spaces
        keys = sorted(key for key in observation_keys if key in observation_space.spaces
                      and key not in {"select_native_indices", "select_card_ids",
                                      "select_card_costs", "select_card_upgrades"})
        if "select_native_indices" not in observation_space.spaces:
            keys = [key for key in keys if key not in {
                "select_card_ids", "select_card_upgrades", "pending_card_slot",
            }]
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
        if self.public_choices:
            self.choice_embedding = nn.Embedding(
                int(observation_space["select_card_ids"].high.max()) + 1, 16, padding_idx=0)
            self.choice_key = nn.Linear(18, 64)
            self.choice_query = nn.Linear(256, 64, bias=False)
            self.choice_score = nn.Linear(64, 1)
            width += 18
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
        if self.public_choices:
            choices = torch.cat((self.choice_embedding(obs["select_card_ids"].long()),
                obs["select_card_upgrades"].float().unsqueeze(-1),
                obs["select_card_costs"].float().unsqueeze(-1)), dim=-1)
            parts.append(self._pool(choices, obs["select_native_indices"].ne(0).float()))
        features = self.trunk(torch.cat(parts, dim=-1))
        logits = self.policy_head(self.policy_mlp(features))
        if self.public_choices:
            logits[:, A._PICK:A._PICK + A.G_MAX] = self.choice_score(torch.tanh(
                self.choice_key(choices) + self.choice_query(features).unsqueeze(1))).squeeze(-1)
        return logits, self.value_head(self.value_mlp(features))
