"""A one-block combat Transformer used to condition the compact action scorer."""

import numpy as np
import torch
from torch import nn

from rl.policies.small_playout import SmallPlayoutPolicy, PreparedPlayoutSamples

from env.search.token_context import (
    TOKEN_COUNT,
    TOKEN_FEATURES,
    WIDTH,
    HEADS,
    public_tokens,
)


class CombatTokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(TOKEN_FEATURES, WIDTH)
        self.slots = nn.Embedding(TOKEN_COUNT, WIDTH)
        self.norm1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.mix = nn.Linear(WIDTH, WIDTH)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.ff1 = nn.Linear(WIDTH, 2 * WIDTH)
        self.ff2 = nn.Linear(2 * WIDTH, WIDTH)
        self.final = nn.LayerNorm(WIDTH)
        self.out = nn.Linear(WIDTH, 64, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, tokens):
        present = tokens[:, :, -1] > 0
        present = torch.cat((torch.ones_like(present[:, :1]), present[:, 1:]), dim=1)
        x = self.input(tokens) + self.slots.weight
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(-1, TOKEN_COUNT, 3, HEADS, WIDTH // HEADS)
            .permute(2, 0, 3, 1, 4)
        )
        attended = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=present[:, None, None, :]
        )
        x = x + self.mix(attended.transpose(1, 2).reshape(-1, TOKEN_COUNT, WIDTH))
        x = x + self.ff2(torch.relu(self.ff1(self.norm2(x))))
        x = self.final(x)
        # Pool real updated tokens as well as CLS: entity-to-entity attention
        # affects decisions even with a single Transformer block.
        pooled = (x * present[:, :, None]).sum(1) / present.sum(1, keepdim=True)
        return self.out((x[:, 0] + pooled) * 0.5)


class TokenPlayoutPolicy(SmallPlayoutPolicy):
    def __init__(self, key_count):
        super().__init__(key_count)
        self.encoder = CombatTokenEncoder()
        self.context_mode = "token-root"

    @classmethod
    def from_mature(cls, parent):
        net = cls(parent.key_count).to(parent.fc1.weight.device)
        result = net.load_state_dict(parent.state_dict(), strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {
            "encoder." + k for k in net.encoder.state_dict()
        }:
            raise ValueError("token fork needs a compact parent")
        return net

    def forward_packed(self, state, actions, keys, slots, tokens, mask):
        z = self.fc2(torch.relu(self.fc1(state) + self.encoder(tokens)))
        return self._score_packed(z, actions, keys, slots, mask)

    def forward(self, state, actions, keys, tokens, mask):
        z = self.fc2(torch.relu(self.fc1(state) + self.encoder(tokens)))
        return self._score_actions(z[:, None], actions, keys).masked_fill(
            ~mask, -torch.inf
        )

    def export(self):
        raise ValueError(
            "global token control is an offline scorer, not a v1 native policy"
        )

    def prepare_samples(self, samples, target_mix):
        return PreparedTokenSamples(samples, target_mix)


class PreparedTokenSamples(PreparedPlayoutSamples):
    def __init__(self, samples, target_mix):
        super().__init__(samples, target_mix)
        self.tokens = np.stack([s["public_tokens"] for s in samples])
        if (
            self.tokens.shape != (len(samples), TOKEN_COUNT, TOKEN_FEATURES)
            or not np.isfinite(self.tokens).all()
        ):
            raise ValueError("invalid public token training bank")

    def batch(self, indices, device):
        args, target, teacher = super().batch(indices, device)
        return (
            (
                *args[:-1],
                torch.as_tensor(self.tokens[indices], device=device),
                args[-1],
            ),
            target,
            teacher,
        )
