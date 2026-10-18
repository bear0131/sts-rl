"""Candidate-scoring baseline for the versioned STS2 visible observation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from env.sts2.encoding import ACTION_DIM, STATE_DIM


def compact(observation):
    n = int(observation["mask"].sum())
    return dict(state=observation["state"].copy(), candidates=observation["candidates"][:n].copy(),
                mask=observation["mask"][:n].copy())


def batch_observations(observations, device):
    n = max(len(o["mask"]) for o in observations)
    candidates = np.zeros((len(observations), n, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((len(observations), n), dtype=bool)
    for i, observation in enumerate(observations):
        size = len(observation["mask"])
        candidates[i, :size] = observation["candidates"]
        mask[i, :size] = observation["mask"]
    return {k: torch.as_tensor(v, device=device) for k, v in dict(
        state=np.stack([o["state"] for o in observations]), candidates=candidates, mask=mask).items()}


class CandidatePolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.state_net = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.action_net = nn.Sequential(nn.Linear(ACTION_DIM, hidden), nn.Tanh())
        self.score = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        state = self.state_net(obs["state"])
        logits = self.score(torch.tanh(state[:, None, :] + self.action_net(obs["candidates"]))).squeeze(-1)
        if not obs["mask"].any(dim=-1).all():
            raise ValueError("Cannot sample a terminal/no-action observation")
        logits = logits.masked_fill(~obs["mask"].bool(), -torch.inf)
        return torch.distributions.Categorical(logits=logits), self.value_head(state).squeeze(-1)

    def values(self, observations):
        state = torch.as_tensor(np.stack([o["state"] for o in observations]), device=next(self.parameters()).device)
        return self.value_head(self.state_net(state)).squeeze(-1)

    def evaluate_actions(self, observations, actions, action_masks=None):
        distribution, values = self(observations)
        return values, distribution.log_prob(actions), distribution.entropy()
