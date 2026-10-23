"""A hurdle distribution: death mass at zero and conditional survivor utility."""
import torch
from torch import nn


class CombatDistributionHead(nn.Module):
    def __init__(self, width, quantiles=32):
        super().__init__()
        if quantiles < 2:
            raise ValueError('at least two HP quantiles are required')
        self.trunk = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.win = nn.Linear(width, 1)
        self.spacings = nn.Linear(width, quantiles + 1)

    def forward(self, state):
        features = self.trunk(state)
        win = self.win(features).float()
        # Positive spacings enforce ordered, bounded quantiles without clipping
        # the expectation. Endpoints allow arbitrary proximity to zero and one.
        hp = self.spacings(features).float().softmax(-1).cumsum(-1)[:, :-1]
        value = win.sigmoid() * hp.mean(-1, keepdim=True)
        return value, dict(win_logits=win, hp_quantiles=hp)


class CombatMeanHead(nn.Module):
    """Two expectations only: victory probability and survivor HP utility."""
    def __init__(self, width):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.win = nn.Linear(width, 1)
        self.hp = nn.Linear(width, 1)

    def forward(self, state):
        features = self.trunk(state)
        win = self.win(features).float()
        hp = self.hp(features).float().sigmoid()
        return win.sigmoid() * hp, dict(win_logits=win, hp_mean=hp)


class CombatConditionalValueHead(nn.Module):
    """Win probability times an unbounded conditional curve-return mean."""
    def __init__(self, width):
        super().__init__()
        # Preserve the finite scalar baseline's value branch architecture.
        self.value = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.win = nn.Linear(width, 1)

    def forward(self, state):
        with torch.autocast(device_type=state.device.type, enabled=False):
            win = self.win(state.float())
            value = self.value(state.float())
            return win.sigmoid() * value, dict(win_logits=win, value_given_win=value)
