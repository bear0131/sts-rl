"""Lossless complete-action scores with a source-conditional target policy."""
import torch
from torch import nn

from env.core import action_space as A
from env.core import native_actions as C


def conditional_log_probs(logits, legal):
    """Finite on empty groups; a sole legal target has log probability zero."""
    scores = logits.float().masked_fill(~legal.bool(), -1e9)
    return scores - scores.logsumexp(dim=-1, keepdim=True)


def complete_logits(source_logits, target_logits, subset_logits, legal):
    """Joint CE equals source CE plus pi(source)-weighted conditional CE.

    Single choices and complete multi subsets are directly scored. The leading
    legacy slots remain for overworld/bridge controls, masked outside that use.
    """
    batch = source_logits.shape[0]
    valid_targets = legal[:, 1:91].reshape(batch, 15, 6)
    conditional = conditional_log_probs(target_logits, valid_targets)
    sources = torch.cat((source_logits[:, A._PLAY:A._PLAY + 10],
                         source_logits[:, A._POTION:A._POTION + 5]), dim=1)
    targeted = (sources.float().unsqueeze(-1) + conditional).reshape(batch, 90)
    return torch.cat((source_logits.float(), source_logits[:, :1].float(), targeted,
                      source_logits[:, A._PICK:A._PICK + A.G_MAX].float(),
                      subset_logits.float()), dim=1)


class CompleteCombatHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.source_query = nn.Linear(width, width, bias=False)
        self.global_query = nn.Linear(width, width, bias=False)
        self.target_key = nn.Linear(width, width, bias=False)
        self.subset = nn.Linear(width, 1 << A.HAND_SLOTS)
        self.scale = width ** -0.5
        # Warm-start new target/subset factors uniformly over the legal set.
        nn.init.zeros_(self.source_query.weight)
        nn.init.zeros_(self.global_query.weight)
        nn.init.zeros_(self.subset.weight)
        nn.init.zeros_(self.subset.bias)

    def targets(self, global_token, sources, monsters):
        query = self.source_query(sources) + self.global_query(global_token).unsqueeze(1)
        scores = torch.einsum('bsd,btd->bst', query, self.target_key(monsters)) * self.scale
        return torch.cat((scores, torch.zeros_like(scores[..., :1])), dim=-1)

    def forward(self, source_logits, global_token, sources, monsters, legal):
        return complete_logits(source_logits, self.targets(global_token, sources, monsters),
                               self.subset(global_token), legal)
