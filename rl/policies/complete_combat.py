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
        self.subset = AutoregressiveSubsetHead(width)
        self.scale = width ** -0.5
        # Warm-start new target/subset factors uniformly over the legal set.
        nn.init.zeros_(self.source_query.weight)
        nn.init.zeros_(self.global_query.weight)

    def targets(self, global_token, sources, monsters):
        query = self.source_query(sources) + self.global_query(global_token).unsqueeze(1)
        scores = torch.einsum('bsd,btd->bst', query, self.target_key(monsters)) * self.scale
        return torch.cat((scores, torch.zeros_like(scores[..., :1])), dim=-1)

    def forward(self, source_logits, global_token, sources, monsters, legal, *, evaluate_subsets=True,
                target_logits=None):
        subsets = (self.subset(global_token, sources[:, :10], legal[:, C.JOINT_MULTI-C.JOINT_END:])
                   if evaluate_subsets else global_token.new_zeros((global_token.shape[0], 1024)))
        if target_logits is None:
            target_logits = self.targets(global_token, sources, monsters)
        return complete_logits(source_logits, target_logits,
                               subsets, legal)


class AutoregressiveSubsetHead(nn.Module):
    """Select increasing hand indices or STOP, with exactly one path per set.

    Parameters are shared across all prefixes and hand slots. Enumeration is
    used only to evaluate the small legal search domain (at most ten cards),
    not as a separate learned class for every subset. Prefixes containing card
    nine have a forced STOP and need no decoder evaluation.
    """
    def __init__(self, width, decoder_width=32):
        super().__init__()
        self.hand = nn.Linear(width, decoder_width)
        self.state = nn.Linear(width, decoder_width)
        self.prefix = nn.Linear(decoder_width, decoder_width, bias=False)
        self.choice_key = nn.Linear(decoder_width, decoder_width, bias=False)
        self.stop = nn.Linear(decoder_width, 1)
        nn.init.zeros_(self.choice_key.weight)
        nn.init.zeros_(self.stop.weight)
        nn.init.zeros_(self.stop.bias)
        self.scale = decoder_width ** -.5
        prefixes = torch.arange(512)
        cards = torch.arange(10)
        membership = (prefixes[:, None].bitwise_and(1 << cards[None, :]) != 0).float()
        last = torch.where(membership.bool(), cards[None, :], -1).max(-1).values
        children = prefixes[:, None].bitwise_or(1 << cards[None, :])
        self.register_buffer('membership', membership, persistent=False)
        self.register_buffer('child_prefixes', children, persistent=False)
        self.register_buffer('can_follow', cards[None, :] > last[:, None], persistent=False)
        sets = torch.arange(1024)[:, None]
        before = sets.bitwise_and((1 << cards[None, :]) - 1)
        edges = before * 11 + cards[None, :]
        active = sets.bitwise_and(1 << cards[None, :]) != 0
        stop_edges = torch.arange(1024).clamp_max(511)[:, None] * 11 + 10
        self.register_buffer('path_edges', torch.cat((edges, stop_edges), -1).reshape(-1), persistent=False)
        self.register_buffer('path_active', torch.cat((active, sets < 512), -1), persistent=False)

    def prefix_log_probs(self, global_token, hand_tokens, legal_sets):
        # Counts of legal completions make the initial residual-zero policy
        # uniform over *sets*, instead of preferring short selection paths.
        counts = legal_sets.float().clone()
        for card in range(9, -1, -1):
            size = 1 << card
            counts[:, :size] = counts[:, :size] + counts[:, size:2*size]
        child_counts = counts[:, self.child_prefixes]
        valid_next = (child_counts > 0) & self.can_follow
        hand = self.hand(hand_tokens)
        selected = torch.einsum('ph,bhd->bpd', self.membership.to(hand.dtype), hand)
        query = torch.tanh(self.state(global_token).unsqueeze(1) + self.prefix(selected))
        next_scores = torch.einsum('bpd,bhd->bph', query, self.choice_key(hand)) * self.scale
        next_scores = next_scores.float() + child_counts.clamp_min(1).log()
        scores = torch.cat((next_scores, self.stop(query).float()), -1)
        valid = torch.cat((valid_next, legal_sets[:, :512, None].bool()), -1)
        return conditional_log_probs(scores, valid)

    def forward(self, global_token, hand_tokens, legal_sets):
        result = global_token.new_zeros((global_token.shape[0], 1024), dtype=torch.float32)
        # Multi screens are sparse. Never materialize 512 prefixes for normal
        # combat rows. Callers must route these sparse rows outside graph capture.
        rows = legal_sets.any(dim=-1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return result
        logp = self.prefix_log_probs(global_token[rows], hand_tokens[rows], legal_sets[rows])
        paths = logp.flatten(1)[:, self.path_edges].reshape(-1, 1024, 11)
        scores = (paths * self.path_active).sum(-1)
        return result.index_copy(0, rows, scores)
