"""Additive combat inputs shared by full training and sparse search updates.

These are learned summaries, not lossless state serialization. An entity's
contribution depends only on that entity: context-dependent attention and
normalization are applied *after* summation, so moving one card is O(width).
The simulator remains the owner of exact slots, legal actions and transitions.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from env.core import encode


@dataclass
class CombatSummary:
    """Branch-owned accumulators; tables and deck/map context are shared."""
    piles: torch.Tensor          # [batch, draw/discard/exhaust, width]
    pile_counts: torch.Tensor    # [batch, 3, 1], unnormalized counts
    relics: torch.Tensor         # [batch, width]

    def fork(self):
        return CombatSummary(self.piles.clone(), self.pile_counts.clone(), self.relics.clone())

    def move_cards(self, rows, sources, destinations, contributions, counts=None):
        """-1 denotes hand or outside inventory; repeated lanes accumulate.

        Call with the old contribution when removing and the new contribution
        when inserting a transformed/upgraded card (two separate events).
        No mutation of the caller's card table or sibling summary is permitted.
        """
        counts = (contributions.new_ones((len(rows), 1)) if counts is None
                  else counts.reshape(-1, 1).to(contributions))
        flat = self.piles.view(-1, self.piles.shape[-1])
        sizes = self.pile_counts.view(-1, 1)
        for zones, sign in ((sources, -1), (destinations, 1)):
            valid = zones >= 0
            index = rows[valid] * 3 + zones[valid]
            flat.index_add_(0, index, sign * contributions[valid])
            sizes.index_add_(0, index, sign * counts[valid])

    def replace_relics(self, rows, before, after):
        self.relics.index_add_(0, rows, after - before)


class AdditiveCombatInputs(nn.Module):
    """Pure per-entity terms; reusable in a native CPU or GPU search lane.

    Relic identity gates each counter feature. A global sum of raw counters
    would confuse e.g. Pen Nib and Nunchaku. The unbounded scalar is retained
    alongside cheap periodic features; counts above a lookup-table range must
    use this function, never clamp or wrap the entire counter.
    """
    counter_features = 10

    def __init__(self, card_embedding, n_relics, width=64):
        super().__init__()
        self.card = card_embedding
        self.width = int(width)
        self.card_projection = nn.Linear(card_embedding.embedding_dim, width, bias=False)
        self.upgrade_projection = nn.Parameter(torch.randn(width) * .02)
        self.relic_identity = nn.Embedding(n_relics, width, padding_idx=0)
        self.relic_counter = nn.Embedding(n_relics, self.counter_features * width, padding_idx=0)

    def card_terms(self, ids, upgrades):
        terms = self.card_projection(self.card.versioned_forward(ids, upgrades))
        terms = terms + upgrades[..., None].to(terms) * self.upgrade_projection
        return terms * ids.ne(0)[..., None]

    def relic_terms(self, ids, counters):
        x = counters.float()
        count = (x * encode.COUNTER_SCALE).round()
        # Original scale stays available, including binary live-relic fields.
        basis = torch.stack((x, x.sign() * x.abs().log1p(),
            *(torch.remainder(count, period) / period for period in (2, 3, 4, 5, 6, 7, 8, 10))), -1)
        gates = self.relic_counter(ids.long()).unflatten(-1, (self.counter_features, self.width))
        result = self.relic_identity(ids.long()) + (gates * basis[..., None]).sum(-2)
        return result * ids.ne(0)[..., None]

    def rebuild(self, obs):
        piles, sizes = [], []
        n_cards = self.card.num_embeddings
        ids = torch.arange(n_cards, device=obs['draw_counts'].device)
        base = self.card_terms(ids, torch.zeros_like(ids, dtype=torch.float32))
        # Version lookup and upgrade magnitude are independent. Searing Blow's
        # repeated upgrades must not be collapsed to the binary version bit.
        upgraded = self.card_projection(self.card.versioned_forward(ids, torch.ones_like(ids)))
        delta = (upgraded - base) * ids.ne(0)[:, None]
        for pile in ('draw', 'discard', 'exhaust'):
            counts = obs[pile + '_counts'].float()
            total = counts @ base
            if 'pile_upgraded_counts' in obs:
                zone = len(piles)
                total = total + obs['pile_upgraded_counts'][:, zone].float() @ delta
                total = total + obs['pile_upgrade_sums'][:, zone].float().sum(1, keepdim=True) * self.upgrade_projection
            elif pile + '_card_ids' in obs:
                card_ids = obs[pile + '_card_ids'].long()
                up_counts = obs[pile + '_card_upgraded_counts'].float()
                total = total + (delta[card_ids] * up_counts[..., None]).sum(1)
                up_sum = (obs[pile + '_card_upgrades'].float() * card_ids.ne(0)).sum(1, keepdim=True)
                total = total + up_sum * self.upgrade_projection
            piles.append(total)
            sizes.append(counts[:, 1:].sum(1, keepdim=True))
        relics = self.relic_terms(obs['relics'], obs['relic_counters']).sum(1)
        return CombatSummary(torch.stack(piles, 1), torch.stack(sizes, 1), relics)

    def features(self, summary):
        # Normalization is cheap and must not be stored inside the accumulator:
        # a move changes the denominator without requiring all cards to update.
        count = summary.pile_counts
        return torch.cat((summary.piles / count.clamp_min(1), count.log1p()), -1).flatten(1)

    @torch.no_grad()
    def export_inference_tables(self):
        """One frozen export per weight version, shared by every CPU lane.

        The same arrays can be consumed by a native event producer. No network
        is evaluated when a card moves or a relic counter changes.
        """
        ids = torch.arange(self.card.num_embeddings, device=self.upgrade_projection.device)
        base = self.card_terms(ids, torch.zeros_like(ids))
        upgraded = self.card_projection(self.card.versioned_forward(ids, torch.ones_like(ids)))
        arrays = [base, (upgraded - base) * ids.ne(0)[:, None], self.upgrade_projection,
                  self.relic_identity.weight, self.relic_counter.weight]
        return FrozenCombatTables(*(x.detach().float().cpu().numpy().copy() for x in arrays))


@dataclass(frozen=True)
class FrozenCombatTables:
    card_base: np.ndarray
    card_upgrade: np.ndarray
    upgrade_magnitude: np.ndarray
    relic_identity: np.ndarray
    relic_counter: np.ndarray

    def __post_init__(self):
        for value in vars(self).values():
            value.setflags(write=False)

    def relic_terms(self, ids, counters):
        x = np.asarray(counters, np.float32)
        count = np.rint(x * encode.COUNTER_SCALE)
        basis = np.stack((x, np.sign(x) * np.log1p(np.abs(x)),
            *(np.remainder(count, p) / p for p in (2, 3, 4, 5, 6, 7, 8, 10))), -1)
        gates = self.relic_counter[ids].reshape(*ids.shape, 10, -1)
        return (self.relic_identity[ids] + (gates * basis[..., None]).sum(-2)) * (ids != 0)[..., None]

    def new_state(self, observation):
        state = ResidentCombatSummary(self)
        state.update(observation)
        return state


class ResidentCombatSummary:
    """Frozen inference accumulators; a card event costs O(summary width).

    ``update`` is the reference adapter for full observations: it compares
    integer-count metadata, then calls the sparse event API. Native producers
    can call that API directly without building or transferring dense piles.
    Normalization stays outside the accumulators. No training graph is cached.
    """
    def __init__(self, tables):
        self.tables = tables
        cards, width = tables.card_base.shape
        self.piles = np.zeros((3, width), np.float32)
        self.counts = np.zeros((3, 1), np.float32)
        self.relics = np.zeros(width, np.float32)
        self.metadata = np.zeros((3, 3, cards), np.float32)
        self.relic_ids = self.relic_counters = None
        self.events = 0

    def fork(self):
        child = ResidentCombatSummary(self.tables)
        for key in ('piles', 'counts', 'relics', 'metadata', 'relic_ids', 'relic_counters'):
            value = getattr(self, key)
            setattr(child, key, None if value is None else value.copy())
        return child

    def card_events(self, zones, ids, counts, upgraded_counts, upgrade_sums):
        """Signed per-ID deltas; upgrades use the observation's UPGRADE_SCALE.

        A move is one removal plus one insertion. An upgrade/transform removes
        the old entity and adds the new entity; no whole-pile recomputation.
        Repeated IDs and zones are intentionally accumulated, never overwritten.
        """
        contribution = counts[:, None] * self.tables.card_base[ids]
        contribution += upgraded_counts[:, None] * self.tables.card_upgrade[ids]
        contribution += upgrade_sums[:, None] * self.tables.upgrade_magnitude
        np.add.at(self.piles, zones, contribution)
        np.add.at(self.counts[:, 0], zones, counts * (ids != 0))
        for i, delta in enumerate((counts, upgraded_counts, upgrade_sums)):
            np.add.at(self.metadata[i], (zones, ids), delta)
        self.events += len(ids)

    def relic_events(self, old_ids, old_counters, new_ids, new_counters):
        self.relics += (self.tables.relic_terms(new_ids, new_counters)
                       - self.tables.relic_terms(old_ids, old_counters)).sum(0)
        self.events += len(new_ids)

    def update(self, observation):
        now = np.stack((np.stack([observation[p + '_counts'] for p in ('draw', 'discard', 'exhaust')]),
                        observation['pile_upgraded_counts'], observation['pile_upgrade_sums']))
        zones, ids = np.nonzero(np.any(now != self.metadata, axis=0))
        delta = now[:, zones, ids] - self.metadata[:, zones, ids]
        self.card_events(zones, ids, *delta)
        # Retain exact encoded metadata, not accumulated float differences.
        self.metadata[...] = now
        ids, counters = observation['relics'], observation['relic_counters']
        if self.relic_ids is None:
            self.relic_ids = np.zeros_like(ids)
            self.relic_counters = np.zeros_like(counters)
        changed = (ids != self.relic_ids) | (counters != self.relic_counters)
        if changed.any():
            self.relic_events(self.relic_ids[changed], self.relic_counters[changed], ids[changed], counters[changed])
            self.relic_ids, self.relic_counters = ids.copy(), counters.copy()
        return self
