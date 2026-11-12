"""Frozen batched policy/anchor adapter for the model-free chain scheduler."""
import numpy as np
import torch

from rl.policies.deck_latent import SearchContext
from rl.policies.combat_summary import CombatSummary
from rl.policies.split_policy import COMBAT_OBSERVATION_KEYS


class DeckLatentScorer:
    def __init__(self, policy, root_observations, context=None):
        if policy.training:
            raise ValueError('search scorer requires a frozen inference policy')
        self.core = policy.core
        self.device = next(policy.parameters()).device
        self.roots = root_observations
        self.context_refreshes = 0
        self.context_keys = (*self.core.context.keys, 'phase')
        self.combat_keys = tuple(k for k in root_observations[0] if k in COMBAT_OBSERVATION_KEYS)
        self.anchor_keys = ('player_scalars', 'potions', *self.core.anchor.dynamic_keys)
        with torch.no_grad():
            self.context = (self.core.prepare_search_context(self.pack(root_observations, self.context_keys))
                            if context is None else context)
            self.tables = self.core.combat.summaries.export_inference_tables()
            self.summaries = [self.tables.new_state(o) for o in root_observations]
        self.versions = tuple((id(p), p._version) for p in policy.parameters())

    def pack(self, observations, keys):
        return {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=self.device)
                for k in keys}

    @torch.no_grad()
    def __call__(self, requests):
        if tuple((id(p), p._version) for p in self.core.parameters()) != self.versions:
            raise RuntimeError('search weights changed during a frozen collection')
        if not requests:
            return []
        roots = torch.tensor([r.root for r in requests], device=self.device)
        context = self.context.index(roots)
        # A branch retains its refreshed static context after permanent changes.
        # Counter changes/card moves do not enter this key or invoke the encoder.
        changed, summaries = [], []
        for i, r in enumerate(requests):
            if 'summary' not in r.cache:
                r.cache.update(summary=self.summaries[r.root].fork(), static=self.roots[r.root])
            summaries.append(r.cache['summary'].update(r.observation))
            if any(not np.array_equal(r.observation[k], r.cache['static'][k]) for k in self.core.context.keys):
                changed.append(i)
            elif 'context' in r.cache:
                context.combat[i], context.anchor[i] = r.cache['context']
        if changed:
            rows = torch.tensor(changed, device=self.device)
            refreshed = self.core.prepare_search_context(
                self.pack([requests[i].observation for i in changed], self.context_keys))
            context.combat[rows] = refreshed.combat
            context.anchor[rows] = refreshed.anchor
            self.context_refreshes += len(changed)
            for j, i in enumerate(changed):
                requests[i].cache['static'] = {k: requests[i].observation[k].copy() for k in self.core.context.keys}
                requests[i].cache['context'] = (refreshed.combat[j], refreshed.anchor[j])
        summary = CombatSummary(*(torch.as_tensor(np.stack([getattr(s, key) for s in summaries]), device=self.device)
                                  for key in ('piles', 'counts', 'relics')))
        output = [None] * len(requests)
        for anchor in (False, True):
            rows = [i for i, r in enumerate(requests) if (r.kind == 'anchor') == anchor]
            if not rows:
                continue
            index = torch.tensor(rows, device=self.device)
            # The context and summaries already own static deck/map/pile data.
            # Pack only inputs consumed by this head, preserving row order and
            # the exact values/shape seen by the previous full-dict projection.
            batch = self.pack([requests[i].observation for i in rows],
                              self.anchor_keys if anchor else self.combat_keys)
            z = context.index(index)
            if anchor:
                values = self.core.anchor_with_context(batch, z, summary.relics[index]).flatten().cpu().numpy()
                for i, value in zip(rows, values, strict=True):
                    output[i] = (None, float(value))
            else:
                partial = CombatSummary(summary.piles[index], summary.pile_counts[index], summary.relics[index])
                logits, values = self.core.combat_with_context(batch, z, partial)
                logits, values = logits.cpu().numpy(), values.flatten().cpu().numpy()
                for i, logit, value in zip(rows, logits, values, strict=True):
                    output[i] = (logit, float(value))
        return output
