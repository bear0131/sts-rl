"""Frozen batched policy/anchor adapter for the model-free chain scheduler."""
import numpy as np
import torch

from rl.policies.deck_latent import SearchContext


class DeckLatentScorer:
    def __init__(self, policy, root_observations):
        if policy.training:
            raise ValueError('search scorer requires a frozen inference policy')
        self.core = policy.core
        self.device = next(policy.parameters()).device
        self.roots = root_observations
        self.context_refreshes = 0
        with torch.no_grad():
            self.context = self.core.prepare_search_context(self.pack(root_observations))
        self.versions = tuple((id(p), p._version) for p in policy.parameters())

    def pack(self, observations):
        return {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=self.device)
                for k in observations[0]}

    @torch.no_grad()
    def __call__(self, requests):
        if tuple((id(p), p._version) for p in self.core.parameters()) != self.versions:
            raise RuntimeError('search weights changed during a frozen collection')
        if not requests:
            return []
        obs = self.pack([r.observation for r in requests])
        roots = torch.tensor([r.root for r in requests], device=self.device)
        context = self.context.index(roots)
        # Reference content check. Native lanes use versioned dirty events for
        # these uncommon permanent changes, not a full key scan every step.
        changed = [i for i, r in enumerate(requests) if any(
            not np.array_equal(r.observation[k], self.roots[r.root][k]) for k in self.core.context.keys)]
        if changed:
            rows = torch.tensor(changed, device=self.device)
            refreshed = self.core.prepare_search_context({k: v[rows] for k, v in obs.items()})
            context.combat[rows] = refreshed.combat
            context.anchor[rows] = refreshed.anchor
            self.context_refreshes += len(changed)
        output = [None] * len(requests)
        for anchor in (False, True):
            rows = [i for i, r in enumerate(requests) if (r.kind == 'anchor') == anchor]
            if not rows:
                continue
            index = torch.tensor(rows, device=self.device)
            batch = {k: v[index] for k, v in obs.items()}
            z = context.index(index)
            if anchor:
                values = self.core.anchor_with_context(batch, z).flatten().cpu().numpy()
                for i, value in zip(rows, values, strict=True):
                    output[i] = (None, float(value))
            else:
                logits, values = self.core.combat_with_context(batch, z)
                logits, values = logits.cpu().numpy(), values.flatten().cpu().numpy()
                for i, logit, value in zip(rows, logits, values, strict=True):
                    output[i] = (logit, float(value))
        return output
