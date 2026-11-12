"""Frozen DeckLatent scoring directly from resident CUDA observations."""
import numpy as np
import torch

from env.core.cuda_observation import CudaObservationDecoder
from env.core.outer_return import MISC_SCALE
from rl.policies.combat_summary import CombatSummary
from rl.policies.collection_graphs import SharedCollectionGraphs
from rl.policies.deck_latent import SearchContext
from rl.search.rollout_inference import ExactInputCache


class ResidentDeckScorer:
    def __init__(self, policy, cache_bytes=64*1024**2, graph_capacity=32):
        self.policy, self.core = policy, policy.core
        self.device = next(policy.parameters()).device
        self.decoder = CudaObservationDecoder(policy.observation_space, self.device)
        self.runtime = SharedCollectionGraphs(self.core, capacity=graph_capacity)
        self.anchor_cache = ExactInputCache(cache_bytes // 4)
        self.versions = tuple((id(p), p._version) for p in self.core.parameters())
        self.refreshes = 0
        with torch.no_grad():
            summaries = self.core.combat.summaries
            ids = torch.arange(summaries.card.num_embeddings, device=self.device)
            self.card_base = summaries.card_terms(ids, torch.zeros_like(ids))
            self.card_delta = (summaries.card_projection(summaries.card.versioned_forward(ids,
                torch.ones_like(ids)))-self.card_base)*ids.ne(0)[:, None]

    @torch.no_grad()
    def roots(self, observations, context=None):
        keys = set(self.core.context.keys) | set(self.core.combat.scalar_keys) | {
            'phase', 'relic_counters', 'combat_is_boss'}
        self.root = {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=self.device)
                     for k in keys if k in observations[0]}
        self.context = self.core.prepare_search_context(self.root) if context is None else context
        self.changed_context = {}

    def _forward(self, batch, width, anchor):
        raw = {k[4:]: v for k, v in batch.items() if k.startswith('raw:')}
        roots = {k[5:]: v for k, v in batch.items() if k.startswith('root:')}
        o = self.decoder(raw, roots, torch.arange(len(raw['outcome']), device=self.device))
        if anchor:
            # Cache identity is the complete consumed dynamic state plus the
            # frozen projected context. Combat turn/block do not enter it.
            relics = self.core.combat.summaries.relic_terms(o['relics'], o['relic_counters']).sum(1)
            return o, relics
        o['_choice_width'] = width
        context = SearchContext(batch['context'], None)
        counts = torch.stack([o[k+'_counts'] for k in ('draw', 'discard', 'exhaust')], 1)
        piles = counts @ self.card_base + o['pile_upgraded_counts'] @ self.card_delta
        piles = piles + o['pile_upgrade_sums'].sum(2, keepdim=True)*self.core.combat.summaries.upgrade_projection
        relics = self.core.combat.summaries.relic_terms(o['relics'], o['relic_counters']).sum(1)
        summary = CombatSummary(piles, counts[:, :, 1:].sum(2, keepdim=True), relics)
        return self.core.combat_with_context(o, context, summary)

    @torch.no_grad()
    def score(self, raw, owners, rows, *, anchor=False):
        if tuple((id(p), p._version) for p in self.core.parameters()) != self.versions:
            raise RuntimeError('resident search weights changed during collection')
        rows = torch.as_tensor(rows, device=self.device)
        owner = owners[rows]
        batch = {'raw:'+k: v[rows] for k, v in raw.items()}
        batch.update({'root:'+k: v[owner] for k, v in self.root.items()})
        context = (self.context.anchor if anchor else self.context.combat)[owner].clone()
        # Permanent growth is rare; it owns a refreshed projection. No root or
        # sibling cache is mutated, and every card slot remains in its key.
        misc = batch['raw:permanent_misc']
        original = batch['root:outer_deck_misc']
        projected = torch.where((misc >= 0) & (batch['root:deck_ids'] != 0), misc/MISC_SCALE, original)
        changed = (projected != original).any(1).nonzero().flatten()
        if len(changed):
            obs = {k[5:]: v[changed] for k, v in batch.items() if k.startswith('root:')}
            obs['outer_deck_misc'] = projected[changed]
            keys = torch.cat((owner[changed, None], misc[changed]), 1).cpu().numpy()
            for j, key in enumerate(keys):
                key = key.tobytes()
                if key not in self.changed_context:
                    new = self.core.prepare_search_context({k: v[j:j+1] for k, v in obs.items()})
                    self.changed_context[key] = (new.combat[0].clone(), new.anchor[0].clone())
                    self.refreshes += 1
                context[changed[j]] = self.changed_context[key][int(anchor)]
        batch['context'] = context
        if anchor:
            obs, relics = self._forward(batch, 1, True)
            inputs = {k: obs[k].cpu().numpy() for k in ('gold', 'run_keys', 'potions', 'combat_exit_anchor')}
            inputs.update(hp=obs['player_scalars'][:, :2].cpu().numpy(),
                          context=context.cpu().numpy(), relics=relics.cpu().numpy())
            def forward(host):
                x = {k: torch.as_tensor(v, device=self.device) for k, v in host.items()}
                x['player_scalars'] = x.pop('hp')
                values = self.core.anchor_with_context(x, x.pop('context'), x.pop('relics'))
                return np.empty((len(values), 0), np.float32), values.flatten().cpu().numpy()
            _, values = self.anchor_cache.forward(inputs, forward)
            return None, torch.as_tensor(values, device=self.device)
        required = (batch['raw:categorical'][:, 127:223] > 0) | batch['raw:mask'][:, 91:187].bool()
        columns = required.any(0).nonzero().flatten()
        width = min(96, max(1, int(columns[-1])+1)) if len(columns) else 1
        # Only masked trailing padding changes physical shape; no candidate cap.
        width = min(96, 1 << (width-1).bit_length())
        padded = max(16, 1 << (len(rows)-1).bit_length())
        outputs = self.runtime._replay(self.runtime.entries, ('resident', width, padded), batch,
            padded, lambda x: self._forward(x, width, False), self.runtime.capacity)
        return outputs[0][:len(rows)], outputs[1][:len(rows)].flatten()

    def close(self):
        self.runtime.close()
        self.anchor_cache.clear()
