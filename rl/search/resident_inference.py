"""stepfull480 pinned-input graph transport with current policy heads.

The old CombatLeafGraphs owns transfer, capture, replay, output and eviction.
Only the frozen run-context cache and current policy/anchor tensors adapt it.
No second packed device bank or collection-specific phase scheduler is used.
"""
from types import SimpleNamespace
import numpy as np
import torch

from env.core import encode, native_actions as C, vocab
from rl.search.inference import CombatLeafGraphs, active_width
from rl.search.rollout_inference import ExactInputCache
from env.search.anchored_chains import ChainRequest
from rl.policies.deck_latent import SearchContext
from rl.policies.combat_summary import CombatSummary
from rl.policies.shared_wholerun import MERGED_UPPER_PHASE
from rl.policies.split_policy import COMBAT_OBSERVATION_KEYS
from rl.search.deck_latent import DeckLatentScorer, DeckSearchInferenceCache
from rl.training import reward_monitor, campfire_monitor


class _DeckHead:
    """Bind current head arguments to the old static forward interface."""
    def __init__(self, core, kind):
        self.core, self.kind = core, kind

    @property
    def training(self):
        return self.core.training

    def prepare_inference_cache(self):
        self.core.prepare_inference_cache()

    def combat_actor_critic(self, inputs, *, active_relic_width):
        spec = active_relic_width
        if self.kind == 'context':
            deck, relic, packed = spec
            z = self.core.context(inputs, active_widths=(deck, relic), packed_width=packed)
            return z, z.new_zeros(len(z))
        obs = dict(inputs)
        if self.kind == 'fallback':
            anchor, width = spec
            context, relics = obs.pop('_context'), obs.pop('_relic_summary')
            if anchor:
                value = self.core.anchor_with_context(obs, context, relics)
                return value.new_empty((len(value), 0)), value
            summary = CombatSummary(obs.pop('_piles'), obs.pop('_pile_counts'), relics)
            if width is not None:
                obs['_choice_width'] = width
            return self.core.combat_with_context(obs, SearchContext(context, None), summary)
        route, deck, relic, choice, union = spec
        z, mask = obs.pop('_run_context'), obs.pop('_actor_mask')
        for key in ('relics', 'relic_counters'):
            obs[key] = obs[key][:, :relic]
        combat_context = self.core.combat.context_projection(z)
        anchor_context = self.core.anchor.context_projection(z)
        if route == -2:
            out = z.new_zeros(len(z), C.AUTOREGRESSIVE_SIZE)
            value = self.core.anchor_with_context(obs, anchor_context)
        elif route == vocab.PHASE_VOCAB_INDEX['combat']:
            obs['_choice_width'] = choice
            out, value = self.core.combat_with_context(obs, SearchContext(combat_context, anchor_context))
        else:
            for key in ('deck_ids', 'deck_upgrades'):
                obs[key] = obs[key][:, :deck]
            obs['_run_context'] = z
            out, value = self.core.overworld(obs, self.core.upper_tokenizer, route,
                upper_active_widths=((deck, relic, union) if route == MERGED_UPPER_PHASE else (deck, relic)))
        # The current PPO behavior contract uses the same two normalizations as
        # MaskableCategorical. Publishing its outputs is the consumer adapter.
        normalized = out.float() - out.float().logsumexp(-1, keepdim=True)
        masked = normalized.masked_fill(~mask.bool(), -1e8)
        logprob = masked - masked.logsumexp(-1, keepdim=True)
        return torch.cat((logprob.softmax(-1), logprob, combat_context, anchor_context), 1), value


class ResidentInference:
    def __init__(self, core, cache_bytes, native_host):
        self.core = core
        self.policy = SimpleNamespace(core=core, training=False, parameters=core.parameters)
        self.actor_graphs = self._graphs('actor', 128)
        self.context_graphs = self._graphs('context', 64)
        self.fallback_graphs = self._graphs('fallback', 64)
        self.context_cache = ExactInputCache(cache_bytes)
        self.cache = DeckSearchInferenceCache(core, cache_bytes)
        self.versions = tuple((id(p), p._version) for p in core.parameters())
        self.jobs = {}
        self.context_rows = self.map_rows = 0
        self.tables = core.combat.summaries.export_inference_tables()

    def _graphs(self, kind, capacity):
        # Same pinned dtype buffers and captured H2D/forward/D2H as the old
        # upper/fallback service; batch padding never enters sample counts.
        return CombatLeafGraphs(_DeckHead(self.core, kind), 'cuda', amp='off',
            capacity=capacity, batch_multiple=8, compile_forward=False)

    def _context(self, obs):
        versions = tuple((id(p), p._version) for p in self.core.parameters())
        if versions != self.versions:
            if self.jobs:
                raise RuntimeError('weights changed with unfinished search')
            self.context_cache.clear()
            self.cache = DeckSearchInferenceCache(self.core, self.cache.combat.max_bytes+self.cache.anchor.max_bytes)
            self.tables = self.core.combat.summaries.export_inference_tables()
            self.versions = versions
        fields = {k: obs[k] for k in self.core.context.keys}
        def forward(selected):
            deck = active_width(selected['deck_ids'], 1)
            relic = active_width(selected['relics'], 1)
            live = (1+(selected['deck_ids'] != 0).sum(1)+(selected['relics'] != 0).sum(1)
                    +((selected['map_full_rooms'] != 0) | selected['map_full_features'].any(-1)).sum(1))
            packed = max(16, 1 << (int(live.max())-1).bit_length())
            self.context_rows += len(live)
            return self.context_graphs(selected, (deck, relic, packed))
        return self.context_cache.forward(fields, forward)[0]

    @torch.no_grad()
    def _fallback_head(self, inputs, *, anchor, width):
        return self.fallback_graphs(inputs, (anchor, width))

    @torch.no_grad()
    def actor(self, payload, *, context=False, values_only=False):
        obs, masks, slots, capacity = payload
        count, width = masks.shape
        if (len(slots) != count or len(np.unique(slots)) != count
                or np.any(np.asarray(slots) < 0) or np.any(np.asarray(slots) >= capacity)):
            raise ValueError('actor slot contract changed')
        latent = self._context(obs)
        phase = obs['phase'].reshape(-1)
        routes = phase.copy()
        combat = vocab.PHASE_VOCAB_INDEX['combat']
        routes[(phase == combat) & obs['combat_exit_anchor'].any(1)] = -2
        ordinary = (phase != combat) & (phase != vocab.PHASE_VOCAB_INDEX['card_select'])
        if self.core.map_dag_enabled and 'upper_public_context' in obs:
            routes[ordinary] = MERGED_UPPER_PHASE
        packets, values = None, np.empty(count, np.float32)
        for route in np.unique(routes).tolist():
            rows = np.flatnonzero(routes == route)
            is_combat = route in (combat, -2)
            keys = ((COMBAT_OBSERVATION_KEYS | {'pile_upgraded_counts', 'pile_upgrade_sums'})
                if is_combat else obs if route == MERGED_UPPER_PHASE else encode.upper_observation_keys(route, obs))
            batch = {k: obs[k][rows] for k in obs if k in keys}
            batch['_run_context'] = latent[rows]
            batch['_actor_mask'] = masks[rows]
            relic = active_width(obs['relics'][rows], 4)
            deck = None if is_combat else active_width(obs['deck_ids'][rows], 16)
            choice = union = None
            if route == combat:
                required = obs['select_native_indices'][rows] > 0
                pick = C.JOINT_PICK - C.JOINT_END
                required |= obs['combat_legal'][rows, pick:pick+required.shape[1]].astype(bool)
                choice = active_width(required, 1)
            if route == MERGED_UPPER_PHASE:
                union = self.core.overworld.encoder.merged_width(obs, rows, multiple=16)
            if not is_combat:
                self.map_rows += len(rows)
            packet, value = self.actor_graphs(batch, (route, deck, relic, choice, union))
            if packets is None:
                packets = np.empty((count, packet.shape[1]), np.float32)
            packets[rows], values[rows] = packet, value
        if values_only:
            return values
        dist = SimpleNamespace(distribution=SimpleNamespace(probs=torch.from_numpy(packets[:, :width])))
        result = dict(probabilities=packets[:, :width], logprobs=packets[:, width:2*width], values=values,
            reward=reward_monitor.policy_probabilities(dist, obs),
            campfire=campfire_monitor.policy_probabilities(dist, obs))
        if context:
            split = 2*width+self.core.combat.context_projection.out_features
            result['context'] = packets[:, 2*width:split], packets[:, split:]
            result['anchor_context_host'] = packets[:, split:]
        return result

    @torch.no_grad()
    def fallback_begin(self, payload):
        key, roots, context = payload
        if key in self.jobs:
            raise ValueError('duplicate native fallback')
        context = SearchContext(*(torch.as_tensor(v, device='cuda') for v in context))
        scorer = DeckLatentScorer(self.policy, roots, context=context, tables=self.tables,
            inference_cache=self.cache, head_forward=self._fallback_head)
        self.jobs[key] = scorer, {}

    @torch.no_grad()
    def fallback(self, payload):
        key, rows = payload
        scorer, caches = self.jobs[key]
        requests = [ChainRequest(root, dict(scorer.roots[root], **delta), mask, kind,
                                 caches.setdefault((root, lane), {}), lane)
                    for root, lane, kind, mask, delta in rows]
        return scorer(requests)

    def fallback_end(self, key):
        scorer, _ = self.jobs.pop(key)
        return dict(context_refreshes=scorer.context_refreshes)

    def close(self, cancel=False):
        if self.jobs and not cancel:
            raise RuntimeError('inference has unfinished native searches')
        self.jobs.clear()
        caches = self.actor_graphs, self.context_graphs, self.fallback_graphs
        statistics = [c.statistics() for c in caches]
        keys = ('captures', 'hits', 'evictions', 'capture_seconds', 'copy_seconds', 'output_seconds')
        graphs = {k: sum(s[k] for s in statistics) for k in keys}
        graphs.update(context_rows=self.context_rows, map_rows=self.map_rows)
        stats = dict(graphs=graphs, cache=self.cache.statistics())
        for c in caches:
            c.clear()
        self.context_cache.clear()
        self.cache.clear()
        return stats
