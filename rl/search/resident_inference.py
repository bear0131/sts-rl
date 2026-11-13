"""Legacy actor/fallback RPCs on the existing single CUDA search owner.

Only the deck context and current actor/value heads adapt the old messages.
The collector never waits for a whole vector or creates another search model.
"""
from types import SimpleNamespace
import numpy as np
import torch
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from env.search.anchored_chains import ChainRequest
from rl.policies.collection_graphs import SharedCollectionGraphs
from rl.policies.deck_latent import SearchContext
from rl.search.deck_latent import DeckLatentScorer, DeckSearchInferenceCache
from rl.training import reward_monitor, campfire_monitor


class ResidentInference:
    def __init__(self, core, cache_bytes):
        self.core = core
        self.policy = SimpleNamespace(core=core, training=False, parameters=core.parameters)
        self.graphs = SharedCollectionGraphs(core)
        self.cache = DeckSearchInferenceCache(core, cache_bytes)
        self.jobs = {}

    @torch.no_grad()
    def actor(self, payload):
        obs, masks, slots, capacity = payload
        tensor = {k: torch.as_tensor(v, device='cuda') for k, v in obs.items()}
        self.graphs.bind_host_observations(tensor, obs, context_slots=(slots, capacity))
        logits, values = self.graphs(tensor)
        dist = MaskableCategoricalDistribution(masks.shape[1]).proba_distribution(action_logits=logits.float())
        dist.apply_masking(masks)
        context = self.core.actor_search_context(torch.arange(len(masks), device='cuda'))
        return dict(probabilities=dist.distribution.probs.cpu().numpy(),
                    logprobs=dist.distribution.logits.cpu().numpy(), values=values.flatten().cpu().numpy(),
                    context=tuple(v.cpu().numpy() for v in (context.combat, context.anchor)),
                    reward=reward_monitor.policy_probabilities(dist, obs),
                    campfire=campfire_monitor.policy_probabilities(dist, obs))

    @torch.no_grad()
    def fallback_begin(self, payload):
        key, roots, context = payload
        if key in self.jobs: raise ValueError('duplicate native fallback')
        context = SearchContext(*(torch.as_tensor(v, device='cuda') for v in context))
        scorer = DeckLatentScorer(self.policy, roots, context=context, inference_cache=self.cache)
        self.jobs[key] = scorer, {}

    @torch.no_grad()
    def fallback(self, payload):
        key, rows = payload
        scorer, caches = self.jobs[key]
        requests = [ChainRequest(root, dict(scorer.roots[root], **delta), mask, kind,
                                 caches.setdefault(lane, {}), lane)
                    for root, lane, kind, mask, delta in rows]
        return scorer(requests)

    def fallback_end(self, key):
        scorer, _ = self.jobs.pop(key)
        # Shared-cache totals are reported once at park, not once per job.
        return dict(context_refreshes=scorer.context_refreshes)

    def close(self, cancel=False):
        if self.jobs and not cancel: raise RuntimeError('inference has unfinished native searches')
        self.jobs.clear()
        stats = dict(graphs=dict(self.graphs.stats), cache=self.cache.statistics())
        self.graphs.close()
        self.cache.clear()
        return stats
