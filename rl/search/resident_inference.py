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
    def __init__(self, core, cache_bytes, native_host):
        self.core = core
        self.native_host = native_host
        self.policy = SimpleNamespace(core=core, training=False, parameters=core.parameters)
        self.graphs = SharedCollectionGraphs(core)
        self.cache = DeckSearchInferenceCache(core, cache_bytes)
        self.jobs = {}

    @torch.no_grad()
    def actor(self, payload, *, context=False, values_only=False):
        obs, masks, slots, capacity = payload
        from env.search.cuda_battle import upload_arrays
        groups = {}
        for key, value in obs.items():
            groups.setdefault(value.dtype.str, []).append(key)
        # Concatenate in native NumPy before one upload, avoiding one eager
        # Torch copy per field and a second bank assembly on the device.
        banks = [np.concatenate([obs[k].reshape(len(masks), -1) for k in names], axis=1)
                 for names in groups.values()]
        uploaded = upload_arrays(self.native_host, 'cuda', banks)
        tensor, packed = {}, {}
        for group, (names, bank) in enumerate(zip(groups.values(), uploaded, strict=True)):
            packed['packed'+str(group)] = bank
            offset = 0
            for key in names:
                shape = obs[key].shape[1:]
                end = offset + int(np.prod(shape))
                tensor[key] = bank[:, offset:end].reshape(len(masks), *shape)
                offset = end
        self.graphs.bind_host_observations(tensor, obs, context_slots=(slots, capacity), packed=packed)
        logits, values = self.graphs(tensor)
        if values_only:
            return values.flatten().cpu().numpy()
        dist = MaskableCategoricalDistribution(masks.shape[1]).proba_distribution(action_logits=logits.float())
        dist.apply_masking(masks)
        width = masks.shape[1]
        packet = torch.cat((dist.distribution.probs, dist.distribution.logits,
                            values.reshape(-1, 1)), dim=1).cpu().numpy()
        result = dict(probabilities=packet[:, :width], logprobs=packet[:, width:2*width],
                      values=packet[:, -1], reward=reward_monitor.policy_probabilities(dist, obs),
                      campfire=campfire_monitor.policy_probabilities(dist, obs))
        if context:
            projected = self.core.actor_search_context(torch.arange(len(masks), device='cuda'))
            # This owner admits the root immediately. Its frozen context never
            # has to roundtrip through the collector or host observation codec.
            result['context'] = projected.combat, projected.anchor
        return result

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
