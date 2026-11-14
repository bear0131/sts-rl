"""Legacy actor/fallback RPCs on the existing single CUDA search owner.

Only the deck context and current actor/value heads adapt the old messages.
The collector never waits for a whole vector or creates another search model.
"""
from types import SimpleNamespace
from collections.abc import Mapping
import numpy as np
import torch
from env.search.anchored_chains import ChainRequest
from rl.policies.collection_graphs import SharedCollectionGraphs
from rl.policies.deck_latent import SearchContext
from rl.search.deck_latent import DeckLatentScorer, DeckSearchInferenceCache
from rl.training import reward_monitor, campfire_monitor


class _PackedViews(Mapping):
    """Only materialize CUDA views actually consumed outside captured graphs."""
    def __init__(self, layout, banks, count):
        self.layout, self.banks, self.count, self.views = layout, banks, count, {}

    def __len__(self): return len(self.layout)
    def __iter__(self): return iter(self.layout)
    def __getitem__(self, key):
        if key not in self.views:
            group, lo, hi, shape = self.layout[key]
            self.views[key] = self.banks[group][:,lo:hi].reshape(self.count,*shape)
        return self.views[key]


class ResidentInference:
    def __init__(self, core, cache_bytes, native_host):
        self.core = core
        self.native_host = native_host
        self.policy = SimpleNamespace(core=core, training=False, parameters=core.parameters)
        self.graphs = SharedCollectionGraphs(core, capacity=256)
        self.cache = DeckSearchInferenceCache(core, cache_bytes)
        self.jobs = {}
        self.geometry = None

    @torch.no_grad()
    def actor(self, payload, *, context=False, values_only=False):
        obs, masks, slots, capacity = payload
        from env.search.cuda_battle import upload_arrays
        geometry = tuple((k,v.shape[1:],v.dtype.str) for k,v in obs.items())
        if geometry != self.geometry:
            groups, layout = {}, {}
            for key, value in obs.items():
                groups.setdefault(value.dtype.str, []).append(key)
            for group, names in enumerate(groups.values()):
                offset = 0
                for key in names:
                    shape = obs[key].shape[1:]
                    end = offset+int(np.prod(shape))
                    layout[key] = 'packed'+str(group), offset, end, shape
                    offset = end
            self.geometry, self.groups = geometry, groups
            self.layout = {k:layout[k] for k in obs}
        # Concatenate in native NumPy before one upload, avoiding one eager
        # Torch copy per field and a second bank assembly on the device.
        banks = [np.concatenate([obs[k].reshape(len(masks), -1) for k in names], axis=1)
                 for names in self.groups.values()]
        uploaded = upload_arrays(self.native_host, 'cuda', [*banks, np.asarray(masks,dtype=np.uint8)])
        packed = {'packed'+str(j):v for j,v in enumerate(uploaded[:-1])}
        tensor = _PackedViews(self.layout, packed, len(masks))
        if values_only:
            # Only context/map consumers request views; binding shape checks
            # are unnecessary here because the schema built the exact banks.
            self.graphs.host_inputs = tensor, obs, (slots, capacity), packed, self.layout
            _, values = self.graphs(tensor)
            return values.flatten().cpu().numpy()
        width = masks.shape[1]
        packet = self.graphs.host_actor(tensor, obs, (slots,capacity), packed, uploaded[-1], layout=self.layout).cpu().numpy()
        # Monitoring consumes the already published probabilities on the CPU,
        # without another device gather/copy for each reward or campfire batch.
        dist = SimpleNamespace(distribution=SimpleNamespace(probs=torch.from_numpy(packet[:,:width])))
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
