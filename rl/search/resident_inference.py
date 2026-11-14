"""Legacy actor/fallback RPCs on the existing single CUDA search owner.

Only the deck context and current actor/value heads adapt the old messages.
The collector never waits for a whole vector or creates another search model.
"""
from types import SimpleNamespace
from collections.abc import Mapping
import numpy as np
import torch
from env.search.resident_transport import PackedObservationBatch
from env.search.anchored_chains import ChainRequest
from rl.policies.collection_graphs import SharedCollectionGraphs
from rl.policies.deck_latent import SearchContext
from rl.policies.combat_summary import CombatSummary
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
        # Native fallback heads and ordinary actor shapes share this owner.
        # Retain their bounded working set instead of repeatedly retiring the
        # actor graphs when a long special-state wave alternates frontiers.
        self.graphs = SharedCollectionGraphs(core, capacity=512)
        self.cache = DeckSearchInferenceCache(core, cache_bytes)
        self.jobs = {}
        self.geometry = None
        self.tables = core.combat.summaries.export_inference_tables()

    @torch.no_grad()
    def _fallback_head(self, inputs, *, anchor, width):
        """Old fallback frame: one upload, captured heads, one owned reply.

        Native lanes and their exact incremental summaries remain authoritative.
        The full run encoder is not a consumer of this scoring frontier.
        """
        from env.search.cuda_battle import upload_arrays
        groups = {}
        for key,value in inputs.items():
            groups.setdefault(value.dtype.str,[]).append(key)
        layout, storage = {}, []
        count = len(next(iter(inputs.values())))
        for group,names in enumerate(groups.values()):
            offset=0
            for key in names:
                shape=inputs[key].shape[1:]
                end=offset+int(np.prod(shape))
                layout[key]=group,offset,end,shape
                offset=end
            storage.append(np.concatenate([inputs[k].reshape(count,-1) for k in names],1))
        uploaded=upload_arrays(self.native_host,'cuda',storage)
        batch={str(i):v for i,v in enumerate(uploaded)}
        padded=max(8,1 << (count-1).bit_length())
        signature=('native-fallback-head',anchor,width,padded,
                   tuple((k,v) for k,v in layout.items()),
                   tuple((k,v.dtype) for k,v in batch.items()))
        def forward(columns):
            tensors={k:columns[str(g)][:,lo:hi].reshape(padded,*shape)
                     for k,(g,lo,hi,shape) in layout.items()}
            context=tensors.pop('_context');relics=tensors.pop('_relic_summary')
            if anchor:
                return (self.core.anchor_with_context(tensors,context,relics).reshape(-1,1),)
            summary=CombatSummary(tensors.pop('_piles'),tensors.pop('_pile_counts'),relics)
            if width is not None:
                tensors['_choice_width']=width
            logits,values=self.core.combat_with_context(tensors,SearchContext(context,None),summary)
            return (torch.cat((logits,values.reshape(-1,1)),1),)
        packed=self.graphs._replay(self.graphs.entries,signature,batch,padded,forward,
                                  self.graphs.capacity)[0][:count].cpu().numpy().copy()
        return (np.empty((count,0),np.float32) if anchor else packed[:,:-1]),packed[:,-1]

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
        if isinstance(obs,PackedObservationBatch):
            # Transport layout is grouped by dtype in the same field order.
            # Its owned arrays can go directly to the device upload.
            banks = [column[2] for column in obs.columns]
        else:
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
        device_packet = self.graphs.host_actor(tensor, obs, (slots,capacity), packed, uploaded[-1], layout=self.layout)
        projected = None
        if context:
            projected = self.core.actor_search_context(torch.arange(len(masks), device='cuda'))
            # The anchor cache needs these exact bytes on the host. Publish
            # them with the actor's existing rendezvous, rather than creating
            # a second synchronization at every root admission.
            combined = torch.cat((device_packet, projected.anchor), 1).cpu().numpy()
            packet = combined[:, :device_packet.shape[1]]
            anchor_host = combined[:, device_packet.shape[1]:]
        else:
            packet = device_packet.cpu().numpy()
        # Monitoring consumes the already published probabilities on the CPU,
        # without another device gather/copy for each reward or campfire batch.
        dist = SimpleNamespace(distribution=SimpleNamespace(probs=torch.from_numpy(packet[:,:width])))
        result = dict(probabilities=packet[:, :width], logprobs=packet[:, width:2*width],
                      values=packet[:, -1], reward=reward_monitor.policy_probabilities(dist, obs),
                      campfire=campfire_monitor.policy_probabilities(dist, obs))
        if context:
            # This owner admits the root immediately. Its frozen context never
            # has to roundtrip through the collector or host observation codec.
            result['context'] = projected.combat, projected.anchor
            result['anchor_context_host'] = anchor_host
        return result

    @torch.no_grad()
    def fallback_begin(self, payload):
        key, roots, context = payload
        if key in self.jobs: raise ValueError('duplicate native fallback')
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
        # Shared-cache totals are reported once at park, not once per job.
        return dict(context_refreshes=scorer.context_refreshes)

    def close(self, cancel=False):
        if self.jobs and not cancel: raise RuntimeError('inference has unfinished native searches')
        self.jobs.clear()
        stats = dict(graphs=dict(self.graphs.stats), cache=self.cache.statistics())
        self.graphs.close()
        self.cache.clear()
        return stats
