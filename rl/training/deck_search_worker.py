"""Frozen CPU search adapter in an existing native environment worker.

The shared mmap snapshot is replaced only between collections. The model-free
search itself belongs to env.search; this module owns inference and transport.
"""
from types import SimpleNamespace

import numpy as np
import torch

from env.search.anchored_chains import paired_anchor_search
from env.core.native_observation import NativeRootObservationBatch
from rl.policies.deck_latent import DeckLatentCore, SearchContext
from rl.search.deck_latent import DeckLatentScorer, DeckSearchInferenceCache


def search_options(config):
    return dict(max_depth=config.max_depth, max_active=config.max_active,
                timeout_seconds=config.timeout_seconds)


class DeckSearchWorker:
    def __init__(self, envs):
        self.envs, self.generation, self.policy = envs, None, None
        self.observation_batch = NativeRootObservationBatch(envs[0].unwrapped)

    def run(self, request):
        from rl.algorithms.search_auxiliary import SearchTrainingConfig
        config = SearchTrainingConfig(**request['config'])
        if self.generation != request['generation']:
            torch.set_num_threads(1)
            data = torch.load(request['snapshot'], mmap=True, weights_only=True, map_location='cpu')
            with torch.device('meta'):
                core = DeckLatentCore(self.envs[0].observation_space, **data['geometry'])
            core.load_state_dict(data['state_dict'], assign=True)
            # Nonpersistent lookup indices/HP thresholds are absent from a
            # state_dict, but still need concrete storage after meta creation.
            for name, value in data['buffers'].items():
                parent, _, field = name.rpartition('.')
                core.get_submodule(parent)._buffers[field] = value
            if any(x.is_meta for x in (*core.parameters(), *core.buffers())):
                raise RuntimeError('frozen worker snapshot left a meta tensor unresolved')
            core.eval()
            self.policy = SimpleNamespace(core=core, training=False, parameters=core.parameters)
            self.inference_cache = DeckSearchInferenceCache(core, config.inference_cache_bytes)
            self.tables = core.combat.summaries.export_inference_tables()
            self.generation = request['generation']
        roots = [self.envs[i].unwrapped for i in request['indices']]
        observations = [e._obs() for e in roots]
        context = SearchContext(*(torch.from_numpy(v) for v in request['context']))
        scorer = DeckLatentScorer(self.policy, observations, context=context, tables=self.tables,
                                 inference_cache=self.inference_cache)
        result, stats = paired_anchor_search(roots, request['budgets'], scorer,
            seed=request['seed'], observation_batch=self.observation_batch, **search_options(config))
        stats['context_refreshes'] = scorer.context_refreshes
        stats.update(scorer.cache_statistics())
        return result, stats
