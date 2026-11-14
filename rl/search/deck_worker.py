"""Current model/state consumer for the unchanged legacy worker schedule.

Scheduling lives in resident_worker (stepfull480-01). This module translates
the retained checkpoint's context, critic and anchor target at its boundary.
"""
import gc
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from env.core.run_env import _nosl_compact_lut
from env.search.cuda_battle import CudaBattleArena, load_artifacts
from rl.policies.deck_latent import DeckLatentCore
from rl.search.resident_deck import ResidentDeckScorer
from rl.search.resident_pool import ResidentPool
from rl.search.resident_step import ResidentStepPipeline
from rl.search.resident_inference import ResidentInference


class DeckWorker:
    def __init__(self, settings):
        self.settings = dict(settings)
        self.settings.setdefault('environment_capacity', settings['capacity'])
        self.core = self.engine = self.arena = self.scorer = self.inference = None
        self.compiled = None
        self.queued, self.estimates, self.actor_queue = {}, {}, {}
        self.actor_batches = []
        self.parked_report = None
        self.generation = 0
        self.folder, self.manifest, self.host = load_artifacts(settings['artifacts'])
        if getattr(self.host, 'host_runtime_version', None) != 2:
            raise ValueError('deck state adapter requires native host runtime v2')
        self.begin(settings['snapshot'])

    def begin(self, snapshot):
        if self.engine is not None or self.queued or self.estimates or self.actor_queue:
            raise RuntimeError('weights may change only at a completely drained boundary')
        data = torch.load(snapshot, map_location='cpu', weights_only=False, mmap=True)
        if self.core is None:
            self.geometry, self.space = data['geometry'], data['space']
            with torch.device('meta'):
                self.core = DeckLatentCore(self.space, **self.geometry)
            self.core.load_state_dict(data['state_dict'], assign=True)
            for name, value in data['buffers'].items():
                parent, _, field = name.rpartition('.')
                self.core.get_submodule(parent)._buffers[field] = value
            self.core.to('cuda').eval()
        else:
            if data['geometry'] != self.geometry or data['space'] != self.space:
                raise ValueError('frozen checkpoint geometry changed')
            self.core.load_state_dict(data['state_dict'], assign=False)
            buffers = dict(self.core.named_buffers())
            if buffers.keys() != data['buffers'].keys():
                raise ValueError('frozen buffer schema changed')
            for name, value in data['buffers'].items():
                buffers[name].copy_(value)
        policy = SimpleNamespace(core=self.core, observation_space=self.space,
                                 parameters=self.core.parameters)
        if self.scorer is None:
            self.scorer = ResidentDeckScorer(policy, self.settings['cache_bytes'], native_host=self.host)
        else:
            # Dynamo's callable is bound to this scorer identity. Reinitialize
            # its drained state without leaving it bound to a retired object.
            self.scorer.__init__(policy, self.settings['cache_bytes'],
                                 native_host=self.host, compiled_gather=self.compiled)
        capacity, budget = self.settings['capacity'], self.settings['budget_cap']
        mirrors = self.settings['environment_capacity']
        lanes = capacity * ((budget + 15)//16)*16
        # Physical frames must accommodate the live pool, including boss roots.
        # The ordinary-root budget is not a bound on concurrently live chains.
        frame = min(ResidentStepPipeline.frame_capacity, lanes)
        self.arena = CudaBattleArena(self.folder/'deck.cubin',
            self.manifest['sha256']['deck.cubin'], capacity+lanes+mirrors,
            card_count=_nosl_compact_lut()['layout']['card_count'], paged=True,
            observation_capacity=min(capacity+lanes, frame))
        self.engine = ResidentPool(self.arena, self.scorer, capacity, budget,
            self.settings['max_depth'], self.settings['timeout'], mirror_capacity=mirrors)
        self.inference = ResidentInference(self.core, self.settings['cache_bytes'], self.host)
        self.boundary_seconds = dict(actor=0., admission=0., search=0.)
        self.generation += 1
        self.started = time.monotonic()
        return dict(pid=os.getpid(), generation=self.generation)

    @property
    def pending(self):
        return bool(self.actor_queue or self.queued or (self.engine is not None and self.engine.pending))

    def submit(self, rows):
        from env.search.resident_transport import unpack_actor_roots
        rows, obs = unpack_actor_roots(rows, row_observations=False, packed=True)
        if self.engine is None or (self.estimates.keys() | self.actor_queue.keys()) & rows.keys():
            raise RuntimeError('parked owner or duplicate search')
        if all('context' in row and 'slot' not in row for row in rows.values()):
            # Read compatibility for retained shadow-training checkpoints. The
            # same legacy owner executes already prepared roots without actor RPCs.
            for j, row in enumerate(rows.values()):
                row['observation'] = {k: v[j] for k,v in obs.items()}
            self.queued.update(rows)
            self.estimates.update(dict.fromkeys(rows))
            return
        # Legacy submit only enqueues. Adapt all roots accumulated by the FIFO
        # drain together immediately before its next tick, not once per packet.
        self.actor_queue.update(rows)
        self.actor_batches.append(obs)

    def _prepare_roots(self, upper=None):
        rows, self.actor_queue = self.actor_queue, {}
        if not rows:
            return self.inference.actor(upper) if upper is not None else None
        keys = list(rows)
        batches, self.actor_batches = self.actor_batches, []
        from env.search.resident_transport import PackedObservationBatch
        obs = (batches[0].concatenate(batches[1:]) if isinstance(batches[0],PackedObservationBatch)
               else batches[0] if len(batches) == 1 else
               {k: np.concatenate([b[k] for b in batches]) for k in batches[0]})
        masks = np.stack([r['mask'] for r in rows.values()])
        slots = np.asarray([r['slot'] for r in rows.values()], np.intp)
        if upper is not None:
            upper_obs, upper_masks, upper_slots, capacity = upper
            if capacity != self.settings['environment_capacity']:
                raise ValueError('actor batch environment capacity changed')
            obs = (obs.concatenate([upper_obs]) if isinstance(obs,PackedObservationBatch) else
                   {k: np.concatenate((v, upper_obs[k]), axis=0) for k, v in obs.items()})
            masks = np.concatenate((masks, upper_masks), axis=0)
            slots = np.concatenate((slots, upper_slots), axis=0)
        actor = self.inference.actor((obs, masks, slots, self.settings['environment_capacity']), context=True)
        fields = ResidentDeckScorer.root_fields(self.core)
        for j, key in enumerate(keys):
            row = rows[key]
            self.estimates[key] = dict(logits=actor['logprobs'][j],
                probabilities=actor['probabilities'][j], value=actor['values'][j])
            self.queued[key] = dict(root=row['root'],
                observation={k: obs[k][j] for k in fields},
                context=tuple(v[j] for v in actor['context']),
                anchor_context_host=actor['anchor_context_host'][j],
                worlds=row['worlds'], budget=row['budget'], submitted_at=row['submitted_at'])
        if upper is not None:
            count = len(keys)
            result = {k: actor[k][count:] for k in ('probabilities', 'logprobs', 'values')}
            for name in ('reward', 'campfire'):
                indices, values = actor[name]
                selected = indices >= count
                result[name] = indices[selected]-count, values[selected]
            return result

    def receive(self, steps=1):
        start = time.perf_counter()
        self._prepare_roots()
        now = time.perf_counter()
        self.boundary_seconds['actor'] += now-start
        start = now
        if self.queued:
            accepted = self.engine.admit(self.queued)
            for key in accepted:
                del self.queued[key]
        now = time.perf_counter()
        self.boundary_seconds['admission'] += now-start
        ready = self.engine.receive(steps=steps) if self.engine.pending else {}
        self.boundary_seconds['search'] += time.perf_counter()-now
        for key, result in ready.items():
            actor = self.estimates.pop(key)
            if actor is not None:
                result['actor'] = actor
        return ready

    def infer(self, kind, payload):
        if self.engine is None:
            raise RuntimeError('inference on parked weights')
        if kind == 'upper':
            return self._prepare_roots(payload)
        if kind == 'values':
            observations, masks, slots = payload
            return self.inference.actor((observations, masks, slots,
                max(len(masks), self.settings['environment_capacity'])), values_only=True)
        if kind == 'fallback_begin':
            key, observations, masks, slots, capacity = payload
            actor = self.inference.actor((observations, masks, slots, capacity), context=True)
            roots = [{k:v[j] for k,v in observations.items()} for j in range(len(masks))]
            self.inference.fallback_begin((key, roots, actor.pop('context')))
            scorer, _ = self.inference.jobs[key]
            return dict(actor=actor, fields=scorer.observation_keys, static=scorer.context_keys)
        return getattr(self.inference, kind)(payload)

    def statistics(self):
        if self.engine is None:
            return self.parked_report
        return dict(pid=os.getpid(), generation=self.generation,
            elapsed_seconds=time.monotonic()-self.started, queued_roots=len(self.queued),
            boundary_seconds=dict(self.boundary_seconds),
            pool=self.engine.statistics(), context_refreshes=self.scorer.refreshes,
            anchor_cache=dict(self.scorer.anchor_cache.stats),
            torch_allocated=torch.cuda.memory_allocated(),
            torch_reserved=torch.cuda.memory_reserved(), torch_peak=torch.cuda.max_memory_allocated())

    def park(self):
        if self.pending or self.estimates:
            raise RuntimeError('park requires completed roots and consumed estimates')
        report = self.statistics()
        report['inference'] = self.inference.close()
        self.inference = None
        self.compiled = self.scorer.compiled_gather
        self.engine.close()
        self.scorer.__dict__.clear()
        self.engine = self.arena = None
        self.core._context_cache = self.core._actor_context = None
        self.core.combat._card_inference_tables.clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        report['park'] = dict(allocated=torch.cuda.memory_allocated(),
                             reserved=torch.cuda.memory_reserved(),
                             free_bytes=torch.cuda.mem_get_info()[0])
        self.parked_report = report
        return report

    def close(self, cancel=False):
        cancelled = list(self.estimates)
        if self.engine is not None:
            if cancel:
                report = self.statistics()
                self.inference.close(cancel=True)
                self.engine.close(cancel=True)
                self.engine = self.arena = self.scorer = self.inference = None
            else:
                report = self.park()
        else:
            report = self.parked_report
        self.queued.clear()
        self.actor_queue.clear()
        self.actor_batches.clear()
        self.estimates.clear()
        return dict(report or {}, cancelled=cancelled)
