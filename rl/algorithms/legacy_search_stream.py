"""Frozen current-training consumer of the accepted ready-root Stream contract."""
from pathlib import Path
import gc
import time
import numpy as np
import torch

from env.search.action_selection import search_filtered_choice
from env.search.resident_transport import ResidentRootExporter
from rl.algorithms.search_auxiliary import SearchTrainingConfig, improved_prior
from rl.search.resident_service import collection_service
from rl.search.deck_latent import DECK_SEARCH_CACHE_METRICS


class Stream:
    def __init__(self, model, vec, buffer, positions, seeds):
        self.model, self.vec, self.buffer = model, vec, buffer
        self.positions, self.seeds = positions, seeds
        self.config = SearchTrainingConfig(**model.deck_search_config)
        self.meta, self.queue, self.inline = {}, [], {}
        self.target_batches = []
        self.fallback = None
        self.exporter = ResidentRootExporter(vec.envs, self.config.cuda_artifacts,
                                             observations=vec.buf_obs)
        from env.search.resident_transport import ObservationColumnTransport
        self.observation_transport = ObservationColumnTransport(vec.buf_obs)
        self.stats = dict.fromkeys(('chains','atomic_steps','policy_rows','forced_rows',
            'anchor_rows','cutoff_rows','terminal_rows','max_active','budget_floor_overage',
            'searched_roots','selected_roots','eligible_roots','executed_search_actions',
            'changed_policy_actions','native_fallback_roots','native_fallback_seconds',
            'context_refreshes','search_seconds','worker_search_seconds',
            'resident_frames','resident_frame_roots_max'), 0.)
        self.stats.update(dict.fromkeys(DECK_SEARCH_CACHE_METRICS, 0.))
        self.stats.update(backend_cuda_inference=1., backend_cuda_simulation=1.,
                          frontier_batches=0.,frontier_rows=0.,scoring_seconds=0.,transport_bytes=0.)
        self.snapshot = Path(model.deck_search_snapshot_dir)/'legacy-frozen.pt'
        self.snapshot.parent.mkdir(parents=True, exist_ok=True)
        cfg = model.policy._core_cfg
        torch.save(dict(space=model.policy.observation_space,
            geometry=dict(width=cfg['d_model'], layers=cfg['n_layers'], heads=cfg['n_heads'],
                          combat_encoder=model.policy.deck_combat_encoder),
            state_dict={k:v.detach().cpu() for k,v in model.policy.core.state_dict().items()},
            buffers={k:v.detach().cpu() for k,v in model.policy.core.named_buffers()}), self.snapshot)
        graph = model.__dict__.pop('_shared_collection_graphs', None)
        if graph is not None:
            graph.close()
        model.policy.core._context_cache = model.policy.core._actor_context = None
        model.policy.core.combat._card_inference_tables.clear()
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
        capacity = vec.num_envs  # Old fixed real-root slots, no extra queued root generations.
        self.service = collection_service(model, capacity, self.snapshot, dict(
            artifacts=self.config.cuda_artifacts, cache_bytes=self.config.inference_cache_bytes,
            normal_budget=self.config.normal_budget,
            budget_cap=max(self.config.boss_budget, self.config.normal_budget,
                           model.policy.observation_space['combat_legal'].shape[0]),
            max_depth=self.config.max_depth, timeout=self.config.timeout_seconds))
        self.generation = self.service.status['generation']
        self.closed = False
        self.started = time.monotonic()

    @property
    def pending(self):
        return bool(self.meta or self.queue or self.fallback is not None or self.inline)

    def submit(self, calls):
        if not calls:
            return
        indices = list(calls)
        roots = self.exporter.export(indices, generation=self.generation)
        rows = {}
        for i, root in zip(indices, roots, strict=True):
            position = int(self.positions[i])
            key = position, i
            if key in self.meta:
                raise RuntimeError('real root already in flight')
            mask = calls[i]
            budget = (self.config.boss_budget if self.vec.buf_obs['combat_is_boss'][i].item()
                      else self.config.normal_budget)
            seed = int(self.seeds[position, i])
            worlds = np.random.default_rng(seed).integers(0, 2**63-1,
                max(1, budget//len(root['actions'])), dtype=np.int64)
            row = dict(root=root, mask=mask, worlds=worlds,
                       budget=budget, slot=i, submitted_at=time.monotonic())
            self.meta[key] = dict(slot=i, mask=mask, seed=seed, position=position, row=row)
            if root.get('native_fallback'):
                row['observation'] = {k:v[i] for k,v in self.vec.buf_obs.items()}
                self.queue.append(key)
                self.stats['native_fallback_roots'] += 1
            else:
                rows[key] = row
            self.stats['selected_roots'] += 1
        if rows:
            self.service.submit(rows, columns=self.observation_transport.pack([key[1] for key in rows]))
            self.stats['resident_frames'] += 1
            self.stats['resident_frame_roots_max'] = max(
                self.stats['resident_frame_roots_max'], len(self.service.outstanding))

    def receive(self, timeout=0.):
        out, self.inline = self.inline, {}
        ready = self.service.receive(.001 if timeout and not out else 0.)
        if self.fallback is None and self.queue:
            from env.search.anchor_wave import CpuFallback
            keys, self.queue = self.queue, []
            self.fallback = CpuFallback(keys, self.meta, self.vec.envs, self.service,
                self.config, threads=self.vec.native_threads)
        if self.fallback is not None:
            complete = self.fallback.step()
            if self.fallback.complete:
                ready.update(complete)
                for key, value in self.fallback.stats.items():
                    if key == 'seconds':
                        self.stats['native_fallback_seconds'] += value
                    elif key in self.stats:
                        if key == 'max_active':
                            self.stats[key] = max(self.stats[key], value)
                        else:
                            self.stats[key] += value
                self.fallback.close()
                self.fallback = None
        if ready:
            keys = list(ready)
            logits = np.stack([r['actor']['logits'] for r in ready.values()])
            masks = np.stack([self.meta[k]['mask'] for k in keys])
            values = np.zeros_like(logits, dtype=np.float32)
            worlds = np.empty(len(keys), np.int64)
            for j, (key, result) in enumerate(ready.items()):
                values[j, result['actions']] = result['values']
                worlds[j] = result['visits'][0]
                item = self.meta.pop(key)
                chosen, log_prob = search_filtered_choice(logits[j,result['actions']],
                    np.asarray(result['values'], np.float64), result['visits'],
                    item['seed'] ^ (item['slot']*1_000_003))
                action = int(result['actions'][chosen])
                self.stats['changed_policy_actions'] += int(action != int(np.argmax(logits[j])))
                self.buffer.search_actions[key] = action, log_prob
                out[item['slot']] = dict(action=action, value=float(result['actor']['value']),
                    log_prob=log_prob, reward=None, campfire=None)
            # The auxiliary target has no acting consumer. Preserve the exact
            # frozen inputs and evaluate independent rows together at the
            # collection boundary, instead of launching tiny Torch reductions
            # between every pair of real environment transitions.
            self.target_batches.append((keys, logits, values, masks, worlds))
            self.stats['searched_roots'] += len(ready)
            self.stats['executed_search_actions'] += len(ready)
        return out

    def before(self, actions):
        self.exporter.before_step(actions.values(), indices=actions.keys())

    def finish(self):
        if self.pending or self.service.outstanding:
            raise RuntimeError('collection boundary has unfinished search')
        report = self.service.park()
        for start in range(0, len(self.target_batches), 128):
            batches = self.target_batches[start:start+128]
            keys = [key for batch in batches for key in batch[0]]
            inputs = [torch.from_numpy(np.concatenate([b[i] for b in batches]))
                      for i in range(1, 5)]
            targets = improved_prior(*inputs, self.config.target_std).numpy()
            for (position, slot), target in zip(keys, targets, strict=True):
                self.buffer.add_search_target(slot, target, position=position)
        self.target_batches.clear()
        self.owner_report = report
        for key in ('chains','atomic_steps','policy_rows','forced_rows','anchor_rows',
                    'cutoff_rows','terminal_rows','budget_floor_overage'):
            self.stats[key] += report['pool'].get(key, 0)
        self.stats['max_active'] = max(self.stats['max_active'], report['pool'].get('max_active', 0))
        self.stats['context_refreshes'] += report.get('context_refreshes', 0)
        owner = report['owner_counts']
        self.stats['worker_search_seconds'] = owner.get('gpu_service_seconds', 0.)
        for key in ('submit','upper','fallback_begin','fallback','fallback_end','values','park'):
            self.model.logger.record('pure_rl/legacy_owner_seconds_'+key,owner.get('seconds_'+key,0.))
        for key in ('torch_allocated','torch_reserved','torch_peak'):
            self.model.logger.record('pure_rl/legacy_owner_'+key,report[key])
        self.stats.update(report['inference']['cache'])
        for key,value in report.get('anchor_cache', {}).items():
            name = 'anchor_cache_'+key
            if name in self.stats:
                self.stats[name] += value
        for key,value in report['inference']['graphs'].items():
            self.model.logger.record('pure_rl/collection_graph_'+key,value)
        self.stats['search_seconds'] = time.monotonic()-self.started
        self.model.deck_search_last_stats = self.stats
        self.publish()
        self.closed = True
        return report

    def publish(self):
        for key, value in self.stats.items():
            self.model.logger.record('deck_search/'+key, value)

    def close(self):
        if self.closed:
            return
        if self.fallback is not None:
            self.fallback.close()
        self.service.close(cancel='incomplete collection')
        self.closed = True
