"""Frozen paired search: legacy shadow teaching or explicit search-led actions.

Only real executed transitions enter PPO storage. Search-led actions retain
their own behavior probability/source and are excluded from the actor ratio.
"""
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import os
import time

import numpy as np
import torch

from env.core import vocab
from env.search.anchored_chains import paired_anchor_search
from rl.algorithms.search_auxiliary import SearchTrainingConfig, SearchRolloutBuffer, improved_prior
from rl.policies.deck_latent import MODEL_CONTRACT
from rl.search.deck_latent import DeckLatentScorer, DeckSearchInferenceCache, DECK_SEARCH_CACHE_METRICS
from rl.training.deck_search_worker import search_options


class ShadowSearchCollection:
    def __init__(self, model, env, buffer):
        self.model, self.env, self.buffer = model, env, buffer
        self.config = SearchTrainingConfig(**model.deck_search_config)
        self.direct_actions = self.config.action_selection != 'shadow'
        if getattr(model.policy.core, 'model_contract', None) != MODEL_CONTRACT:
            raise ValueError('shadow search is only supported by deck-latent-combat-v1')
        if not isinstance(buffer, SearchRolloutBuffer) or model.ppo_update_config is None:
            raise ValueError('shadow search requires the packed PPO/search buffer lifecycle')
        if model.policy.combat_value_target != 'pre-reward-anchor':
            raise ValueError('search teaching requires the learned combat-exit anchor target')
        if self.config.backend in ('cuda-batched-native', 'cuda-resident'):
            if model.device.type != 'cuda' or not hasattr(env, 'deck_batched_search'):
                raise ValueError('cuda-batched-native requires a CUDA policy and NativeSubprocVecEnv')
        self.rng = np.random.default_rng(self.config.seed)
        state = getattr(model, 'deck_search_rng', None)
        if state is not None:
            self.rng.bit_generator.state = state
        # Identical keys on every rank, including a rank with zero selected
        # roots, are required by the scalar reduction contract.
        self.stats = dict.fromkeys(('searched_roots', 'eligible_roots', 'selected_roots', 'search_seconds',
            'chains', 'atomic_steps', 'policy_rows', 'forced_rows', 'anchor_rows', 'cutoff_rows',
            'terminal_rows', 'max_active', 'budget_floor_overage', 'worker_search_seconds', 'context_refreshes',
            'scoring_seconds', 'worker_and_transport_seconds', 'frontier_batches', 'frontier_rows', 'transport_bytes',
            'backend_cuda_inference', 'backend_cuda_simulation', 'resident_frames',
            'resident_frame_roots_max', 'native_fallback_roots', 'native_fallback_seconds',
            'stream_submitted_roots', 'stream_retired_roots', 'stream_submissions_while_pending',
            'stream_ticks', 'stream_command_bytes', 'stream_peak_pages',
            'stream_page_oom_retries', 'stream_deferred_admissions',
            'executed_search_actions', 'changed_policy_actions',
            *DECK_SEARCH_CACHE_METRICS), 0.)
        self.stats['backend_cuda_inference'] = float(self.config.backend in ('cuda-batched-native', 'cuda-resident'))
        self.stats['backend_cuda_simulation'] = float(self.config.backend == 'cuda-resident')
        self.hook = None
        self.previous = env.step_async
        self.snapshot = None

    def start(self, *, hooks=True):
        self.inference_cache = DeckSearchInferenceCache(self.model.policy.core, self.config.inference_cache_bytes)
        if self.config.backend == 'cuda-resident':
            import gc
            from rl.search.resident_service import collection_service
            from rl.search.resident_deck import ResidentDeckScorer
            self.resident_root_keys = ResidentDeckScorer.root_fields(self.model.policy.core)
            folder = Path(self.model.deck_search_snapshot_dir)
            folder.mkdir(parents=True, exist_ok=True)
            self.snapshot = folder / 'stream-frozen.pt'
            cfg = self.model.policy._core_cfg
            torch.save(dict(space=self.model.policy.observation_space,
                geometry=dict(width=cfg['d_model'],layers=cfg['n_layers'],heads=cfg['n_heads'],
                              combat_encoder=self.model.policy.deck_combat_encoder),
                state_dict={k:v.detach().cpu() for k,v in self.model.policy.core.state_dict().items()},
                buffers={k:v.detach().cpu() for k,v in self.model.policy.core.named_buffers()}),self.snapshot)
            torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
            root_capacity = max(self.env.num_envs,
                (self.config.cuda_wave_capacity+self.config.normal_budget-1)//self.config.normal_budget)
            self.resident_service = collection_service(self.model,root_capacity,self.snapshot,dict(
                artifacts=self.config.cuda_artifacts,cache_bytes=self.config.inference_cache_bytes,
                budget_cap=max(self.config.boss_budget,self.config.normal_budget,
                               self.model.policy.observation_space['combat_legal'].shape[0]),
                max_depth=self.config.max_depth,timeout=self.config.timeout_seconds))
            self.resident_items = {}
        elif self.config.backend == 'cuda-batched-native':
            # Weights stay on the learner device for the entire frozen
            # collection. Export only the small summary tables, once.
            self.tables = self.model.policy.core.combat.summaries.export_inference_tables()
        elif hasattr(self.env, 'deck_shadow_search'):
            folder = Path(self.model.deck_search_snapshot_dir)
            folder.mkdir(parents=True, exist_ok=True)
            self.snapshot = folder / 'frozen.pt'
            cfg = self.model.policy._core_cfg
            temporary = folder / 'frozen.tmp'
            torch.save(dict(geometry=dict(width=cfg['d_model'], layers=cfg['n_layers'], heads=cfg['n_heads'],
                combat_encoder=self.model.policy.deck_combat_encoder),
                state_dict={k: v.detach().cpu() for k, v in self.model.policy.core.state_dict().items()},
                buffers={k: v.detach().cpu() for k, v in self.model.policy.core.named_buffers()}), temporary)
            os.replace(temporary, self.snapshot)
        self.generation = (int(self.model.num_timesteps), time.time_ns())
        if not hooks and self.config.backend == 'cuda-resident':
            from rl.training.native_frontier_stream import NativeFrontierStream
            from stable_baselines3.common.vec_env import VecMonitor
            vec = self.env.venv if isinstance(self.env, VecMonitor) else self.env
            self.native_stream = NativeFrontierStream(vec, self.resident_service,
                self.model.policy.core, self.config, self.generation)
        if hooks:
            self.hook = self.model.policy.register_forward_hook(self.observe, with_kwargs=True)
            self.env.step_async = MethodType(lambda env, actions: self.step(actions), self.env)
        return self

    @torch.no_grad()
    def observe(self, policy, args, kwargs, output):
        obs, masks = self.model._last_obs, np.asarray(kwargs['action_masks'], bool)
        legal = ((obs['phase'].reshape(-1) == vocab.PHASE_VOCAB_INDEX['combat'])
                 & ~obs['combat_exit_anchor'].any(1) & (masks.sum(1) > 1))
        self.stats['eligible_roots'] += int(legal.sum())
        indices = np.flatnonzero(legal & (self.rng.random(len(legal)) < self.config.fraction))
        self.pending = None
        if len(indices):
            rows = torch.as_tensor(indices, device=self.model.device)
            logits = policy.action_dist.distribution.logits[rows].detach().cpu()
            context = policy.core.actor_search_context(rows)
            self.pending = (indices, masks[indices], logits, context)
        if self.direct_actions:
            # Finish selected current roots before SB3 can advance any real
            # environment. Returning changed forward outputs also changes the
            # action and log probability that SB3 records in its rollout.
            self._search_step(None)
            actions, values, log_prob = output
            actions, log_prob = actions.clone(), log_prob.clone()
            chosen = [(i, self.buffer.search_actions[(self.buffer.pos, int(i))])
                      for i in indices if (self.buffer.pos, int(i)) in self.buffer.search_actions]
            if chosen:
                ids = torch.as_tensor([i for i, _ in chosen], device=actions.device)
                picked = torch.as_tensor([v[0] for _, v in chosen], device=actions.device)
                self.stats['changed_policy_actions'] += int((actions.flatten()[ids] != picked).sum())
                actions.flatten()[ids] = picked
                log_prob.flatten()[ids] = torch.as_tensor([v[1] for _, v in chosen], device=log_prob.device)
                self.stats['executed_search_actions'] += len(chosen)
            return actions, values, log_prob

    def step(self, actions):
        if self.direct_actions:
            return self.previous(actions)
        return self._search_step(actions)

    def _search_step(self, actions):
        if self.config.backend == 'cuda-resident':
            return self._resident_step(actions)
        if self.pending is not None:
            started = time.monotonic()
            indices, masks, logits, context = self.pending
            observations = [{k: v[i] for k, v in self.model._last_obs.items()} for i in indices]
            budgets = [self.config.boss_budget if self.model._last_obs['combat_is_boss'][i].item()
                       else self.config.normal_budget for i in indices]
            seed = int(self.rng.integers(0, 2**63 - 1))
            if self.config.backend == 'cuda-batched-native':
                scorer = DeckLatentScorer(self.model.policy, observations, context=context, tables=self.tables,
                    inference_cache=self.inference_cache)
                result, stats = self.env.deck_batched_search(dict(indices=indices.tolist(), budgets=budgets,
                    seed=seed, generation=self.generation, options=search_options(self.config)), scorer)
            elif self.snapshot is not None:
                result, stats = self.env.deck_shadow_search(dict(indices=indices.tolist(), budgets=budgets,
                    context=(context.combat.detach().cpu().numpy(), context.anchor.detach().cpu().numpy()),
                    config=asdict(self.config), seed=seed, generation=self.generation, snapshot=str(self.snapshot)))
            else:
                if not hasattr(self.env, 'envs'):
                    raise ValueError('unsupported shadow search environment transport')
                roots = [self.env.envs[i].unwrapped for i in indices]
                scorer = DeckLatentScorer(self.model.policy, observations, context=context,
                    inference_cache=self.inference_cache)
                result, stats = paired_anchor_search(roots, budgets, scorer, seed=seed, **search_options(self.config))
                stats['context_refreshes'] = scorer.context_refreshes
                stats.update(scorer.cache_statistics())
            for j, (index, row) in enumerate(zip(indices, result, strict=True)):
                if row.get('skipped'):
                    continue
                q = torch.zeros_like(logits[j:j+1])
                q[0, row['actions']] = torch.as_tensor(row['values'], dtype=q.dtype)
                target = improved_prior(logits[j:j+1], q, torch.from_numpy(masks[j:j+1]),
                    torch.tensor([int(row['visits'][0])]), self.config.target_std)[0].numpy()
                self.buffer.add_search_target(int(index), target)
                if self.direct_actions:
                    self._record_action(int(index), self.buffer.pos, row, logits[j].numpy(),
                                        seed ^ (int(index)*1_000_003))
                self.stats['searched_roots'] += 1
            self.stats['selected_roots'] += len(indices)
            self.stats['search_seconds'] += time.monotonic() - started
            for k, v in stats.items():
                k = 'worker_search_seconds' if k == 'seconds' else k
                self.stats[k] = max(self.stats[k], float(v)) if k == 'max_active' else self.stats[k] + float(v)
        return None if self.direct_actions else self.previous(actions)

    def _record_action(self, index, position, row, logits, nonce):
        from env.search.action_selection import search_filtered_choice
        choices = np.asarray(row['actions'])
        chosen, log_prob = search_filtered_choice(logits[choices], np.asarray(row['values'],np.float64),
                                                  row['visits'], nonce)
        self.buffer.search_actions[(position,index)] = int(choices[chosen]), log_prob

    @torch.no_grad()
    def _resident_step(self, actions):
        started = time.monotonic()
        self._receive_resident()
        if self.pending is not None:
            indices, masks, logits, context = self.pending
            observations = [{k: self.model._last_obs[k][i] for k in self.resident_root_keys}
                            for i in indices]
            budgets = [self.config.boss_budget if self.model._last_obs['combat_is_boss'][i].item()
                       else self.config.normal_budget for i in indices]
            independent = getattr(self, 'stream_positions', None) is not None
            seed = 0 if independent else int(self.rng.integers(0, 2**63-1))
            roots, groups = self.env.deck_resident_roots(indices.tolist(), self.config.cuda_artifacts,
                                                       generation=self.generation)
            # A worker exports an origin once per collection, including roots
            # whose whole worker group subsequently takes the native fallback.
            # Keep that origin until its GPU admission actually occurs.
            if not hasattr(self, 'resident_origins'):
                self.resident_origins = {}
            for root in roots:
                identity = root.get('mirror')
                if identity is None:
                    continue
                env_id, serial = identity['env'], identity['serial']
                if root['snapshot'] is not None:
                    self.resident_origins[env_id] = serial, root['snapshot']
                cached_serial, snapshot = self.resident_origins[env_id]
                if cached_serial != serial:
                    raise ValueError('missing resident battle origin')
                root['snapshot'] = snapshot
            item = dict(position=self.buffer.pos, indices=indices.copy(),
                masks=masks.copy(), logits=logits.clone(), context=context, observations=observations,
                roots=roots, groups=groups, budgets=budgets, seed=seed)
            if independent:
                item['positions'] = [int(self.stream_positions[i]) for i in indices]
                item['seeds'] = [int(self.stream_seeds[self.stream_positions[i], i]) for i in indices]
            fallback_groups = {g for g, root in zip(groups, roots) if root.get('native_fallback')}
            if fallback_groups:
                bad = [i for i, g in enumerate(groups)
                       if (roots[i].get('native_fallback') if independent else g in fallback_groups)]
                batches = [[i] for i in bad] if independent else [bad]
                for rows in batches:
                    selected = ShadowSearchCollection._select_resident(item, rows)
                    selected['observations'] = [{k: v[i] for k, v in self.model._last_obs.items()}
                                                for i in selected['indices']]
                    if independent:
                        self.native_stream.submit(selected)
                        key = (selected['positions'][0], int(selected['indices'][0]))
                        self.resident_items[key] = selected
                        self.stats['native_fallback_roots'] += 1
                        continue
                    fallback_start = time.monotonic()
                    scorer = DeckLatentScorer(self.model.policy, selected['observations'],
                        context=selected['context'], inference_cache=self.inference_cache)
                    fallback_seed = selected['seeds'][0] if independent else seed
                    result, stats = self.env.deck_batched_search(dict(indices=selected['indices'].tolist(),
                        budgets=selected['budgets'], seed=fallback_seed, generation=self.generation,
                        options=search_options(self.config)), scorer)
                    self.stats['searched_roots'] += ShadowSearchCollection._store_targets(self, [selected], result)
                    self.stats['native_fallback_roots'] += len(rows)
                    self.stats['native_fallback_seconds'] += time.monotonic()-fallback_start
                    for k,v in stats.items():
                        if k in DECK_SEARCH_CACHE_METRICS:
                            continue
                        k = 'worker_search_seconds' if k == 'seconds' else k
                        self.stats[k] = max(self.stats[k],float(v)) if k == 'max_active' else self.stats[k]+float(v)
                item = ShadowSearchCollection._select_resident(item,
                    [i for i in range(len(groups)) if i not in bad])
            # Paired world draws belong to the original CPU worker group, not
            # to admission batches or newly leased physical pages.
            generators, queued = {}, {}
            host_context = tuple(x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in
                                 (item['context'].combat,item['context'].anchor))
            for j, (root, group, budget) in enumerate(zip(item['roots'],item['groups'],item['budgets'])):
                if root.get('skipped'):
                    continue
                root_seed = item['seeds'][j] if independent else seed
                generator_key = (group, j) if independent else group
                if generator_key not in generators:
                    group_seed = int(np.random.SeedSequence([root_seed,int(group)]).generate_state(1,dtype=np.uint64)[0])
                    generators[generator_key] = np.random.default_rng(group_seed)
                worlds = generators[generator_key].integers(0,2**63-1,max(1,budget//len(root['actions'])),dtype=np.int64)
                position = item['positions'][j] if independent else self.buffer.pos
                key = (position,int(item['indices'][j]))
                queued[key] = dict(root=root,observation={k:v for k,v in item['observations'][j].items()
                                                        if k in self.resident_root_keys},
                    context=tuple(x[j] for x in host_context),worlds=worlds,budget=budget,
                    submitted_at=time.monotonic())
                # The FIFO synchronously serializes the frozen root/context.
                # Only the prior and its original buffer position survive here;
                # do not launch per-root CUDA gathers for discarded context.
                self.resident_items[key] = dict(position=position,
                    indices=item['indices'][j:j+1],masks=item['masks'][j:j+1],
                    logits=item['logits'][j:j+1],roots=[None],seed=root_seed)
            while queued:
                self._receive_resident()
                free = self.resident_service.capacity-len(self.resident_service.outstanding)
                if not free:
                    self._receive_resident(.05)
                    continue
                keys = list(queued)[:free]
                self.resident_service.submit({key:queued.pop(key) for key in keys})
                self.stats['resident_frames'] += 1
                self.stats['resident_frame_roots_max'] = max(self.stats['resident_frame_roots_max'],
                    len(self.resident_service.outstanding))
            self.stats['selected_roots'] += len(indices)
        direct = getattr(self, 'direct_actions', False)
        streaming = getattr(self, 'stream_positions', None) is not None
        if not streaming and (direct or self.buffer.pos == self.buffer.buffer_size-1):
            while self.resident_service.outstanding:
                self._receive_resident(.05)
        if not streaming and self.buffer.pos == self.buffer.buffer_size-1:
            self.resident_service.park()
        self.stats['search_seconds'] += time.monotonic()-started
        return None if direct else self.previous(actions)

    def _receive_resident(self, timeout=0.):
        if hasattr(self, 'native_stream'):
            for item, result, stats in self.native_stream.poll():
                key = (item['positions'][0], int(item['indices'][0]))
                self.resident_items.pop(key)
                self.stats['searched_roots'] += ShadowSearchCollection._store_targets(self, [item], result)
                for k, v in stats.items():
                    k = 'worker_search_seconds' if k == 'seconds' else k
                    self.stats[k] = max(self.stats[k], float(v)) if k == 'max_active' else self.stats[k]+float(v)
        ready = self.resident_service.receive(timeout)
        if ready:
            items = [self.resident_items.pop(key) for key in ready]
            self.stats['searched_roots'] += ShadowSearchCollection._store_targets(self,items,list(ready.values()))
        return ready

    @staticmethod
    def _select_resident(item, rows):
        from rl.policies.deck_latent import SearchContext
        result = dict(item)
        for name in ('indices', 'masks', 'logits'):
            result[name] = item[name][rows]
        for name in ('observations', 'roots', 'groups', 'budgets', 'positions', 'seeds'):
            if name not in item:
                continue
            result[name] = [item[name][i] for i in rows]
        result['context'] = SearchContext(item['context'].combat[rows], item['context'].anchor[rows])
        return result

    def _store_targets(self, items, result):
        offset = 0
        target_logits, target_masks, target_rows, destinations = [], [], [], []
        for item in items:
            selected = []
            for j, index in enumerate(item['indices']):
                row = result[offset+j]
                if row.get('skipped'):
                    continue
                selected.append(j)
                target_rows.append(row)
                position = item.get('positions', [item['position']]*len(item['indices']))[j]
                seed = item.get('seeds', [item['seed']]*len(item['indices']))[j]
                destinations.append((int(index), position))
                if self.config.action_selection != 'shadow':
                    self._record_action(int(index), position, row, item['logits'][j].numpy(),
                                        seed ^ (int(index)*1_000_003))
            if selected:
                complete = len(selected) == len(item['indices'])
                target_logits.append(item['logits'] if complete else item['logits'][selected])
                target_masks.append(item['masks'] if complete else item['masks'][selected])
            offset += len(item['roots'])
        if target_rows:
            # The legacy GPU pipeline finalizes a frame in bulk. These are
            # independent per-root softmaxes; batching never mixes baselines,
            # worlds or the original real (time, environment) destination.
            logits = torch.cat(target_logits)
            q = np.zeros(tuple(logits.shape), np.float32)
            worlds = np.empty(len(target_rows), np.int64)
            for j, row in enumerate(target_rows):
                q[j, row['actions']] = row['values']
                worlds[j] = row['visits'][0]
            targets = improved_prior(logits, torch.from_numpy(q),
                torch.from_numpy(np.concatenate(target_masks)), torch.from_numpy(worlds),
                self.config.target_std).numpy()
            for (index, position), target in zip(destinations, targets, strict=True):
                self.buffer.add_search_target(index, target, position=position)
        return len(destinations)

    def close(self):
        if getattr(self, '_closed', False): return
        self._closed = True
        if self.hook is not None:
            self.hook.remove()
        self.env.step_async = self.previous
        self.inference_cache.clear()
        if self.config.backend == 'cuda-resident' and hasattr(self,'resident_service'):
            service = self.resident_service
            aborted = bool(service.outstanding or self.resident_items or service.requests)
            if hasattr(self, 'native_stream'): self.native_stream.close()
            if aborted or service.failure:
                service.close(cancel='collection aborted')
            elif not service.parked:
                service.park()
            report = service.status
            for k, v in report.get('inference', {}).get('cache', {}).items():
                self.stats[k] += v
            pool = report.get('pool',{})
            for key in ('chains','atomic_steps','policy_rows','forced_rows','anchor_rows',
                        'cutoff_rows','terminal_rows','budget_floor_overage'):
                self.stats[key] += pool.get(key,0)
            self.stats['max_active'] = max(self.stats['max_active'],pool.get('max_active',0))
            self.stats['worker_search_seconds'] += report.get('active_tick_seconds',0)
            self.stats['context_refreshes'] += report.get('context_refreshes',0)
            for key,value in report.get('anchor_cache',{}).items():
                if 'anchor_cache_'+key in self.stats:
                    self.stats['anchor_cache_'+key] += value
            for key in ('submitted_roots','retired_roots','submissions_while_pending','ticks',
                        'page_oom_retries','deferred_admissions'):
                self.stats['stream_'+key] = pool.get(key,0)
            self.stats['stream_command_bytes'] = service.counts['command_bytes']
            self.stats['stream_peak_pages'] = pool.get('state_pages',{}).get('peak_leased_pages',0)
            self.resident_items.clear()
        self.model.deck_search_rng = self.rng.bit_generator.state
        self.model.deck_search_last_stats = self.stats
        for k, v in self.stats.items():
            self.model.logger.record('deck_search/' + k, v)
