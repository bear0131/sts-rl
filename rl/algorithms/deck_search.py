"""On-policy real actions with frozen, off-trajectory search teaching.

No search action replaces a sampled PPO action. No simulated transition is
written into the rollout buffer. A collection failure never accepts a partial
paired teacher. The auxiliary loss shares the existing PPO optimizer steps.
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
            'resident_frame_roots_max', *DECK_SEARCH_CACHE_METRICS), 0.)
        self.stats['backend_cuda_inference'] = float(self.config.backend in ('cuda-batched-native', 'cuda-resident'))
        self.stats['backend_cuda_simulation'] = float(self.config.backend == 'cuda-resident')
        self.hook = None
        self.previous = env.step_async
        self.snapshot = None
        self.resident_pending = []
        self.resident_pending_roots = 0

    def start(self):
        self.inference_cache = DeckSearchInferenceCache(self.model.policy.core, self.config.inference_cache_bytes)
        if self.config.backend == 'cuda-resident':
            from env.search.cuda_battle import CudaBattleArena, load_artifacts
            from env.core.run_env import _nosl_compact_lut
            from rl.search.resident_deck import ResidentDeckScorer
            folder, manifest, _ = load_artifacts(self.config.cuda_artifacts)
            self.resident_arena = CudaBattleArena(folder/'deck.cubin', manifest['sha256']['deck.cubin'],
                2*self.env.num_envs+self.config.cuda_wave_capacity,
                card_count=_nosl_compact_lut()['layout']['card_count'])
            self.resident_scorer = ResidentDeckScorer(self.model.policy, self.config.inference_cache_bytes)
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
        if not len(indices):
            return
        rows = torch.as_tensor(indices, device=self.model.device)
        logits = policy.action_dist.distribution.logits[rows].detach().cpu()
        # The ordinary action forward has just produced this exact context.
        # Reuse its output, without rerunning E or consuming policy RNG.
        context = policy.core.actor_search_context(rows)
        self.pending = (indices, masks[indices], logits, context)

    def step(self, actions):
        if self.config.backend == 'cuda-resident':
            return self._resident_step(actions)
        if self.pending is not None:
            started = time.monotonic()
            indices, masks, logits, context = self.pending
            observations = [{k: v[i] for k, v in self.model._last_obs.items()} for i in indices]
            budgets = [self.config.boss_budget if o['combat_is_boss'].item() else self.config.normal_budget
                       for o in observations]
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
                self.stats['searched_roots'] += 1
            self.stats['selected_roots'] += len(indices)
            self.stats['search_seconds'] += time.monotonic() - started
            for k, v in stats.items():
                k = 'worker_search_seconds' if k == 'seconds' else k
                self.stats[k] = max(self.stats[k], float(v)) if k == 'max_active' else self.stats[k] + float(v)
        return self.previous(actions)

    @torch.no_grad()
    def _resident_step(self, actions):
        # Shadow teachers never select real actions. Keep the frozen roots of
        # several real steps in one frame, as in the original GPU pipeline.
        # The original (time, env) and per-call worker RNG streams stay owned by
        # each item; neither physical grouping nor flushing changes them.
        started = time.monotonic()
        if self.pending is not None:
            indices, masks, logits, context = self.pending
            observations = [{k: np.array(v[i], copy=True) for k, v in self.model._last_obs.items()}
                            for i in indices]
            budgets = [self.config.boss_budget if o['combat_is_boss'].item() else self.config.normal_budget
                       for o in observations]
            seed = int(self.rng.integers(0, 2**63-1))
            roots, groups = self.env.deck_resident_roots(indices.tolist(), self.config.cuda_artifacts)
            from rl.policies.deck_latent import SearchContext
            context = SearchContext(context.combat.clone(), context.anchor.clone())
            self.resident_pending.append(dict(position=self.buffer.pos, indices=indices.copy(),
                masks=masks.copy(), logits=logits.clone(), context=context, observations=observations,
                roots=roots, groups=groups, budgets=budgets, seed=seed))
            self.resident_pending_roots += len(roots)
            self.stats['selected_roots'] += len(indices)
        if (self.resident_pending_roots >= self.env.num_envs
                or self.buffer.pos == self.buffer.buffer_size-1):
            self._flush_resident()
        self.stats['search_seconds'] += time.monotonic()-started
        return self.previous(actions)

    @torch.no_grad()
    def _flush_resident(self):
        if not self.resident_pending:
            return
        from env.search.resident_chains import resident_anchor_search
        from rl.policies.deck_latent import SearchContext
        items = self.resident_pending
        roots, observations, budgets, groups, group_seeds = [], [], [], [], {}
        for frame, item in enumerate(items):
            roots.extend(item['roots']); observations.extend(item['observations']); budgets.extend(item['budgets'])
            for worker in item['groups']:
                key = frame, int(worker)
                groups.append(key)
                group_seeds[key] = int(np.random.SeedSequence([item['seed'], int(worker)]).generate_state(1, dtype=np.uint64)[0])
        context = SearchContext(torch.cat([i['context'].combat for i in items]),
                                torch.cat([i['context'].anchor for i in items]))
        self.resident_scorer.roots(observations, context)
        result, stats = resident_anchor_search(roots, budgets, self.resident_scorer,
            self.resident_arena, seed=0, groups=groups, group_seeds=group_seeds,
            max_depth=self.config.max_depth, wave_capacity=self.config.cuda_wave_capacity,
            timeout_seconds=self.config.timeout_seconds)
        offset = 0
        for item in items:
            for j, index in enumerate(item['indices']):
                row = result[offset+j]
                if row.get('skipped'):
                    continue
                logits, masks = item['logits'], item['masks']
                q = torch.zeros_like(logits[j:j+1])
                q[0, row['actions']] = torch.as_tensor(row['values'], dtype=q.dtype)
                target = improved_prior(logits[j:j+1], q, torch.from_numpy(masks[j:j+1]),
                    torch.tensor([int(row['visits'][0])]), self.config.target_std)[0].numpy()
                self.buffer.add_search_target(int(index), target, position=item['position'])
                self.stats['searched_roots'] += 1
            offset += len(item['roots'])
        for k, v in stats.items():
            k = 'worker_search_seconds' if k == 'seconds' else k
            self.stats[k] = max(self.stats[k], float(v)) if k == 'max_active' else self.stats[k]+float(v)
        self.stats['resident_frames'] += 1
        self.stats['resident_frame_roots_max'] = max(self.stats['resident_frame_roots_max'], len(roots))
        self.resident_pending.clear()
        self.resident_pending_roots = 0

    def close(self):
        if self.hook is not None:
            self.hook.remove()
        self.env.step_async = self.previous
        self.inference_cache.clear()
        self.resident_pending.clear()  # Failed/aborted rollouts never flush a partial teacher.
        if self.config.backend == 'cuda-resident':
            for key, value in self.resident_scorer.anchor_cache.stats.items():
                metric = 'anchor_cache_' + key
                if metric in self.stats:
                    self.stats[metric] = value
            self.stats['context_refreshes'] = self.resident_scorer.refreshes
            self.resident_scorer.close()
            self.resident_arena.close()
        self.model.deck_search_rng = self.rng.bit_generator.state
        self.model.deck_search_last_stats = self.stats
        for k, v in self.stats.items():
            self.model.logger.record('deck_search/' + k, v)
