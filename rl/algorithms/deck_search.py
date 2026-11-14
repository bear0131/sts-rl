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
from rl.search.deck_latent import DeckLatentScorer
from rl.training.deck_search_worker import search_options


class ShadowSearchCollection:
    def __init__(self, model, env, buffer):
        self.model, self.env, self.buffer = model, env, buffer
        self.config = SearchTrainingConfig(**model.deck_search_config)
        self.direct_actions = self.config.action_selection != 'shadow'
        if self.config.backend != 'native-cpu':
            raise RuntimeError(
                'Combat49 GPU sampler withdrawn at user request; no CPU/shadow fallback. '
                'See docs/2026-11-14/combat49-old-sampling-handoff.md before resuming.')
        if getattr(model.policy.core, 'model_contract', None) != MODEL_CONTRACT:
            raise ValueError('shadow search is only supported by deck-latent-combat-v1')
        if not isinstance(buffer, SearchRolloutBuffer) or model.ppo_update_config is None:
            raise ValueError('shadow search requires the packed PPO/search buffer lifecycle')
        if model.policy.combat_value_target != 'pre-reward-anchor':
            raise ValueError('search teaching requires the learned combat-exit anchor target')
        self.rng = np.random.default_rng(self.config.seed)
        state = getattr(model, 'deck_search_rng', None)
        if state is not None:
            self.rng.bit_generator.state = state
        # Identical keys on every rank, including a rank with zero selected
        # roots, are required by the scalar reduction contract.
        self.stats = dict.fromkeys(('searched_roots', 'eligible_roots', 'selected_roots', 'search_seconds',
            'chains', 'atomic_steps', 'policy_rows', 'forced_rows', 'anchor_rows', 'cutoff_rows',
            'terminal_rows', 'max_active', 'budget_floor_overage', 'worker_search_seconds', 'context_refreshes',
            'executed_search_actions', 'changed_policy_actions'), 0.)
        self.hook = None
        self.previous = env.step_async
        self.snapshot = None

    def start(self):
        if hasattr(self.env, 'deck_shadow_search'):
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
        if self.pending is not None:
            started = time.monotonic()
            indices, masks, logits, context = self.pending
            observations = [{k: v[i] for k, v in self.model._last_obs.items()} for i in indices]
            budgets = [self.config.boss_budget if self.model._last_obs['combat_is_boss'][i].item()
                       else self.config.normal_budget for i in indices]
            seed = int(self.rng.integers(0, 2**63 - 1))
            if self.snapshot is not None:
                result, stats = self.env.deck_shadow_search(dict(indices=indices.tolist(), budgets=budgets,
                    context=(context.combat.detach().cpu().numpy(), context.anchor.detach().cpu().numpy()),
                    config=asdict(self.config), seed=seed, generation=self.generation, snapshot=str(self.snapshot)))
            else:
                if not hasattr(self.env, 'envs'):
                    raise ValueError('unsupported shadow search environment transport')
                roots = [self.env.envs[i].unwrapped for i in indices]
                scorer = DeckLatentScorer(self.model.policy, observations, context=context)
                result, stats = paired_anchor_search(roots, budgets, scorer, seed=seed, **search_options(self.config))
                stats['context_refreshes'] = scorer.context_refreshes
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


    def close(self):
        if self.hook is not None:
            self.hook.remove()
        self.env.step_async = self.previous
        self.model.deck_search_rng = self.rng.bit_generator.state
        self.model.deck_search_last_stats = self.stats
        for k, v in self.stats.items():
            self.model.logger.record('deck_search/' + k, v)
