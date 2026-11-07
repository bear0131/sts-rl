"""Learned root budgets with randomized, independent-world counterfactual audits."""
import time
import numpy as np
import torch
import slaythespire as sts

from rl.search.paired_token_rollout import PairedTokenBatch
from rl.policies.token_regret import encode_public_roots, pack_rows
from rl.policies.search_budget import (
    BUDGETS, AUDIT_PROBABILITY, AUDIT_WORLDS, COST_PRICE, budget_manifest, choose_budget,
)


def pick_action(bits, logits, values, visits, nonce):
    """Keep the original near-tie policy rule; zero budget uses the full policy."""
    bits, logits = np.asarray(bits), np.asarray(logits)
    visits, values = np.asarray(visits), np.asarray(values)
    eligible = np.flatnonzero((visits > 0) & (values >= values[visits > 0].max() - .005)) \
        if np.any(visits > 0) else np.arange(len(bits))
    tied = eligible[logits[eligible] == logits[eligible].max()]
    return int(bits[np.random.default_rng(int(nonce) ^ 0x524F4F54544945).choice(tied)])


def audit_targets(values, actions, choices, costs, units):
    """Independent panel means, signed against the neural-only action."""
    panel = np.asarray(values, np.float64)
    if panel.ndim != 2 or panel.shape[1] != len(actions) or not np.isfinite(panel).all():
        raise ValueError('audit requires complete finite independent worlds')
    means = panel.mean(0)
    by_action = {int(a): means[i] for i, a in enumerate(actions)}
    gains = [(by_action[a] - by_action[choices[0]]) * units / 10 for a in choices[1:]]
    return np.r_[gains, np.log1p(np.asarray(costs[1:], np.float64) / 1000)].astype(np.float32)


class AdaptiveBudgetBatch(PairedTokenBatch):
    def __init__(self, envs, pool, pair, policy, *, fixed_budget=None, cost_price=None):
        super().__init__(envs, pool, pair, policy)
        if pair.get('budget_config') != budget_manifest():
            raise ValueError('incompatible learned budget contract')
        if getattr(sts, 'small_paired_world_max_simulations', 0) < max(BUDGETS):
            raise ValueError('learned budgets require the native world batch backend')
        if fixed_budget is not None and (pair['online'] or type(fixed_budget) is not int or not 0 <= fixed_budget <= 4096):
            raise ValueError('fixed-budget reference requires frozen evaluation and 0..4096 simulations')
        if cost_price is not None and (pair['online'] or not np.isfinite(cost_price) or cost_price < 0):
            raise ValueError('a finite nonnegative cost-price override requires frozen evaluation')
        self.cost_price = COST_PRICE if cost_price is None else float(cost_price)
        self.fixed_budget = fixed_budget
        self.budgets = BUDGETS if fixed_budget is None else (0, fixed_budget) if fixed_budget else (0,)
        self._contexts = {}

    @torch.inference_mode()
    def prepare(self, calls):
        out, jobs = super().prepare(calls)
        retained = []
        for start in range(0, len(jobs), 128):
            group = jobs[start:start + 128]
            raw = [sts.small_playout_public_features(self.envs[j[0]].thick.gc,
                                                    self.envs[j[0]].thick.bc) for j in group]
            tokens = encode_public_roots(raw)
            rows = [dict(state=r['state'], actions=r['actions'], keys=r['keys'],
                         action_bits=r['action_bits'], tokens=t) for r, t in zip(raw, tokens, strict=True)]
            # Forced roots do not need the model or search; preserve native action bits.
            multi = [(j, r) for j, r in zip(group, rows, strict=True) if len(r['action_bits']) > 1]
            for j, row in zip(group, rows, strict=True):
                if len(row['action_bits']) == 1:
                    out[j[0]] = self.envs[j[0]]._apply_rulebase_turn_decision(int(row['action_bits'][0]), {})
                    self.stats['budget/forced_roots'] = self.stats.get('budget/forced_roots', 0) + 1
            if not multi:
                continue
            inputs = pack_rows([r for _, r in multi], self.device)
            scores, predictions = self.policy.score_with_budget(inputs,
                torch.tensor([j[2].boss for j, _ in multi], device=self.device))
            scores, predictions = scores.cpu().numpy(), predictions.cpu().numpy()
            for k, (job, row) in enumerate(multi):
                index, tier, cfg, nonce = job
                choice = (choose_budget(predictions[k], nonce, training=self.pair['online'],
                                        round_index=self.pair['budget_round'], cost_price=self.cost_price) if self.fixed_budget is None
                          else int(self.fixed_budget > 0))
                audit = self.pair['online'] and np.random.default_rng(nonce ^ 0x4155444954).random() < AUDIT_PROBABILITY
                cfg.simulations = self.budgets[choice]
                self._contexts[index] = dict(row=row, logits=scores[k, :len(row['action_bits'])].copy(),
                                             choice=choice, audit=bool(audit))
                retained.append(job)
        return out, retained

    def decisions(self, calls, *, pre_step=None, start_steps=None):
        if pre_step is not None or start_steps is not None:
            raise ValueError('adaptive roots must complete scoring before environment advancement')
        out, jobs = self.prepare(calls)
        if jobs:
            out.update(self.accept_ready(list(zip(jobs, self.search_jobs(self.pool, jobs), strict=True))))
        return {i: out[i] for i in calls}

    def search_jobs(self, pool, jobs):
        started = time.perf_counter()
        expanded, owners = [], []
        for pos, job in enumerate(jobs):
            ctx = self._contexts[job[0]]
            choices = range(1, len(self.budgets)) if ctx['audit'] else [ctx['choice']]
            for choice in choices:
                if choice == 0:
                    continue
                cfg = sts.SmallPlayoutConfig()
                cfg.paired_worldlines = True
                cfg.simulations, cfg.playout_depth = self.budgets[choice], job[2].playout_depth
                cfg.boss, cfg.temperature = job[2].boss, job[2].temperature
                expanded.append((job[0], job[1], cfg, job[3]))
                owners.append((pos, choice))
        measured = {}
        # Frozen evaluation can supply more roots than the stream's 64-root cap.
        for start in range(0, len(expanded), 480):
            results = super().search_jobs(pool, expanded[start:start + 480])
            measured.update(zip(owners[start:start + 480], results, strict=True))
        output = []
        for pos, job in enumerate(jobs):
            ctx = self._contexts[job[0]]
            row, logits, choice = ctx['row'], ctx['logits'], ctx['choice']
            n = len(row['action_bits'])
            baseline = {k: row[k] for k in ('state', 'actions', 'keys', 'action_bits')}
            baseline.update(values=np.zeros(n), visits=np.zeros(n), stats=dict(simulations=0,
                            playout_transitions=0, policy_calls=0, search_seconds=0.))
            candidates = {0: baseline}
            candidates.update({c: measured[pos, c] for c in range(1, len(self.budgets)) if (pos, c) in measured})
            for result in candidates.values():
                if not np.array_equal(row['action_bits'], result['action_bits']):
                    raise RuntimeError('budget search changed public root action order')
                result['action'] = pick_action(result['action_bits'], logits, result['values'], result['visits'], job[3])
            selected = candidates[choice]
            stats = selected['stats']
            stats['budget_decisions'] = 1
            stats[f'budget_selected_{self.budgets[choice]}'] = 1
            stats['budget_audits'] = int(ctx['audit'])
            targets = np.zeros(8, np.float32)
            units = 100. if job[2].boss else float(round(float(row['state'][1]) * 200))
            extra_simulations = extra_transitions = extra_policy_calls = 0
            if ctx['audit']:
                actions = [candidates[c]['action'] for c in range(len(BUDGETS))]
                unique = list(dict.fromkeys(actions))
                panel = sts.small_paired_worldlines(self.envs[job[0]].thick.bc, self.net, job[2],
                    unique, AUDIT_WORLDS * len(unique), job[3] ^ 0x48454C444F5554)
                costs = [candidates[c]['stats']['playout_transitions'] for c in range(len(BUDGETS))]
                targets = audit_targets(panel['values'], unique, actions, costs, units)
                for c, result in candidates.items():
                    if c != choice:
                        extra_simulations += result['stats']['simulations']
                        extra_transitions += result['stats']['playout_transitions']
                        extra_policy_calls += result['stats']['policy_calls']
                extra_simulations += panel['root_transitions']
                extra_transitions += panel['root_transitions'] + panel['playout_transitions']
                extra_policy_calls += panel['policy_calls']
                stats['budget_audit_completed'] = panel['completed']
                stats['budget_audit_attempts'] = panel['root_transitions']
            stats.update(budget_audit_simulations=extra_simulations,
                         budget_audit_transitions=extra_transitions,
                         budget_audit_policy_calls=extra_policy_calls,
                         budget_total_simulations=stats['simulations'] + extra_simulations,
                         budget_total_transitions=stats['playout_transitions'] + extra_transitions)
            selected['budget_sample'] = dict(tokens=row['tokens'], prior_logits=logits,
                return_units=units, values=np.asarray(selected['values'], np.float32),
                budget_searched=choice > 0, budget_audit=ctx['audit'], budget_boss=job[2].boss,
                budget_targets=targets)
            output.append(selected)
        # This wall-time attribution sums to dispatch time, including audit work.
        elapsed = time.perf_counter() - started
        for z in output:
            z['stats']['budget_dispatch_seconds'] = elapsed / len(output)
        return output

    def accept_ready(self, completed):
        out = {}
        for job, result in completed:
            decision = self.accept(job, result)
            if self.pair['online']:
                self.samples[-1].update(result['budget_sample'])
            self._contexts.pop(job[0])
            out[job[0]] = decision
        return out
