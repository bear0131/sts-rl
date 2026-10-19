"""Matched combat-start counterfactuals on a declared reference trajectory.

The episode runner still owns all stepping/budgets. Forks live only in this
measurement layer; the rule policy never receives a transition oracle.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time

import numpy as np

from env.core.combat_env import _OUTCOME_VICTORY

from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate
from rl.evaluation.baselines.combat_rules import visible_combat, rank_actions


def public_state(view):
    data = asdict(view)
    # tuple keys are useful inside the scorer but not JSON object keys.
    data['targets'] = {f'{kind}:{slot}': targets for (kind, slot), targets in view.targets.items()}
    return data


class CombatAuditStrategy:
    """Trace all combat decisions and optionally audit first N combat starts."""
    def __init__(self, reference, *, candidates=None, limit=0, rows=None,
                 trace=None, max_steps=2000, timeout=300.):
        self.reference = reference
        self.name = reference.name
        self.candidates = dict(candidates or {})
        self.limit = limit
        self.rows = rows if rows is not None else []
        self.trace = trace
        self.max_steps, self.timeout = max_steps, timeout
        self._active = {}
        self._starts = 0

    def decide(self, requests, *, check):
        views = {}
        for index, request in enumerate(requests):
            check()
            key = id(request.env)
            if request.info.get('phase') != 'combat':
                self._active.pop(key, None)
                continue
            view = visible_combat(request)
            views[index] = view
            identity = (request.seed, request.info.get('run_stats', {}).get('floor'), request.env.unwrapped.thick.bc)
            if self._active.get(key) != identity:
                self._active[key] = identity
                if self._starts < self.limit:
                    self._audit(request, view, check)
                    self._starts += 1
        decisions = self.reference.decide(requests, check=check)
        if self.trace:
            for index, view in views.items():
                request, decision = requests[index], decisions[index]
                ranking = rank_actions(view, np.flatnonzero(request.env.action_masks()))
                native = getattr(request.env.unwrapped, '_nosl_cpp_pending', None)
                native_action = None
                if native is not None:
                    import slaythespire as sts
                    complete = sts.Action(int(native[1]))
                    native_action = dict(bits=int(native[1]), kind=complete.action_type.name,
                                         source=complete.source_idx, target=complete.target_idx)
                self.trace(dict(kind='reference_decision', seed=request.seed, step=request.step,
                    floor=request.info.get('run_stats', {}).get('floor'), source=decision.source,
                    action=decision.action, native_action=native_action, rule_action=ranking[0]['action'],
                    state=public_state(view), ranked=ranking))
        return decisions

    def _audit(self, request, view, check):
        root = request.env.unwrapped
        initial = public_state(view)
        digest = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        context = dict(combat=self._starts, seed=request.seed, reference_step=request.step,
                       floor=request.info.get('run_stats', {}).get('floor', 0),
                       initial_hp=view.player['hp'], initial_potions=list(view.potions),
                       visible_state_sha256=digest,
                       encounter=[m['id'] for m in view.alive])
        if self.trace:
            self.trace(dict(kind='combat_start', **context, state=initial))
        for name, candidate in self.candidates.items():
            check()
            branch = root.fork_combat()
            def outcome(env, info, terminated, truncated):
                raw = env.unwrapped.thick.bc.to_dict()
                won = env.unwrapped.thick.bc.outcome == _OUTCOME_VICTORY
                return dict(result='win' if won else 'other_truncation' if truncated else 'escape' if raw['outcome'] == 3 else 'death',
                            floor=context['floor'], final_hp=raw['player']['hp'],
                            final_potions=raw['potions'])
            scenario = Scenario('matched-combat-start', lambda: branch, outcome,
                                reset=lambda env, seed: (env._obs(), env._info()))
            config = EvaluationConfig(episodes=1, seed=request.seed, max_steps=self.max_steps,
                                      timeout=self.timeout)
            # Parent deadline remains in force during nested search leaf batches.
            def cancelled():
                check()
                return False
            if self.trace:
                candidate = CombatAuditStrategy(candidate,
                    trace=lambda event: self.trace(dict(event, candidate=name, combat=self._starts)))
            started = time.monotonic()
            rows = list(evaluate(candidate, scenario, config, cancelled=cancelled))
            row = dict(rows[0], **context, candidate=name)
            row['wall_seconds'] = time.monotonic()-started
            row['hp_loss'] = context['initial_hp']-row['final_hp']
            self.rows.append(row)
            if self.trace:
                self.trace(dict(kind='combat_result', **row))


def summarize_pairs(rows):
    """Report paired HP only for fights both candidates won; keep failures."""
    by_combat = {}
    for row in rows:
        by_combat.setdefault(row['combat'], {})[row['candidate']] = row
    comparisons = {}
    names = sorted({r['candidate'] for r in rows} - {'rules'})
    for name in names:
        pairs = [items for items in by_combat.values() if 'rules' in items and name in items]
        wins = [p for p in pairs if p['rules']['result'] == p[name]['result'] == 'win']
        differences = [p['rules']['hp_loss']-p[name]['hp_loss'] for p in wins]
        comparisons[name] = dict(paired=len(pairs), both_won=len(wins),
            rules_wins=sum(p['rules']['result']=='win' for p in pairs),
            opponent_wins=sum(p[name]['result']=='win' for p in pairs),
            rules_minus_opponent_hp_loss_mean=sum(differences)/len(differences) if differences else None,
            largest_rule_losses=sorted([dict(combat=p['rules']['combat'], seed=p['rules']['seed'],
                floor=p['rules']['floor'], encounter=p['rules']['encounter'],
                extra_hp_loss=p['rules']['hp_loss']-p[name]['hp_loss']) for p in wins],
                key=lambda x: -x['extra_hp_loss'])[:10])
    return comparisons
