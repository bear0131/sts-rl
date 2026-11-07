"""Aggregate additive search-budget counters; never average rank percentages."""
from __future__ import annotations

import math

BUDGETS = (0, 32, 128, 512, 2048)
TIERS = ('common', 'elite', 'boss')


def merge_budget_records(existing, incoming):
    records = {(int(r['global_timesteps']), int(r['rank'])): r
               for r in existing + incoming}
    return [records[key] for key in sorted(records)]


def budget_rows(records, expected_ranks):
    groups = {}
    for record in records:
        groups.setdefault(int(record['global_timesteps']), []).append(record)
    result = []
    for step, group in sorted(groups.items()):
        tiers, valid = {}, True
        for tier in TIERS:
            counters = {}
            for record in group:
                for key, value in record['stats'].items():
                    prefix = tier + '/budget_'
                    if not key.startswith(prefix):
                        continue
                    if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                        valid = False
                        continue
                    name = key[len(prefix):]
                    counters[name] = counters.get(name, 0) + value
            n = counters.get('decisions', 0)
            counts = [counters.get('selected_' + str(b), 0) for b in BUDGETS]
            valid &= sum(counts) == n
            if n:
                valid &= all(key in counters for key in ('total_simulations',
                    'audit_simulations', 'total_transitions', 'audit_transitions'))
            total = counters.get('total_simulations', 0)
            audit = counters.get('audit_simulations', 0)
            transitions = counters.get('total_transitions', 0)
            audit_transitions = counters.get('audit_transitions', 0)
            valid &= total >= audit and transitions >= audit_transitions
            tiers[tier] = dict(decisions=n, counts=counts,
                simulations=total-audit, audit_simulations=audit,
                transitions=transitions-audit_transitions,
                audit_transitions=audit_transitions,
                audits=counters.get('audits', 0))
        all_tiers = {key: sum(t[key] for t in tiers.values())
                     for key in ('decisions', 'simulations', 'audit_simulations',
                                 'transitions', 'audit_transitions', 'audits')}
        all_tiers['counts'] = [sum(t['counts'][i] for t in tiers.values())
                               for i in range(len(BUDGETS))]
        tiers['all'] = all_tiers
        ranks = sorted({int(r['rank']) for r in group})
        rounds = {r.get('rollout_round') for r in group}
        valid &= len(rounds) == 1 and None not in rounds
        result.append(dict(global_timesteps=step,
            rollout_round=group[0].get('rollout_round'), ranks=ranks,
            expected_ranks=expected_ranks,
            complete=ranks == list(range(expected_ranks)), valid=bool(valid),
            tiers=tiers))
    return result
