"""Evaluation adapter for the shared public-state overworld rules."""
import numpy as np

from env.search.overworld_rules import (
    AOE, CARD_VALUE, CURSES, DRAW, EXHAUST, OVERWORLD_RULE_VERSION,
    PERMANENT_CURSES, SCALING, STRENGTH, card_value, deck_counts, event_scores,
    has_relic, rank_overworld, relic_value, remove_value, route_scores,
    upgrade_value,
)


class OverworldRuleStrategy:
    name = 'overworld-rules'
    version = OVERWORLD_RULE_VERSION

    def __init__(self, fallback=None, *, trace=None):
        self.fallback, self.trace = fallback, trace

    def decide(self, requests, *, check=lambda: None):
        from rl.evaluation.strategies import Decision
        result, fallback_requests, fallback_indices, traces = [None] * len(requests), [], [], []
        for index, request in enumerate(requests):
            check()
            base = request.env.unwrapped.thick
            state = base.obs_canonical()
            legal = np.flatnonzero(request.env.action_masks())
            ranked = rank_overworld(state, legal, rmap=base.rmap,
                                    current_y=base.gc.cur_map_node_y)
            if ranked is None:
                if self.fallback is None:
                    raise ValueError(f'unsupported overworld screen without fallback: {state["phase"]}/{state.get("options", {})}')
                fallback_indices.append(index)
                fallback_requests.append(request)
            else:
                result[index] = Decision(ranked[0]['action'], self.name, {'overworld_rule_decisions': 1})
            if self.trace:
                traces.append(dict(kind='overworld_decision', seed=request.seed,
                    step=request.step, phase=state['phase'], version=self.version,
                    ranked=ranked, state=state))
        if fallback_requests:
            decisions = self.fallback.decide(fallback_requests, check=check)
            if len(decisions) != len(fallback_requests):
                raise ValueError('overworld fallback returned a wrong decision count')
            for index, decision in zip(fallback_indices, decisions):
                diagnostics = dict(decision.diagnostics)
                diagnostics['overworld_fallback_decisions'] = 1
                result[index] = Decision(decision.action, f'{self.name}:fallback:{decision.source}', diagnostics)
        if self.trace:
            for trace, decision in zip(traces, result):
                self.trace(dict(trace, action=decision.action, source=decision.source))
        return result
