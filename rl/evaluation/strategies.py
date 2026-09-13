"""Composable batched decision strategies; no episode loops or reporting.

New strategies implement DecisionStrategy.decide(requests, check=...). They may
hold their own models/search state and return diagnostics with each decision.
PhaseStrategy composes them without adding solver branches to the runner/UI.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol
import time

import numpy as np

from rl.policies.model import observation_for_policy
from rl.search.native import NativeSearchConfig, complete_searches, forward_root, supports_compact


@dataclass(frozen=True)
class DecisionRequest:
    env: Any
    observation: dict
    info: dict
    seed: int
    step: int


@dataclass(frozen=True)
class Decision:
    action: int
    source: str
    diagnostics: dict = field(default_factory=dict)


class DecisionStrategy(Protocol):
    name: str

    def decide(self, requests: list[DecisionRequest], *, check: Callable[[], None]) -> list[Decision]: ...


class PolicyStrategy:
    name = 'policy'

    def __init__(self, model, *, deterministic=True, lock=None):
        self.model, self.deterministic, self.lock = model, deterministic, lock

    def decide(self, requests, *, check=lambda: None):
        if not requests:
            return []
        check()
        observations = {key: np.stack([r.observation[key] for r in requests]) for key in requests[0].observation}
        masks = np.stack([r.env.action_masks() for r in requests])
        with self.lock or nullcontext():
            actions, _ = self.model.predict(observation_for_policy(self.model, observations),
                                            action_masks=masks, deterministic=self.deterministic)
        return [Decision(int(action), self.name) for action in np.asarray(actions).reshape(-1)]


class NativeStrategy:
    name = 'native'

    def __init__(self, model, config: NativeSearchConfig, *, master_nonce=0, lock=None):
        self.model, self.config, self.lock = model, config, lock
        self.master_nonce = int(master_nonce)

    def decide(self, requests, *, check=lambda: None):
        if not requests:
            return []
        started = time.monotonic()
        compact = supports_compact(self.model)
        observations = {key: np.stack([r.observation[key] for r in requests]) for key in requests[0].observation}
        logits, values = forward_root(self.model, observations, lock=self.lock)
        responses = {}
        try:
            for index, request in enumerate(requests):
                check()
                nonce = (self.master_nonce ^ request.seed ^ (request.step * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
                responses[index] = request.env.unwrapped.nosl_is_az_cpp_begin(
                    logits[index], root_value=float(values[index]), state_nonce=nonce,
                    decision_index=request.step, stochastic=False, compact_observation=compact,
                    dirichlet_alpha=0.0, dirichlet_epsilon=0.0, **asdict(self.config))
            results, stats = complete_searches(
                self.model, lambda method, calls: {
                    i: getattr(requests[i].env.unwrapped, method)(*args, **kw)
                    for i, (args, kw) in calls.items()}, responses,
                leaf_batch_size=self.config.leaf_batch_size, compact=compact, lock=self.lock, check=check)
        except BaseException:
            for request in requests:
                request.env.unwrapped.nosl_is_az_cpp_abort()
            raise
        # Batch costs are attributed equally, never multiplied by worker count.
        return [Decision(int(results[i]['action']), self.name, {
            'search_decisions': 1, 'search_simulations': self.config.simulations,
            'search_leaf_evaluations': stats['by_worker'][i]['leaf_evaluations'],
            'search_seconds': (time.monotonic() - started) / len(requests),
        }) for i in range(len(requests))]


class PhaseStrategy:
    """Dispatch each request once, preserving batch and result order."""
    def __init__(self, default: DecisionStrategy, routes: dict[str, DecisionStrategy], *, predicates=None):
        self.default, self.routes = default, dict(routes)
        self.predicates = dict(predicates or {})
        self.name = '+'.join([default.name, *[f'{phase}:{solver.name}' for phase, solver in self.routes.items()]])

    def decide(self, requests, *, check=lambda: None):
        grouped = {}
        for index, request in enumerate(requests):
            phase = request.info.get('phase')
            solver = self.routes.get(phase, self.default)
            if phase in self.predicates and not self.predicates[phase](request):
                solver = self.default
            group = grouped.setdefault(id(solver), (solver, []))
            group[1].append((index, request))
        result = [None] * len(requests)
        for solver, indexed in grouped.values():
            decisions = solver.decide([r for _, r in indexed], check=check)
            if len(decisions) != len(indexed):
                raise ValueError(f'{solver.name} returned a wrong decision count')
            for (index, _), decision in zip(indexed, decisions):
                result[index] = decision
        return result


class LeftmostStrategy:
    name = 'leftmost'

    def decide(self, requests, *, check=lambda: None):
        check()
        return [Decision(int(np.flatnonzero(r.env.action_masks())[0]), self.name) for r in requests]


class ValueStrategy:
    name = 'value'

    def __init__(self, model, config, lock=None):
        from rl.search.combat_value_search import CombatValueSearch
        self.search, self.lock = CombatValueSearch(model, config), lock

    def decide(self, requests, *, check=lambda: None):
        result = []
        for request in requests:
            check()
            with self.lock or nullcontext():
                item = self.search.search(request.env, request.observation, request.env.action_masks())
            result.append(Decision(item.action, self.name, dict(search_decisions=1,
                search_nodes=item.expanded_nodes, search_seconds=item.elapsed_seconds,
                search_changes=int(item.action != item.model_action))))
        return result


class MapRouteStrategy:
    name = 'min-elite-max-rest'

    def __init__(self):
        self.random = {}

    def decide(self, requests, *, check=lambda: None):
        from env.core import action_space as A
        from rl.evaluation.baselines.eval_starter_policy import _dp_route_choices
        result = []
        for request in requests:
            check()
            legal = np.flatnonzero(request.env.action_masks())
            choices = [(offset, action) for action in legal for segment, offset in [A.segment_of(action)]
                       if segment == 'MAP_NODE']
            base = request.env.unwrapped.thick
            optimal, _ = _dp_route_choices(choices, base.rmap, base.gc.cur_map_node_y)
            rng = self.random.setdefault(request.seed, np.random.default_rng(request.seed ^ 0x51A7_7E12))
            result.append(Decision(int(rng.choice(optimal)), self.name, {'route_choices': 1}))
        return result


def policy_strategy(model, args, lock=None):
    return PolicyStrategy(model, deterministic=not args.stochastic, lock=lock)


def native_strategy(model, args, lock=None):
    modes = {'puct': 0, 'gumbel': 1, 'full': 2, 'strict': 2}
    config = NativeSearchConfig.from_model(model, simulations=args.simulations, depth=args.depth,
        leaf_batch_size=args.leaf_batch_size, root_search_mode=modes[args.search_mode],
        paper_full_gumbel=args.search_mode == 'strict')
    return PhaseStrategy(policy_strategy(model, args, lock), {'combat': NativeStrategy(
        model, config, master_nonce=getattr(model, 'nosl_is_az_master_nonce', 0), lock=lock)},
        predicates={'combat': lambda request: getattr(request.env.unwrapped, '_pending', None) is None})


def value_strategy(model, args, lock=None):
    from rl.search.combat_value_search import config_from_args
    if args.combat_search_depth <= 0:
        args.combat_search_depth = 3
    return PhaseStrategy(policy_strategy(model, args, lock),
                         {'combat': ValueStrategy(model, config_from_args(args), lock)})


# Registration is deliberately a plain mapping, not discovery/import magic.
# Embedders can pass any DecisionStrategy directly without registering a CLI.
STRATEGIES = {'policy': policy_strategy, 'native': native_strategy, 'value': value_strategy,
              'leftmost': lambda model, args, lock=None: LeftmostStrategy()}
