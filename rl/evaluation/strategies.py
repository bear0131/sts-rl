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
import torch

from rl.policies.model import observation_for_policy
from rl.search.native import MODEL_LEAF_BACKEND, NativeSearchConfig, complete_searches, forward_root, supports_compact
from rl.search.inference import CombatLeafGraphs


@dataclass(frozen=True)
class DecisionRequest:
    env: Any
    observation: dict
    info: dict
    seed: int
    step: int


@dataclass(frozen=True)
class Decision:
    action: int | None
    source: str
    diagnostics: dict = field(default_factory=dict)
    native_action: int | None = None

    def __post_init__(self):
        if (self.action is None) == (self.native_action is None):
            raise ValueError('Decision requires exactly one thin or complete native action')


class DecisionStrategy(Protocol):
    name: str

    def decide(self, requests: list[DecisionRequest], *, check: Callable[[], None]) -> list[Decision]: ...


def strategy_manifest(strategy):
    """Record effective composition and search settings, including inherited values."""
    result = {'name': strategy.name}
    if getattr(strategy, 'combat_action_contract', None):
        result['combat_action_contract'] = strategy.combat_action_contract
    if getattr(strategy, 'version', None):
        result['version'] = strategy.version
    if callable(getattr(strategy, 'manifest', None)):
        result.update(strategy.manifest())
    if getattr(strategy, 'fallback', None) is not None:
        result['fallback'] = strategy_manifest(strategy.fallback)
    if isinstance(strategy, PhaseStrategy):
        result['default'] = strategy_manifest(strategy.default)
        result['phases'] = {phase: strategy_manifest(solver) for phase, solver in strategy.routes.items()}
        result['predicate_phases'] = sorted(strategy.predicates)
    if isinstance(strategy, NativeStrategy):
        result.update(config=asdict(strategy.config), master_nonce=strategy.master_nonce)
        backend = (getattr(strategy.model, '_native_leaf_inference', None)
                   if strategy.leaf_backend is MODEL_LEAF_BACKEND else strategy.leaf_backend)
        result['leaf_inference'] = {'backend': 'eager' if backend is None else type(backend).__name__}
        if isinstance(backend, CombatLeafGraphs):
            result['leaf_inference'] = dict(backend='cuda-graph', amp=backend.amp,
                capacity=backend.capacity, batch_multiple=backend.batch_multiple,
                statistics=backend.statistics())
    if isinstance(strategy, PolicyStrategy):
        result['deterministic'] = strategy.deterministic
    return result


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

    def __init__(self, model, config: NativeSearchConfig, *, master_nonce=0, lock=None,
                 leaf_backend=MODEL_LEAF_BACKEND):
        self.model, self.config, self.lock = model, config, lock
        self.master_nonce = int(master_nonce)
        self.leaf_backend = leaf_backend

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
                leaf_batch_size=self.config.leaf_batch_size, compact=compact, lock=self.lock,
                check=check, leaf_backend=self.leaf_backend)
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
        from rl.evaluation.baselines.routes import optimal_route_choices
        result = []
        for request in requests:
            check()
            legal = np.flatnonzero(request.env.action_masks())
            choices = [(offset, action) for action in legal for segment, offset in [A.segment_of(action)]
                       if segment == 'MAP_NODE']
            base = request.env.unwrapped.thick
            optimal, _ = optimal_route_choices(choices, base.rmap, base.gc.cur_map_node_y)
            rng = self.random.setdefault(request.seed, np.random.default_rng(request.seed ^ 0x51A7_7E12))
            result.append(Decision(int(rng.choice(optimal)), self.name, {'route_choices': 1}))
        return result


def policy_strategy(model, args, lock=None):
    return PolicyStrategy(model, deterministic=not args.stochastic, lock=lock)


def native_strategy(model, args, lock=None):
    leaf_inference = getattr(args, 'leaf_inference', 'eager')
    backend = None
    if leaf_inference not in ('eager', 'cuda-graph'):
        raise ValueError('leaf inference must be eager or cuda-graph')
    if leaf_inference == 'cuda-graph':
        if torch.device(model.device).type != 'cuda':
            raise ValueError('cuda-graph leaf inference requires a CUDA combat model')
        if not supports_compact(model):
            raise ValueError('cuda-graph leaf inference requires compact combat observations')
        if model.policy.training or any(p.dtype != torch.float32 for p in model.policy.parameters()):
            raise ValueError('cuda-graph evaluation requires an eval-mode FP32 combat model')
        # Exact shapes preserve the eager batch geometry. This runtime-only
        # cache changes neither the model weights nor per-tree leaf concurrency.
        backend = CombatLeafGraphs(model.policy.core, model.device,
            amp='off', capacity=32, batch_multiple=0)
    modes = {'puct': 0, 'gumbel': 1, 'full': 2, 'strict': 2}
    config = NativeSearchConfig.from_model(model, simulations=args.simulations, depth=args.depth,
        leaf_batch_size=args.leaf_batch_size, root_search_mode=modes[args.search_mode],
        paper_full_gumbel=args.search_mode == 'strict')
    return PhaseStrategy(policy_strategy(model, args, lock), {'combat': NativeStrategy(
        model, config, master_nonce=getattr(model, 'nosl_is_az_master_nonce', 0), lock=lock,
        leaf_backend=backend)},
        predicates={'combat': lambda request: getattr(request.env.unwrapped, '_pending', None) is None})


def value_strategy(model, args, lock=None):
    from rl.search.combat_value_search import config_from_args
    if args.combat_search_depth <= 0:
        args.combat_search_depth = 3
    return PhaseStrategy(policy_strategy(model, args, lock),
                         {'combat': ValueStrategy(model, config_from_args(args), lock)})


def rule_strategy(model, args, lock=None):
    from rl.evaluation.baselines.combat_rules import CombatRuleStrategy
    return PhaseStrategy(policy_strategy(model, args, lock), {'combat': CombatRuleStrategy()})


def turn_planner_strategy(model, args, lock=None):
    from env.search.turn_planner import TurnPlannerConfig
    from rl.evaluation.baselines.turn_planner import TurnPlannerStrategy
    config = TurnPlannerConfig(worlds=getattr(args, 'planner_worlds', 2),
        node_budget=getattr(args, 'planner_nodes', 512),
        beam_width=getattr(args, 'planner_beam', 8),
        max_depth=getattr(args, 'planner_depth', 16),
        master_nonce=getattr(args, 'planner_nonce', 20260918))
    return PhaseStrategy(policy_strategy(model, args, lock), {'combat': TurnPlannerStrategy(config)})


# Registration is deliberately a plain mapping, not discovery/import magic.
# Embedders can pass any DecisionStrategy directly without registering a CLI.
STRATEGIES = {'policy': policy_strategy, 'native': native_strategy, 'value': value_strategy,
              'leftmost': lambda model, args, lock=None: LeftmostStrategy(), 'rules': rule_strategy,
              'turn-planner': turn_planner_strategy}


def compose_strategy(upper_model, args, combat_model=None, *, trace=None):
    """Compose independently chosen upper/combat components outside the runner."""
    strategy = STRATEGIES[args.solver](combat_model if combat_model is not None else upper_model, args)
    if args.solver == 'turn-planner':
        strategy.routes['combat'].trace = trace
    if combat_model is not None:
        strategy = PhaseStrategy(policy_strategy(upper_model, args), {'combat': strategy})
    if getattr(args, 'upper_strategy', 'model') == 'rules':
        from rl.evaluation.baselines.overworld_rules import OverworldRuleStrategy
        upper = OverworldRuleStrategy(fallback=policy_strategy(upper_model, args), trace=trace)
        strategy = PhaseStrategy(upper, {'combat': strategy})
    if getattr(args, 'route', 'model') != 'model':
        strategy = PhaseStrategy(strategy, {'map': MapRouteStrategy()})
    return strategy
