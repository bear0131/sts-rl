"""Composable batched decision strategies; no episode loops or reporting.

New strategies implement DecisionStrategy.decide(requests, check=...). They may
hold their own models/search state and return diagnostics with each decision.
PhaseStrategy composes them without adding solver branches to the runner/UI.
"""
from contextlib import nullcontext
from functools import partial
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol
import time

import numpy as np
import torch
from env.core import native_actions as C

from rl.policies.model import (
    observation_for_policy, action_mask_for_policy, validate_reward_policy_contract,
)
from rl.policies.precision import bf16_autocast
from rl.search.native import (
    MODEL_LEAF_BACKEND,
    NativeSearchConfig,
    complete_searches,
    complete_searches_native,
    forward_root,
    supports_compact,
)
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


class SmallPlayoutStrategy:
    """Complete native battle actions; compose with an upper via PhaseStrategy."""
    name = 'small-playout'
    combat_action_contract = 'native-v1'

    def __init__(self, search):
        self.search = search

    def manifest(self):
        return self.search.manifest()

    def decide(self, requests, *, check=lambda: None):
        check()
        if any(r.info.get('phase') != 'combat' for r in requests):
            raise ValueError('small playout policy requires combat requests')
        if any(getattr(getattr(r.env, 'unwrapped', r.env), 'combat_action_contract', None) != 'native-v1'
               for r in requests):
            raise ValueError('small playout strategy requires native-v1 complete-action environments')
        from dataclasses import replace
        from env.core.boss_value import is_boss_room
        configs = [replace(self.search.config, boss=is_boss_room(r.env.thick.gc.cur_room)) for r in requests]
        results = self.search.search([r.env.thick.bc for r in requests],
                                     [r.seed * 1000000007 + r.step for r in requests], configs)
        check()
        return [Decision(None, self.name, result['stats'], native_action=result['action']) for result in results]


class OnlinePlayoutEvaluationStrategy:
    """Frozen online lower policy through the actual collection decision path.

    Thin autoregressive actions preserve pending multi-pick bookkeeping. The
    shared worker owns tier budgets, public-information Dome fallback and search.
    """
    name = 'small-playout'
    combat_action_contract = 'autoregressive-v2'

    def __init__(self, policy, *, threads=1, paired_worldlines=False, fixed_budget=None,
                 cost_price=None, sample_collector=None):
        from rl.policies.small_playout import export_online_search_pair
        if threads < 1:
            raise ValueError('native search threads must be positive')
        from rl.policies.token_regret import OnlineTokenRegretPolicy
        self.root_policy = policy if isinstance(policy, OnlineTokenRegretPolicy) else None
        if self.root_policy is None:
            self.pair = export_online_search_pair(policy, collect_samples=False, paired_worldlines=paired_worldlines)
        else:
            self.pair = self.root_policy.collection_pair(collect_samples=sample_collector is not None)
            paired_worldlines = True
        if paired_worldlines:
            self.name = 'paired-worldlines'
        if fixed_budget is not None:
            if 'budget_config' not in self.pair or type(fixed_budget) is not int or not 0 <= fixed_budget <= 4096:
                raise ValueError('fixed-budget reference needs an adaptive checkpoint and 0..4096 simulations')
            self.name = 'fixed-budget-token'
        if (cost_price is not None or sample_collector is not None) and 'budget_config' not in self.pair:
            raise ValueError('budget cost/calibration requires an adaptive policy')
        if fixed_budget is not None and cost_price is not None:
            raise ValueError('a fixed budget cannot also have an adaptive cost price')
        if sample_collector is not None and (fixed_budget is not None or cost_price is not None):
            raise ValueError('audit collection uses the declared training budget distribution')
        self.fixed_budget = fixed_budget
        self.cost_price = cost_price
        self.sample_collector = sample_collector
        self.threads = threads
        self._batch = self._pool = None
        self._indices = {}
        self._statistics = {}

    def bind_environments(self, environments):
        import slaythespire as sts
        from env.search.small_playout import SmallPlayoutBatch
        self.unbind_environments()
        envs = [getattr(env, 'unwrapped', env) for env in environments]
        if any(env.combat_action_contract != self.combat_action_contract for env in envs):
            raise ValueError('online small-playout evaluation requires autoregressive-v2 environments')
        if len({id(env) for env in envs}) != len(envs):
            raise ValueError('online small-playout environments must be distinct')
        self._statistics = {}
        pool = sts.NeuralSearchBatch(self.threads)
        try:
            if self.root_policy is None:
                batch = SmallPlayoutBatch(envs, pool, self.pair)
            else:
                from rl.search.paired_token_rollout import PairedTokenBatch
                from rl.search.adaptive_budget import AdaptiveBudgetBatch
                factory = AdaptiveBudgetBatch if 'budget_config' in self.pair else PairedTokenBatch
                batch = factory(envs, pool, self.pair, self.root_policy,
                                **(dict(fixed_budget=self.fixed_budget, cost_price=self.cost_price)
                                   if 'budget_config' in self.pair else {}))
        except BaseException:
            pool.clear()
            raise
        self._pool, self._batch = pool, batch
        self._indices = {id(env): index for index, env in enumerate(envs)}

    def unbind_environments(self):
        try:
            if self._batch is not None:
                self._statistics = self._batch.scalar_stats()
                if self.sample_collector is not None:
                    self.sample_collector.save(self.root_policy)
        finally:
            self._batch = None
            self._indices = {}
            if self._pool is not None:
                self._pool.clear()
                self._pool = None

    def decide(self, requests, *, check=lambda: None):
        check()
        if not requests:
            return []
        if self._batch is None:
            raise RuntimeError('bind evaluation environments before small-playout decisions')
        if any(r.info.get('phase') != 'combat' for r in requests):
            raise ValueError('small-playout requires combat requests')
        indices = [self._indices[id(getattr(r.env, 'unwrapped', r.env))] for r in requests]
        if len(set(indices)) != len(indices):
            raise ValueError('duplicate environment in decision batch')
        # Episode-local nonces keep frozen evaluation independent of batch order.
        # These are independent evaluation seeds, not training RNG replay.
        calls = {i: ((), {'state_nonce': (int(r.seed)*1000000007 + int(r.step)) & ((1 << 64)-1)})
                 for i, r in zip(indices, requests, strict=True)}
        results = self._batch.decisions(calls)
        if self.sample_collector is not None:
            self.sample_collector.add(self.root_policy, self._batch.samples,
                                      episode_seeds=(request.seed for request in requests))
            self._batch.samples.clear()
        check()
        return [Decision(results[i]['action'], self.name, results[i]['diagnostics']) for i in indices]

    def manifest(self):
        from rl.policies.token_regret import TOKEN_REGRET_CONTRACT
        from rl.policies.search_budget import BUDGET_CONTRACT
        contract = BUDGET_CONTRACT if 'budget_config' in self.pair else TOKEN_REGRET_CONTRACT if self.root_policy is not None else self.pair['contract']
        return dict(contract=contract, search_contract=self.pair['search_contract'],
                    budget_config=self.pair.get('budget_config'),
                    fixed_budget_simulations=self.fixed_budget, cost_price_override=self.cost_price,
                    lower_sha256=self.pair['sha256'], native_threads=self.threads,
                    config={k: self.pair[k] for k in ('common_simulations', 'elite_simulations',
                            'boss_simulations', 'playout_depth', 'tree_depth')},
                    prior=False, collect_training_samples=self.sample_collector is not None,
                    dome='shared collection public-feature policy; no hidden-intent search',
                    nonce='uint64(episode_seed*1000000007+episode_step)',
                    search_statistics=self._batch.scalar_stats() if self._batch is not None else self._statistics)


def _forced_end_batch(strategy, requests, check):
    """Resolve forced execution roots; internal search leaves still evaluate V."""
    if strategy.combat_action_contract not in C.NEURAL_CONTRACTS:
        return None
    forced = {}
    for index, request in enumerate(requests):
        mask = request.env.action_masks()
        if request.info.get('phase') == 'combat' and len(mask) != C.policy_size(strategy.combat_action_contract):
            raise ValueError('joint-v2 combat policy requires a joint-v2 environment')
        if request.info.get('phase') == 'combat' and mask[C.policy_offset(strategy.combat_action_contract)] and np.count_nonzero(mask) == 1:
            forced[index] = Decision(C.policy_offset(strategy.combat_action_contract), 'forced_end_turn',
                                    {'root_forward_skipped':1, 'search_simulations':0})
    if not forced:
        return None
    check()
    remaining = [i for i in range(len(requests)) if i not in forced]
    normal = strategy.decide([requests[i] for i in remaining], check=check) if remaining else []
    forced.update(zip(remaining, normal))
    return [forced[i] for i in range(len(requests))]


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
        result['native_search_threads'] = strategy.native_search_threads
        backend = (getattr(strategy.model, '_native_leaf_inference', None)
                   if strategy.leaf_backend is MODEL_LEAF_BACKEND else strategy.leaf_backend)
        result['leaf_inference'] = {'backend': 'eager' if backend is None else type(backend).__name__}
        if isinstance(backend, CombatLeafGraphs):
            result['leaf_inference'] = dict(backend='cuda-graph', amp=backend.amp,
                capacity=backend.capacity, batch_multiple=backend.batch_multiple,
                compile_forward=backend.compile_forward,
                statistics=backend.statistics())
    if isinstance(strategy, FrozenUpperStrategy):
        result.update(deterministic=True, precision='FP32',
                      teacher_sha256=strategy.model.policy.core.teacher.assert_integrity(),
                      reward_policy_contract='checkpoint-v7-act3')
    if isinstance(strategy, PolicyStrategy):
        result['deterministic'] = strategy.deterministic
        result['reward_policy_contract'] = strategy.reward_policy_contract
    return result


class FrozenUpperStrategy:
    name = 'frozen-upper'

    def __init__(self, model, lock=None):
        self.model, self.lock = model, lock
        self.combat_action_contract = C.policy_contract(model.observation_space)

    def decide(self, requests, *, check=lambda: None):
        if not requests:
            return []
        check()
        observations = {key: np.stack([r.observation[key] for r in requests]) for key in requests[0].observation}
        masks = np.stack([r.env.action_masks() for r in requests])
        with self.lock or nullcontext():
            tensor, _ = self.model.policy.obs_to_tensor(observations)
            actions, _ = self.model.policy.core.teacher.decide(tensor, masks)
        return [Decision(int(action), self.name) for action in actions.cpu().tolist()]


class PolicyStrategy:
    name = 'policy'

    def __init__(self, model, *, deterministic=True, lock=None, autocast=nullcontext,
                 reward_policy_contract="native"):
        self.model, self.deterministic, self.lock = model, deterministic, lock
        self.autocast = autocast
        self.combat_action_contract = C.policy_contract(getattr(model, 'observation_space', None))
        validate_reward_policy_contract(model, reward_policy_contract)
        self.reward_policy_contract = reward_policy_contract

    def decide(self, requests, *, check=lambda: None):
        if not requests:
            return []
        if getattr(getattr(getattr(self.model, 'policy', None), 'core', None), 'combat_only', False):
            if any(r.info.get('phase') != 'combat' for r in requests):
                raise ValueError('combat-only model requires a separate overworld strategy')
        from env.core.encode import ACT_SCALE
        masks = np.stack([
            action_mask_for_policy(
                self.model, r.env.action_masks(), phase=r.info['phase'],
                act=(float(r.observation['act'].item()) * ACT_SCALE
                     if self.reward_policy_contract == 'checkpoint-v7-act3' else None),
                reward_policy_contract=self.reward_policy_contract)
            for r in requests
        ])
        forced = _forced_end_batch(self, requests, check)
        if forced is not None:
            return forced
        check()
        observations = {key: np.stack([r.observation[key] for r in requests]) for key in requests[0].observation}
        if self.combat_action_contract in C.NEURAL_CONTRACTS and 'combat_legal' not in observations:
            if any(r.info.get('phase') == 'combat' for r in requests):
                raise ValueError('joint-v2 combat policy requires combat_legal observations')
            # A legacy combat solver may share the unchanged overworld policy.
            observations['combat_legal'] = np.zeros((len(requests), (C.policy_size(self.combat_action_contract)-C.policy_offset(self.combat_action_contract))), dtype=bool)
        if self.combat_action_contract in C.NEURAL_CONTRACTS and masks.shape[1] == C.JOINT_END:
            masks = np.pad(masks, ((0, 0), (0, C.policy_size(self.combat_action_contract)-C.JOINT_END)))
        if self.combat_action_contract == 'legacy' and masks.shape[1] > self.model.action_space.n:
            if any(r.info.get('phase') == 'combat' for r in requests):
                raise ValueError('legacy policy cannot consume a joint-v2 combat environment')
            masks = masks[:, :self.model.action_space.n]
        groups = [
            r.env.equivalent_action_groups()
            if (self.combat_action_contract in C.NEURAL_CONTRACTS
                and getattr(self.model, 'nosl_action_group_contract', 'equivalent-plays-v1') != 'individual-actions'
                and r.info.get('phase') == 'combat')
            else np.arange(masks.shape[1]) for r in requests
        ]
        grouped = any(np.any(g != np.arange(len(g))) for g in groups)
        with self.lock or nullcontext(), self.autocast():
            if grouped:
                self.model.policy.set_training_mode(False)
                with torch.no_grad():
                    tensor, _ = self.model.policy.obs_to_tensor(
                        observation_for_policy(self.model, observations))
                    distribution = self.model.policy.get_distribution(tensor, action_masks=masks)
                    probabilities = distribution.distribution.probs
                    mapping = torch.as_tensor(np.stack(groups), device=probabilities.device)
                    mass = torch.zeros_like(probabilities).scatter_add_(1, mapping, probabilities)
                    actions = (mass.argmax(dim=1) if self.deterministic else
                               torch.multinomial(mass, 1).flatten()).cpu().numpy()
            else:
                actions, _ = self.model.predict(observation_for_policy(self.model, observations),
                                                action_masks=masks, deterministic=self.deterministic)
        # Contract identity belongs to strategy_manifest; per-step diagnostics
        # are numeric counters accumulated by the generic episode runner.
        return [Decision(int(action), self.name,
                         {'equivalent_play_mass_decisions': 1} if grouped else {})
                for action in np.asarray(actions).reshape(-1)]


class NativeStrategy:
    name = 'native'

    def __init__(self, model, config: NativeSearchConfig, *, master_nonce=0, lock=None,
                 leaf_backend=MODEL_LEAF_BACKEND, native_search_threads=0,
                 root_autocast=nullcontext):
        self.model, self.config, self.lock = model, config, lock
        self.master_nonce = int(master_nonce)
        self.leaf_backend = leaf_backend
        self.root_autocast = root_autocast
        self.native_search_threads = int(native_search_threads)
        if self.native_search_threads < 0:
            raise ValueError('native search threads must be non-negative')
        self._native_batch = None
        self._native_env_indices = None
        self.combat_action_contract = C.policy_contract(getattr(model, 'observation_space', None))

    def bind_environments(self, environments):
        self.unbind_environments()
        if not self.native_search_threads:
            return
        from env.search.native_batch import NativeSearchBatch
        self._native_batch = NativeSearchBatch(environments, self.native_search_threads)
        self._native_env_indices = {id(env): index for index, env in enumerate(environments)}

    def unbind_environments(self):
        if self._native_batch is not None:
            self._native_batch.close()
        self._native_batch = None
        self._native_env_indices = None

    def decide(self, requests, *, check=lambda: None):
        if not requests:
            return []
        forced = _forced_end_batch(self, requests, check)
        if forced is not None:
            return forced
        started = time.monotonic()
        compact = supports_compact(self.model)
        observations = {key: np.stack([r.observation[key] for r in requests]) for key in requests[0].observation}
        logits, values = forward_root(
            self.model, observations, lock=self.lock, autocast=self.root_autocast)
        responses = {}
        begin_calls = {}
        local_to_native = {}
        try:
            for index, request in enumerate(requests):
                check()
                nonce = (self.master_nonce ^ request.seed ^ (request.step * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
                kwargs = dict(
                    root_value=float(values[index]),
                    state_nonce=nonce,
                    decision_index=request.step,
                    stochastic=False,
                    compact_observation=compact,
                    dirichlet_alpha=0.0,
                    dirichlet_epsilon=0.0,
                    **asdict(self.config),
                )
                if self._native_batch is None:
                    responses[index] = request.env.unwrapped.nosl_is_az_cpp_begin(
                        logits[index], **kwargs)
                else:
                    native_index = self._native_env_indices.get(id(request.env))
                    if native_index is None:
                        raise ValueError('native search received an environment outside its bound pool')
                    local_to_native[index] = native_index
                    begin_calls[native_index] = ((logits[index],), kwargs)
            if self._native_batch is None:
                results, stats = complete_searches(
                    self.model, lambda method, calls: {
                        i: getattr(requests[i].env.unwrapped, method)(*args, **kw)
                        for i, (args, kw) in calls.items()}, responses,
                    leaf_batch_size=self.config.leaf_batch_size, compact=compact, lock=self.lock,
                    check=check, leaf_backend=self.leaf_backend)
            else:
                native_results, stats = complete_searches_native(
                    self.model,
                    self._native_batch,
                    begin_calls,
                    leaf_backend=self.leaf_backend,
                )
                results = {index: native_results[native_index]
                           for index, native_index in local_to_native.items()}
        except BaseException:
            if self._native_batch is not None:
                self._native_batch.clear()
            else:
                for request in requests:
                    request.env.unwrapped.nosl_is_az_cpp_abort()
            raise
        # Batch costs are attributed equally, never multiplied by worker count.
        return [Decision(int(results[i]['action']), self.name, {
            'search_decisions': 1, 'search_simulations': self.config.simulations,
            'search_leaf_evaluations': stats['by_worker'][
                local_to_native.get(i, i)
            ]['leaf_evaluations'],
            'search_seconds': (time.monotonic() - started) / len(requests),
        }) for i in range(len(requests))]


class PhaseStrategy:
    """Dispatch each request once, preserving batch and result order."""
    def __init__(self, default: DecisionStrategy, routes: dict[str, DecisionStrategy], *, predicates=None):
        self.default, self.routes = default, dict(routes)
        self.predicates = dict(predicates or {})
        self.name = '+'.join([default.name, *[f'{phase}:{solver.name}' for phase, solver in self.routes.items()]])

    @property
    def combat_action_contract(self):
        possible = [self.routes.get('combat', self.default)]
        if 'combat' in self.routes and 'combat' in self.predicates:
            possible.append(self.default)
        contracts = {
            getattr(strategy, 'combat_action_contract', 'legacy')
            for strategy in possible
        }
        unknown = contracts - {'legacy', 'native-v1', *C.NEURAL_CONTRACTS}
        if unknown:
            raise ValueError(f'unsupported combat action contracts: {unknown}')
        if contracts & set(C.NEURAL_CONTRACTS):
            if len(contracts) != 1:
                raise ValueError('conditional combat strategies must share the joint-v2 contract')
            return next(iter(contracts))
        return 'native-v1' if 'native-v1' in contracts else 'legacy'

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

    def _unique_strategies(self):
        seen = set()
        for strategy in (self.default, *self.routes.values()):
            if id(strategy) not in seen:
                seen.add(id(strategy))
                yield strategy

    def bind_environments(self, environments):
        bound = []
        try:
            for strategy in self._unique_strategies():
                bind = getattr(strategy, 'bind_environments', None)
                if bind is not None:
                    bind(environments)
                    bound.append(strategy)
        except BaseException:
            for strategy in reversed(bound):
                strategy.unbind_environments()
            raise

    def unbind_environments(self):
        for strategy in self._unique_strategies():
            unbind = getattr(strategy, 'unbind_environments', None)
            if unbind is not None:
                unbind()


class LeftmostStrategy:
    name = 'leftmost'

    def decide(self, requests, *, check=lambda: None):
        check()
        return [Decision(int(np.flatnonzero(r.env.action_masks())[0]), self.name) for r in requests]


class ValueStrategy:
    name = 'value'

    def __init__(self, model, config, lock=None):
        from rl.search.combat_value_search import CombatValueSearch
        if 'combat_legal' in model.observation_space.spaces:
            raise ValueError('joint-v2 supports policy or native search, not legacy value search')
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
    autocast = (bf16_autocast if getattr(args, 'leaf_amp', 'off') == 'bf16'
                else nullcontext)
    combat = PolicyStrategy(model, deterministic=not args.stochastic, lock=lock, autocast=autocast)
    if getattr(getattr(model.policy, 'core', None), 'frozen_upper', False):
        return PhaseStrategy(FrozenUpperStrategy(model, lock), {'combat': combat})
    return combat


def native_strategy(model, args, lock=None):
    if (getattr(getattr(model.policy, 'core', None), 'frozen_upper', False) and
            getattr(args, 'native_search_threads', 0) < 1):
        raise ValueError('frozen-upper search requires --native-search-threads >= 1 for complete run exits')
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
        backend = CombatLeafGraphs(
            model.policy.core,
            model.device,
            amp=getattr(args, 'leaf_amp', 'off'),
            capacity=32,
            batch_multiple=getattr(args, 'leaf_graph_batch_multiple', 0),
            compile_forward=getattr(args, 'leaf_compile', True),
        )
    modes = {'puct': 0, 'gumbel': 1, 'full': 2, 'strict': 2}
    config = NativeSearchConfig.from_model(model, simulations=args.simulations, depth=args.depth,
        leaf_batch_size=args.leaf_batch_size, root_search_mode=modes[args.search_mode],
        paper_full_gumbel=args.search_mode == 'strict',
        fixed_worldline=getattr(args, 'fixed_worldline', None))
    prior_temperature = getattr(args, 'prior_temperature', None)
    if prior_temperature is not None:
        prior_temperature = float(prior_temperature)
        if not np.isfinite(prior_temperature) or prior_temperature <= 0.0:
            raise ValueError('prior temperature must be finite and positive')
        # NativeSearchConfig canonicalizes strict paper settings in
        # __post_init__. Apply an explicit evaluation override afterwards;
        # the native strict branch still owns all other canonical defaults.
        object.__setattr__(config, 'prior_temperature', prior_temperature)
    gumbel_scale = getattr(args, 'gumbel_scale', None)
    if gumbel_scale is not None:
        gumbel_scale = float(gumbel_scale)
        if not np.isfinite(gumbel_scale) or gumbel_scale < 0.0:
            raise ValueError('gumbel scale must be finite and non-negative')
        object.__setattr__(config, 'gumbel_scale', gumbel_scale)
    root_autocast = (bf16_autocast if getattr(args, 'leaf_amp', 'off') == 'bf16'
                     else nullcontext)
    return PhaseStrategy(policy_strategy(model, args, lock), {'combat': NativeStrategy(
        model, config, master_nonce=getattr(model, 'nosl_is_az_master_nonce', 0), lock=lock,
        leaf_backend=backend,
        native_search_threads=getattr(args, 'native_search_threads', 0),
        root_autocast=root_autocast)},
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
    config = TurnPlannerConfig(
        worlds=getattr(args, 'planner_worlds', 2),
        node_budget=getattr(args, 'planner_nodes', 512),
        beam_width=getattr(args, 'planner_beam', 8),
        max_depth=getattr(args, 'planner_depth', 16),
        lookahead_turns=getattr(args, 'planner_turns', 1),
        leaf_profile=getattr(args, 'planner_leaf_profile', 'time-clock-v9'),
        master_nonce=getattr(args, 'planner_nonce', 20260918),
    )
    return PhaseStrategy(
        policy_strategy(model, args, lock),
        {'combat': TurnPlannerStrategy(config)},
    )


# Registration is deliberately a plain mapping, not discovery/import magic.
# Embedders can pass any DecisionStrategy directly without registering a CLI.
def small_playout_strategy(model, args, lock=None, *, paired_worldlines=False, fixed_reference=False):
    from env.search.small_playout import SEARCH_CONTRACT
    from rl.policies.token_regret import OnlineTokenRegretPolicy, TOKEN_REGRET_CONTRACT
    from rl.policies.search_budget import AdaptiveTokenRegretPolicy, BUDGET_CONTRACT
    lower = getattr(model.policy, 'small_playout', None)
    expected = BUDGET_CONTRACT if isinstance(lower, AdaptiveTokenRegretPolicy) else TOKEN_REGRET_CONTRACT if isinstance(lower, OnlineTokenRegretPolicy) else SEARCH_CONTRACT
    if (getattr(args, 'task', 'wholerun') != 'wholerun'
            or C.policy_contract(model.observation_space) != 'autoregressive-v2'
            or not hasattr(model.policy, 'small_playout')
            or getattr(model, 'small_playout_search_contract', None) != expected):
        raise ValueError('small-playout evaluation requires an online autoregressive-v2 whole-run checkpoint with the current search contract')
    if getattr(args, 'trace', None) or getattr(args, 'compare_combats', 0):
        raise ValueError('small-playout does not support the separate combat-audit wrapper; use trajectory-output')
    threads = getattr(args, 'native_search_threads', 0)
    if threads < 0:
        raise ValueError('native search threads must be non-negative')
    fixed_budget = getattr(args, 'simulations', None) if fixed_reference else None
    if fixed_reference and fixed_budget is None:
        raise ValueError('fixed-budget-token requires an explicit --simulations budget')
    for name in ('simulations', 'depth', 'prior_temperature', 'gumbel_scale', 'fixed_worldline', 'leaf_batch_size'):
        if fixed_reference and name == 'simulations':
            continue
        if getattr(args, name, None) is not None:
            raise ValueError('small-playout uses the collection search budget; cannot override '+name)
    collector = None
    if getattr(args, 'budget_dataset', None):
        from rl.algorithms.budget_calibration import BudgetSampleCollector
        collector = BudgetSampleCollector(args.budget_dataset, lower)
    return PhaseStrategy(policy_strategy(model, args, lock), {'combat':
        OnlinePlayoutEvaluationStrategy(model.policy.small_playout,
            threads=max(1, threads), paired_worldlines=paired_worldlines, fixed_budget=fixed_budget,
            cost_price=getattr(args, 'search_cost_price', None), sample_collector=collector)})


STRATEGIES = {'policy': policy_strategy, 'native': native_strategy, 'value': value_strategy,
              'leftmost': lambda model, args, lock=None: LeftmostStrategy(), 'rules': rule_strategy,
              'turn-planner': turn_planner_strategy, 'small-playout': small_playout_strategy,
              'paired-worldlines': partial(small_playout_strategy, paired_worldlines=True),
              'fixed-budget-token': partial(small_playout_strategy, fixed_reference=True)}


def compose_strategy(upper_model, args, combat_model=None, *, trace=None):
    """Compose independently chosen upper/combat components outside the runner."""
    frozen = getattr(getattr(getattr(upper_model, 'policy', None), 'core', None), 'frozen_upper', False)
    if frozen and (getattr(args, 'upper_strategy', 'model') != 'model' or getattr(args, 'route', 'model') != 'model'):
        raise ValueError('frozen upper evaluation cannot override its actor or map route')
    if frozen and (args.solver not in ('policy', 'native') or combat_model is not None):
        raise ValueError('frozen-upper evaluation supports its own lower with policy/native only')
    if combat_model is not None and getattr(getattr(getattr(combat_model, 'policy', None), 'core', None), 'frozen_upper', False):
        raise ValueError('use the complete frozen-upper checkpoint; a separate upper would change its teacher contract')
    strategy = STRATEGIES[args.solver](combat_model if combat_model is not None else upper_model, args)
    if args.solver == 'turn-planner':
        strategy.routes['combat'].trace = trace
    if combat_model is not None:
        # A pre-reward exit anchor is a forced settlement row, not a live
        # battle for a separately loaded search policy to explore.
        spaces = getattr(getattr(upper_model, 'observation_space', None), 'spaces', {})
        predicates = ({'combat': lambda r: not np.any(r.observation.get('combat_exit_anchor', False))}
                      if 'combat_exit_anchor' in spaces else None)
        strategy = PhaseStrategy(policy_strategy(upper_model, args), {'combat': strategy},
                                 predicates=predicates)
    combat_only_upper = getattr(getattr(getattr(upper_model,'policy',None),'core',None),'combat_only',False)
    if getattr(args, 'upper_strategy', 'model') == 'rules' or combat_only_upper:
        from rl.evaluation.baselines.overworld_rules import OverworldRuleStrategy
        upper = OverworldRuleStrategy(fallback=None if combat_only_upper else policy_strategy(upper_model, args), trace=trace)
        strategy = PhaseStrategy(upper, {'combat': strategy})
    if getattr(args, 'route', 'model') != 'model':
        strategy = PhaseStrategy(strategy, {'map': MapRouteStrategy()})
    return strategy
