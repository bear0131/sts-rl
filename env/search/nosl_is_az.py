"""Information-set AlphaZero style search for whole-run combat training.

The search deliberately keeps the policy/value update in the learner process,
while the live simulator state stays in a VecEnv worker.  A simulation clones
and redeterminizes *once at the root* and then advances that branch in place.
This is the important NOSL property: no simulation can observe the hidden
future RNG of the current world line.
"""

from __future__ import annotations

import math
from itertools import count
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from env.core import encode
from env.search.nosl_sampler import NOSLSampler


@dataclass(frozen=True)
class NOSLISAZConfig:
    simulations: int = 8
    depth: int = 3
    # Retained as a checkpoint/CLI compatibility field.  AlphaZero-style
    # search evaluates the leaf directly and never consumes this value.
    rollout_steps: int = 0
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.0
    dirichlet_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("NOSL-IS-AZ simulations must be positive")
        if self.depth < 0:
            raise ValueError(
                "NOSL-IS-AZ depth must be non-negative (0 means unlimited)"
            )
        if self.rollout_steps < 0:
            raise ValueError("NOSL-IS-AZ rollout_steps must be non-negative")
        if self.c_puct < 0.0:
            raise ValueError("NOSL-IS-AZ c_puct must be non-negative")
        if self.temperature < 0.0:
            raise ValueError("NOSL-IS-AZ temperature must be non-negative")
        if self.dirichlet_alpha < 0.0:
            raise ValueError("NOSL-IS-AZ Dirichlet alpha must be non-negative")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError(
                "NOSL-IS-AZ Dirichlet epsilon must be in [0, 1]"
            )
        if self.dirichlet_epsilon > 0.0 and self.dirichlet_alpha <= 0.0:
            raise ValueError(
                "NOSL-IS-AZ Dirichlet alpha must be positive when epsilon > 0"
            )


@dataclass(frozen=True)
class NOSLISAZResult:
    action: int
    visit_policy: np.ndarray
    root_value: float
    root_visits: np.ndarray
    simulations: int

    def as_dict(self) -> dict[str, Any]:
        # VecEnv IPC handles ordinary Python/numpy values reliably; avoid
        # requiring callers to import this dataclass in subprocess workers.
        return {
            "action": int(self.action),
            "visit_policy": np.asarray(self.visit_policy, dtype=np.float32),
            "root_value": float(self.root_value),
            "root_visits": np.asarray(self.root_visits, dtype=np.int32),
            "simulations": int(self.simulations),
        }


@dataclass
class _Edge:
    visits: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0


def _redeterminized_root(env, sampler: NOSLSampler, state_nonce: int, index: int):
    branch = env.fork_combat()
    branch.thick.bc = sampler.clone_and_redeterminize(
        branch.thick.bc,
        state_nonce=state_nonce,
        simulation_index=index,
    )
    invalidate = getattr(branch, "_invalidate_backend_cache", None)
    if invalidate is not None:
        invalidate()
    return branch


def _terminal_value(branch) -> float:
    hp = branch.thick.bc.projected_exit_hp(branch.thick.gc)
    return min(1.0, float(max(0.0, hp)) / encode.HP_SCALE)


def _masked_prior(logits, legal: list[int], action_count: int) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != action_count:
        raise ValueError(
            f"prior logits have {values.size} actions, expected {action_count}"
        )
    selected = values[np.asarray(legal, dtype=np.int64)]
    selected = selected - np.max(selected)
    probabilities = np.exp(np.clip(selected, -80.0, 80.0))
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.ones(len(legal), dtype=np.float64)
        total = float(len(legal))
    result = np.zeros(action_count, dtype=np.float64)
    result[np.asarray(legal, dtype=np.int64)] = probabilities / total
    return result


class NOSLISAZSession:
    """Incremental PUCT session used for batched learner-side leaf inference.

    A session advances exactly one simulation between ``next_request`` and
    ``backprop``.  The worker retains the cloned branch while the learner
    evaluates its public observation, so sibling simulations still use the
    accumulated ``N/W/Q`` statistics and each branch is redeterminized only at
    its root.
    """

    def __init__(
        self,
        env,
        legal_mask: np.ndarray,
        *,
        prior_logits,
        sampler: NOSLSampler,
        state_nonce: int,
        decision_index: int,
        config: NOSLISAZConfig,
    ) -> None:
        legal = [int(action) for action in np.flatnonzero(legal_mask)]
        if not legal:
            raise ValueError("NOSL-IS-AZ root has no legal action")
        self.env = env
        self.legal = legal
        self.action_count = int(np.asarray(legal_mask).size)
        self.prior = _masked_prior(prior_logits, legal, self.action_count)
        if config.dirichlet_epsilon > 0.0:
            # The root noise is sampled from a nonce-derived stream, so it is
            # independent of the live world's hidden RNG and reproducible for
            # a given information-set root.
            noise_seed = sampler.nonce(
                state_nonce=state_nonce,
                simulation_index=decision_index * 1_000_000 + 777_777,
            )
            noise_rng = np.random.default_rng(noise_seed)
            eta = noise_rng.dirichlet(
                np.full(len(legal), config.dirichlet_alpha, dtype=np.float64)
            )
            self.prior[np.asarray(legal, dtype=np.int64)] = (
                (1.0 - config.dirichlet_epsilon)
                * self.prior[np.asarray(legal, dtype=np.int64)]
                + config.dirichlet_epsilon * eta
            )
        self.node_priors: dict[tuple[int, ...], np.ndarray] = {
            (): self.prior
        }
        self.edges: dict[tuple[int, ...], _Edge] = {}
        self.node_visits: dict[tuple[int, ...], int] = {(): 0}
        self.root_visits = np.zeros(self.action_count, dtype=np.int32)
        self.root_total = 0.0
        self.simulation = 0
        self.sampler = sampler
        self.state_nonce = int(state_nonce)
        self.decision_index = int(decision_index)
        self.config = config
        self.pending: tuple[Any, list[int], np.ndarray] | None = None

    def _select_branch(self):
        branch = _redeterminized_root(
            self.env,
            self.sampler,
            self.state_nonce,
            self.decision_index * self.config.simulations + self.simulation,
        )
        path: list[int] = []
        mask = np.asarray(branch.action_masks(), dtype=bool)
        for _depth in (count() if self.config.depth == 0 else range(self.config.depth)):
            prefix = tuple(path)
            node_prior = self.node_priors.get(prefix)
            if node_prior is None:
                # AlphaZero expands and evaluates the first unseen leaf.  Its
                # policy prior is installed by backprop() for later visits;
                # do not walk through an unexpanded node with a synthetic
                # uniform policy.
                return branch, path, mask
            current_legal = [int(action) for action in np.flatnonzero(mask)]
            if not current_legal:
                break
            parent_visits = self.node_visits.get(prefix, 0)
            def score(action: int) -> tuple[float, int]:
                edge = self.edges.get((*prefix, action))
                visits = edge.visits if edge is not None else 0
                mean = edge.mean if edge is not None else 0.0
                return (
                    mean
                    + self.config.c_puct
                    * float(node_prior[action])
                    * math.sqrt(max(parent_visits, 1))
                    / (1 + visits),
                    -action,
                )
            action = max(current_legal, key=score)
            terminated, next_mask = branch.search_step(action, validate=False)
            path.append(action)
            if terminated:
                break
            mask = np.asarray(next_mask, dtype=bool)
        return branch, path, mask

    def _backup(self, value: float, path: list[int]) -> None:
        value = float(value)
        if not np.isfinite(value):
            value = 0.0
        self.root_total += value
        self.node_visits[()] = self.node_visits.get((), 0) + 1
        if path:
            self.root_visits[path[0]] += 1
        for length in range(1, len(path) + 1):
            prefix = tuple(path[:length])
            self.node_visits[prefix] = self.node_visits.get(prefix, 0) + 1
            edge = self.edges.setdefault(prefix, _Edge())
            edge.visits += 1
            edge.total += value
        self.simulation += 1

    def _result(self) -> dict[str, Any]:
        root_visits = self.root_visits
        if int(root_visits.sum()) == 0:
            root_visits[np.asarray(self.legal, dtype=np.int64)] = 1
        if self.config.temperature == 0.0:
            target = np.zeros(self.action_count, dtype=np.float32)
            target[int(np.argmax(root_visits))] = 1.0
        else:
            weights = np.zeros(self.action_count, dtype=np.float64)
            legal_indices = np.asarray(self.legal, dtype=np.int64)
            weights[legal_indices] = np.power(
                root_visits[legal_indices].astype(np.float64),
                1.0 / self.config.temperature,
            )
            total = float(weights.sum())
            target = (weights / total if total > 0.0 else self.prior).astype(
                np.float32
            )
        target = np.asarray(target, dtype=np.float32)
        target_sum = float(target.sum())
        if not np.isfinite(target_sum) or target_sum <= 0.0:
            target = self.prior.astype(np.float32)
            target_sum = float(target.sum())
        target /= target_sum
        rng = np.random.default_rng(
            self.sampler.nonce(
                state_nonce=self.state_nonce,
                simulation_index=self.decision_index * 1_000_000 + 999_999,
            )
        )
        action = int(rng.choice(self.action_count, p=target))
        return {
            "action": action,
            "visit_policy": target,
            "root_value": self.root_total / self.config.simulations,
            "root_visits": root_visits,
            "simulations": self.config.simulations,
        }

    def next_request(self) -> dict[str, Any]:
        """Return a leaf request or a completed result."""
        while self.simulation < self.config.simulations:
            branch, path, mask = self._select_branch()
            if branch.thick.bc.is_battle_over:
                self._backup(_terminal_value(branch), path)
                continue
            self.pending = (branch, path, mask)
            return {
                "done": False,
                "observation": branch._obs(),
                "action_mask": np.asarray(mask, dtype=bool),
            }
        return {"done": True, "result": self._result()}

    def backprop(self, value: float, policy_logits=None) -> dict[str, Any]:
        if self.pending is None:
            raise RuntimeError("NOSL-IS-AZ session has no pending leaf")
        _branch, path, mask = self.pending
        if policy_logits is not None:
            legal = [int(action) for action in np.flatnonzero(mask)]
            if legal:
                self.node_priors[tuple(path)] = _masked_prior(
                    policy_logits, legal, self.action_count
                )
        self.pending = None
        self._backup(float(value), path)
        return self.next_request()


class NOSLISAZSearch:
    """In-process NOSL PUCT search with direct value-network leaf bootstrap."""

    def __init__(self, config: NOSLISAZConfig | None = None) -> None:
        self.config = config or NOSLISAZConfig()

    def choose(
        self,
        env,
        legal_mask: np.ndarray,
        *,
        prior_logits,
        sampler: NOSLSampler,
        state_nonce: int,
        decision_index: int,
        leaf_value: float | None = None,
        value_fn: Callable[[Any], float] | None = None,
    ) -> NOSLISAZResult:
        """Run NOSL PUCT and evaluate each non-terminal leaf directly.

        ``value_fn`` is the AlphaZero interface for callers that can evaluate
        branch observations locally.  ``leaf_value`` remains a compatibility
        shortcut for callers that have one scalar bootstrap available.  The
        staged ``NOSLISAZSession`` used by training evaluates each public leaf
        in the learner and then backprops its value, so it does not use a
        fixed-policy rollout and does not expose hidden branch state.
        """
        if leaf_value is not None and value_fn is not None:
            raise ValueError("pass leaf_value or value_fn, not both")
        legal = [int(action) for action in np.flatnonzero(legal_mask)]
        if not legal:
            raise ValueError("NOSL-IS-AZ root has no legal action")
        action_count = int(np.asarray(legal_mask).size)
        prior = _masked_prior(prior_logits, legal, action_count)
        edges: dict[tuple[int, ...], _Edge] = {}
        node_visits: dict[tuple[int, ...], int] = {(): 0}
        root_total = 0.0
        root_visits = np.zeros(action_count, dtype=np.int32)

        for simulation in range(self.config.simulations):
            branch = _redeterminized_root(
                env,
                sampler,
                state_nonce,
                decision_index * self.config.simulations + simulation,
            )
            path: list[int] = []
            mask = np.asarray(branch.action_masks(), dtype=bool)
            for depth in (count() if self.config.depth == 0 else range(self.config.depth)):
                current_legal = [int(a) for a in np.flatnonzero(mask)]
                if not current_legal:
                    break
                prefix = tuple(path)
                parent_visits = node_visits.get(prefix, 0)
                candidates = [
                    action for action in current_legal
                    if (edges.get((*prefix, action)) is None
                        or edges[(*prefix, action)].visits == 0)
                ]
                if candidates:
                    rng = np.random.default_rng(
                        sampler.nonce(
                            state_nonce=state_nonce,
                            simulation_index=(
                                decision_index * 1_000_000
                                + simulation * 100
                                + depth
                            ),
                        )
                    )
                    action = int(rng.choice(candidates))
                else:
                    # Root receives the learned policy prior.  The worker does
                    # not have a policy callback for imagined observations, so
                    # deeper nodes use a uniform legal-action prior.
                    p = {
                        a: (
                            float(prior[a])
                            if not path
                            else 1.0 / len(current_legal)
                        )
                        for a in current_legal
                    }
                    action = max(
                        current_legal,
                        key=lambda a: (
                            edges[(*prefix, a)].mean
                            + self.config.c_puct
                            * p[a]
                            * math.sqrt(max(parent_visits, 1))
                            / (1 + edges[(*prefix, a)].visits),
                            -a,
                        ),
                    )
                terminated, next_mask = branch.search_step(
                    action, validate=False
                )
                path.append(action)
                if terminated:
                    break
                mask = np.asarray(next_mask, dtype=bool)

                # With no depth cap, expand one unseen edge then evaluate it.
                # This also bounds a simulation in non-terminating toy worlds.
                if self.config.depth == 0 and candidates:
                    break

            if branch.thick.bc.is_battle_over:
                value = _terminal_value(branch)
            elif value_fn is not None:
                value = float(value_fn(branch))
            elif leaf_value is not None:
                value = float(leaf_value)
            else:
                # A value callback is optional for model-free unit tests and
                # diagnostic searches.  Zero is the neutral one-player value.
                value = 0.0
            if not np.isfinite(value):
                value = 0.0
            root_total += value
            node_visits[()] = node_visits.get((), 0) + 1
            if path:
                root_visits[path[0]] += 1
            for length in range(1, len(path) + 1):
                prefix = tuple(path[:length])
                node_visits[prefix] = node_visits.get(prefix, 0) + 1
                edge = edges.setdefault(prefix, _Edge())
                edge.visits += 1
                edge.total += value

        if int(root_visits.sum()) == 0:
            root_visits[np.asarray(legal, dtype=np.int64)] = 1
        if self.config.temperature == 0.0:
            target = np.zeros(action_count, dtype=np.float32)
            target[int(np.argmax(root_visits))] = 1.0
        else:
            weights = np.zeros(action_count, dtype=np.float64)
            legal_indices = np.asarray(legal, dtype=np.int64)
            weights[legal_indices] = np.power(
                root_visits[legal_indices].astype(np.float64),
                1.0 / self.config.temperature,
            )
            total = float(weights.sum())
            target = (weights / total if total > 0.0 else prior).astype(
                np.float32
            )
        target = np.asarray(target, dtype=np.float32)
        target_sum = float(target.sum())
        if not np.isfinite(target_sum) or target_sum <= 0.0:
            target = prior.astype(np.float32)
            target_sum = float(target.sum())
        target /= target_sum
        rng = np.random.default_rng(
            sampler.nonce(
                state_nonce=state_nonce,
                simulation_index=decision_index * 1_000_000 + 999_999,
            )
        )
        action = int(rng.choice(action_count, p=target))
        return NOSLISAZResult(
            action=action,
            visit_policy=target,
            root_value=(root_total / self.config.simulations),
            root_visits=root_visits,
            simulations=self.config.simulations,
        )


def search_result_dict(result: NOSLISAZResult) -> dict[str, Any]:
    """Public helper for worker methods and unit tests."""
    return result.as_dict()
