"""Model-free combat search policies built on the NOSL sampler.

All methods operate on a thin ``SlayTheSpireEnv`` combat branch.  They never
read a policy/value head.  Every simulator transition used for lookahead is
made from a cloned branch whose future RNG is redeterminized by NOSL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl.nosl_sampler import NOSLSampler


# Combat encounters normally terminate far below this bound. It only protects
# the interactive dashboard from a malformed or stalled simulator state when
# the caller requests depth=0 (automatic expansion).
MCTS_AUTO_EXPANSION_LIMIT = 4096


def _redet_branch(env, sampler: NOSLSampler, state_nonce: int, index: int):
    branch = env.fork_combat()
    thick = getattr(branch, "thick", None)
    battle_context = getattr(thick, "bc", None)
    if battle_context is None or not hasattr(battle_context, "nosl_redeterminize"):
        raise RuntimeError(
            "NOSL search requires a BattleContext with nosl_redeterminize()"
        )
    branch.thick.bc = sampler.clone_and_redeterminize(
        battle_context, state_nonce=state_nonce, simulation_index=index
    )
    invalidate = getattr(branch, "_invalidate_backend_cache", None)
    if invalidate is not None:
        invalidate()
    return branch


def _terminal_hp(branch) -> float:
    return float(max(0, branch.thick.bc.projected_exit_hp(branch.thick.gc)))


def _apply(env, action: int, sampler: NOSLSampler, state_nonce: int, index: int):
    branch = _redet_branch(env, sampler, state_nonce, index)
    advances = bool(branch.combat_action_advances(int(action)))
    terminated, mask = branch.search_step(int(action), validate=False)
    if terminated:
        mask = np.zeros(0, dtype=bool)
    return branch, None, np.asarray(mask, dtype=bool), bool(terminated), advances


def random_action(
    mask: np.ndarray,
    *,
    sampler: NOSLSampler,
    state_nonce: int,
    index: int,
) -> int:
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        raise ValueError("combat state has no legal action")
    rng = np.random.default_rng(
        sampler.nonce(state_nonce=state_nonce, simulation_index=index)
    )
    return int(rng.choice(legal))


def random_rollout(
    env,
    *,
    sampler: NOSLSampler,
    state_nonce: int,
    index: int,
    max_steps: int,
) -> tuple[float, bool]:
    """Return terminal HP and whether the rollout timed out."""
    branch = _redet_branch(env, sampler, state_nonce, index)
    terminated = False
    mask = np.asarray(branch.action_masks(), dtype=bool)
    for step in range(max_steps):
        if terminated:
            break
        if not mask.any():
            return -1.0e6, True
        action = random_action(
            mask,
            sampler=sampler,
            state_nonce=state_nonce,
            index=index * 10000 + step,
        )
        terminated, next_mask = branch.search_step(action, validate=False)
        mask = (
            np.asarray(next_mask, dtype=bool)
            if next_mask is not None
            else np.zeros(0, dtype=bool)
        )
    if not terminated:
        return -1.0e6, True
    return _terminal_hp(branch), False


@dataclass(frozen=True)
class BeamConfig:
    depth: int = 2
    width: int = 4
    rollout_steps: int = 128
    max_branching: int = 12


@dataclass
class _BeamNode:
    env: Any
    first_action: int
    depth: int
    terminal: bool
    score: float = -1.0e6


class NOSLBeamSearch:
    """Breadth-limited model-free beam search with random terminal rollouts."""

    def __init__(self, config: BeamConfig | None = None):
        self.config = config or BeamConfig()
        if self.config.depth < 1 or self.config.width < 1:
            raise ValueError("beam depth and width must be positive")
        if self.config.rollout_steps < 1 or self.config.max_branching < 1:
            raise ValueError("beam rollout and branching limits must be positive")

    def choose(
        self,
        env,
        mask: np.ndarray,
        *,
        sampler: NOSLSampler,
        state_nonce: int,
        decision_index: int,
    ) -> int:
        roots = [int(action) for action in np.flatnonzero(mask)]
        if not roots:
            raise ValueError("combat state has no legal action")
        frontier: list[_BeamNode] = []
        serial = 0
        for action in roots[: self.config.max_branching]:
            child, _obs, _mask, terminal, _advances = _apply(
                env,
                action,
                sampler,
                state_nonce,
                decision_index * 100000 + serial,
            )
            serial += 1
            frontier.append(_BeamNode(child, action, 1, terminal))

        for depth in range(1, self.config.depth):
            expanded: list[_BeamNode] = []
            for node in frontier:
                if node.terminal:
                    expanded.append(node)
                    continue
                child_mask = np.asarray(node.env.action_masks(), dtype=bool)
                actions = [
                    int(action)
                    for action in np.flatnonzero(child_mask)[: self.config.max_branching]
                ]
                for action in actions:
                    child, _obs, _mask, terminal, _advances = _apply(
                        node.env,
                        action,
                        sampler,
                        state_nonce,
                        decision_index * 100000 + serial,
                    )
                    serial += 1
                    expanded.append(
                        _BeamNode(child, node.first_action, depth + 1, terminal)
                    )
            if not expanded:
                break
            frontier = self._score_and_keep(
                expanded,
                sampler=sampler,
                state_nonce=state_nonce,
                decision_index=decision_index,
                serial_start=serial,
            )
            serial += len(frontier)

        frontier = self._score_and_keep(
            frontier,
            sampler=sampler,
            state_nonce=state_nonce,
            decision_index=decision_index,
            serial_start=serial,
        )
        best = max(frontier, key=lambda node: (node.score, -node.first_action))
        return int(best.first_action)

    def _score_and_keep(
        self,
        nodes: list[_BeamNode],
        *,
        sampler: NOSLSampler,
        state_nonce: int,
        decision_index: int,
        serial_start: int,
    ) -> list[_BeamNode]:
        for offset, node in enumerate(nodes):
            if node.terminal:
                node.score = _terminal_hp(node.env)
            else:
                node.score, _timeout = random_rollout(
                    node.env,
                    sampler=sampler,
                    state_nonce=state_nonce,
                    index=decision_index * 1000000 + serial_start + offset,
                    max_steps=self.config.rollout_steps,
                )
        return sorted(nodes, key=lambda node: (node.score, -node.first_action), reverse=True)[
            : self.config.width
        ]


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 32
    depth: int = 3
    rollout_steps: int = 128
    exploration: float = 1.41421356237
    max_branching: int = 12
    temperature: float = 0.0


@dataclass
class _MCTSStat:
    visits: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else -1.0e6


class NOSLMCTS:
    """UCT search with NOSL transitions and model-free random rollouts."""

    def __init__(self, config: MCTSConfig | None = None):
        self.config = config or MCTSConfig()
        # Keep the latest root statistics available to interactive callers.
        # ``choose`` still returns the selected action for compatibility, while
        # dashboards can distinguish search statistics from the policy prior.
        self.last_root_stats: dict[int, dict[str, float | int]] = {}
        if self.config.simulations < 1 or self.config.depth < 0:
            raise ValueError(
                "MCTS simulations must be positive and depth must be "
                "non-negative (0 means automatic expansion)"
            )
        if self.config.rollout_steps < 1 or self.config.max_branching < 1:
            raise ValueError("MCTS rollout and branching limits must be positive")
        if self.config.temperature < 0.0:
            raise ValueError("MCTS temperature must be non-negative")

    def choose(
        self,
        env,
        mask: np.ndarray,
        *,
        sampler: NOSLSampler,
        state_nonce: int,
        decision_index: int,
    ) -> int:
        root_actions = [int(action) for action in np.flatnonzero(mask)]
        if not root_actions:
            raise ValueError("combat state has no legal action")
        stats: dict[tuple[int, ...], _MCTSStat] = {}
        for simulation in range(self.config.simulations):
            branch = env
            path: list[int] = []
            # A zero depth asks the search to keep expanding until the
            # simulator reaches a terminal combat state.  The rollout budget
            # remains a defensive bound for malformed/stalled simulator
            # states; normal combats terminate well before it.
            depth_limit = (
                self.config.depth
                if self.config.depth > 0
                else MCTS_AUTO_EXPANSION_LIMIT
            )
            for depth in range(depth_limit):
                child_mask = np.asarray(branch.action_masks(), dtype=bool)
                legal = [
                    int(action)
                    for action in np.flatnonzero(child_mask)[: self.config.max_branching]
                ]
                if not legal:
                    break
                parent = stats.setdefault(tuple(path), _MCTSStat())
                action = self._select_action(
                    legal,
                    parent,
                    stats,
                    tuple(path),
                    sampler=sampler,
                    state_nonce=state_nonce,
                    index=decision_index * 100000 + simulation * 100 + depth,
                )
                branch, _obs, _mask, terminal, _advances = _apply(
                    branch,
                    action,
                    sampler,
                    state_nonce,
                    decision_index * 100000 + simulation * 100 + depth,
                )
                path.append(action)
                if terminal:
                    break
            if branch.thick.bc.is_battle_over:
                value, _timeout = _terminal_hp(branch), False
            else:
                value, _timeout = random_rollout(
                    branch,
                    sampler=sampler,
                    state_nonce=state_nonce,
                    index=decision_index * 1000000 + simulation,
                    max_steps=self.config.rollout_steps,
                )
            for length in range(1, len(path) + 1):
                node = stats.setdefault(tuple(path[:length]), _MCTSStat())
                node.visits += 1
                node.total += value
            root = stats.setdefault((), _MCTSStat())
            root.visits += 1
            root.total += value

        # Save only root-level data.  The full tree can be large, and callers
        # only need enough information to explain the action selected at this
        # decision.  A missing root edge is represented with zero visits and
        # the same sentinel mean used by the selector.
        self.last_root_stats = {}
        for action in root_actions:
            stat = stats.get((action,), _MCTSStat())
            self.last_root_stats[int(action)] = {
                "visits": int(stat.visits),
                "value": float(stat.mean),
            }

        def key(action: int):
            stat = stats.get((action,), _MCTSStat())
            return stat.mean, stat.visits, -action

        if self.config.temperature > 0.0:
            visits = np.asarray(
                [stats.get((action,), _MCTSStat()).visits for action in root_actions],
                dtype=np.float64,
            )
            weights = np.power(visits, 1.0 / self.config.temperature)
            total = float(weights.sum())
            if not np.isfinite(total) or total <= 0.0:
                weights = np.ones(len(root_actions), dtype=np.float64)
                total = float(len(root_actions))
            rng = np.random.default_rng(
                sampler.nonce(
                    state_nonce=state_nonce,
                    simulation_index=decision_index * 1_000_000 + 999_999,
                )
            )
            return int(rng.choice(root_actions, p=weights / total))
        return max(root_actions, key=key)

    def _select_action(
        self,
        legal: list[int],
        parent: _MCTSStat,
        stats: dict[tuple[int, ...], _MCTSStat],
        path: tuple[int, ...],
        *,
        sampler: NOSLSampler,
        state_nonce: int,
        index: int,
    ) -> int:
        unvisited = [action for action in legal if not stats.get((*path, action), _MCTSStat()).visits]
        if unvisited:
            rng = np.random.default_rng(
                sampler.nonce(state_nonce=state_nonce, simulation_index=index)
            )
            return int(rng.choice(unvisited))
        log_parent = math.log(max(parent.visits, 1))
        return max(
            legal,
            key=lambda action: (
                stats[(*path, action)].mean
                + self.config.exploration
                * math.sqrt(log_parent / stats[(*path, action)].visits),
                -action,
            ),
        )
