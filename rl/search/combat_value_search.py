"""Inference-time combat lookahead guided by the policy's own value head.

The search is deliberately combat-local. It clones ``BattleContext`` and stops
at the combat boundary, so combat and overworld critics are never compared.
Policy probabilities only prune the branching factor; completed search leaves
are ranked by the combat value estimate. For hierarchical combat-HP checkpoints,
a terminal leaf uses the exact training target (remaining HP / ``HP_SCALE``).
The beam reserves same-depth root-action coverage when capacity permits, and an
override requires a configurable value margin so critic noise does not churn it.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from env.core import encode
from env.core.boss_value import context_terminal_target
from env.core.combat_value import combat_terminal_target
from env.search.nosl_sampler import NOSLSampler
from rl.policies.model import observation_for_policy


@dataclass(frozen=True)
class CombatValueSearchConfig:
    depth: int = 3
    beam_width: int = 16
    policy_top_k: int = 4
    root_top_k: int = 0
    argument_top_k: int = 5
    argument_beam_width: int = 32
    max_argument_steps: int = 12
    node_budget: int = 512
    min_value_gain: float = 0.01
    terminal_hp: bool | None = None
    boss_transition_target: bool | None = None
    nosl: bool = True
    master_nonce: int = 0x4E4F534C_7631

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("combat search depth must be at least 1")
        if self.beam_width < 1:
            raise ValueError("combat search beam width must be at least 1")
        if self.policy_top_k < 1:
            raise ValueError("combat search policy top-k must be at least 1")
        if self.root_top_k < 0:
            raise ValueError("combat search root top-k must be non-negative")
        if self.argument_top_k < 1:
            raise ValueError("combat search argument top-k must be at least 1")
        if self.argument_beam_width < 1:
            raise ValueError(
                "combat search argument beam width must be at least 1"
            )
        if self.max_argument_steps < 1:
            raise ValueError("combat search max argument steps must be at least 1")
        if self.node_budget < 1:
            raise ValueError("combat search node budget must be at least 1")
        if self.min_value_gain < 0.0:
            raise ValueError(
                "combat search minimum value gain must be non-negative"
            )
        if int(self.master_nonce) < 0:
            raise ValueError("combat search master nonce must be non-negative")


@dataclass(frozen=True)
class CombatSearchCandidate:
    action: int
    value: float
    depth: int
    terminal: bool
    root_probability: float
    path: tuple[int, ...]


@dataclass(frozen=True)
class CombatSearchResult:
    action: int
    model_action: int
    root_value: float
    value: float
    depth_reached: int
    expanded_nodes: int
    evaluated_states: int
    elapsed_seconds: float
    principal_variation: tuple[int, ...]
    candidates: tuple[CombatSearchCandidate, ...]


@dataclass
class _Node:
    env: Any
    obs: dict[str, np.ndarray]
    mask: np.ndarray
    first_action: int | None = None
    root_probability: float = 1.0
    path: tuple[int, ...] = ()
    depth: int = 0
    log_probability: float = 0.0
    terminal: bool = False
    value: float | None = None
    probabilities: np.ndarray | None = None


def add_combat_search_args(parser) -> None:
    parser.add_argument(
        "--combat-search-depth",
        type=int,
        default=0,
        help="Combat value-search depth in completed simulator decisions; 0 disables.",
    )
    parser.add_argument(
        "--combat-search-beam-width",
        type=int,
        default=16,
        help="Number of value-ranked combat states retained at each depth.",
    )
    parser.add_argument(
        "--combat-search-policy-top-k",
        type=int,
        default=4,
        help="Policy-ranked legal actions expanded below the root.",
    )
    parser.add_argument(
        "--combat-search-node-budget",
        type=int,
        default=512,
        help="Maximum cloned action transitions per combat decision.",
    )
    parser.add_argument(
        "--combat-search-min-value-gain",
        type=float,
        default=0.01,
        help="Minimum leaf-value advantage required to override policy argmax.",
    )
    parser.add_argument(
        "--combat-search-nosl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redeterminize every cloned combat branch with independent NOSL RNG.",
    )
    parser.add_argument(
        "--combat-search-master-nonce",
        type=lambda value: int(value, 0),
        default=0x4E4F534C_7631,
        help="Opaque master nonce for NOSL combat-search branches.",
    )


def config_from_args(args) -> CombatValueSearchConfig | None:
    depth = int(args.combat_search_depth)
    if depth <= 0:
        return None
    return CombatValueSearchConfig(
        depth=depth,
        beam_width=int(args.combat_search_beam_width),
        policy_top_k=int(args.combat_search_policy_top_k),
        node_budget=int(args.combat_search_node_budget),
        min_value_gain=float(args.combat_search_min_value_gain),
        nosl=bool(args.combat_search_nosl),
        master_nonce=int(args.combat_search_master_nonce),
    )


class CombatValueSearch:
    """Policy-guided beam search whose leaf objective is combat ``V(s)``."""

    def __init__(
        self,
        model: Any,
        config: CombatValueSearchConfig | None = None,
        *,
        model_lock: Any | None = None,
    ) -> None:
        self.model = model
        self.policy = model.policy
        if getattr(self.policy, 'entity_config', {}).get('boss_win_weight', 0.0):
            raise ValueError('boss utility currently requires native AR search')
        self.config = config or CombatValueSearchConfig()
        self.model_lock = model_lock
        self.expanded_nodes = 0
        self.evaluated_states = 0
        self._terminal_hp = (
            bool(getattr(model, "hierarchical_combat_hp", False))
            if self.config.terminal_hp is None
            else self.config.terminal_hp
        )
        self._boss_transition_target = (
            bool(getattr(model, "nosl_is_az_boss_transition_target", False))
            if self.config.boss_transition_target is None
            else bool(self.config.boss_transition_target)
        )
        self._nosl_sampler = (
            NOSLSampler(self.config.master_nonce) if self.config.nosl else None
        )
        self._search_state_nonce = 0
        self._search_invocation = 0
        set_training_mode = getattr(self.policy, "set_training_mode", None)
        if set_training_mode is not None:
            set_training_mode(False)

    @staticmethod
    def _base_env(env):
        unwrapped = getattr(env, "unwrapped", env)
        if not hasattr(unwrapped, "fork_combat"):
            raise TypeError("environment does not support combat-state forking")
        return unwrapped

    @staticmethod
    def _phase(env) -> str:
        phase = getattr(env, "_phase", None)
        if phase is not None:
            return str(phase())
        return str(env.thick.phase())

    @staticmethod
    def _stack_observations(
        observations: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        keys = tuple(observations[0])
        expected = set(keys)
        for obs in observations[1:]:
            if set(obs) != expected:
                raise ValueError("combat search observations have mismatched fields")
        return {
            key: np.stack([np.asarray(obs[key]) for obs in observations])
            for key in keys
        }

    def _lock_context(self):
        return self.model_lock if self.model_lock is not None else nullcontext()

    def _evaluate(
        self,
        nodes: list[_Node],
        *,
        policy_probabilities: bool,
    ) -> None:
        if not nodes:
            return
        batch = self._stack_observations([node.obs for node in nodes])
        masks = np.stack([node.mask for node in nodes])
        with self._lock_context():
            obs_tensor, _ = self.policy.obs_to_tensor(observation_for_policy(self.policy, batch))
            with torch.inference_mode():
                core = getattr(self.policy, "core", None)
                if core is not None and hasattr(core, "actor_critic"):
                    logits, values = core.actor_critic(obs_tensor)
                    if policy_probabilities:
                        mask_tensor = torch.as_tensor(
                            masks, dtype=torch.bool, device=logits.device
                        )
                        logits = logits.float().masked_fill(
                            ~mask_tensor, torch.finfo(torch.float32).min
                        )
                        probabilities = torch.softmax(logits, dim=-1)
                else:
                    values = self.policy.predict_values(obs_tensor)
                    if policy_probabilities:
                        distribution = self.policy.get_distribution(
                            obs_tensor, action_masks=masks
                        )
                        probabilities = distribution.distribution.probs

        value_array = values.float().reshape(-1).cpu().numpy()
        probability_array = (
            probabilities.float().cpu().numpy()
            if policy_probabilities
            else None
        )
        for idx, node in enumerate(nodes):
            node.value = float(value_array[idx])
            if probability_array is not None:
                node.probabilities = probability_array[idx]
        self.evaluated_states += len(nodes)

    def _populate(self, nodes: list[_Node], *, need_policy: bool) -> None:
        missing = [
            node
            for node in nodes
            if node.value is None
            or (need_policy and node.probabilities is None)
        ]
        self._evaluate(missing, policy_probabilities=need_policy)

    @staticmethod
    def _terminal_hp_value(node: _Node) -> float:
        return combat_terminal_target(node.env.thick.bc)

    def _score_nodes(self, nodes: list[_Node]) -> None:
        terminal = [node for node in nodes if node.terminal]
        live = [node for node in nodes if not node.terminal]
        self._populate(live, need_policy=True)
        if self._terminal_hp:
            for node in terminal:
                # Bind the per-search objective without mutating branch state.
                target = context_terminal_target(
                    enabled=self._boss_transition_target,
                    gc=node.env.thick.gc,
                    bc=node.env.thick.bc,
                )
                node.value = (
                    float(target)
                    if target is not None
                    else self._terminal_hp_value(node)
                )
        else:
            self._populate(terminal, need_policy=False)

    @staticmethod
    def _rank_key(node: _Node) -> tuple[float, float]:
        assert node.value is not None
        return node.value, node.log_probability

    @classmethod
    def _best(cls, nodes: list[_Node], limit: int) -> list[_Node]:
        return sorted(nodes, key=cls._rank_key, reverse=True)[:limit]

    def _prune_frontier(self, nodes: list[_Node], model_action: int) -> list[_Node]:
        """Keep root-action coverage before filling the beam by global value."""
        if len(nodes) <= self.config.beam_width:
            return self._best(nodes, len(nodes))

        by_action: dict[int, list[_Node]] = {}
        for node in nodes:
            assert node.first_action is not None
            by_action.setdefault(node.first_action, []).append(node)
        representatives = {
            action: max(group, key=self._rank_key)
            for action, group in by_action.items()
        }

        selected: list[_Node] = []
        model_representative = representatives.get(model_action)
        if model_representative is not None:
            selected.append(model_representative)
        others = sorted(
            (
                node
                for action, node in representatives.items()
                if action != model_action
            ),
            key=self._rank_key,
            reverse=True,
        )
        selected.extend(others[: self.config.beam_width - len(selected)])

        selected_ids = {id(node) for node in selected}
        remaining = self.config.beam_width - len(selected)
        if remaining > 0:
            extras = sorted(
                (node for node in nodes if id(node) not in selected_ids),
                key=self._rank_key,
                reverse=True,
            )
            selected.extend(extras[:remaining])
        return sorted(selected, key=self._rank_key, reverse=True)

    @staticmethod
    def _top_actions(node: _Node, limit: int) -> list[int]:
        assert node.probabilities is not None
        legal = np.flatnonzero(node.mask)
        ranked = sorted(
            (int(action) for action in legal),
            key=lambda action: float(node.probabilities[action]),
            reverse=True,
        )
        return ranked if limit == 0 else ranked[:limit]

    def _redeterminized_root(self, env):
        branch = env.fork_combat()
        thick = getattr(branch, "thick", None)
        battle_context = getattr(thick, "bc", None)
        if battle_context is None or not hasattr(
            battle_context, "nosl_redeterminize"
        ):
            raise RuntimeError(
                "NOSL combat search requires a BattleContext with "
                "nosl_redeterminize()"
            )
        branch.thick.bc = self._nosl_sampler.clone_and_redeterminize(
            battle_context,
            state_nonce=self._search_state_nonce,
            simulation_index=0,
        )
        invalidate = getattr(branch, "_invalidate_backend_cache", None)
        if invalidate is not None:
            invalidate()
        return branch

    def _apply(self, node: _Node, action: int) -> tuple[_Node, bool]:
        branch = node.env.fork_combat()
        advances = bool(branch.combat_action_advances(action))
        probability = float(node.probabilities[action])
        terminated, next_mask = branch.search_step(action, validate=False)
        obs = branch._obs()
        mask = (
            np.zeros_like(node.mask)
            if terminated
            else np.asarray(next_mask, dtype=bool)
        )
        first_action = node.first_action if node.first_action is not None else action
        child = _Node(
            env=branch,
            obs=obs,
            mask=mask,
            first_action=first_action,
            root_probability=(
                node.root_probability
                if node.first_action is not None
                else probability
            ),
            path=(*node.path, action),
            depth=node.depth + int(advances),
            log_probability=(
                node.log_probability + math.log(max(probability, 1.0e-30))
            ),
            terminal=bool(terminated),
        )
        self.expanded_nodes += 1
        return child, advances

    def _expand_one_decision(self, frontier: list[_Node]) -> list[_Node]:
        partial = list(frontier)
        completed: list[_Node] = []

        for argument_step in range(self.config.max_argument_steps):
            if not partial or self.expanded_nodes >= self.config.node_budget:
                break
            self._populate(partial, need_policy=True)
            if argument_step > 0 and len(partial) > self.config.argument_beam_width:
                partial = self._best(partial, self.config.argument_beam_width)

            specs: list[tuple[_Node, int]] = []
            for node in partial:
                if node.first_action is None:
                    limit = self.config.root_top_k
                elif argument_step == 0:
                    limit = self.config.policy_top_k
                else:
                    limit = self.config.argument_top_k
                specs.extend(
                    (node, action) for action in self._top_actions(node, limit)
                )

            remaining = self.config.node_budget - self.expanded_nodes
            specs = specs[:remaining]
            next_partial: list[_Node] = []
            for node, action in specs:
                child, advances = self._apply(node, action)
                if advances:
                    completed.append(child)
                else:
                    next_partial.append(child)
            partial = next_partial

        return completed

    def search(
        self,
        env,
        obs: dict[str, np.ndarray] | None = None,
        action_mask: np.ndarray | None = None,
    ) -> CombatSearchResult:
        started = time.perf_counter()
        self.expanded_nodes = 0
        self.evaluated_states = 0
        self._search_state_nonce = self._nosl_sampler.nonce(
            state_nonce=self._search_invocation,
            simulation_index=0,
        ) if self._nosl_sampler is not None else 0
        self._search_invocation += 1

        env = self._base_env(env)
        if self._phase(env) != "combat":
            raise ValueError("combat value search cannot run outside combat")
        if self._nosl_sampler is not None:
            # Match the stable training search contract: sample hidden future
            # once at the root, then preserve that sampled world while the beam
            # clones and advances it.  Re-sampling between thin argument steps
            # can invalidate action ids and erases known future state.
            env = self._redeterminized_root(env)
            obs = env._obs()
            action_mask = env.action_masks()
        else:
            obs = env._obs() if obs is None else obs
            action_mask = (
                env.action_masks() if action_mask is None else action_mask
            )
        root = _Node(
            env=env,
            obs=obs,
            mask=np.asarray(action_mask, dtype=bool),
        )
        self._populate([root], need_policy=True)
        model_action = self._top_actions(root, 1)[0]
        root_value = float(root.value)

        frontier = [root]
        finished: list[_Node] = []
        depth_reached = 0
        for _depth in range(self.config.depth):
            children = self._expand_one_decision(frontier)
            if not children:
                break
            self._score_nodes(children)
            depth_reached = max(depth_reached, max(node.depth for node in children))
            finished.extend(node for node in children if node.terminal)
            live = [node for node in children if not node.terminal]
            frontier = self._prune_frontier(live, model_action)
            if not frontier:
                break

        leaves = [*finished, *frontier]
        if not leaves:
            best = root
            best.first_action = model_action
            best.path = (model_action,)

        by_action: dict[int, _Node] = {}
        for leaf in leaves:
            if leaf.first_action is None:
                continue
            previous = by_action.get(leaf.first_action)
            if previous is None or self._rank_key(leaf) > self._rank_key(previous):
                by_action[leaf.first_action] = leaf
        if leaves:
            best = max(leaves, key=self._rank_key)
            baseline = by_action.get(model_action)
            if best.first_action != model_action and (
                baseline is None
                or float(best.value) < (
                    float(baseline.value) + self.config.min_value_gain
                )
            ):
                best = baseline or root
                best.first_action = model_action
                if best is root:
                    best.path = (model_action,)
        candidates = tuple(
            CombatSearchCandidate(
                action=int(node.first_action),
                value=float(node.value),
                depth=node.depth,
                terminal=node.terminal,
                root_probability=node.root_probability,
                path=node.path,
            )
            for node in sorted(by_action.values(), key=self._rank_key, reverse=True)
        )
        return CombatSearchResult(
            action=int(best.first_action),
            model_action=model_action,
            root_value=root_value,
            value=float(best.value),
            depth_reached=depth_reached,
            expanded_nodes=self.expanded_nodes,
            evaluated_states=self.evaluated_states,
            elapsed_seconds=time.perf_counter() - started,
            principal_variation=best.path,
            candidates=candidates,
        )
