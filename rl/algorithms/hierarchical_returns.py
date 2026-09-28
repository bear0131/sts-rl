"""Phase-local returns for hierarchical whole-run PPO.

Combat is a low-level option whose critic terminates at the end of each fight.
The overworld critic treats the complete fight as environment dynamics and
connects the decision before combat directly to the first state after combat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HierarchicalReturnStats:
    combat_samples: int
    overworld_samples: int
    valid_overworld_samples: int
    unresolved_overworld_samples: int
    combats_finished: int
    combat_end_hp_mean: float


def _as_array(name: str, value, shape: tuple[int, ...], dtype=None):
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    return array


def compute_hierarchical_returns(
    *,
    phases: np.ndarray,
    values: np.ndarray,
    raw_rewards: np.ndarray,
    dones: np.ndarray,
    next_phases: np.ndarray,
    next_hp: np.ndarray,
    interrupted: np.ndarray,
    last_values: np.ndarray,
    combat_phase: int,
    overworld_gamma: float,
    overworld_gae_lambda: float,
    combat_terminal_targets: np.ndarray | None = None,
    combat_gamma: float = 1.0,
    combat_gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, HierarchicalReturnStats]:
    """Build independent combat and overworld lambda returns.

    ``next_hp`` is current HP on a fixed global scale (the observation's first
    ``run_hp`` scalar), not HP/max-HP. An interrupted combat gets zero terminal
    reward so a timeout cannot profit by stalling at high HP.

    A rollout can end while an overworld action's combat option is still in
    progress. That one macro transition has no valid post-combat bootstrap, so
    its advantage/return are marked NaN and omitted by the PPO loss. At most one
    such transition per environment is omitted per rollout.
    """
    phases = np.asarray(phases)
    if phases.ndim != 2:
        raise ValueError(f"phases must be [steps, envs], got {phases.shape}")
    shape = phases.shape
    values = _as_array("values", values, shape, np.float32)
    raw_rewards = _as_array("raw_rewards", raw_rewards, shape, np.float32)
    dones = _as_array("dones", dones, shape, bool)
    next_phases = _as_array("next_phases", next_phases, shape)
    next_hp = _as_array("next_hp", next_hp, shape, np.float32)
    if combat_terminal_targets is not None:
        combat_terminal_targets = _as_array(
            "combat_terminal_targets", combat_terminal_targets, shape, np.float32
        )
    interrupted = _as_array("interrupted", interrupted, shape, bool)
    last_values = _as_array("last_values", last_values, (shape[1],), np.float32)

    if not 0.0 <= overworld_gamma <= 1.0:
        raise ValueError("overworld_gamma must be in [0, 1]")
    if not 0.0 <= overworld_gae_lambda <= 1.0:
        raise ValueError("overworld_gae_lambda must be in [0, 1]")
    if not 0.0 <= combat_gamma <= 1.0:
        raise ValueError("combat_gamma must be in [0, 1]")
    if not 0.0 <= combat_gae_lambda <= 1.0:
        raise ValueError("combat_gae_lambda must be in [0, 1]")

    steps, n_envs = shape
    combat = phases == combat_phase
    effective_rewards = np.zeros(shape, dtype=np.float32)
    advantages = np.full(shape, np.nan, dtype=np.float32)
    returns = np.full(shape, np.nan, dtype=np.float32)

    combat_finished_rewards: list[float] = []

    # Low-level trajectory: zero reward until the fight ends, then current HP.
    # The phase boundary is terminal for this return even though the Gym episode
    # continues into combat_reward/map/etc.
    for env_idx in range(n_envs):
        next_gae = 0.0
        for step in range(steps - 1, -1, -1):
            if not combat[step, env_idx]:
                next_gae = 0.0
                continue

            local_terminal = bool(
                dones[step, env_idx]
                or next_phases[step, env_idx] != combat_phase
            )
            reward = 0.0
            if local_terminal and not interrupted[step, env_idx]:
                reward_source = next_hp[step, env_idx]
                if combat_terminal_targets is not None:
                    override = combat_terminal_targets[step, env_idx]
                    if np.isfinite(override):
                        reward_source = override
                reward = max(0.0, float(reward_source))
                combat_finished_rewards.append(reward)
            effective_rewards[step, env_idx] = reward

            if local_terminal:
                next_non_terminal = 0.0
                next_value = 0.0
            elif step == steps - 1:
                next_non_terminal = 1.0
                next_value = float(last_values[env_idx])
            else:
                next_non_terminal = 1.0
                next_value = float(values[step + 1, env_idx])

            delta = (
                reward
                + combat_gamma * next_value * next_non_terminal
                - float(values[step, env_idx])
            )
            next_gae = (
                delta
                + combat_gamma
                * combat_gae_lambda
                * next_non_terminal
                * next_gae
            )
            advantages[step, env_idx] = next_gae
            returns[step, env_idx] = next_gae + values[step, env_idx]

    valid_overworld = 0
    unresolved_overworld = 0

    # High-level trajectory: accumulate the original reward while the combat
    # option runs, then bootstrap from the next overworld state. Discount is
    # applied once per strategic transition, not once per combat click.
    for env_idx in range(n_envs):
        upper_indices = np.flatnonzero(~combat[:, env_idx])
        transitions: dict[int, tuple[float, int | None, float, bool]] = {}

        for upper_step in upper_indices:
            reward_sum = 0.0
            cursor = int(upper_step)
            while True:
                reward_sum += float(raw_rewards[cursor, env_idx])

                if dones[cursor, env_idx]:
                    transitions[int(upper_step)] = (reward_sum, None, 0.0, True)
                    break

                if cursor == steps - 1:
                    if next_phases[cursor, env_idx] == combat_phase:
                        unresolved_overworld += 1
                    else:
                        transitions[int(upper_step)] = (
                            reward_sum,
                            None,
                            float(last_values[env_idx]),
                            False,
                        )
                    break

                next_step = cursor + 1
                if not combat[next_step, env_idx]:
                    transitions[int(upper_step)] = (
                        reward_sum,
                        next_step,
                        float(values[next_step, env_idx]),
                        False,
                    )
                    break
                cursor = next_step

        for upper_step in reversed(upper_indices.tolist()):
            transition = transitions.get(int(upper_step))
            if transition is None:
                continue
            reward_sum, next_step, next_value, terminal = transition
            effective_rewards[upper_step, env_idx] = reward_sum
            continuation = 0.0
            if next_step is not None and np.isfinite(advantages[next_step, env_idx]):
                continuation = float(advantages[next_step, env_idx])
            next_non_terminal = 0.0 if terminal else 1.0
            delta = (
                reward_sum
                + overworld_gamma * next_value * next_non_terminal
                - float(values[upper_step, env_idx])
            )
            advantage = (
                delta
                + overworld_gamma
                * overworld_gae_lambda
                * next_non_terminal
                * continuation
            )
            advantages[upper_step, env_idx] = advantage
            returns[upper_step, env_idx] = advantage + values[upper_step, env_idx]
            valid_overworld += 1

    stats = HierarchicalReturnStats(
        combat_samples=int(combat.sum()),
        overworld_samples=int((~combat).sum()),
        valid_overworld_samples=valid_overworld,
        unresolved_overworld_samples=unresolved_overworld,
        combats_finished=len(combat_finished_rewards),
        combat_end_hp_mean=(
            float(np.mean(combat_finished_rewards))
            if combat_finished_rewards
            else 0.0
        ),
    )
    return effective_rewards, returns, advantages, stats
