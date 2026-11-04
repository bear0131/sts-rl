"""MaskablePPO with optimizer-level gradient accumulation."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Thread
from types import MethodType

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

from env.core import vocab as V
from rl.policies.precision import bf16_autocast
from env.core import native_actions as C
from env.core.combat_value import VALUE_CONTRACT
from rl.search.native import (
    LeafInferenceSchedule,
    NATIVE_TIMING_KEYS,
    WORKER_TIMING_KEYS,
    complete_searches,
    complete_searches_native,
    complete_searches_streaming,
    observation_as_tensor as _leaf_observation_as_tensor,
)
from rl.algorithms.ppo import PPOLossComponents, ppo_loss_components
from rl.algorithms.updates import ppo_forward, optimizer_step, reasoning_microbatch_backward
from rl.search.native import apply_strict_training_settings
from rl.algorithms.hierarchical_returns import compute_hierarchical_returns
from rl.algorithms.upper_ppo import enabled as joint_upper, combat_update, capture_step, update_upper
from rl.training import reward_monitor, campfire_monitor
from rl.algorithms.nosl_replay import NOSLISAZReplayBuffer
from rl.algorithms import outer_return as outer


NOSL_IS_AZ_SEARCH_MODES = {
    "puct": 0,
    "gumbel": 1,
    "full_gumbel": 2,
}


def _append_nosl_value_only_sample(
    samples: list[dict], observation, action_dim: int, *, sample_kind: str
) -> bool:
    """Append an observed trajectory state that has value but no action target."""
    if observation is None:
        return False
    samples.append({
        "observation": {
            key: np.asarray(value).copy()
            for key, value in observation.items()
        },
        "action_mask": np.zeros(action_dim, dtype=bool),
        "policy_target": np.zeros(action_dim, dtype=np.float32),
        "policy_weight": 0.0,
        "value_target": np.nan,
        "search_value": np.nan,
        "sample_kind": sample_kind,
    })
    return True




def _fill_nosl_combat_value_targets(
    samples: list[dict], final_hp: float, *, clip_hp: bool = True
) -> int:
    """Assign one normalized terminal-HP target to every root in a fight."""
    target = float(np.clip(final_hp, 0.0, 1.0) if clip_hp else final_hp)
    if not np.isfinite(target) or target < 0:
        raise ValueError("final combat HP must be finite and nonnegative")
    for sample in samples:
        sample["value_target"] = target
    return len(samples)


def _normalized_player_hp(observation) -> float:
    """Read the public current-player-HP feature used by combat roots."""
    if not isinstance(observation, dict):
        return float("nan")
    # run_hp is present in all phases; player_scalars is the combat-specific
    # fallback for older checkpoints/observation adapters.
    for key in ("run_hp", "player_scalars"):
        value = observation.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size:
            return float(array[0])
    return float("nan")








class AccumulatingMaskablePPO(MaskablePPO):
    """MaskablePPO whose optimizer can consume several micro-batches per step.

    ``batch_size`` remains the number of rollout samples materialized for one
    forward/backward pass. ``gradient_accumulation_steps`` controls how many of
    those gradients are averaged before clipping and stepping the optimizer.
    """

    def __init__(
        self,
        *args,
        gradient_accumulation_steps: int = 1,
        performance_profile: bool = False,
        amp_mode: str = "off",
        hierarchical_combat_hp: bool = False,
        nosl_is_az_boss_transition_target: bool = False,
        combat_hp_gamma: float = 1.0,
        combat_hp_gae_lambda: float = 0.95,
        reasoning_coef: float = 0.0,
        reasoning_batch_size: int = 16,
        reasoning_updates: int = 1,
        reasoning_gradient_accumulation_steps: int = 1,
        nosl_is_az: bool = False,
        nosl_is_az_inner_only: bool = False,
        nosl_is_az_upper_strategy: str = "model",
        nosl_is_az_combat_strategy: str = "model",
        rulebase_planner_worlds: int = 2,
        rulebase_planner_nodes: int = 8192,
        rulebase_planner_beam: int = 8,
        rulebase_planner_depth: int = 16,
        rulebase_planner_turns: int = 1,
        rulebase_planner_leaf_profile: str = "legacy-v8",
        rulebase_planner_nonce: int = 20260918,
        nosl_is_az_cpp: bool = False,
        nosl_is_az_simulations: int = 8,
        nosl_is_az_depth: int = 3,
        nosl_is_az_end_turn_as_leaf: bool = False,
        nosl_is_az_fixed_worldline: bool = False,
        nosl_is_az_rollout_steps: int = 0,
        nosl_is_az_c_puct: float = 1.5,
        nosl_is_az_temperature: float = 1.0,
        nosl_is_az_prior_temperature: float = 1.0,
        nosl_is_az_search_mode: str = "puct",
        nosl_is_az_paper_full_gumbel: bool = False,
        nosl_is_az_gumbel_candidates: int = 0,
        nosl_is_az_gumbel_scale: float = 1.0,
        nosl_is_az_merge_equivalent_actions: bool = True,
        nosl_is_az_gumbel_value_scale: float = 0.1,
        nosl_is_az_gumbel_max_visit_init: float = 50.0,
        nosl_is_az_gumbel_rescale_q: bool = True,
        nosl_is_az_gumbel_use_mixed_value: bool = True,
        nosl_is_az_gumbel_interior_uniformity: float = 0.0,
        nosl_is_az_gumbel_interior_coverage_fraction: float = 0.0,
        nosl_is_az_gumbel_interior_value_mode: int = 2,
        nosl_is_az_gumbel_interior_max_mix_visits: float = 64.0,
        nosl_is_az_dirichlet_alpha: float = 0.0,
        nosl_is_az_dirichlet_epsilon: float = 0.0,
        nosl_is_az_master_nonce: int = 0x4E4F534C4953415A,
        nosl_is_az_batch_size: int = 64,
        nosl_is_az_micro_batch_size: int = 2048,
        nosl_is_az_replay_enabled: bool = True,
        nosl_is_az_replay_capacity: int = 131072,
        nosl_is_az_replay_min_size: int = 1,
        nosl_is_az_replay_max_old_fraction: float = 1.0,
        nosl_is_az_leaf_batch_size: int = 64,
        nosl_is_az_inference_batch_size: int = 32,
        nosl_is_az_inference_max_wait_ms: float = 0.5,
        nosl_is_az_inference_schedule: str = "fixed_wait",
        nosl_is_az_updates: int = 1,
        nosl_is_az_update_schedule: str | dict | None = None,
        nosl_is_az_policy_coef: float = 1.0,
        nosl_is_az_value_coef: float = 0.5,
        **kwargs,
    ) -> None:
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if amp_mode not in ("off", "bf16"):
            raise ValueError("amp_mode must be 'off' or 'bf16'")
        if not 0.0 <= combat_hp_gamma <= 1.0:
            raise ValueError("combat_hp_gamma must be in [0, 1]")
        if not 0.0 <= combat_hp_gae_lambda <= 1.0:
            raise ValueError("combat_hp_gae_lambda must be in [0, 1]")
        if reasoning_coef < 0.0:
            raise ValueError("reasoning_coef must be non-negative")
        if reasoning_batch_size < 1:
            raise ValueError("reasoning_batch_size must be at least 1")
        if reasoning_updates < 0:
            raise ValueError("reasoning_updates must be non-negative")
        if reasoning_gradient_accumulation_steps < 1:
            raise ValueError(
                "reasoning_gradient_accumulation_steps must be at least 1"
            )
        if nosl_is_az_simulations < 1 or nosl_is_az_depth < 0:
            raise ValueError(
                "NOSL-IS-AZ simulations must be positive and depth must be "
                "non-negative (0 means unlimited node expansion)"
            )
        if nosl_is_az_rollout_steps < 0:
            raise ValueError("NOSL-IS-AZ rollout steps must be non-negative")
        if nosl_is_az_c_puct < 0.0 or nosl_is_az_temperature < 0.0:
            raise ValueError("NOSL-IS-AZ c_puct/temperature must be non-negative")
        if not np.isfinite(nosl_is_az_prior_temperature) or nosl_is_az_prior_temperature <= 0.0:
            raise ValueError("NOSL-IS-AZ prior temperature must be positive and finite")
        if nosl_is_az_search_mode not in NOSL_IS_AZ_SEARCH_MODES:
            raise ValueError(
                "NOSL-IS-AZ search mode must be puct, gumbel, or full_gumbel"
            )
        if nosl_is_az_search_mode != "puct" and not nosl_is_az_cpp:
            raise ValueError("Gumbel NOSL-IS-AZ search requires the C++ backend")
        if nosl_is_az_gumbel_candidates < 0:
            raise ValueError("NOSL-IS-AZ Gumbel candidates must be non-negative")
        for name, value in (
            ("Gumbel scale", nosl_is_az_gumbel_scale),
            ("Gumbel value scale", nosl_is_az_gumbel_value_scale),
            ("Gumbel max visit init", nosl_is_az_gumbel_max_visit_init),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"NOSL-IS-AZ {name} must be non-negative and finite")
        if nosl_is_az_dirichlet_alpha < 0.0:
            raise ValueError("NOSL-IS-AZ Dirichlet alpha must be non-negative")
        if not 0.0 <= nosl_is_az_dirichlet_epsilon <= 1.0:
            raise ValueError("NOSL-IS-AZ Dirichlet epsilon must be in [0, 1]")
        if nosl_is_az_dirichlet_epsilon > 0.0 and nosl_is_az_dirichlet_alpha <= 0.0:
            raise ValueError(
                "NOSL-IS-AZ Dirichlet alpha must be positive when epsilon > 0"
            )
        if nosl_is_az_batch_size < 1 or nosl_is_az_updates < 1:
            raise ValueError("NOSL-IS-AZ batch size/updates must be positive")
        if nosl_is_az_micro_batch_size < 1:
            raise ValueError("NOSL-IS-AZ micro batch size must be positive")
        if nosl_is_az_replay_capacity < 1:
            raise ValueError("NOSL-IS-AZ replay capacity must be positive")
        if not 1 <= nosl_is_az_replay_min_size <= nosl_is_az_replay_capacity:
            raise ValueError(
                "NOSL-IS-AZ replay min size must be in [1, replay capacity]"
            )
        if (
            not np.isfinite(nosl_is_az_replay_max_old_fraction)
            or not 0.0 <= nosl_is_az_replay_max_old_fraction <= 1.0
        ):
            raise ValueError(
                "NOSL-IS-AZ replay max old fraction must be finite and in [0, 1]"
            )
        if nosl_is_az_leaf_batch_size < 1:
            raise ValueError("NOSL-IS-AZ leaf batch size must be positive")
        if nosl_is_az_inference_batch_size < 1:
            raise ValueError("NOSL-IS-AZ inference batch size must be positive")
        if (
            not np.isfinite(nosl_is_az_inference_max_wait_ms)
            or nosl_is_az_inference_max_wait_ms < 0.0
        ):
            raise ValueError(
                "NOSL-IS-AZ inference maximum wait must be finite and non-negative"
            )
        if nosl_is_az_inference_schedule not in ("fixed_wait", "adaptive"):
            raise ValueError(
                "NOSL-IS-AZ inference schedule must be fixed_wait or adaptive"
            )
        if nosl_is_az_policy_coef < 0.0 or nosl_is_az_value_coef < 0.0:
            raise ValueError("NOSL-IS-AZ loss coefficients must be non-negative")
        if nosl_is_az_inner_only and not nosl_is_az:
            raise ValueError("NOSL-IS-AZ inner-only mode requires nosl_is_az")
        if nosl_is_az_upper_strategy not in ("model", "rules"):
            raise ValueError("NOSL-IS-AZ upper strategy must be model or rules")
        if nosl_is_az_upper_strategy == "rules" and not (
            nosl_is_az and nosl_is_az_inner_only
        ):
            raise ValueError("NOSL upper rules require inner-only NOSL training")
        if nosl_is_az_combat_strategy not in ("model", "turn-planner", "small-playout"):
            raise ValueError("NOSL-IS-AZ combat strategy must be model or turn-planner")
        if nosl_is_az_combat_strategy in ("turn-planner", "small-playout") and not (
            nosl_is_az and nosl_is_az_inner_only
        ):
            raise ValueError("turn-planner combat requires inner-only NOSL collection")
        from env.search.turn_planner import TurnPlannerConfig
        TurnPlannerConfig(
            worlds=int(rulebase_planner_worlds),
            node_budget=int(rulebase_planner_nodes),
            beam_width=int(rulebase_planner_beam),
            max_depth=int(rulebase_planner_depth),
            lookahead_turns=int(rulebase_planner_turns),
            leaf_profile=str(rulebase_planner_leaf_profile),
            master_nonce=int(rulebase_planner_nonce),
        )
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.performance_profile = bool(performance_profile)
        self.amp_mode = amp_mode
        # Combat AZ targets are fight-local terminal-HP values.  Keep the
        # overworld/combat return split enabled automatically so combat PPO can
        # be removed without changing the strategic return calculation.
        self.nosl_is_az = bool(nosl_is_az)
        # Optional boss terminal objective.  This is an additive, serialized
        # setting; legacy checkpoints and fresh runs keep the old HP target.
        self.nosl_is_az_boss_transition_target = bool(
            nosl_is_az_boss_transition_target
        )
        self.nosl_value_contract = (
            "legacy-boss-transition" if self.nosl_is_az_boss_transition_target
            else VALUE_CONTRACT
        )
        self.nosl_is_az_inner_only = bool(nosl_is_az_inner_only)
        self.nosl_is_az_upper_strategy = str(nosl_is_az_upper_strategy)
        self.nosl_is_az_combat_strategy = str(nosl_is_az_combat_strategy)
        self.rulebase_planner_worlds = int(rulebase_planner_worlds)
        self.rulebase_planner_nodes = int(rulebase_planner_nodes)
        self.rulebase_planner_beam = int(rulebase_planner_beam)
        self.rulebase_planner_depth = int(rulebase_planner_depth)
        self.rulebase_planner_turns = int(rulebase_planner_turns)
        self.rulebase_planner_leaf_profile = str(rulebase_planner_leaf_profile)
        self.rulebase_planner_nonce = int(rulebase_planner_nonce)
        self.nosl_is_az_cpp = bool(nosl_is_az_cpp)
        self._hierarchical_hp_requested = bool(hierarchical_combat_hp)
        self.hierarchical_combat_hp = bool(
            hierarchical_combat_hp or self.nosl_is_az
        )
        self.combat_hp_gamma = float(combat_hp_gamma)
        self.combat_hp_gae_lambda = float(combat_hp_gae_lambda)
        self.reasoning_coef = float(reasoning_coef)
        self.reasoning_batch_size = int(reasoning_batch_size)
        self.reasoning_updates = int(reasoning_updates)
        self.reasoning_gradient_accumulation_steps = int(
            reasoning_gradient_accumulation_steps
        )
        self.nosl_is_az_simulations = int(nosl_is_az_simulations)
        self.nosl_is_az_depth = int(nosl_is_az_depth)
        self.nosl_is_az_end_turn_as_leaf = bool(nosl_is_az_end_turn_as_leaf)
        self.nosl_is_az_fixed_worldline = bool(nosl_is_az_fixed_worldline)
        if self.nosl_is_az_fixed_worldline and not nosl_is_az_cpp:
            raise ValueError('fixed-worldline training requires the C++ search backend')
        self.nosl_is_az_rollout_steps = int(nosl_is_az_rollout_steps)
        self.nosl_is_az_c_puct = float(nosl_is_az_c_puct)
        self.nosl_is_az_temperature = float(nosl_is_az_temperature)
        self.nosl_is_az_prior_temperature = float(nosl_is_az_prior_temperature)
        self.nosl_is_az_search_mode = str(nosl_is_az_search_mode)
        self.nosl_is_az_paper_full_gumbel = bool(
            nosl_is_az_paper_full_gumbel
        )
        self.nosl_is_az_root_search_mode = NOSL_IS_AZ_SEARCH_MODES[
            self.nosl_is_az_search_mode
        ]
        self.nosl_is_az_gumbel_candidates = int(nosl_is_az_gumbel_candidates)
        self.nosl_is_az_gumbel_scale = float(nosl_is_az_gumbel_scale)
        self.nosl_is_az_merge_equivalent_actions = bool(nosl_is_az_merge_equivalent_actions)
        self.nosl_is_az_gumbel_value_scale = float(nosl_is_az_gumbel_value_scale)
        self.nosl_is_az_gumbel_max_visit_init = float(
            nosl_is_az_gumbel_max_visit_init
        )
        self.nosl_is_az_gumbel_rescale_q = bool(nosl_is_az_gumbel_rescale_q)
        self.nosl_is_az_gumbel_use_mixed_value = bool(
            nosl_is_az_gumbel_use_mixed_value
        )
        self.nosl_is_az_gumbel_interior_uniformity = float(
            nosl_is_az_gumbel_interior_uniformity
        )
        self.nosl_is_az_gumbel_interior_coverage_fraction = float(
            nosl_is_az_gumbel_interior_coverage_fraction
        )
        self.nosl_is_az_gumbel_interior_value_mode = int(
            nosl_is_az_gumbel_interior_value_mode
        )
        self.nosl_is_az_gumbel_interior_max_mix_visits = float(
            nosl_is_az_gumbel_interior_max_mix_visits
        )
        if not 0.0 <= self.nosl_is_az_gumbel_interior_coverage_fraction <= 1.0:
            raise ValueError(
                "NOSL Full-Gumbel interior coverage fraction must be in [0, 1]"
            )
        if self.nosl_is_az_gumbel_interior_value_mode not in (0, 1, 2):
            raise ValueError(
                "NOSL Full-Gumbel interior value mode must be 0, 1, or 2"
            )
        if (
            not np.isfinite(self.nosl_is_az_gumbel_interior_max_mix_visits)
            or self.nosl_is_az_gumbel_interior_max_mix_visits <= 0.0
        ):
            raise ValueError(
                "NOSL Full-Gumbel interior max-mix visits must be positive and finite"
            )
        self.nosl_is_az_dirichlet_alpha = float(nosl_is_az_dirichlet_alpha)
        self.nosl_is_az_dirichlet_epsilon = float(nosl_is_az_dirichlet_epsilon)
        self.nosl_is_az_master_nonce = int(nosl_is_az_master_nonce)
        self.nosl_is_az_batch_size = int(nosl_is_az_batch_size)
        self.nosl_is_az_micro_batch_size = int(nosl_is_az_micro_batch_size)
        self.nosl_is_az_replay_enabled = bool(nosl_is_az_replay_enabled)
        self.nosl_is_az_replay_capacity = int(nosl_is_az_replay_capacity)
        self.nosl_is_az_replay_min_size = int(nosl_is_az_replay_min_size)
        self.nosl_is_az_replay_max_old_fraction = float(
            nosl_is_az_replay_max_old_fraction
        )
        self.nosl_is_az_leaf_batch_size = int(nosl_is_az_leaf_batch_size)
        self.nosl_is_az_inference_batch_size = int(
            nosl_is_az_inference_batch_size
        )
        self.nosl_is_az_inference_max_wait_ms = float(
            nosl_is_az_inference_max_wait_ms
        )
        self.nosl_is_az_inference_schedule = str(
            nosl_is_az_inference_schedule
        )
        self._nosl_leaf_batch_controller = None
        if self.nosl_is_az_paper_full_gumbel:
            apply_strict_training_settings(self)
        self.nosl_is_az_updates = int(nosl_is_az_updates)
        from rl.algorithms.update_schedule import CombatUpdateSchedule
        self.nosl_is_az_update_schedule = CombatUpdateSchedule.from_config(nosl_is_az_update_schedule).config()
        self.nosl_is_az_policy_coef = float(nosl_is_az_policy_coef)
        self.nosl_is_az_value_coef = float(nosl_is_az_value_coef)
        self._nosl_is_az_samples = []
        # When replay is disabled, this transient list is the complete
        # effective batch for the current update.  It is intentionally not
        # serialized into checkpoints.
        self._nosl_is_az_fresh_samples = []
        self._nosl_is_az_pending_fights = []
        self._nosl_is_az_replay = None
        self._nosl_is_az_replay_prepare_stats = {}
        self._reasoning_rollout_samples = []
        self._reasoning_compile_thread: Thread | None = None
        self._reasoning_compile_error: BaseException | None = None
        self._performance_rollout = None
        self._hierarchical_rollout_stats = None
        super().__init__(*args, **kwargs)
        if self.amp_mode != "off" and self.device.type != "cuda":
            raise ValueError("bf16 AMP requires a CUDA device")

    def _setup_model(self) -> None:
        super()._setup_model()
        from env.core.encode import OBS_VERSION
        self.observation_version = OBS_VERSION
        from rl.algorithms.action_groups import CONTRACT
        self.nosl_action_group_contract = (CONTRACT if self.nosl_is_az and
            self.nosl_is_az_merge_equivalent_actions else 'individual-actions')
        self.nosl_supervision_contract = getattr(self.policy.core,'training_objective','scalar-hp-v1') if hasattr(self.policy,'core') else 'scalar-hp-v1'
        from rl.algorithms.combat_distribution import OUTCOME_CONTRACTS
        # Both heads share observed outcome labels and the combat replay pipeline.
        self._combat_distribution = self.nosl_supervision_contract in OUTCOME_CONTRACTS
        self._outer_return = self.nosl_supervision_contract in outer.SCALAR_OBJECTIVES
        self._frozen_upper = bool(getattr(getattr(self.policy, 'core', None), 'frozen_upper', False))
        if self._outer_return:
            if self._hierarchical_hp_requested:
                raise ValueError('outer return cannot enable hierarchical HP targets')
            self.hierarchical_combat_hp = False
        if self._frozen_upper:
            outer.setup(self)
        from rl.algorithms.update_schedule import CombatUpdateSchedule
        CombatUpdateSchedule.from_config(self.nosl_is_az_update_schedule).validate_model(self)
        if self._combat_distribution and not (self.nosl_is_az and self.nosl_is_az_inner_only
                and self.nosl_is_az_cpp and self.nosl_is_az_upper_strategy == ('model' if joint_upper(self) or self._frozen_upper else 'rules')):
            raise ValueError('entity-combat training requires C++ inner-only NOSL and rules upper strategy')
        self.combat_action_contract = C.policy_contract(self.observation_space)
        if self.combat_action_contract in C.NEURAL_CONTRACTS:
            if getattr(self, "nosl_is_az_boss_transition_target", False):
                raise ValueError("joint-v2 requires combat-end-hp-v1")
            if self.nosl_is_az and not self.nosl_is_az_cpp:
                raise ValueError("joint-v2 NOSL requires the C++ search backend")

    def dump_logs(self, iteration=0):
        # SB3's narrow console columns otherwise collapse distinct long NOSL
        # metric names into the same key and raise before the learner update.
        from stable_baselines3.common.logger import HumanOutputFormat
        width = max((len(key)+3 for key in self.logger.name_to_value), default=36)
        for output in self.logger.output_formats:
            if isinstance(output, HumanOutputFormat):
                output.max_length = max(output.max_length, width)
        return super().dump_logs(iteration)

    def _excluded_save_params(self) -> list[str]:
        """Exclude transient reasoning transcripts from SB3 checkpoints.

        The captured rollout traces are regenerated after every rollout.  They
        can contain hundreds of thousands of long token sequences and are not
        needed to resume the policy, so serializing them makes checkpoints
        needlessly multi-gigabyte.
        """
        return [
            *super()._excluded_save_params(),
            "_native_leaf_inference",
            "_rollout_inference",
            "_nosl_leaf_batch_controller",
            "_reasoning_rollout_samples",
            "_nosl_is_az_samples",
            "_nosl_is_az_fresh_samples",
            "_nosl_is_az_pending_fights",
            "_outer_pending_rewards", "_outer_collection",
            "_upper_pending", "_upper_completed", "_upper_episode_ids",
            "_upper_auxiliary",
            "_upper_reward_counts",
            "_small_playout_samples",
            "_nosl_is_az_replay",
            "_nosl_is_az_replay_prepare_stats",
            "_nosl_is_az_rollout_hp_order",
            "_nosl_is_az_rollout_upper_counts",
            "_nosl_is_az_metric_sums",
            "_nosl_is_az_metric_sumsq",
            "_nosl_is_az_metric_counts",
            "_nosl_is_az_metric_count",
            "_nosl_is_az_policy_loss_sum",
            "_nosl_is_az_policy_weight_sum",
            "_nosl_is_az_value_loss_sum",
        ]

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        if joint_upper(self):
            state_dicts = [*state_dicts, 'policy.upper_optimizer']
        if hasattr(self.policy, 'small_playout_optimizer'):
            state_dicts = [*state_dicts, 'policy.small_playout_optimizer']
        return state_dicts, variables

    def _ensure_nosl_is_az_replay(self) -> NOSLISAZReplayBuffer:
        """Create runtime replay state, including when loading an old checkpoint."""
        # A disabled replay path must never allocate the large fixed-capacity
        # arrays.  Keep this guard explicit so an accidental call from a new
        # code path fails loudly instead of silently changing the experiment.
        if not bool(getattr(self, "nosl_is_az_replay_enabled", True)):
            raise RuntimeError("NOSL-IS-AZ replay is disabled")
        capacity = int(getattr(self, "nosl_is_az_replay_capacity", 131072))
        self.nosl_is_az_replay_capacity = capacity
        self.nosl_is_az_replay_min_size = int(
            getattr(self, "nosl_is_az_replay_min_size", 1)
        )
        self.nosl_is_az_replay_max_old_fraction = float(
            getattr(self, "nosl_is_az_replay_max_old_fraction", 1.0)
        )
        self.nosl_is_az_micro_batch_size = int(
            getattr(self, "nosl_is_az_micro_batch_size", 2048)
        )
        replay = getattr(self, "_nosl_is_az_replay", None)
        value_contract = (
            "legacy-boss-transition"
            if getattr(self, "nosl_is_az_boss_transition_target", False)
            else VALUE_CONTRACT
        )
        if getattr(self, '_outer_return', False):
            value_contract = self.nosl_supervision_contract
        if replay is not None:
            if replay.supervision_contract != getattr(self,'nosl_supervision_contract','scalar-hp-v1'):
                raise ValueError("NOSL replay supervision changed; collect new replay")
            if getattr(replay, "value_contract", None) != value_contract:
                raise ValueError("NOSL replay value contract changed; collect new replay")
            if replay.capacity != capacity:
                raise RuntimeError(
                    "cannot resize a populated NOSL replay buffer at runtime"
                )
            return replay

        core = getattr(self.policy, "core", None)
        observation_keys = getattr(
            core, "combat_observation_keys", self.observation_space.spaces.keys()
        )
        action_dim = int(getattr(self.action_space, "n", 0))
        replay = NOSLISAZReplayBuffer(
            capacity=capacity,
            observation_space=self.observation_space,
            observation_keys=observation_keys,
            action_dim=action_dim,
            value_contract=value_contract,
            supervision_contract=getattr(self,'nosl_supervision_contract','scalar-hp-v1'),
            objective_identity=getattr(self, 'outer_return_contract', None),
        )
        self._nosl_is_az_replay = replay
        return replay

    def _nosl_is_az_observation_keys(self) -> tuple[str, ...]:
        """Return the combat observation contract without creating replay."""
        core = getattr(self.policy, "core", None)
        keys = getattr(core, "combat_observation_keys", None)
        if keys is None:
            keys = self.observation_space.spaces.keys()
        return tuple(
            key for key in keys if key in self.observation_space.spaces
        )

    def _nosl_samples_to_batch(self, samples: list[dict]) -> dict[str, object]:
        """Materialize fresh labelled roots in the replay batch contract."""
        if not samples:
            raise ValueError("cannot materialize an empty NOSL sample batch")
        keys = self._nosl_is_az_observation_keys()
        action_dim = int(getattr(self.action_space, "n", 0))
        observations = {
            key: np.stack([sample["observation"][key] for sample in samples])
            for key in keys
        }
        action_masks = np.stack(
            [np.asarray(sample["action_mask"], dtype=bool) for sample in samples]
        )
        policy_targets = np.stack(
            [np.asarray(sample["policy_target"], dtype=np.float32) for sample in samples]
        )
        policy_weights = np.asarray(
            [sample.get("policy_weight", 1.0) for sample in samples],
            dtype=np.float32,
        )
        value_targets = np.asarray(
            [sample["value_target"] for sample in samples], dtype=np.float32
        )
        expected = (len(samples), action_dim)
        if action_masks.shape != expected or policy_targets.shape != expected:
            raise ValueError(
                "invalid fresh NOSL action/policy shape: "
                f"{action_masks.shape}/{policy_targets.shape}; expected {expected}"
            )
        if policy_weights.shape != (len(samples),) or not np.all(
            np.isfinite(policy_weights) & (policy_weights >= 0.0)
        ):
            raise ValueError(
                "fresh NOSL policy weights must be finite and non-negative"
            )
        from rl.algorithms.action_groups import sample_groups
        extra = {"action_groups": sample_groups(samples, action_dim)}
        if getattr(self,'_combat_distribution',False):
            from rl.algorithms.combat_distribution import validate_targets
            extra['combat_targets'] = np.stack([sample['combat_targets'] for sample in samples]).astype(np.float32)
            validate_targets(extra['combat_targets'],value_targets, action_dim=action_dim)
        return {
            **extra,
            "observations": observations,
            "action_masks": action_masks,
            "policy_targets": policy_targets,
            "policy_weights": policy_weights,
            "value_targets": value_targets,
            # The no-replay path has no persistent sample ids.  Keeping a
            # local ordinal makes diagnostics and compatibility callers easy.
            "sample_ids": np.arange(len(samples), dtype=np.uint64),
        }

    def autocast_context(self):
        if self.amp_mode == "bf16":
            return bf16_autocast()
        return nullcontext()

    def reset_nosl_is_az_shared_embeddings(self, seed: int) -> int:
        """Reset shared semantic embedding tables with a reproducible seed."""
        if not self.nosl_is_az or not self.nosl_is_az_inner_only:
            raise ValueError("shared embedding reset requires NOSL inner-only mode")
        core = getattr(self.policy, "core", None)
        shared = getattr(core, "shared", None)
        if getattr(core, 'combat_only', False) or joint_upper(self):
            # EntityCombat owns its embeddings inside the freshly initialized
            # combat network; there is no separately pretrained shared table.
            # Joint upper likewise owns a fresh independent tokenizer.
            return 0
        if shared is None:
            raise ValueError("NOSL inner-only mode requires shared semantic tokenizers")
        embeddings = [
            module for module in shared.modules()
            if isinstance(module, th.nn.Embedding)
        ]
        if not embeddings:
            raise ValueError("shared semantic tokenizer has no embedding tables")
        cuda_devices = []
        parameter = next(shared.parameters(), None)
        if parameter is not None and parameter.is_cuda:
            device_index = parameter.device.index
            if device_index is not None:
                cuda_devices = [device_index]
        with th.random.fork_rng(devices=cuda_devices):
            th.manual_seed(int(seed))
            if cuda_devices:
                th.cuda.manual_seed_all(int(seed))
            for module in embeddings:
                module.reset_parameters()
        # A frozen card-rule encoder may have a materialized lookup cache.  It
        # must not survive a reset of the underlying embedding parameters.
        if hasattr(shared, "card_vector_cache"):
            shared.card_vector_cache = None
            shared.card_vector_cache_enabled = False
        return len(embeddings)

    def configure_nosl_is_az_inner_only(
        self, *, train_shared_embeddings: bool = False
    ) -> tuple[int, int]:
        """Freeze everything except combat and optionally shared embeddings."""
        if not self.nosl_is_az or not self.nosl_is_az_inner_only:
            raise ValueError("NOSL-IS-AZ inner-only mode is not enabled")
        core = getattr(self.policy, "core", None)
        combat = getattr(core, "combat", None)
        if getattr(core, 'frozen_upper', False):
            if train_shared_embeddings:
                raise ValueError('frozen upper does not permit shared-embedding training')
            core.teacher.assert_integrity()
            return (sum(p.numel() for p in core.teacher.parameters()),
                    sum(p.numel() for p in combat.parameters() if p.requires_grad))
        if getattr(core,'combat_only',False) or joint_upper(self):
            for parameter in self.policy.parameters():
                parameter.requires_grad_(True)
            if not core.config.auxiliary_loss_coef:
                core.disable_transition_auxiliary()
            frozen = sum(p.numel() for p in self.policy.parameters() if not p.requires_grad)
            trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
            return frozen, trainable
        if combat is None or getattr(core, "overworld", None) is None:
            raise ValueError(
                "NOSL-IS-AZ inner-only mode requires a split specialist policy"
            )
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        for parameter in combat.parameters():
            parameter.requires_grad_(True)
        if train_shared_embeddings:
            shared = getattr(core, "shared", None)
            if shared is None:
                raise ValueError(
                    "shared embedding training requires a split specialist policy"
                )
            for module in shared.modules():
                if isinstance(module, th.nn.Embedding):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        frozen = sum(
            parameter.numel()
            for parameter in self.policy.parameters()
            if not parameter.requires_grad
        )
        trainable = sum(
            parameter.numel()
            for parameter in self.policy.parameters()
            if parameter.requires_grad
        )
        return frozen, trainable

    def configure_rulebase_lower(self) -> tuple[int, int]:
        """Freeze the learned combat core for upper-only rulebase collection."""
        if getattr(self, "nosl_is_az_combat_strategy", "model") not in ("turn-planner", "small-playout"):
            raise ValueError("rulebase lower is not enabled")
        if not joint_upper(self):
            raise ValueError("rulebase lower requires the joint strategic policy")
        for parameter in self.policy.core.combat.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        frozen = sum(
            parameter.numel() for parameter in self.policy.core.combat.parameters()
            if not parameter.requires_grad
        )
        trainable = sum(
            parameter.numel() for parameter in self.policy.core.upper_parameters()
            if parameter.requires_grad
        )
        return frozen, trainable

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            th.cuda.synchronize(self.device)

    def configure_hierarchical_combat_hp(
        self,
        *,
        enabled: bool,
        combat_gamma: float = 1.0,
        combat_gae_lambda: float = 0.95,
        reset_combat_value: bool = False,
        reset_seed: int = 0,
    ) -> int:
        if getattr(self, '_outer_return', False):
            if enabled or reset_combat_value:
                raise ValueError('outer return cannot enable or reset an HP value head')
            self.hierarchical_combat_hp = False
            return 0
        if not 0.0 <= combat_gamma <= 1.0:
            raise ValueError("combat_gamma must be in [0, 1]")
        if not 0.0 <= combat_gae_lambda <= 1.0:
            raise ValueError("combat_gae_lambda must be in [0, 1]")
        if self.nosl_is_az and not enabled:
            raise ValueError(
                "NOSL-IS-AZ requires hierarchical combat returns; disabling "
                "the combat phase would re-enable the PPO objective"
            )
        self.hierarchical_combat_hp = bool(enabled or self.nosl_is_az)
        self.combat_hp_gamma = float(combat_gamma)
        self.combat_hp_gae_lambda = float(combat_gae_lambda)
        if not reset_combat_value:
            return 0
        return self.reset_combat_value_head(seed=reset_seed)

    def reset_combat_value_head(self, *, seed: int = 0) -> int:
        """Reset only the critic whose target changes to terminal combat HP."""
        core = getattr(self.policy, "core", None)
        combat = getattr(core, "combat", None)
        value_mlp = getattr(combat, "value_mlp", None)
        value_head = getattr(combat, "value_head", None)
        if value_mlp is None or value_head is None:
            raise ValueError(
                "hierarchical combat-HP mode requires a split specialist policy"
            )

        modules = (value_mlp, value_head)
        parameters = {
            parameter
            for module in modules
            for parameter in module.parameters()
        }
        cuda_devices = sorted({
            parameter.device.index
            for parameter in parameters
            if parameter.is_cuda and parameter.device.index is not None
        })
        with th.random.fork_rng(devices=cuda_devices):
            th.manual_seed(seed)
            if cuda_devices:
                th.cuda.manual_seed_all(seed)
            for module in modules:
                for child in module.modules():
                    if isinstance(child, th.nn.Linear):
                        child.reset_parameters()

        optimizer = getattr(self.policy, "optimizer", None)
        if optimizer is not None:
            for parameter in parameters:
                optimizer.state.pop(parameter, None)
        return len(parameters)

    @staticmethod
    def _next_phase_hp(new_obs, dones, infos):
        phases = np.asarray(new_obs["phase"]).reshape(len(dones), -1)[:, 0].copy()
        hp = np.asarray(new_obs["run_hp"], dtype=np.float32).reshape(
            len(dones), -1
        )[:, 0].copy()
        interrupted = np.zeros(len(dones), dtype=bool)
        for idx, done in enumerate(dones):
            info = infos[idx]
            interrupted[idx] = bool(
                info.get("TimeLimit.truncated", False)
                or info.get("dead_screen", False)
            )
            terminal_obs = info.get("terminal_observation") if done else None
            if isinstance(terminal_obs, dict):
                phases[idx] = np.asarray(terminal_obs["phase"]).reshape(-1)[0]
                hp[idx] = np.asarray(
                    terminal_obs["run_hp"], dtype=np.float32
                ).reshape(-1)[0]
        return phases, hp, interrupted

    def _apply_hierarchical_returns(self, rollout_buffer, captured) -> None:
        if len(captured) != rollout_buffer.buffer_size:
            raise RuntimeError(
                "hierarchical transition capture length mismatch: "
                f"{len(captured)} != {rollout_buffer.buffer_size}"
            )
        with th.no_grad():
            last_values = self.policy.predict_values(
                obs_as_tensor(self._last_obs, self.device)
            ).detach().cpu().numpy().reshape(-1)

        phases = np.asarray(rollout_buffer.observations["phase"])[..., 0]
        raw_rewards = np.stack([item[0] for item in captured])
        dones = np.stack([item[1] for item in captured])
        next_phases = np.stack([item[2] for item in captured])
        next_hp = np.stack([item[3] for item in captured])
        interrupted = np.stack([item[4] for item in captured])
        combat_terminal_targets = (
            np.stack([item[5] for item in captured])
            if len(captured[0]) > 5
            else None
        )
        rewards, returns, advantages, stats = compute_hierarchical_returns(
            phases=phases,
            values=rollout_buffer.values,
            raw_rewards=raw_rewards,
            dones=dones,
            next_phases=next_phases,
            next_hp=next_hp,
            combat_terminal_targets=combat_terminal_targets,
            interrupted=interrupted,
            last_values=last_values,
            combat_phase=V.PHASE_VOCAB_INDEX["combat"],
            overworld_gamma=self.gamma,
            overworld_gae_lambda=self.gae_lambda,
            combat_gamma=self.combat_hp_gamma,
            combat_gae_lambda=self.combat_hp_gae_lambda,
        )
        rollout_buffer.rewards[...] = rewards
        rollout_buffer.returns[...] = returns
        rollout_buffer.advantages[...] = advantages
        self._hierarchical_rollout_stats = stats
        for key, value in vars(stats).items():
            self.logger.record(f"hierarchical/{key}", value)

    def collect_rollouts(self, env, callback, rollout_buffer,
                         n_rollout_steps, use_masking=True):
        # A previous update normally joins before starting the next rollout;
        # keep this guard so an early callback/error cannot leave a compiler
        # thread racing with a later buffer reuse.
        self._wait_for_reasoning_samples()
        if self.amp_mode == "bf16" and not getattr(
            self.policy, "supports_bf16_rollout", False
        ):
            raise TypeError(
                "bf16 rollout requires a policy that returns float32 values "
                "and log-probabilities"
            )
        captured = []
        if joint_upper(self):
            self._upper_completed = []
            self._upper_censored = 0
            reward_monitor.begin_collection(self)
            campfire_monitor.begin_collection(self)
            from rl.algorithms import upper_auxiliary
            upper_auxiliary.begin_collection(self, n_rollout_steps)
        reasoning_samples = []
        original_step_wait = env.step_wait
        original_step_async = env.step_async
        uncaptured_step_wait = env.step_wait

        if self.hierarchical_combat_hp:
            def capture_step_wait(_env):
                new_obs, rewards, dones, infos = uncaptured_step_wait()
                phases, hp, interrupted = self._next_phase_hp(
                    new_obs, dones, infos
                )
                terminal_targets = np.full(
                    len(infos), np.nan, dtype=np.float32
                )
                for index, info in enumerate(infos):
                    if "nosl_value_terminal_target" in info:
                        value = info["nosl_value_terminal_target"]
                        if value is not None:
                            terminal_targets[index] = float(value)
                captured.append((
                    np.asarray(rewards, dtype=np.float32).copy(),
                    np.asarray(dones, dtype=bool).copy(),
                    phases,
                    hp,
                    interrupted,
                    terminal_targets,
                ))
                return new_obs, rewards, dones, infos

            env.step_wait = MethodType(capture_step_wait, env)

        reasoning_enabled = (
            self.reasoning_coef > 0.0
            and self.reasoning_updates > 0
            and hasattr(getattr(self.policy, "core", None), "reasoning_loss")
        )
        pending_reasoning = {}
        # Keep only the small action/trace side channel during collection.  A
        # previous implementation copied every observation field for all
        # environments in step_async, even though SB3 writes the same
        # pre-action observations into rollout_buffer immediately afterwards.
        # That copy was pure CPU/memory-bandwidth overhead on every vector step.
        raw_reasoning_steps = []
        if reasoning_enabled:
            captured_step_wait = env.step_wait

            def capture_reasoning_step_async(_env, actions):
                pending_reasoning["actions"] = (
                    np.asarray(actions).reshape(-1).copy()
                )
                return original_step_async(actions)

            def capture_reasoning_step_wait(_env):
                result = captured_step_wait()
                _new_obs, _rewards, _dones, infos = result
                actions = pending_reasoning.pop("actions", None)
                if actions is None:
                    raise RuntimeError("reasoning step_wait has no matching step_async")
                raw_reasoning_steps.append(
                    (actions, [info.get("reasoning_trace") for info in infos])
                )
                return result

            env.step_async = MethodType(capture_reasoning_step_async, env)
            env.step_wait = MethodType(capture_reasoning_step_wait, env)

        try:
            if self._outer_return:
                from rl.search.rollout_inference import RolloutInference
                self._rollout_inference = RolloutInference(self, upper_value_only=True)
            with self.autocast_context():
                if self.nosl_is_az:
                    result = self._collect_nosl_is_az_rollouts(
                        env, callback, rollout_buffer, n_rollout_steps, use_masking
                    )
                elif not self.performance_profile:
                    result = super().collect_rollouts(
                        env, callback, rollout_buffer, n_rollout_steps, use_masking
                    )
                else:
                    result = self._collect_profiled_rollouts(
                        env, callback, rollout_buffer, n_rollout_steps, use_masking
                    )
        finally:
            env.step_wait = original_step_wait
            env.step_async = original_step_async
            runtime = self.__dict__.pop('_rollout_inference', None)
            if runtime is not None:
                try:
                    for key, value in runtime.statistics().items():
                        self.logger.record(f'nosl_is_az/{key}', value)
                finally:
                    runtime.clear()

        if (result and self.hierarchical_combat_hp
                and self.nosl_is_az_combat_strategy != 'small-playout'):
            self._apply_hierarchical_returns(rollout_buffer, captured)
        if result and reasoning_enabled:
            if len(raw_reasoning_steps) != rollout_buffer.buffer_size:
                raise RuntimeError(
                    "reasoning transition capture length mismatch: "
                    f"{len(raw_reasoning_steps)} != "
                    f"{rollout_buffer.buffer_size}"
                )
            # RolloutBuffer stores the observation that preceded each action;
            # snapshot it before RolloutBuffer.get() swaps/ flattens the
            # arrays in place.  The compiler is pure CPU work, so run it while
            # PPO performs its GPU backward pass and join only immediately
            # before the reasoning loss consumes it.  One batch copy is much
            # cheaper than copying every field for every env at every step.
            observation_snapshot = {
                key: np.array(value, copy=True)
                for key, value in rollout_buffer.observations.items()
            }
            self._reasoning_rollout_samples = []
            self._reasoning_compile_error = None
            self._reasoning_compile_thread = Thread(
                target=self._compile_reasoning_samples,
                args=(raw_reasoning_steps, observation_snapshot),
                name="reasoning-trace-compile",
                daemon=True,
            )
            self._reasoning_compile_thread.start()
        else:
            self._reasoning_rollout_samples = []
        return result

    def _collect_nosl_is_az_rollouts(
        self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True
    ):
        """Collect a whole-run rollout with worker-side combat search.

        This is intentionally close to sb3-contrib's MaskablePPO collector.
        The only extra work is (1) obtaining public policy logits, (2) asking
        each combat worker for a NOSL search decision, and (3) retaining the
        search target for the AZ update. Upper rules are supported only with
        the frozen outer policy in inner-only mode.
        """
        from stable_baselines3.common.vec_env import VecEnv

        if not isinstance(env, VecEnv):
            raise TypeError("NOSL-IS-AZ collection requires a VecEnv")
        if not is_masking_supported(env):
            raise ValueError("NOSL-IS-AZ requires an action-masking VecEnv")
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        self._nosl_is_az_samples = []
        combat_phase = V.PHASE_VOCAB_INDEX["combat"]
        # AlphaZero value labels are only known when the current fight ends.
        # Keep each environment's roots across rollout boundaries and label the
        # whole sequence only when the real fight ends.
        active_combat_samples = getattr(
            self, "_nosl_is_az_pending_fights", None
        )
        if (
            not isinstance(active_combat_samples, list)
            or len(active_combat_samples) != env.num_envs
        ):
            active_combat_samples = [[] for _ in range(env.num_envs)]
        self._nosl_is_az_pending_fights = active_combat_samples
        if self._outer_return:
            outer.begin_collection(self, env.num_envs)
        incomplete_combat_roots = 0
        completed_combat_roots = 0
        collected_value_boundaries = 0
        collected_combat_terminals = 0
        completed_combats = 0
        rollout_hp_order_count = 0
        rollout_hp_order_matches = 0
        cpp_leaf_batches = 0
        cpp_leaf_evaluations = 0
        leaf_graph = getattr(self, '_native_leaf_inference', None)
        graph_before = leaf_graph.statistics() if hasattr(leaf_graph, 'statistics') else {}
        # Keep the staged NOSL pipeline measurable.  These counters include
        # worker IPC where named, so the next run can distinguish simulator
        # work from Python packing and learner inference without sampling every
        # individual leaf.
        nosl_timings = {
            "root_policy_seconds": 0.0,
            "cpp_begin_rpc_seconds": 0.0,
            "cpp_leaf_decode_seconds": 0.0,
            "cpp_leaf_model_seconds": 0.0,
            "cpp_backprop_rpc_seconds": 0.0,
            "cpp_stream_wait_seconds": 0.0,
            "cpp_ready_empty_wait_seconds": 0.0,
            "env_step_seconds": 0.0,
            "selection_prefix_decisions": 0,
            "simulator_transitions": 0,
        }
        nosl_timings.update({name: 0.0 for name in WORKER_TIMING_KEYS + NATIVE_TIMING_KEYS})
        inference_batch_size_sum = 0.0
        inference_batch_count = 0
        inference_batch_size_max = 0
        inference_schedule_stats = {
            "target_sum": 0.0,
            "target_count": 0,
            "ready_backlog_sum": 0.0,
            "ready_backlog_count": 0,
            "ready_backlog_max": 0,
            "leaf_queue_wait_seconds": 0.0,
            "leaf_worker_wait_seconds": 0.0,
            "full_flushes": 0,
            "drained_flushes": 0,
            "watchdog_flushes": 0,
            "controller_changes": 0,
            "controller_increases": 0,
            "controller_decreases": 0,
            "controller_rollbacks": 0,
            "target_current": float(getattr(
                self, "nosl_is_az_inference_batch_size", 32
            )),
            "watchdog_seconds": float(getattr(
                self, "nosl_is_az_inference_max_wait_ms", 0.5
            )) / 1000.0,
        }
        from rl.training import episode_rollout
        cohort = episode_rollout.begin(self, env)
        executed_actions = 0
        n_steps = 0
        action_masks = None
        startup_paused: set[int] = set()
        if getattr(self, "_startup_occupancy_sampling", False):
            startup_paused = {
                index for index, state in enumerate(
                    env.env_method("startup_occupancy_state")
                )
                if state["accepted"] and not state["released"]
            }
        started = time.perf_counter()

        upper_rules = getattr(self, "nosl_is_az_upper_strategy", "model") == "rules"
        rulebase_lower = getattr(
            self, "nosl_is_az_combat_strategy", "model"
        ) in ("turn-planner", "small-playout")
        small_lower = getattr(self, "nosl_is_az_combat_strategy", "model") == "small-playout"
        if small_lower:
            from rl.training.small_playout import configure_collection
            configure_collection(self, env)
        if upper_rules and not self.nosl_is_az_inner_only:
            raise ValueError("NOSL upper rules require inner-only NOSL training")
        overworld_rule_decisions = overworld_fallback_decisions = 0
        rulebase_lower_decisions = rulebase_lower_nodes = 0
        rulebase_lower_roots = 0
        rulebase_lower_seconds = 0.0
        rulebase_lower_batch_seconds = 0.0
        forced_end_turn_roots = 0
        end_action = C.policy_offset(self.combat_action_contract)

        def call_workers(method_name, calls):
            if not calls:
                return {}
            batch_call = getattr(env, "env_method_batch", None)
            if batch_call is not None:
                return batch_call(method_name, calls)
            return {
                index: env.env_method(
                    method_name, *args, indices=index, **kwargs,
                )[0]
                for index, (args, kwargs) in calls.items()
            }

        while (env.episode_rollout_active.any() if cohort is not None else n_steps < n_rollout_steps):
            if cohort is not None:
                if not episode_rollout.check_budget(cohort, n_steps, env):
                    break
                startup_paused = set(np.flatnonzero(~env.episode_rollout_active).tolist())
            active_count = env.num_envs - len(startup_paused) if cohort is not None else env.num_envs
            policy_started = time.perf_counter()
            forced = np.zeros(env.num_envs, dtype=bool)
            with th.inference_mode():
                # Root, leaf, and learner batches use the same compact
                # dtype-packed transport.  The router still selects only the
                # fields needed by each phase after this value-preserving move.
                if use_masking:
                    action_masks = get_action_masks(env)
                # Whole-run policies expose this core API.  Keeping a strict
                # check here makes failures explicit for unsupported policies.
                core = getattr(self.policy, "core", None)
                if core is None or not hasattr(core, "actor_critic"):
                    raise TypeError(
                        "NOSL-IS-AZ requires a policy exposing core.actor_critic"
                    )
                compact_cpp_observations = bool(
                    self.nosl_is_az_cpp
                    and hasattr(core, "combat_actor_critic")
                    and hasattr(core, "combat_observation_keys")
                )
                phase_now = np.asarray(
                    self._last_obs["phase"]
                ).reshape(env.num_envs, -1)[:, 0]
                if (self.nosl_is_az_inner_only and getattr(core, "joint_actions", False)
                        and action_masks is not None):
                    forced = ((phase_now == combat_phase) & action_masks[:, end_action]
                              & (action_masks.sum(axis=1) == 1))
                    if startup_paused:
                        forced[list(startup_paused)] = False
                mixed_raw = self._outer_return and outer.is_mixed_raw(self)
                # Forced end-turn still belongs to the executed trajectory mean.
                eligible = np.ones(env.num_envs, dtype=bool) if mixed_raw else ~forced
                if rulebase_lower:
                    rulebase_roots = phase_now == combat_phase
                    if cohort is not None:
                        rulebase_roots &= env.episode_rollout_active
                    eligible = eligible & ~rulebase_roots
                    rulebase_lower_roots += int(rulebase_roots.sum())
                if getattr(core, 'combat_only', False):
                    eligible = eligible & (phase_now == combat_phase)
                if cohort is not None:
                    eligible &= env.episode_rollout_active
                active = np.flatnonzero(eligible)
                if len(active) < env.num_envs:
                    # These buffer values are unused by inner-only training.
                    # Diagnostics below explicitly mark the missing predictions.
                    logits = th.zeros((env.num_envs, self.action_space.n), device=self.device)
                    values = th.zeros((env.num_envs, 1), device=self.device)
                    if len(active):
                        obs_tensor = _leaf_observation_as_tensor(
                            {k:v[active] for k,v in self._last_obs.items()}, self.device)
                        active_logits, active_values = core.actor_critic(obs_tensor)
                        logits[active] = active_logits.float()
                        values[active] = active_values.float()
                else:
                    obs_tensor = _leaf_observation_as_tensor(self._last_obs, self.device)
                    logits, values = core.actor_critic(obs_tensor)
                distribution = self.policy.action_dist.proba_distribution(
                    action_logits=logits
                )
                if action_masks is not None:
                    distribution.apply_masking(action_masks)
                actions = distribution.get_actions(deterministic=upper_rules)
                values = values.float()
            nosl_timings["root_policy_seconds"] += (
                time.perf_counter() - policy_started
            )

            actions_np = actions.cpu().numpy().reshape(-1)
            reward_probabilities = (reward_monitor.policy_probabilities(distribution, self._last_obs)
                                    if joint_upper(self) else None)
            campfire_probabilities = (campfire_monitor.policy_probabilities(distribution, self._last_obs)
                                      if joint_upper(self) else None)
            if self._frozen_upper:
                outer.upper_actions(self, self._last_obs, action_masks, actions_np, logits)
            # BF16 autocast tensors do not have a numpy representation on
            # several PyTorch versions; IPC always receives public FP32 logits.
            logits_np = logits.detach().float().cpu().numpy()
            if joint_upper(self):
                from rl.algorithms import upper_auxiliary
                upper_auxiliary.collect(self, self._last_obs, action_masks, logits_np,
                                        n_steps, call_workers, startup_paused)
            values_np = values.detach().float().cpu().numpy().reshape(-1)
            values_np = values_np.copy()
            forced_end_turn_roots += int(forced.sum())
            if not mixed_raw:
                values_np[forced] = np.nan
            phase = np.asarray(self._last_obs["phase"]).reshape(
                env.num_envs, -1
            )[:, 0]
            for env_index, phase_id in enumerate(phase):
                if (int(phase_id) == combat_phase or not active_combat_samples[env_index]
                        or outer.is_monte_carlo(self)):
                    continue
                # The environment can be externally reset between collectors.
                # Such roots have no objective terminal HP and must not leak
                # into the next fight.
                incomplete_combat_roots += len(active_combat_samples[env_index])
                active_combat_samples[env_index] = []
                if self._outer_return:
                    self._outer_pending_rewards[env_index] = 0.
            if action_masks is None:
                action_masks = np.ones(
                    (env.num_envs, int(self.action_space.n)), dtype=bool
                )
            # Run NOSL PUCT simulations in the workers, but evaluate every
            # non-terminal leaf with the current learner network in batches.
            # ``env_method`` broadcasts one argument tuple; the batch helper
            # sends distinct logits/nonces to all combat workers concurrently.
            combat_indices = [
                index for index in range(env.num_envs)
                if int(phase[index]) == combat_phase and index not in startup_paused
                and not forced[index]
            ]
            if upper_rules:
                upper_results = call_workers("overworld_rule_decision", {
                    index: ((), {}) for index in range(env.num_envs)
                    if int(phase[index]) != combat_phase and index not in startup_paused
                })
                unsupported = [index for index, action in upper_results.items()
                               if action is None and getattr(self, '_combat_distribution', False)]
                if unsupported:
                    diagnostics = call_workers("overworld_rule_diagnostics", {
                        index: ((), {}) for index in unsupported
                    })
                    raise ValueError(
                        "entity-combat has no overworld model fallback; "
                        f"unsupported rule screens: {diagnostics!r}"
                    )
                for index, action in upper_results.items():
                    if action is None:
                        overworld_fallback_decisions += 1
                        continue
                    action = int(action)
                    if not 0 <= action < action_masks.shape[1] or not action_masks[index, action]:
                        raise ValueError(f"illegal overworld rule action: env={index} action={action}")
                    actions_np[index] = action
                    overworld_rule_decisions += 1
            result_by_index = {}
            if combat_indices and small_lower:
                small_started = time.perf_counter()
                prepare_steps = getattr(env, 'prepare_small_playout_steps', None)
                if prepare_steps is not None:
                    prepare_steps(actions_np, [index for index in range(env.num_envs)
                        if int(phase[index]) != combat_phase and index not in startup_paused])
                result_by_index = call_workers('small_playout_decision', {
                    index: ((), {'state_nonce': (int(self.nosl_is_az_master_nonce)
                        ^ int(self.num_timesteps) * 0x9E3779B97F4A7C15
                        ^ int(n_steps) * 0xD1B54A32D192ED03 ^ index) & ((1 << 64)-1)})
                    for index in combat_indices})
                rulebase_lower_batch_seconds += time.perf_counter()-small_started
            if combat_indices and rulebase_lower and not small_lower:
                rule_batch_started = time.perf_counter()
                results = call_workers("rulebase_turn_decision", {
                    env_index: ((), {
                        "state_nonce": (
                            int(self.nosl_is_az_master_nonce)
                            ^ (int(self.num_timesteps) * 0x9E3779B97F4A7C15)
                            ^ (int(n_steps) * 0xD1B54A32D192ED03)
                            ^ int(env_index)
                        ),
                        "worlds": int(getattr(self, "rulebase_planner_worlds", 2)),
                        "node_budget": int(getattr(self, "rulebase_planner_nodes", 8192)),
                        "beam_width": int(getattr(self, "rulebase_planner_beam", 8)),
                        "max_depth": int(getattr(self, "rulebase_planner_depth", 16)),
                        "master_nonce": int(getattr(self, "rulebase_planner_nonce", 20260918)),
                        "lookahead_turns": int(getattr(self, "rulebase_planner_turns", 1)),
                        "leaf_profile": str(getattr(
                            self, "rulebase_planner_leaf_profile", "legacy-v8"
                        )),
                    }) for env_index in combat_indices
                })
                rulebase_lower_batch_seconds += time.perf_counter() - rule_batch_started
                for env_index, response in results.items():
                    result_by_index[env_index] = response
                    diagnostics = response.get("diagnostics", {})
                    rulebase_lower_decisions += int(
                        diagnostics.get("search_decisions", 0)
                        or diagnostics.get("rule_decisions", 0)
                    )
                    rulebase_lower_nodes += int(diagnostics.get("expanded_nodes", 0))
                    rulebase_lower_seconds += float(diagnostics.get("search_seconds", 0.0))
            if combat_indices and not rulebase_lower:
                begin_calls = {}
                for env_index in combat_indices:
                    startup_nonce_rng = getattr(
                        self, "_startup_occupancy_nonce_rng", None
                    )
                    state_nonce = (
                        int(startup_nonce_rng.integers(
                            0, 1 << 64, dtype=np.uint64
                        ))
                        if startup_nonce_rng is not None else
                        int(self.nosl_is_az_master_nonce)
                        ^ (int(self.num_timesteps) * 0x9E3779B97F4A7C15)
                        ^ (int(n_steps) * 0xD1B54A32D192ED03)
                        ^ int(env_index)
                    )
                    if self.nosl_is_az_cpp:
                        begin_calls[env_index] = (
                            (logits_np[env_index],),
                            {
                                "state_nonce": state_nonce,
                                "simulations": self.nosl_is_az_simulations,
                                "depth": self.nosl_is_az_depth,
                                "end_turn_as_leaf": self.nosl_is_az_end_turn_as_leaf,
                                "fixed_worldline": bool(getattr(self, 'nosl_is_az_fixed_worldline', False)),
                                "c_puct": self.nosl_is_az_c_puct,
                                "leaf_batch_size": self.nosl_is_az_leaf_batch_size,
                                "dirichlet_alpha": self.nosl_is_az_dirichlet_alpha,
                                "dirichlet_epsilon": self.nosl_is_az_dirichlet_epsilon,
                                "root_value": float(values_np[env_index]),
                                "prior_temperature": self.nosl_is_az_prior_temperature,
                                "root_search_mode": self.nosl_is_az_root_search_mode,
                                "paper_full_gumbel": self.nosl_is_az_paper_full_gumbel,
                                "gumbel_candidates": self.nosl_is_az_gumbel_candidates,
                                "gumbel_scale": self.nosl_is_az_gumbel_scale,
                                "merge_equivalent_actions": self.nosl_is_az_merge_equivalent_actions,
                                "boss_win_weight": 0.0 if self._outer_return else getattr(self.policy, "entity_config", {}).get("boss_win_weight", 0.0),
                                "outer_return": self._outer_return,
                                "gumbel_value_scale": self.nosl_is_az_gumbel_value_scale,
                                "gumbel_max_visit_init": self.nosl_is_az_gumbel_max_visit_init,
                                "gumbel_rescale_q": self.nosl_is_az_gumbel_rescale_q,
                                "gumbel_use_mixed_value": self.nosl_is_az_gumbel_use_mixed_value,
                                "gumbel_interior_uniformity": self.nosl_is_az_gumbel_interior_uniformity,
                                "gumbel_interior_min_visits": int(getattr(
                                    self,
                                    "nosl_is_az_gumbel_interior_min_visits",
                                    1,
                                )),
                                "gumbel_interior_coverage_fraction": self.nosl_is_az_gumbel_interior_coverage_fraction,
                                "gumbel_interior_value_mode": self.nosl_is_az_gumbel_interior_value_mode,
                                "gumbel_interior_max_mix_visits": self.nosl_is_az_gumbel_interior_max_mix_visits,
                                "compact_observation": compact_cpp_observations,
                                "boss_transition_target": (
                                    self.nosl_is_az_boss_transition_target
                                ),
                            },
                        )
                    else:
                        begin_calls[env_index] = (
                            (logits_np[env_index],),
                            {
                                "state_nonce": state_nonce,
                                "master_nonce": self.nosl_is_az_master_nonce,
                                "simulations": self.nosl_is_az_simulations,
                                "depth": self.nosl_is_az_depth,
                                "rollout_steps": self.nosl_is_az_rollout_steps,
                                "c_puct": self.nosl_is_az_c_puct,
                                "temperature": self.nosl_is_az_temperature,
                                "dirichlet_alpha": self.nosl_is_az_dirichlet_alpha,
                                "dirichlet_epsilon": self.nosl_is_az_dirichlet_epsilon,
                                "boss_transition_target": (
                                    self.nosl_is_az_boss_transition_target
                                ),
                            },
                        )

                if self.nosl_is_az_cpp:
                    # C++ owns one tree per worker environment. Grouped workers
                    # stream independent trees through bounded cross-tree GPU
                    # batches when each tree has one pending leaf; other VecEnv
                    # layouts retain the synchronous batch protocol.
                    stream_factory = getattr(env, "env_method_stream", None)
                    native_batch = getattr(env, "native_search_batch", None)
                    if native_batch is not None:
                        result_by_index, shared_stats = complete_searches_native(
                            self, native_batch, begin_calls, autocast=self.autocast_context,
                            root_observations=self._last_obs if self._frozen_upper else None)
                        for key in NATIVE_TIMING_KEYS:
                            nosl_timings[key] += shared_stats[key]
                    elif (
                        callable(stream_factory)
                        and self.nosl_is_az_leaf_batch_size == 1
                    ):
                        result_by_index, shared_stats = (
                            complete_searches_streaming(
                                self,
                                stream_factory(),
                                begin_calls,
                                leaf_batch_size=1,
                                schedule=LeafInferenceSchedule(
                                    target_batch_size=int(getattr(
                                        self,
                                        "nosl_is_az_inference_batch_size",
                                        32,
                                    )),
                                    max_wait_seconds=float(getattr(
                                        self,
                                        "nosl_is_az_inference_max_wait_ms",
                                        0.5,
                                    )) / 1000.0,
                                    mode=str(getattr(
                                        self,
                                        "nosl_is_az_inference_schedule",
                                        "fixed_wait",
                                    )),
                                ),
                                compact=compact_cpp_observations,
                                autocast=self.autocast_context,
                            )
                        )
                        nosl_timings["cpp_begin_rpc_seconds"] += shared_stats[
                            "cpp_begin_rpc_seconds"
                        ]
                        nosl_timings["cpp_stream_wait_seconds"] += shared_stats[
                            "cpp_stream_wait_seconds"
                        ]
                        nosl_timings["cpp_ready_empty_wait_seconds"] += (
                            shared_stats["cpp_ready_empty_wait_seconds"]
                        )
                    else:
                        rpc_started = time.perf_counter()
                        responses = call_workers(
                            "nosl_is_az_cpp_begin", begin_calls
                        )
                        nosl_timings["cpp_begin_rpc_seconds"] += (
                            time.perf_counter() - rpc_started
                        )
                        result_by_index, shared_stats = complete_searches(
                            self, call_workers, responses,
                            leaf_batch_size=self.nosl_is_az_leaf_batch_size,
                            compact=compact_cpp_observations,
                            autocast=self.autocast_context,
                        )
                    cpp_leaf_batches += shared_stats["leaf_batches"]
                    cpp_leaf_evaluations += shared_stats["leaf_evaluations"]
                    inference_batch_size_sum += shared_stats["leaf_evaluations"]
                    inference_batch_count += shared_stats["leaf_batches"]
                    inference_batch_size_max = max(
                        inference_batch_size_max,
                        int(shared_stats.get("inference_batch_size_max", 0)),
                    )
                    inference_schedule_stats["target_sum"] += shared_stats.get(
                        "inference_target_batch_sum", 0
                    )
                    inference_schedule_stats["target_count"] += shared_stats.get(
                        "inference_target_batch_count", 0
                    )
                    inference_schedule_stats["ready_backlog_sum"] += shared_stats.get(
                        "inference_ready_backlog_sum", 0
                    )
                    inference_schedule_stats["ready_backlog_count"] += shared_stats.get(
                        "inference_ready_backlog_count", 0
                    )
                    inference_schedule_stats["ready_backlog_max"] = max(
                        inference_schedule_stats["ready_backlog_max"],
                        int(shared_stats.get("inference_ready_backlog_max", 0)),
                    )
                    for source, target_name in (
                        ("inference_leaf_queue_wait_seconds", "leaf_queue_wait_seconds"),
                        ("inference_leaf_worker_wait_seconds", "leaf_worker_wait_seconds"),
                        ("inference_full_flushes", "full_flushes"),
                        ("inference_drained_flushes", "drained_flushes"),
                        ("inference_watchdog_flushes", "watchdog_flushes"),
                        ("inference_controller_changes", "controller_changes"),
                        ("inference_controller_increases", "controller_increases"),
                        ("inference_controller_decreases", "controller_decreases"),
                        ("inference_controller_rollbacks", "controller_rollbacks"),
                    ):
                        inference_schedule_stats[target_name] += shared_stats.get(
                            source, 0
                        )
                    inference_schedule_stats["target_current"] = shared_stats.get(
                        "inference_batch_target_current",
                        inference_schedule_stats["target_current"],
                    )
                    inference_schedule_stats["watchdog_seconds"] = shared_stats.get(
                        "inference_watchdog_seconds",
                        inference_schedule_stats["watchdog_seconds"],
                    )
                    for key in ("cpp_leaf_decode_seconds", "cpp_leaf_model_seconds", "cpp_backprop_rpc_seconds"):
                        nosl_timings[key] += shared_stats[key]
                    for key in WORKER_TIMING_KEYS:
                        nosl_timings[key] += shared_stats.get(key, 0.0)
                else:
                    requests = call_workers("nosl_is_az_begin", begin_calls)
                    active = {
                        index: request
                        for index, request in requests.items()
                        if not request.get("done", False)
                    }
                    for index, request in requests.items():
                        if request.get("done", False):
                            result_by_index[index] = request["result"]

                    while active:
                        active_indices = sorted(active)
                        leaf_observations = {
                            key: np.stack([
                                np.asarray(active[index]["observation"][key])
                                for index in active_indices
                            ])
                            for key in active[active_indices[0]]["observation"]
                        }
                        leaf_tensor = _leaf_observation_as_tensor(
                            leaf_observations, self.device
                        )
                        with th.inference_mode():
                            with self.autocast_context():
                                leaf_logits, leaf_values = core.actor_critic(
                                    leaf_tensor
                                )
                        leaf_logits_np = (
                            leaf_logits.detach().float().cpu().numpy()
                        )
                        leaf_values_np = (
                            leaf_values.detach().float().cpu().numpy().reshape(-1)
                        )
                        backprop_calls = {
                            index: (
                                (float(leaf_values_np[position]),
                                 leaf_logits_np[position]),
                                {},
                            )
                            for position, index in enumerate(active_indices)
                        }
                        responses = call_workers(
                            "nosl_is_az_backprop", backprop_calls
                        )
                        active = {
                            index: response
                            for index, response in responses.items()
                            if not response.get("done", False)
                        }
                        for index, response in responses.items():
                            if response.get("done", False):
                                result_by_index[index] = response["result"]

            for index in np.flatnonzero(forced):
                result_by_index[int(index)] = dict(action=end_action,
                    visit_policy=action_masks[index].astype(np.float32), root_value=np.nan)
            for env_index in range(env.num_envs):
                if (
                    int(phase[env_index]) != combat_phase
                    or env_index in startup_paused
                ):
                    continue
                result = result_by_index[env_index]
                actions_np[env_index] = int(result["action"])
                if (self.combat_action_contract in C.NEURAL_CONTRACTS
                        and not rulebase_lower and not forced[env_index]
                        and "action_groups" not in result):
                    raise RuntimeError("neural search result omitted action groups")
                if not rulebase_lower:
                    active_combat_samples[env_index].append(
                        {
                        # Replay outlives the PPO rollout buffer, so retain an
                        # independent combat-only observation at every root.
                        "observation": {
                            key: np.asarray(self._last_obs[key][env_index]).copy()
                            for key in getattr(
                                core,
                                "combat_observation_keys",
                                self._last_obs.keys(),
                            )
                            if key in self._last_obs
                        },
                        "action_mask": np.asarray(
                            action_masks[env_index], dtype=bool
                        ).copy(),
                        "policy_target": np.asarray(
                            result["visit_policy"], dtype=np.float32
                        ).copy(),
                        "action_groups": np.asarray(result.get("action_groups",
                            np.arange(self.action_space.n)), dtype=np.int16).copy(),
                        "policy_weight": 1.0,
                        # Filled with objective final combat HP below.  The
                        # search root value remains diagnostic only.
                        "value_target": np.nan,
                        "search_value": float(result["root_value"]),
                        "estimated_value": float(values_np[env_index]),
                        "current_hp": _normalized_player_hp(
                            {
                                key: self._last_obs[key][env_index]
                                for key in ("run_hp", "player_scalars")
                                if key in self._last_obs
                            }
                        ),
                        "sample_kind": "combat_root",
                        }
                    )
            actions = th.as_tensor(actions_np, dtype=th.long, device=self.device)
            # Search/rules may replace model actions. Recompute their old
            # log-probability so the rollout remains internally consistent even
            # though combat samples are excluded from the PPO objective below.
            log_probs = distribution.log_prob(actions).float()
            env_started = time.perf_counter()
            if self._outer_return:
                outer.stamp_roots(self, active_combat_samples)
            new_obs, rewards, dones, infos = env.step(actions_np)
            if cohort is not None:
                episode_rollout.record_terminals(cohort, dones, infos)
            if self._outer_return:
                completed, discarded, fights = outer.finish_step(self, self._last_obs,
                    new_obs, dones, infos, active_combat_samples)
                completed_combat_roots += completed
                incomplete_combat_roots += discarded
                completed_combats += fights
            if joint_upper(self):
                capture_step(self, self._last_obs, actions_np, action_masks,
                    values_np, log_probs.detach().cpu().numpy(), new_obs, dones, infos)
                reward_monitor.record_step(self, self._last_obs, actions_np, action_masks,
                    reward_probabilities, infos)
                campfire_monitor.record_step(self, actions_np, action_masks, campfire_probabilities, infos)
            prefix_steps = sum(info.get('simulator_transition') is False
                               and not info.get('episode_cohort_hold') for info in infos)
            nosl_timings['selection_prefix_decisions'] += prefix_steps
            nosl_timings['simulator_transitions'] += active_count - prefix_steps
            if getattr(self, "_startup_occupancy_sampling", False):
                startup_paused.update(
                    index for index, info in enumerate(infos)
                    if info.get("startup_occupancy_accepted")
                )
                reset_indices = np.flatnonzero(dones).tolist()
                if reset_indices:
                    startup_paused.update(
                        index for index, state in zip(
                            reset_indices,
                            env.env_method(
                                "startup_occupancy_state", indices=reset_indices,
                            ),
                        )
                        if state["accepted"] and not state["released"]
                    )
            nosl_timings["env_step_seconds"] += (
                time.perf_counter() - env_started
            )
            self.num_timesteps += active_count
            executed_actions += active_count

            next_phases, next_hp, interrupted = self._next_phase_hp(
                new_obs, dones, infos
            )
            for env_index in range(env.num_envs):
                if self._outer_return:
                    continue
                if (
                    int(phase[env_index]) != combat_phase
                    or env_index in startup_paused
                ):
                    continue
                # Only the boundary on the action actually taken belongs to
                # this trajectory.  It receives the fight's eventual HP label
                # but no policy target; hypothetical search leaves must never
                # be assigned the selected line's return.
                boundary_observation = infos[env_index].pop(
                    "nosl_value_boundary_observation", None
                )
                if _append_nosl_value_only_sample(
                    active_combat_samples[env_index],
                    boundary_observation,
                    int(self.action_space.n),
                    sample_kind="end_turn_boundary",
                ):
                    collected_value_boundaries += 1
                terminal_observation = infos[env_index].pop(
                    "nosl_value_terminal_observation", None
                )
                terminal_target = infos[env_index].pop(
                    "nosl_value_terminal_target", np.nan
                )
                if (getattr(self,'_combat_distribution',False)
                        and self.policy.core.config.auxiliary_loss_coef):
                    from rl.algorithms.combat_distribution import transition_features
                    after = terminal_observation
                    if after is None and not dones[env_index] and next_phases[env_index] == combat_phase:
                        after = {key:value[env_index] for key,value in new_obs.items()}
                    if after is not None and infos[env_index].get('simulator_transition', True):
                        roots = active_combat_samples[env_index]
                        for root in reversed(roots):
                            if root.get('sample_kind') == 'combat_root':
                                root['executed_action'] = int(actions_np[env_index])
                                root['transition_target'] = transition_features(after)-transition_features(root['observation'])
                                break
                if _append_nosl_value_only_sample(
                    active_combat_samples[env_index],
                    terminal_observation,
                    int(self.action_space.n),
                    sample_kind="combat_terminal",
                ):
                    collected_combat_terminals += 1
                local_terminal = bool(
                    dones[env_index]
                    or next_phases[env_index] != combat_phase
                    or terminal_observation is not None
                )
                if not local_terminal:
                    continue
                pending = active_combat_samples[env_index]
                active_combat_samples[env_index] = []
                if interrupted[env_index]:
                    incomplete_combat_roots += len(pending)
                    continue
                if getattr(self,'_combat_distribution',False):
                    from rl.algorithms.combat_distribution import outcome_targets
                    outcome = infos[env_index].get('nosl_value_terminal_outcome')
                    if outcome not in ('win','loss') or not np.isfinite(terminal_target):
                        incomplete_combat_roots += len(pending)
                        continue
                    outcome_targets(
                        pending, outcome,
                        include_transition=bool(self.policy.core.config.auxiliary_loss_coef),
                    )
                target = next_hp[env_index]
                if np.isfinite(terminal_target):
                    target = terminal_target
                mode = getattr(getattr(self.policy, 'core', None), 'boss_ablation', None)
                if mode is not None:
                    from rl.algorithms.combat_distribution import boss_ablation_target
                    target = boss_ablation_target(mode,
                        boss=bool(pending and pending[0]['observation']['combat_is_boss'][0]),
                        outcome=infos[env_index].get('nosl_value_terminal_outcome'), value=target,
                        terminal_max_hp=infos[env_index].get('nosl_value_terminal_max_hp'))
                future_hp = float(target)
                if np.isfinite(future_hp):
                    for sample in pending:
                        if sample.get("sample_kind") != "combat_root":
                            continue
                        estimated_value = float(sample.get("estimated_value", np.nan))
                        current_hp = float(sample.get("current_hp", np.nan))
                        if not (np.isfinite(estimated_value) and np.isfinite(current_hp)):
                            continue
                        rollout_hp_order_count += 1
                        if estimated_value > current_hp and current_hp >= future_hp:
                            rollout_hp_order_matches += 1
                completed = _fill_nosl_combat_value_targets(
                    pending, target, clip_hp=mode != 'full-or-zero')
                self._nosl_is_az_samples.extend(pending)
                completed_combat_roots += completed
                completed_combats += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            actions_for_buffer = actions_np
            if isinstance(self.action_space, spaces.Discrete):
                actions_for_buffer = actions_for_buffer.reshape(-1, 1)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                    and not self._outer_return
                    and not small_lower
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    # ``predict_values`` returns a one-element tensor for a
                    # single terminal observation.  Convert it to a Python
                    # scalar before adding it to the scalar reward slot;
                    # assigning a shape-(1,) array can fail when the reward
                    # buffer uses a NumPy scalar dtype.
                    terminal_value_scalar = float(
                        terminal_value.detach().float().reshape(-1)[0].cpu()
                    )
                    rewards[idx] += self.gamma * terminal_value_scalar

            if cohort is None:
                rollout_buffer.add(
                    self._last_obs,
                    actions_for_buffer,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    action_masks=action_masks,
                )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = (th.zeros(env.num_envs, device=self.device) if small_lower else
                      self.policy.predict_values(obs_as_tensor(new_obs, self.device)))
        if small_lower:
            self._small_playout_stats = env.small_playout_stats()
            self._small_playout_samples = env.take_small_playout_samples()
            if cohort is not None:
                episode_rollout.exclude_samples(self, env)
            for key, value in self._small_playout_stats.items():
                self.logger.record('small_playout/'+key, value)
        if self._outer_return:
            # Only completed segment targets train this experiment. PPO's
            # rollout is retained for transport/lifecycle, never an HP target.
            rollout_buffer.returns.fill(np.nan)
            rollout_buffer.advantages.fill(0.)
            outer.finish_collection(self)
        elif cohort is None:
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if cohort is not None:
            episode_rollout.finish(self, env, cohort, n_steps)
        if self._frozen_upper:
            self.policy.core.teacher.assert_integrity()
            self.logger.record('frozen_upper/upper_updates', 0)
        callback.on_rollout_end()
        pending_combat_roots = sum(len(pending) for pending in active_combat_samples)
        self.logger.record(
            "nosl_is_az/incomplete_combat_roots", incomplete_combat_roots
        )
        self.logger.record(
            "nosl_is_az/pending_combat_roots", pending_combat_roots
        )
        self.logger.record(
            "nosl_is_az/completed_combat_roots", completed_combat_roots
        )
        self.logger.record(
            "nosl_is_az/collected_value_boundaries",
            collected_value_boundaries,
        )
        self.logger.record(
            "nosl_is_az/collected_combat_terminals",
            collected_combat_terminals,
        )
        # Keep a direct copy for the distributed learner.  The SB3 logger can
        # be dumped/reset between collection and the inner update, so relying
        # only on ``name_to_value`` makes these diagnostics disappear from
        # distributed/W&B records.
        self._nosl_is_az_rollout_timings = dict(nosl_timings)
        self._nosl_is_az_rollout_upper_counts = {
            "overworld_rule_decisions": float(overworld_rule_decisions),
            "overworld_fallback_decisions": float(overworld_fallback_decisions),
            "forced_end_turn_roots": float(forced_end_turn_roots),
            "rulebase_lower_roots": float(rulebase_lower_roots),
            "root_value_predictions_skipped": float(
                rulebase_lower_roots if rulebase_lower
                else (0 if self._outer_return and outer.is_mixed_raw(self) else forced_end_turn_roots)
            ),
            "rulebase_lower_decisions": float(rulebase_lower_decisions),
            "rulebase_lower_expanded_nodes": float(rulebase_lower_nodes),
            "rulebase_lower_search_seconds": float(rulebase_lower_seconds),
            "rulebase_lower_batch_seconds": float(rulebase_lower_batch_seconds),
        }
        for name, count in self._nosl_is_az_rollout_upper_counts.items():
            self.logger.record("nosl_is_az/" + name, count)
        rollout_hp_order_fraction = (
            float(rollout_hp_order_matches / rollout_hp_order_count)
            if rollout_hp_order_count
            else 0.0
        )
        self._nosl_is_az_rollout_hp_order = {
            "count": float(rollout_hp_order_count),
            "matches": float(rollout_hp_order_matches),
            "fraction": rollout_hp_order_fraction,
        }
        self.logger.record(
            "nosl_is_az/rollout_value_gt_current_hp_ge_future_hp_fraction",
            rollout_hp_order_fraction,
        )
        self.logger.record(
            "nosl_is_az/rollout_value_gt_current_hp_ge_future_hp_count",
            rollout_hp_order_count,
        )
        self.logger.record(
            "nosl_is_az/rollout_value_gt_current_hp_ge_future_hp_matches",
            rollout_hp_order_matches,
        )
        if self.nosl_is_az_cpp:
            self.logger.record("nosl_is_az/cpp_leaf_batches", cpp_leaf_batches)
            self.logger.record(
                "nosl_is_az/cpp_leaf_evaluations", cpp_leaf_evaluations
            )
            self.logger.record(
                "nosl_is_az/inference_batch_size_mean",
                (
                    inference_batch_size_sum / inference_batch_count
                    if inference_batch_count
                    else 0.0
                ),
            )
            self.logger.record(
                "nosl_is_az/inference_batch_size_max",
                inference_batch_size_max,
            )
            target_count = inference_schedule_stats["target_count"]
            backlog_count = inference_schedule_stats["ready_backlog_count"]
            self.logger.record(
                "nosl_is_az/inference_batch_target_mean",
                (
                    inference_schedule_stats["target_sum"] / target_count
                    if target_count else 0.0
                ),
            )
            self.logger.record(
                "nosl_is_az/inference_batch_target_current",
                inference_schedule_stats["target_current"],
            )
            self.logger.record(
                "nosl_is_az/inference_ready_backlog_mean",
                (
                    inference_schedule_stats["ready_backlog_sum"] / backlog_count
                    if backlog_count else 0.0
                ),
            )
            self.logger.record(
                "nosl_is_az/inference_ready_backlog_max",
                inference_schedule_stats["ready_backlog_max"],
            )
            leaf_count = max(1, cpp_leaf_evaluations)
            self.logger.record(
                "nosl_is_az/inference_leaf_queue_wait_ms_mean",
                1000.0 * inference_schedule_stats["leaf_queue_wait_seconds"]
                / leaf_count,
            )
            self.logger.record(
                "nosl_is_az/inference_leaf_worker_wait_ms_mean",
                1000.0 * inference_schedule_stats["leaf_worker_wait_seconds"]
                / leaf_count,
            )
            for name in (
                "full_flushes", "drained_flushes", "watchdog_flushes",
                "controller_changes", "controller_increases",
                "controller_decreases", "controller_rollbacks",
            ):
                self.logger.record(
                    f"nosl_is_az/inference_{name}",
                    inference_schedule_stats[name],
                )
            self.logger.record(
                "nosl_is_az/inference_watchdog_ms",
                1000.0 * inference_schedule_stats["watchdog_seconds"],
            )
            for name, seconds in nosl_timings.items():
                self.logger.record(f"nosl_is_az/{name}", seconds)
        self.logger.record(
            "nosl_is_az/completed_combats", completed_combats
        )
        self.logger.record(
            "nosl_is_az/collection_seconds", time.perf_counter() - started
        )
        if graph_before:
            graph_after = leaf_graph.statistics()
            for name, value in graph_after.items():
                if isinstance(value, (int, float)):
                    self.logger.record('nosl_is_az/leaf_graph_' + name,
                        value if name == 'cached_graphs' else value-graph_before[name])
        if self.performance_profile:
            total = time.perf_counter() - started
            self._performance_rollout = {
                "policy": 0.0,
                "env": total,
                "mask": 0.0,
                "overhead": 0.0,
                "total": total,
                "transitions": executed_actions,
                "calls": {"policy": n_steps, "env": n_steps, "mask": 0},
            }
        self.logger.record("nosl_is_az/combat_roots", len(self._nosl_is_az_samples))
        self.logger.record(
            "nosl_is_az/combat_policy_roots",
            sum(
                float(sample.get("policy_weight", 1.0)) > 0.0
                for sample in self._nosl_is_az_samples
            ),
        )
        self.logger.record(
            "nosl_is_az/value_only_samples",
            sum(
                float(sample.get("policy_weight", 1.0)) == 0.0
                for sample in self._nosl_is_az_samples
            ),
        )
        return True

    def _compile_reasoning_samples(self, raw_reasoning_steps, buffer_observations):
        """Materialize captured traces without blocking the next GPU phase."""
        try:
            reasoning_samples = []
            for step_index, (actions, traces) in enumerate(raw_reasoning_steps):
                observations = {
                    key: np.asarray(value[step_index])
                    for key, value in buffer_observations.items()
                }
                for env_index, trace in enumerate(traces):
                    if trace is None:
                        continue
                    sample = self._reasoning_sample(
                        observations,
                        env_index,
                        int(actions[env_index]),
                        trace,
                    )
                    if sample is not None:
                        reasoning_samples.append(sample)
            self._reasoning_rollout_samples = reasoning_samples
        except BaseException as error:
            self._reasoning_compile_error = error

    def _wait_for_reasoning_samples(self) -> None:
        thread = self._reasoning_compile_thread
        if thread is None:
            return
        thread.join()
        self._reasoning_compile_thread = None
        error = self._reasoning_compile_error
        self._reasoning_compile_error = None
        if error is not None:
            raise error

    @staticmethod
    def _reasoning_sample(observations, env_index, action, trace):
        """Compile one simulator trace and immediately discard its snapshots."""
        from env.core import action_space as A
        from rl.policies.wholerun_reasoning import (
            build_interactive_transcript,
            card_stochastic_flags,
            events_from_snapshots,
        )

        snapshots = trace.get("snapshots") or ()
        events = trace.get("events")
        if events is None:
            if len(snapshots) < 2:
                return None
            events = events_from_snapshots(snapshots)
        action_card = trace.get("action_card")
        flags = card_stochastic_flags(action_card)
        segment, offset = A.segment_of(action)
        transcript = build_interactive_transcript(
            events=events,
            action_segment=segment,
            action_offset=offset,
            action_card=action_card,
            **flags,
        )
        inputs, labels = transcript.teacher_labels()
        return {
            "observation": {
                key: np.asarray(value[env_index]).copy()
                for key, value in observations.items()
            },
            "action": int(action),
            "inputs": np.asarray(inputs, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
            "precision": str(trace.get("precision", "atomic")),
            "environment_tokens": int(sum(
                owner == 0 for owner in transcript.owners
            )),
        }

    def _collect_profiled_rollouts(self, env, callback, rollout_buffer,
                                   n_rollout_steps, use_masking):

        timings = {"policy": 0.0, "env": 0.0, "mask": 0.0}
        calls = {"policy": 0, "env": 0, "mask": 0}
        had_forward = "forward" in self.policy.__dict__
        saved_forward = self.policy.__dict__.get("forward")
        original_forward = self.policy.forward
        original_step_wait = env.step_wait
        original_env_method = env.env_method

        def timed_forward(_policy, *args, **kwargs):
            self._sync_device()
            started = time.perf_counter()
            result = original_forward(*args, **kwargs)
            self._sync_device()
            timings["policy"] += time.perf_counter() - started
            calls["policy"] += 1
            return result

        def timed_step_wait(_env):
            started = time.perf_counter()
            result = original_step_wait()
            timings["env"] += time.perf_counter() - started
            calls["env"] += 1
            return result

        def timed_env_method(_env, method_name, *args, **kwargs):
            started = time.perf_counter()
            result = original_env_method(method_name, *args, **kwargs)
            if method_name == "action_masks":
                timings["mask"] += time.perf_counter() - started
                calls["mask"] += 1
            return result

        self.policy.forward = MethodType(timed_forward, self.policy)
        env.step_wait = MethodType(timed_step_wait, env)
        env.env_method = MethodType(timed_env_method, env)
        self._sync_device()
        started = time.perf_counter()
        try:
            result = super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps, use_masking
            )
        finally:
            self._sync_device()
            total = time.perf_counter() - started
            if had_forward:
                self.policy.forward = saved_forward
            else:
                del self.policy.forward
            env.step_wait = original_step_wait
            env.env_method = original_env_method

        transitions = n_rollout_steps * env.num_envs
        known = timings["policy"] + timings["env"] + timings["mask"]
        self._performance_rollout = {
            **timings,
            "overhead": max(0.0, total - known),
            "total": total,
            "transitions": transitions,
            "calls": calls,
        }
        return result

    def _materialize_legacy_nosl_observations(self, samples: list[dict]) -> None:
        """Resolve pre-replay buffer indices used by older callers and tests."""
        missing = [sample for sample in samples if "observation" not in sample]
        if not missing:
            return
        if self.rollout_buffer.generator_ready:
            observations = self.rollout_buffer.observations
        else:
            observations = {
                key: self.rollout_buffer.swap_and_flatten(value)
                for key, value in self.rollout_buffer.observations.items()
            }
        keys = self._nosl_is_az_observation_keys()
        for sample in missing:
            index = int(sample.get("buffer_index", -1))
            if index < 0:
                raise ValueError("NOSL sample has neither observation nor buffer index")
            sample["observation"] = {
                key: np.asarray(observations[key][index]).copy()
                for key in keys
            }

    def _prepare_nosl_is_az_replay(self) -> dict[str, float]:
        """Prepare newly labelled fights for the next AZ update.

        The historical default is replay-enabled for checkpoint
        compatibility.  With replay explicitly disabled, samples remain in a
        short-lived fresh list and are consumed exactly once by the update.
        """
        raw_samples = list(getattr(self, "_nosl_is_az_samples", []))
        samples = [
            sample
            for sample in raw_samples
            if np.isfinite(float(sample.get("value_target", np.nan)))
        ]
        dropped = len(raw_samples) - len(samples)
        self._materialize_legacy_nosl_observations(samples)
        replay_enabled = bool(getattr(self, "nosl_is_az_replay_enabled", True))
        if not replay_enabled:
            self._nosl_is_az_fresh_samples = samples
            self._nosl_is_az_samples = []
            stats = {
                "fresh_roots": float(len(samples)),
                "dropped_unlabeled_roots": float(dropped),
                "replay_size": 0.0,
                "replay_capacity": 0.0,
                "replay_fill_ratio": 0.0,
                "replay_allocated_mb": 0.0,
                "replay_ready": float(bool(samples)),
                "replay_enabled": 0.0,
                "fresh_start_id": 0.0,
            }
            self._nosl_is_az_replay_prepare_stats = stats
            logger = getattr(self, "_logger", None)
            if logger is not None:
                logger.record("nosl_is_az/dropped_unlabeled_roots", dropped)
            return stats

        self._nosl_is_az_fresh_samples = []
        replay = self._ensure_nosl_is_az_replay()
        fresh_start_id = replay.total_added
        added = replay.add_samples(samples)
        self._nosl_is_az_samples = []
        ready = replay.size >= int(self.nosl_is_az_replay_min_size)
        stats = {
            "fresh_roots": float(added),
            "dropped_unlabeled_roots": float(dropped),
            "replay_size": float(replay.size),
            "replay_capacity": float(replay.capacity),
            "replay_fill_ratio": float(replay.size / replay.capacity),
            "replay_allocated_mb": float(replay.allocated_bytes / (1024**2)),
            "replay_ready": float(ready),
            "replay_enabled": 1.0,
            "fresh_start_id": float(fresh_start_id),
        }
        self._nosl_is_az_replay_prepare_stats = stats
        logger = getattr(self, "_logger", None)
        if logger is not None:
            logger.record("nosl_is_az/dropped_unlabeled_roots", dropped)
        return stats

    def _nosl_is_az_replay_train_count(self) -> int:
        if not bool(getattr(self, "nosl_is_az_replay_enabled", True)):
            return len(getattr(self, "_nosl_is_az_fresh_samples", ()))
        replay = self._ensure_nosl_is_az_replay()
        if replay.size < int(self.nosl_is_az_replay_min_size):
            return 0
        max_old_fraction = float(getattr(
            self, "nosl_is_az_replay_max_old_fraction", 1.0
        ))
        if max_old_fraction < 1.0:
            fresh_roots = float(getattr(
                self, "_nosl_is_az_replay_prepare_stats", {}
            ).get("fresh_roots", 0.0))
            # A protected update must never silently fall back to an all-old
            # batch.  Wait for the next completed fight collection instead.
            if fresh_roots < 1.0:
                return 0
        return min(int(self.nosl_is_az_batch_size), replay.size)

    @combat_update
    def _train_nosl_is_az(
        self,
        *,
        step_optimizer: bool = True,
        gradient_scale: float = 1.0,
        prepare_replay: bool = True,
        update_count: int | None = None,
        value_only_updates: int = 0,
        consume_fresh_samples: bool = True,
    ) -> dict[str, object]:
        """Train the combat specialist from a recent self-play replay window.

        One effective batch is sampled without replacement per requested AZ
        update. It is split into bounded microbatches for memory, accumulated as
        an exact sample-weighted mean, and stepped once. Distributed training
        leaves that mean gradient installed for the ring all-reduce. ``update_count``
        is an internal override used by distributed training to run one replay
        update per synchronized optimizer step while retaining the configured
        total number of inner updates.  Such callers set
        ``consume_fresh_samples=False`` until their final synchronized update;
        otherwise a no-replay batch would disappear after the first step.
        """
        if update_count is None:
            from rl.algorithms.update_schedule import CombatUpdateSchedule
            completed = max(0, (int(self.num_timesteps) - 1) // (self.n_steps * self.n_envs))
            update_count, value_only_updates = CombatUpdateSchedule.from_config(
                self.nosl_is_az_update_schedule).plan(completed, int(self.nosl_is_az_updates))
        else:
            update_count = int(update_count)
        if update_count < 1:
            raise ValueError("NOSL-IS-AZ update_count must be positive")
        if not 0 <= value_only_updates <= update_count or (value_only_updates and not self._combat_distribution):
            raise ValueError("value-only updates require a valid entity-combat budget")
        if prepare_replay:
            prepare_stats = self._prepare_nosl_is_az_replay()
        else:
            prepare_stats = dict(
                getattr(self, "_nosl_is_az_replay_prepare_stats", {})
            )
        replay_enabled = bool(getattr(self, "nosl_is_az_replay_enabled", True))
        replay = (
            self._ensure_nosl_is_az_replay() if replay_enabled else None
        )
        train_count = self._nosl_is_az_replay_train_count()
        requested_batch_size = int(self.nosl_is_az_batch_size)
        batch_fill_ratio = (
            float(train_count / requested_batch_size)
            if requested_batch_size > 0 else 0.0
        )
        if train_count < 1:
            if not step_optimizer:
                self.policy.optimizer.zero_grad()
            self._nosl_is_az_metric_sums = {}
            self._nosl_is_az_metric_sumsq = {}
            self._nosl_is_az_metric_counts = {}
            self._nosl_is_az_metric_count = 0.0
            self._nosl_is_az_policy_loss_sum = 0.0
            self._nosl_is_az_policy_weight_sum = 0.0
            self._nosl_is_az_value_loss_sum = 0.0
            return {
                "roots": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "updates": 0.0,
                "value_only_updates": 0.0,
                "joint_updates": 0.0,
                "gradient_batches": 0.0,
                "replay_sampled_roots": 0.0,
                "replay_old_fraction": 0.0,
                "replay_current_fraction": 0.0,
                "replay_max_old_fraction": float(getattr(
                    self, "nosl_is_az_replay_max_old_fraction", 1.0
                )),
                "replay_enabled": float(replay_enabled),
                "batch_size_requested": float(requested_batch_size),
                "batch_size_used": 0.0,
                "batch_fill_ratio": 0.0,
                "metric_count": 0.0,
                "gradient_norm_preclip": 0.0,
                **{
                    key: value
                    for key, value in prepare_stats.items()
                    if key != "fresh_start_id"
                },
            }

        core = getattr(self.policy, "core", None)
        if core is None or not hasattr(core, "actor_critic"):
            raise TypeError("NOSL-IS-AZ requires policy core.actor_critic")
        actor_critic = getattr(core, "combat_actor_critic", core.actor_critic)
        micro_batch_size = int(
            getattr(self, "nosl_is_az_micro_batch_size", 2048)
        )
        fresh_start_id = int(
            prepare_stats.get(
                "fresh_start_id", replay.total_added if replay is not None else 0
            )
        )
        fresh_samples = list(
            getattr(self, "_nosl_is_az_fresh_samples", ())
        ) if not replay_enabled else None
        policy_loss_sum = 0.0
        sampled_policy_weight = 0.0
        value_loss_sum = 0.0
        sampled_roots = 0
        sampled_old = 0
        gradient_batches = 0
        optimizer_steps = 0
        gradient_norms = []

        monitor_names = (
            "target_entropy",
            "policy_entropy",
            "target_policy_kl",
            "policy_cross_entropy",
            "target_effective_actions",
            "target_top1_agreement",
            "legal_actions",
            "illegal_target_mass",
            "target_sum_error",
            "value_target",
            "pred_value",
            "value_mae",
            "value_bias",
            "finite_fraction",
        )
        if getattr(self,'_combat_distribution',False):
            from rl.algorithms.combat_distribution import metrics_for
            monitor_names += metrics_for(self.nosl_supervision_contract)
        monitor_sum = {name: 0.0 for name in monitor_names}
        monitor_sumsq = {name: 0.0 for name in monitor_names}
        monitor_min = {name: float("inf") for name in monitor_names}
        monitor_max = {name: float("-inf") for name in monitor_names}
        monitor_count = {name: 0 for name in monitor_names}

        def accumulate_monitor(name: str, array: np.ndarray) -> None:
            finite = np.isfinite(array)
            if not np.any(finite):
                return
            array = array[finite].astype(np.float64, copy=False)
            monitor_sum[name] += float(array.sum(dtype=np.float64))
            monitor_sumsq[name] += float(np.square(array).sum(dtype=np.float64))
            monitor_min[name] = min(monitor_min[name], float(array.min()))
            monitor_max[name] = max(monitor_max[name], float(array.max()))
            monitor_count[name] += int(array.size)

        host_transfer_sources = []
        is_cuda = th.device(self.device).type == "cuda"

        def transfer_array(array: np.ndarray, dtype=None) -> th.Tensor:
            source = th.from_numpy(array)
            if is_cuda:
                # Stage compact storage in pinned memory. Pageable .to() can
                # block behind the previous microbatch on the CUDA stream,
                # even with non_blocking=True. Cast on the GPU after the copy
                # so float16/uint replay fields do not expand on the host.
                pinned = th.empty(source.shape, dtype=source.dtype, pin_memory=True)
                pinned.copy_(source)
                host_transfer_sources.append(pinned)
                source = pinned
            return source.to(device=self.device, non_blocking=True).to(dtype=dtype)

        def transfer_observations(observation: dict) -> dict:
            if not is_cuda:
                return _leaf_observation_as_tensor({
                    key: np.asarray(value, dtype=self.observation_space[key].dtype)
                    for key, value in observation.items()
                }, self.device)
            groups = {}
            for key, value in observation.items():
                array = np.ascontiguousarray(value)
                groups.setdefault(array.dtype, []).append((key, array))
            result = {}
            for fields in groups.values():
                flats = [array.reshape(-1) for _, array in fields]
                packed = np.concatenate(flats) if len(flats) > 1 else flats[0]
                tensor = transfer_array(packed)
                offset = 0
                for key, array in fields:
                    dtype = th.from_numpy(np.empty(0, dtype=self.observation_space[key].dtype)).dtype
                    result[key] = tensor[offset:offset + array.size].reshape(array.shape).to(dtype=dtype)
                    offset += array.size
            return result

        micro_batches_per_update = (train_count + micro_batch_size - 1) // micro_batch_size
        diagnostics_size = len(monitor_names) * train_count + 3 * micro_batches_per_update
        diagnostics_device = None

        for _update in range(update_count):
            diagnostic_rng = (
                np.random.get_state() if replay_enabled and not step_optimizer
                else None
            )
            indices = replay.sample_indices(
                train_count,
                fresh_start_id=fresh_start_id,
                max_old_fraction=float(getattr(
                    self, "nosl_is_az_replay_max_old_fraction", 1.0
                )),
            ) if replay_enabled else None
            if replay_enabled:
                assert indices is not None and replay is not None
                # Materialize the sampled effective batch once.  Slicing this
                # batch below keeps each micro-batch as a view and avoids a
                # replay-buffer advanced-index copy for every forward pass.
                sampled_batch = replay.get_batch(
                    indices, workers=4 if is_cuda and train_count >= 32768 else 1
                )
            else:
                assert fresh_samples is not None
                sampled_batch = self._nosl_samples_to_batch(
                    fresh_samples[:train_count]
                )
            if not step_optimizer:
                # Retain the exact ordered batch only until the distributed
                # gradient check.  On a failure it can be saved for replay;
                # on success the caller releases it before checkpointing.
                self._nosl_is_az_diagnostic_batch = sampled_batch
                self._nosl_is_az_diagnostic_rng = diagnostic_rng
                self._nosl_is_az_diagnostic_fresh_start_id = fresh_start_id
            self.policy.optimizer.zero_grad()
            diagnostics_offset = 0
            for start in range(0, train_count, micro_batch_size):
                end = min(start + micro_batch_size, train_count)
                micro_count = end - start
                batch = {
                    "observations": {
                        key: value[start:end]
                        for key, value in sampled_batch["observations"].items()
                    },
                    "action_masks": sampled_batch["action_masks"][start:end],
                    "action_groups": sampled_batch["action_groups"][start:end],
                    "policy_targets": sampled_batch["policy_targets"][start:end],
                    "policy_weights": sampled_batch["policy_weights"][start:end],
                    "value_targets": sampled_batch["value_targets"][start:end],
                    "sample_ids": sampled_batch["sample_ids"][start:end],
                }
                if getattr(self,'_combat_distribution',False):
                    batch['combat_targets'] = sampled_batch['combat_targets'][start:end]
                host_observations = batch["observations"]
                observations = transfer_observations(host_observations)
                actor_kwargs = {}
                if "relics" in host_observations and hasattr(core, "combat_actor_critic"):
                    # Match the encoder's ids.long().ne(0) prefix for this
                    # exact microbatch, without CUDA nonzero/item syncs.
                    relics = host_observations["relics"].astype(np.int64, copy=False)
                    active = np.flatnonzero(np.any(relics != 0, axis=0))
                    actor_kwargs["active_relic_width"] = (
                        int(active[-1]) + 1 if active.size else 1
                    )
                masks = transfer_array(
                    batch["action_masks"], dtype=th.bool
                )
                targets = transfer_array(
                    batch["policy_targets"],
                    dtype=th.float32,
                )
                value_targets = transfer_array(
                    batch["value_targets"],
                    dtype=th.float32,
                )
                policy_weights = transfer_array(
                    batch["policy_weights"],
                    dtype=th.float32,
                ).clamp_min(0.0)
                with self.autocast_context():
                    distribution_metrics = {}
                    if getattr(self,'_combat_distribution',False):
                        combat_targets = transfer_array(batch['combat_targets'],dtype=th.float32)
                        logits, values, distribution_outputs = core.combat_training_outputs(
                            observations,combat_targets[:,1],**actor_kwargs)
                    else:
                        logits, values = actor_critic(observations, **actor_kwargs)
                    illegal_target_mass = (targets * (~masks).to(targets.dtype)).sum(dim=-1)
                    from rl.algorithms.action_groups import quotient_distribution
                    groups = transfer_array(batch["action_groups"], dtype=th.long)
                    log_probs, targets, masks = quotient_distribution(logits, targets, masks, groups)
                    target_sum = targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    target_sum_error = (target_sum.squeeze(-1) - 1.0).abs()
                    targets = targets / target_sum
                    target_positive = targets > 0.0
                    target_log = th.where(
                        target_positive,
                        th.log(targets.clamp_min(1e-8)),
                        th.zeros_like(targets),
                    )
                    target_entropy_per_sample = -(targets * target_log).sum(dim=-1)
                    masked_probs = th.where(
                        masks, th.exp(log_probs), th.zeros_like(log_probs)
                    )
                    prob_sum = masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    masked_probs = masked_probs / prob_sum
                    policy_log = th.where(
                        masked_probs > 0.0,
                        th.log(masked_probs.clamp_min(1e-8)),
                        th.zeros_like(masked_probs),
                    )
                    policy_entropy_per_sample = -(
                        masked_probs * policy_log
                    ).sum(dim=-1)
                    policy_cross_entropy_per_sample = -(
                        targets * log_probs
                    ).sum(dim=-1)
                    policy_kl_per_sample = (
                        policy_cross_entropy_per_sample
                        - target_entropy_per_sample
                    )
                    legal_actions = masks.sum(dim=-1).float()
                    target_top1 = targets.argmax(dim=-1)
                    policy_top1 = masked_probs.argmax(dim=-1)
                    target_top1_agreement = (
                        target_top1 == policy_top1
                    ).float()
                    value_loss_per_sample = F.mse_loss(
                        values.float().flatten(), value_targets,
                        reduction="none",
                    )
                    if getattr(self,'_combat_distribution',False):
                        from rl.algorithms.combat_distribution import distribution_loss, mean_outcome_loss, MEAN_CONTRACT
                        if self.nosl_supervision_contract == MEAN_CONTRACT:
                            value_loss_per_sample, distribution_metrics = mean_outcome_loss(
                                distribution_outputs, combat_targets, value_targets)
                        else:
                            value_loss_per_sample, distribution_metrics = distribution_loss(
                                distribution_outputs,combat_targets,value_targets,
                                mean_coef=core.config.mean_loss_coef,
                                auxiliary_coef=core.config.auxiliary_loss_coef)
                    loss_per_sample = self.nosl_is_az_value_coef * value_loss_per_sample
                    if _update >= value_only_updates:
                        loss_per_sample = (loss_per_sample + self.nosl_is_az_policy_coef
                            * policy_cross_entropy_per_sample * policy_weights)

                # Every sample contributes its exact fraction of the effective
                # batch.  A boundary's zero policy weight removes only its
                # policy gradient while retaining its value supervision.
                backward_loss = (
                    loss_per_sample.sum() / float(train_count)
                )
                if not step_optimizer:
                    backward_loss = backward_loss * float(gradient_scale)
                backward_loss.backward()
                loss_sums = (
                    (policy_cross_entropy_per_sample * policy_weights).sum(),
                    policy_weights.sum(),
                    value_loss_per_sample.sum(),
                )
                sampled_roots += micro_count
                if replay_enabled:
                    sampled_old += int(
                        np.count_nonzero(batch["sample_ids"] < fresh_start_id)
                    )
                gradient_batches += 1

                pred_values = values.float().flatten()
                target_values = value_targets.float().flatten()
                if getattr(self, "_combat_distribution", False) and "boss_win_weight" in distribution_outputs:
                    from env.core.boss_value import win_hp_utility
                    target_values = win_hp_utility(combat_targets[:, 0].float(), target_values,
                        distribution_outputs["boss_win_weight"].float().flatten())
                monitor_values = {
                    "target_entropy": target_entropy_per_sample,
                    "policy_entropy": policy_entropy_per_sample,
                    "target_policy_kl": policy_kl_per_sample,
                    "policy_cross_entropy": policy_cross_entropy_per_sample,
                    "target_effective_actions": th.exp(
                        target_entropy_per_sample
                    ),
                    "target_top1_agreement": target_top1_agreement,
                    "legal_actions": legal_actions,
                    "illegal_target_mass": illegal_target_mass,
                    "target_sum_error": target_sum_error,
                    "value_target": target_values,
                    "pred_value": pred_values,
                    "value_mae": (pred_values - target_values).abs(),
                    "value_bias": pred_values - target_values,
                    "finite_fraction": (
                        th.isfinite(logits).all(dim=-1)
                        & th.isfinite(pred_values)
                    ).float(),
                    **distribution_metrics,
                }
                # Preserve the original reductions and detached metric values,
                # but avoid a host synchronization after each microbatch.
                if diagnostics_device is None:
                    diagnostics_dtype = th.float32
                    for value in loss_sums:
                        diagnostics_dtype = th.promote_types(diagnostics_dtype, value.dtype)
                    diagnostics_device = th.empty(
                        diagnostics_size, dtype=diagnostics_dtype, device=self.device
                    )
                diagnostics_end = diagnostics_offset + 3 + len(monitor_names) * micro_count
                th.cat([
                    *(value.detach().reshape(-1) for value in loss_sums),
                    *(value.detach().float().reshape(-1)
                      for value in monitor_values.values()),
                ], out=diagnostics_device[diagnostics_offset:diagnostics_end])
                diagnostics_offset = diagnostics_end

            # This blocking D2H completes all preceding H2D/compute work. Host
            # transfer storage can now be released, while CPU aggregation keeps
            # the original microbatch order and float64 reduction boundaries.
            diagnostics_host = diagnostics_device.cpu().numpy()
            host_transfer_sources.clear()
            diagnostics_offset = 0
            for start in range(0, train_count, micro_batch_size):
                end = min(start + micro_batch_size, train_count)
                micro_count = end - start
                diagnostics_end = diagnostics_offset + 3 + len(monitor_names) * micro_count
                diagnostics = diagnostics_host[diagnostics_offset:diagnostics_end]
                policy_loss_sum += float(diagnostics[0])
                sampled_policy_weight += float(diagnostics[1])
                value_loss_sum += float(diagnostics[2])
                metric_rows = diagnostics[3:].reshape(len(monitor_names), micro_count)
                policy_rows = np.asarray(
                    sampled_batch["policy_weights"][start:end], dtype=np.float32
                ) > 0
                for name, array in zip(monitor_names, metric_rows, strict=True):
                    if name in monitor_names[:9]:
                        array = array[policy_rows]
                    accumulate_monitor(name, array)
                diagnostics_offset = diagnostics_end
            if step_optimizer:
                from rl.algorithms.updates import optimizer_step
                gradient_norms.append(float(optimizer_step(self, value_only=_update < value_only_updates)))
                optimizer_steps += 1

        if not replay_enabled and consume_fresh_samples:
            # Fresh roots are deliberately one-shot.  This also bounds memory
            # when a fight spans many collection updates.  Distributed
            # multi-update training delays this until all synchronized steps
            # have consumed the same collection batch.
            self._nosl_is_az_fresh_samples = []

        def metric_mean(name: str) -> float:
            return (
                float(monitor_sum[name] / monitor_count[name])
                if monitor_count[name] else 0.0
            )

        def metric_std(name: str) -> float:
            if not monitor_count[name]:
                return 0.0
            mean = monitor_sum[name] / monitor_count[name]
            return float(
                np.sqrt(max(
                    0.0,
                    monitor_sumsq[name] / monitor_count[name] - mean * mean,
                ))
            )

        def metric_minimum(name: str) -> float:
            return 0.0 if not np.isfinite(monitor_min[name]) else float(monitor_min[name])

        def metric_maximum(name: str) -> float:
            return 0.0 if not np.isfinite(monitor_max[name]) else float(monitor_max[name])

        rollout_hp_order = getattr(
            self,
            "_nosl_is_az_rollout_hp_order",
            {"count": 0.0, "matches": 0.0, "fraction": 0.0},
        )
        result = {
            "roots": float(sampled_roots),
            "policy_loss": float(
                policy_loss_sum / max(1.0, sampled_policy_weight)
            ),
            "value_loss": float(value_loss_sum / sampled_roots),
            **{name:metric_mean(name) for name in monitor_names[14:]},
            "updates": float(optimizer_steps),
            "value_only_updates": float(value_only_updates if step_optimizer else 0),
            "joint_updates": float(optimizer_steps - value_only_updates if step_optimizer else 0),
            "gradient_batches": float(gradient_batches),
            "replay_sampled_roots": float(sampled_roots if replay_enabled else 0.0),
            "replay_old_fraction": float(sampled_old / sampled_roots),
            "replay_current_fraction": float(
                (sampled_roots - sampled_old) / sampled_roots
            ),
            "replay_max_old_fraction": float(getattr(
                self, "nosl_is_az_replay_max_old_fraction", 1.0
            )),
            "replay_enabled": float(replay_enabled),
            "batch_size_requested": float(requested_batch_size),
            "batch_size_used": float(train_count),
            "batch_fill_ratio": float(batch_fill_ratio),
            "metric_count": float(monitor_count["value_target"]),
            "target_entropy": metric_mean("target_entropy"),
            "policy_entropy": metric_mean("policy_entropy"),
            "target_policy_kl": metric_mean("target_policy_kl"),
            "policy_cross_entropy": metric_mean("policy_cross_entropy"),
            "target_effective_actions": metric_mean("target_effective_actions"),
            "target_top1_agreement": metric_mean("target_top1_agreement"),
            "legal_actions_mean": metric_mean("legal_actions"),
            "illegal_target_mass": metric_mean("illegal_target_mass"),
            "target_sum_error": metric_mean("target_sum_error"),
            "value_target_mean": metric_mean("value_target"),
            "value_target_std": metric_std("value_target"),
            "value_target_min": metric_minimum("value_target"),
            "value_target_max": metric_maximum("value_target"),
            "pred_value_mean": metric_mean("pred_value"),
            "pred_value_std": metric_std("pred_value"),
            "value_mae": metric_mean("value_mae"),
            "value_bias": metric_mean("value_bias"),
            "finite_fraction": metric_mean("finite_fraction"),
            "rollout_value_gt_current_hp_ge_future_hp_fraction": float(
                rollout_hp_order.get("fraction", 0.0)
            ),
            "rollout_value_gt_current_hp_ge_future_hp_count": float(
                rollout_hp_order.get("count", 0.0)
            ),
            "rollout_value_gt_current_hp_ge_future_hp_matches": float(
                rollout_hp_order.get("matches", 0.0)
            ),
            "gradient_norm_preclip": float(
                np.mean(gradient_norms) if gradient_norms else 0.0
            ),
        }
        result.update(
            {
                key: value
                for key, value in prepare_stats.items()
                if key != "fresh_start_id"
            }
        )
        # Keep exact sums on the runtime object for the distributed caller.
        # They must not be returned in the logger-facing dict: TensorBoard and
        # W&B expect scalar values and reject nested dictionaries.
        self._nosl_is_az_metric_sums = monitor_sum
        self._nosl_is_az_metric_sumsq = monitor_sumsq
        self._nosl_is_az_metric_counts = monitor_count
        self._nosl_is_az_metric_count = float(monitor_count["value_target"])
        self._nosl_is_az_policy_loss_sum = float(policy_loss_sum)
        self._nosl_is_az_policy_weight_sum = float(sampled_policy_weight)
        self._nosl_is_az_value_loss_sum = float(value_loss_sum)
        return result

    def _optimizer_step(self, micro_batches: int) -> float:
        return float(optimizer_step(self, micro_batches))

    def _train_reasoning(self) -> dict[str, float]:
        self._wait_for_reasoning_samples()
        samples = self._reasoning_rollout_samples
        core = getattr(self.policy, "core", None)
        if (
            not samples
            or self.reasoning_coef <= 0.0
            or self.reasoning_updates <= 0
            or not hasattr(core, "reasoning_loss")
        ):
            return {}

        # All captured transcripts participate.  The decoder can extend its
        # positional representation beyond the initial learned table, so a
        # long trace is no longer silently removed from the auxiliary loss.
        eligible = list(samples)
        dropped = 0
        if not eligible:
            return {
                "samples": float(len(samples)),
                "eligible_samples": 0.0,
                "dropped_too_long": float(dropped),
                "sequence_length_mean": 0.0,
                "sequence_length_max": 0.0,
            }

        losses = []
        accuracies = []
        supervised = 0.0
        sampled_samples = 0
        optimizer_steps = 0
        accumulation_steps = self.reasoning_gradient_accumulation_steps
        self.policy.optimizer.zero_grad()
        for reasoning_update in range(self.reasoning_updates):
            stats = reasoning_microbatch_backward(
                self, self.reasoning_batch_size, synchronize=False)
            sampled_samples += int(stats["sampled_samples"])
            group_finished = (
                (reasoning_update + 1) % accumulation_steps == 0
                or reasoning_update + 1 >= self.reasoning_updates
            )
            if group_finished:
                group_size = min(
                    accumulation_steps,
                    reasoning_update + 1
                    - (reasoning_update // accumulation_steps) * accumulation_steps,
                )
                self._optimizer_step(group_size)
                optimizer_steps += 1
            losses.append(stats["loss"])
            accuracies.append(stats["token_accuracy"])
            supervised += stats["supervised_tokens"]

        atomic = sum(sample["precision"] == "atomic" for sample in eligible)
        environment_tokens = sum(sample["environment_tokens"] for sample in eligible)
        lengths = [len(sample["inputs"]) for sample in eligible]
        return {
            "loss": float(np.mean(losses)),
            "token_accuracy": float(np.mean(accuracies)),
            "supervised_tokens": supervised,
            "sampled_samples": float(sampled_samples),
            "samples": float(len(samples)),
            "eligible_samples": float(len(eligible)),
            "dropped_too_long": float(dropped),
            "atomic_trace_rate": atomic / len(eligible),
            "environment_tokens": float(environment_tokens),
            "optimizer_steps": float(optimizer_steps),
            "gradient_accumulation_steps": float(accumulation_steps),
            "sequence_length_mean": float(np.mean(lengths)),
            "sequence_length_max": float(max(lengths)),
        }

    def train(self) -> None:
        """Update the policy, accumulating gradients across micro-batches."""
        if self.performance_profile:
            self._sync_device()
            train_started = time.perf_counter()
            if self.device.type == "cuda":
                th.cuda.reset_peak_memory_stats(self.device)
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        combat_value_losses, overworld_value_losses = [], []
        clip_fractions = []
        valid_samples = combat_samples = overworld_samples = 0
        continue_training = True
        optimizer_steps = 0
        loss = th.zeros((), device=self.device)
        approx_kl_divs = []

        inner_only = bool(getattr(self, "nosl_is_az_inner_only", False))
        ppo_epochs = 0 if inner_only else self.n_epochs
        for epoch in range(ppo_epochs):
            approx_kl_divs = []
            accumulated = 0
            self.policy.optimizer.zero_grad()

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                components, loss = ppo_forward(
                    self, rollout_data, clip_range,
                    clip_range_vf if self.clip_range_vf is not None else None,
                )
                policy_loss = components.policy_loss
                value_loss = components.value_loss
                entropy_loss = components.entropy_loss
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(components.clip_fraction.item())
                valid_samples += components.valid_samples
                combat_samples += components.combat_samples
                overworld_samples += components.overworld_samples
                if components.combat_value_loss is not None:
                    combat_value_losses.append(
                        components.combat_value_loss.item()
                    )
                if components.overworld_value_loss is not None:
                    overworld_value_losses.append(
                        components.overworld_value_loss.item()
                    )

                approx_kl_div = float(components.approx_kl.detach().cpu())
                approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max kl: {approx_kl_div:.2f}"
                        )
                    break

                loss.backward()
                accumulated += 1
                if accumulated == self.gradient_accumulation_steps:
                    self._optimizer_step(accumulated)
                    optimizer_steps += 1
                    accumulated = 0

            # Preserve updates from micro-batches preceding a target-KL stop,
            # and support callers whose rollout does not form a complete group.
            if accumulated:
                self._optimizer_step(accumulated)
                optimizer_steps += 1

            self._n_updates += 1
            if not continue_training:
                break

        reasoning_stats = {} if inner_only else self._train_reasoning()
        nosl_is_az_stats = (
            self._train_nosl_is_az()
            if self.nosl_is_az and getattr(
                self, "nosl_is_az_combat_strategy", "model"
            ) == "model"
            else {}
        )
        if joint_upper(self):
            if hasattr(self.policy, 'small_playout'):
                from rl.algorithms.small_playout import update_online
                update_online(self)
            update_upper(self)

        flat_values = self.rollout_buffer.values.flatten()
        flat_returns = self.rollout_buffer.returns.flatten()
        valid_returns = np.isfinite(flat_returns)
        explained_var = explained_variance(
            flat_values[valid_returns],
            flat_returns[valid_returns],
        )

        mean_or_nan = lambda values: np.mean(values) if values else np.nan
        self.logger.record("train/entropy_loss", mean_or_nan(entropy_losses))
        self.logger.record("train/policy_gradient_loss", mean_or_nan(pg_losses))
        self.logger.record("train/value_loss", mean_or_nan(value_losses))
        self.logger.record("train/approx_kl", mean_or_nan(approx_kl_divs))
        self.logger.record("train/clip_fraction", mean_or_nan(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record(
            "train/gradient_accumulation_steps",
            self.gradient_accumulation_steps,
        )
        self.logger.record(
            "train/effective_batch_size",
            self.batch_size * self.gradient_accumulation_steps,
        )
        self.logger.record("train/optimizer_steps", optimizer_steps)
        self.logger.record("train/outer_updates_enabled", float(not inner_only))
        for key, value in reasoning_stats.items():
            self.logger.record(f"reasoning/{key}", value)
        for key, value in nosl_is_az_stats.items():
            self.logger.record(f"nosl_is_az/{key}", value)
        if self.hierarchical_combat_hp:
            flat_phases = np.asarray(
                self.rollout_buffer.observations["phase"]
            )[..., 0].flatten()
            combat_mask = flat_phases == V.PHASE_VOCAB_INDEX["combat"]
            combat_returns = combat_mask & valid_returns
            overworld_returns = ~combat_mask & valid_returns
            self.logger.record("train/valid_samples", valid_samples)
            self.logger.record("train/combat_samples", combat_samples)
            self.logger.record("train/overworld_samples", overworld_samples)
            self.logger.record(
                "train/combat_value_loss",
                np.mean(combat_value_losses) if combat_value_losses else np.nan,
            )
            self.logger.record(
                "train/overworld_value_loss",
                np.mean(overworld_value_losses)
                if overworld_value_losses else np.nan,
            )
            self.logger.record(
                "train/combat_explained_variance",
                explained_variance(
                    flat_values[combat_returns], flat_returns[combat_returns]
                ) if combat_returns.any() else np.nan,
            )
            self.logger.record(
                "train/overworld_explained_variance",
                explained_variance(
                    flat_values[overworld_returns],
                    flat_returns[overworld_returns],
                ) if overworld_returns.any() else np.nan,
            )
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        if self.performance_profile:
            self._sync_device()
            train_seconds = time.perf_counter() - train_started
            rollout = self._performance_rollout
            self.logger.record("perf/train_update_seconds", train_seconds)
            if self.device.type == "cuda":
                self.logger.record(
                    "perf/gpu_peak_memory_mb",
                    th.cuda.max_memory_allocated(self.device) / (1024 ** 2),
                )
            if rollout is not None:
                total = max(rollout["total"], 1e-12)
                transitions = rollout["transitions"]
                for key in ("policy", "env", "mask", "overhead", "total"):
                    self.logger.record(f"perf/rollout_{key}_seconds", rollout[key])
                    self.logger.record(
                        f"perf/rollout_{key}_ms_per_transition",
                        rollout[key] * 1000.0 / transitions,
                    )
                self.logger.record(
                    "perf/rollout_policy_fraction", rollout["policy"] / total
                )
                self.logger.record(
                    "perf/rollout_env_fraction", rollout["env"] / total
                )
                self.logger.record(
                    "perf/rollout_mask_fraction", rollout["mask"] / total
                )
                bottleneck = max(
                    ("gpu-policy", rollout["policy"]),
                    ("cpu-env", rollout["env"]),
                    ("cpu-mask-ipc", rollout["mask"]),
                    ("python-overhead", rollout["overhead"]),
                    key=lambda item: item[1],
                )[0]
                print(
                    "[perf] "
                    f"rollout={rollout['total']:.2f}s "
                    f"gpu_policy={rollout['policy']:.2f}s "
                    f"cpu_env={rollout['env']:.2f}s "
                    f"mask_ipc={rollout['mask']:.2f}s "
                    f"overhead={rollout['overhead']:.2f}s "
                    f"update={train_seconds:.2f}s bottleneck={bottleneck}",
                    flush=True,
                )
