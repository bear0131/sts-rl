"""MaskablePPO with optimizer-level gradient accumulation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MethodType

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

from env.core import vocab as V
from rl.hierarchical_returns import compute_hierarchical_returns


@dataclass(frozen=True)
class PPOLossComponents:
    policy_loss: th.Tensor
    value_loss: th.Tensor
    entropy_loss: th.Tensor
    approx_kl: th.Tensor
    clip_fraction: th.Tensor
    valid_samples: int
    combat_samples: int
    overworld_samples: int
    combat_value_loss: th.Tensor | None
    overworld_value_loss: th.Tensor | None


def _phase_balanced_mean(
    values: th.Tensor,
    *,
    valid: th.Tensor,
    combat: th.Tensor,
) -> th.Tensor:
    means = []
    for group in (valid & combat, valid & ~combat):
        if bool(group.any()):
            means.append(values[group].mean())
    if not means:
        raise RuntimeError("PPO minibatch has no valid hierarchical samples")
    return th.stack(means).mean()


def ppo_loss_components(
    *,
    hierarchical: bool,
    normalize_advantage: bool,
    observations,
    advantages: th.Tensor,
    returns: th.Tensor,
    old_values: th.Tensor,
    old_log_prob: th.Tensor,
    values: th.Tensor,
    log_prob: th.Tensor,
    entropy: th.Tensor | None,
    clip_range: float,
    clip_range_vf: float | None,
) -> PPOLossComponents:
    """Compute the shared native/distributed PPO objective.

    Hierarchical rollouts use NaN to mark an overworld macro transition whose
    combat option did not finish before the rollout boundary. Those samples are
    excluded. Combat and overworld means are then weighted equally so the much
    larger number of combat clicks cannot drown out strategic decisions.
    """
    if clip_range_vf is None:
        values_pred = values
    else:
        values_pred = old_values + th.clamp(
            values - old_values,
            -clip_range_vf,
            clip_range_vf,
        )

    ratio = th.exp(log_prob - old_log_prob)
    clipped_ratio = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    entropy_per_sample = log_prob if entropy is None else -entropy
    kl_per_sample = (th.exp(log_prob - old_log_prob) - 1) - (
        log_prob - old_log_prob
    )
    clipped_per_sample = (th.abs(ratio - 1) > clip_range).float()

    if not hierarchical:
        normalized_advantages = advantages
        if normalize_advantage:
            normalized_advantages = (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-8)
        policy_loss = -th.min(
            normalized_advantages * ratio,
            normalized_advantages * clipped_ratio,
        ).mean()
        value_loss = F.mse_loss(returns, values_pred)
        return PPOLossComponents(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_per_sample.mean(),
            approx_kl=kl_per_sample.mean(),
            clip_fraction=clipped_per_sample.mean(),
            valid_samples=int(advantages.numel()),
            combat_samples=0,
            overworld_samples=0,
            combat_value_loss=None,
            overworld_value_loss=None,
        )

    phase = observations["phase"].long().reshape(-1)
    combat = phase == V.PHASE_VOCAB_INDEX["combat"]
    valid = th.isfinite(advantages) & th.isfinite(returns)
    normalized_advantages = th.where(valid, advantages, th.zeros_like(advantages))
    if normalize_advantage:
        normalized_advantages = normalized_advantages.clone()
        for group in (valid & combat, valid & ~combat):
            count = int(group.sum().item())
            if count > 1:
                selected = normalized_advantages[group]
                normalized_advantages[group] = (
                    selected - selected.mean()
                ) / (selected.std(unbiased=False) + 1e-8)

    policy_per_sample = -th.min(
        normalized_advantages * ratio,
        normalized_advantages * clipped_ratio,
    )
    safe_returns = th.where(valid, returns, values_pred.detach())
    value_per_sample = (safe_returns - values_pred).square()
    policy_loss = _phase_balanced_mean(
        policy_per_sample, valid=valid, combat=combat
    )
    value_loss = _phase_balanced_mean(
        value_per_sample, valid=valid, combat=combat
    )
    entropy_loss = _phase_balanced_mean(
        entropy_per_sample, valid=valid, combat=combat
    )

    combat_valid = valid & combat
    overworld_valid = valid & ~combat
    return PPOLossComponents(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        approx_kl=kl_per_sample[valid].mean(),
        clip_fraction=clipped_per_sample[valid].mean(),
        valid_samples=int(valid.sum().item()),
        combat_samples=int(combat_valid.sum().item()),
        overworld_samples=int(overworld_valid.sum().item()),
        combat_value_loss=(
            value_per_sample[combat_valid].mean()
            if bool(combat_valid.any()) else None
        ),
        overworld_value_loss=(
            value_per_sample[overworld_valid].mean()
            if bool(overworld_valid.any()) else None
        ),
    )


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
        hierarchical_combat_hp: bool = False,
        combat_hp_gamma: float = 1.0,
        combat_hp_gae_lambda: float = 0.95,
        **kwargs,
    ) -> None:
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if not 0.0 <= combat_hp_gamma <= 1.0:
            raise ValueError("combat_hp_gamma must be in [0, 1]")
        if not 0.0 <= combat_hp_gae_lambda <= 1.0:
            raise ValueError("combat_hp_gae_lambda must be in [0, 1]")
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.performance_profile = bool(performance_profile)
        self.hierarchical_combat_hp = bool(hierarchical_combat_hp)
        self.combat_hp_gamma = float(combat_hp_gamma)
        self.combat_hp_gae_lambda = float(combat_hp_gae_lambda)
        self._performance_rollout = None
        self._hierarchical_rollout_stats = None
        super().__init__(*args, **kwargs)

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
        if not 0.0 <= combat_gamma <= 1.0:
            raise ValueError("combat_gamma must be in [0, 1]")
        if not 0.0 <= combat_gae_lambda <= 1.0:
            raise ValueError("combat_gae_lambda must be in [0, 1]")
        self.hierarchical_combat_hp = bool(enabled)
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
        rewards, returns, advantages, stats = compute_hierarchical_returns(
            phases=phases,
            values=rollout_buffer.values,
            raw_rewards=raw_rewards,
            dones=dones,
            next_phases=next_phases,
            next_hp=next_hp,
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
        captured = []
        uncaptured_step_wait = env.step_wait

        if self.hierarchical_combat_hp:
            def capture_step_wait(_env):
                new_obs, rewards, dones, infos = uncaptured_step_wait()
                phases, hp, interrupted = self._next_phase_hp(
                    new_obs, dones, infos
                )
                captured.append((
                    np.asarray(rewards, dtype=np.float32).copy(),
                    np.asarray(dones, dtype=bool).copy(),
                    phases,
                    hp,
                    interrupted,
                ))
                return new_obs, rewards, dones, infos

            env.step_wait = MethodType(capture_step_wait, env)

        try:
            if not self.performance_profile:
                result = super().collect_rollouts(
                    env, callback, rollout_buffer, n_rollout_steps, use_masking
                )
            else:
                result = self._collect_profiled_rollouts(
                    env, callback, rollout_buffer, n_rollout_steps, use_masking
                )
        finally:
            if self.hierarchical_combat_hp:
                env.step_wait = uncaptured_step_wait

        if result and self.hierarchical_combat_hp:
            self._apply_hierarchical_returns(rollout_buffer, captured)
        return result

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

    def _optimizer_step(self, micro_batches: int) -> None:
        """Average accumulated gradients, clip once, and update parameters."""
        if micro_batches < 1:
            return
        scale = 1.0 / micro_batches
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        self.policy.optimizer.zero_grad()

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

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            accumulated = 0
            self.policy.optimizer.zero_grad()

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()

                components = ppo_loss_components(
                    hierarchical=self.hierarchical_combat_hp,
                    normalize_advantage=self.normalize_advantage,
                    observations=rollout_data.observations,
                    advantages=rollout_data.advantages,
                    returns=rollout_data.returns,
                    old_values=rollout_data.old_values,
                    old_log_prob=rollout_data.old_log_prob,
                    values=values,
                    log_prob=log_prob,
                    entropy=entropy,
                    clip_range=clip_range,
                    clip_range_vf=(
                        clip_range_vf
                        if self.clip_range_vf is not None else None
                    ),
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

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
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

        flat_values = self.rollout_buffer.values.flatten()
        flat_returns = self.rollout_buffer.returns.flatten()
        valid_returns = np.isfinite(flat_returns)
        explained_var = explained_variance(
            flat_values[valid_returns],
            flat_returns[valid_returns],
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
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
