"""MaskablePPO with optimizer-level gradient accumulation."""

from __future__ import annotations

import time
from types import MethodType

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


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
        **kwargs,
    ) -> None:
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.performance_profile = bool(performance_profile)
        self._performance_rollout = None
        super().__init__(*args, **kwargs)

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            th.cuda.synchronize(self.device)

    def collect_rollouts(self, env, callback, rollout_buffer,
                         n_rollout_steps, use_masking=True):
        if not self.performance_profile:
            return super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps, use_masking
            )

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
        clip_fractions = []
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

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (
                        advantages - advantages.mean()
                    ) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
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

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
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
