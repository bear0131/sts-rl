"""STS2 collection and recovery, using the repository's shared PPO update primitives."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from env.sts2.game import EnvConfig, Sts2Env
from rl.algorithms.ppo import ppo_loss_components
from rl.algorithms.updates import optimizer_step
from rl.policies.sts2 import CandidatePolicy, batch_observations, compact


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    num_envs: int = 4
    rollout_steps: int = 64
    epochs: int = 2
    minibatch: int = 64
    hidden: int = 128
    lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    threads: int = 1
    device: str = "cpu"

    def __post_init__(self):
        if min(self.num_envs, self.rollout_steps, self.epochs, self.hidden, self.threads) < 1 or self.minibatch < 2:
            raise ValueError("Positive budgets and minibatch >= 2 required")
        if self.num_envs * self.rollout_steps < 2 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid rollout/GAE configuration")
        if not all(math.isfinite(x) and x > 0 for x in (self.lr, self.clip, self.max_grad_norm)):
            raise ValueError("Learning rate, clip and gradient norm must be finite and positive")
        if not all(math.isfinite(x) and x >= 0 for x in (self.entropy_coef, self.value_coef)):
            raise ValueError("Loss coefficients must be finite and nonnegative")


def gae(rewards, values, next_values, terminated, boundaries, gamma, lam):
    """Truncations bootstrap, but never carry advantage into a reset episode."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * ~terminated[t] - values[t]
        last = delta + gamma * lam * last * ~boundaries[t]
        advantages[t] = last
    return advantages, advantages + values


class Trainer:
    def __init__(self, config=TrainConfig(), env_config=EnvConfig()):
        self.config = config
        self.env_config = env_config
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        self.device = torch.device(config.device)
        self.policy = CandidatePolicy(config.hidden).to(self.device)
        self.policy.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.max_grad_norm = config.max_grad_norm
        self.update_rng = np.random.default_rng(config.seed)
        self.envs = [Sts2Env(env_config) for _ in range(config.num_envs)]
        self.next_seed = config.seed
        self.observations = [self._reset(env) for env in self.envs]
        self.iteration = self.transitions = self.optimizer_updates = 0

    def _reset(self, env):
        obs, _ = env.reset(seed=self.next_seed)
        self.next_seed += 1
        return compact(obs)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def iteration_step(self, check=lambda: None):
        cfg = self.config
        started = time.perf_counter()
        process = psutil.Process()
        cpu0 = sum(process.cpu_times()[:2])
        observations, actions, logprobs, values, rewards, next_values, terminals, boundaries = [], [], [], [], [], [], [], []
        outcomes, phases = Counter(), Counter()
        self.policy.eval()
        for _ in range(cfg.rollout_steps):
            check()
            with torch.no_grad():
                distribution, val = self.policy(batch_observations(self.observations, self.device))
                action = distribution.sample()
                lp = distribution.log_prob(action)
            observations.extend(self.observations)
            actions.append(action.cpu().numpy())
            logprobs.append(lp.cpu().numpy())
            values.append(val.cpu().numpy())
            post, rs, ts, bs = [], [], [], []
            for i, env in enumerate(self.envs):
                phases[env.rm.phase()] += 1
                obs, reward, terminated, truncated, info = env.step(int(actions[-1][i]))
                post.append(compact(obs))
                rs.append(reward)
                ts.append(terminated)
                bs.append(terminated or truncated)
                if terminated or truncated:
                    outcomes[info["outcome"]] += 1
                self.observations[i] = self._reset(env) if terminated or truncated else post[-1]
            with torch.no_grad():
                next_values.append(self.policy.values(post).cpu().numpy())
            rewards.append(rs)
            terminals.append(ts)
            boundaries.append(bs)
        self._sync()
        collect_seconds = time.perf_counter()-started
        adv, returns = gae(np.asarray(rewards, np.float32), np.asarray(values), np.asarray(next_values),
                           np.asarray(terminals), np.asarray(boundaries), cfg.gamma, cfg.gae_lambda)
        action = np.asarray(actions).reshape(-1)
        old_lp = np.asarray(logprobs).reshape(-1)
        old_values = np.asarray(values).reshape(-1)
        adv, returns = adv.reshape(-1), returns.reshape(-1)
        n = len(action)
        before = [p.detach().clone() for p in self.policy.parameters()]
        statistics = []
        self.policy.train()
        update_started = time.perf_counter()
        for _ in range(cfg.epochs):
            order = self.update_rng.permutation(n)
            # Avoid singleton minibatches (sample std in shared PPO normalization).
            batches = np.array_split(order, max(1, n // cfg.minibatch))
            for indices in batches:
                check()
                obs = batch_observations([observations[i] for i in indices], self.device)
                distribution, value = self.policy(obs)
                tensor = lambda x: torch.as_tensor(x[indices], device=self.device)
                loss_parts = ppo_loss_components(hierarchical=False, normalize_advantage=True,
                    observations=obs, advantages=tensor(adv), returns=tensor(returns),
                    old_values=tensor(old_values), old_log_prob=tensor(old_lp), values=value,
                    log_prob=distribution.log_prob(tensor(action)), entropy=distribution.entropy(),
                    clip_range=cfg.clip, clip_range_vf=None)
                loss = loss_parts.policy_loss + cfg.value_coef * loss_parts.value_loss + cfg.entropy_coef * loss_parts.entropy_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.policy.parameters()):
                    raise FloatingPointError("Nonfinite PPO gradient")
                norm = optimizer_step(self)
                statistics.append(dict(loss=float(loss.detach()), kl=float(loss_parts.approx_kl.detach()), grad_norm=float(norm)))
                self.optimizer_updates += 1
        self._sync()
        update_seconds = time.perf_counter()-update_started
        delta = sum(float((p.detach()-old).square().sum()) for p, old in zip(self.policy.parameters(), before))**0.5
        self.iteration += 1
        self.transitions += n
        elapsed = time.perf_counter()-started
        cpu_seconds = sum(process.cpu_times()[:2])-cpu0
        cuda = self.device.type == "cuda"
        return dict(iteration=self.iteration, transitions=self.transitions, samples=n,
            optimizer_updates=self.optimizer_updates, outcomes=dict(outcomes), phases=dict(phases),
            loss=float(np.mean([s["loss"] for s in statistics])),
            approx_kl=float(np.mean([s["kl"] for s in statistics])), parameter_delta_l2=delta,
            collect_seconds=collect_seconds, update_seconds=update_seconds, iteration_seconds=elapsed,
            cpu_seconds=cpu_seconds, cpu_util_percent=100*cpu_seconds/elapsed,
            rss_bytes=process.memory_info().rss, system_available_bytes=psutil.virtual_memory().available,
            torch_allocated_bytes=torch.cuda.memory_allocated(self.device) if cuda else None,
            torch_reserved_bytes=torch.cuda.memory_reserved(self.device) if cuda else None,
            torch_peak_bytes=torch.cuda.max_memory_allocated(self.device) if cuda else None,
            device_vram_bytes=None, gpu_util_percent=None,
            unavailable_metrics="device-level GPU sampler not enabled; Torch CUDA metrics null on CPU",
            config=asdict(cfg), contract=self.envs[0].contract)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format="sts2-ppo-v1", config=asdict(self.config), env_config=asdict(self.env_config),
            contract=self.envs[0].contract, policy=self.policy.state_dict(), optimizer=self.policy.optimizer.state_dict(),
            iteration=self.iteration, transitions=self.transitions, optimizer_updates=self.optimizer_updates,
            next_seed=self.next_seed, update_rng=self.update_rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            environments=[env.snapshot() for env in self.envs])
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def restore(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format"] != "sts2-ppo-v1" or payload["config"] != asdict(self.config) or payload["contract"] != self.envs[0].contract:
            raise ValueError("Checkpoint learner/environment/backend contract mismatch")
        self.policy.load_state_dict(payload["policy"])
        self.policy.optimizer.load_state_dict(payload["optimizer"])
        self.observations = [compact(env.restore(state)[0]) for env, state in zip(self.envs, payload["environments"], strict=True)]
        self.iteration, self.transitions, self.optimizer_updates = (payload[k] for k in ("iteration", "transitions", "optimizer_updates"))
        self.next_seed = payload["next_seed"]
        self.update_rng.bit_generator.state = payload["update_rng"]
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])

    def close(self):
        for env in self.envs:
            env.close()
