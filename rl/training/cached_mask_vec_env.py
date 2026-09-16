"""Subprocess vector environment that reuses masks returned by each step."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_connections
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.utils import EXPECTED_METHOD_NAME
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import (
    CloudpickleWrapper,
    _patch_env,
    _stack_obs,
)


class CachedMaskSubprocVecEnv(SubprocVecEnv):
    """Avoid a second worker round trip for the next action mask.

    Gymnasium step/reset info already contains ``action_mask``. SB3-contrib
    normally asks every subprocess for ``action_masks()`` again before the next
    policy call. Cache the info value while retaining ``env_method`` fallback
    behavior for environments that do not provide it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._remote_indices = {
            remote: index for index, remote in enumerate(self.remotes)
        }
        self._cached_action_masks: list[np.ndarray | None] = [
            None for _ in range(self.num_envs)
        ]

    @staticmethod
    def _info_mask(info: dict[str, Any]) -> np.ndarray | None:
        mask = info.get("action_mask")
        if mask is None:
            return None
        return np.asarray(mask, dtype=bool).copy()

    def reset(self):
        observations = super().reset()
        self._cached_action_masks = [
            self._info_mask(info) for info in self.reset_infos
        ]
        return observations

    def step_wait(self):
        # SB3's SubprocVecEnv receives pipes in fixed index order.  With many
        # environments, one slow simulator can hold up the whole receive loop
        # while other workers block trying to send their (larger) observations
        # and reasoning traces.  Drain whichever pipe is ready first, then
        # restore the original index order before stacking so the public VecEnv
        # contract is unchanged.
        ready = set(self.remotes)
        results: list[Any] = [None] * self.num_envs
        while ready:
            for remote in wait_for_connections(tuple(ready)):
                index = self._remote_indices[remote]
                results[index] = remote.recv()
                ready.remove(remote)
        self.waiting = False
        observations, rewards, dones, infos, self.reset_infos = zip(
            *results, strict=True
        )
        observations, rewards, dones, infos = (
            _stack_obs(observations, self.observation_space),
            np.stack(rewards),
            np.stack(dones),
            infos,
        )
        for index, (done, info) in enumerate(zip(dones, infos, strict=True)):
            source = self.reset_infos[index] if done else info
            self._cached_action_masks[index] = self._info_mask(source)
        return observations, rewards, dones, infos

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        if (
            method_name == EXPECTED_METHOD_NAME
            and not method_args
            and not method_kwargs
        ):
            selected = list(self._get_indices(indices))
            cached = [self._cached_action_masks[index] for index in selected]
            if all(mask is not None for mask in cached):
                return [mask.copy() for mask in cached]

            values = super().env_method(
                method_name,
                indices=selected,
            )
            for index, value in zip(selected, values, strict=True):
                self._cached_action_masks[index] = np.asarray(
                    value, dtype=bool
                ).copy()
            return values

        return super().env_method(
            method_name,
            *method_args,
            indices=indices,
            **method_kwargs,
        )

    def env_method_batch(
        self,
        method_name: str,
        calls: dict[int, tuple[tuple[Any, ...], dict[str, Any]]],
    ):
        """Call one method with distinct arguments on several workers.

        ``SubprocVecEnv.env_method`` broadcasts one argument tuple to all
        selected workers.  Staged NOSL leaf evaluation needs a different value
        and policy-logit response for every worker, so send those requests in
        parallel and restore caller index order on receive.
        """
        if not calls:
            return {}
        pending: dict[Any, int] = {}
        for index, (args, kwargs) in calls.items():
            remote = self.remotes[int(index)]
            remote.send(("env_method", (method_name, args, kwargs)))
            pending[remote] = int(index)
        results: dict[int, Any] = {}
        while pending:
            for remote in wait_for_connections(tuple(pending)):
                index = pending.pop(remote)
                results[index] = remote.recv()
        return results


def _group_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrappers: list[CloudpickleWrapper],
) -> None:
    """Run several independent Gym envs behind one pipe.

    The environments are deliberately kept independent: each owns its own
    simulator state, RNG and NOSL search session.  Commands are batched only
    at the transport boundary; no search tree or hidden state is shared.
    """
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    envs = [_patch_env(wrapper.var()) for wrapper in env_fn_wrappers]
    reset_infos: list[dict[str, Any]] = [{} for _ in envs]
    try:
        while True:
            try:
                cmd, data = remote.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd == "step":
                results = []
                for local_index, (env, action) in enumerate(
                    zip(envs, data, strict=True)
                ):
                    observation, reward, terminated, truncated, info = env.step(
                        action
                    )
                    done = bool(terminated or truncated)
                    info["TimeLimit.truncated"] = bool(
                        truncated and not terminated
                    )
                    if done:
                        info["terminal_observation"] = observation
                        observation, reset_infos[local_index] = env.reset()
                    results.append(
                        (observation, reward, done, info, reset_infos[local_index])
                    )
                remote.send(results)
            elif cmd == "reset":
                results = []
                for env, (seed, options) in zip(envs, data, strict=True):
                    observation, info = env.reset(
                        seed=seed,
                        **({"options": options} if options else {}),
                    )
                    results.append((observation, info))
                reset_infos = [info for _obs, info in results]
                remote.send(results)
            elif cmd == "render":
                remote.send([env.render() for env in envs])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                spaces = [(env.observation_space, env.action_space) for env in envs]
                first = spaces[0]
                if any(pair != first for pair in spaces[1:]):
                    raise ValueError("grouped envs must have identical spaces")
                remote.send(first)
            elif cmd == "env_method":
                local_indices, method_name, args, kwargs = data
                remote.send({
                    int(index): envs[int(index)].get_wrapper_attr(method_name)(
                        *args, **kwargs
                    )
                    for index in local_indices
                })
            elif cmd == "env_method_batch":
                calls = data
                remote.send({
                    int(index): envs[int(index)].get_wrapper_attr(method_name)(
                        *args, **kwargs
                    )
                    for index, method_name, args, kwargs in calls
                })
            elif cmd == "get_attr":
                local_indices, attr_name = data
                remote.send({
                    int(index): envs[int(index)].get_wrapper_attr(attr_name)
                    for index in local_indices
                })
            elif cmd == "has_attr":
                local_indices, attr_name = data
                values = {}
                for index in local_indices:
                    try:
                        envs[int(index)].get_wrapper_attr(attr_name)
                        values[int(index)] = True
                    except AttributeError:
                        values[int(index)] = False
                remote.send(values)
            elif cmd == "set_attr":
                local_indices, attr_name, value = data
                for index in local_indices:
                    setattr(envs[int(index)], attr_name, value)
                remote.send(None)
            elif cmd == "is_wrapped":
                local_indices, wrapper_class = data
                remote.send({
                    int(index): is_wrapped(envs[int(index)], wrapper_class)
                    for index in local_indices
                })
            else:
                raise NotImplementedError(
                    f"`{cmd}` is not implemented in grouped worker"
                )
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


@dataclass(frozen=True)
class GroupedMethodResponse:
    """One completed worker-group command, restored to logical env indices."""

    group_index: int
    method_name: str
    responses: dict[int, Any]
    elapsed_seconds: float = 0.0


class GroupedMethodStream:
    """Keep independent worker groups in flight without a global receive barrier.

    A worker process still executes one command at a time and preserves the
    order of every environment it owns.  The stream only allows the caller to
    receive and resubmit groups independently, which is useful for batching
    neural leaves from unrelated search trees.
    """

    def __init__(self, owner: "GroupedCachedMaskSubprocVecEnv") -> None:
        self._owner = owner
        self._pending: dict[Any, tuple[int, str, float]] = {}

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    def submit(
        self,
        method_name: str,
        calls: dict[int, tuple[tuple[Any, ...], dict[str, Any]]],
    ) -> None:
        """Submit distinct calls, requiring each affected group to be idle."""
        if not calls:
            return
        grouped_calls: dict[
            int, list[tuple[int, str, tuple[Any, ...], dict[str, Any]]]
        ] = defaultdict(list)
        for index, (args, kwargs) in calls.items():
            group_index, local_index = self._owner._env_to_group[int(index)]
            grouped_calls[group_index].append(
                (local_index, method_name, args, kwargs)
            )
        for group_index in grouped_calls:
            remote = self._owner.remotes[group_index]
            if remote in self._pending:
                raise RuntimeError(
                    f"worker group {group_index} already has an in-flight command"
                )
        for group_index, group_calls in grouped_calls.items():
            remote = self._owner.remotes[group_index]
            remote.send(("env_method_batch", group_calls))
            self._pending[remote] = (group_index, method_name, time.perf_counter())

    def receive(
        self,
        timeout: float | None = None,
    ) -> list[GroupedMethodResponse]:
        """Return every group currently ready, or an empty list on timeout."""
        if not self._pending:
            return []
        ready = wait_for_connections(tuple(self._pending), timeout=timeout)
        packets = []
        for remote in ready:
            group_index, method_name, submitted_at = self._pending.pop(remote)
            local_responses = remote.recv()
            group = self._owner._groups[group_index]
            responses = {
                int(group[int(local_index)]): response
                for local_index, response in local_responses.items()
            }
            packets.append(
                GroupedMethodResponse(
                    group_index=group_index,
                    method_name=method_name,
                    responses=responses,
                    elapsed_seconds=time.perf_counter() - submitted_at,
                )
            )
        return packets


class GroupedCachedMaskSubprocVecEnv(VecEnv):
    """Vector env whose subprocesses each own multiple independent envs.

    ``envs_per_worker`` controls the grouping granularity.  A group is stepped
    sequentially inside its process, but one command/response carries all of
    its environments.  The public SB3 VecEnv ordering remains the original
    logical env order, so rollout buffers and action masks are unchanged.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        *,
        envs_per_worker: int = 1,
        start_method: str | None = None,
    ) -> None:
        if not env_fns:
            raise ValueError("grouped VecEnv requires at least one environment")
        if envs_per_worker < 1:
            raise ValueError("envs_per_worker must be positive")
        self.waiting = False
        self.closed = False
        self._groups: list[list[int]] = [
            list(range(start, min(start + envs_per_worker, len(env_fns))))
            for start in range(0, len(env_fns), envs_per_worker)
        ]
        self._env_to_group: dict[int, tuple[int, int]] = {
            env_index: (group_index, local_index)
            for group_index, group in enumerate(self._groups)
            for local_index, env_index in enumerate(group)
        }
        if start_method is None:
            start_method = (
                "forkserver"
                if "forkserver" in mp.get_all_start_methods()
                else "spawn"
            )
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _group in self._groups], strict=True
        )
        self.processes = []
        for work_remote, remote, group in zip(
            self.work_remotes, self.remotes, self._groups, strict=True
        ):
            wrappers = [CloudpickleWrapper(env_fns[index]) for index in group]
            process = ctx.Process(
                target=_group_worker,
                args=(work_remote, remote, wrappers),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, action_space)
        self._cached_action_masks: list[np.ndarray | None] = [
            None for _ in range(self.num_envs)
        ]

    @staticmethod
    def _info_mask(info: dict[str, Any]) -> np.ndarray | None:
        mask = info.get("action_mask")
        return None if mask is None else np.asarray(mask, dtype=bool).copy()

    def _grouped(self, indices) -> dict[int, list[tuple[int, int]]]:
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for index in self._get_indices(indices):
            group_index, local_index = self._env_to_group[int(index)]
            grouped[group_index].append((int(index), local_index))
        return grouped

    def _receive_groups(self, pending: dict[Any, int]) -> dict[int, Any]:
        results: dict[int, Any] = {}
        while pending:
            for remote in wait_for_connections(tuple(pending)):
                group_index = pending.pop(remote)
                results[group_index] = remote.recv()
        return results

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        if len(actions) != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} actions, got {len(actions)}"
            )
        for group_index, group in enumerate(self._groups):
            self.remotes[group_index].send(
                ("step", [actions[index] for index in group])
            )
        self.waiting = True

    def step_wait(self):
        pending = {
            remote: group_index
            for group_index, remote in enumerate(self.remotes)
        }
        grouped_results = self._receive_groups(pending)
        results = [
            grouped_results[group_index][local_index]
            for group_index, group in enumerate(self._groups)
            for local_index, _env_index in enumerate(group)
        ]
        self.waiting = False
        observations, rewards, dones, infos, self.reset_infos = zip(
            *results, strict=True
        )
        observations, rewards, dones, infos = (
            _stack_obs(observations, self.observation_space),
            np.stack(rewards),
            np.stack(dones),
            infos,
        )
        for index, (done, info) in enumerate(zip(dones, infos, strict=True)):
            source = self.reset_infos[index] if done else info
            self._cached_action_masks[index] = self._info_mask(source)
        return observations, rewards, dones, infos

    def reset(self):
        for group_index, group in enumerate(self._groups):
            requests = [
                (self._seeds[index], self._options[index]) for index in group
            ]
            self.remotes[group_index].send(("reset", requests))
        pending = {
            remote: group_index
            for group_index, remote in enumerate(self.remotes)
        }
        grouped_results = self._receive_groups(pending)
        results = [
            grouped_results[group_index][local_index]
            for group_index, group in enumerate(self._groups)
            for local_index, _env_index in enumerate(group)
        ]
        observations, self.reset_infos = zip(*results, strict=True)
        self.reset_infos = list(self.reset_infos)
        self._reset_seeds()
        self._reset_options()
        self._cached_action_masks = [
            self._info_mask(info) for info in self.reset_infos
        ]
        return _stack_obs(observations, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            self.step_wait()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self):
        for remote in self.remotes:
            remote.send(("render", None))
        grouped = self._receive_groups({
            remote: group_index
            for group_index, remote in enumerate(self.remotes)
        })
        return [
            grouped[group_index][local_index]
            for group_index, group in enumerate(self._groups)
            for local_index, _env_index in enumerate(group)
        ]

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        selected = list(self._get_indices(indices))
        if (
            method_name == EXPECTED_METHOD_NAME
            and not method_args
            and not method_kwargs
        ):
            cached = [self._cached_action_masks[index] for index in selected]
            if all(mask is not None for mask in cached):
                return [mask.copy() for mask in cached]
        requests = {}
        for group_index, members in self._grouped(selected).items():
            local_indices = [local for _global, local in members]
            self.remotes[group_index].send(
                ("env_method", (local_indices, method_name, method_args, method_kwargs))
            )
            requests[self.remotes[group_index]] = group_index
        grouped = self._receive_groups(requests)
        values = [grouped[self._env_to_group[index][0]][self._env_to_group[index][1]] for index in selected]
        if method_name == EXPECTED_METHOD_NAME:
            for index, value in zip(selected, values, strict=True):
                self._cached_action_masks[index] = np.asarray(value, dtype=bool).copy()
        return values

    def env_method_batch(self, method_name: str, calls: dict[int, tuple[tuple[Any, ...], dict[str, Any]]]):
        if not calls:
            return {}
        grouped_calls: dict[int, list[tuple[int, str, tuple, dict]]] = defaultdict(list)
        for index, (args, kwargs) in calls.items():
            group_index, local_index = self._env_to_group[int(index)]
            grouped_calls[group_index].append(
                (local_index, method_name, args, kwargs)
            )
        requests = {}
        for group_index, group_calls in grouped_calls.items():
            self.remotes[group_index].send(("env_method_batch", group_calls))
            requests[self.remotes[group_index]] = group_index
        grouped = self._receive_groups(requests)
        return {
            int(index): grouped[self._env_to_group[int(index)][0]][
                self._env_to_group[int(index)][1]
            ]
            for index in calls
        }

    def env_method_stream(self) -> GroupedMethodStream:
        """Open a search-scoped stream for independently scheduled groups.

        No other VecEnv operation may use the worker pipes until the returned
        stream has no pending commands.  NOSL collection satisfies that rule by
        completing every combat search before the environment step.
        """
        if self.waiting:
            raise RuntimeError("cannot open an env-method stream during step_wait")
        return GroupedMethodStream(self)

    def get_attr(self, attr_name: str, indices=None):
        selected = list(self._get_indices(indices))
        requests = {}
        for group_index, members in self._grouped(selected).items():
            self.remotes[group_index].send(
                ("get_attr", ([local for _global, local in members], attr_name))
            )
            requests[self.remotes[group_index]] = group_index
        grouped = self._receive_groups(requests)
        return [
            grouped[self._env_to_group[index][0]][self._env_to_group[index][1]]
            for index in selected
        ]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        requests = {}
        for group_index, members in self._grouped(indices).items():
            self.remotes[group_index].send(
                ("set_attr", ([local for _global, local in members], attr_name, value))
            )
            requests[self.remotes[group_index]] = group_index
        self._receive_groups(requests)

    def has_attr(self, attr_name: str) -> bool:
        selected = list(range(self.num_envs))
        requests = {}
        for group_index, members in self._grouped(selected).items():
            self.remotes[group_index].send(
                ("has_attr", ([local for _global, local in members], attr_name))
            )
            requests[self.remotes[group_index]] = group_index
        grouped = self._receive_groups(requests)
        return all(
            grouped[group_index][local]
            for group_index, group in enumerate(self._groups)
            for local, _global in enumerate(group)
        )

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices=None):
        selected = list(self._get_indices(indices))
        requests = {}
        for group_index, members in self._grouped(selected).items():
            self.remotes[group_index].send(
                ("is_wrapped", ([local for _global, local in members], wrapper_class))
            )
            requests[self.remotes[group_index]] = group_index
        grouped = self._receive_groups(requests)
        return [
            grouped[self._env_to_group[index][0]][self._env_to_group[index][1]]
            for index in selected
        ]
