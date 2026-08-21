"""Subprocess vector environment that reuses masks returned by each step."""

from __future__ import annotations

from typing import Any

import numpy as np
from sb3_contrib.common.maskable.utils import EXPECTED_METHOD_NAME
from stable_baselines3.common.vec_env import SubprocVecEnv


class CachedMaskSubprocVecEnv(SubprocVecEnv):
    """Avoid a second worker round trip for the next action mask.

    Gymnasium step/reset info already contains ``action_mask``. SB3-contrib
    normally asks every subprocess for ``action_masks()`` again before the next
    policy call. Cache the info value while retaining ``env_method`` fallback
    behavior for environments that do not provide it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        observations, rewards, dones, infos = super().step_wait()
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
