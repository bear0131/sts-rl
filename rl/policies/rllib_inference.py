"""Standalone inference adapter for RLlib ``policy_state.pkl`` checkpoints."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from env.core.run_env import SlayTheSpireEnv
from rl.policies.split_policy import CombatStSPolicyCore, SplitStSPolicyCore


class RLLibInferencePolicy:
    """Minimal SB3-like inference API over the shared policy core."""

    def __init__(
        self,
        core: CombatStSPolicyCore | SplitStSPolicyCore,
        observation_space,
        device: torch.device,
    ):
        self.core = core
        self.observation_space = observation_space
        self.device = device

    def obs_to_tensor(self, observation):
        tensors = {}
        vectorized = None
        for key, value in observation.items():
            array = np.asarray(value)
            shape = self.observation_space[key].shape
            if array.shape == shape:
                is_vectorized = False
                array = np.expand_dims(array, axis=0)
            elif array.shape[1:] == shape:
                is_vectorized = True
            else:
                raise ValueError(
                    f"observation {key!r} has shape {array.shape}; "
                    f"expected {shape} or (batch, {', '.join(map(str, shape))})"
                )
            if vectorized is not None and vectorized != is_vectorized:
                raise ValueError("observation fields mix batched and unbatched values")
            vectorized = is_vectorized
            tensors[key] = torch.as_tensor(array, device=self.device)
        return tensors, bool(vectorized)

    def get_distribution(self, observation, action_masks=None):
        logits = self.core.action_logits(observation)
        if action_masks is not None:
            mask = torch.as_tensor(
                action_masks, device=logits.device, dtype=torch.bool
            )
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != logits.shape:
                raise ValueError(
                    f"action mask shape {tuple(mask.shape)} does not match "
                    f"logits {tuple(logits.shape)}"
                )
            logits = logits.masked_fill(~mask, -1.0e9)
        return SimpleNamespace(
            distribution=torch.distributions.Categorical(logits=logits)
        )

    def predict_values(self, observation):
        return self.core.value(observation)


class RLLibInferenceModel:
    """Restore inference state from one RLlib ``policy_state.pkl`` file."""

    def __init__(self, path: Path | str, device: str = "auto"):
        path = Path(path)
        try:
            with path.open("rb") as stream:
                state = pickle.load(stream)
        except Exception as exc:
            raise ValueError(f"cannot read RLlib policy_state.pkl: {exc}") from exc

        if not isinstance(state, dict) or not isinstance(state.get("weights"), dict):
            raise ValueError(
                "RLlib policy_state.pkl must contain a policy 'weights' mapping"
            )
        policy_spec = state.get("policy_spec")
        if not isinstance(policy_spec, dict):
            raise ValueError("RLlib policy_state.pkl has no serialized policy_spec")
        config = policy_spec.get("config")
        if not isinstance(config, dict):
            raise ValueError("RLlib policy_state.pkl has no policy config")

        fields = self._observation_fields(policy_spec)
        env = SlayTheSpireEnv(
            global_map_view="map_full_rooms" in fields,
            card_language="draw_card_ids" in fields,
            combat_language="player_status_language" in fields,
            deck_size_observation="deck_size" in fields,
            neow_observation="neow_bonus_ids" in fields,
            event_option_observation="event_option_ids" in fields,
        )
        try:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        finally:
            env.close()

        custom = config.get("model", {}).get("custom_model_config", {})
        if not isinstance(custom, dict):
            raise ValueError("RLlib policy config has no custom model config")
        task = custom.get("task")
        if task not in {"combat", "wholerun"}:
            raise ValueError(
                "RLlib policy config has no current specialized model task"
            )
        core_config = {
            "d_model": int(custom.get("d_model", 256)),
            "n_layers": int(custom.get("n_layers", 4)),
            "n_heads": int(custom.get("n_heads", 8)),
            "card_language": bool(custom.get("card_language", False)),
            "combat_language": bool(custom.get("combat_language", False)),
            "map_dag": bool(custom.get("map_dag", True)),
        }
        resolved_device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        core_class = (
            CombatStSPolicyCore if task == "combat" else SplitStSPolicyCore
        )
        core = core_class(self.observation_space, **core_config)
        core_weights = {
            key.removeprefix("core."): torch.as_tensor(value)
            for key, value in state["weights"].items()
            if key.startswith("core.")
        }
        if not core_weights:
            raise ValueError("RLlib policy has no specialized policy-core weights")
        try:
            core.load_state_dict(core_weights, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"RLlib policy architecture does not match this checkout: {exc}"
            ) from exc
        core.to(resolved_device).eval()

        self.policy = RLLibInferencePolicy(
            core, self.observation_space, resolved_device
        )
        self.global_timestep = int(state.get("global_timestep", 0))
        self.config = config

    @staticmethod
    def _observation_fields(policy_spec: dict) -> set[str]:
        try:
            outer = policy_spec["observation_space"]["space"]
            observations = outer["spaces"]["observations"]
            return set(observations["spaces"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "RLlib policy_state.pkl has an unsupported observation-space format"
            ) from exc

    def set_random_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def predict(self, observation, action_masks=None, deterministic=True):
        obs_tensor, _ = self.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.policy.get_distribution(
                obs_tensor, action_masks=action_masks
            ).distribution
            action = (
                distribution.probs.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
        return action.cpu().numpy(), None
