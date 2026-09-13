"""Focused RLlib adapter tests for the specialized policy cores.

Run from the repository root:

    python -m tests.unit.test_rllib_split_policy
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from env.core import action_space as A
from env.core import encode
from env.core import vocab as V
from rl.rllib_trainer import (
    MASK_LOG_FLOOR,
    StSRLLibTorchModel,
    _apply_sb3_weights,
)
from rl.split_policy import CombatStSPolicyCore, SplitStSPolicyCore


torch.set_num_threads(1)


def _observation_space():
    return encode.obs_space(
        global_map_view=True,
        card_language=False,
        combat_language=False,
        deck_size_observation=True,
        neow_observation=True,
        event_option_observation=True,
    )


def _wrapped_space(observation_space):
    return spaces.Dict({
        "observations": observation_space,
        "action_mask": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(A.N_ACTIONS,),
            dtype=np.float32,
        ),
    })


def _observation(observation_space, phase):
    observation = {}
    for key, field in observation_space.spaces.items():
        dtype = torch.long if field.dtype.kind in ("i", "u") else torch.float32
        observation[key] = torch.zeros((1, *field.shape), dtype=dtype)
    observation["phase"][0, 0] = V.PHASE_VOCAB_INDEX[phase]
    return observation


def _model(observation_space, task):
    return StSRLLibTorchModel(
        _wrapped_space(observation_space),
        spaces.Discrete(A.N_ACTIONS),
        A.N_ACTIONS,
        {
            "custom_model_config": {
                "task": task,
                "d_model": 32,
                "n_layers": 1 if task == "combat" else 2,
                "n_heads": 4,
                "card_language": False,
                "combat_language": False,
                "map_dag": True,
            }
        },
        f"test-{task}",
    )


def test_rllib_constructs_the_task_specific_core_and_masks_logits():
    observation_space = _observation_space()
    for task, phase, expected_type in (
        ("combat", "combat", CombatStSPolicyCore),
        ("wholerun", "event", SplitStSPolicyCore),
    ):
        model = _model(observation_space, task)
        assert isinstance(model.core, expected_type)
        mask = torch.zeros((1, A.N_ACTIONS), dtype=torch.float32)
        valid_action = A.seg_base("END_TURN")
        mask[0, valid_action] = 1.0
        logits, state = model.forward(
            {
                "obs": {
                    "observations": _observation(observation_space, phase),
                    "action_mask": mask,
                }
            },
            [],
            None,
        )
        assert state == []
        assert logits.shape == (1, A.N_ACTIONS)
        assert model.value_function().shape == (1,)
        invalid = torch.ones(A.N_ACTIONS, dtype=torch.bool)
        invalid[valid_action] = False
        assert torch.all(logits[0, invalid] == MASK_LOG_FLOOR)


class _RunnerGroup:
    def __init__(self):
        self.synced = False

    def sync_weights(self):
        self.synced = True


class _Algorithm:
    def __init__(self, core):
        self._policy = SimpleNamespace(model=SimpleNamespace(core=core))
        self.env_runner_group = _RunnerGroup()

    def get_policy(self):
        return self._policy


def test_combat_checkpoint_initializes_only_the_matching_split_parameters():
    observation_space = _observation_space()
    source = CombatStSPolicyCore(
        observation_space,
        d_model=32,
        n_layers=1,
        n_heads=4,
    )
    target = SplitStSPolicyCore(
        observation_space,
        d_model=32,
        n_layers=2,
        n_heads=4,
        map_dag=True,
    )
    overworld_before = {
        key: value.detach().clone()
        for key, value in target.overworld.state_dict().items()
    }
    algorithm = _Algorithm(target)
    checkpoint = SimpleNamespace(core=source, task="combat")

    _apply_sb3_weights(algorithm, checkpoint, freeze_encoder=True)

    source_state = source.state_dict()
    target_state = target.state_dict()
    for key, value in source_state.items():
        torch.testing.assert_close(target_state[key], value)
    for key, value in overworld_before.items():
        torch.testing.assert_close(target.overworld.state_dict()[key], value)
    assert algorithm.env_runner_group.synced
    assert all(not parameter.requires_grad for parameter in target.shared.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in target.combat.encoder.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in target.combat.fixed_head.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in target.overworld.parameters()
    )


if __name__ == "__main__":
    test_rllib_constructs_the_task_specific_core_and_masks_logits()
    test_combat_checkpoint_initializes_only_the_matching_split_parameters()
    print("RLlib specialized core construction, masking, and migration OK")
