import unittest
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from env.core import vocab as V
from rl.accumulating_ppo import ppo_loss_components
from rl.distributed_sb3 import ppo_microbatch_backward
from rl.hierarchical_returns import compute_hierarchical_returns


COMBAT = 1
OVERWORLD = 2


def calculate(phases, rewards, dones, next_phases, next_hp, interrupted=None,
              values=None, last_values=None):
    shape = np.asarray(phases).shape
    return compute_hierarchical_returns(
        phases=np.asarray(phases),
        values=np.zeros(shape, np.float32) if values is None else values,
        raw_rewards=np.asarray(rewards, np.float32),
        dones=np.asarray(dones, bool),
        next_phases=np.asarray(next_phases),
        next_hp=np.asarray(next_hp, np.float32),
        interrupted=(
            np.zeros(shape, bool) if interrupted is None else interrupted
        ),
        last_values=(
            np.zeros(shape[1], np.float32)
            if last_values is None else last_values
        ),
        combat_phase=COMBAT,
        overworld_gamma=1.0,
        overworld_gae_lambda=1.0,
        combat_gamma=1.0,
        combat_gae_lambda=1.0,
    )


class HierarchicalReturnsTest(unittest.TestCase):
    def test_combat_is_local_and_overworld_skips_it(self):
        rewards, returns, advantages, stats = calculate(
            phases=[[OVERWORLD], [COMBAT], [COMBAT], [OVERWORLD]],
            rewards=[[0.0], [0.0], [50.0], [1.0]],
            dones=[[False], [False], [False], [True]],
            next_phases=[[COMBAT], [COMBAT], [OVERWORLD], [OVERWORLD]],
            next_hp=[[0.0], [0.0], [0.4], [0.0]],
        )

        np.testing.assert_allclose(returns[:, 0], [51.0, 0.4, 0.4, 1.0])
        np.testing.assert_allclose(advantages, returns)
        np.testing.assert_allclose(rewards[:, 0], [50.0, 0.0, 0.4, 1.0])
        self.assertEqual(stats.combats_finished, 1)
        self.assertAlmostEqual(stats.combat_end_hp_mean, 0.4)

    def test_rollout_cut_mid_combat_omits_only_pending_upper_action(self):
        _rewards, returns, advantages, stats = calculate(
            phases=[[OVERWORLD], [COMBAT]],
            rewards=[[0.0], [0.0]],
            dones=[[False], [False]],
            next_phases=[[COMBAT], [COMBAT]],
            next_hp=[[0.0], [0.5]],
            last_values=np.asarray([0.25], np.float32),
        )

        self.assertTrue(np.isnan(returns[0, 0]))
        self.assertTrue(np.isnan(advantages[0, 0]))
        self.assertAlmostEqual(returns[1, 0], 0.25)
        self.assertEqual(stats.unresolved_overworld_samples, 1)

    def test_death_terminates_both_objectives_with_zero_hp(self):
        _rewards, returns, _advantages, stats = calculate(
            phases=[[OVERWORLD], [COMBAT], [COMBAT]],
            rewards=[[0.0], [0.0], [0.0]],
            dones=[[False], [False], [True]],
            next_phases=[[COMBAT], [COMBAT], [COMBAT]],
            next_hp=[[0.0], [0.0], [0.0]],
        )

        np.testing.assert_allclose(returns[:, 0], [0.0, 0.0, 0.0])
        self.assertEqual(stats.combats_finished, 1)

    def test_interruption_does_not_pay_surviving_hp(self):
        rewards, returns, _advantages, stats = calculate(
            phases=[[COMBAT]],
            rewards=[[0.0]],
            dones=[[True]],
            next_phases=[[COMBAT]],
            next_hp=[[0.7]],
            interrupted=np.asarray([[True]]),
        )

        self.assertEqual(rewards[0, 0], 0.0)
        self.assertEqual(returns[0, 0], 0.0)
        self.assertEqual(stats.combats_finished, 0)

    def test_phase_balanced_loss_ignores_unresolved_macro_sample(self):
        values = torch.zeros(4, requires_grad=True)
        log_prob = torch.zeros(4, requires_grad=True)
        components = ppo_loss_components(
            hierarchical=True,
            normalize_advantage=True,
            observations={
                "phase": torch.tensor(
                    [[COMBAT], [COMBAT], [OVERWORLD], [OVERWORLD]]
                )
            },
            advantages=torch.tensor([1.0, -1.0, 2.0, float("nan")]),
            returns=torch.tensor([1.0, 0.0, 2.0, float("nan")]),
            old_values=torch.zeros(4),
            old_log_prob=torch.zeros(4),
            values=values,
            log_prob=log_prob,
            entropy=torch.ones(4),
            clip_range=0.2,
            clip_range_vf=None,
        )
        loss = (
            components.policy_loss
            + 0.01 * components.entropy_loss
            + 0.5 * components.value_loss
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(torch.isfinite(log_prob.grad).all())
        self.assertEqual(components.valid_samples, 3)
        self.assertEqual(components.combat_samples, 2)
        self.assertEqual(components.overworld_samples, 1)
        self.assertAlmostEqual(components.value_loss.item(), 2.25)

    def test_phasewise_backward_matches_combined_minibatch_gradient(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 3)

            def evaluate_actions(self, obs, actions, action_masks=None):
                del actions, action_masks
                output = self.projection(obs["features"])
                return output[:, :1], output[:, 1], output[:, 2]

        class Model:
            def __init__(self, policy):
                self.policy = policy
                self.action_space = spaces.Discrete(2)
                self.device = torch.device("cpu")
                self.hierarchical_combat_hp = True
                self.normalize_advantage = True
                self.ent_coef = 0.01
                self.vf_coef = 0.5

            @staticmethod
            def autocast_context():
                return nullcontext()

        torch.manual_seed(7)
        combined_policy = Policy()
        phasewise_policy = deepcopy(combined_policy)
        combat = V.PHASE_VOCAB_INDEX["combat"]
        map_phase = V.PHASE_VOCAB_INDEX["map"]
        rollout = SimpleNamespace(
            observations={
                "phase": torch.tensor([
                    [combat], [map_phase], [combat], [map_phase], [combat],
                    [map_phase],
                ]),
                "features": torch.randn(6, 3),
            },
            actions=torch.tensor([[0], [1], [0], [1], [0], [1]]),
            action_masks=torch.ones(6, 2, dtype=torch.bool),
            advantages=torch.tensor([1.2, -0.4, -0.7, 0.9, 0.3, float("nan")]),
            returns=torch.tensor([0.8, 1.1, 0.2, -0.3, 0.5, float("nan")]),
            old_values=torch.tensor([0.1, 0.2, 0.3, -0.1, 0.4, 0.0]),
            old_log_prob=torch.tensor([-0.2, -0.3, -0.1, -0.4, -0.2, 0.0]),
        )

        combined = ppo_microbatch_backward(
            Model(combined_policy), rollout, 0.2, None
        )
        phasewise = ppo_microbatch_backward(
            Model(phasewise_policy),
            rollout,
            0.2,
            None,
            phasewise_backward=True,
        )

        self.assertAlmostEqual(combined["loss"], phasewise["loss"], places=6)
        self.assertAlmostEqual(
            combined["combat_value_loss"],
            phasewise["combat_value_loss"],
            places=6,
        )
        self.assertAlmostEqual(
            combined["overworld_value_loss"],
            phasewise["overworld_value_loss"],
            places=6,
        )
        for combined_parameter, phasewise_parameter in zip(
            combined_policy.parameters(), phasewise_policy.parameters(), strict=True
        ):
            torch.testing.assert_close(
                combined_parameter.grad,
                phasewise_parameter.grad,
                atol=1e-6,
                rtol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
