import unittest

import numpy as np
import torch

from rl.accumulating_ppo import ppo_loss_components
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


if __name__ == "__main__":
    unittest.main()
