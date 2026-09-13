import copy
import unittest

from env.core.combat_env import SimCombatEnv
from env.search.nosl_sampler import NOSLSampler


class NOSLSamplerTest(unittest.TestCase):
    def setUp(self):
        self.env = SimCombatEnv()
        self.env.reset(seed=17)

    def tearDown(self):
        self.env.close()

    @staticmethod
    def _draw_signature(bc):
        return tuple(tuple(card) for card in bc.to_dict()["draw"])

    @staticmethod
    def _state_signature(bc):
        # to_dict is a debug/public-state view; it intentionally excludes RNG
        # internals while retaining the visible combat state and draw order.
        return repr(bc.to_dict())

    def test_same_nonce_erases_prior_hidden_world_line(self):
        first = self.env.bc.clone()
        second = self.env.bc.clone()

        first.nosl_redeterminize(0x1111)
        second.nosl_redeterminize(0x2222)
        first.nosl_redeterminize(0xABCDEF)
        second.nosl_redeterminize(0xABCDEF)

        self.assertEqual(self._state_signature(first), self._state_signature(second))

    def test_clone_isolated_and_sampler_nonce_is_independent(self):
        original = copy.deepcopy(self.env.bc.to_dict())
        sampler = NOSLSampler(master_nonce=0x123456789)
        branches = [
            sampler.clone_and_redeterminize(
                self.env.bc,
                state_nonce=0xCAFE,
                simulation_index=index,
            )
            for index in range(8)
        ]

        self.assertEqual(self.env.bc.to_dict(), original)
        # A non-trivial draw pile should not collapse all independent samples
        # to one hidden permutation.
        signatures = {self._draw_signature(branch) for branch in branches}
        self.assertGreater(len(signatures), 1)

    def test_future_observation_changes_with_nonce(self):
        sampler = NOSLSampler(master_nonce=0xBEEF)
        observations = []
        for index in range(8):
            branch = sampler.clone_and_redeterminize(
                self.env.bc,
                state_nonce=0xCAFE,
                simulation_index=index,
            )
            end_turn = next(
                action
                for action in branch.legal_actions()
                if action.action_type.name == "END_TURN"
            )
            branch.execute(end_turn)
            observations.append(self._state_signature(branch))

        self.assertGreater(len(set(observations)), 1)


if __name__ == "__main__":
    unittest.main()
