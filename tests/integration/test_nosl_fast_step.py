import unittest
from types import SimpleNamespace

from env.core import actions, game_actions as ga
from env.core.combat_env import SimCombatEnv
from env.core.game_env import _CombatBranchGameEnv


class NOSLFastStepTest(unittest.TestCase):
    def test_fast_branch_step_matches_regular_branch(self):
        env = SimCombatEnv()
        try:
            env.reset(seed=17)
            source = SimpleNamespace(bc=env.bc, gc=env.gc)
            regular = _CombatBranchGameEnv(source)
            fast = _CombatBranchGameEnv(source)
            semantic = ga.combat(actions.END_TURN)

            regular._apply(semantic)
            regular_terminal = regular.bc.is_battle_over
            fast_terminal = fast.step_fast(semantic)

            self.assertEqual(fast_terminal, regular_terminal)
            self.assertEqual(fast.bc.to_dict(), regular.bc.to_dict())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
