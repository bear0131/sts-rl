import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from env.core.combat_env import SimCombatEnv
from rl.combat_value_search import CombatValueSearch, CombatValueSearchConfig


class _ToyPolicy:
    def obs_to_tensor(self, observation):
        return {
            key: torch.as_tensor(value)
            for key, value in observation.items()
        }, True

    def predict_values(self, observation):
        state = observation["state"].long().reshape(-1)
        values = torch.tensor(
            [0.0, 0.2, 0.5, 0.3, 0.9], dtype=torch.float32
        )
        return values[state].unsqueeze(-1)

    def get_distribution(self, observation, action_masks=None):
        state = observation["state"].long().reshape(-1)
        logits = torch.zeros((len(state), 2), dtype=torch.float32)
        logits[state == 0] = torch.tensor([3.0, 0.0])
        mask = torch.as_tensor(action_masks, dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1.0e9)
        return SimpleNamespace(
            distribution=torch.distributions.Categorical(logits=logits)
        )


class _ToyModel:
    policy = _ToyPolicy()
    hierarchical_combat_hp = False


class _ToyCombatEnv:
    _TRANSITIONS = {
        (0, 0): (1, False),
        (0, 1): (2, False),
        (1, 0): (3, True),
        (2, 0): (4, True),
    }

    def __init__(self, state=0):
        self.state = state

    @property
    def unwrapped(self):
        return self

    def _phase(self):
        return "combat"

    def _obs(self):
        return {"state": np.asarray([self.state], dtype=np.int64)}

    def action_masks(self):
        mask = np.zeros(2, dtype=bool)
        for state, action in self._TRANSITIONS:
            if state == self.state:
                mask[action] = True
        return mask

    def fork_combat(self):
        return copy.copy(self)

    def combat_action_advances(self, action):
        if not self.action_masks()[int(action)]:
            raise ValueError(action)
        return True

    def step(self, action):
        self.state, terminal = self._TRANSITIONS[(self.state, int(action))]
        obs = self._obs()
        info = {"action_mask": self.action_masks()}
        return obs, 0.0, terminal, False, info


class _ToyBattleContext:
    redeterminized = []

    def clone(self):
        return copy.copy(self)

    def nosl_redeterminize(self, nonce):
        self.redeterminized.append(int(nonce))


class _ToyNoslCombatEnv(_ToyCombatEnv):
    def __init__(self, state=0):
        super().__init__(state)
        self.thick = SimpleNamespace(bc=_ToyBattleContext())

    def fork_combat(self):
        branch = copy.copy(self)
        branch.thick = SimpleNamespace(bc=self.thick.bc.clone())
        return branch


class CombatValueSearchTest(unittest.TestCase):
    def test_nosl_is_default_for_native_branches(self):
        _ToyBattleContext.redeterminized.clear()
        search = CombatValueSearch(
            _ToyModel(),
            CombatValueSearchConfig(
                depth=1,
                beam_width=2,
                policy_top_k=1,
                node_budget=4,
                terminal_hp=False,
            ),
        )

        search.search(_ToyNoslCombatEnv())

        self.assertGreaterEqual(len(_ToyBattleContext.redeterminized), 1)

    def test_value_lookahead_can_override_policy_argmax(self):
        search = CombatValueSearch(
            _ToyModel(),
            CombatValueSearchConfig(
                depth=2,
                beam_width=2,
                policy_top_k=1,
                node_budget=16,
                terminal_hp=False,
                nosl=False,
            ),
        )

        result = search.search(_ToyCombatEnv())

        self.assertEqual(result.model_action, 0)
        self.assertEqual(result.action, 1)
        self.assertEqual(result.principal_variation, (1, 0))
        self.assertAlmostEqual(result.value, 0.9)
        self.assertEqual(result.depth_reached, 2)

    def test_small_value_gain_falls_back_to_policy(self):
        search = CombatValueSearch(
            _ToyModel(),
            CombatValueSearchConfig(
                depth=2,
                beam_width=2,
                policy_top_k=1,
                node_budget=16,
                min_value_gain=1.0,
                terminal_hp=False,
                nosl=False,
            ),
        )

        result = search.search(_ToyCombatEnv())

        self.assertEqual(result.model_action, 0)
        self.assertEqual(result.action, 0)
        self.assertAlmostEqual(result.value, 0.3)

    def test_native_battle_clone_is_isolated(self):
        env = SimCombatEnv()
        try:
            env.reset(seed=17)
            original = env.bc.to_dict()
            branch = env.bc.clone()
            end_turn = next(
                action
                for action in branch.legal_actions()
                if int(action.action_type) == 0
            )
            branch.execute(end_turn)

            self.assertEqual(env.bc.to_dict(), original)
            self.assertNotEqual(branch.to_dict(), original)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
