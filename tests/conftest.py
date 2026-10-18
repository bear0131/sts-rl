"""Shared deterministic native fixtures; missing native builds are errors."""
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import pytest
import torch
import numpy as np

from env import core  # Preserve STS1 path setup without requiring its binary.


@pytest.fixture(scope="session", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def combat_factory():
    from env.core.run_env import SlayTheSpireEnv
    import slaythespire as sts
    created = []

    def make(deck, encounter="Jaw Worm", *, seed=42, relics=(),
             combat_action_contract="legacy"):
        env = SlayTheSpireEnv(combat_action_contract=combat_action_contract)
        env.reset(seed=seed)
        # This fixture's seed is the native game seed, not the wrapper RNG seed.
        env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
        env.thick.gc.floor_num = 1
        env.thick.gc.set_deck([(card, 0) for card in deck])
        env.thick.gc.set_relics(list(relics))
        env.thick.bc = sts.create_battle(env.thick.gc, encounter)
        env._invalidate_backend_cache()
        created.append(env)
        return env

    yield make
    for env in created:
        env.close()


@pytest.fixture
def short_combat_env_type():
    from env.core.run_env import SlayTheSpireEnv
    class ShortCombatEnv(SlayTheSpireEnv):
        def reset(self, **kwargs):
            super().reset(**kwargs)
            # Enter through the real run state machine so exit callbacks exist.
            for _ in range(64):
                if self.thick.bc is not None:
                    break
                self.step(int(np.flatnonzero(self.action_masks())[0]))
            assert self.thick.bc is not None, "fixture failed to reach combat"
            self.thick.bc.set_player_strength(999)
            self._invalidate_backend_cache()
            return self._obs(), self._info()
    return ShortCombatEnv


@pytest.fixture
def sts2_choice_env(request):
    from env.sts2.game import Sts2Env
    card, wire_id = getattr(request, 'param', ('BurningPact', 'BURNING_PACT'))

    class ChoiceEnv(Sts2Env):
        def _new_run(self, seed):
            return self.native.RunManager(seed=seed, neow_start=False,
                deck=[card, 'BurningPact', 'StrikeIronclad', 'DefendIronclad', 'Bash'])

    env = ChoiceEnv()
    env.reset(seed=42)
    index = next(i for i, (_, desc) in enumerate(env.candidates)
                 if desc.get('card', {}).get('id') == wire_id)
    env.step(index)
    yield env
    env.close()
