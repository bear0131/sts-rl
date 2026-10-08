"""Shared deterministic native fixtures; missing native builds are errors."""
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import pytest
import torch
import numpy as np

from env.core.run_env import SlayTheSpireEnv
import slaythespire as sts


@pytest.fixture(scope="session", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def combat_factory():
    created = []

    def make(deck, encounter="Jaw Worm", *, seed=42, relics=(),
             combat_action_contract="legacy", card_language=False):
        env = SlayTheSpireEnv(combat_action_contract=combat_action_contract,card_language=card_language)
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
def replayable_selection_env():
    from env.core.game_env import SimGameEnv
    from env.core import native_actions as C
    class SelectionBackend(SimGameEnv):
        def reset(self, *, seed=None):
            super().reset(seed=seed)
            for _ in range(64):
                if self.bc is not None:
                    break
                self.step(self.legal_semantic_actions()[0])
            assert self.bc is not None
            self.bc.apply_custom_state({'hand': ['PURITY'] + ['STRIKE_RED'] * 5})
            self.bc.set_player_strength(999)
            return self._canonical(), self._info()
    env = SlayTheSpireEnv(backend_factory=SelectionBackend,
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, card_language=True)
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def chest_env_type():
    """Real treasure/reward state machine, reached through a replayable route."""
    from env.core.game_env import SimGameEnv
    from env.core import game_actions as ga

    class ChestBackend(SimGameEnv):
        def reset(self, *, seed=None):
            super().reset(seed=seed)
            self.gc.skip_battles = True
            for _ in range(500):
                if self.phase() == 'chest':
                    break
                acts = self.legal_semantic_actions()
                self.step(ga.proceed() if ga.proceed() in acts else acts[0])
            assert self.phase() == 'chest'
            self.gc.skip_battles = False
            return self._canonical(), self._info()

        def _settle(self):
            super()._settle()
            if self.bc is not None:
                self.bc.set_player_strength(999)

    class ChestEnv(SlayTheSpireEnv):
        def __init__(self, **kwargs):
            super().__init__(backend_factory=ChestBackend, **kwargs)

    return ChestEnv
