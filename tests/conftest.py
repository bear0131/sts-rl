"""Shared deterministic native fixtures; missing native builds are errors."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import pytest
import torch
import numpy as np

from env.core import _BUILD
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
def gauntlet_factory():
    from env.core.combat_backend import SimCombatBackend

    def make(deck, *, relics=(), hp=80, seed=42):
        def factory(_rng):
            def make_gc():
                gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
                gc.floor_num = 1
                gc.set_relics(list(relics))
                gc.set_deck([(card, 0) for card in deck])
                gc.cur_hp = hp
                return gc
            return make_gc, lambda _gc: "Jaw Worm"
        env = SimCombatBackend(factory)
        env.reset(seed=42)
        return env

    return make


_DESIGNER_DECK = [('STRIKE_RED', 0), ('DEFEND_RED', 0), ('BASH', 0),
         ('ANGER', 0), ('INFLAME', 0)]
# Public map entry fixtures, not a claim of naturally reaching Act 2.
_DESIGNER_ENTRIES = {
    (0, False): (165, 5, 1), (0, True): (284, 4, 1),
    (15, False): (164, 3, 2), (15, True): (235, 4, 1),
}


@pytest.fixture
def designer_factory():
    def make(ascension, option):
        seed, x, y = _DESIGNER_ENTRIES[ascension, option in (1, 3)]
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
        gc.act, gc.floor_num, gc.gold, gc.cur_hp = 2, 20, 500, 60
        gc.monster_chance = gc.shop_chance = gc.treasure_chance = 0
        gc.cur_map_node_y = y - 1
        gc.transition_to_map_node(x)
        gc.set_deck(_DESIGNER_DECK)
        assert gc.cur_event_name == 'Designer'
        assert option in gc.legal_event_options()
        return gc
    return make


@pytest.fixture(scope='session')
def game_context_executable():
    build = Path(_BUILD).resolve()
    source = Path(__file__).resolve().parents[1] / 'sim' / 'sts_lightspeed'
    cache = (build / 'CMakeCache.txt').read_text()
    home = next(line.split('=', 1)[1] for line in cache.splitlines()
                if line.startswith('CMAKE_HOME_DIRECTORY:INTERNAL='))
    assert Path(home).resolve() == source, 'native tests require a build of this checkout'
    cmake = Path(sys.executable).parent / ('cmake.exe' if sys.platform == 'win32' else 'cmake')
    command = str(cmake) if cmake.exists() else shutil.which('cmake')
    assert command, 'install the locked test dependencies (CMake is required)'
    result = subprocess.run([command, '--build', str(build), '--target', 'game-context-contract',
                             '--config', 'Release', '--parallel', '2'],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    name = 'game-context-contract' + ('.exe' if sys.platform == 'win32' else '')
    executable = next((p for p in (build / name, build / 'Release' / name) if p.is_file()), None)
    assert executable, 'CMake did not produce the contract executable'
    run_env = os.environ.copy()
    if sys.platform == 'win32':
        compiler = next(line.split('=', 1)[1] for line in cache.splitlines()
                        if line.startswith('CMAKE_CXX_COMPILER:FILEPATH='))
        # add_dll_directory handles in the Python parent do not reach a native child.
        run_env['PATH'] = str(Path(compiler).parent) + os.pathsep + run_env.get('PATH', '')
    return executable, run_env
