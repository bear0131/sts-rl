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
def isolated_torch_compile_cache():
    """Do not spend another test model's Dynamo recompilation budget.

    A single test retains all shape, precision, eviction and weight-update
    specializations so its runtime cache behavior remains under test.
    """
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


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
def upper_search_env_factory():
    """Real public room entries with native callbacks, not a saved-run claim."""
    created = []
    def make(kind, *, relics=(), room_type=None):
        env = SlayTheSpireEnv(global_map_view=True, upper_public_context=True,
            event_option_observation=True, combat_action_contract='autoregressive-v2')
        created.append(env)
        # Event identities are seeded native draws. Bound discovery instead of
        # changing opaque GameContext unions or mocking the transition code.
        for seed in range(1, 65):
            env.reset(seed=seed)
            gc = env.thick.gc
            room = room_type or ('MONSTER' if kind == 'combat_reward' else kind.upper())
            choices = [(x, y) for y in range(15) for x in range(7)
                       if env.thick.rmap.room_type(x, y) == room]
            if not choices:
                continue
            x, y = choices[0]
            gc.set_relics(list(relics))
            gc.gold, gc.cur_hp = 400, 40
            gc.skip_battles = kind == 'combat_reward'
            gc.monster_chance = gc.shop_chance = gc.treasure_chance = 0
            gc.cur_map_node_y = y - 1
            gc.cur_map_node_x = (-1 if y == 0 else next(px for px in range(7)
                                    if x in env.thick.rmap.reachable(px, y - 1)))
            gc.transition_to_map_node(x)
            env._invalidate_backend_cache()
            if kind != 'event' or gc.cur_event_name in ('The Cleric', 'Purifier', 'Upgrade Shrine'):
                assert env._phase() == kind
                return env
        raise AssertionError('bounded native fixture did not find requested room')
    yield make
    for env in created:
        env.close()


@pytest.fixture
def run_combat_factory():
    """Controlled battle with a real run exit continuation; seed 42101."""
    from env.core.game_env import SimGameEnv

    def make(encounter='Looter', potions=(), **env_options):
        env = SimGameEnv(**env_options)
        env.reset(seed=42101)
        for _ in range(64):
            if env.bc is not None:
                break
            env.step(env.legal_semantic_actions()[0])
        assert env.bc is not None, 'fixture failed to enter a real combat room'
        env.gc.gold = 99
        env.gc.set_deck([('STRIKE_RED', 0)] * 10)
        env.gc.set_potion_state(list(potions))
        env.bc = sts.create_battle(env.gc, encounter)
        return env

    return make


@pytest.fixture
def campfire_env_type():
    """Replayable native rest entry; setup is repeated inside backend reset."""
    from functools import partial
    from env.core.game_env import SimGameEnv

    class CampfireBackend(SimGameEnv):
        def reset(self, *, seed=None):
            super().reset(seed=seed)
            x, y = next((x, y) for y in range(15) for x in range(7)
                        if self.rmap.room_type(x, y) == 'REST')
            self.gc.set_relics(['Girya', 'Peace Pipe', 'Shovel'])
            self.gc.cur_map_node_y = y - 1
            self.gc.cur_map_node_x = (-1 if y == 0 else next(px for px in range(7)
                                        if x in self.rmap.reachable(px, y - 1)))
            self.gc.transition_to_map_node(x)
            assert self.phase() == 'rest'
            return self._canonical(), self._info()

        def _settle(self):
            super()._settle()
            if self.bc is not None:
                self.bc.set_player_strength(999)

    class CampfireEnv(SlayTheSpireEnv):
        def __init__(self, **kwargs):
            super().__init__(backend_factory=partial(CampfireBackend,
                global_map_view=kwargs.get('global_map_view', False),
                upper_public_context=kwargs.get('upper_public_context', False)), **kwargs)

    return CampfireEnv


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
def short_natural_game_env_type():
    """Replayable strategic entry followed by a genuine early combat loss.

    Controlled HP/strength keep lifecycle tests short. The simulator still
    owns termination; these are not quality-evaluation games.
    """
    from functools import partial
    from env.core.game_env import SimGameEnv
    class Backend(SimGameEnv):
        def __init__(self, *, start_on_turn_50, **kwargs):
            super().__init__(**kwargs)
            self.start_on_turn_50 = start_on_turn_50

        def reset(self, *, seed=None):
            super().reset(seed=seed)
            for _ in range(16):
                if self.phase() == 'map':
                    break
                self.step(self.legal_semantic_actions()[0])
            assert self.phase() == 'map'
            return self._canonical(), self._info()

        def _settle(self):
            super()._settle()
            if self.bc is not None:
                state = {'player': {'hp': 80 if self.start_on_turn_50 else 1, 'powers': [('STRENGTH', -999)]}}
                if self.start_on_turn_50 and self.bc.turn == 0:
                    state['turn'] = 49
                self.bc.apply_custom_state(state)

    class Env(SlayTheSpireEnv):
        def __init__(self, *, start_on_turn_50=False, **kwargs):
            super().__init__(backend_factory=partial(Backend, start_on_turn_50=start_on_turn_50,
                global_map_view=kwargs.get('global_map_view', False),
                upper_public_context=kwargs.get('upper_public_context', False)), **kwargs)
    return Env


@pytest.fixture
def act3_boss_map_root():
    """Controlled final boss entry; the native victory callback advances the act."""
    def configure(env):
        gc = env.thick.gc
        gc.act = 3
        gc.red_key = gc.blue_key = gc.green_key = True
        gc.cur_map_node_x = next(x for x in range(7)
                                if env.thick.rmap.room_type(x, 14) == 'REST')
        gc.cur_map_node_y, gc.floor_num = 14, 50
        gc.screen_state = sts.ScreenState.MAP_SCREEN
        gc.skip_battles = True
        env._invalidate_backend_cache()
        return gc.cur_map_node_x
    return configure


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
            from functools import partial
            super().__init__(backend_factory=partial(ChestBackend,
                global_map_view=kwargs.get('global_map_view', False),
                upper_public_context=kwargs.get('upper_public_context', False)), **kwargs)

    return ChestEnv


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
                        if line.startswith('CMAKE_CXX_COMPILER:'))
        # add_dll_directory handles in the Python parent do not reach a native child.
        run_env['PATH'] = str(Path(compiler).parent) + os.pathsep + run_env.get('PATH', '')
    return executable, run_env


@pytest.fixture
def entity_material_source(tmp_path):
    """Small tensor-only checkpoint with the real cross-layer buffer alias."""
    import hashlib
    import io
    import json
    from zipfile import ZipFile

    card = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    state = {
        'core.upper_tokenizer.card.vectors': card,
        'core.upper_tokenizer.projection.weight': torch.ones(2, 3),
        'core.overworld.actor.weight': torch.ones(2, 2),
        'core.overworld.value.weight': torch.ones(1, 2),
        'core.combat.card.vectors': card,
        'core.combat.encoder.weight': torch.ones(2, 3),
        'core.combat.pointer_query.weight': torch.ones(2, 2),
        'core.combat.distribution.win.weight': torch.ones(1, 2),
        'core.combat.transition.0.weight': torch.ones(2, 2),
    }
    metadata = {
        'num_timesteps': 123, 'policy_class': {'__module__': 'rl.policies.entity_joint'},
        'observation_space': {'fixture': 'original-space'}, 'action_space': {'n': 198},
        'policy_kwargs': {'entity_config': {'card_vector_checkpoint': '/old/cards.pt'}},
    }
    source, cards, profile, out = [tmp_path / n for n in ('source.zip', 'cards.pt', 'profile.json', 'material')]
    payload = io.BytesIO()
    torch.save(state, payload)
    with ZipFile(source, 'w') as archive:
        archive.writestr('data', json.dumps(metadata))
        archive.writestr('policy.pth', payload.getvalue())
        archive.writestr('policy.optimizer.pth', b'must never be imported')
    torch.save({'vectors': card}, cards)
    plan = {
        'kind': 'combat01-preparation-v1', 'launch_ready': False,
        'source': {'model_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'card_vectors_sha256': hashlib.sha256(cards.read_bytes()).hexdigest()},
        'source_checkpoint_contract': {'num_timesteps': 123},
        'source_components': {'upper': {'tensors': 4, 'elements': 24},
                              'lower_init': {'tensors': 3, 'elements': 22},
                              'discarded': {'tensors': 2, 'elements': 6}},
        'simulator': {'runtime_commit': 'fixture'},
    }
    profile.write_text(json.dumps(plan), encoding='utf-8')
    return source, cards, profile, out, state, plan


@pytest.fixture
def frozen_card_vectors_factory():
    """Canonical synthetic frozen tables without private policy checkpoints."""
    from rl.rules.rule_vector_model import FROZEN_CARD_VECTOR_TABLE_SCHEMA
    from env.core.cards_vocab import CARD_VOCAB

    def make(path, width, *, seed=42101):
        generator = torch.Generator().manual_seed(seed)
        torch.save(dict(schema=FROZEN_CARD_VECTOR_TABLE_SCHEMA, card_vocab=list(CARD_VOCAB),
                        vectors=torch.randn(len(CARD_VOCAB), 2, width, generator=generator)), path)
        return path

    return make


@pytest.fixture
def frozen_upper_source(tmp_path, frozen_card_vectors_factory):
    """A small real entity-joint core with TRAIN47's shared frozen card table."""
    from gymnasium import spaces
    from rl.policies.entity_combat import EntityCombatConfig
    from rl.policies.entity_joint import EntityJointCore, UpperConfig

    env = SlayTheSpireEnv(card_language=False, global_map_view=True,
                         combat_action_contract='autoregressive-v2')
    observation, _ = env.reset(seed=42101)
    old_space = spaces.Dict({k: v for k, v in env.observation_space.spaces.items()
                            if k not in ('run_keys', 'reward_keys')})
    cards = frozen_card_vectors_factory(tmp_path / 'cards.pt', 12)
    with torch.random.fork_rng():
        torch.manual_seed(42101)
        core = EntityJointCore(old_space,
            EntityCombatConfig(width=32, layers=1, heads=4, quantiles=8,
                               boss_win_weight=.9, card_vector_checkpoint=str(cards)),
            UpperConfig(width=32, layers=1, heads=4))
    core.eval()
    yield core, old_space, env, {k: torch.as_tensor(v[None]) for k, v in observation.items()}
    env.close()


@pytest.fixture
def outer_return_core(frozen_upper_source):
    from rl.policies.outer_return import OuterReturnCore
    source, source_space, _, _ = frozen_upper_source
    env = SlayTheSpireEnv(global_map_view=True, combat_action_contract='autoregressive-v2',
                         outer_return_context=True)
    env.reset(seed=42101)
    core = OuterReturnCore(source, source_space, env.observation_space, source_sha256='1' * 64,
                           lower_initialization='876')
    yield core, source, env
    env.close()


@pytest.fixture
def outer_material(frozen_upper_source, tmp_path):
    """Real small source architecture in the production immutable material format."""
    import hashlib
    import json
    from dataclasses import asdict
    from stable_baselines3.common.save_util import data_to_json
    core, space, env, _ = frozen_upper_source
    directory = tmp_path / 'material'
    directory.mkdir()
    state = core.state_dict()
    torch.save({'core.' + k: v for k, v in state.items()
                if k.startswith(('overworld.', 'upper_tokenizer.'))}, directory / 'upper.pt')
    torch.save({'core.' + k: v for k, v in state.items() if k.startswith('combat.')
                and not k.startswith(('combat.distribution.', 'combat.transition.'))}, directory / 'lower-init.pt')
    shutil.copy2(core.config.card_vector_checkpoint, directory / 'frozen-card-vectors.pt')
    (directory / 'constructor.json').write_text(data_to_json(dict(observation_space=space,
        action_space=env.action_space, policy_kwargs=dict(entity_config=asdict(core.config),
                                                        upper_config=asdict(core.upper_config)))))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(schema='entity-joint-initialization-material-v1', source_model_sha256='1' * 64,
                    optimizer_imported=False, replay_imported=False,
                    files={p.name: dict(sha256=digest(p)) for p in directory.iterdir()})
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return dict(material_directory=str(directory), material_sha256=digest(directory / 'manifest.json'),
                source_sha256='1' * 64)


@pytest.fixture
def finite_curve_bank(frozen_upper_source):
    from env.core.finite_curves import SCHEMA, curve_key
    from rl.policies.frozen_upper import FrozenUpperTeacher
    import slaythespire as sts
    source, space, _, _ = frozen_upper_source
    teacher = FrozenUpperTeacher(source, space, source_sha256='1'*64)
    curves = {}
    for act in (1, 2, 3):
        for encounter in sts.encounter_names():
            for outlet in ('win', 'escape'):
                base, slope = (.2, .3) if outlet == 'win' else (.1, .2)
                curves[curve_key(act, encounter.upper().replace(' ', '_'), outlet)] = [
                    [base+.025*p+slope*x for x in (.01, .5, 1.)] for p in range(6)]
    return dict(schema=SCHEMA, teacher_sha256=teacher.assert_integrity(),
                hp_fractions=[.01, .5, 1.], potion_counts=list(range(6)), curves=curves)


@pytest.fixture
def upper_phase_batch():
    """Mixed real observation schema, including PAD pointer candidates and map choices."""
    from copy import deepcopy
    from env.core import vocab
    from env.core.run_env import SlayTheSpireEnv
    env = SlayTheSpireEnv(card_language=True, global_map_view=True, upper_public_context=True,
        event_option_observation=True, neow_observation=True,
        combat_action_contract='autoregressive-v2')
    frame, _ = env.reset(seed=20261109)
    frames = []
    for name in ('shop', 'map', 'combat_reward', 'card_select', 'event', 'boss_relic', 'rest', 'chest'):
        row = deepcopy(frame)
        row['phase'][0] = vocab.PHASE_VOCAB_INDEX[name]
        row['relics'][:] = 0
        row['relics'][3] = 1
        row['relic_counters'][:] = .7
        row['deck_ids'][1::2] = 0
        row['selected_card_ids'][:] = 0
        row['select_card_ids'][:] = 0
        row['select_native_indices'][:] = 0
        row['select_native_indices'][0] = 1  # Executable skip carries PAD card ID.
        row['map_full_rooms'][-1] = 0
        row['map_full_features'][-1, 3] = 1  # Choice at a PAD map slot.
        frames.append(row)
    yield env, frames
    env.close()
