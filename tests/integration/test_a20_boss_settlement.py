"""The first A20 Act3 boss must settle into a playable second fight."""
import pytest

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from rl.evaluation.runner import is_win

pytestmark = pytest.mark.native


def _finish_battle(env, native):
    encounter = env.thick.gc.encounter
    for _ in range(4):
        battle = env.thick.bc
        assert battle is not None, 'live combat phase must own its BattleContext'
        battle.apply_custom_state({
            'player': {'hp': 80, 'energy': 3, 'powers': [('STRENGTH', 999)]},
            'hand': ['CLEAVE'],
        })
        rebirthing = any(m['half_dead'] for m in battle.to_dict()['monsters'])
        env._invalidate_backend_cache()
        result = (env.step_native_combat((4 << 29) if rebirthing else 0) if native
                  else env.step(A._END if rebirthing else A._PLAY))
        if not native and env._pending is not None:
            result = env.step(int(env.action_masks().nonzero()[0][0]))
        if env.thick.bc is None or env.thick.gc.encounter != encounter:
            return result
    pytest.fail('overpowered fixture failed to clear a boss within four actions')


@pytest.mark.parametrize('ascension', [19, 20])
@pytest.mark.parametrize('native', [False, True])
def test_act3_victory_requires_all_native_bosses_and_never_exposes_dead_screen(ascension, native):
    env = SlayTheSpireEnv(ascension=ascension,
                         combat_action_contract='native-v1' if native else 'legacy')
    try:
        env.reset(seed=42)
        env.thick.gc.set_deck([('CLEAVE', 0)] * 10)
        # Jump only map rows; real native boss generation and victory callbacks
        # remain intact when moving through all three acts.
        for act in (1, 2, 3):
            env.thick.gc.cur_map_node_y = 14
            env.thick.gc.transition_to_map_node(0)
            env.thick._settle()
            first_encounter = env.thick.gc.encounter
            result = _finish_battle(env, native)
            _, _, terminal, truncated, info = result
            assert not truncated and not info.get('dead_screen', False)
            if act < 3:
                assert not terminal and not is_win(env)
                env.thick.gc.reward_proceed()
                env.thick.gc.choose_boss_relic(3)  # legal native skip
                env.thick._settle()
                continue
            if ascension == 20:
                assert not terminal and not is_win(env)
                assert env.thick.gc.encounter != first_encounter
                assert env.thick.bc is not None and not env.thick.bc.is_battle_over
                assert env.thick.phase() == 'combat' and env.action_masks().any()
                _, _, terminal, truncated, info = _finish_battle(env, native)
                assert not truncated and not info.get('dead_screen', False)
            assert terminal and is_win(env)
    finally:
        env.close()
