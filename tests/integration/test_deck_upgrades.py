"""DECK02: upgrade eligibility follows cards, not the number of upgrade calls."""
import subprocess

import pytest
import slaythespire as sts

from env.core.run_env import SlayTheSpireEnv
from env.core import action_space as A
from env.core import game_actions as ga


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(5), ids=['repeated-searing', 'random-mixed',
    'ordinary-card', 'whetstone', 'mind-bloom'])
def test_deck_upgrade_count_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'deck-upgrade', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('initial_upgrades', [0, 3])
def test_searing_blow_remains_available_at_later_campfires(ascension, initial_upgrades):
    env = SlayTheSpireEnv(ascension=ascension, combat_action_contract='native-v1')
    try:
        env.reset(seed=42)
        gc = env.thick.gc
        gc.set_deck([('SEARING_BLOW', initial_upgrades)])
        # Actual room transitions initialize their continuation; controlled map
        # entry isolates the eligibility contract from whole-run decisions.
        x, y = next((x, y) for y in range(15) for x in range(7)
                    if env.thick.rmap.room_type(x, y) == 'REST')
        for i in range(4):
            gc.cur_map_node_y = y - 1
            gc.transition_to_map_node(x)
            env._invalidate_backend_cache()
            assert 1 in gc.legal_campfire_options()
            assert ga.campfire('smith') in env.thick.legal_semantic_actions()
            assert env.action_masks()[A._CAMP + 1]
            env.step(A._CAMP + 1)
            assert gc.get_select_cards() == [('SEARING_BLOW', initial_upgrades + i)]
            gc.choose_select_card_option(0)
            assert gc.deck_ids() == [('SEARING_BLOW', initial_upgrades + i + 1)]
            assert gc.screen_state == sts.ScreenState.MAP_SCREEN
    finally:
        env.close()
