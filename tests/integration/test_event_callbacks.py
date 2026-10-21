"""CALLBACK01: delayed event rewards must belong to their copied GameContext."""
import subprocess

import pytest


@pytest.mark.timeout(240)  # The shared native fixture may compile on first use.
@pytest.mark.parametrize('ascension', [0, 15])
@pytest.mark.parametrize('mode,variant', [('event', i) for i in range(5)] + [('boss', 1), ('boss', 2)],
                         ids=['dead-no-relic', 'dead-with-relic', 'mind-bloom', 'mushrooms',
                              'masked-bandits-control', 'act1-boss', 'act2-boss'])
def test_copied_callbacks_are_independent_of_origin(game_context_executable, mode, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), mode, str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 15])
@pytest.mark.parametrize('variant', range(7), ids=[
    'dead-no-relic', 'dead-with-relic', 'mind-bloom', 'mushrooms', 'bandits',
    'elite', 'act2-boss'])
def test_independent_single_reward_worldline(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'worldline', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')
