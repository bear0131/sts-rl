"""BOTTLE01: transform eligibility depends on the originating event/relic."""
import subprocess

import pytest


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(6), ids=['augmenter-candidates', 'augmenter-result',
    'ordinary-transform', 'ordinary-remove', 'astrolabe', 'augmenter-count'])
def test_bottle_transform_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'bottle-transform', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(18), ids=[
    *[f'cage-{n}' for n in range(5)], *[f'astrolabe-{n}' for n in range(5)],
    'boss-cage-auto', 'boss-cage-select', 'boss-astrolabe-auto', 'boss-astrolabe-select',
    'reward-cage-auto', 'reward-cage-select', 'reward-astrolabe-auto', 'reward-astrolabe-select'])
def test_relic_selection_threshold_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'relic-selection', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')
