"""SHOP03: egg previews and obtaining the previewed card are idempotent."""
import subprocess

import pytest


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(6), ids=['initial-stock', 'buy-molten',
    'buy-toxic', 'buy-frozen', 'repeat-preview', 'obtain-counts'])
def test_shop_egg_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'shop-egg', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')
