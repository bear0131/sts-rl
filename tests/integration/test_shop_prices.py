"""SHOP02: initial inflation, restock pricing and fixed removal prices."""
import subprocess

import pytest


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 15, 16, 20])
@pytest.mark.parametrize('variant', range(10), ids=[
    'initial', 'courier', 'membership', 'both', 'restock-card-discounts',
    'restock-relic-price', 'mask-fixed', 'membership-with-mask', 'used-removal',
    'both-removal-discounts',
])
def test_shop_price_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'shop-price', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')
