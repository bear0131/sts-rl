"""Pending callbacks survive forks; consumed/cancelled captures are released."""
import subprocess

import pytest


@pytest.mark.native
@pytest.mark.timeout(240)
@pytest.mark.parametrize('variant', [0, 1, 2, 3, 4],
                         ids=['fork-fifo', 'ownership', 'victory', 'growth-forks', 'growth-victory'])
@pytest.mark.parametrize('offset', range(50))
def test_pending_queue_contract(game_context_executable, offset, variant):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'action-queue', str(offset), str(variant)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')
