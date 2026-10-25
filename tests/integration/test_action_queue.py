"""Callback ownership must survive dequeue, ring reuse, and context copying."""
import subprocess

import pytest


@pytest.mark.parametrize("push_front", [0, 1])
def test_dequeued_callback_is_transferred_and_released(game_context_executable, push_front):
    executable, run_env = game_context_executable
    result = subprocess.run(
        [str(executable), "queue-ownership", str(push_front), "0"],
        env=run_env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
