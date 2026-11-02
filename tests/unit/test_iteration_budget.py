"""Hard wall deadlines cover a blocked call, and cancellation leaves it alive."""
import json
import subprocess
import sys


def test_deadline_and_cancel(tmp_path):
    code = ('from pathlib import Path; import time; '
            'from rl.training.iteration_budget import IterationBudget; '
            f'b=IterationBudget(.1,rank=0,directory=Path({str(tmp_path)!r})); '
            'b.start(7); time.sleep(3)')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=5)
    assert result.returncode == 124
    report = json.loads((tmp_path / 'round-timeout-rank0.json').read_text())
    assert report['iteration'] == 7 and report['elapsed'] < 1
    result = subprocess.run([sys.executable, '-c', code.replace('time.sleep(3)', 'b.cancel(); time.sleep(.2)')],
                            capture_output=True, timeout=5)
    assert result.returncode == 0
