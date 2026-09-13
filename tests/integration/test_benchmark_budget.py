"""The benchmark supervisor must actually enforce its advertised wall budget."""
import json
import subprocess
import sys


def test_benchmark_timeout_exits_with_machine_readable_status(tmp_path):
    output = tmp_path / "timeout.json"
    result = subprocess.run(
        [sys.executable, "-m", "rl.benchmarks.train", "--output", str(output),
         "--timeout", "0.001"], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 124, result.stderr
    assert json.loads(output.read_text())["status"] == "timeout"
