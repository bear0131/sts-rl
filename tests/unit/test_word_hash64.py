import json
from pathlib import Path
import subprocess

import pytest


REFERENCE = json.loads((Path(__file__).parents[1] / 'fixtures/word_hash64_reference.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: f"{c['seed']}-{c['words']}")
def test_word_stream_matches_upstream_xxh64(game_context_executable, case):
    executable, environment = game_context_executable
    result = subprocess.run([str(executable), 'word-hash64', str(case['seed']), str(case['words'])],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert int(result.stdout.strip()) == case['expected']
