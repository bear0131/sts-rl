"""Launcher-level contracts that are independent of a remote host."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_fork_resume_explicitly_resets_environment_geometry():
    launcher = (ROOT / "experiments/launchers/launch_nosl_replay.sh").read_text()

    fork_branch = re.search(
        r'if \[ "\$\{NOSL_FORK_RESUME:-0\}" = 1 \]; then\n'
        r'(?P<body>.*?)\n  else\n',
        launcher,
        flags=re.DOTALL,
    )
    assert fork_branch is not None
    assert '--fork-state "$resume_state" --fork-reset-environments' in fork_branch.group("body")
    assert 'extra+=(--resume-state "$resume_state")' in launcher
