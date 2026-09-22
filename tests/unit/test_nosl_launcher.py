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
    assert 'resume_state="$NOSL_RESUME_ROOT/rank$rank/$generation"' in launcher
    assert 'extra+=(--resume-state "$resume_state")' in launcher


def test_fourway_manager_resolves_fork_generation_path():
    manager = (ROOT / "experiments/launchers/manage_nosl_replay_4way.ps1").read_text()
    assert '$(Resolve-RemotePath $ResumeRoot)/rank$($Peer.Rank)/$generation' in manager


def test_launcher_exposes_checkpoint_cadence_override():
    launcher = (ROOT / "experiments/launchers/launch_nosl_replay.sh").read_text()
    assert 'save_every_updates=${NOSL_SAVE_EVERY_UPDATES:-1}' in launcher
    assert '--save-every-updates "$save_every_updates"' in launcher
    assert '--max-checkpoints "${NOSL_MAX_CHECKPOINTS:-2}"' in launcher
    assert "must be a positive integer" in launcher


def test_fourway_manager_defaults_to_two_atomic_generations():
    manager = (ROOT / "experiments/launchers/manage_nosl_replay_4way.ps1").read_text()
    assert re.search(
        r'\[ValidateRange\(2, 100\)\]\s*'
        r'\[int\]\$MaxCheckpoints = 2,',
        manager,
    )
