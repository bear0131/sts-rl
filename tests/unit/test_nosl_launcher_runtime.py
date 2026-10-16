"""Execute the real launcher with bounded process stand-ins, never a trainer."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('rank', [0, 1])
def test_launcher_attaches_once_per_rank0_incarnation_and_passes_new_log_boundary(tmp_path, rank):
    source = Path(os.environ.get('STS_LAUNCHER_UNDER_TEST',
                                 ROOT / 'experiments/launchers/launch_nosl_replay.sh'))
    repo = tmp_path / 'repo'
    (repo / '.venv/bin').mkdir(parents=True)
    (repo / 'tools/operations').mkdir(parents=True)
    (repo / 'tools/operations/wandb_heartbeat.py').touch()
    (repo / 'args').touch()
    (repo / 'checkpoint').touch()
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    executable = repo / '.venv/bin/python'
    executable.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
args = sys.argv[1:]
if '-c' in args:
    sys.exit(0)
if '-m' in args:
    (root / ('trainer-' + str(os.getpid()))).write_text('alive')
    print('{"global_timesteps":1000}', flush=True)
    time.sleep(15)
else:
    (root / ('helper-' + str(os.getpid()))).write_text(json.dumps(args))
''')
    executable.chmod(0o755)
    for name, code in [('pgrep', 'import sys;sys.exit(1)'),
                       ('stat', 'import pathlib,sys;print(pathlib.Path(sys.argv[-1]).stat().st_size)')]:
        path = bindir / name
        path.write_text(f'#!{sys.executable}\n{code}\n')
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               FIXTURE_ROOT=str(tmp_path), NOSL_REPO=str(repo), NOSL_RUN_ID='fixture',
               NOSL_WANDB_ID='fixture-wandb', NOSL_ARGS_FILE=str(repo / 'args'),
               NOSL_CHECKPOINT=str(repo / 'checkpoint'), NOSL_SKIP_TRANSPORT_CLEANUP='1')
    log = repo / f'runs/fixture/rank{rank}.log'
    log.parent.mkdir(parents=True)
    log.write_text('{"global_timesteps":9000}\n')
    previous_pid = None
    for attempt in range(2):
        before = log.stat().st_size
        env['NOSL_RESUME_STEP'] = str(1000 * attempt)
        result = subprocess.run(['bash', str(source), str(rank)], env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        pid = int(log.with_suffix('.pid').read_text())
        try:
            assert pid != previous_pid
            if rank == 0:
                helper_pid = int((log.parent / 'rank0.heartbeat.pid').read_text())
                evidence = tmp_path / f'helper-{helper_pid}'
                deadline = time.monotonic() + 3
                while not evidence.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                args = json.loads(evidence.read_text())
                assert args[args.index('--pid') + 1] == str(pid)
                assert args[args.index('--wandb-id') + 1] == 'fixture-wandb'
                assert args[args.index('--project') + 1] == 'sts-rl'
                assert '--auto' in args and '--core-pid' not in args and '--run' not in args
                offset = int(args[args.index('--log-start') + 1])
                assert offset >= before
                assert b'9000' not in log.read_bytes()[offset:]
                assert b'"global_timesteps":1000' in log.read_bytes()[offset:]
                assert len(list(tmp_path.glob('helper-*'))) == attempt + 1
            else:
                assert not list(tmp_path.glob('helper-*'))
                assert not (log.parent / 'rank0.heartbeat.pid').exists()
        finally:
            # Only the fixture-owned stand-in, never a process found by pattern.
            assert (tmp_path / f'trainer-{pid}').exists()
            os.kill(pid, signal.SIGTERM)
        previous_pid = pid
