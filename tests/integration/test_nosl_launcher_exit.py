"""Exercise the real shell launcher with a bounded fake trainer on Linux."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(sys.platform != 'linux', reason='Production launcher uses Linux process signals')
@pytest.mark.parametrize('finish,expected', [('success', 0), ('error', 17), ('killed', 137)])
def test_launcher_retains_child_pid_and_exit_status(tmp_path, finish, expected):
    repo = tmp_path / 'repo'
    (repo / '.venv/bin').mkdir(parents=True)
    (repo / 'native').mkdir()
    checkpoint = repo / 'checkpoint.zip'
    checkpoint.write_bytes(b'fixture')
    args_file = repo / 'test.args'
    args_file.write_text('')
    stub = repo / 'fake.py'
    stub.write_text(
        'import os,sys,json,time,signal\n'
        'from pathlib import Path\n'
        'Path(os.environ["FIXTURE_IDENTITY"]).write_text(json.dumps({"pid":os.getpid(),'
        '"argv":sys.argv[1:],"faulthandler":os.environ.get("PYTHONFAULTHANDLER")}))\n'
        'time.sleep(2)\n'
        'mode=os.environ["FIXTURE_FINISH"]\n'
        'if mode=="killed":os.kill(os.getpid(),signal.SIGKILL)\n'
        'sys.exit(17 if mode=="error" else 0)\n'
    )
    import shlex
    python = repo / '.venv/bin/python'
    python.write_text('#!/bin/bash\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(stub)) + ' "$@"\n')
    python.chmod(0o755)
    run_id = f'launcher-exit-{os.getpid()}-{finish}'
    identity = repo / 'identity.json'
    env = os.environ.copy()
    env.update(NOSL_REPO=str(repo), NOSL_RUN_ID=run_id, NOSL_WANDB_ID='fixture',
               NOSL_ARGS_FILE=str(args_file), NOSL_CHECKPOINT=str(checkpoint),
               STS_LIGHTSPEED_BUILD_DIR=str(repo/'native'), NOSL_SKIP_TRANSPORT_CLEANUP='1',
               FIXTURE_IDENTITY=str(identity), FIXTURE_FINISH=finish)
    pid = None
    try:
        result = subprocess.run(['bash', str(ROOT/'experiments/launchers/launch_nosl_replay.sh'), '3'],
                                env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        run = repo / 'runs' / run_id
        pid = int((run/'rank3.pid').read_text())
        child = json.loads(identity.read_text())
        assert child['pid'] == pid
        assert child['faulthandler'] == '1'
        assert child['argv'][:3] == ['-u', '-m', 'rl.training.distributed_sb3']
        deadline = time.monotonic() + 10
        while not (run/'rank3.exit.json').exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        ended = json.loads((run/'rank3.exit.json').read_text())
        assert ended['pid'] == pid and ended['exit_code'] == expected
        assert ended['utc'].endswith('Z')
        assert f'TRAINER_EXIT pid={pid} exit_code={expected}' in (run/'rank3.log').read_text()
    finally:
        if pid is not None:
            try:os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:pass
