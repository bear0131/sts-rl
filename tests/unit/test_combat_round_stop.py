"""One explicit experiment boundary; no real SSH or training."""
import json
from types import SimpleNamespace

import pytest

from experiments.launchers import combat_deployment as d


@pytest.mark.parametrize('step,expected', [(0, 'wait'), (40*786432, 'wait'),
                                          (48*786432, 'wait'), (49*786432, 'request-stop')])
def test_only_previous_published_round_opens_stop_window(step, expected):
    assert d.round_stop_decision(step, 50, 786432) == expected


@pytest.mark.parametrize('step', [50*786432, 51*786432, -1, 49*786432+1])
def test_late_or_partial_boundaries_fail_closed(step):
    with pytest.raises(ValueError):
        d.round_stop_decision(step, 50, 786432)


@pytest.fixture
def watcher(tmp_path, monkeypatch):
    import psutil
    run = tmp_path/'runs/test-run'
    code = tmp_path/'rl/training/distributed_sb3.py'
    code.parent.mkdir(parents=True)
    code.write_text('pinned trainer')
    command = ['python', '-m', 'rl.training.distributed_sb3', '--continuous']
    identity = dict(trainer=100, created=10., supervisor=99, supervisor_created=9.)
    spec = dict(root=str(tmp_path), run_id='test-run', identity=identity, command=command,
                trainer_source_sha256=d.digest(code), watcher_timeout_seconds=30,
                target_round=50, slots_per_round=786432, poll_seconds=5)
    path = tmp_path/'request.json'
    d.write(path, spec)
    d.write(run/'rank0.pid.json', identity)
    def process(pid=None):
        return SimpleNamespace(pid=pid, create_time=lambda: {100:10.,99:9.}.get(pid, 11.),
            cwd=lambda:str(tmp_path), ppid=lambda:99, cmdline=lambda:command)
    monkeypatch.setattr(psutil, 'Process', process)
    now = [0.]
    monkeypatch.setattr(d.time, 'monotonic', lambda:now[0])
    def publish(step):
        d.write(run/f'recovery/rank0/step-{step:012d}/manifest.json',
            dict(run_id='test-run', rank=0, world_size=4, schema_version=2, global_step=step))
    publish(48*786432)
    def sleep(seconds):
        now[0] += seconds
        publish(49*786432)
    monkeypatch.setattr(d.time, 'sleep', sleep)
    return path, run, spec, publish, now


def test_wait_then_single_stop_and_no_rearming(watcher):
    path, run, _, _, _ = watcher
    d.request_stop_after_round(path)
    assert d.read(run/'STOP')['completed_step'] == 49*786432
    assert d.read(path.with_suffix('.receipt.json'))['status'] == 'stop-requested'
    with pytest.raises(FileExistsError):
        d.request_stop_after_round(path)


@pytest.mark.parametrize('failure', ['source', 'birth', 'command', 'manifest', 'late', 'timeout', 'existing-stop'])
def test_failure_never_creates_a_stop(watcher, monkeypatch, failure):
    path, run, spec, publish, now = watcher
    if failure == 'source':
        spec['trainer_source_sha256'] = 'wrong'
    elif failure == 'birth':
        spec['identity']['created'] = 20.
    elif failure == 'command':
        spec['command'] = ['different']
    elif failure == 'manifest':
        p = next((run/'recovery/rank0').glob('*/manifest.json'))
        v = d.read(p); v['run_id'] = 'another-run'; d.write(p, v)
    elif failure == 'late':
        publish(50*786432)
    elif failure == 'timeout':
        spec['watcher_timeout_seconds'] = 0
    else:
        (run/'STOP').write_text('user stop')
    d.write(path, spec)
    with pytest.raises((ValueError, TimeoutError)):
        d.request_stop_after_round(path)
    assert (run/'STOP').exists() == (failure == 'existing-stop')
    if failure == 'existing-stop':
        assert (run/'STOP').read_text() == 'user stop'
    assert d.read(path.with_suffix('.receipt.json'))['status'] == 'failed'
