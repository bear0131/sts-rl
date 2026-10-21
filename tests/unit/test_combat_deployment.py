"""Deployment contract tests; no SSH, provisioning, or external training."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from experiments.launchers.combat_deployment import validate, training_command, agree, common_update, supervise
from rl.training.distributed_sb3 import build_parser

ROOT = Path(__file__).resolve().parents[2]


def test_remote_script_wire_preserves_posix_lines_and_utf8():
    import sys
    from experiments.launchers.combat_deployment import run_ssh_script
    child = 'import sys; data=sys.stdin.buffer.read(); assert b"\\r" not in data; sys.stdout.buffer.write(data)'
    result = run_ssh_script([sys.executable, '-c', child], 'mkdir /tmp/example\r\nprintf 测试\n', 10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'mkdir /tmp/example\nprintf 测试\n'


def configuration():
    config = json.loads((ROOT/'experiments/launchers/combat01_876_4x5090.json').read_text(encoding='utf-8-sig'))
    config['remote_repo'] = '/root/autodl-tmp/combat01-new'
    config['ring_port'] = 29880
    for rank in config['ranks']:
        rank.update(host='localhost', port=22000+rank['rank'])
    return config


def test_deployment_preserves_whole_lower_training_and_876_geometry():
    config = validate(configuration(), connected=True)
    profile = json.loads((ROOT/config['profile']).read_text())
    for rank in range(4):
        command = training_command(config, profile, rank)
        args = build_parser().parse_args(command[4:])
        assert args.rank == rank and args.world_size == 4 and len(args.peer) == 3
        assert args.gamma == 1. and args.learning_rate == 5e-5
        assert (args.n_envs, args.n_steps) == (384, 512)
        assert (args.nosl_is_az_batch_size, args.nosl_is_az_micro_batch_size, args.nosl_is_az_updates) == (102400, 8192, 40)
        assert args.nosl_is_az_update_schedule == 'constant'
        assert args.hierarchical_combat_hp is None and args.resume_state is None and args.fork_state is None
        assert args.leaf_inference == 'cuda-graph' and args.leaf_compile
        assert args.barrier_timeout == 900
        assert args.updates == 2 and args.save_every_updates == 1 and args.max_checkpoints == 2
        assert 'train47' not in str(args.checkpoint)
        assert str(args.checkpoint).replace('\\', '/').endswith('/material/initial.zip')


def test_production_ssh_socket_fits_even_with_long_deployment_paths():
    from rl.training.distributed_sb3 import parse_peer, ssh_control_directory, ssh_options
    config = configuration()
    config['remote_repo'] = '/root/autodl-tmp/' + 'long-deployment-' * 8
    config['run_id'] = 'combat01-876-outer-v1-4x5090-retry'
    profile = json.loads((ROOT/config['profile']).read_text())
    for rank in range(4):
        args = build_parser().parse_args(training_command(config, profile, rank)[4:])
        directory = ssh_control_directory(Path(args.ssh_control_dir), args.run_id, rank)
        options = ssh_options(parse_peer(args.peer[0]), Path(args.ssh_key), directory)
        assert 'ControlMaster=auto' in options
        assert len(str(directory).encode()) + 1 + 40 + 17 < 104


@pytest.mark.parametrize('mutation', [
    lambda c: c['ranks'][0].update(host=None),
    lambda c: c['ranks'][1].update(host=c['ranks'][0]['host'], port=c['ranks'][0]['port']),
    lambda c: c['budget'].update(max_seconds=0),
    lambda c: c['budget'].update(rounds=0),
    lambda c: c.update(remote_repo='/'),
    lambda c: c.update(remote_repo='/root/../etc'),
    lambda c: c.update(resume_step=688914432),
])
def test_deployment_rejects_missing_endpoints_unbounded_or_old_state(mutation):
    config = configuration()
    mutation(config)
    with pytest.raises(ValueError):
        validate(config, connected=True)


def test_rank_agreement_checks_all_identities_not_zip_timestamps():
    reports = [dict(rank=i, source_commit='commit', config_sha256='config', profile_sha256='profile',
        native_sha256='binary', teacher_sha256='teacher', lower_sha256='lower', torch='version',
        python='3.11', objective_identity={'context':'v2'}, runtime_files={'a.py':'hash'}, initial_sha256=str(i)) for i in range(4)]
    assert len(agree(reports)) == 4
    for key in ('source_commit', 'native_sha256', 'teacher_sha256', 'lower_sha256', 'objective_identity', 'torch', 'runtime_files'):
        changed = deepcopy(reports)
        changed[2][key] = 'changed'
        with pytest.raises(ValueError, match=key):
            agree(changed)
    with pytest.raises(ValueError, match='four'):
        agree(reports[:3])


def test_update_requires_four_published_generations_and_actual_lower_updates():
    states = [dict(rank=i, generations=[786432], metric=dict(global_timesteps=786432,
        optimizer_steps=40, **{'outer_return/upper_updates': 0, 'outer_return/labelled_roots': 100}))
        for i in range(4)]
    assert common_update(states) == 786432
    for mutation in (lambda s: s.update(generations=[]), lambda s: s.update(metric=None),
                     lambda s: s['metric'].update(optimizer_steps=0),
                     lambda s: s['metric'].update({'outer_return/upper_updates': 1})):
        incomplete = deepcopy(states)
        mutation(incomplete[3])
        assert common_update(incomplete) is None


def test_status_checks_real_metrics_bundle_hashes_and_pid_identity(tmp_path, monkeypatch):
    import os
    import psutil
    from experiments.launchers import combat_deployment as deployment
    config = configuration()
    config['remote_repo'] = str(tmp_path)
    path = tmp_path/'deployment.json'
    deployment.write(path, config)
    monkeypatch.setattr(deployment, 'validate', lambda c, **kw: c)
    run = tmp_path/'runs'/config['run_id']
    objective = {'context': 'v2', 'teacher': 'fixed'}
    deployment.write(run/'preflight-rank0.json', {'objective_identity': objective})
    deployment.write(run/'rank0.pid.json', dict(trainer=os.getpid(), created=psutil.Process().create_time()))
    metric = dict(global_timesteps=786432, optimizer_steps=40)
    (run/'rank0.log').write_text('startup\n'+json.dumps(metric)+'\n', encoding='utf-8')
    bundle = run/'recovery/rank0/step-000000786432'
    bundle.mkdir(parents=True)
    for name in ('model.zip', 'learner.pt', 'environment.pt'):
        (bundle/name).write_bytes(b'checkpoint')
    deployment.write(bundle/'manifest.json', dict(schema_version=2, rank=0, world_size=4,
        global_step=786432, run_id=config['run_id'], outer_return_contract=objective,
        sha256={p.name: deployment.digest(p) for p in bundle.iterdir()}))
    status = deployment.remote_status(path, 0, verify=True)
    assert status['metric'] == metric and status['generations'] == [786432]
    assert not status['alive'], 'an existing PID running pytest is not a trainer'
    (bundle/'learner.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        deployment.remote_status(path, 0, verify=True)


@pytest.mark.parametrize('exit_at', [None, 30., 83.])
def test_supervisor_finishes_or_requests_save_then_kills_at_deadline(tmp_path, monkeypatch, exit_at):
    import subprocess
    from types import SimpleNamespace
    from experiments.launchers import combat_deployment as deployment
    now, signals = [0.], []
    stop = tmp_path/'STOP'
    class Process:
        pid = 123
        returncode = None
        def poll(self):
            if exit_at is not None and now[0] >= exit_at:
                self.returncode = 0
            return self.returncode
        def wait(self, timeout):
            if self.poll() is not None:
                return self.returncode
            now[0] += timeout
            if self.poll() is not None:
                return self.returncode
            raise subprocess.TimeoutExpired('test child', timeout)
    process = Process()
    # POSIX signals are passed as opaque constants, so this contract runs on Windows too.
    monkeypatch.setattr(deployment, 'signal', SimpleNamespace(SIGTERM=15, SIGKILL=9))
    def kill(pid, value):
        assert pid == 123 and stop.exists()
        signals.append((now[0], value))
        if value == 9:
            process.returncode = -9
    monkeypatch.setattr(deployment.os, 'killpg', kill, raising=False)
    monkeypatch.setattr(deployment.time, 'time', lambda: now[0])
    code = supervise(process, stop, 100)
    if exit_at is None:
        assert code == 124 and signals == [(90., 15), (100., 9)]
    else:
        assert code == 0 and not signals
        assert stop.exists() == (exit_at > 80.)
