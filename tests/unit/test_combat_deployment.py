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


@pytest.mark.parametrize('profile_name', ['combat_finite_win', 'combat_matched_curves', 'combat_value_units'])
def test_curve_profile_uses_existing_deployment_plan(tmp_path, monkeypatch, capsys, profile_name):
    from experiments.launchers import combat_deployment as deployment
    config = configuration()
    config['profile'] = f'configs/train/{profile_name}_876_4x5090.json'
    profile = tmp_path/config['profile']
    profile.parent.mkdir(parents=True)
    profile.write_bytes((ROOT/config['profile']).read_bytes())
    path = tmp_path/'deployment.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(deployment, '__file__', str(tmp_path/'experiments/launchers/combat_deployment.py'))
    monkeypatch.setattr(deployment.subprocess, 'check_output', lambda *a, **kw: 'test-commit')
    deployment.manage(path, 'plan')
    plan = json.loads(capsys.readouterr().out)
    assert plan['upper_updates'] == 0 and plan['global_slots'] == 2*786432
    assert plan['maximum_updates'] == 80


def configuration():
    config = json.loads((ROOT/'experiments/launchers/combat01_876_4x5090.json').read_text(encoding='utf-8-sig'))
    config['remote_repo'] = '/root/autodl-tmp/combat01-new'
    config['ring_port'] = 29880
    for rank in config['ranks']:
        rank.update(host='localhost', port=22000+rank['rank'])
    return config


@pytest.mark.parametrize('arm,physical', [('curves', (0, 1)), ('manual', (2, 3))])
def test_strict_pair_isolated_local_ranks_and_workload(arm, physical):
    config = configuration()
    config.update(world_size=2, profile=f'configs/train/combat10_{arm}_876_2x5090.json')
    config['ranks'] = [dict(config['ranks'][p], rank=i) for i, p in enumerate(physical)]
    validate(config, connected=True)
    profile = json.loads((ROOT/config['profile']).read_text())
    assert profile['contract']['external_value_unit_scale'] == 1
    assert profile['training']['slot_steps_global_per_round'] == 49152
    assert profile['search']['gumbel_rescale_q'] is True
    assert profile['search']['gumbel_value_scale'] == .1
    from rl.search.native import NativeSearchConfig
    search = NativeSearchConfig.from_mapping(profile['search'])
    assert search.paper_full_gumbel and search.gumbel_rescale_q
    assert search.gumbel_value_scale == .1 and search.leaf_batch_size == 1
    for rank in range(2):
        args = build_parser().parse_args(training_command(config, profile, rank)[4:])
        assert (args.rank, args.world_size, args.n_envs, args.n_steps) == (rank, 2, 96, 256)
        assert args.peer == [f'{1-rank}=root@localhost:{22000+physical[1-rank]}']
        assert args.nosl_is_az_updates == 40 and args.nosl_is_az_simulations == 128
    with pytest.raises(ValueError, match='world size'):
        training_command(dict(config, world_size=4), profile, 0)


def test_pair_agreement_and_common_save_require_both_local_ranks():
    reports = [dict(rank=i, source_commit='x', config_sha256='x', profile_sha256='x',
        native_sha256='x', teacher_sha256='x', lower_sha256='x', torch='x', python='x',
        objective_identity={}, runtime_files={}) for i in range(2)]
    assert len(agree(reports, 2)) == 2
    with pytest.raises(ValueError):
        agree(reports[:1], 2)
    states = [dict(rank=i, generations=[49152], metric=dict(global_timesteps=49152,
        optimizer_steps=40, **{'frozen_upper/upper_updates': 0, 'nosl_is_az/completed_combat_roots': 10})) for i in range(2)]
    assert common_update(states, 2) == 49152
    states[1]['generations'] = []
    assert common_update(states, 2) is None


def test_raw_six_profile_changes_only_curve_scale_contract():
    old = json.loads((ROOT/'configs/train/combat11_six_curves_876_2x5090.json').read_text())
    raw_path = 'configs/train/combat12_six_curves_raw_876_2x5090.json'
    raw = json.loads((ROOT/raw_path).read_text())
    for key in ('source', 'source_checkpoint_contract', 'training', 'search', 'simulator'):
        assert raw[key] == old[key]
    changed = {k for k in set(old['contract']) | set(raw['contract'])
               if old['contract'].get(k) != raw['contract'].get(k)}
    assert changed == {'experiment_number', 'comparison', 'curve_bank_file_sha256',
                       'curve_bank_identity', 'payoff', 'full_hp_target'}
    config = configuration()
    config.update(world_size=2, profile=raw_path)
    config['ranks'] = config['ranks'][:2]
    validate(config, connected=True)
    for rank in (0, 1):
        args = build_parser().parse_args(training_command(config, raw, rank)[4:])
        assert (args.rank, args.world_size, args.n_envs, args.n_steps) == (rank, 2, 96, 256)
        assert args.nosl_is_az_updates == 40 and args.nosl_is_az_simulations == 128


def test_small_conditional_profile_uses_resident_two_rank_deployment():
    config = configuration()
    config.update(world_size=2, continuous=True,
        profile='configs/train/combat14_small_conditional_876_2x5090.json')
    config['ranks'] = [dict(config['ranks'][i+2],rank=i) for i in range(2)]
    config['budget'].update(rounds=None,max_seconds=None,seed=9002000)
    validate(config,connected=True)
    profile = json.loads((ROOT/config['profile']).read_text())
    for rank in range(2):
        cmd=training_command(config,profile,rank)
        args=build_parser().parse_args(cmd[4:])
        assert args.continuous and '--updates' not in cmd and '--total-steps' not in cmd
        assert (args.rank,args.world_size,args.n_envs,args.n_steps,args.seed)==(rank,2,96,256,9002000)
        assert args.nosl_is_az_updates==40 and args.nosl_is_az_simulations==128


@pytest.mark.parametrize('profile_name', ['combat_outer_return', 'combat_matched_curves', 'combat_value_units'])
def test_deployment_preserves_whole_lower_training_and_876_geometry(profile_name):
    config = configuration()
    config['profile'] = f'configs/train/{profile_name}_876_4x5090.json'
    config = validate(config, connected=True)
    profile = json.loads((ROOT/config['profile']).read_text())
    for rank in range(4):
        command = training_command(config, profile, rank)
        args = build_parser().parse_args(command[4:])
        assert args.rank == rank and args.world_size == 4 and len(args.peer) == 3
        assert args.gamma == 1. and args.learning_rate == 5e-5
        assert (args.n_envs, args.n_steps) == (384, 512)
        assert args.nosl_is_az_simulations == 128
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


def test_explicit_unbounded_supervision_never_requests_timed_stop(tmp_path, monkeypatch):
    import experiments.launchers.combat_deployment as deployment
    def unexpected_clock():
        raise AssertionError('unbounded supervision must not set a deadline')
    monkeypatch.setattr(deployment.time, 'time', unexpected_clock)
    class Process:
        def wait(self):
            return 0
    stop = tmp_path/'STOP'
    assert supervise(Process(), stop, None) == 0
    assert not stop.exists()


def test_continuous_deployment_has_no_finite_training_target():
    config = configuration()
    config['continuous'] = True
    with pytest.raises(ValueError, match='null round'):
        validate(config, connected=True)
    config['budget'].update(rounds=None, max_seconds=None)
    validate(config, connected=True)
    profile = json.loads((ROOT/config['profile']).read_text())
    command = training_command(config, profile, 0)
    args = build_parser().parse_args(command[4:])
    assert args.continuous and args.total_steps is None
    assert '--updates' not in command and '--total-steps' not in command
    assert args.stop_file and args.training_state_dir
    assert args.save_every_updates == 1 and args.max_checkpoints == 2
    config['continuous'] = False
    with pytest.raises(ValueError, match='positive round'):
        validate(config, connected=True)
