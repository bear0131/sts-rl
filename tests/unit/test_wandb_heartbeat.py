"""OPS02: liveness must never outlive bounded, real training progress."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.operations.wandb_heartbeat import Client, Guard, Process, read_process, read_progress


def process(cpu=10, start=100, state='S'):
    return Process(start, cpu, state)


def guard():
    return Guard(process(), process(start=200), 1000, 0, 0, 1500, 300)


def test_long_collection_can_heartbeat_without_new_metric():
    g = guard()
    for second in range(60, 721, 60):
        g.check(process(cpu=10 + second), process(start=200), 1000, second)
    g.check(process(cpu=800), process(start=200), 2000, 750)
    assert g.step == 2000


@pytest.mark.parametrize('which', ['trainer', 'core'])
@pytest.mark.parametrize('state', ['Z', 'X', 'T', 't'])
def test_no_heartbeat_for_dead_or_stopped_process(which, state):
    trainer, core = process(), process(start=200)
    if which == 'trainer':
        trainer = process(state=state)
    else:
        core = process(start=200, state=state)
    with pytest.raises(ValueError, match='exited/stopped'):
        guard().check(trainer, core, 1000, 60)


@pytest.mark.parametrize('trainer,core', [(process(start=101), process(start=200)),
                                         (process(), process(start=201))])
def test_pid_reuse_is_not_training_liveness(trainer, core):
    with pytest.raises(ValueError, match='identity'):
        guard().check(trainer, core, 1000, 60)


def test_cpu_busy_loop_does_not_hide_stalled_training():
    with pytest.raises(ValueError, match='completed round'):
        guard().check(process(cpu=10000), process(start=200), 1000, 1500)


def test_idle_process_does_not_receive_heartbeat():
    with pytest.raises(ValueError, match='CPU'):
        guard().check(process(), process(start=200), 1000, 300)


def test_resume_regression_stops_old_heartbeat():
    with pytest.raises(ValueError, match='regressed'):
        guard().check(process(cpu=20), process(start=200), 900, 60)


def test_stale_log_does_not_get_fresh_grace_period():
    g = Guard(process(), process(start=200), 1000, 0, 1600, 1500, 300)
    with pytest.raises(ValueError, match='completed round'):
        g.check(process(cpu=20), process(start=200), 1000, 1)


def test_partial_log_line_is_not_progress(tmp_path):
    path = tmp_path / 'rank0.log'
    path.write_text('startup\n' + json.dumps({'global_timesteps': 123, 'x/finite_fraction': 1})
                    + '\n{"global_timesteps":124')
    assert read_progress(path) == 123
    path.write_text(json.dumps({'global_timesteps': 124, 'x/finite_fraction': .9}))
    with pytest.raises(ValueError, match='non-finite'):
        read_progress(path)


def test_proc_stat_command_parentheses_do_not_shift_fields(tmp_path):
    (tmp_path / '42').mkdir()
    fields = ['S'] + ['0'] * 21
    fields[11], fields[12], fields[19] = '15', '7', '123'
    (tmp_path / '42' / 'stat').write_text('42 (worker (test)) ' + ' '.join(fields))
    assert read_process(42, tmp_path) == Process(123, 22, 'S')


def test_heartbeat_has_no_explicit_step_or_training_metric():
    from unittest.mock import Mock
    client = object.__new__(Client)
    client.attached = Mock()
    with patch('tools.operations.wandb_heartbeat.time.time', return_value=1234.5):
        client.heartbeat()
    client.attached.log.assert_called_once_with({'ops/heartbeat_unix': 1234.5})
    client.attached.finish.assert_not_called()


@pytest.mark.parametrize('state', ['finished', 'failed', 'killed', 'preempted', 'pending'])
def test_terminal_server_state_is_never_resurrected(state, tmp_path):
    from unittest.mock import Mock
    from tools.operations.wandb_heartbeat import pulse
    client = Mock()
    client.status.return_value = {'state': state}
    with patch('tools.operations.wandb_heartbeat.read_process', side_effect=[process(), process(start=200)]), \
         patch('tools.operations.wandb_heartbeat.read_progress', return_value=1000), \
         patch('tools.operations.wandb_heartbeat.time.monotonic', return_value=1):
        assert pulse(client, guard(), 1, 2, tmp_path)['event'] == 'stop'
    client.heartbeat.assert_not_called()


def test_trainer_exit_during_status_request_prevents_heartbeat(tmp_path):
    from unittest.mock import Mock
    from tools.operations.wandb_heartbeat import pulse
    client = Mock()
    client.status.return_value = {'state': 'running'}
    with patch('tools.operations.wandb_heartbeat.read_process',
               side_effect=[process(), process(start=200), FileNotFoundError()]), \
         patch('tools.operations.wandb_heartbeat.read_progress', return_value=1000), \
         patch('tools.operations.wandb_heartbeat.time.monotonic', return_value=1):
        with pytest.raises(FileNotFoundError):
            pulse(client, guard(), 1, 2, tmp_path)
    client.heartbeat.assert_not_called()


def test_query_failure_does_not_blindly_send_heartbeat(tmp_path):
    from unittest.mock import Mock
    from tools.operations.wandb_heartbeat import pulse
    client = Mock()
    client.status.side_effect = TimeoutError()
    with patch('tools.operations.wandb_heartbeat.read_process', side_effect=[process(), process(start=200)]), \
         patch('tools.operations.wandb_heartbeat.read_progress', return_value=1000), \
         patch('tools.operations.wandb_heartbeat.time.monotonic', return_value=1):
        with pytest.raises(TimeoutError):
            pulse(client, guard(), 1, 2, tmp_path)
    client.heartbeat.assert_not_called()


def test_discover_only_listener_owned_by_expected_core(tmp_path):
    from tools.operations.wandb_heartbeat import core_socket
    (tmp_path / '2' / 'fd').mkdir(parents=True)
    (tmp_path / '2' / 'fd' / '8').symlink_to('socket:[123]')
    (tmp_path / 'net').mkdir()
    table = tmp_path / 'net' / 'unix'
    table.write_text('header\n'
                     '0: 2 0 00010000 0001 01 999 /tmp/wandb-9-9-0/socket\n'
                     '0: 2 0 00010000 0001 01 123 /tmp/wandb-1-2-0/socket\n'
                     '0: 2 0 00000000 0001 03 123 /tmp/wandb-1-2-0/socket\n')
    assert core_socket(1, 2, tmp_path) == '/tmp/wandb-1-2-0/socket'
    table.write_text('header\n0: 2 0 00010000 0001 01 999 /tmp/wandb-1-2-0/socket\n')
    with pytest.raises(ValueError, match='exactly one'):
        core_socket(1, 2, tmp_path)


def test_resume_ignores_old_log_and_partial_new_record(tmp_path):
    log = tmp_path / 'rank0.log'
    old = b'{"global_timesteps":9000}\n'
    log.write_bytes(old + b'initializing\n{"global_timesteps":100')
    assert read_progress(log, len(old), -1) == -1
    with log.open('ab') as stream:
        stream.write(b'0}\n')
    assert read_progress(log, len(old), -1) == 1000
    log.write_text('truncated')
    with pytest.raises(ValueError, match='truncated'):
        read_progress(log, len(old), -1)


def test_startup_can_heartbeat_before_first_completed_round(tmp_path):
    from unittest.mock import Mock
    from tools.operations.wandb_heartbeat import pulse
    log = tmp_path / 'rank0.log'
    log.write_text('initializing\n')
    g = Guard(process(), process(start=200), -1, 0, 0, 1500, 300)
    client = Mock()
    client.status.return_value = {'state': 'running'}
    with patch('tools.operations.wandb_heartbeat.read_process',
               side_effect=[process(cpu=20), process(start=200)] * 2), \
         patch('tools.operations.wandb_heartbeat.time.monotonic', return_value=60):
        result = pulse(client, g, 1, 2, log, log_start=0, initial_step=-1)
    assert result['event'] == 'heartbeat' and result['step'] == -1
    client.heartbeat.assert_called_once()


def test_core_discovery_is_limited_to_trainer_children(tmp_path):
    from tools.operations.wandb_heartbeat import find_core
    (tmp_path / '1' / 'task' / '1').mkdir(parents=True)
    (tmp_path / '1' / 'task' / '1' / 'children').write_text('2 3 4')
    for pid, command in [(2, b'python\0worker\0'), (3, b'/bin/wandb-core\0--pid\099\0'),
                         (4, b'/bin/wandb-core\0--pid\x001\0')]:
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / 'cmdline').write_bytes(command)
    assert find_core(1, tmp_path) == 4


def test_auto_waits_for_initialization_then_attaches_once():
    from tools.operations.wandb_heartbeat import wait_for_client
    with patch('tools.operations.wandb_heartbeat.read_process', return_value=process()), \
         patch('tools.operations.wandb_heartbeat.find_core', side_effect=[None, 2, 2]), \
         patch('tools.operations.wandb_heartbeat.core_socket'), \
         patch('tools.operations.wandb_heartbeat.time.monotonic', side_effect=[0, 2, 4]), \
         patch('tools.operations.wandb_heartbeat.time.sleep'), \
         patch('tools.operations.wandb_heartbeat.Client', side_effect=[RuntimeError('not initialized'), 'client']) as client:
        assert wait_for_client(1, 100, 'run', 'project', 10) == ('client', 2)
    assert client.call_count == 2
    assert all(call.kwargs == {'wandb_id': 'run', 'project': 'project'} for call in client.call_args_list)


def test_auto_stops_when_trainer_exits_before_wandb():
    from tools.operations.wandb_heartbeat import wait_for_client
    with patch('tools.operations.wandb_heartbeat.read_process', return_value=process(state='Z')), \
         patch('tools.operations.wandb_heartbeat.time.monotonic', return_value=0), \
         patch('tools.operations.wandb_heartbeat.Client') as client:
        with pytest.raises(ValueError, match='exited or changed'):
            wait_for_client(1, 100, 'run', 'project', 10)
    client.assert_not_called()


def test_auto_wait_has_a_deadline():
    from tools.operations.wandb_heartbeat import wait_for_client
    with patch('tools.operations.wandb_heartbeat.time.monotonic', return_value=11), \
         patch('tools.operations.wandb_heartbeat.Client') as client:
        with pytest.raises(ValueError, match='deadline'):
            wait_for_client(1, 100, 'run', 'project', 10)
    client.assert_not_called()


def test_duplicate_helper_lock_releases_on_exit_and_new_launch_is_independent(tmp_path):
    from tools.operations.wandb_heartbeat import acquire_lock
    log = tmp_path / 'rank0.log'
    first = acquire_lock(log, 1, 100)
    try:
        with pytest.raises(ValueError, match='already owns'):
            acquire_lock(log, 1, 100)
        # Same PID reused by a new trainer must not inherit the old lock.
        acquire_lock(log, 1, 200).close()
    finally:
        first.close()
    acquire_lock(log, 1, 100).close()
