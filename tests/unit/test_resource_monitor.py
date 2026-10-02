"""Unavailable device sensors are not zero; changing power sources stay visible."""
from copy import deepcopy
import sys
import threading
from types import SimpleNamespace

import pytest

from rl.training.resource_monitor import ResourceMonitor


@pytest.mark.parametrize('device_row', [
    '0, 42, 13, 1024, 8192, 210, 405, 4.13, 51, P8',
    '0, [N/A], [N/A], 1024, 8192',
])
def test_power_and_clock_telemetry_preserves_missing_values(monkeypatch, device_row):
    import rl.training.resource_monitor as monitor_module
    monkeypatch.setattr(monitor_module.shutil, 'which', lambda _: 'nvidia-smi')
    monkeypatch.setattr(monitor_module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=device_row))
    plugged = False
    monkeypatch.setattr(monitor_module.psutil, 'sensors_battery', lambda: SimpleNamespace(power_plugged=plugged))
    with ResourceMonitor(interval=60, gpu=True) as monitor:
        collection = monitor.snapshot()
        saved_collection = deepcopy(collection)
        plugged = True
    assert collection == saved_collection
    assert collection['power_plugged_observed'] == [False]
    collection['gpu_devices'][0]['graphics_clock_mhz_mean'] = -1
    assert monitor.summary['power_plugged_observed'] == [False, True]
    device = monitor.summary['gpu_devices'][0]
    if '[N/A]' in device_row:
        assert device['gpu_util_percent_mean'] is None
        assert device['graphics_clock_mhz_mean'] is None
        assert device['power_watts_mean'] is None
        assert device['performance_states_observed'] is None
    else:
        assert device['graphics_clock_mhz_min'] == device['graphics_clock_mhz_max'] == 210
        assert device['power_watts_mean'] == 4.13
        assert device['performance_states_observed'] == ['P8']
    assert not monitor.summary['errors']


def test_absent_power_sensor_stays_null(monkeypatch):
    import rl.training.resource_monitor as monitor_module
    monkeypatch.setattr(monitor_module.psutil, 'sensors_battery', lambda: None)
    with ResourceMonitor(interval=60) as monitor:
        pass
    assert monitor.summary['power_plugged_observed'] is None
    assert monitor.summary['gpu_devices'] is None


def test_snapshot_counts_cpu_once_and_keeps_sampling(monkeypatch):
    import rl.training.resource_monitor as monitor_module
    cpu = {'parent': 10.0, 'child': 3.0}
    child = SimpleNamespace(
        pid=2, create_time=lambda: 2,
        cpu_times=lambda: SimpleNamespace(user=cpu['child'], system=0.0),
        memory_info=lambda: SimpleNamespace(rss=30),
    )
    parent = SimpleNamespace(
        pid=1, create_time=lambda: 1, children=lambda **_: [child],
        cpu_times=lambda: SimpleNamespace(user=cpu['parent'], system=0.0),
        memory_info=lambda: SimpleNamespace(rss=70),
    )
    monkeypatch.setattr(monitor_module.psutil, 'Process', lambda: parent)
    with ResourceMonitor(interval=60) as monitor:
        thread = monitor._thread
        cpu.update(parent=12.0, child=4.0)
        collection = monitor.snapshot()
        assert collection['sample_count'] == 2
        assert collection['tree_cpu_seconds'] == 3.0
        assert collection['tree_rss_peak_sampled_bytes'] == 100
        assert collection['elapsed_seconds'] > 0
        assert collection['tree_cpu_percent_mean'] == pytest.approx(
            300.0 / collection['elapsed_seconds'])
        assert thread.is_alive()
        saved = deepcopy(collection)
        cpu.update(parent=15.0, child=6.0)
        later = monitor.snapshot()
        assert monitor._thread is thread
        assert later['tree_cpu_seconds'] == 8.0
        assert later['sample_count'] == 3
    assert collection == saved
    assert monitor.summary['sample_count'] == 4
    assert monitor.summary['tree_cpu_seconds'] == 8.0
    assert monitor.summary['elapsed_seconds'] >= later['elapsed_seconds']
    assert not thread.is_alive()
    later['notes'].clear()
    assert monitor.summary['notes']


def test_context_exception_stops_monitor_thread():
    monitor = ResourceMonitor(interval=60)
    with pytest.raises(ValueError, match='training failed'):
        with monitor:
            monitor.snapshot()
            raise ValueError('training failed')
    assert not monitor._thread.is_alive()
    assert monitor.summary['sample_count'] == 3
    with pytest.raises(RuntimeError, match='active'):
        monitor.snapshot()


def test_snapshot_serializes_with_background_sampling(monkeypatch):
    import rl.training.resource_monitor as monitor_module
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    memory = monitor_module.psutil.virtual_memory
    monitor = ResourceMonitor(interval=0.01)
    snapshots, errors = [], []

    def blocked_memory_sample():
        if threading.current_thread() is monitor._thread:
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test did not release background sample')
        return memory()

    def snapshot():
        try:
            snapshots.append(monitor.snapshot())
        except BaseException as exc:
            errors.append(exc)
        finally:
            completed.set()

    monkeypatch.setattr(monitor_module.psutil, 'virtual_memory', blocked_memory_sample)
    with monitor:
        try:
            assert entered.wait(2)
            reader = threading.Thread(target=snapshot)
            reader.start()
            assert not completed.wait(0.03)
        finally:
            release.set()
        reader.join(timeout=2)
        assert completed.is_set()
        assert not errors
        assert len(snapshots) == 1
        assert snapshots[0]['sample_count'] >= 3
    assert not monitor._thread.is_alive()


@pytest.fixture
def torch_allocator(monkeypatch):
    import torch

    state = SimpleNamespace(initialized=True, device=2, allocated=100, reserved=200,
                            current_calls=[], stats_calls=[], sampled=threading.Event())

    def forbidden(*args, **kwargs):
        pytest.fail('telemetry must not initialize, switch, synchronize, or reset CUDA')

    def current_device():
        state.current_calls.append(threading.current_thread())
        assert threading.current_thread() is threading.main_thread()
        return state.device

    def memory_stats(device):
        state.stats_calls.append((threading.current_thread(), device))
        if threading.current_thread() is not threading.main_thread():
            state.sampled.set()
        return {'allocated_bytes.all.current': state.allocated,
                'reserved_bytes.all.current': state.reserved,
                # The monitor must not mislabel lifetime peaks as its own peak.
                'allocated_bytes.all.peak': 9999, 'reserved_bytes.all.peak': 99999}

    state.cuda = SimpleNamespace(is_initialized=lambda: state.initialized,
        current_device=current_device, memory_stats=memory_stats,
        init=forbidden, _lazy_init=forbidden, set_device=forbidden,
        device=forbidden, synchronize=forbidden, reset_peak_memory_stats=forbidden)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(device=torch.device, cuda=state.cuda))
    return state


@pytest.mark.parametrize('device', ['cuda', 'cuda:2'])
def test_torch_sampling_pins_device_and_reports_sampled_peaks(torch_allocator, device):
    state = torch_allocator
    with ResourceMonitor(interval=.01, torch_device=device) as monitor:
        assert monitor.samples[0]['torch_allocated_bytes'] == 100
        state.device = 7  # Changing the caller's default must not retarget samples.
        state.allocated, state.reserved = 300, 600
        assert state.sampled.wait(2), 'the real background sampler must query the allocator'
        high = monitor.snapshot()
        state.allocated, state.reserved = 0, 0
    assert state.current_calls == ([threading.main_thread()] if device == 'cuda' else [])
    assert state.stats_calls and all(index == 2 for _, index in state.stats_calls)
    assert any(thread is monitor._thread for thread, _ in state.stats_calls)
    assert high['torch_allocated_bytes'] == high['torch_allocated_peak_sampled_bytes'] == 300
    assert high['torch_reserved_bytes'] == high['torch_reserved_peak_sampled_bytes'] == 600
    assert monitor.summary['torch_device'] == 'cuda:2'
    assert monitor.summary['torch_allocated_bytes'] == monitor.summary['torch_reserved_bytes'] == 0
    assert monitor.summary['torch_allocated_peak_sampled_bytes'] == 300
    assert monitor.summary['torch_reserved_peak_sampled_bytes'] == 600
    assert monitor.summary['torch_memory_unavailable_reason'] is None


@pytest.mark.parametrize('device,initialized,reason', [
    (None, True, 'not requested'),
    ('cpu', True, 'requires a CUDA device'),
    ('cuda', False, 'not initialized'),
    ('cuda:2', False, 'not initialized'),
])
def test_unavailable_torch_sampling_never_initializes_cuda(torch_allocator, device, initialized, reason):
    torch_allocator.initialized = initialized
    with ResourceMonitor(interval=60, torch_device=device) as monitor:
        pass
    assert not torch_allocator.current_calls and not torch_allocator.stats_calls
    for name in ('allocated', 'reserved'):
        assert monitor.summary[f'torch_{name}_bytes'] is None
        assert monitor.summary[f'torch_{name}_peak_sampled_bytes'] is None
    assert reason in monitor.summary['torch_memory_unavailable_reason']
    assert not monitor.summary['errors']


@pytest.mark.parametrize('stats', [{}, {'allocated_bytes.all.current': 0}, 'query-error'])
def test_missing_torch_allocator_values_stay_null(torch_allocator, stats):
    def memory_stats(device):
        assert device == 2
        if stats == 'query-error':
            raise RuntimeError('allocator unavailable')
        return stats

    torch_allocator.cuda.memory_stats = memory_stats
    with ResourceMonitor(interval=60, torch_device='cuda:2') as monitor:
        pass
    assert monitor.summary['torch_reserved_bytes'] is None
    assert monitor.summary['torch_reserved_peak_sampled_bytes'] is None
    expected = 0 if isinstance(stats, dict) and stats else None
    assert monitor.summary['torch_allocated_bytes'] == expected
    assert monitor.summary['torch_allocated_peak_sampled_bytes'] == expected
    assert monitor.summary['torch_memory_unavailable_reason']


def test_missing_torch_module_reports_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    with ResourceMonitor(interval=60, torch_device='cuda:2') as monitor:
        pass
    assert monitor.summary['torch_allocated_bytes'] is None
    assert monitor.summary['torch_reserved_bytes'] is None
    assert 'Torch allocator sampling unavailable' in monitor.summary['torch_memory_unavailable_reason']
