"""Unavailable device sensors are not zero; changing power sources stay visible."""
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
        plugged = True
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
