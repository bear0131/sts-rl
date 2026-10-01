"""Process-tree RSS/CPU and device-wide NVIDIA telemetry, with explicit units."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
import math
from pathlib import Path

import psutil


class ResourceMonitor:
    def __init__(self, *, interval: float = 0.5, gpu: bool = False):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('resource sample interval must be positive')
        self.interval = interval
        self.gpu = gpu
        self.samples = []
        self.errors = set()
        self._stop = threading.Event()
        self._previous_cpu = {}
        self._thread = None
        self._smi = shutil.which("nvidia-smi") if gpu else None
        if gpu and self._smi is None:
            self.errors.add("nvidia-smi unavailable; device metrics are null")

    def _sample(self):
        parent = psutil.Process()
        cpu_delta, rss = 0.0, 0
        for process in [parent, *parent.children(recursive=True)]:
            try:
                key = (process.pid, process.create_time())
                times = process.cpu_times()
                total = times.user + times.system
                cpu_delta += max(0.0, total - self._previous_cpu.get(key, total))
                self._previous_cpu[key] = total
                rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.errors.add("some process samples unavailable")
        row = {"elapsed_seconds": time.perf_counter() - self.started,
               "cpu_seconds_delta": cpu_delta, "tree_rss_bytes": rss,
               "system_available_bytes": psutil.virtual_memory().available,
               "cgroup_memory_bytes": self._cgroup_number('memory.current'),
               "gpu_devices": None, "power_plugged": None}
        try:
            battery = getattr(psutil, 'sensors_battery', lambda: None)()
            if battery is not None:
                row['power_plugged'] = battery.power_plugged
        except (OSError, NotImplementedError):
            self.errors.add('battery power source unavailable')
        if self._smi:
            try:
                result = subprocess.run(
                    [self._smi, "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,pstate",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=True,
                )
                devices = []
                for line in result.stdout.strip().splitlines():
                    cells = line.split(",")
                    def number(i):
                        try:
                            value = float(cells[i].strip())
                            return value if math.isfinite(value) else None
                        except (ValueError, IndexError):
                            return None
                    device = dict(zip(
                        ("index", "gpu_util_percent", "memory_controller_util_percent",
                         "vram_used_mib", "vram_total_mib", "graphics_clock_mhz",
                         "memory_clock_mhz", "power_watts", "temperature_c"),
                        [number(i) for i in range(9)],
                    ))
                    state = cells[9].strip() if len(cells)>9 else ''
                    device['performance_state'] = state if state.startswith('P') and state[1:].isdigit() else None
                    devices.append(device)
                row["gpu_devices"] = devices
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.errors.add(type(exc).__name__ + ": device telemetry unavailable")
        self.samples.append(row)

    @staticmethod
    def _cgroup_number(name):
        try:
            return int((Path('/sys/fs/cgroup') / name).read_text().strip())
        except (OSError,ValueError):
            return None

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.started = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join()
        self._sample()
        elapsed = self.samples[-1]["elapsed_seconds"]
        devices = {}
        for sample in self.samples:
            for device in sample['gpu_devices'] or []:
                devices.setdefault(device['index'],[]).append(device)
        gpu_summary = []
        for index,rows in devices.items():
            utilization = [row['gpu_util_percent'] for row in rows if row['gpu_util_percent'] is not None]
            memory = [row['vram_used_mib'] for row in rows if row['vram_used_mib'] is not None]
            summary = dict(index=index,
                gpu_util_percent_mean=sum(utilization)/len(utilization) if utilization else None,
                vram_peak_sampled_mib=max(memory) if memory else None)
            for key in ('graphics_clock_mhz', 'memory_clock_mhz', 'power_watts', 'temperature_c'):
                values = [row[key] for row in rows if row[key] is not None]
                summary[key+'_mean'] = sum(values)/len(values) if values else None
                summary[key+'_min'] = min(values) if values else None
                summary[key+'_max'] = max(values) if values else None
            summary['performance_states_observed'] = sorted({row['performance_state'] for row in rows
                if row['performance_state'] is not None}) or None
            gpu_summary.append(summary)
        cgroup_memory = [s['cgroup_memory_bytes'] for s in self.samples if s['cgroup_memory_bytes'] is not None]
        try:
            quota,period=(Path('/sys/fs/cgroup')/'cpu.max').read_text().split()
            cpu_quota=float(quota)/float(period)
        except (OSError,ValueError,ZeroDivisionError):
            cpu_quota=None
        self.summary = {
            "sample_count": len(self.samples),
            "interval_seconds": self.interval,
            "tree_cpu_percent_mean": 100 * sum(s["cpu_seconds_delta"] for s in self.samples) / max(elapsed, 1e-9),
            "tree_rss_peak_sampled_bytes": max(s["tree_rss_bytes"] for s in self.samples),
            "system_available_min_bytes": min(s["system_available_bytes"] for s in self.samples),
            "cgroup_memory_peak_sampled_bytes": max(cgroup_memory) if cgroup_memory else None,
            "cgroup_memory_limit_bytes": self._cgroup_number('memory.max'),
            "cgroup_cpu_quota_cores": cpu_quota,
            "gpu_devices": gpu_summary or None,
            "power_plugged_observed": sorted({s['power_plugged'] for s in self.samples
                                              if s['power_plugged'] is not None}) or None,
            "errors": sorted(self.errors),
            "notes": [
                "100% CPU means one logical core; short-lived children can be undersampled.",
                "RSS sums may double-count shared pages; sampled peaks are lower bounds.",
                "NVIDIA telemetry is device-wide and can include other processes.",
                "Telemetry sampling has overhead; compare identical intervals and hardware.",
            ],
        }
