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
    def __init__(self, *, interval: float = 0.5, gpu: bool = False, torch_device=None,
                 thread_details: bool = False):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('resource sample interval must be positive')
        self.interval = interval
        self.gpu = gpu
        self._requested_torch_device = torch_device
        self._torch_cuda = None
        self._torch_device_index = None
        self._torch_device = None
        self._torch_memory_reason = 'Torch allocator sampling not requested'
        self.samples = []
        self.errors = set()
        self._stop = threading.Event()
        self._sample_lock = threading.RLock()
        self._previous_cpu = {}
        self.thread_details = thread_details
        self._process_cpu = {}
        self._previous_threads = {}
        self._thread_cpu = {}
        self._thread = None
        self._active = False
        self._smi = shutil.which("nvidia-smi") if gpu else None
        if gpu and self._smi is None:
            self.errors.add("nvidia-smi unavailable; device metrics are null")

    def _prepare_torch_sampling(self):
        # Resolve CUDA's thread-local default on the main thread once. Sampling
        # must not initialize CUDA, switch devices, synchronize, or reset peaks.
        if self._requested_torch_device is None:
            return
        try:
            import torch
            device = torch.device(self._requested_torch_device)
            self._torch_device = str(device)
            if device.type != 'cuda':
                self._torch_memory_reason = 'Torch allocator sampling requires a CUDA device'
                return
            if not torch.cuda.is_initialized():
                self._torch_memory_reason = 'Torch CUDA is not initialized'
                return
            index = device.index
            if index is None:
                if threading.current_thread() is not threading.main_thread():
                    self._torch_memory_reason = 'CUDA device index must be resolved on the main thread'
                    return
                index = torch.cuda.current_device()
            self._torch_device_index = index
            self._torch_device = f'cuda:{index}'
            self._torch_cuda = torch.cuda
            self._torch_memory_reason = None
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            self._torch_memory_reason = f'{type(exc).__name__}: Torch allocator sampling unavailable'

    def _torch_memory_sample(self):
        result = dict(torch_device=self._torch_device, torch_allocated_bytes=None,
                      torch_reserved_bytes=None,
                      torch_memory_unavailable_reason=self._torch_memory_reason)
        if self._torch_cuda is None:
            return result
        try:
            stats = self._torch_cuda.memory_stats(self._torch_device_index)
            for name in ('allocated', 'reserved'):
                # memory_allocated/reserved return zero for missing stats;
                # absence must stay distinguishable from an empty allocator.
                result[f'torch_{name}_bytes'] = stats.get(f'{name}_bytes.all.current')
            if any(result[f'torch_{name}_bytes'] is None for name in ('allocated', 'reserved')):
                result['torch_memory_unavailable_reason'] = 'Torch allocator statistics are missing'
        except (OSError, RuntimeError, ValueError) as exc:
            result['torch_memory_unavailable_reason'] = f'{type(exc).__name__}: Torch allocator query failed'
        return result

    def _sample(self):
        with self._sample_lock:
            self._sample_locked()

    def _sample_locked(self):
        parent = psutil.Process()
        cpu_delta, rss = 0.0, 0
        for process in [parent, *parent.children(recursive=True)]:
            try:
                key = (process.pid, process.create_time())
                times = process.cpu_times()
                total = times.user + times.system
                delta = max(0.0, total - self._previous_cpu.get(key, total))
                cpu_delta += delta
                self._previous_cpu[key] = total
                self._process_cpu[key] = self._process_cpu.get(key, 0.0) + delta
                rss += process.memory_info().rss
                if self.thread_details:
                    for thread in process.threads():
                        identity = (*key, thread.id)
                        used = thread.user_time + thread.system_time
                        increment = max(0., used-self._previous_threads.get(identity, used))
                        self._thread_cpu[identity] = self._thread_cpu.get(identity, 0.) + increment
                        self._previous_threads[identity] = used
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.errors.add("some process samples unavailable")
        row = {"elapsed_seconds": time.perf_counter() - self.started,
               "cpu_seconds_delta": cpu_delta, "tree_rss_bytes": rss,
               "system_available_bytes": psutil.virtual_memory().available,
               "cgroup_memory_bytes": self._cgroup_number('memory.current'),
               "gpu_devices": None, "power_plugged": None}
        row.update(self._torch_memory_sample())
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
        with self._sample_lock:
            if self._thread is not None:
                raise RuntimeError('ResourceMonitor instances cannot be reused')
            self._prepare_torch_sampling()
            self.started = time.perf_counter()
            self._sample_locked()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._active = True
            try:
                self._thread.start()
            except BaseException:
                self._active = False
                self._stop.set()
                raise
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join()
        with self._sample_lock:
            try:
                self._sample_locked()
                self.summary = self._summarize_locked()
            finally:
                self._active = False

    def snapshot(self):
        """Sample now and summarize elapsed work without stopping the monitor.

        The returned nested values are independent of later samples and the
        final summary. Sampling is serialized with the background thread so
        process CPU deltas are counted only once.
        """
        with self._sample_lock:
            if not self._active or self._stop.is_set():
                raise RuntimeError('snapshot requires an active ResourceMonitor')
            self._sample_locked()
            return self._summarize_locked()

    def _summarize_locked(self):
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
        cpu_seconds = sum(s["cpu_seconds_delta"] for s in self.samples)
        torch_memory = {key: self.samples[-1][key] for key in (
            'torch_device', 'torch_allocated_bytes', 'torch_reserved_bytes',
            'torch_memory_unavailable_reason')}
        for name in ('allocated', 'reserved'):
            values = [s[f'torch_{name}_bytes'] for s in self.samples
                      if s[f'torch_{name}_bytes'] is not None]
            torch_memory[f'torch_{name}_peak_sampled_bytes'] = max(values) if values else None
        return {
            **torch_memory,
            "sample_count": len(self.samples),
            "interval_seconds": self.interval,
            "elapsed_seconds": elapsed,
            "tree_cpu_seconds": cpu_seconds,
            "tree_cpu_percent_mean": 100 * cpu_seconds / max(elapsed, 1e-9),
            "process_cpu": [dict(pid=pid, created_at=created, cpu_seconds=seconds,
                cpu_percent_mean=100*seconds/max(elapsed, 1e-9))
                for (pid, created), seconds in sorted(self._process_cpu.items())],
            "thread_cpu": ([dict(pid=pid, created_at=created, tid=tid, cpu_seconds=seconds,
                cpu_percent_mean=100*seconds/max(elapsed, 1e-9))
                for (pid, created, tid), seconds in sorted(self._thread_cpu.items())]
                if self.thread_details else None),
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
                "Per-process/thread percentages use the whole monitored interval, including idle time; thread totals are not added to process totals.",
                "RSS sums may double-count shared pages; sampled peaks are lower bounds.",
                "NVIDIA telemetry is device-wide and can include other processes.",
                "Torch allocator values cover this process and the selected CUDA device; sampled peaks are lower bounds, not allocator high-water marks.",
                "Telemetry sampling has overhead; compare identical intervals and hardware.",
            ],
        }
