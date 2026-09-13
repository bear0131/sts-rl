"""Process-tree RSS/CPU and device-wide NVIDIA telemetry, with explicit units."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time

import psutil


class ResourceMonitor:
    def __init__(self, *, interval: float = 0.5, gpu: bool = False):
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
               "gpu_devices": None}
        if self._smi:
            try:
                result = subprocess.run(
                    [self._smi, "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=True,
                )
                devices = []
                for line in result.stdout.strip().splitlines():
                    cells = line.split(",")
                    def number(i):
                        try:
                            return float(cells[i].strip())
                        except ValueError:
                            return None
                    devices.append(dict(zip(
                        ("index", "gpu_util_percent", "memory_controller_util_percent",
                         "vram_used_mib", "vram_total_mib"),
                        [number(i) for i in range(5)],
                    )))
                row["gpu_devices"] = devices
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.errors.add(type(exc).__name__ + ": device telemetry unavailable")
        self.samples.append(row)

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
        self.summary = {
            "sample_count": len(self.samples),
            "interval_seconds": self.interval,
            "tree_cpu_percent_mean": 100 * sum(s["cpu_seconds_delta"] for s in self.samples) / max(elapsed, 1e-9),
            "tree_rss_peak_sampled_bytes": max(s["tree_rss_bytes"] for s in self.samples),
            "system_available_min_bytes": min(s["system_available_bytes"] for s in self.samples),
            "errors": sorted(self.errors),
            "notes": [
                "100% CPU means one logical core; short-lived children can be undersampled.",
                "RSS sums may double-count shared pages; sampled peaks are lower bounds.",
                "NVIDIA telemetry is device-wide and can include other processes.",
                "Telemetry sampling has overhead; compare identical intervals and hardware.",
            ],
        }
