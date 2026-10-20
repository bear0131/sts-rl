"""Process identities shared by bounded real-game supervisors and audits.

This module neither launches nor signals processes. Callers may signal only a
Process returned by matching_process(), whose kill() also checks PID reuse.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import psutil


def _pid(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Process identity requires a positive integer PID")
    return value


def _timestamp(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Process identity requires an actual finite timestamp")
    return float(value)


def capture_process_identity(process: psutil.Process) -> dict[str, Any]:
    """Capture a live incarnation; missing birth time is never guessed.

    Descriptive metadata denied by the OS is explicitly null with a reason.
    If the process disappears or is reused while reading, discard this capture.
    """
    pid = _pid(process.pid)
    created = _timestamp(process.create_time())
    if not process.is_running():
        raise psutil.NoSuchProcess(pid)
    identity: dict[str, Any] = dict(version=1, pid=pid, create_time=created)
    errors = {}
    for field, getter in (
        ("parent_pid", process.ppid), ("name", process.name), ("exe", process.exe),
    ):
        try:
            identity[field] = getter()
        except psutil.AccessDenied:
            identity[field] = None
            errors[field] = "AccessDenied"
    if not process.is_running():
        raise psutil.NoSuchProcess(pid)
    identity["observed_at"] = time.time()
    identity["metadata_errors"] = errors
    return identity


def matching_process(identity: Mapping[str, Any]) -> psutil.Process | None:
    """Return this exact live incarnation, not another occupant of its PID.

    AccessDenied is unknown, not exited. Invalid or legacy records are rejected;
    they cannot authorize cleanup. Exact timestamps survive JSON round trips.
    """
    if type(identity.get("version")) is not int or identity["version"] != 1:
        raise ValueError("Missing or unsupported process identity version")
    pid = _pid(identity["pid"])
    created = _timestamp(identity["create_time"])
    try:
        process = psutil.Process(pid)
        if process.create_time() != created or not process.is_running():
            return None
        return process
    except psutil.NoSuchProcess:
        return None


def inspect_legacy_pid(pid: int, *, observed_before: float) -> dict[str, Any]:
    """Read-only assessment of an old numeric record, never a cleanup handle.

    observed_before must bound when the original PID had already been observed,
    not an invented creation time. A newer birth proves reuse; an older birth
    does NOT prove ownership. In particular, final status mtime is too weak to
    resolve reuse during a run. Unverifiable results require provenance review.
    """
    pid = _pid(pid)
    bound = _timestamp(observed_before)
    try:
        current = capture_process_identity(psutil.Process(pid))
    except psutil.NoSuchProcess:
        return dict(pid=pid, status="not_present", observed_before=bound)
    status = "reused" if current["create_time"] > bound else "unverifiable"
    return dict(pid=pid, status=status, observed_before=bound, current=current)
