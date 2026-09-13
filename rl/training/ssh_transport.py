"""Small SSH/SCP transport primitives shared by distributed experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import time


@dataclass(frozen=True)
class Peer:
    rank: int
    host: str
    port: int
    user: str = "root"


def parse_peer(text: str) -> Peer:
    rank_text, target = text.split("=", 1)
    rank = int(rank_text)
    if "@" in target:
        user, host_port = target.split("@", 1)
    else:
        user, host_port = "root", target
    host, port_text = host_port.rsplit(":", 1)
    return Peer(rank=rank, user=user, host=host, port=int(port_text))


def ssh_options(peer: Peer, key: Path, control_dir: Path) -> list[str]:
    return [
        "-i", str(key),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=10",
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=600",
        "-o", f"ControlPath={control_dir}/%C",
    ]


def ssh_cmd(
    peer: Peer, key: Path, control_dir: Path, command: str,
    *, timeout: float | None = None,
) -> None:
    subprocess.run(
        [
            "ssh", *ssh_options(peer, key, control_dir), "-p", str(peer.port),
            f"{peer.user}@{peer.host}", command,
        ],
        check=True,
        timeout=timeout,
    )


def scp_to(
    peer: Peer, key: Path, control_dir: Path, source: Path, destination: str,
    *, timeout: float | None = None,
) -> None:
    subprocess.run(
        [
            "scp", *ssh_options(peer, key, control_dir), "-P", str(peer.port),
            str(source), f"{peer.user}@{peer.host}:{destination}",
        ],
        check=True,
        timeout=timeout,
    )


def _wait_for_file(path: Path, expected_size: int, timeout: float) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if path.exists():
            if path.stat().st_size != expected_size:
                raise RuntimeError(f"unexpected barrier marker size: {path}")
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for barrier marker: {path}")


def all_rank_file_barrier(
    *, tag: str, rank: int, peers: list[Peer], key: Path,
    control_dir: Path, local_root: Path, remote_root: str,
    run_id: str, timeout: float,
) -> None:
    marker = local_root / f"barrier_rank{rank}_{tag}.ready"
    marker.write_bytes(b"ready")
    for peer in peers:
        destination = (
            f"{remote_root}/{run_id}/rank{rank}/{marker.name}"
        )
        temporary = destination + ".tmp"
        scp_to(peer, key, control_dir, marker, temporary, timeout=timeout)
        ssh_cmd(
            peer, key, control_dir,
            f"mv {shlex.quote(temporary)} {shlex.quote(destination)}",
            timeout=timeout,
        )
    peer_markers = [
        Path(remote_root) / run_id / f"rank{peer.rank}"
        / f"barrier_rank{peer.rank}_{tag}.ready"
        for peer in peers
    ]
    for path in peer_markers:
        _wait_for_file(path, expected_size=5, timeout=timeout)
    for path in [marker, *peer_markers]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
