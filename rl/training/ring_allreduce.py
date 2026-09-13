"""Persistent socket ring all-reduce for flat CPU FP32 gradients."""

from __future__ import annotations

import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


class RingAllReduce:
    """Average flat tensors over a unidirectional full-duplex socket ring."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        outgoing: socket.socket,
        incoming: socket.socket,
        timeout: float,
    ) -> None:
        if world_size < 2:
            raise ValueError("ring all-reduce requires at least two ranks")
        self.rank = rank
        self.world_size = world_size
        self.outgoing = outgoing
        self.incoming = incoming
        self.outgoing.settimeout(timeout)
        self.incoming.settimeout(timeout)
        self._sender = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"ring-send-{rank}"
        )

    @staticmethod
    def _byte_view(array: np.ndarray) -> memoryview:
        return memoryview(array).cast("B")

    @staticmethod
    def _recv_exact(sock: socket.socket, target: np.ndarray) -> None:
        view = RingAllReduce._byte_view(target)
        received = 0
        while received < len(view):
            count = sock.recv_into(view[received:])
            if count == 0:
                raise ConnectionError("ring peer closed the connection")
            received += count

    def _exchange(self, send: np.ndarray, receive: np.ndarray) -> None:
        future = self._sender.submit(
            self.outgoing.sendall, self._byte_view(send)
        )
        self._recv_exact(self.incoming, receive)
        future.result()

    def average(self, flat: torch.Tensor) -> tuple[torch.Tensor, int]:
        if flat.device.type != "cpu" or flat.dtype != torch.float32:
            raise TypeError("ring input must be a CPU float32 tensor")
        if not flat.is_contiguous():
            raise ValueError("ring input must be contiguous")

        values = flat.numpy()
        chunk_sizes = [
            (values.size + self.world_size - index - 1) // self.world_size
            for index in range(self.world_size)
        ]
        offsets = np.cumsum([0, *chunk_sizes], dtype=np.int64)
        receive_buffer = np.empty(max(chunk_sizes), dtype=np.float32)
        bytes_sent = 0

        # Reduce-scatter. Rank r finishes with reduced chunk (r + 1) % p.
        for step in range(self.world_size - 1):
            send_index = (self.rank - step) % self.world_size
            receive_index = (self.rank - step - 1) % self.world_size
            send = values[offsets[send_index]:offsets[send_index + 1]]
            receive = receive_buffer[:chunk_sizes[receive_index]]
            self._exchange(send, receive)
            values[offsets[receive_index]:offsets[receive_index + 1]] += receive
            bytes_sent += send.nbytes

        # All-gather the reduced chunks back around the ring.
        for step in range(self.world_size - 1):
            send_index = (self.rank - step + 1) % self.world_size
            receive_index = (self.rank - step) % self.world_size
            send = values[offsets[send_index]:offsets[send_index + 1]]
            receive = values[
                offsets[receive_index]:offsets[receive_index + 1]
            ]
            self._exchange(send, receive)
            bytes_sent += send.nbytes

        flat.mul_(1.0 / self.world_size)
        return flat, bytes_sent

    def close(self) -> None:
        self._sender.shutdown(wait=True, cancel_futures=True)
        for sock in (self.outgoing, self.incoming):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()


class SshRingAllReduce(RingAllReduce):
    """Ring connection whose outgoing edge is an OpenSSH local forward."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        next_host: str,
        next_port: int,
        next_user: str,
        ssh_key: Path,
        listen_port: int,
        timeout: float,
        ready_barrier,
    ) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", listen_port))
        server.listen(1)
        server.settimeout(timeout)

        tunnel = None
        outgoing = incoming = None
        try:
            # Every rank must be listening before an SSH forward is exercised.
            ready_barrier()

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                local_port = probe.getsockname()[1]

            deadline = time.perf_counter() + timeout
            last_error = None
            while time.perf_counter() < deadline and outgoing is None:
                tunnel = subprocess.Popen(
                    [
                        "ssh",
                        "-i", str(ssh_key),
                        "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR",
                        "-o", "ConnectTimeout=10",
                        "-o", "ExitOnForwardFailure=yes",
                        "-o", "ServerAliveInterval=30",
                        "-o", "ServerAliveCountMax=3",
                        "-o", "Compression=no",
                        "-N",
                        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{listen_port}",
                        "-p", str(next_port),
                        f"{next_user}@{next_host}",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                attempt_deadline = min(
                    deadline, time.perf_counter() + 15.0
                )
                while time.perf_counter() < attempt_deadline:
                    if tunnel.poll() is not None:
                        last_error = RuntimeError(
                            "SSH ring tunnel exited with code "
                            f"{tunnel.returncode}"
                        )
                        tunnel = None
                        break
                    try:
                        outgoing = socket.create_connection(
                            ("127.0.0.1", local_port), timeout=1.0
                        )
                        break
                    except OSError:
                        time.sleep(0.1)
                if outgoing is not None:
                    break
                if tunnel is not None:
                    tunnel.terminate()
                    try:
                        tunnel.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        tunnel.kill()
                # Public SSH endpoints can briefly reject a burst of rank
                # startup connections (MaxStartups/kex).  Retrying here is
                # safe because the local listener remains open.
                time.sleep(min(1.0, max(0.0, deadline - time.perf_counter())))
            if outgoing is None:
                if last_error is not None:
                    raise last_error
                raise TimeoutError(
                    "timed out connecting to SSH ring tunnel"
                )

            incoming, _address = server.accept()
            for sock in (outgoing, incoming):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except BaseException:
            for sock in (outgoing, incoming):
                if sock is not None:
                    sock.close()
            if tunnel is not None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
            raise
        finally:
            server.close()

        self._tunnel = tunnel
        super().__init__(
            rank=rank,
            world_size=world_size,
            outgoing=outgoing,
            incoming=incoming,
            timeout=timeout,
        )

    def close(self) -> None:
        super().close()
        self._tunnel.terminate()
        try:
            self._tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._tunnel.kill()
            self._tunnel.wait(timeout=5)
