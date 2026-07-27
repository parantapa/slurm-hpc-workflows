"""Common utilities."""

from __future__ import annotations

import signal
import socket
import random
import string
import logging
import subprocess
from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import contextmanager, closing

import ifaddr


def gen_random_string(k: int = 32) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


def gen_error_id() -> str:
    return "ERROR_" + gen_random_string()


@dataclass
class RemoteExecutionError:
    error: str
    error_id: str


@contextmanager
def ignoring_sigint():
    """SIGINT is ignored inside this context manager."""
    handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, handler)


def terminate_gracefully(
    proc: subprocess.Popen, timeout: int = 5, proc_name: str = "process"
):
    """Terminal a process gracefully."""
    if proc.poll() is None:
        print(f"Terminating {proc_name} ...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            print(f"Killing {proc_name} ...", flush=True)
            proc.kill()


class Closeable(ABC):
    """Base class for objects that require cleanup."""

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        exc_type, exc_val, exc_tb = exc_type, exc_val, exc_tb
        self.close()

    def __del__(self):
        self.close()


def data_address(interface: str | None, default: str = "0.0.0.0") -> str:
    """Find the network address for data transfer."""
    adapters = [a.name for a in ifaddr.get_adapters()]
    if interface is None:
        if "ib0" in adapters:
            interface = "ib0"
    if interface is None:
        return default
    a = [a for a in ifaddr.get_adapters() if a.name == interface][0]
    ip = [ip for ip in a.ips if ip.is_IPv4][0].ip
    assert isinstance(ip, str)
    return ip


def arbitrary_free_port(host: str) -> int:
    """Request a free port from OS."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


LOG_FORMAT: str = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
LOG_LEVEL = logging.INFO
