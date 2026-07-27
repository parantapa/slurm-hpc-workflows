"""Run ds-service."""

import time
import shlex
import socket
import subprocess

from .utils import (
    Closeable,
    arbitrary_free_port,
    data_address,
    terminate_gracefully,
    ignoring_sigint,
)


READY_POLL_INTERVAL_S: float = 0.01


class DsService(Closeable):
    """Run ds-service server locally"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        server_exe: str = "ds-service",
    ):
        if port is None:
            port = arbitrary_free_port(host)

        self.host = host
        self.port = port
        self.server_exe = server_exe
        self._proc: subprocess.Popen | None = None

    def start(self):
        cmd = self.server_exe + f" --address {self.host}:{self.port}"
        cmd = shlex.split(cmd)
        print("Starting server ...")
        print("executing: ", " ".join(cmd))

        assert self._proc is None
        with ignoring_sigint():
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
            )

        print(f"server address: {self.host}:{self.port}")

    def wait_until_ready(self, timeout: float = 30.0) -> None:
        if self._proc is None:
            raise RuntimeError("Server not started; call start() first")

        # 0.0.0.0 and :: mean "every interface" to bind(); they are not
        # connectable destinations, so probe loopback instead.
        host = self.host
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"

        deadline = time.monotonic() + timeout
        while True:
            returncode = self._proc.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"ds-service exited before becoming ready "
                    f"(returncode={returncode})"
                )

            try:
                with socket.create_connection((host, self.port), timeout=1.0):
                    return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"ds-service at {host}:{self.port} was not ready "
                        f"within {timeout}s"
                    )
                time.sleep(READY_POLL_INTERVAL_S)

    def get_address(self, interface: str | None = None) -> str:
        ip = data_address(interface, self.host)
        return f"{ip}:{self.port}"

    def close(self):
        if self._proc is not None:
            terminate_gracefully(self._proc)
            self._proc = None
