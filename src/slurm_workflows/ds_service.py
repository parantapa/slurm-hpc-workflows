"""Run ds-service."""

import shlex
import subprocess

from .utils import (
    Closeable,
    arbitrary_free_port,
    data_address,
    terminate_gracefully,
    ignoring_sigint,
)


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

    def get_address(self, interface: str | None = None) -> str:
        ip = data_address(interface, self.host)
        return f"{ip}:{self.port}"

    def close(self):
        if self._proc is not None:
            terminate_gracefully(self._proc)
            self._proc = None
