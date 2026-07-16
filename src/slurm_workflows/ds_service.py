"""Run ds-service."""

import shlex
import subprocess

from .utils import (
    Closeable,
    terminate_gracefully,
    ignoring_sigint,
)


class DsService(Closeable):
    """Run ds-service server locally"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5051,
        server_exe: str = "ds-service",
    ):
        self.address = f"{host}:{port}"
        self.server_exe = server_exe
        self._proc: subprocess.Popen | None = None

    def start(self):
        cmd = self.server_exe + f" --address {self.address}"
        cmd = shlex.split(cmd)
        print("Starting server ...")
        print("executing: ", " ".join(cmd))

        assert self._proc is None
        with ignoring_sigint():
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
            )

        print(f"server address: {self.address}")

    def close(self):
        if self._proc is not None:
            terminate_gracefully(self._proc)
            self._proc = None
