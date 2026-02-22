"""Run a postgres database."""

import shlex
import subprocess
from pathlib import Path

from .utils import (
    Closeable,
    cmd_str,
    data_address,
    find_ds_server,
    terminate_gracefully,
    ignoring_sigint,
)


class DsServer(Closeable):
    """Run a postgres database locally"""

    def __init__(
        self,
        ds_server_exe: Path | str | None = None,
        interface: str | None = None,
        port: int = 5051,
    ):
        host = data_address(interface)

        self.address = f"{host}:{port}"
        self.ds_server_exe = find_ds_server(ds_server_exe)

        self._proc: subprocess.Popen | None = None

    def start(self):
        print("Starting server ...")
        cmd = f"'{self.ds_server_exe!s}' --address {self.address}"
        print("executing:", cmd_str(cmd))
        cmd = shlex.split(cmd)

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
