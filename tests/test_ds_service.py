"""Tests for DsService.wait_until_ready.

These drive the real ds-service binary; they skip when it is not installed.
"""

from __future__ import annotations

import sys
import time

import pytest
from ds_service_client import DsServiceClient

from conftest import find_ds_service_exe, free_port
from slurm_workflows import ds_service as ds_service_mod
from slurm_workflows.ds_service import DsService


@pytest.fixture
def probe_spy(monkeypatch) -> list[tuple[str, int]]:
    """Record every address wait_until_ready tries to connect to."""
    seen: list[tuple[str, int]] = []
    real = ds_service_mod.socket.create_connection

    def spy(address, *args, **kwargs):
        seen.append(tuple(address))
        return real(address, *args, **kwargs)

    monkeypatch.setattr(ds_service_mod.socket, "create_connection", spy)
    return seen


@pytest.fixture
def server_exe() -> str:
    exe = find_ds_service_exe()
    if exe is None:
        pytest.skip(
            "ds-service executable not found; set DS_SERVICE_EXE to its path "
            "or put `ds-service` on PATH"
        )
    return exe


@pytest.fixture
def service(server_exe):
    """A DsService bound to loopback on a free port, always cleaned up."""
    svc = DsService(host="127.0.0.1", port=free_port(), server_exe=server_exe)
    try:
        yield svc
    finally:
        svc.close()


class TestWaitUntilReady:
    def test_returns_once_the_server_accepts_connections(self, service):
        service.start()

        service.wait_until_ready(timeout=10)

        # Ready means ready: an RPC must work with no further retrying.
        client = DsServiceClient(f"127.0.0.1:{service.port}")
        try:
            client.task_get_count_by_state()
        finally:
            client.close()

    def test_is_idempotent(self, service):
        service.start()

        service.wait_until_ready(timeout=10)
        service.wait_until_ready(timeout=10)

    def test_returns_promptly(self, service):
        """It must poll, not sleep a fixed amount."""
        service.start()

        started = time.monotonic()
        service.wait_until_ready(timeout=10)

        assert time.monotonic() - started < 2.0

    def test_binding_all_interfaces_probes_loopback(self, server_exe, probe_spy):
        """host='0.0.0.0' is a bind target, not a connectable destination.

        Linux happens to route a connect() to 0.0.0.0 to localhost, so this
        asserts on the address actually probed rather than on success alone.
        """
        svc = DsService(host="0.0.0.0", port=free_port(), server_exe=server_exe)
        try:
            svc.start()
            svc.wait_until_ready(timeout=10)
        finally:
            svc.close()

        assert probe_spy
        assert all(host == "127.0.0.1" for host, _ in probe_spy)

    def test_a_concrete_host_is_probed_as_given(self, service, probe_spy):
        service.start()

        service.wait_until_ready(timeout=10)

        # May be probed more than once: the first attempt can race the bind.
        assert probe_spy
        assert set(probe_spy) == {("127.0.0.1", service.port)}


class TestFailureModes:
    def test_rejects_a_server_that_was_never_started(self, service):
        with pytest.raises(RuntimeError, match="not started"):
            service.wait_until_ready(timeout=1)

    def test_reports_a_server_that_exited(self, server_exe):
        """Fail fast on a dead process instead of waiting out the timeout."""
        svc = DsService(host="127.0.0.1", port=free_port(), server_exe="/bin/false")
        try:
            svc.start()
            started = time.monotonic()
            with pytest.raises(RuntimeError, match="exited before becoming ready"):
                svc.wait_until_ready(timeout=10)
            assert time.monotonic() - started < 5.0
        finally:
            svc.close()

    def test_times_out_when_the_server_never_listens(self, server_exe):
        alive_but_deaf = f"{sys.executable} -c 'import time; time.sleep(60)'"
        svc = DsService(host="127.0.0.1", port=free_port(), server_exe=alive_but_deaf)
        try:
            svc.start()
            with pytest.raises(TimeoutError, match="was not ready within"):
                svc.wait_until_ready(timeout=0.5)
        finally:
            svc.close()
