"""Optuna storage backed by the ds-service Journal.

Optuna's `JournalStorage` needs a backend
that does exactly two things: append a JSON-serializable record,
and read every record from a given index onward.
ds-service's Journal is an append-only list of byte strings per key
with those two operations,
so the mapping is direct ---
one journal key holds the whole log of one storage,
and an entry's index in that journal *is* its Optuna log number.

Usage::

    from slurm_workflows.optuna_storage import create_optuna_storage

    storage = create_optuna_storage(server_address, prefix="tuning")
    study = optuna.create_study(storage=storage, study_name="tuning")
    study.optimize(objective, n_trials=100)

Every process pointed at the same server and prefix sees the same study,
which is what makes this useful here:
hand the storage (or the study) to `SlurmPilotExecutor.submit`,
and pilot workers on compute nodes run trials against the same journal
the login node reads from.
The backend is picklable for exactly that reason
--- it carries the server address, not the connection,
and reconnects on first use after being unpickled.

Why no lock object, unlike `JournalFileBackend`:
ds-service serializes journal appends and reads server-side,
so an append is atomic and a read returns a consistent prefix
without any client-side critical section.

Optuna is not a dependency of this package;
install it (or `slurm-workflows[optuna]`) before importing this module.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any
from collections.abc import Generator

from ds_service_client import Client
from optuna.storages import JournalStorage
from optuna.storages.journal import BaseJournalBackend

from optuna.storages.journal._base import BaseJournalSnapshot
from .utils import Closeable

log = logging.getLogger(__name__)

DEFAULT_PREFIX: str = "optuna"

# Entries fetched per journal_read RPC.
# Logs are typically a few hundred bytes,
# so this keeps a read well inside the server's message-size limit
# while still replaying a long journal in few round trips.
DEFAULT_READ_CHUNK_SIZE: int = 4096


class DsServiceJournalBackend(BaseJournalBackend, BaseJournalSnapshot, Closeable):
    """Optuna journal backend that keeps its log in a ds-service journal.

    Args:
        server_address: `host:port` of the ds-service server.
            Defaults to the `DS_SERVER_ADDRESS` environment variable.
        prefix: Namespace for the keys this backend owns.
            Two backends with different prefixes on the same server
            are independent storages.
        timeout: Per-RPC deadline in seconds; `None` uses the client default.
        read_chunk_size: Log entries to fetch per read RPC.
    """

    def __init__(
        self,
        server_address: str | None = None,
        prefix: str = DEFAULT_PREFIX,
        timeout: float | None = None,
        read_chunk_size: int = DEFAULT_READ_CHUNK_SIZE,
    ):
        if server_address is None:
            server_address = os.environ.get("DS_SERVER_ADDRESS")
            if server_address is None:
                raise ValueError(
                    "server_address was not given "
                    "and DS_SERVER_ADDRESS is not set in the environment"
                )
        if read_chunk_size < 1:
            raise ValueError(f"read_chunk_size must be >= 1, got {read_chunk_size}")

        # Resolved eagerly rather than left to the client,
        # so that a backend created on the login node
        # carries a concrete address to the compute nodes it is pickled to.
        self.server_address = server_address
        self.prefix = prefix
        self.timeout = timeout
        self.read_chunk_size = read_chunk_size

        self._client: Client | None = None

    @property
    def log_key(self) -> str:
        """Journal key holding the log."""
        return f"{self.prefix}:log"

    @property
    def snapshot_key(self) -> str:
        """Map key holding the replay snapshot."""
        return f"{self.prefix}:snapshot"

    def _get_client(self) -> Client:
        """Connect on first use.

        The grpc channel does not survive pickling; the address does.
        """
        if self._client is None:
            kwargs = {} if self.timeout is None else {"timeout": self.timeout}
            self._client = Client(self.server_address, **kwargs)
        return self._client

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def read_logs(self, log_number_from: int) -> Generator[dict[str, Any]]:
        client = self._get_client()
        key = self.log_key

        # A journal only ever grows,
        # so the size read here delimits a prefix that stays valid
        # even while other processes append behind us.
        size = client.journal_size(key)

        chunk_size = self.read_chunk_size
        start = log_number_from
        while start < size:
            end = min(start + chunk_size, size)
            try:
                entries = client.journal_read(key, start, end)
            except ValueError:
                # The response outgrew the server's message-size limit.
                # Any single entry small enough to have been appended
                # is small enough to read back,
                # so halving always converges.
                if end - start <= 1:
                    raise
                chunk_size = max(1, (end - start) // 2)
                continue

            if not entries:
                break
            for entry in entries:
                yield json.loads(entry)
            start += len(entries)

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        # Each append is atomic on the server,
        # but a multi-log batch is not appended as a unit:
        # a concurrent writer's entry may land between two of these.
        # Optuna writes one log per call, so that case does not arise in practice.
        client = self._get_client()
        key = self.log_key
        for entry in logs:
            client.journal_append(
                key, json.dumps(entry, separators=(",", ":")).encode("utf-8")
            )

    def save_snapshot(self, snapshot: bytes) -> None:
        try:
            self._get_client().map_set(self.snapshot_key, snapshot)
        except ValueError:
            # A snapshot only shortens replay;
            # the journal remains the source of truth.
            # One too large to send is worth a warning,
            # not a dead optimization run.
            log.warning(
                "Optuna snapshot of %d bytes is too large for one RPC; "
                "not storing it. Replay will read the full journal instead.",
                len(snapshot),
            )

    def load_snapshot(self) -> bytes | None:
        try:
            return self._get_client().map_get(self.snapshot_key)
        except KeyError:
            return None

    def close(self) -> None:
        # getattr: __del__ can fire on an instance whose __init__ raised.
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
            self._client = None


def create_optuna_storage(
    server_address: str | None = None,
    prefix: str = DEFAULT_PREFIX,
    timeout: float | None = None,
    read_chunk_size: int = DEFAULT_READ_CHUNK_SIZE,
) -> JournalStorage:
    """Make an Optuna storage whose journal lives in ds-service.

    Pass the result as `optuna.create_study(storage=...)`.
    Arguments are those of `DsServiceJournalBackend`.
    """
    return JournalStorage(
        DsServiceJournalBackend(
            server_address=server_address,
            prefix=prefix,
            timeout=timeout,
            read_chunk_size=read_chunk_size,
        )
    )


def resolve_backend(storage: Any) -> DsServiceJournalBackend:
    """Accept either the backend or the storage wrapping it.

    Samplers in this package take the storage the study is using,
    rather than a server address of their own,
    so that their keys cannot end up on a different server
    or under a different prefix than the study they sample for.
    """
    if isinstance(storage, DsServiceJournalBackend):
        return storage
    if isinstance(storage, JournalStorage):
        backend = storage._backend
        if isinstance(backend, DsServiceJournalBackend):
            return backend
    raise TypeError(
        "storage must be a DsServiceJournalBackend, "
        "or a JournalStorage built on one (as create_optuna_storage returns); "
        f"got {type(storage).__name__}"
    )
