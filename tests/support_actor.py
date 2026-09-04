"""Actor classes for the actor tests.

Workers resolve actors with `importlib`,
so these must be importable by name (`support_actor.CounterActor`).
conftest puts this directory on `sys.path`.
"""

from __future__ import annotations

# Instances created in this process, so tests can assert on per-worker state.
INSTANCES: list["CounterActor"] = []


def reset() -> None:
    INSTANCES.clear()


class CounterActor:
    """Keeps state across tasks, and records that it was closed."""

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False
        INSTANCES.append(self)

    def bump(self, n: int = 1) -> int:
        self.calls += n
        return self.calls

    def echo(self, value):
        return value

    def boom(self):
        raise ValueError("actor failure")

    def close(self) -> None:
        self.closed = True


class ConfiguredActor:
    """Keeps whatever constructor arguments it was handed."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def config(self) -> tuple[tuple, dict]:
        return self.args, self.kwargs


class NoCloseActor:
    """Has no close(); exercises the optional-cleanup branch."""

    def ping(self) -> str:
        return "pong"
