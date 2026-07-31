"""Common utilities."""

from __future__ import annotations

import random
import string
import logging
from dataclasses import dataclass


def gen_random_string(k: int = 32) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


def gen_error_id() -> str:
    return "ERROR_" + gen_random_string()


@dataclass
class RemoteExecutionError:
    error: str
    error_id: str


LOG_FORMAT: str = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
LOG_LEVEL = logging.INFO
