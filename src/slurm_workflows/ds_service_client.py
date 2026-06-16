"""Ds Service Client."""

import os
from contextlib import contextmanager

from .ds_service_pb2 import *
from .ds_service_pb2_grpc import *

GRPC_CLIENT_OPTIONS = [
    ("grpc.keepalive_time_ms", 120 * 1000),
    ("grpc.keepalive_timeout_ms", 30 * 000),
    ("grpc.http2.max_pings_without_data", 5),
    ("grpc.keepalive_permit_without_calls", 1),
]


@contextmanager
def translate_grpc_error():
    try:
        yield
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise KeyError(e.details())
        elif e.code() == grpc.StatusCode.ALREADY_EXISTS:
            raise ValueError(e.details())
        elif e.code() == grpc.StatusCode.UNAVAILABLE:
            raise TimeoutError(e.details())
        else:
            raise


class Client:
    def __init__(self, address: str | None = None):
        if address is None:
            self.address = os.environ["DS_SERVER_ADDRESS"]
        else:
            self.address = address

        self.channel = grpc.insecure_channel(self.address, options=GRPC_CLIENT_OPTIONS)
        self.stub = DsServiceStub(self.channel)

    def close(self):
        self.channel.close()

    def map_set(self, key: str, value: bytes) -> None:
        with translate_grpc_error():
            self.stub.MapSet(MapSetRequest(key=key, value=value))

    def map_get(self, key: str) -> bytes:
        with translate_grpc_error():
            response: MapGetResponse = self.stub.MapGet(MapGetRequest(key=key))
            return response.value

    def task_add(
        self,
        task_id: str,
        queue: str | list[str],
        priority: float,
        function: bytes,
        input: bytes,
    ) -> None:
        if isinstance(queue, str):
            queue = [queue]

        with translate_grpc_error():
            self.stub.TaskAdd(
                TaskAddRequest(
                    task_id=task_id,
                    queue=queue,
                    priority=priority,
                    function=function,
                    input=input,
                )
            )

    def task_status(self, task_id: str) -> TaskStatusResponse:
        with translate_grpc_error():
            return self.stub.TaskStatus(TaskStatusRequest(task_id=task_id))

    def task_get(self, worker_id: str, queue: str | list[str]) -> TaskGetResponse:
        if isinstance(queue, str):
            queue = [queue]

        with translate_grpc_error():
            return self.stub.TaskGet(TaskGetRequest(worker_id=worker_id, queue=queue))

    def task_done(self, task_id: str, output: bytes):
        with translate_grpc_error():
            return self.stub.TaskDone(TaskDoneRequest(task_id=task_id, output=output))

    def requeue(self, timeout_s: float):
        with translate_grpc_error():
            return self.stub.Requeue(RequeueRequest(timeout_s=timeout_s))
