from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TaskState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Ready: _ClassVar[TaskState]
    Running: _ClassVar[TaskState]
    Complete: _ClassVar[TaskState]
Ready: TaskState
Running: TaskState
Complete: TaskState

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MapSetRequest(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: bytes
    def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...

class MapGetRequest(_message.Message):
    __slots__ = ("key",)
    KEY_FIELD_NUMBER: _ClassVar[int]
    key: str
    def __init__(self, key: _Optional[str] = ...) -> None: ...

class MapGetResponse(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: bytes
    def __init__(self, value: _Optional[bytes] = ...) -> None: ...

class TaskAddRequest(_message.Message):
    __slots__ = ("task_id", "queue", "priority", "function", "input")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    queue: _containers.RepeatedScalarFieldContainer[str]
    priority: float
    function: bytes
    input: bytes
    def __init__(self, task_id: _Optional[str] = ..., queue: _Optional[_Iterable[str]] = ..., priority: _Optional[float] = ..., function: _Optional[bytes] = ..., input: _Optional[bytes] = ...) -> None: ...

class TaskStatusRequest(_message.Message):
    __slots__ = ("task_id",)
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    def __init__(self, task_id: _Optional[str] = ...) -> None: ...

class TaskStatusResponse(_message.Message):
    __slots__ = ("state", "output")
    STATE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    state: TaskState
    output: bytes
    def __init__(self, state: _Optional[_Union[TaskState, str]] = ..., output: _Optional[bytes] = ...) -> None: ...

class TaskGetRequest(_message.Message):
    __slots__ = ("worker_id", "queue")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    QUEUE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    queue: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, worker_id: _Optional[str] = ..., queue: _Optional[_Iterable[str]] = ...) -> None: ...

class TaskGetResponse(_message.Message):
    __slots__ = ("task_id", "function", "input")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    function: bytes
    input: bytes
    def __init__(self, task_id: _Optional[str] = ..., function: _Optional[bytes] = ..., input: _Optional[bytes] = ...) -> None: ...

class TaskDoneRequest(_message.Message):
    __slots__ = ("task_id", "output")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    output: bytes
    def __init__(self, task_id: _Optional[str] = ..., output: _Optional[bytes] = ...) -> None: ...

class RequeueRequest(_message.Message):
    __slots__ = ("timeout_s",)
    TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    timeout_s: float
    def __init__(self, timeout_s: _Optional[float] = ...) -> None: ...
