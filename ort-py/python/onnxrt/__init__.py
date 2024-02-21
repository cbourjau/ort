from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
import numpy as np

from onnxrt._onnxrt import PySession


@runtime_checkable
class ToProtobufBytes(Protocol):
    def SerializeToString(self) -> bytes:
        ...


class Session:
    _py_session = PySession

    def __init__(
        self,
        *,
        path: Path | str | None = None,
        model_proto: ToProtobufBytes | bytes | None = None,
    ):
        if path is None != model_proto is None:
            raise ValueError("Exactly one of `path` or `model_proto` must be set.")
        
        if path is not None:
            self._py_session = PySession(path=Path(path))
        else:
            if isinstance(model_proto, ToProtobufBytes):
                bytes_ = model_proto.SerializeToString()
            else:
                bytes_ = model_proto  # type: ignore
            self._py_session = PySession(model_proto=bytes_)
            

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self._py_session.run(inputs)

    @property
    def input_dtypes(self) -> dict[str, np.dtype]:
        raise NotImplementedError()

    @property
    def output_dtypes(self) -> dict[str, np.dtype]:
        raise NotImplementedError()

    @property
    def input_shapes(self) -> dict[str, tuple[int | str | None, ...]]:
        raise NotImplementedError()

    @property
    def output_shapes(self) -> dict[str, tuple[int | str | None, ...]]:
        raise NotImplementedError()

