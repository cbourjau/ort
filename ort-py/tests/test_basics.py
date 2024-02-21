import numpy as np
from spox import build, Tensor, argument
import spox.opset.ai.onnx.v18 as op
import onnx
import pytest

from onnxrt import Session


@pytest.fixture
def add_model():
    a = argument(Tensor(np.float64, ("N",)))
    b = op.add(a, a)
    return build({"a": a}, {"b": b})
    

def test_basics_session_from_file(tmp_path, add_model: onnx.ModelProto):
    path = tmp_path / "model.onnx"
    onnx.save(add_model, path)

    sess = Session(path=path)


def test_basics_session_from_bytes(add_model: onnx.ModelProto):
    sess = Session(model_proto=add_model.SerializeToString())


def test_basics_session_from_protobuf_object(add_model: onnx.ModelProto):
    sess = Session(model_proto=add_model)


def test_run(add_model):
    sess = Session(model_proto=add_model)
    breakpoint()
    sess.run({"a": np.array([1, 2], np.float64), "b": np.array([1, 2], np.float64)})
