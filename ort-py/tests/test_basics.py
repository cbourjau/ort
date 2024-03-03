import numpy as np
from spox import build, Tensor, argument
import spox.opset.ai.onnx.v18 as op
import onnx
import pytest

from onnxrt import Session


@pytest.fixture
def add_model():
    a = argument(Tensor(np.float32, ("N", )))
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


def test_type_infos(add_model):
    sess = Session(model_proto=add_model)

    assert isinstance(sess.input_infos, dict)
    assert {"a"} == sess.input_infos.keys()

    assert isinstance(sess.output_infos, dict)
    assert {"b"} == sess.output_infos.keys()

    
def test_run(add_model):
    sess = Session(model_proto=add_model)
    outputs = sess.run({"a": np.array([1, 2], np.float32)})

    assert {"b"} == outputs.keys()
    np.testing.assert_array_equal(outputs["b"], np.array([2, 4], np.float32))
