import numpy as np
from spox import build, Tensor, argument
import spox.opset.ai.onnx.v18 as op
import onnx
import pytest

from onnxrt import Session


def make_add_model(dtype):
    a = argument(Tensor(dtype, ("N",)))
    b = op.add(a, a)
    return build({"a": a}, {"b": b})


def make_identity_model(dtype):
    a = argument(Tensor(dtype, ("N",)))
    b = op.identity(a)
    return build({"a": a}, {"b": b})


@pytest.fixture
def add_model():
    return make_add_model(np.float32)


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


def test_metadata(add_model: onnx.ModelProto):
    # Add some meta data
    entry = add_model.metadata_props.add()
    entry.key = "🦀"
    entry.value = "🚀"

    sess = Session(model_proto=add_model)

    assert {"🦀": "🚀"} == sess.metadata


def test_run(add_model):
    sess = Session(model_proto=add_model)
    outputs = sess.run({"a": np.array([1, 2], np.float32)})

    assert {"b"} == outputs.keys()
    np.testing.assert_array_equal(outputs["b"], np.array([2, 4], np.float32))


def test_string_inputs():
    model = make_identity_model(np.str_)
    sess = Session(model_proto=model)

    exp = np.array(["a", "foo" * 10], np.object_)
    (candidate,) = sess.run({"a": exp}).values()

    np.testing.assert_array_equal(exp, candidate)


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_numeric_and_bool_data_types(dtype):
    model = make_identity_model(dtype)
    sess = Session(model_proto=model)

    inp = np.array([1, 2, 3], dtype)
    candidate = sess.run({"a": inp})

    exp = inp
    assert candidate.keys() == {"b"}
    np.testing.assert_array_equal(exp, candidate["b"])
