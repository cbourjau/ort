mod api;
mod error;
mod session;
mod tensor_dtype;
mod type_info;
mod value;

pub use crate::session::Session;
pub use crate::type_info::{TensorInfo, TypeInfo};
pub use crate::value::{IntoValue, Tensor, Value};

pub const API_VERSION: u32 = 16;

pub(crate) use crate::{api::Api, error::ErrorStatus, tensor_dtype::TensorDataType};

// TODO: this should not be pub
pub struct Wrapper<T> {
    ptr: *mut T,
    destructor: unsafe extern "C" fn(*mut T) -> (),
}

impl<T> Drop for Wrapper<T> {
    fn drop(&mut self) {
        unsafe { (self.destructor)(self.ptr) }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::write;

    use ndarray::array;
    use onnx_protos::{
        Dim, Dtype, Graph, Input, Model, Node, Operation, Output, TensorInfo, ValueInfo,
    };
    use tempfile::NamedTempFile;

    use super::*;

    fn make_info(rank: usize, dtype: Dtype) -> ValueInfo {
        ValueInfo::Tensor(TensorInfo {
            shape: vec![Dim::Unknown; rank],
            dtype,
        })
    }

    fn make_abs_model() -> Model {
        Model {
            opsets: [("ai.onnx".to_string(), 19)].into(),
            graph: Graph {
                name: "graph".to_string(),
                inputs: vec![Input {
                    name: "a".into(),
                    info: make_info(1, Dtype::F32),
                }],
                outputs: vec![Output {
                    name: "b".into(),
                    info: make_info(1, Dtype::F32),
                }],
                nodes: vec![Node {
                    name: "abs".into(),
                    inputs: vec!["a".into()],
                    outputs: vec!["b".into()],
                    operation: Operation {
                        name: "Abs".into(),
                        domain: "ai.onnx".into(),
                    },
                    attributes: HashMap::new(),
                    doc_string: None,
                }],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn session_from_path() {
        let f = NamedTempFile::new().unwrap();
        let model = make_abs_model().into_bytes();

        write(f.path(), model).unwrap();

        Session::from_path(f.path()).unwrap();
    }

    #[test]
    fn abs_low_level() {
        let api = Api::new();
        let model = make_abs_model().into_bytes();

        let sess = Session::from_bytes(model).unwrap();
        let rt_opts = api.create_run_options().unwrap();

        let val = array![-1.0f32, -2.0].into_dyn().into_value().unwrap();

        let input = val.into_ort_value();

        let out = sess
            .run_ort_values(&[("a".to_string(), input)].into(), rt_opts)
            .unwrap();

        let out = unsafe {
            api.get_tensor_data_mut::<f32>(out.get("b").unwrap().ptr)
                .unwrap()
        };

        let expected = vec![1.0, 2.0];
        let candidate = unsafe { std::slice::from_raw_parts(out, 2) };

        assert_eq!(expected, candidate);
    }

    #[test]
    fn abs_higher_level() {
        let model = make_abs_model().into_bytes();

        let sess = Session::from_bytes(model).unwrap();
        let input = array![-1.0f32, -2.0].into_dyn().into_value().unwrap();

        let mut out = sess.run([("a".to_string(), input)].into(), None).unwrap();

        assert_eq!(out.len(), 1);

        if let Value::Tensor(Tensor::F32(data)) = out.remove("b").unwrap() {
            let expected = array![1.0, 2.0].into_dyn();
            assert_eq!(data.array_view(), expected);
        } else {
            panic!("Expected `F32` output.")
        }
    }
}
