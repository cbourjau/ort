use std::collections::HashMap;
use std::path::PathBuf;

use ort::{self, Session, Value};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pyo3::types::PyDict;

mod py_value;

use py_value::PyValue;

/// A Python module implemented in Rust.
#[pymodule]
fn _onnxrt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySession>()?;
    m.add_class::<TypeInfo>()?;
    Ok(())
}

#[pyclass(unsendable)]
pub struct PySession {
    session: Session,
}

#[pymethods]
impl PySession {
    #[new]
    #[pyo3(signature = (*, path=None, model_proto=None))]
    fn new(path: Option<PathBuf>, model_proto: Option<Vec<u8>>) -> PyResult<Self> {
        if let Some(path) = path {
            let session = Session::from_path(path.as_path()).unwrap();
            Ok(Self { session })
        } else if let Some(bytes) = model_proto {
            let session = Session::from_bytes(bytes).unwrap();
            Ok(Self { session })
        } else {
            Err(PyValueError::new_err(
                "Exactly one of `path` or `model_proto` must be set.",
            ))
        }
    }

    fn run<'py>(&self, inputs: &'py PyDict) -> PyResult<&'py PyDict> {
        let py = inputs.py();

        // TODO: Avoid copy of inputs if possible?
        let inputs: HashMap<String, PyValue> = inputs.extract()?;
        let inputs: HashMap<&str, &Value> =
            inputs.iter().map(|(k, v)| (k.as_str(), &v.0)).collect();

        let outputs = py.allow_threads(|| self.session.run(inputs, None).unwrap());

        // Allocate outputs on the Python side
        let out_dict = PyDict::new(py);

        for (k, value) in outputs.into_iter() {
            let pyobj = PyValue(value).to_object(py);

            out_dict.set_item(k, pyobj)?;
        }

        Ok(out_dict)
    }

    fn get_input_type_infos(&self) -> Vec<(&str, TypeInfo)> {
        self.session
            .get_input_infos()
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k, TypeInfo(v)))
            .collect()
    }

    fn get_output_type_infos(&self) -> Vec<(&str, TypeInfo)> {
        self.session
            .get_output_infos()
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k, TypeInfo(v)))
            .collect()
    }

    fn get_model_metadata(&self) -> HashMap<String, String> {
        self.session.get_model_metadata().unwrap()
    }
}

#[pyclass]
struct TypeInfo(ort::TypeInfo);

#[pymethods]
impl TypeInfo {
    fn __str__(&self) -> String {
        match self.0 {
            ort::TypeInfo::Tensor(ref t) => {
                format!("Tensor(dtype: {:?}, shape: {:?})", t.dtype, t.shape)
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("TypeInfo.{}", self.__str__())
    }
}
