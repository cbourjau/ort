use std::collections::HashMap;
use std::path::PathBuf;

use ort::{IntoValue, Session, Tensor, TypeInfo, Value};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::{npyffi::NPY_TYPES, PyArray, PyReadonlyArrayDyn, PyUntypedArray};
use pyo3::types::PyDict;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _onnxrt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<PySession>()?;
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

        let inputs: HashMap<String, PyValue> = inputs.extract()?;
        let inputs = inputs.into_iter().map(|(k, v)| (k, v.0)).collect();

        let outputs = self.session.run(inputs, None).unwrap();

        // Allocate outputs on the Python side
        let out_dict = PyDict::new(py);

        for (k, value) in outputs.into_iter() {
            let pyobj = match value {
                Value::Tensor(tensor) => match tensor {
                    Tensor::F64(data) => {
                        let arr = data.array_view();
                        // from_array copies data into Python heap
                        PyArray::from_array(py, &arr).to_object(py)
                    }
                    Tensor::F32(data) => {
                        let arr = data.array_view();
                        PyArray::from_array(py, &arr).to_object(py)
                    }
                },
            };

            out_dict.set_item(k, pyobj)?;
        }

        Ok(out_dict)
    }

    fn get_input_type_infos(&self) -> Vec<(&str, PyTypeInfo)> {
        self.session
            .get_input_infos()
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k, PyTypeInfo(v)))
            .collect()
    }
}

struct PyValue(Value);

impl<'source> FromPyObject<'source> for PyValue {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        // Get the GIL marker.
        let dtype = ob.extract::<&PyUntypedArray>()?.dtype();

        use NPY_TYPES::*;

        Ok(match i32_as_numpy_type(dtype.num()) {
            NPY_FLOAT => {
                let arr = ob.extract::<PyReadonlyArrayDyn<f32>>()?;
                PyValue(arr.as_array().to_owned().into_value().unwrap())
            }
            NPY_DOUBLE => {
                let arr = ob.extract::<PyReadonlyArrayDyn<f64>>()?;
                PyValue(arr.as_array().to_owned().into_value().unwrap())
            }
            npy_type => panic!("Unsupported NumPy type: '{:?}'", npy_type),
        })
    }
}

fn i32_as_numpy_type(num: i32) -> NPY_TYPES {
    use NPY_TYPES::*;

    match num {
        0 => NPY_BOOL,
        1 => NPY_BYTE,
        2 => NPY_UBYTE,
        3 => NPY_SHORT,
        4 => NPY_USHORT,
        5 => NPY_INT,
        6 => NPY_UINT,
        7 => NPY_LONG,
        8 => NPY_ULONG,
        9 => NPY_LONGLONG,
        10 => NPY_ULONGLONG,
        11 => NPY_FLOAT,
        12 => NPY_DOUBLE,
        13 => NPY_LONGDOUBLE,
        14 => NPY_CFLOAT,
        15 => NPY_CDOUBLE,
        16 => NPY_CLONGDOUBLE,
        17 => NPY_OBJECT,
        18 => NPY_STRING,
        19 => NPY_UNICODE,
        20 => NPY_VOID,
        21 => NPY_DATETIME,
        22 => NPY_TIMEDELTA,
        23 => NPY_HALF,
        24 => NPY_NTYPES,
        25 => NPY_NOTYPE,
        26 => NPY_CHAR,
        256 => NPY_USERDEF,
        num => panic!("Unrecognized NumPy type id {}", num),
    }
}

#[pyclass]
struct PyTypeInfo(TypeInfo);

#[pymethods]
impl PyTypeInfo {
    fn __str__(&self) -> String {
        match self.0 {
            TypeInfo::Tensor(ref t) => {
                format!("Tensor(dtype: {:?}, shape: {:?})", t.dtype, t.shape)
            }
        }
    }
}
