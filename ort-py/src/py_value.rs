use numpy::{npyffi::NPY_TYPES, PyArray, PyArrayDescr, PyReadonlyArrayDyn, PyUntypedArray};

use ort::{self, IntoValue, Tensor, Value};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pyo3::types::PyString;

pub struct PyValue(pub Value);

impl<'source> FromPyObject<'source> for PyValue {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        // Get the GIL marker.
        let dtype = ob.extract::<&PyUntypedArray>()?.dtype();

        use NPY_TYPES::*;

        Ok(match dtype_num_as_npy_type(dtype)? {
            // Unsigned
            NPY_UBYTE => {
                let arr = ob.extract::<PyReadonlyArrayDyn<u8>>()?;
                PyValue::new(arr)?
            }
            NPY_USHORT => {
                let arr = ob.extract::<PyReadonlyArrayDyn<u16>>()?;
                PyValue::new(arr)?
            }
            NPY_UINT => {
                let arr = ob.extract::<PyReadonlyArrayDyn<u32>>()?;
                PyValue::new(arr)?
            }
            NPY_ULONGLONG | NPY_ULONG => {
                let arr = ob.extract::<PyReadonlyArrayDyn<u64>>()?;
                PyValue::new(arr)?
            }

            // Signed
            NPY_BYTE => {
                let arr = ob.extract::<PyReadonlyArrayDyn<i8>>()?;
                PyValue::new(arr)?
            }
            NPY_SHORT => {
                let arr = ob.extract::<PyReadonlyArrayDyn<i16>>()?;
                PyValue::new(arr)?
            }
            NPY_INT => {
                let arr = ob.extract::<PyReadonlyArrayDyn<i32>>()?;
                PyValue::new(arr)?
            }
            NPY_LONGLONG | NPY_LONG => {
                // FIXME: Probably wrong on Windows!
                let arr = ob.extract::<PyReadonlyArrayDyn<i64>>()?;
                PyValue::new(arr)?
            }

            // Floating point
            NPY_FLOAT => {
                let arr = ob.extract::<PyReadonlyArrayDyn<f32>>()?;
                PyValue::new(arr)?
            }
            NPY_DOUBLE => {
                let arr = ob.extract::<PyReadonlyArrayDyn<f64>>()?;
                PyValue::new(arr)?
            }

            // Bool
            NPY_BOOL => {
                let arr = ob.extract::<PyReadonlyArrayDyn<bool>>()?;
                PyValue::new(arr)?
            }

            // Strings
            NPY_OBJECT => {
                let py = ob.py();
                let arr = ob.extract::<PyReadonlyArrayDyn<PyObject>>()?;
                let arr = arr.as_array();

                let arr = arr.map(|el| el.extract::<&PyString>(py).unwrap().to_str().unwrap());

                PyValue(arr.into_value().unwrap())
            }
            num => Err(PyValueError::new_err(format!(
                "Unsupported NumPy data type: '{:?}', (num: '{:?}')",
                dtype, num
            )))?,
        })
    }
}

impl PyValue {
    fn new<T>(arr: PyReadonlyArrayDyn<T>) -> PyResult<Self>
    where
        T: Copy + numpy::Element + ort::TensorDataType,
    {
        let val = arr
            .as_array()
            .into_value()
            .map_err(|err| PyValueError::new_err(format!("{}", err)))?;
        Ok(PyValue(val))
    }
}

impl ToPyObject for PyValue {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        // from_array copies data into Python heap
        match &self.0 {
            Value::Tensor(tensor) => match tensor {
                Tensor::U8(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }
                Tensor::U16(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }
                Tensor::U32(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }
                Tensor::U64(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }

                Tensor::I8(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }
                Tensor::I16(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }
                Tensor::I32(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }
                Tensor::I64(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }

                Tensor::F64(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }
                Tensor::F32(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }

                Tensor::Bool(data) => {
                    let arr = data.array_view();
                    PyArray::from_array(py, &arr).to_object(py)
                }

                Tensor::String(data) => {
                    let container = data.str_container();
                    let arr = container.array();
                    let arr = arr.map(|el| el.to_object(py));
                    PyArray::from_array(py, &arr).to_object(py)
                }
            },
        }
    }
}

fn dtype_num_as_npy_type(dtype: &PyArrayDescr) -> PyResult<NPY_TYPES> {
    use NPY_TYPES::*;

    Ok(match dtype.num() {
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
        _ => Err(PyValueError::new_err(format!(
            "Unrecongnized NumPy dtype: `{}`",
            dtype
        )))?,
    })
}
