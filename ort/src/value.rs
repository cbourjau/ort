use std::marker::PhantomData;

use crate::{api::Api, error::ErrorStatus, TensorDataType, Wrapper};
use ndarray::{ArrayD, ArrayViewD};
use ort_sys::OrtValue;

/// A struct that works for inputs and output. The goal is to not make
/// a unnecessary copies from the Rust to the Python heap and vice
/// versa.
pub enum Value {
    Tensor(Tensor),
}

pub enum Tensor {
    F64(Data<f64>),
    F32(Data<f32>),
    String(Data<String>),
}

pub struct Data<T> {
    ort_value: Wrapper<OrtValue>,
    shape: Vec<usize>,
    phantom_type: PhantomData<T>,
}

impl Value {
    pub(crate) fn ref_ort_value(&self) -> &Wrapper<OrtValue> {
        match self {
            Value::Tensor(Tensor::F64(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::F32(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::String(ref data)) => &data.ort_value,
        }
    }
}

impl<T> Data<T> {
    pub fn array_view(&self) -> ArrayViewD<'_, T> {
        let api = Api::new();

        unsafe {
            ndarray::ArrayView::from_shape_ptr(
                dbg!(self.shape.clone()),
                api.get_tensor_data_mut(self.ort_value.ptr).unwrap(),
            )
        }
    }
}

pub trait IntoValue {
    /// Create a new value with the copy of the data.
    fn into_value(self) -> Result<Value, ErrorStatus>;
}

impl<T> IntoValue for ArrayD<T>
where
    T: Clone + TensorDataType,
{
    fn into_value(self) -> Result<Value, ErrorStatus> {
        let api = Api::new();

        let arr = self.as_standard_layout();
        let slice = arr.as_slice().unwrap();
        let shape = self.shape();
        let ort_value = api.create_tensor_with_cloned_data(slice, shape)?;

        ort_value.into_value()
    }
}

impl IntoValue for Wrapper<OrtValue> {
    fn into_value(self) -> Result<Value, ErrorStatus> {
        let api = Api::new();
        let val = unsafe {
            let type_info = api.get_type_info_from_ort_value(self.ptr)?;
            let container_type = api.get_onnx_type(type_info.ptr)?;
            match container_type {
                ort_sys::ONNXType_ONNX_TYPE_TENSOR => {
                    let tensor_info = api.type_info_as_tensor_type_info(&*type_info.ptr)?;
                    let onnx_dtype = api.get_tensor_data_type(tensor_info)?;
                    let shape = api
                        .get_tensor_shape(tensor_info)?
                        .into_iter()
                        .map(|el| el as usize)
                        .collect();

                    Value::Tensor(match onnx_dtype {
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
                            Tensor::F64(Data {
                                ort_value: self,
                                shape,
                                phantom_type: PhantomData::<_>,
                            })
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                            Tensor::F32(Data {
                                ort_value: self,
                                shape,
                                phantom_type: PhantomData::<_>,
                            })
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => {
                            Tensor::String(Data {
                                ort_value: self,
                                shape: shape.to_vec(),
                                phantom_type: PhantomData::<String>,
                            })
                        }
                        _ => todo!(),
                    })
                }
                _ => todo!(),
            }
        };
        Ok(val)
    }
}

#[cfg(test)]
mod tests {
    use std::slice;

    use ndarray::array;

    use super::*;

    #[test]
    fn test_roundtrip_array_to_value_to_array() {
        let arr = array![1.0, 2.0].into_dyn();
        let val = arr.clone().into_value().unwrap();

        let api = Api::new();
        let ort_val = val.ref_ort_value();

        let ptr: *const f64 = unsafe { api.get_tensor_data_mut(ort_val.ptr).unwrap() };

        let round_trip = unsafe { slice::from_raw_parts(ptr, 2) };

        assert_eq!(arr.as_slice().unwrap(), round_trip);
    }
}
