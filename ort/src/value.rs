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
    U8(Data<u8>),
    U16(Data<u16>),
    U32(Data<u32>),
    U64(Data<u64>),

    I8(Data<i8>),
    I16(Data<i16>),
    I32(Data<i32>),
    I64(Data<i64>),

    F64(Data<f64>),
    F32(Data<f32>),

    Bool(Data<bool>),

    String(Data<String>),
}

pub struct Data<T> {
    ort_value: Wrapper<OrtValue>,
    shape: Vec<usize>,
    phantom_type: PhantomData<T>,
}

/// Container holding a contiguous buffer of `str`s (NOT null terminated) and their respective offsets.
pub struct StringContainer {
    contiguous_buffer: Vec<u8>,
    byte_offsets: Vec<usize>,
    shape: Vec<usize>,
}

impl Value {
    pub(crate) fn ref_ort_value(&self) -> &Wrapper<OrtValue> {
        match self {
            Value::Tensor(Tensor::U8(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::U16(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::U32(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::U64(ref data)) => &data.ort_value,

            Value::Tensor(Tensor::I8(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::I16(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::I32(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::I64(ref data)) => &data.ort_value,

            Value::Tensor(Tensor::F64(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::F32(ref data)) => &data.ort_value,

            Value::Tensor(Tensor::Bool(ref data)) => &data.ort_value,
            Value::Tensor(Tensor::String(ref data)) => &data.ort_value,
        }
    }
}

impl<T> Data<T>
where
    T: Copy,
{
    pub fn array_view(&self) -> ArrayViewD<'_, T> {
        let api = Api::new();

        unsafe {
            ndarray::ArrayView::from_shape_ptr(
                self.shape.clone(),
                api.get_tensor_data_mut(self.ort_value.ptr).unwrap(),
            )
        }
    }
}

impl<String> Data<String> {
    pub fn str_container(&self) -> StringContainer {
        let api = Api::new();

        let n_elements = self.shape.iter().product();

        let (contiguous_buffer, byte_offsets) = unsafe {
            api.get_string_tensor_buffer(self.ort_value.ptr, n_elements)
                .unwrap()
        };

        StringContainer {
            contiguous_buffer,
            byte_offsets,
            shape: self.shape.clone(),
        }
    }
}

impl StringContainer {
    pub fn array(&self) -> ArrayD<&str> {
        let starts = self.byte_offsets.iter();
        let stops = self
            .byte_offsets
            .iter()
            .skip(1)
            .cloned()
            .chain(std::iter::once(self.contiguous_buffer.len()));

        let strs: Vec<&str> = starts
            .zip(stops)
            .map(|(&start, stop)| {
                std::str::from_utf8(&self.contiguous_buffer[start..stop]).unwrap()
            })
            .collect();

        ArrayD::from_shape_vec(self.shape.clone(), strs).unwrap()
    }
}

pub trait IntoValue {
    /// Create a new value with the copy of the data.
    fn into_value(self) -> Result<Value, ErrorStatus>;
}

impl<'a, T> IntoValue for ArrayViewD<'a, T>
where
    T: Copy + TensorDataType,
{
    fn into_value(self) -> Result<Value, ErrorStatus> {
        let api = Api::new();

        let arr = self.as_standard_layout();
        let slice = arr.as_slice().unwrap();
        let shape = self.shape();
        let ort_value = api.create_tensor_with_copied_data(slice, shape)?;

        ort_value.into_value()
    }
}

impl<'b> IntoValue for ArrayD<&'b str> {
    fn into_value(self) -> Result<Value, ErrorStatus> {
        let api = Api::new();

        let arr = self.as_standard_layout();
        let slice = arr.as_slice().unwrap();
        let shape = self.shape();
        let ort_value = api.create_string_tensor(slice, shape)?;

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

                    macro_rules! make_data {
                        () => {
                            Data {
                                ort_value: self,
                                shape,
                                phantom_type: PhantomData::<_>,
                            }
                        };
                    }

                    Value::Tensor(match onnx_dtype {
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => {
                            Tensor::U8(make_data!())
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => {
                            Tensor::U16(make_data!())
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => {
                            Tensor::U32(make_data!())
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => {
                            Tensor::U64(make_data!())
                        }

                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
                            Tensor::I8(make_data!())
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
                            Tensor::I16(make_data!())
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
                            Tensor::I32(make_data!())
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
                            Tensor::I64(make_data!())
                        }

                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
                            Tensor::F64(make_data!())
                        }
                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                            Tensor::F32(make_data!())
                        }

                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
                            Tensor::Bool(make_data!())
                        }

                        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => {
                            Tensor::String(make_data!())
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
        let val = arr.view().into_value().unwrap();

        let api = Api::new();
        let ort_val = val.ref_ort_value();

        let ptr: *const f64 = unsafe { api.get_tensor_data_mut(ort_val.ptr).unwrap() };

        let round_trip = unsafe { slice::from_raw_parts(ptr, 2) };

        assert_eq!(arr.as_slice().unwrap(), round_trip);
    }
}
