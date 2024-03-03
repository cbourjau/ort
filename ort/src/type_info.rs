use std::ffi::CStr;

use ort_sys::OrtTypeInfo;

use crate::{Api, ErrorStatus, Wrapper};

#[derive(Debug, PartialEq)]
pub enum TypeInfo {
    Tensor(TensorInfo),
}

#[derive(Debug, PartialEq)]
pub struct TensorInfo {
    // TODO: Proper enum rather than an integer
    pub dtype: TensorDataType,
    pub shape: Vec<Dim>,
}

#[derive(Debug, PartialEq)]
pub enum Dim {
    Unknown,
    Fixed(usize),
    Dynamic(String),
}

#[derive(Debug, PartialEq)]
pub enum TensorDataType {
    F64,
    F32,
}

impl TypeInfo {
    pub fn new(api: &Api, ort_type_info: &Wrapper<OrtTypeInfo>) -> Result<Self, ErrorStatus> {
        let onnx_type = unsafe { api.get_onnx_type(ort_type_info.ptr)? };

        match onnx_type {
            ort_sys::ONNXType_ONNX_TYPE_TENSOR => unsafe {
                let ort_type_info = &*ort_type_info.ptr;

                let tensor_info = api.type_info_as_tensor_type_info(ort_type_info)?;

                let dtype = {
                    let type_id = api.get_tensor_data_type(tensor_info)?;
                    // TODO: error handling
                    TensorDataType::from_onnx_tensor_element_data_type(type_id).unwrap()
                };
                let fixed_dims = api.get_tensor_shape(tensor_info)?;
                let sym_dims = api.get_tensor_shape_symbolic(tensor_info)?;
                let mut shape = Vec::with_capacity(fixed_dims.len());
                for (fixed, sym) in fixed_dims.into_iter().zip(sym_dims.into_iter()) {
                    let sym = CStr::from_ptr(sym).to_str().unwrap();

                    let dim = if fixed > 0 {
                        Dim::Fixed(fixed as usize)
                    } else if !sym.is_empty() {
                        Dim::Dynamic(sym.to_string())
                    } else {
                        Dim::Unknown
                    };
                    shape.push(dim);
                }
                Ok(Self::Tensor(TensorInfo { dtype, shape }))
            },
            _ => todo!(),
        }
    }
}

impl TensorDataType {
    fn from_onnx_tensor_element_data_type(id: u32) -> Result<Self, ()> {
        use TensorDataType::*;
        Ok(match id {
            ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => F64,
            ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => F32,
            _ => todo!(),
        })
    }
}
