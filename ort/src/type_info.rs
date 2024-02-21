use ort_sys::OrtTypeInfo;

use crate::{Api, ErrorStatus, Wrapper};

pub enum TypeInfo {
    Tensor(TensorInfo),
}

pub struct TensorInfo {
    // TODO: Proper enum rather than an integer
    pub dtype: TensorDataType,
    pub shape: Vec<usize>,
}

#[derive(Debug)]
pub enum TensorDataType {
    F64,
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
                let shape = api.get_tensor_shape(tensor_info)?;
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
            _ => todo!(),
        })
    }
}
