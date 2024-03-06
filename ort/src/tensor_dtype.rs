use ort_sys::{
    ONNXTensorElementDataType, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
};

pub trait TensorDataType {
    fn tensor_dtype() -> ONNXTensorElementDataType;
}
macro_rules! impl_tensor_dtype {
    ($ty:ty, $val:expr) => {
        impl TensorDataType for $ty {
            fn tensor_dtype() -> ONNXTensorElementDataType {
                $val
            }
        }
    };
}

impl_tensor_dtype!(
    f32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
);
impl_tensor_dtype!(
    f64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
);
impl_tensor_dtype!(
    i8,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
);
impl_tensor_dtype!(
    i16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
);
impl_tensor_dtype!(
    i32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
);
impl_tensor_dtype!(
    i64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
);
impl_tensor_dtype!(
    u8,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
);
impl_tensor_dtype!(
    u16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
);
impl_tensor_dtype!(
    u32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
);
impl_tensor_dtype!(
    u64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
);
impl_tensor_dtype!(
    bool,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
);
impl_tensor_dtype!(
    String,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
);
