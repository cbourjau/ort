use std::mem::size_of;
use std::path::PathBuf;

use crate::protos::{
    tensor_proto::{DataLocation, DataType},
    type_proto::Value,
    StringStringEntryProto, TensorProto, TypeProto, ValueInfoProto,
};
use ndarray::{Array, ArrayD};

use protobuf::Enum;

use crate::{utils::shape_to_tensor_shape_proto, Error};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub tensor: TensorValue,
    /// Relative path from where the data was read or to which it
    /// will be written.
    pub path: Option<ExternalData>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalData {
    /// POSIX filesystem path relative to the directory where the ONNX
    /// protobuf model was stored.
    location: PathBuf,
    /// Position of byte at which stored data begins.  Offset values
    /// SHOULD be multiples 4096 (page size) to enable mmap support.
    offset: Option<usize>,
    /// Number of bytes containing data.
    length: Option<usize>,
    /// SHA1 digest of file specified in under 'location' key.}
    checksum: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TensorValue {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I64(ArrayD<i64>),
    I32(ArrayD<i32>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    Bool(ArrayD<bool>),
    String(ArrayD<String>),
    // `Unknown` does not have a defined serialization format inside a TensorProto
    // Unknown { shape: Vec<usize>, data: Vec<u8> },
}

impl TensorValue {
    fn shape(&self) -> Vec<usize> {
        use TensorValue::*;

        match self {
            F32(t) => t.shape().to_vec(),
            F64(t) => t.shape().to_vec(),
            I8(t) => t.shape().to_vec(),
            I16(t) => t.shape().to_vec(),
            I64(t) => t.shape().to_vec(),
            I32(t) => t.shape().to_vec(),
            U8(t) => t.shape().to_vec(),
            U16(t) => t.shape().to_vec(),
            U32(t) => t.shape().to_vec(),
            U64(t) => t.shape().to_vec(),
            Bool(t) => t.shape().to_vec(),
            String(t) => t.shape().to_vec(),
        }
    }

    fn dtype(&self) -> DataType {
        use TensorValue::*;

        match self {
            F32(_) => DataType::FLOAT,
            F64(_) => DataType::DOUBLE,
            I8(_) => DataType::INT8,
            I16(_) => DataType::INT16,
            I64(_) => DataType::INT64,
            I32(_) => DataType::INT32,
            U8(_) => DataType::UINT8,
            U16(_) => DataType::UINT16,
            U32(_) => DataType::UINT32,
            U64(_) => DataType::UINT64,
            Bool(_) => DataType::BOOL,
            String(_) => DataType::STRING,
        }
    }
}

impl Tensor {
    pub fn tensor_proto(self, name: String) -> TensorProto {
        let mut tp = TensorProto {
            dims: self
                .tensor
                .shape()
                .into_iter()
                .map(|el| el as i64)
                .collect(),
            data_type: self.tensor.dtype() as _,
            name,
            ..Default::default()
        };

        macro_rules! to_le_bytes {
            ($arr:expr) => {
                $arr.mapv(|el| el.to_le_bytes())
                    .into_iter()
                    .flatten()
                    .collect()
            };
        }

        match self.tensor {
            TensorValue::F32(arr) => tp.float_data = arr.into_raw_vec(),
            TensorValue::F64(arr) => tp.double_data = arr.into_raw_vec(),
            TensorValue::I64(arr) => tp.int64_data = arr.into_raw_vec(),
            TensorValue::I32(arr) => tp.int32_data = arr.into_raw_vec(),
            TensorValue::U64(arr) => tp.uint64_data = arr.into_raw_vec(),
            TensorValue::String(arr) => {
                tp.string_data = arr.mapv(|el| el.into_bytes()).into_raw_vec()
            }

            TensorValue::I8(arr) => tp.raw_data = to_le_bytes!(arr),
            TensorValue::I16(arr) => tp.raw_data = to_le_bytes!(arr),
            TensorValue::U8(arr) => tp.raw_data = to_le_bytes!(arr),
            TensorValue::U16(arr) => tp.raw_data = to_le_bytes!(arr),
            TensorValue::U32(arr) => tp.raw_data = to_le_bytes!(arr),
            TensorValue::Bool(arr) => {
                tp.raw_data = arr
                    .mapv(|el| (el as u8).to_le_bytes())
                    .into_iter()
                    .flatten()
                    .collect()
            }
        }
        tp
    }

    pub fn value_info_proto(&self, name: String) -> ValueInfoProto {
        // dentoation is optional (and IMHO pointless/harmful)
        // https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition
        let type_denotation = "".to_string();

        let shape = self.tensor.shape();
        ValueInfoProto {
            name,
            type_: Some(TypeProto {
                denotation: type_denotation,
                value: Some(Value::TensorType(crate::protos::type_proto::Tensor {
                    elem_type: self.tensor.dtype() as _,
                    shape: Some(shape_to_tensor_shape_proto(shape.as_slice())).into(),
                    ..Default::default()
                })),
                ..Default::default()
            })
            .into(),
            doc_string: String::new(),
            ..Default::default()
        }
    }
}

impl TryFrom<TensorProto> for Tensor {
    type Error = Error;

    fn try_from(proto: TensorProto) -> Result<Self, Self::Error> {
        macro_rules! read_raw {
            ($ty:ty) => {{
                assert_eq!(proto.raw_data.len() % size_of::<$ty>(), 0);
                proto
                    .raw_data
                    .chunks_exact(size_of::<$ty>())
                    .map(|slice| slice.try_into().unwrap())
                    .map(<$ty>::from_le_bytes)
            }};
        }

        let shape: Vec<_> = proto.dims.iter().map(|v| *v as usize).collect();

        macro_rules! match_arm {
            ($var:tt, $expl_field:expr, $rust_ty:ty) => {{
                let explicit_field = &$expl_field;
                let vec: Vec<_> = if !explicit_field.is_empty() {
                    explicit_field.clone()
                } else {
                    read_raw!($rust_ty).collect::<Vec<_>>()
                };
                TensorValue::$var(
                    Array::from_vec(vec)
                        .into_dyn()
                        .into_shape(shape)
                        .unwrap()
                        .into(),
                )
            }};
        }

        let data_type = DataType::from_i32(proto.data_type).ok_or_else(|| {
            Error::new_validation(format!(
                "Failed to parse tensor data type: `{}`",
                proto.data_type
            ))
        })?;
        let tval = match dbg!(data_type) {
            DataType::FLOAT => match_arm!(F32, proto.float_data, f32),
            DataType::DOUBLE => match_arm!(F64, proto.double_data, f64),

            DataType::INT8 => match_arm!(I8, vec![], i8),
            DataType::INT16 => match_arm!(I16, vec![], i16),
            DataType::INT32 => match_arm!(I32, proto.int32_data, i32),
            DataType::INT64 => match_arm!(I64, proto.int64_data, i64),

            DataType::UINT8 => match_arm!(U8, vec![], u8),
            DataType::UINT16 => match_arm!(U16, vec![], u16),
            DataType::UINT32 => match_arm!(U32, vec![], u32),
            DataType::UINT64 => match_arm!(U64, proto.uint64_data, u64),

            DataType::BOOL => {
                let explicit_field = &proto.int32_data;
                let vec: Vec<_> = if !explicit_field.is_empty() {
                    explicit_field.iter().map(|&el| el != 0).collect()
                } else {
                    read_raw!(u8).map(|v| v > 0).collect()
                };
                TensorValue::Bool(Array::from_vec(vec).into_dyn().into_shape(shape).unwrap())
            }
            DataType::STRING => {
                let explicit_field = &proto.string_data;
                // Strings are always in the explicit field (from proto3 source)
                let vec: Vec<_> = explicit_field
                    .clone()
                    .into_iter()
                    .map(|u8_vec| String::from_utf8(u8_vec).unwrap())
                    .collect();
                TensorValue::String(Array::from_vec(vec).into_dyn().into_shape(shape).unwrap())
            }
            v => {
                return Err(Error::new_validation(format!(
                    "Unsupported DataType: `{:?}`",
                    v
                )))
            }
        };
        let path = match proto
            .data_location
            .enum_value()
            .map_err(|v| Error::new_validation(format!("Unexpected data location id: {}", v)))?
        {
            DataLocation::DEFAULT => None,
            DataLocation::EXTERNAL => Some(proto.external_data.try_into()?),
        };
        Ok(Tensor { tensor: tval, path })
    }
}

impl TryFrom<Vec<StringStringEntryProto>> for ExternalData {
    type Error = Error;

    fn try_from(protos: Vec<StringStringEntryProto>) -> Result<Self, Self::Error> {
        let mut location = None;
        let mut offset = None;
        let mut length = None;
        let mut checksum = None;
        for StringStringEntryProto { key, value, .. } in protos.into_iter() {
            match key.as_str() {
                "location" => location = Some(value),
                "offset" => {
                    offset = Some(value.parse().map_err(|e| {
                        Error::new_validation(format!("Failed to parse offset: {:?}", e))
                    })?);
                }
                "length" => {
                    length = Some(value.parse().map_err(|e| {
                        Error::new_validation(format!("Failed to parse offset: {:?}", e))
                    })?)
                }
                "checksum" => checksum = Some(value.into_bytes()),
                _ => Err(Error::new_validation(format!(
                    "Unexpected location key: {}",
                    key
                )))?,
            }
        }

        Ok(Self {
            location: location
                .ok_or_else(|| Error::new_validation("No data location specified.".into()))?
                .into(),
            offset,
            length,
            checksum,
        })
    }
}
