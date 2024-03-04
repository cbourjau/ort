use crate::protos::{
    tensor_proto::DataType,
    tensor_shape_proto::{dimension, Dimension},
    type_proto, TensorShapeProto, TypeProto, ValueInfoProto,
};
use protobuf::Enum;

use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum ValueInfo {
    Tensor(TensorInfo),
    // Only allow sequence of Tensors. Everything else is madness
    Sequence(TensorInfo),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub shape: Vec<Dim>,
    pub dtype: Dtype,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    Unknown,
    Dynamic(String),
    Fixed(usize),
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub enum Dtype {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    String,
    Bool,
}

impl From<Dim> for Dimension {
    fn from(dim: Dim) -> Self {
        Dimension {
            value: match dim {
                Dim::Unknown => None,
                Dim::Fixed(v) => Some(dimension::Value::DimValue(v as _)),
                Dim::Dynamic(v) => Some(dimension::Value::DimParam(v)),
            },
            ..Default::default()
        }
    }
}

impl TryFrom<Dimension> for Dim {
    type Error = Error;

    fn try_from(proto: Dimension) -> Result<Self, Self::Error> {
        Ok(match proto.value {
            Some(dimension::Value::DimValue(v)) => Dim::Fixed(v as _),
            Some(dimension::Value::DimParam(v)) => Dim::Dynamic(v),
            None => Dim::Unknown,
        })
    }
}

impl TryFrom<ValueInfoProto> for ValueInfo {
    type Error = Error;

    fn try_from(proto: ValueInfoProto) -> Result<Self, Self::Error> {
        let typro = proto.type_.into_option();
        if let Some(TypeProto { value: Some(v), .. }) = typro {
            Ok(match v {
                type_proto::Value::TensorType(t) => ValueInfo::Tensor(t.try_into()?),
                _ => todo!(),
            })
        } else {
            Err(Error::new_validation(format!(
                "Invalid TypeProto: `{:?}`",
                typro
            )))
        }
    }
}

impl From<TensorInfo> for type_proto::Value {
    fn from(t: TensorInfo) -> Self {
        let data_type: DataType = t.dtype.into();
        type_proto::Value::TensorType(crate::protos::type_proto::Tensor {
            elem_type: data_type as _,
            shape: Some(TensorShapeProto {
                dim: t.shape.into_iter().map(|dim| dim.into()).collect(),
                ..Default::default()
            })
            .into(),
            ..Default::default()
        })
    }
}

impl TryFrom<type_proto::Tensor> for TensorInfo {
    type Error = Error;

    fn try_from(
        type_proto::Tensor {
            elem_type, shape, ..
        }: type_proto::Tensor,
    ) -> Result<Self, Self::Error> {
        let shape = shape
            .into_option()
            .ok_or_else(|| Error::new_validation("Incomplete dimenson data.".into()))?
            .dim
            .into_iter()
            .map(|el| el.try_into())
            .collect::<Result<_, Self::Error>>()?;

        let dtype = elem_type.try_into()?;
        Ok(TensorInfo { shape, dtype })
    }
}

impl TryFrom<i32> for Dtype {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        use DataType::*;
        let value = DataType::from_i32(value);
        Ok(match value {
            Some(UINT8) => Self::U8,
            Some(UINT16) => Self::U16,
            Some(UINT32) => Self::U32,
            Some(UINT64) => Self::U64,
            Some(INT8) => Self::I8,
            Some(INT16) => Self::I16,
            Some(INT32) => Self::I32,
            Some(INT64) => Self::I64,
            Some(FLOAT) => Self::F32,
            Some(DOUBLE) => Self::F64,
            Some(BOOL) => Self::Bool,
            Some(STRING) => Self::String,
            _ => {
                return Err(Error::new_validation(format!(
                    "Unsupported data type `{:?}`",
                    value
                )))
            }
        })
    }
}

impl From<Dtype> for DataType {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::U8 => DataType::UINT8,
            Dtype::U16 => DataType::UINT16,
            Dtype::U32 => DataType::UINT32,
            Dtype::U64 => DataType::UINT64,
            Dtype::I8 => DataType::INT8,
            Dtype::I16 => DataType::INT16,
            Dtype::I32 => DataType::INT32,
            Dtype::I64 => DataType::INT64,
            Dtype::F32 => DataType::FLOAT,
            Dtype::F64 => DataType::DOUBLE,
            Dtype::String => DataType::STRING,
            Dtype::Bool => DataType::BOOL,
        }
    }
}

impl ValueInfo {
    pub fn value_info_proto(self, name: String) -> ValueInfoProto {
        ValueInfoProto {
            name,
            type_: Some(TypeProto {
                value: match self {
                    Self::Tensor(t) => Some(t.into()),
                    Self::Sequence(_s) => todo!(),
                },
                ..Default::default()
            })
            .into(),
            doc_string: String::new(),
            ..Default::default()
        }
    }
}
