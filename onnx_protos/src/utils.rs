use crate::protos::{
    tensor_shape_proto::{dimension, Dimension},
    TensorShapeProto,
};

pub fn shape_to_dimensions(shape: &[usize]) -> Vec<Dimension> {
    shape
        .iter()
        .map(|dim| Dimension {
            value: Some(dimension::Value::DimValue(*dim as _)),
            ..Default::default()
        })
        .collect()
}

pub fn shape_to_tensor_shape_proto(shape: &[usize]) -> TensorShapeProto {
    TensorShapeProto {
        dim: shape_to_dimensions(shape),
        ..Default::default()
    }
}

pub fn if_not_empty(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}
