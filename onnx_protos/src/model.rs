use std::collections::HashMap;

use crate::protos::{ModelProto, OperatorSetIdProto, StringStringEntryProto, Version};
use protobuf::{Message, MessageField};

use crate::{utils::if_not_empty, Error, Function, Graph};

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Model {
    /// Domain and version of opsets used in this model.
    pub opsets: HashMap<String, i64>,
    pub producer_name: Option<String>,
    pub producer_version: Option<String>,
    /// Reverse-DNS name indicating the model namespace or domain, for example, 'org.onnx'
    ///
    /// Unclear what this would every do...
    pub domain: Option<String>,
    pub model_version: i64,
    pub doc_string: Option<String>,
    pub graph: Graph,
    pub metadata: HashMap<String, String>,
    pub functions: Vec<Function>,
}

impl Model {
    pub fn into_bytes(self) -> Vec<u8> {
        let proto: ModelProto = self.try_into().unwrap();
        proto.write_to_bytes().unwrap()
    }
}

impl TryFrom<ModelProto> for Model {
    type Error = Error;

    fn try_from(proto: ModelProto) -> Result<Self, Self::Error> {
        Ok(Self {
            opsets: proto
                .opset_import
                .into_iter()
                .map(|el| (el.domain, el.version))
                .collect(),
            producer_name: if_not_empty(proto.producer_name),
            producer_version: if_not_empty(proto.producer_version),
            domain: if_not_empty(proto.domain),
            model_version: proto.model_version,
            doc_string: if_not_empty(proto.doc_string),
            graph: proto
                .graph
                .into_option()
                .ok_or_else(|| Error::new_validation("Model must define a graph.".to_string()))?
                .try_into()?,
            metadata: proto
                .metadata_props
                .into_iter()
                .map(|entry| (entry.key, entry.value))
                .collect(),
            functions: proto
                .functions
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<Vec<_>, Error>>()?,
        })
    }
}

impl TryFrom<Model> for ModelProto {
    type Error = Error;

    fn try_from(model: Model) -> Result<Self, Self::Error> {
        Ok(Self {
            ir_version: Version::IR_VERSION as _,
            opset_import: model
                .opsets
                .into_iter()
                .map(|(domain, version)| OperatorSetIdProto {
                    domain,
                    version,
                    ..Default::default()
                })
                .collect(),
            producer_name: model.producer_name.unwrap_or_default(),
            producer_version: model.producer_version.unwrap_or_default(),
            domain: model.domain.unwrap_or_default(),
            model_version: model.model_version,
            doc_string: model.doc_string.unwrap_or_default(),
            graph: MessageField::some(model.graph.into()),
            metadata_props: model
                .metadata
                .into_iter()
                .map(|(key, value)| StringStringEntryProto {
                    key,
                    value,
                    ..Default::default()
                })
                .collect(),
            training_info: vec![],
            functions: model
                .functions
                .into_iter()
                .map(|f| f.try_into())
                .collect::<Result<_, _>>()?,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{
        edge_info::{Dim, Dtype, TensorInfo},
        graph::{Input, Output},
        Attribute, Node, Operation, Tensor, ValueInfo,
    };

    use super::*;

    use crate::protos::ModelProto;

    #[test]
    fn test_roundtrip() {
        let expected = Model {
            opsets: [("ai.foo".to_string(), 1)].into(),
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: 42,
            doc_string: None,
            graph: Graph {
                name: "graph".to_string(),
                doc_string: None,
                inputs: vec![Input {
                    name: "foo".into(),
                    info: ValueInfo::Tensor(TensorInfo {
                        shape: vec![Dim::Unknown, Dim::Dynamic("N".into()), Dim::Fixed(42)],
                        dtype: Dtype::U8,
                    }),
                }],
                initializers: HashMap::new(),
                outputs: vec![Output {
                    name: "bar".into(),
                    info: ValueInfo::Tensor(TensorInfo {
                        shape: vec![Dim::Unknown, Dim::Dynamic("N".into()), Dim::Fixed(42)],
                        dtype: Dtype::U8,
                    }),
                }],
                nodes: vec![Node {
                    name: "baz".into(),
                    inputs: vec!["foo".into()],
                    outputs: vec!["bar".into()],
                    operation: Operation {
                        name: "DoThings".into(),
                        domain: "ai.foo".into(),
                    },
                    attributes: [
                        ("attr_int".to_string(), Attribute::I64(42)),
                        ("attr_f32s".to_string(), Attribute::F32s(vec![0.0])),
                        (
                            "attr_tensor".to_string(),
                            Attribute::Tensor(Tensor {
                                tensor: crate::tensor::TensorValue::U8(
                                    ndarray::array![1, 2, 3].into_dyn(),
                                ),
                                path: None,
                            }),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    doc_string: None,
                }],
                edge_infos: HashMap::new(),
            },
            metadata: [("foo".to_string(), "oof".to_string())]
                .into_iter()
                .collect(),
            functions: vec![],
        };

        let proto: ModelProto = expected.clone().try_into().unwrap();
        let candidate: Model = proto.try_into().unwrap();

        assert_eq!(expected, candidate);
    }
}
