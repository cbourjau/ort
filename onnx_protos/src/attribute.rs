use crate::protos::{attribute_proto::AttributeType, AttributeProto};

use crate::{Error, Graph, Tensor};

#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Tensor(Tensor),

    F32(f32),
    I64(i64),
    String(String),

    Strings(Vec<String>),
    F32s(Vec<f32>),
    I64s(Vec<i64>),

    Graph(Graph),

    /// Reference to attribute in outer scope
    RefAttrName(String),
}

impl Attribute {
    pub fn into_proto(self, name: String) -> AttributeProto {
        let default = AttributeProto {
            name,
            ..Default::default()
        };
        match self {
            Attribute::F32(f) => AttributeProto {
                f,
                type_: AttributeType::FLOAT.into(),
                ..default
            },
            Attribute::I64(i) => AttributeProto {
                i,
                type_: AttributeType::INT.into(),
                ..default
            },
            Attribute::String(s) => AttributeProto {
                s: s.into_bytes(),
                type_: AttributeType::STRING.into(),
                ..default
            },
            Attribute::F32s(floats) => AttributeProto {
                floats,
                type_: AttributeType::FLOATS.into(),
                ..default
            },
            Attribute::I64s(ints) => AttributeProto {
                ints,
                type_: AttributeType::INTS.into(),
                ..default
            },
            Attribute::Strings(s) => AttributeProto {
                strings: s.into_iter().map(|s| s.into_bytes()).collect(),
                type_: AttributeType::STRINGS.into(),
                ..default
            },
            Attribute::RefAttrName(ref_attr_name) => AttributeProto {
                ref_attr_name,
                // TODO: Unclear which type to use in this case
                type_: AttributeType::UNDEFINED.into(),
                ..default
            },
            Attribute::Graph(g) => AttributeProto {
                g: Some(g.into()).into(),
                type_: AttributeType::GRAPH.into(),
                ..default
            },
            Attribute::Tensor(t) => AttributeProto {
                t: Some(t.tensor_proto("".to_string())).into(),
                type_: AttributeType::TENSOR.into(),
                ..default
            },
        }
    }
}

impl TryFrom<AttributeProto> for Attribute {
    type Error = Error;

    fn try_from(ap: AttributeProto) -> Result<Self, Self::Error> {
        let ty = ap.type_.unwrap();

        Ok(match (ty, ap) {
            (AttributeType::FLOAT, AttributeProto { f, .. }) => Attribute::F32(f),
            (AttributeType::INT, AttributeProto { i, .. }) => Attribute::I64(i),
            (AttributeType::STRING, AttributeProto { s, .. }) => {
                Attribute::String(String::from_utf8(s).map_err(|e| {
                    Error::new_validation(format!("Atribute validation error: `{}`", e))
                })?)
            }
            (AttributeType::INTS, AttributeProto { ints, .. }) => Attribute::I64s(ints),
            (AttributeType::FLOATS, AttributeProto { floats, .. }) => Attribute::F32s(floats),
            (AttributeType::STRINGS, AttributeProto { strings, .. }) => {
                let strings = strings
                    .iter()
                    .map(|chunk| {
                        String::from_utf8(chunk.to_owned()).map_err(|e| {
                            Error::new_validation(format!("Atribute validation error: `{}`", e))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Attribute::Strings(strings)
            }
            (AttributeType::TENSOR, AttributeProto { t, .. }) => {
                let t = t
                    .into_option()
                    .ok_or_else(|| Error::new_validation("Tensor has not data".into()))?;
                Attribute::Tensor(t.try_into()?)
            }
            // (AttributeType::GRAPH, AttributeProto { g, .. }) => {
            //     let g = g
            //         .into_option()
            //         .ok_or_else(|| Err(Error::new_validation("Graph has not data".into())));
            //     Attribute::Graph(g.try_into()?)
            // }
            (ty, ap) => Err(Error::new_validation(format!(
                "Cannot handle type `{:#?}` for `{:#?}`",
                ty, ap
            )))?,
        })
    }
}
