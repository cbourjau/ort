use std::collections::HashMap;

use crate::protos::NodeProto;

use crate::{utils::if_not_empty, Attribute, Error, Operation};

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub operation: Operation,
    pub attributes: HashMap<String, Attribute>,
    pub doc_string: Option<String>,
}

impl From<Node> for NodeProto {
    fn from(node: Node) -> Self {
        Self {
            name: node.name,
            input: node.inputs,
            output: node.outputs,
            op_type: node.operation.name,
            domain: node.operation.domain,
            doc_string: node.doc_string.unwrap_or_default(),
            attribute: node
                .attributes
                .into_iter()
                .map(|(name, attr)| attr.into_proto(name))
                .collect(),
            ..Default::default()
        }
    }
}

impl TryFrom<NodeProto> for Node {
    type Error = Error;

    fn try_from(proto: NodeProto) -> Result<Self, Self::Error> {
        Ok(Node {
            name: proto.name,
            inputs: proto.input,
            outputs: proto.output,
            operation: Operation {
                name: proto.op_type,
                domain: proto.domain,
            },
            attributes: proto
                .attribute
                .into_iter()
                .map(|ap| Ok((ap.name.clone(), ap.try_into()?)))
                .collect::<Result<_, Self::Error>>()?,
            doc_string: if_not_empty(proto.doc_string),
        })
    }
}
