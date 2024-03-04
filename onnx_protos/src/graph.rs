use std::collections::HashMap;

use crate::protos::GraphProto;
use crate::protos::ValueInfoProto;

use crate::utils::if_not_empty;
use crate::Error;
use crate::Node;
use crate::Tensor;
use crate::ValueInfo;

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Graph {
    pub name: String,
    pub doc_string: Option<String>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<Input>,
    pub initializers: HashMap<String, Tensor>,
    pub outputs: Vec<Output>,
    /// Optional information about internal edges of the graph
    pub edge_infos: HashMap<String, ValueInfo>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    pub name: String,
    pub info: ValueInfo,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub name: String,
    pub info: ValueInfo,
}

impl TryFrom<GraphProto> for Graph {
    type Error = Error;

    fn try_from(proto: GraphProto) -> Result<Self, Self::Error> {
        let initializers: HashMap<String, Tensor> = proto
            .initializer
            .into_iter()
            .map(|tp| Ok((tp.name.clone(), tp.try_into()?)))
            .collect::<Result<_, Self::Error>>()?;
        let inputs = proto
            .input
            .into_iter()
            .map(|vip| {
                Ok(Input {
                    name: vip.name.clone(),
                    info: vip.try_into()?,
                })
            })
            .collect::<Result<_, Self::Error>>()?;

        Ok(Self {
            name: proto.name,
            doc_string: if_not_empty(proto.doc_string),
            nodes: proto
                .node
                .into_iter()
                .map(|n| n.try_into())
                .collect::<Result<_, _>>()?,
            inputs,
            initializers,
            outputs: proto
                .output
                .into_iter()
                .map(|vi| vi.try_into())
                .collect::<Result<_, _>>()?,
            edge_infos: proto
                .value_info
                .into_iter()
                .map(|el| Ok((el.name.clone(), el.try_into()?)))
                .collect::<Result<_, Self::Error>>()?,
        })
    }
}

impl From<Graph> for GraphProto {
    fn from(graph: Graph) -> Self {
        let initializer: Vec<_> = graph
            .initializers
            .into_iter()
            .map(|(name, t)| t.tensor_proto(name))
            .collect();
        let mut input = vec![];

        for Input { name, info: kind } in graph.inputs.into_iter() {
            input.push(kind.value_info_proto(name))
        }
        Self {
            node: graph.nodes.into_iter().map(|n| n.into()).collect(),
            name: graph.name,
            input,
            initializer,
            sparse_initializer: vec![],
            doc_string: graph.doc_string.unwrap_or_default(),
            output: graph.outputs.into_iter().map(|el| el.into()).collect(),
            value_info: graph
                .edge_infos
                .into_iter()
                .map(|(name, ty)| ty.value_info_proto(name))
                .collect(),
            ..Default::default()
        }
    }
}

impl From<Output> for ValueInfoProto {
    fn from(value: Output) -> Self {
        value.info.value_info_proto(value.name.clone())
    }
}

impl TryFrom<ValueInfoProto> for Output {
    type Error = Error;

    fn try_from(proto: ValueInfoProto) -> Result<Self, Self::Error> {
        let name = proto.name.clone();
        Ok(Self {
            name,
            info: proto.try_into()?,
        })
    }
}
