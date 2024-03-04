use std::collections::HashMap;

use crate::protos::FunctionProto;

use crate::{Attribute, Error, Node, Operation};

/// Duck-typed local function definition. A frankly rather broken concept.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    operation: Operation,
    doc_string: Option<String>,
    attributes: HashMap<String, FunctionAttribute>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    nodes: Vec<Node>,
    opsets: HashMap<String, i64>,
}

#[allow(unused)]
#[derive(Clone, Debug, PartialEq)]
enum FunctionAttribute {
    Mandatory(String),
    Optional(Attribute),
}

impl TryFrom<FunctionProto> for Function {
    type Error = Error;

    fn try_from(_value: FunctionProto) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl TryFrom<Function> for FunctionProto {
    type Error = Error;

    fn try_from(_value: Function) -> Result<Self, Self::Error> {
        todo!()
    }
}
