mod scoped {
    include!(concat!(env!("OUT_DIR"), "/generated_with_pure/mod.rs"));
}

pub use scoped::onnx_proto3::*;
