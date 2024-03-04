use protobuf_codegen::Codegen;

fn main() {
    Codegen::new()
        .pure()
        .cargo_out_dir("generated_with_pure")
        .input("./protobuf_gen/onnx.proto3")
        .include("./protobuf_gen/")
        .run_from_script();
}
