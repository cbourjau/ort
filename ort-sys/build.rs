use std::env;

fn main() {
    if let Ok(prefix) = env::var("CONDA_PREFIX") {
        println!("cargo:rustc-link-search=native={}/lib", prefix);
    }
    println!("cargo:rustc-link-lib=onnxruntime");
}
