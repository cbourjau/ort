#! /bin/bash

curl https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.3/include/onnxruntime/core/session/onnxruntime_c_api.h -o onnxruntime_c_api.h

~/.cargo/bin/bindgen --no-copy "Ort.*" --allowlist-type "OrtApi.*" --allowlist-function "OrtGetApiBase" -o ../src/bindings.rs ./onnxruntime_c_api.h 
