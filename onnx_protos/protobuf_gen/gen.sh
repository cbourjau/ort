#! /bin/bash

set -exuo pipefail


ONNX_VERSION="1.15.0"

curl https://raw.githubusercontent.com/onnx/onnx/v${ONNX_VERSION}/onnx/onnx.proto3 -o onnx.proto3
