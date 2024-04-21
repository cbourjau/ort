#!/usr/bin/env bash

set -o xtrace -o nounset -o pipefail -o errexit

if [[ "${target_platform}" == osx-* ]] ; then
  export RUSTFLAGS="-C link-args=-Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib"
else
  export RUSTFLAGS="-C link-arg=-Wl,-rpath-link,${PREFIX}/lib -L${PREFIX}/lib"
fi

cargo-bundle-licenses --format yaml --output ${SRC_DIR}/THIRDPARTY.yml

maturin build -m ort-py/Cargo.toml --release --skip-auditwheel
python -m pip install target/wheels/onnxrt*.whl -vv --no-deps --no-build-isolation --prefix=${PREFIX}

# remove extra build file
rm -f "${PREFIX}/.crates.toml"
