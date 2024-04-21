:: build
maturin build -m ort-py/Cargo.toml --release --skip-auditwheel
python -m pip install target/wheels/onnxrt*.whl -vv --no-deps --no-build-isolation --prefix="%PREFIX%"

:: remove extra build file
del /F /Q "%PREFIX%\.crates.toml"

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
