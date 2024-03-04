use std::backtrace::Backtrace;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Deserialization faild: `{0}`")]
    DeserializingError(#[from] protobuf::Error),
    #[error("Validation failed: `{msg}`: {bt:?}")]
    ValidationError { msg: String, bt: String },
}

impl Error {
    pub fn new_validation(msg: String) -> Self {
        Self::ValidationError {
            msg,
            bt: Backtrace::force_capture().to_string(),
        }
    }
}
