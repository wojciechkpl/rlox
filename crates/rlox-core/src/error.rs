use thiserror::Error;

/// Top-level error type for the rlox workspace.
///
/// Each variant corresponds to a distinct subsystem. Callers can match
/// on the variant to decide how to recover or propagate.
#[derive(Debug, Error)]
pub enum RloxError {
    /// An action was invalid for the environment's action space.
    #[error("Invalid action: {0}")]
    InvalidAction(String),

    /// An environment-level error (reset failure, physics divergence, etc.).
    #[error("Environment error: {0}")]
    EnvError(String),

    /// A shape/dimension mismatch between expected and actual data.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// A replay-buffer or sampling error.
    #[error("Buffer error: {0}")]
    BufferError(String),

    /// A neural-network or training-related error (e.g. Candle backend failures).
    #[error("NN error: {0}")]
    NNError(String),

    /// An I/O error (file access, mmap, etc.).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<rlox_rl_ops::RlOpsError> for RloxError {
    fn from(e: rlox_rl_ops::RlOpsError) -> Self {
        match e {
            rlox_rl_ops::RlOpsError::ShapeMismatch { expected, got } => {
                RloxError::ShapeMismatch { expected, got }
            }
        }
    }
}
