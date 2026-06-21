/// Error type for `rlox-rl-ops`.
///
/// Kept minimal: only the `ShapeMismatch` variant is needed for the arithmetic
/// ops in this crate.  The crate has zero dependency on `rlox-core::error`.
#[derive(Debug, thiserror::Error)]
pub enum RlOpsError {
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },
}
