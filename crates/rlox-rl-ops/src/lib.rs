//! Estimator-agnostic advantage and token-KL ops for rlox.
//!
//! This crate has no dependency on `rlox-core` (no environment physics, replay
//! buffers, or GAE).  It is intentionally slim so that `rlox-sandbox` can depend
//! on it without pulling in the entire training data-plane.
//!
//! # Public API
//!
//! - [`error::RlOpsError`] — the single error type for this crate.
//! - [`estimator::AdvantageEstimator`] — pluggable trait for advantage algorithms.
//! - [`grpo::GroupRelativeEstimator`] — GRPO implementation (z-score normalisation).
//! - [`kl`] — token-level KL ops (exact + Schulman 2020, f32 + f64 sub-modules).
//!
//! The f64 KL functions are re-exported at crate root via `kl::*`.

pub mod error;
pub mod estimator;
pub mod grpo;
pub mod kl;

// Flatten the top-level API: re-export the most commonly used items.
pub use error::RlOpsError;
pub use estimator::AdvantageEstimator;
pub use grpo::GroupRelativeEstimator;
// Re-export all f64 kl functions at crate root for ergonomic access.
pub use kl::*;
