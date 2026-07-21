use crate::error::RlOpsError;

/// Compute per-rollout advantages from a flat slice of scalar rewards.
///
/// `rewards` is a flat slice of length `n_groups * group_size`.
/// Returns a `Vec` of the same length with per-rollout advantage estimates.
///
/// # Contract
///
/// Implementations MUST be:
/// - **Deterministic**: same inputs always produce identical outputs.
/// - **Free of autograd operations**: this crate has no tensor/gradient deps.
/// - **Thread-safe**: `Send + Sync` so the estimator can live inside `Arc`.
pub trait AdvantageEstimator: Send + Sync {
    fn compute(&self, rewards: &[f32], group_size: usize) -> Result<Vec<f32>, RlOpsError>;
}
