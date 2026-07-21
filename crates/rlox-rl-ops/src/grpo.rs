//! Group-Relative Policy Optimisation (GRPO) advantage estimator.

use crate::error::RlOpsError;
use crate::estimator::AdvantageEstimator;
use crate::kl::f32_ops::{compute_batch_group_advantages, compute_group_advantages};

/// Group-Relative Policy Optimisation advantage estimator.
///
/// Normalises rewards within each group: `z = (r - mean) / std`.
/// Returns zeros when `std < 1e-8` (constant-reward group).
///
/// This is the canonical GRPO implementation matching the math in the original
/// `rlox-core/src/llm/ops.rs` — extracted here so `rlox-sandbox` can depend on
/// this slim crate instead of all of `rlox-core`.
pub struct GroupRelativeEstimator;

impl AdvantageEstimator for GroupRelativeEstimator {
    /// Compute per-rollout z-score normalised advantages.
    ///
    /// `rewards` must be a flat slice of length `n_groups * group_size`.
    /// Groups are contiguous: `rewards[0..group_size]` is group 0, etc.
    ///
    /// Uses rayon parallel dispatch when `rewards.len() >= 4096` elements.
    fn compute(&self, rewards: &[f32], group_size: usize) -> Result<Vec<f32>, RlOpsError> {
        compute_batch_group_advantages(rewards, group_size)
    }
}

/// Convenience free function: compute advantages for a single group.
///
/// Equivalent to calling `GroupRelativeEstimator.compute(rewards, rewards.len())`.
/// Exported so callers that only have one group don't need to pass `group_size`.
pub fn compute_single_group_advantages(rewards: &[f32]) -> Vec<f32> {
    compute_group_advantages(rewards)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Core contract test for the `AdvantageEstimator` trait implementation.
    ///
    /// Input:  rewards = [1, 0, 1, 0], group_size = 4
    /// Expected: advantages = [+1, -1, +1, -1] (z-score normalised)
    ///
    /// Verification:
    ///   mean = 0.5, variance = 0.25, std = 0.5
    ///   adv[i] = (r[i] - 0.5) / 0.5
    ///          => (1 - 0.5)/0.5 = +1.0, (0 - 0.5)/0.5 = -1.0
    #[test]
    fn advantage_estimator_trait_binary_rewards() {
        let estimator = GroupRelativeEstimator;
        let rewards = [1.0f32, 0.0, 1.0, 0.0];
        let adv = estimator
            .compute(&rewards, 4)
            .expect("compute must succeed");
        assert_eq!(adv.len(), 4);

        let expected = [1.0f32, -1.0, 1.0, -1.0];
        for (i, (&got, &exp)) in adv.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "adv[{i}]: expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn advantage_estimator_trait_constant_group_returns_zeros() {
        let estimator = GroupRelativeEstimator;
        let rewards = [5.0f32, 5.0, 5.0, 5.0];
        let adv = estimator
            .compute(&rewards, 4)
            .expect("compute must succeed");
        assert!(
            adv.iter().all(|&v| v == 0.0),
            "constant rewards must produce all-zero advantages"
        );
    }

    #[test]
    fn advantage_estimator_trait_multiple_groups() {
        let estimator = GroupRelativeEstimator;
        // Two groups: [1,0,1,0] and [1,1,1,1]
        let rewards = [1.0f32, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let adv = estimator
            .compute(&rewards, 4)
            .expect("compute must succeed");
        assert_eq!(adv.len(), 8);
        // First group: z-scored [+1, -1, +1, -1]
        assert!((adv[0] - 1.0).abs() < 1e-5);
        assert!((adv[1] + 1.0).abs() < 1e-5);
        // Second group: all same → all zeros
        assert!(adv[4..8].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn advantage_estimator_trait_bad_group_size_returns_err() {
        let estimator = GroupRelativeEstimator;
        // 3 rewards with group_size=2 — not divisible
        let result = estimator.compute(&[1.0f32, 0.0, 1.0], 2);
        assert!(result.is_err(), "non-divisible group_size must return Err");

        // group_size = 0 must return Err
        let result2 = estimator.compute(&[1.0f32], 0);
        assert!(result2.is_err(), "group_size=0 must return Err");
    }

    #[test]
    fn single_group_convenience_function() {
        let rewards = [2.0f32, 0.0];
        let adv = compute_single_group_advantages(&rewards);
        assert_eq!(adv.len(), 2);
        assert!((adv[0] - 1.0).abs() < 1e-5);
        assert!((adv[1] + 1.0).abs() < 1e-5);
    }

    /// Verify the estimator is object-safe and can be stored behind Arc<dyn>.
    #[test]
    fn advantage_estimator_is_dyn_compatible() {
        use std::sync::Arc;
        let estimator: Arc<dyn AdvantageEstimator> = Arc::new(GroupRelativeEstimator);
        let rewards = [1.0f32, 0.0, 1.0, 0.0];
        let adv = estimator.compute(&rewards, 4).unwrap();
        assert_eq!(adv.len(), 4);
    }
}
