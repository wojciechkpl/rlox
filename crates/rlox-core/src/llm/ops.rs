//! LLM advantage and KL ops — forwarding re-exports from `rlox-rl-ops`.
//!
//! The implementations live in `rlox-rl-ops` so that `rlox-sandbox` can depend
//! on that slim crate without pulling in all of `rlox-core`.  Everything that
//! was previously defined here is still **publicly accessible under the same
//! path** (`rlox_core::llm::ops::*`, `rlox_core::llm::ops::f32_ops::*`, etc.)
//! so existing callers — including `rlox-python` PyO3 bindings — compile and
//! behave identically without any change.
//!
//! The only item that remains defined here rather than re-exported is
//! [`DPOPair`], which is a training-plane data container unrelated to advantage
//! estimation.

// ---------------------------------------------------------------------------
// Re-export everything from rlox-rl-ops::kl (the macro-generated f64 + f32
// sub-modules and the module-level f64 re-exports).
// ---------------------------------------------------------------------------

// f64 sub-module — keep accessible as `rlox_core::llm::ops::f64_ops`.
pub use rlox_rl_ops::kl::f64_ops;

// f32 sub-module — keep accessible as `rlox_core::llm::ops::f32_ops`.
pub use rlox_rl_ops::kl::f32_ops;

// Module-level f64 convenience re-exports (same as before: `pub use f64_ops::*`).
pub use rlox_rl_ops::kl::{
    compute_batch_group_advantages, compute_batch_token_kl, compute_batch_token_kl_schulman,
    compute_group_advantages, compute_token_kl, compute_token_kl_schulman,
};

// ---------------------------------------------------------------------------
// DPOPair — stays in rlox-core (training-plane data container).
// ---------------------------------------------------------------------------

/// A DPO preference pair holding tokenized prompt, chosen, and rejected sequences.
#[derive(Debug, Clone)]
pub struct DPOPair {
    pub prompt_tokens: Vec<u32>,
    pub chosen_tokens: Vec<u32>,
    pub rejected_tokens: Vec<u32>,
}

impl DPOPair {
    pub fn new(
        prompt_tokens: Vec<u32>,
        chosen_tokens: Vec<u32>,
        rejected_tokens: Vec<u32>,
    ) -> Self {
        Self {
            prompt_tokens,
            chosen_tokens,
            rejected_tokens,
        }
    }

    pub fn chosen_len(&self) -> usize {
        self.chosen_tokens.len()
    }

    pub fn rejected_len(&self) -> usize {
        self.rejected_tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_advantages_basic() {
        let rewards = [1.0, 0.5, 0.8];
        let adv = compute_group_advantages(&rewards);
        assert_eq!(adv.len(), 3);
        let mean: f64 = adv.iter().sum::<f64>() / adv.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_group_advantages_constant_rewards() {
        let rewards = [5.0, 5.0, 5.0];
        let adv = compute_group_advantages(&rewards);
        assert!(adv.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_group_advantages_empty() {
        let adv = compute_group_advantages(&[]);
        assert!(adv.is_empty());
    }

    #[test]
    fn test_token_kl_identical() {
        let log_p = [-1.0, -2.0, -0.5];
        let kl = compute_token_kl(&log_p, &log_p).unwrap();
        assert!(kl.abs() < 1e-15);
    }

    #[test]
    fn test_token_kl_known_value() {
        let log_p = [-1.0];
        let log_q = [-2.0];
        let kl = compute_token_kl(&log_p, &log_q).unwrap();
        assert!((kl - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_token_kl_mismatched_lengths_returns_err() {
        let result = compute_token_kl(&[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn token_kl_mismatched_lengths_returns_err_not_panic() {
        let log_p = vec![-1.0f64, -2.0];
        let log_q = vec![-1.0f64];
        let result = compute_token_kl(&log_p, &log_q);
        assert!(result.is_err(), "mismatched lengths must return Err");
    }

    #[test]
    fn token_kl_matching_lengths_returns_ok() {
        let log_p = vec![-1.0f64, -2.0, -0.5];
        let log_q = vec![-1.0f64, -2.0, -0.5];
        let result = compute_token_kl(&log_p, &log_q);
        assert!(result.is_ok());
        assert!(result.unwrap().abs() < 1e-15);
    }

    #[test]
    fn token_kl_empty_slices_returns_zero() {
        let result = compute_token_kl(&[], &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn token_kl_nan_input_propagates_to_output() {
        let log_p = vec![f64::NAN];
        let log_q = vec![-1.0f64];
        let result = compute_token_kl(&log_p, &log_q);
        if let Ok(v) = result {
            assert!(v.is_nan(), "NaN input should produce NaN output");
        }
    }

    #[test]
    fn token_kl_inf_input_does_not_panic() {
        let log_p = vec![f64::INFINITY];
        let log_q = vec![-1.0f64];
        let _result = compute_token_kl(&log_p, &log_q);
    }

    #[test]
    fn token_kl_known_value_still_correct_after_refactor() {
        let log_p = vec![-1.0f64];
        let log_q = vec![-2.0f64];
        let kl = compute_token_kl(&log_p, &log_q).unwrap();
        assert!((kl - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_batch_group_advantages() {
        let rewards = [1.0, 2.0, 3.0, 10.0, 10.0, 10.0];
        let adv = compute_batch_group_advantages(&rewards, 3).unwrap();
        assert_eq!(adv.len(), 6);
        let g1_mean: f64 = adv[..3].iter().sum::<f64>() / 3.0;
        assert!(g1_mean.abs() < 1e-10);
        assert!(adv[3..6].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_batch_group_advantages_bad_size() {
        assert!(compute_batch_group_advantages(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(compute_batch_group_advantages(&[1.0], 0).is_err());
    }

    #[test]
    fn test_token_kl_schulman_identical() {
        let log_p = [-1.0, -2.0, -0.5];
        let kl = compute_token_kl_schulman(&log_p, &log_p).unwrap();
        assert!(kl.abs() < 1e-15);
    }

    #[test]
    fn test_token_kl_schulman_known_value() {
        let log_p = [-1.0];
        let log_q = [-2.0];
        let kl = compute_token_kl_schulman(&log_p, &log_q).unwrap();
        assert!((kl - (1.0_f64.exp() - 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_token_kl_schulman_non_negative() {
        let log_p = [-0.5, -1.0, -3.0, 0.0];
        let log_q = [-1.0, -0.5, -0.1, -2.0];
        let kl = compute_token_kl_schulman(&log_p, &log_q).unwrap();
        assert!(kl >= 0.0, "Schulman KL should be non-negative, got {kl}");
    }

    #[test]
    fn test_dpo_pair() {
        let pair = DPOPair::new(vec![1, 2, 3], vec![4, 5], vec![6, 7, 8]);
        assert_eq!(pair.chosen_len(), 2);
        assert_eq!(pair.rejected_len(), 3);
        assert_eq!(pair.prompt_tokens.len(), 3);
    }

    // --- Batched KL tests ---

    #[test]
    fn test_batch_token_kl_matches_unbatched() {
        let log_p = vec![-1.0, -2.0, -0.5, -1.5, -0.3, -2.5];
        let log_q = vec![-1.1, -1.9, -0.6, -1.4, -0.4, -2.4];
        let batched = compute_batch_token_kl(&log_p, &log_q, 3).unwrap();
        let kl0 = compute_token_kl(&log_p[..3], &log_q[..3]).unwrap();
        let kl1 = compute_token_kl(&log_p[3..], &log_q[3..]).unwrap();
        assert_eq!(batched.len(), 2);
        assert!((batched[0] - kl0).abs() < 1e-12);
        assert!((batched[1] - kl1).abs() < 1e-12);
    }

    #[test]
    fn test_batch_token_kl_schulman_matches_unbatched() {
        let log_p = vec![-1.0, -2.0, -0.5, -1.5, -0.3, -2.5];
        let log_q = vec![-1.1, -1.9, -0.6, -1.4, -0.4, -2.4];
        let batched = compute_batch_token_kl_schulman(&log_p, &log_q, 3).unwrap();
        let kl0 = compute_token_kl_schulman(&log_p[..3], &log_q[..3]).unwrap();
        let kl1 = compute_token_kl_schulman(&log_p[3..], &log_q[3..]).unwrap();
        assert_eq!(batched.len(), 2);
        assert!((batched[0] - kl0).abs() < 1e-12);
        assert!((batched[1] - kl1).abs() < 1e-12);
    }

    #[test]
    fn test_batch_token_kl_bad_seq_len() {
        assert!(compute_batch_token_kl(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 0).is_err());
        assert!(compute_batch_token_kl(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 2).is_err());
    }

    #[test]
    fn test_batch_token_kl_mismatched_lengths() {
        assert!(compute_batch_token_kl(&[1.0, 2.0], &[1.0], 1).is_err());
    }

    // --- f32 variant tests ---

    #[test]
    fn test_f32_token_kl_identical() {
        let log_p: Vec<f32> = vec![-1.0, -2.0, -0.5];
        let kl = f32_ops::compute_token_kl(&log_p, &log_p).unwrap();
        assert!(kl.abs() < 1e-6);
    }

    #[test]
    fn test_f32_batch_token_kl_schulman() {
        let log_p: Vec<f32> = vec![-1.0, -2.0, -0.5, -1.5];
        let log_q: Vec<f32> = vec![-1.1, -1.9, -0.6, -1.4];
        let batched = f32_ops::compute_batch_token_kl_schulman(&log_p, &log_q, 2).unwrap();
        assert_eq!(batched.len(), 2);
        let kl0 = f32_ops::compute_token_kl_schulman(&log_p[..2], &log_q[..2]).unwrap();
        assert!((batched[0] - kl0).abs() < 1e-6);
    }

    #[test]
    fn test_f32_group_advantages() {
        let rewards: Vec<f32> = vec![1.0, 2.0, 3.0];
        let adv = f32_ops::compute_group_advantages(&rewards);
        assert_eq!(adv.len(), 3);
        let mean: f32 = adv.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5);
    }
}
