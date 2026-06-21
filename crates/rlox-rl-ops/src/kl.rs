//! Token-level KL divergence operations (exact and Schulman 2020 estimator).
//!
//! The `impl_kl_ops!` macro generates both `f64_ops` and `f32_ops` sub-modules,
//! each exposing the same function set for their respective float type.  The
//! f64 variants are re-exported at module level for ergonomic use as the default.

macro_rules! impl_kl_ops {
    ($mod_name:ident, $float:ty) => {
        pub mod $mod_name {
            use crate::error::RlOpsError;

            /// GRPO group advantage: `(reward - mean) / std`.
            /// Returns zeros if std < 1e-8.
            pub fn compute_group_advantages(rewards: &[$float]) -> Vec<$float> {
                if rewards.is_empty() {
                    return Vec::new();
                }

                let n = rewards.len() as $float;
                let mean = rewards.iter().sum::<$float>() / n;
                let variance = rewards
                    .iter()
                    .map(|&r| (r - mean) * (r - mean))
                    .sum::<$float>()
                    / n;
                let std = variance.sqrt();

                if std < 1e-8 as $float {
                    return vec![0.0 as $float; rewards.len()];
                }

                let inv_std = 1.0 as $float / std;
                rewards.iter().map(|&r| (r - mean) * inv_std).collect()
            }

            /// Token-level KL divergence: `sum(exp(log_p) * (log_p - log_q))`.
            pub fn compute_token_kl(
                log_probs_policy: &[$float],
                log_probs_ref: &[$float],
            ) -> Result<$float, RlOpsError> {
                if log_probs_policy.len() != log_probs_ref.len() {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: format!("len={}", log_probs_policy.len()),
                        got: format!("len={}", log_probs_ref.len()),
                    });
                }

                Ok(log_probs_policy
                    .iter()
                    .zip(log_probs_ref.iter())
                    .map(|(&log_p, &log_q)| log_p.exp() * (log_p - log_q))
                    .sum())
            }

            /// Batched GRPO group advantages: process all groups in a single call.
            ///
            /// `rewards` is a flat slice of length `n_prompts * group_size`.
            /// Returns a Vec of the same length with per-group z-score normalisation.
            pub fn compute_batch_group_advantages(
                rewards: &[$float],
                group_size: usize,
            ) -> Result<Vec<$float>, RlOpsError> {
                if group_size == 0 {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: "group_size > 0".to_string(),
                        got: "0".to_string(),
                    });
                }
                if rewards.len() % group_size != 0 {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: format!("len divisible by {group_size}"),
                        got: format!("len={}", rewards.len()),
                    });
                }

                const PAR_ELEMENT_THRESHOLD: usize = 4096;
                if rewards.len() >= PAR_ELEMENT_THRESHOLD {
                    use rayon::prelude::*;
                    let out: Vec<$float> = rewards
                        .par_chunks_exact(group_size)
                        .flat_map_iter(|group| compute_group_advantages(group))
                        .collect();
                    Ok(out)
                } else {
                    let mut out = Vec::with_capacity(rewards.len());
                    for group in rewards.chunks_exact(group_size) {
                        out.extend_from_slice(&compute_group_advantages(group));
                    }
                    Ok(out)
                }
            }

            /// Token-level KL divergence using the Schulman (2020) estimator:
            /// `sum(exp(log_p - log_q) - (log_p - log_q) - 1)`.
            ///
            /// This is the estimator used by TRL (HuggingFace). It is unbiased and
            /// numerically more stable than the exact `exp(log_p) * (log_p - log_q)`.
            pub fn compute_token_kl_schulman(
                log_probs_policy: &[$float],
                log_probs_ref: &[$float],
            ) -> Result<$float, RlOpsError> {
                if log_probs_policy.len() != log_probs_ref.len() {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: format!("len={}", log_probs_policy.len()),
                        got: format!("len={}", log_probs_ref.len()),
                    });
                }

                Ok(log_probs_policy
                    .iter()
                    .zip(log_probs_ref.iter())
                    .map(|(&log_p, &log_q)| {
                        let r = log_p - log_q;
                        r.exp() - r - 1.0 as $float
                    })
                    .sum())
            }

            /// Batched token-level KL divergence: process all sequences in a single call.
            ///
            /// `log_probs_policy` and `log_probs_ref` are flat slices of length `batch * seq_len`.
            /// Returns a Vec of length `batch` with per-sequence KL values.
            pub fn compute_batch_token_kl(
                log_probs_policy: &[$float],
                log_probs_ref: &[$float],
                seq_len: usize,
            ) -> Result<Vec<$float>, RlOpsError> {
                if log_probs_policy.len() != log_probs_ref.len() {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: format!("len={}", log_probs_policy.len()),
                        got: format!("len={}", log_probs_ref.len()),
                    });
                }
                if seq_len == 0 {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: "seq_len > 0".to_string(),
                        got: "0".to_string(),
                    });
                }
                if log_probs_policy.len() % seq_len != 0 {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: format!("len divisible by {seq_len}"),
                        got: format!("len={}", log_probs_policy.len()),
                    });
                }

                const PAR_ELEMENT_THRESHOLD: usize = 4096;
                let batch_size = log_probs_policy.len() / seq_len;

                let kl_for_seq = |i: usize| -> $float {
                    let off = i * seq_len;
                    let ps = &log_probs_policy[off..off + seq_len];
                    let qs = &log_probs_ref[off..off + seq_len];
                    ps.iter()
                        .zip(qs.iter())
                        .map(|(&log_p, &log_q)| log_p.exp() * (log_p - log_q))
                        .sum()
                };

                let out = if log_probs_policy.len() >= PAR_ELEMENT_THRESHOLD {
                    use rayon::prelude::*;
                    (0..batch_size).into_par_iter().map(kl_for_seq).collect()
                } else {
                    (0..batch_size).map(kl_for_seq).collect()
                };
                Ok(out)
            }

            /// Batched token-level KL divergence using the Schulman (2020) estimator.
            ///
            /// `log_probs_policy` and `log_probs_ref` are flat slices of length `batch * seq_len`.
            /// Returns a Vec of length `batch` with per-sequence KL values.
            pub fn compute_batch_token_kl_schulman(
                log_probs_policy: &[$float],
                log_probs_ref: &[$float],
                seq_len: usize,
            ) -> Result<Vec<$float>, RlOpsError> {
                if log_probs_policy.len() != log_probs_ref.len() {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: format!("len={}", log_probs_policy.len()),
                        got: format!("len={}", log_probs_ref.len()),
                    });
                }
                if seq_len == 0 {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: "seq_len > 0".to_string(),
                        got: "0".to_string(),
                    });
                }
                if log_probs_policy.len() % seq_len != 0 {
                    return Err(RlOpsError::ShapeMismatch {
                        expected: format!("len divisible by {seq_len}"),
                        got: format!("len={}", log_probs_policy.len()),
                    });
                }

                const PAR_ELEMENT_THRESHOLD: usize = 4096;
                let batch_size = log_probs_policy.len() / seq_len;

                let kl_for_seq = |i: usize| -> $float {
                    let off = i * seq_len;
                    let ps = &log_probs_policy[off..off + seq_len];
                    let qs = &log_probs_ref[off..off + seq_len];
                    ps.iter()
                        .zip(qs.iter())
                        .map(|(&log_p, &log_q)| {
                            let r = log_p - log_q;
                            r.exp() - r - 1.0 as $float
                        })
                        .sum()
                };

                let out = if log_probs_policy.len() >= PAR_ELEMENT_THRESHOLD {
                    use rayon::prelude::*;
                    (0..batch_size).into_par_iter().map(kl_for_seq).collect()
                } else {
                    (0..batch_size).map(kl_for_seq).collect()
                };
                Ok(out)
            }
        }
    };
}

impl_kl_ops!(f64_ops, f64);
impl_kl_ops!(f32_ops, f32);

// Re-export f64 versions at module level for ergonomic use as the default.
pub use f64_ops::*;

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
