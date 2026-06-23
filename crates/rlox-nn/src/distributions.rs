//! Pure-Rust distribution utilities for CPU-based sampling and log-prob.
//! These are backend-independent helpers that can be used by any backend
//! or for testing without a NN framework.

/// Compute log(softmax(logits)) in a numerically stable way.
/// Returns a vector of the same length as logits.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

/// Sample from a categorical distribution given logits.
/// Uses the Gumbel-max trick for differentiable-friendly sampling.
pub fn categorical_sample(logits: &[f32], uniform_rand: f32) -> usize {
    let log_probs = log_softmax(logits);
    let probs: Vec<f32> = log_probs.iter().map(|&lp| lp.exp()).collect();

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if uniform_rand < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

/// Compute log_prob for a categorical distribution.
pub fn categorical_log_prob(logits: &[f32], action: usize) -> f32 {
    let log_probs = log_softmax(logits);
    log_probs[action]
}

/// Compute entropy of a categorical distribution from logits.
pub fn categorical_entropy(logits: &[f32]) -> f32 {
    let log_probs = log_softmax(logits);
    let probs: Vec<f32> = log_probs.iter().map(|&lp| lp.exp()).collect();
    -probs
        .iter()
        .zip(log_probs.iter())
        .map(|(&p, &lp)| if p > 0.0 { p * lp } else { 0.0 })
        .sum::<f32>()
}

/// Compute log_prob for a normal distribution.
pub fn normal_log_prob(x: f32, mean: f32, std: f32) -> f32 {
    let var = std * std;
    let log_std = std.ln();
    -0.5 * ((x - mean) * (x - mean) / var + 2.0 * log_std + (2.0 * std::f32::consts::PI).ln())
}

/// Compute entropy of a normal distribution.
pub fn normal_entropy(std: f32) -> f32 {
    0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E * std * std).ln()
}

/// Tanh squashing log-prob correction: log_prob -= log(1 - tanh(x)^2 + eps)
pub fn tanh_log_prob_correction(pre_tanh: f32) -> f32 {
    let tanh_x = pre_tanh.tanh();
    -(1.0 - tanh_x * tanh_x + 1e-6).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);
        let sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_log_softmax_with_large_logits() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let log_probs = log_softmax(&logits);
        let sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-3, "numerical stability: {sum}");
    }

    #[test]
    fn test_log_softmax_uniform() {
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let log_probs = log_softmax(&logits);
        let expected = -(4.0_f32).ln();
        for &lp in &log_probs {
            assert!((lp - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_categorical_sample_boundaries() {
        let logits = vec![0.0, 0.0]; // equal probs
        assert_eq!(categorical_sample(&logits, 0.0), 0);
        assert_eq!(categorical_sample(&logits, 0.49), 0);
        assert_eq!(categorical_sample(&logits, 0.51), 1);
        assert_eq!(categorical_sample(&logits, 0.99), 1);
    }

    #[test]
    fn test_categorical_sample_skewed() {
        // logits [10, 0] -> prob ~[1, 0]
        let logits = vec![10.0, 0.0];
        assert_eq!(categorical_sample(&logits, 0.5), 0);
    }

    #[test]
    fn test_categorical_log_prob() {
        let logits = vec![0.0, 0.0, 0.0];
        let lp = categorical_log_prob(&logits, 1);
        let expected = -(3.0_f32).ln();
        assert!((lp - expected).abs() < 1e-5);
    }

    #[test]
    fn test_categorical_entropy_uniform() {
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let ent = categorical_entropy(&logits);
        let expected = (4.0_f32).ln(); // max entropy for 4 categories
        assert!(
            (ent - expected).abs() < 1e-5,
            "expected {expected}, got {ent}"
        );
    }

    #[test]
    fn test_categorical_entropy_deterministic() {
        let logits = vec![100.0, -100.0, -100.0];
        let ent = categorical_entropy(&logits);
        assert!(
            ent < 0.01,
            "near-deterministic should have low entropy: {ent}"
        );
    }

    #[test]
    fn test_normal_log_prob() {
        // Standard normal: log_prob(0) = -0.5 * ln(2π)
        let lp = normal_log_prob(0.0, 0.0, 1.0);
        let expected = -0.5 * (2.0 * std::f32::consts::PI).ln();
        assert!((lp - expected).abs() < 1e-5);
    }

    #[test]
    fn test_normal_log_prob_shifted() {
        let lp = normal_log_prob(2.0, 2.0, 1.0);
        let lp_center = normal_log_prob(0.0, 0.0, 1.0);
        assert!((lp - lp_center).abs() < 1e-5, "shifted mean at center");
    }

    #[test]
    fn test_normal_entropy() {
        let ent = normal_entropy(1.0);
        let expected = 0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln();
        assert!((ent - expected).abs() < 1e-5);
    }

    #[test]
    fn test_normal_entropy_wider_is_larger() {
        let ent1 = normal_entropy(1.0);
        let ent2 = normal_entropy(2.0);
        assert!(ent2 > ent1, "wider distribution should have higher entropy");
    }

    #[test]
    fn test_tanh_correction_at_zero() {
        let correction = tanh_log_prob_correction(0.0);
        // tanh(0) = 0, so correction = -ln(1 + eps) ≈ 0
        assert!(correction.abs() < 0.01);
    }

    #[test]
    fn test_tanh_correction_increases_at_extremes() {
        let c_small = tanh_log_prob_correction(0.5);
        let c_large = tanh_log_prob_correction(3.0);
        assert!(
            c_large > c_small,
            "correction should increase at extremes: {c_small} vs {c_large}"
        );
    }
}
