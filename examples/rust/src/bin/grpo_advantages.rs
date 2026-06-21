//! GRPO advantage estimation using `rlox-rl-ops`.
//!
//! Demonstrates the estimator-agnostic advantage API introduced for the
//! agentic-RL benchmark.  The Group-Relative Policy Optimisation (GRPO)
//! estimator z-scores rewards within each group so that the policy gradient
//! only sees relative quality — not absolute reward magnitude.
//!
//! ```bash
//! cd examples/rust && cargo run --bin grpo_advantages
//! ```

use rlox_rl_ops::{AdvantageEstimator, GroupRelativeEstimator, RlOpsError};
use rlox_rl_ops::grpo::compute_single_group_advantages;

fn main() {
    // ── Multi-group batch ─────────────────────────────────────────────────────
    //
    // Two groups of 4 rollouts each (group_size = 4, 8 total rewards):
    //
    //   Group 0 — binary rewards [1, 0, 1, 0]
    //     mean = 0.5, std = 0.5
    //     advantages ≈ [+1, -1, +1, -1]   (z-score normalised)
    //
    //   Group 1 — constant rewards [5, 5, 5, 5]
    //     std < 1e-8  → all zeros (no relative signal to exploit)

    let rewards: &[f32] = &[
        // Group 0: binary pass/fail
        1.0, 0.0, 1.0, 0.0,
        // Group 1: constant reward (all rollouts identical quality)
        5.0, 5.0, 5.0, 5.0,
    ];
    let group_size = 4;

    let estimator = GroupRelativeEstimator;

    // The `AdvantageEstimator` trait is dyn-compatible; keep it behind the
    // trait reference to show that callers need not name the concrete type.
    let estimator_ref: &dyn AdvantageEstimator = &estimator;

    let advantages = estimator_ref
        .compute(rewards, group_size)
        .expect("compute must succeed for valid inputs");

    println!("GRPO advantage estimation — multi-group batch");
    println!("  rewards:    {rewards:?}");
    println!("  group_size: {group_size}");
    println!();
    println!("Group | Rollout | Reward | Advantage");
    println!("------+---------+--------+----------");
    for (idx, (&reward, &advantage)) in rewards.iter().zip(advantages.iter()).enumerate() {
        let group = idx / group_size;
        let rollout = idx % group_size;
        println!(
            "    {group} |       {rollout} |  {reward:.2}  |  {:+.4}",
            advantage
        );
    }

    // ── Verify the computed values ────────────────────────────────────────────

    // Group 0: binary rewards → ±1
    for i in 0..4 {
        let expected = if rewards[i] > 0.5 { 1.0f32 } else { -1.0f32 };
        assert!(
            (advantages[i] - expected).abs() < 1e-5,
            "group 0 rollout {i}: expected {expected}, got {}",
            advantages[i]
        );
    }
    // Group 1: constant rewards → all zero
    for i in 4..8 {
        assert!(
            advantages[i].abs() < 1e-8,
            "group 1 rollout {}: expected 0.0, got {}",
            i - 4,
            advantages[i]
        );
    }

    // ── Single-group convenience function ─────────────────────────────────────
    println!();
    println!("Single-group convenience (compute_single_group_advantages)");

    let single_rewards: &[f32] = &[2.0, 0.0];
    let single_adv = compute_single_group_advantages(single_rewards);

    println!("  rewards:    {single_rewards:?}");
    println!("  advantages: {single_adv:?}");

    // mean = 1.0, std = 1.0  → advantages = [+1, -1]
    assert!((single_adv[0] - 1.0).abs() < 1e-5);
    assert!((single_adv[1] + 1.0).abs() < 1e-5);

    // ── Error path: non-divisible group_size ──────────────────────────────────
    println!();
    println!("Error path — non-divisible group_size");

    let bad_result: Result<Vec<f32>, RlOpsError> =
        estimator.compute(&[1.0, 0.0, 1.0], 2);

    match bad_result {
        Err(RlOpsError::ShapeMismatch { expected, got }) => {
            println!(
                "  compute(&[1.0, 0.0, 1.0], group_size=2) => \
                 Err(ShapeMismatch {{ expected: \"{expected}\", got: \"{got}\" }})"
            );
        }
        Ok(_) => panic!("expected Err for non-divisible group_size, got Ok"),
    }

    println!();
    println!("All assertions passed.");
}
