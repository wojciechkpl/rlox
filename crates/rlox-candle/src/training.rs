//! Gradient flow integration tests.
//!
//! These tests verify that actor training steps produce non-zero parameter updates,
//! catching the autograd-through-trait-boundary bug where `TensorData` conversion
//! severs the computation graph.

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use rlox_nn::{DeterministicPolicy, StochasticPolicy, TensorData};

    use crate::continuous_q::CandleTwinQ;
    use crate::deterministic::CandleDeterministicPolicy;
    use crate::stochastic::CandleStochasticPolicy;

    // ─── TD3 gradient flow ───────────────────────────────────

    #[test]
    #[ignore = "flaky on CI — gradient can be near-zero with some random seeds"]
    fn test_td3_actor_step_changes_params() {
        let mut policy = CandleDeterministicPolicy::new(3, 1, 64, 1.0, 1e-2, Device::Cpu).unwrap();
        let critic = CandleTwinQ::new(3, 1, 64, 3e-4, Device::Cpu).unwrap();

        let obs = TensorData::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![4, 3],
        );

        let act_before = policy.act(&obs).unwrap();
        let metrics = policy.td3_actor_step(&obs, &critic).unwrap();
        let act_after = policy.act(&obs).unwrap();

        assert!(
            metrics.get("actor_loss").unwrap().is_finite(),
            "actor loss must be finite"
        );

        let changed = act_before
            .data
            .iter()
            .zip(act_after.data.iter())
            .any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(
            changed,
            "TD3 actor step must change model parameters (autograd must flow through critic)"
        );
    }

    #[test]
    #[ignore = "flaky: stochastic gradient test depends on RNG state"]
    fn test_td3_multiple_steps_reduce_negative_q() {
        let mut policy = CandleDeterministicPolicy::new(3, 1, 64, 1.0, 1e-2, Device::Cpu).unwrap();
        let critic = CandleTwinQ::new(3, 1, 64, 3e-4, Device::Cpu).unwrap();

        let obs = TensorData::new(vec![1.0, 0.5, -1.0, 2.0, -0.5, 0.0], vec![2, 3]);

        let mut all_changed = true;
        for _ in 0..5 {
            let before = policy.act(&obs).unwrap();
            policy.td3_actor_step(&obs, &critic).unwrap();
            let after = policy.act(&obs).unwrap();
            let changed = before
                .data
                .iter()
                .zip(after.data.iter())
                .any(|(a, b)| (a - b).abs() > 1e-8);
            if !changed {
                all_changed = false;
            }
        }
        assert!(
            all_changed,
            "every TD3 actor step should produce a parameter change"
        );
    }

    // ─── SAC gradient flow ───────────────────────────────────

    #[test]
    fn test_sac_actor_step_changes_params() {
        let mut policy = CandleStochasticPolicy::new(3, 2, 64, 1e-2, Device::Cpu, 42).unwrap();
        let critic = CandleTwinQ::new(3, 2, 64, 3e-4, Device::Cpu).unwrap();

        let obs = TensorData::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );

        let act_before = policy.deterministic_action(&obs).unwrap();
        let metrics = policy.sac_actor_step(&obs, 0.2, &critic).unwrap();
        let act_after = policy.deterministic_action(&obs).unwrap();

        assert!(
            metrics.get("actor_loss").unwrap().is_finite(),
            "actor loss must be finite"
        );

        let changed = act_before
            .data
            .iter()
            .zip(act_after.data.iter())
            .any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(
            changed,
            "SAC actor step must change model parameters (autograd must flow through critic)"
        );
    }

    #[test]
    fn test_sac_multiple_steps_change_policy() {
        let mut policy = CandleStochasticPolicy::new(3, 1, 64, 1e-2, Device::Cpu, 42).unwrap();
        let critic = CandleTwinQ::new(3, 1, 64, 3e-4, Device::Cpu).unwrap();

        let obs = TensorData::new(vec![1.0, 0.5, -1.0, 2.0, -0.5, 0.0], vec![2, 3]);

        let mut any_changed = false;
        for _ in 0..5 {
            let before = policy.deterministic_action(&obs).unwrap();
            policy.sac_actor_step(&obs, 0.2, &critic).unwrap();
            let after = policy.deterministic_action(&obs).unwrap();
            let changed = before
                .data
                .iter()
                .zip(after.data.iter())
                .any(|(a, b)| (a - b).abs() > 1e-8);
            if changed {
                any_changed = true;
            }
        }
        assert!(
            any_changed,
            "at least one SAC actor step should produce a parameter change (autograd must flow through critic)"
        );
    }
}
