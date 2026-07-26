//! Gradient flow integration tests.
//!
//! These tests verify that actor training steps produce non-zero parameter updates,
//! catching the autograd-through-trait-boundary bug where `TensorData` conversion
//! severs the computation graph.

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::prelude::Backend;
    use rlox_nn::{DeterministicPolicy, StochasticPolicy, TensorData};

    use crate::continuous_q::BurnTwinQ;
    use crate::deterministic::BurnDeterministicPolicy;
    use crate::stochastic::BurnStochasticPolicy;

    type TestBackend = Autodiff<NdArray>;
    type TestDevice = <NdArray as Backend>::Device;

    fn device() -> TestDevice {
        Default::default()
    }

    /// Serialises these tests and seeds the backend so weight init is reproducible.
    /// Hold the returned guard for the whole test: `let _g = seeded();`
    ///
    /// Why both halves are needed:
    ///
    /// *Seeding* — these tests assert that a training step moves the parameters
    /// by more than 1e-8. `BurnDeterministicPolicy::new` takes no seed (unlike
    /// `BurnStochasticPolicy`), so with unseeded init an unlucky draw — a
    /// saturated tanh, a near-zero gradient — makes a step vanish and the test
    /// fail. That is how `test_td3_multiple_steps_reduce_negative_q` failed CI on
    /// a docs-only change, and why its sibling had been quarantined with
    /// `#[ignore]`. Seeded, the measured margins are 0.075–1.46 against
    /// thresholds of 1e-8/1e-7 — seven orders of magnitude, so they are also
    /// insensitive to cross-platform float differences.
    ///
    /// *The lock* — `Backend::seed` sets **process-global** RNG state, and cargo
    /// runs a test binary's tests on multiple threads. Seeding alone therefore
    /// does not make init deterministic: concurrent tests consume each other's
    /// RNG. Measured: 2 failures in 25 runs with seeding but no lock, 0 in 15
    /// with `--test-threads=1`. The mutex gives that serialisation for these
    /// tests only, without forcing it on the whole workspace or adding a
    /// `serial_test` dependency.
    fn seeded() -> std::sync::MutexGuard<'static, ()> {
        static BACKEND_RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());
        // A panicking test poisons the mutex; recover so one failure does not
        // cascade into spurious failures in the rest.
        let guard = BACKEND_RNG.lock().unwrap_or_else(|e| e.into_inner());
        <TestBackend as Backend>::seed(42);
        guard
    }

    // ─── TD3 gradient flow ───────────────────────────────────

    // Un-ignored: this was quarantined for the unseeded-init flake that
    // `seeded()` now fixes at the source. With seed 42 the single-step
    // parameter change measures 1.46 against a 1e-7 threshold — seven orders of
    // magnitude of margin, so it is not sensitive to cross-platform float
    // differences. It guards the autograd-through-trait-boundary regression this
    // module exists for, so it is worth having back rather than skipped.
    #[test]
    fn test_td3_actor_step_changes_params() {
        let _guard = seeded();
        let mut policy =
            BurnDeterministicPolicy::<TestBackend>::new(3, 1, 64, 1.0, 1e-2, device().into());
        let critic = BurnTwinQ::<TestBackend>::new(3, 1, 64, 3e-4, device().into());

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
    fn test_td3_multiple_steps_reduce_negative_q() {
        let _guard = seeded();
        let mut policy =
            BurnDeterministicPolicy::<TestBackend>::new(3, 1, 64, 1.0, 1e-2, device().into());
        let critic = BurnTwinQ::<TestBackend>::new(3, 1, 64, 3e-4, device().into());

        let obs = TensorData::new(vec![1.0, 0.5, -1.0, 2.0, -0.5, 0.0], vec![2, 3]);

        // Multiple steps should consistently change the policy
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
        let _guard = seeded();
        let mut policy =
            BurnStochasticPolicy::<TestBackend>::new(3, 2, 64, 1e-2, device().into(), 42);
        let critic = BurnTwinQ::<TestBackend>::new(3, 2, 64, 3e-4, device().into());

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
        let _guard = seeded();
        let mut policy =
            BurnStochasticPolicy::<TestBackend>::new(3, 1, 64, 1e-2, device().into(), 42);
        let critic = BurnTwinQ::<TestBackend>::new(3, 1, 64, 3e-4, device().into());

        let obs = TensorData::new(vec![1.0, 0.5, -1.0, 2.0, -0.5, 0.0], vec![2, 3]);

        let mut all_changed = true;
        for _ in 0..5 {
            let before = policy.deterministic_action(&obs).unwrap();
            policy.sac_actor_step(&obs, 0.2, &critic).unwrap();
            let after = policy.deterministic_action(&obs).unwrap();
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
            "every SAC actor step should produce a parameter change"
        );
    }
}
