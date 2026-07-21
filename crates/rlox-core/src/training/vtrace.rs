use crate::error::RloxError;

/// Compute V-trace targets and policy gradient advantages (Espeholt et al. 2018).
///
/// Processes backwards from t=n-1 to t=0:
///   rho_t = min(rho_bar, exp(log_rhos[t]))
///   c_t   = min(c_bar,   exp(log_rhos[t]))
///   non_terminal = 1.0 - dones[t]
///   delta_t = rho_t * (rewards[t] + gamma * non_terminal * values[t+1] - values[t])
///   vs[t]   = values[t] + delta_t + gamma * non_terminal * c_t * (vs[t+1] - values[t+1])
///   pg_advantages[t] = rho_t * (rewards[t] + gamma * non_terminal * vs[t+1] - values[t])
///
/// Uses `bootstrap_value` for values[n] and vs[n], zeroed when the last step
/// is terminal (`dones[n-1] == 1.0`).
///
/// Returns `(vs, pg_advantages)`.
// V-trace requires 4 parallel slices + 4 scalar hyperparams; each is an independent math arg.
#[allow(clippy::too_many_arguments)]
pub fn compute_vtrace(
    log_rhos: &[f32],
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    bootstrap_value: f32,
    gamma: f32,
    rho_bar: f32,
    c_bar: f32,
) -> Result<(Vec<f32>, Vec<f32>), RloxError> {
    let n = log_rhos.len();

    if rewards.len() != n || values.len() != n || dones.len() != n {
        return Err(RloxError::ShapeMismatch {
            expected: format!("all slices length {n}"),
            got: format!(
                "log_rhos={}, rewards={}, values={}, dones={}",
                n,
                rewards.len(),
                values.len(),
                dones.len()
            ),
        });
    }

    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut vs = vec![0.0f32; n];
    let mut pg_advantages = vec![0.0f32; n];

    // Handle last step (t = n-1) outside the loop
    let last = n - 1;
    {
        let ratio = log_rhos[last].exp();
        let rho_t = rho_bar.min(ratio);
        let non_terminal = 1.0 - dones[last];
        let next_value = bootstrap_value * non_terminal;

        let delta_t = rho_t * (rewards[last] + gamma * next_value - values[last]);
        // vs_next for the last step is bootstrap_value (zeroed if terminal)
        let vs_next_val = bootstrap_value * non_terminal;
        vs[last] = values[last]
            + delta_t
            + gamma * non_terminal * rho_bar.min(ratio).min(c_bar) * (vs_next_val - next_value);
        pg_advantages[last] = rho_t * (rewards[last] + gamma * vs_next_val - values[last]);
    }

    // Iterate backwards for remaining steps
    let mut vs_next = vs[last];

    for t in (0..last).rev() {
        let ratio = log_rhos[t].exp();
        let rho_t = rho_bar.min(ratio);
        let c_t = c_bar.min(ratio);
        let non_terminal = 1.0 - dones[t];

        let next_value = values[t + 1];

        let delta_t = rho_t * (rewards[t] + gamma * non_terminal * next_value - values[t]);
        vs[t] = values[t] + delta_t + gamma * non_terminal * c_t * (vs_next - next_value);
        pg_advantages[t] = rho_t * (rewards[t] + gamma * non_terminal * vs_next - values[t]);

        vs_next = vs[t];
    }

    Ok((vs, pg_advantages))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vtrace_empty_input() {
        let (vs, adv) = compute_vtrace(&[], &[], &[], &[], 0.0, 0.99, 1.0, 1.0).unwrap();
        assert!(vs.is_empty());
        assert!(adv.is_empty());
    }

    #[test]
    fn vtrace_mismatched_lengths() {
        let result = compute_vtrace(&[0.0], &[1.0, 2.0], &[0.5], &[0.0], 0.0, 0.99, 1.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn vtrace_on_policy_matches_gae_like() {
        // When log_rhos = 0 (on-policy), rho=1, c=1 => V-trace reduces to
        // something close to GAE(lambda=1)
        let log_rhos = vec![0.0; 3];
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0];
        let bootstrap = 0.0;
        let gamma = 0.99;

        let dones = vec![0.0; 3];
        let (vs, _adv) = compute_vtrace(
            &log_rhos, &rewards, &values, &dones, bootstrap, gamma, 1.0, 1.0,
        )
        .unwrap();

        // On-policy with rho=c=1:
        // t=2: delta = 1*(1 + 0.99*0 - 0) = 1, vs[2] = 0 + 1 + 0.99*1*(0-0) = 1
        // t=1: delta = 1*(1 + 0.99*0 - 0) = 1, vs[1] = 0 + 1 + 0.99*1*(1-0) = 1.99
        // t=0: delta = 1*(1 + 0.99*0 - 0) = 1, vs[0] = 0 + 1 + 0.99*1*(1.99-0) = 2.9701
        assert!((vs[2] - 1.0).abs() < 1e-5);
        assert!((vs[1] - 1.99).abs() < 1e-5);
        assert!((vs[0] - 2.9701).abs() < 1e-4);
    }

    #[test]
    fn vtrace_single_step() {
        // Use rho_bar large enough so clipping doesn't engage
        let log_rho = 0.5_f32;
        let log_rhos = vec![log_rho];
        let rewards = vec![1.0];
        let values = vec![0.5];
        let bootstrap = 0.0;
        let gamma = 0.99;
        let rho_bar = 10.0; // no clipping
        let c_bar = 10.0;

        let dones = vec![0.0];
        let (vs, adv) = compute_vtrace(
            &log_rhos, &rewards, &values, &dones, bootstrap, gamma, rho_bar, c_bar,
        )
        .unwrap();

        let rho = log_rho.exp(); // ~1.6487
        let _c = c_bar.min(rho);
        // t=0 (only step, n-1): next_value = bootstrap = 0, vs_next = bootstrap = 0
        // delta = rho * (1.0 + 0.99*0 - 0.5) = rho * 0.5
        // vs[0] = 0.5 + rho*0.5 + 0.99*c*(bootstrap - bootstrap) = 0.5 + rho*0.5
        // pg_adv[0] = rho * (1.0 + 0.99*bootstrap - 0.5) = rho * 0.5
        let expected_vs = 0.5 + rho * 0.5;
        let expected_adv = rho * 0.5;

        assert!(
            (vs[0] - expected_vs).abs() < 1e-5,
            "vs[0]={}, expected={}",
            vs[0],
            expected_vs
        );
        assert!(
            (adv[0] - expected_adv).abs() < 1e-5,
            "adv[0]={}, expected={}",
            adv[0],
            expected_adv
        );
    }

    #[test]
    fn vtrace_clipping_reduces_correction() {
        // With very large importance ratio, clipping should limit the correction
        let log_rhos = vec![5.0]; // exp(5) ~ 148, way above rho_bar=1
        let rewards = vec![1.0];
        let values = vec![0.0];
        let bootstrap = 0.0;
        let gamma = 0.99;

        let dones = vec![0.0];
        let (vs_clipped, _) = compute_vtrace(
            &log_rhos, &rewards, &values, &dones, bootstrap, gamma, 1.0, 1.0,
        )
        .unwrap();
        let (vs_unclipped, _) = compute_vtrace(
            &log_rhos, &rewards, &values, &dones, bootstrap, gamma, 200.0, 200.0,
        )
        .unwrap();

        // With rho_bar=1.0, rho is clamped to 1.0 => delta = 1*(1-0) = 1 => vs = 1.0
        assert!((vs_clipped[0] - 1.0).abs() < 1e-5);
        // Without clipping, rho ~ 148 => delta = 148*(1-0) = 148 => vs = 148
        assert!(vs_unclipped[0] > 100.0);
    }

    #[test]
    fn vtrace_output_lengths_match_input() {
        let n = 10;
        let log_rhos = vec![0.0; n];
        let rewards = vec![1.0; n];
        let values = vec![0.5; n];
        let dones = vec![0.0; n];
        let (vs, adv) =
            compute_vtrace(&log_rhos, &rewards, &values, &dones, 0.0, 0.99, 1.0, 1.0).unwrap();
        assert_eq!(vs.len(), n);
        assert_eq!(adv.len(), n);
    }

    #[test]
    fn vtrace_reference_implementation() {
        // Reference: manually compute for a 3-step trajectory
        let gamma = 0.9_f32;
        let rho_bar = 1.5_f32;
        let c_bar = 1.2_f32;

        let log_rhos = vec![0.2, -0.3, 0.8];
        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 1.0, 1.5];
        let bootstrap = 2.0;

        // Manually compute backwards:
        // t=2: rho = min(1.5, exp(0.8)) = min(1.5, 2.2255) = 1.5
        //       c  = min(1.2, 2.2255) = 1.2
        //       next_val = bootstrap = 2.0
        //       delta = 1.5 * (3.0 + 0.9*2.0 - 1.5) = 1.5 * 3.3 = 4.95
        //       vs[2] = 1.5 + 4.95 + 0.9*1.2*(2.0 - 2.0) = 6.45
        //       pg_adv[2] = 1.5 * (3.0 + 0.9*2.0 - 1.5) = 4.95
        //       vs_next = 6.45
        let rho_2 = 1.5_f32;
        let c_2 = 1.2_f32;
        let delta_2 = rho_2 * (3.0 + 0.9 * 2.0 - 1.5);
        let vs_2 = 1.5 + delta_2 + 0.9 * c_2 * (2.0 - 2.0);
        let pg_2 = rho_2 * (3.0 + 0.9 * 2.0 - 1.5);

        // t=1: rho = min(1.5, exp(-0.3)) = min(1.5, 0.7408) = 0.7408
        //       c  = min(1.2, 0.7408) = 0.7408
        //       next_val = values[2] = 1.5
        //       delta = 0.7408 * (2.0 + 0.9*1.5 - 1.0) = 0.7408 * 2.35 = 1.74088
        //       vs[1] = 1.0 + 1.74088 + 0.9*0.7408*(6.45 - 1.5) = 1.0 + 1.74088 + 0.9*0.7408*4.95
        //       pg_adv[1] = 0.7408 * (2.0 + 0.9*6.45 - 1.0) = 0.7408 * (2.0 + 5.805 - 1.0) = 0.7408 * 6.805
        let rho_1 = (-0.3_f32).exp();
        let c_1 = c_bar.min(rho_1);
        let delta_1 = rho_1 * (2.0 + 0.9 * 1.5 - 1.0);
        let vs_1 = 1.0 + delta_1 + 0.9 * c_1 * (vs_2 - 1.5);
        let pg_1 = rho_1 * (2.0 + 0.9 * vs_2 - 1.0);

        // t=0: rho = min(1.5, exp(0.2)) = min(1.5, 1.2214) = 1.2214
        //       c  = min(1.2, 1.2214) = 1.2
        //       next_val = values[1] = 1.0
        //       delta = 1.2214 * (1.0 + 0.9*1.0 - 0.5) = 1.2214 * 1.4
        //       vs[0] = 0.5 + delta + 0.9*1.2*(vs_1 - 1.0)
        //       pg_adv[0] = 1.2214 * (1.0 + 0.9*vs_1 - 0.5)
        let rho_0 = (0.2_f32).exp();
        let c_0 = c_bar.min(rho_0);
        let delta_0 = rho_0 * (1.0 + 0.9 * 1.0 - 0.5);
        let vs_0 = 0.5 + delta_0 + 0.9 * c_0 * (vs_1 - 1.0);
        let pg_0 = rho_0 * (1.0 + 0.9 * vs_1 - 0.5);

        let dones = vec![0.0; 3];
        let (vs, adv) = compute_vtrace(
            &log_rhos, &rewards, &values, &dones, bootstrap, gamma, rho_bar, c_bar,
        )
        .unwrap();

        assert!(
            (vs[0] - vs_0).abs() < 1e-4,
            "vs[0]: got {}, expected {}",
            vs[0],
            vs_0
        );
        assert!(
            (vs[1] - vs_1).abs() < 1e-4,
            "vs[1]: got {}, expected {}",
            vs[1],
            vs_1
        );
        assert!(
            (vs[2] - vs_2).abs() < 1e-4,
            "vs[2]: got {}, expected {}",
            vs[2],
            vs_2
        );
        assert!(
            (adv[0] - pg_0).abs() < 1e-4,
            "adv[0]: got {}, expected {}",
            adv[0],
            pg_0
        );
        assert!(
            (adv[1] - pg_1).abs() < 1e-4,
            "adv[1]: got {}, expected {}",
            adv[1],
            pg_1
        );
        assert!(
            (adv[2] - pg_2).abs() < 1e-4,
            "adv[2]: got {}, expected {}",
            adv[2],
            pg_2
        );
    }

    #[test]
    fn vtrace_with_dones_resets_at_boundary() {
        // 4-step trajectory with a done at t=1. Episode boundary should
        // prevent rewards from leaking across episodes.
        let gamma = 0.99_f32;
        let log_rhos = vec![0.0; 4]; // on-policy
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        let values = vec![0.0; 4];
        let dones = vec![0.0, 1.0, 0.0, 0.0]; // done at t=1
        let bootstrap = 0.0;

        let (vs_with_dones, _) = compute_vtrace(
            &log_rhos, &rewards, &values, &dones, bootstrap, gamma, 1.0, 1.0,
        )
        .unwrap();

        // Without dones, rewards leak across episodes
        let no_dones = vec![0.0; 4];
        let (vs_no_dones, _) = compute_vtrace(
            &log_rhos, &rewards, &values, &no_dones, bootstrap, gamma, 1.0, 1.0,
        )
        .unwrap();

        // After the boundary (t=0), the done-aware version should produce
        // a LOWER vs because future rewards beyond the boundary are zeroed.
        assert!(
            vs_with_dones[0] < vs_no_dones[0],
            "vs_with_dones[0]={} should be < vs_no_dones[0]={}",
            vs_with_dones[0],
            vs_no_dones[0]
        );

        // Steps after the boundary (t=2, t=3) should be unaffected
        assert!(
            (vs_with_dones[3] - vs_no_dones[3]).abs() < 1e-5,
            "t=3 should be identical"
        );
    }

    #[test]
    fn vtrace_without_dones_matches_old_behavior() {
        // Passing all-zeros dones should reproduce the original behavior
        let gamma = 0.9_f32;
        let rho_bar = 1.5_f32;
        let c_bar = 1.2_f32;
        let log_rhos = vec![0.2, -0.3, 0.8];
        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 1.0, 1.5];
        let bootstrap = 2.0;
        let dones = vec![0.0; 3];

        let (vs, adv) = compute_vtrace(
            &log_rhos, &rewards, &values, &dones, bootstrap, gamma, rho_bar, c_bar,
        )
        .unwrap();

        // Manually computed reference (same as vtrace_reference_implementation)
        let rho_2 = 1.5_f32;
        let c_2 = 1.2_f32;
        let delta_2 = rho_2 * (3.0 + 0.9 * 2.0 - 1.5);
        let vs_2 = 1.5 + delta_2 + 0.9 * c_2 * (2.0 - 2.0);
        let pg_2 = rho_2 * (3.0 + 0.9 * 2.0 - 1.5);

        let rho_1 = (-0.3_f32).exp();
        let c_1 = c_bar.min(rho_1);
        let delta_1 = rho_1 * (2.0 + 0.9 * 1.5 - 1.0);
        let vs_1 = 1.0 + delta_1 + 0.9 * c_1 * (vs_2 - 1.5);
        let pg_1 = rho_1 * (2.0 + 0.9 * vs_2 - 1.0);

        let rho_0 = (0.2_f32).exp();
        let c_0 = c_bar.min(rho_0);
        let delta_0 = rho_0 * (1.0 + 0.9 * 1.0 - 0.5);
        let vs_0 = 0.5 + delta_0 + 0.9 * c_0 * (vs_1 - 1.0);
        let pg_0 = rho_0 * (1.0 + 0.9 * vs_1 - 0.5);

        assert!(
            (vs[0] - vs_0).abs() < 1e-4,
            "vs[0]: got {}, expected {}",
            vs[0],
            vs_0
        );
        assert!(
            (vs[1] - vs_1).abs() < 1e-4,
            "vs[1]: got {}, expected {}",
            vs[1],
            vs_1
        );
        assert!(
            (vs[2] - vs_2).abs() < 1e-4,
            "vs[2]: got {}, expected {}",
            vs[2],
            vs_2
        );
        assert!(
            (adv[0] - pg_0).abs() < 1e-4,
            "adv[0]: got {}, expected {}",
            adv[0],
            pg_0
        );
        assert!(
            (adv[1] - pg_1).abs() < 1e-4,
            "adv[1]: got {}, expected {}",
            adv[1],
            pg_1
        );
        assert!(
            (adv[2] - pg_2).abs() < 1e-4,
            "adv[2]: got {}, expected {}",
            adv[2],
            pg_2
        );

        // Suppress unused-variable warnings
        let _ = (c_0, c_1, c_2, pg_0, pg_1, pg_2, delta_0, delta_1, delta_2);
    }

    #[test]
    fn vtrace_dones_at_last_step_zeros_bootstrap() {
        // When the last step is terminal, bootstrap should be zeroed
        let gamma = 0.99_f32;
        let log_rhos = vec![0.0]; // on-policy, single step
        let rewards = vec![1.0];
        let values = vec![0.5];
        let bootstrap = 10.0; // large bootstrap to make the effect visible

        // With done at last step
        let dones_terminal = vec![1.0];
        let (vs_term, adv_term) = compute_vtrace(
            &log_rhos,
            &rewards,
            &values,
            &dones_terminal,
            bootstrap,
            gamma,
            1.0,
            1.0,
        )
        .unwrap();

        // Without done
        let dones_none = vec![0.0];
        let (vs_cont, adv_cont) = compute_vtrace(
            &log_rhos,
            &rewards,
            &values,
            &dones_none,
            bootstrap,
            gamma,
            1.0,
            1.0,
        )
        .unwrap();

        // Terminal: delta = 1*(1.0 + 0.99*0*10 - 0.5) = 0.5, vs = 0.5 + 0.5 = 1.0
        // Non-terminal: delta = 1*(1.0 + 0.99*10 - 0.5) = 10.4, vs = 0.5 + 10.4 = 10.9
        assert!(
            (vs_term[0] - 1.0).abs() < 1e-5,
            "terminal vs[0]={}, expected 1.0",
            vs_term[0]
        );
        assert!(
            vs_cont[0] > vs_term[0],
            "non-terminal vs should be larger due to bootstrap"
        );

        // Terminal advantage: rho*(r + gamma*0*vs_next - v) = 1*(1+0-0.5) = 0.5
        assert!(
            (adv_term[0] - 0.5).abs() < 1e-5,
            "terminal adv[0]={}, expected 0.5",
            adv_term[0]
        );
        assert!(
            adv_cont[0] > adv_term[0],
            "non-terminal adv should be larger"
        );
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn vtrace_output_length_matches_input(n in 0..200usize) {
                let log_rhos = vec![0.0; n];
                let rewards = vec![1.0; n];
                let values = vec![0.5; n];
                let dones = vec![0.0; n];
                let (vs, adv) = compute_vtrace(&log_rhos, &rewards, &values, &dones, 0.0, 0.99, 1.0, 1.0).unwrap();
                prop_assert_eq!(vs.len(), n);
                prop_assert_eq!(adv.len(), n);
            }

            #[test]
            fn vtrace_on_policy_vs_are_finite(n in 1..100usize) {
                let log_rhos = vec![0.0; n];
                let rewards: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
                let values: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
                let dones = vec![0.0; n];
                let (vs, adv) = compute_vtrace(&log_rhos, &rewards, &values, &dones, 0.0, 0.99, 1.0, 1.0).unwrap();
                for i in 0..n {
                    prop_assert!(vs[i].is_finite(), "vs[{}] is not finite: {}", i, vs[i]);
                    prop_assert!(adv[i].is_finite(), "adv[{}] is not finite: {}", i, adv[i]);
                }
            }
        }
    }
}
