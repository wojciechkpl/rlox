use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{SendTimeoutError, Sender};

/// How long the collector waits on a full channel before re-checking the stop
/// flag. Bounds shutdown latency so `stop()`/`Drop` cannot deadlock behind a
/// blocked `send()` when the consumer has stopped draining.
const SEND_POLL_INTERVAL: Duration = Duration::from_millis(50);

use crate::env::batch::BatchSteppable;
use crate::env::spaces::Action;
use crate::pipeline::channel::RolloutBatch;
use crate::training::gae;

/// Asynchronous rollout collector that runs env stepping and GAE computation
/// in a background thread, sending completed batches through a channel.
///
/// The collector requires two external function pointers:
/// - `action_fn`: given flat observations `[n_envs * obs_dim]`, returns
///   `(actions_flat, log_probs)` where actions are `[n_envs * act_dim]`
///   and log_probs are `[n_envs]`.
/// - `value_fn`: given flat observations `[n_envs * obs_dim]`, returns
///   value estimates `[n_envs]`.
///
/// This design lets the Python side own the neural network while Rust handles
/// the tight env-stepping loop.
pub struct AsyncCollector {
    handle: Option<JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
}

impl AsyncCollector {
    /// Start the collector in a background thread.
    ///
    /// The collector will repeatedly:
    /// 1. Collect `n_steps` of experience from all envs
    /// 2. Compute GAE advantages and returns
    /// 3. Send the resulting `RolloutBatch` through `tx`
    ///
    /// It stops when `stop()` is called or the channel is disconnected.
    pub fn start(
        mut envs: Box<dyn BatchSteppable>,
        n_steps: usize,
        gamma: f64,
        gae_lambda: f64,
        tx: Sender<RolloutBatch>,
        value_fn: Arc<dyn Fn(&[f32]) -> Vec<f64> + Send + Sync>,
        action_fn: Arc<dyn Fn(&[f32]) -> (Vec<f32>, Vec<f64>) + Send + Sync>,
    ) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop = stop_flag.clone();

        let handle = thread::spawn(move || {
            let n_envs = envs.num_envs();

            // Derive obs_dim from the observation space
            let obs_dim = match envs.obs_space() {
                crate::env::spaces::ObsSpace::Discrete(_) => 1,
                crate::env::spaces::ObsSpace::Box { shape, .. } => shape.iter().product(),
                crate::env::spaces::ObsSpace::MultiDiscrete(v) => v.len(),
                crate::env::spaces::ObsSpace::Dict(entries) => entries.iter().map(|(_, d)| d).sum(),
            };

            // Derive act_dim from the action space
            let act_dim = match envs.action_space() {
                crate::env::spaces::ActionSpace::Discrete(_) => 1,
                crate::env::spaces::ActionSpace::Box { shape, .. } => shape.iter().product(),
                crate::env::spaces::ActionSpace::MultiDiscrete(v) => v.len(),
            };

            // Get initial observations
            let init_obs = match envs.reset_batch(None) {
                Ok(obs) => obs,
                Err(_) => return,
            };
            let mut current_obs: Vec<f32> =
                init_obs.into_iter().flat_map(|o| o.into_inner()).collect();

            while !stop.load(Ordering::Relaxed) {
                let total = n_steps * n_envs;

                let mut all_obs = Vec::with_capacity(total * obs_dim);
                let mut all_actions = Vec::with_capacity(total * act_dim);
                let mut all_rewards = Vec::with_capacity(total);
                let mut all_dones = Vec::with_capacity(total);
                let mut all_values = Vec::with_capacity(total);
                let mut all_log_probs = Vec::with_capacity(total);

                // Collect n_steps of experience
                let mut ok = true;
                for _ in 0..n_steps {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }

                    // Get values and actions from the policy
                    let values = value_fn(&current_obs);
                    let (actions_flat, log_probs) = action_fn(&current_obs);

                    // Convert flat actions to Action enum for stepping
                    let actions: Vec<Action> = match envs.action_space() {
                        crate::env::spaces::ActionSpace::Discrete(_) => actions_flat
                            .iter()
                            .map(|&a| Action::Discrete(a as u32))
                            .collect(),
                        crate::env::spaces::ActionSpace::Box { shape, .. } => {
                            let dim: usize = shape.iter().product();
                            actions_flat
                                .chunks(dim)
                                .map(|chunk| Action::Continuous(chunk.to_vec()))
                                .collect()
                        }
                        crate::env::spaces::ActionSpace::MultiDiscrete(_) => actions_flat
                            .iter()
                            .map(|&a| Action::Discrete(a as u32))
                            .collect(),
                    };

                    // Store current obs, actions, values, log_probs
                    all_obs.extend_from_slice(&current_obs);
                    all_actions.extend_from_slice(&actions_flat);
                    all_values.extend(&values);
                    all_log_probs.extend(&log_probs);

                    // Step environments
                    match envs.step_batch(&actions) {
                        Ok(transition) => {
                            for i in 0..n_envs {
                                all_rewards.push(transition.rewards[i]);
                                let done = if transition.terminated[i] || transition.truncated[i] {
                                    1.0
                                } else {
                                    0.0
                                };
                                all_dones.push(done);
                            }
                            // Update current observations (reuse allocation)
                            let mut offset = 0;
                            for obs_vec in transition.obs {
                                current_obs[offset..offset + obs_vec.len()]
                                    .copy_from_slice(&obs_vec);
                                offset += obs_vec.len();
                            }
                        }
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }

                if !ok {
                    break;
                }

                // Bootstrap value for GAE
                let last_values = value_fn(&current_obs);

                // Transpose step-major -> env-major for batched GAE
                let mut env_major_rewards = vec![0.0; total];
                let mut env_major_values = vec![0.0; total];
                let mut env_major_dones = vec![0.0; total];
                for t in 0..n_steps {
                    for e in 0..n_envs {
                        env_major_rewards[e * n_steps + t] = all_rewards[t * n_envs + e];
                        env_major_values[e * n_steps + t] = all_values[t * n_envs + e];
                        env_major_dones[e * n_steps + t] = all_dones[t * n_envs + e];
                    }
                }

                let (env_major_adv, env_major_ret) = gae::compute_gae_batched(
                    &env_major_rewards,
                    &env_major_values,
                    &env_major_dones,
                    &last_values,
                    n_steps,
                    gamma,
                    gae_lambda,
                );

                // Transpose back env-major -> step-major
                let mut advantages = vec![0.0; total];
                let mut returns = vec![0.0; total];
                for e in 0..n_envs {
                    for t in 0..n_steps {
                        advantages[t * n_envs + e] = env_major_adv[e * n_steps + t];
                        returns[t * n_envs + e] = env_major_ret[e * n_steps + t];
                    }
                }

                let batch = RolloutBatch {
                    observations: all_obs,
                    actions: all_actions,
                    rewards: all_rewards,
                    dones: all_dones,
                    log_probs: all_log_probs,
                    values: all_values,
                    advantages,
                    returns,
                    obs_dim,
                    act_dim,
                    n_steps,
                    n_envs,
                };

                // Send with backpressure, but stay responsive to stop(): a
                // plain blocking send() on a full channel never re-checks the
                // stop flag, so a stopped (non-draining) consumer would wedge
                // the thread inside send() and make stop()/Drop join() forever.
                // Poll with a timeout, re-checking the stop flag between tries.
                let mut pending = Some(batch);
                while let Some(b) = pending.take() {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    match tx.send_timeout(b, SEND_POLL_INTERVAL) {
                        Ok(()) => {}
                        Err(SendTimeoutError::Timeout(b)) => pending = Some(b),
                        Err(SendTimeoutError::Disconnected(_)) => return, // receiver dropped
                    }
                }
            }
        });

        Self {
            handle: Some(handle),
            stop_flag,
        }
    }

    /// Signal the collector to stop and wait for the thread to finish.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Check whether the collector has been asked to stop.
    pub fn is_stopped(&self) -> bool {
        self.stop_flag.load(Ordering::Relaxed)
    }
}

impl Drop for AsyncCollector {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::builtins::CartPole;
    use crate::env::parallel::VecEnv;
    use crate::env::RLEnv;
    use crate::pipeline::channel::Pipeline;
    use crate::seed::derive_seed;

    fn make_vec_env(n: usize, seed: u64) -> Box<dyn BatchSteppable> {
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| Box::new(CartPole::new(Some(derive_seed(seed, i)))) as Box<dyn RLEnv>)
            .collect();
        Box::new(VecEnv::new(envs).unwrap())
    }

    #[test]
    fn test_async_collector_produces_batches() {
        let pipe = Pipeline::new(4);
        let tx = pipe.sender();

        let value_fn: Arc<dyn Fn(&[f32]) -> Vec<f64> + Send + Sync> =
            Arc::new(|obs: &[f32]| vec![0.0; obs.len() / 4]); // CartPole obs_dim=4
        let action_fn: Arc<dyn Fn(&[f32]) -> (Vec<f32>, Vec<f64>) + Send + Sync> =
            Arc::new(|obs: &[f32]| {
                let n = obs.len() / 4;
                (vec![0.0; n], vec![0.0; n]) // always action 0
            });

        let mut collector = AsyncCollector::start(
            make_vec_env(2, 42),
            8, // n_steps
            0.99,
            0.95,
            tx,
            value_fn,
            action_fn,
        );

        // Should receive at least one batch
        let batch = pipe.recv().unwrap();
        assert_eq!(batch.n_steps, 8);
        assert_eq!(batch.n_envs, 2);
        assert_eq!(batch.obs_dim, 4);
        assert_eq!(batch.act_dim, 1);
        assert_eq!(batch.observations.len(), 8 * 2 * 4);
        assert_eq!(batch.rewards.len(), 8 * 2);
        assert_eq!(batch.advantages.len(), 8 * 2);

        collector.stop();
        assert!(collector.is_stopped());
    }

    #[test]
    fn test_async_collector_stop_is_idempotent() {
        let pipe = Pipeline::new(2);
        let tx = pipe.sender();

        let value_fn: Arc<dyn Fn(&[f32]) -> Vec<f64> + Send + Sync> =
            Arc::new(|obs: &[f32]| vec![0.0; obs.len() / 4]);
        let action_fn: Arc<dyn Fn(&[f32]) -> (Vec<f32>, Vec<f64>) + Send + Sync> =
            Arc::new(|obs: &[f32]| {
                let n = obs.len() / 4;
                (vec![0.0; n], vec![0.0; n])
            });

        let mut collector =
            AsyncCollector::start(make_vec_env(1, 0), 4, 0.99, 0.95, tx, value_fn, action_fn);

        collector.stop();
        collector.stop(); // should not panic
    }

    /// Regression: `stop()` must terminate even when the bounded channel is
    /// full and the consumer never drains it. Previously the collection thread
    /// blocked forever inside `send()` on the full channel and `stop()`'s
    /// `join()` deadlocked (it hung the whole pytest suite for hours via the
    /// `CandleCollector` binding). A watchdog thread bounds the wait so a
    /// regression FAILS fast instead of hanging the test runner.
    #[test]
    fn test_stop_terminates_when_channel_full_and_undrained() {
        use crossbeam_channel::bounded;

        // Tiny buffer that we deliberately never drain -> the collector fills
        // it and parks in send().
        let pipe = Pipeline::new(1);
        let tx = pipe.sender();

        let value_fn: Arc<dyn Fn(&[f32]) -> Vec<f64> + Send + Sync> =
            Arc::new(|obs: &[f32]| vec![0.0; obs.len() / 4]);
        let action_fn: Arc<dyn Fn(&[f32]) -> (Vec<f32>, Vec<f64>) + Send + Sync> =
            Arc::new(|obs: &[f32]| {
                let n = obs.len() / 4;
                (vec![0.0; n], vec![0.0; n])
            });

        let mut collector =
            AsyncCollector::start(make_vec_env(1, 7), 4, 0.99, 0.95, tx, value_fn, action_fn);

        // Let the collector fill the undrained channel and block in send().
        thread::sleep(Duration::from_millis(200));

        // Run stop() on a watchdog thread so a deadlock surfaces as a timeout,
        // not a hung suite.
        let (done_tx, done_rx) = bounded::<()>(1);
        let watchdog = thread::spawn(move || {
            collector.stop();
            let _ = done_tx.send(());
        });

        assert!(
            done_rx.recv_timeout(Duration::from_secs(5)).is_ok(),
            "AsyncCollector::stop() deadlocked on a full, undrained channel"
        );
        watchdog.join().unwrap();

        // Keep the receiver alive (channel connected, send genuinely blocked on
        // a full buffer) through the assertion above.
        drop(pipe);
    }

    #[test]
    fn test_async_collector_gae_values_are_finite() {
        let pipe = Pipeline::new(4);
        let tx = pipe.sender();

        let value_fn: Arc<dyn Fn(&[f32]) -> Vec<f64> + Send + Sync> =
            Arc::new(|obs: &[f32]| vec![0.5; obs.len() / 4]);
        let action_fn: Arc<dyn Fn(&[f32]) -> (Vec<f32>, Vec<f64>) + Send + Sync> =
            Arc::new(|obs: &[f32]| {
                let n = obs.len() / 4;
                (vec![1.0; n], vec![-0.5; n])
            });

        let mut collector =
            AsyncCollector::start(make_vec_env(4, 42), 16, 0.99, 0.95, tx, value_fn, action_fn);

        let batch = pipe.recv().unwrap();
        for &a in &batch.advantages {
            assert!(a.is_finite(), "advantage must be finite, got {a}");
        }
        for &r in &batch.returns {
            assert!(r.is_finite(), "return must be finite, got {r}");
        }

        collector.stop();
    }
}
