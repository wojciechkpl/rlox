//! Candle-powered rollout collector.
//!
//! Connects `CandleActorCritic` directly to `AsyncCollector` so that
//! policy inference runs in pure Rust — no Python calls during collection.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  Background Thread (pure Rust)          │
//! │                                         │
//! │  loop {                                 │
//! │    obs = VecEnv.step_all()     ~96μs    │
//! │    act = Candle.act(obs)       ~15μs    │  ← no Python dispatch
//! │    val = Candle.value(obs)     ~10μs    │
//! │    buffer.push(obs, act, val)           │
//! │    GAE = compute_gae_batched() ~5μs     │
//! │    channel.send(batch)                  │
//! │  }                                      │
//! └─────────────────────────────────────────┘
//!         ↕ crossbeam channel
//! ┌─────────────────────────────────────────┐
//! │  Python Main Thread                     │
//! │                                         │
//! │  batch = collector.recv()               │
//! │  loss = ppo_loss(batch)    (PyTorch)    │
//! │  loss.backward()                        │
//! │  optimizer.step()                       │
//! │  collector.sync_weights(flat_params)    │
//! └─────────────────────────────────────────┘
//! ```

use std::sync::{Arc, RwLock};

use rlox_nn::{ActorCritic, TensorData};

use crate::actor_critic::CandleActorCritic;

/// A shared, thread-safe function that maps flat observations to per-env value estimates.
type ValueFn = Arc<dyn Fn(&[f32]) -> Vec<f64> + Send + Sync>;

/// A shared, thread-safe function that maps flat observations to (actions, log-probs).
type ActionFn = Arc<dyn Fn(&[f32]) -> (Vec<f32>, Vec<f64>) + Send + Sync>;

/// Shared policy wrapper that allows the collection thread to read the policy
/// while the main thread updates weights.
pub struct SharedPolicy {
    inner: Arc<RwLock<CandleActorCritic>>,
}

impl SharedPolicy {
    pub fn new(policy: CandleActorCritic) -> Self {
        Self {
            inner: Arc::new(RwLock::new(policy)),
        }
    }

    /// Get a clone of the Arc for the collection thread.
    pub fn clone_ref(&self) -> Arc<RwLock<CandleActorCritic>> {
        self.inner.clone()
    }

    /// Synchronize weights from a flat f32 buffer (exported from PyTorch).
    ///
    /// The buffer must contain all parameters in the same order as
    /// `CandleActorCritic`'s VarMap, flattened and concatenated.
    pub fn sync_weights(&self, flat_params: &[f32]) -> Result<(), rlox_nn::NNError> {
        let mut policy = self
            .inner
            .write()
            .map_err(|e| rlox_nn::NNError::Backend(format!("Failed to acquire write lock: {e}")))?;
        load_flat_params(&mut policy, flat_params)
    }

    /// Extract weights as a flat f32 buffer (for PyTorch initialization).
    pub fn get_weights(&self) -> Result<Vec<f32>, rlox_nn::NNError> {
        let policy = self
            .inner
            .read()
            .map_err(|e| rlox_nn::NNError::Backend(format!("Failed to acquire read lock: {e}")))?;
        extract_flat_params(&policy)
    }
}

/// Build `action_fn` and `value_fn` closures that read from a shared policy.
///
/// These closures are compatible with [`AsyncCollector::start`] and perform
/// policy inference entirely in Rust (no GIL, no Python dispatch).
pub fn make_candle_callbacks(
    policy: Arc<RwLock<CandleActorCritic>>,
    obs_dim: usize,
) -> (ActionFn, ValueFn) {
    let policy_act = policy.clone();
    let policy_val = policy;

    let action_fn: ActionFn = Arc::new(move |obs_flat: &[f32]| {
        let n_envs = obs_flat.len() / obs_dim;
        let obs = TensorData::new(obs_flat.to_vec(), vec![n_envs, obs_dim]);

        let p = policy_act.read().unwrap();
        let output = p.act(&obs).expect("Candle act() failed");

        let actions = output.actions.data;
        let log_probs: Vec<f64> = output.log_probs.data.iter().map(|&x| x as f64).collect();
        (actions, log_probs)
    });

    let value_fn: ValueFn = Arc::new(move |obs_flat: &[f32]| {
        let n_envs = obs_flat.len() / obs_dim;
        let obs = TensorData::new(obs_flat.to_vec(), vec![n_envs, obs_dim]);

        let p = policy_val.read().unwrap();
        let values = p.value(&obs).expect("Candle value() failed");
        values.data.iter().map(|&x| x as f64).collect()
    });

    (action_fn, value_fn)
}

/// Load parameters from a flat f32 buffer into a CandleActorCritic's VarMap.
fn load_flat_params(
    policy: &mut CandleActorCritic,
    flat_params: &[f32],
) -> Result<(), rlox_nn::NNError> {
    let vars = policy.varmap.all_vars();
    let mut offset = 0;

    for var in &vars {
        let numel = var.elem_count();
        if offset + numel > flat_params.len() {
            return Err(rlox_nn::NNError::ShapeMismatch {
                expected: format!("at least {} elements", offset + numel),
                got: format!("{} elements", flat_params.len()),
            });
        }

        let slice = &flat_params[offset..offset + numel];
        let shape = var.dims();
        let tensor = candle_core::Tensor::from_vec(slice.to_vec(), shape, var.device())
            .map_err(|e| rlox_nn::NNError::Backend(e.to_string()))?;
        var.set(&tensor)
            .map_err(|e| rlox_nn::NNError::Backend(e.to_string()))?;
        offset += numel;
    }

    if offset != flat_params.len() {
        return Err(rlox_nn::NNError::ShapeMismatch {
            expected: format!("{} total elements", offset),
            got: format!("{} elements", flat_params.len()),
        });
    }

    Ok(())
}

/// Extract all parameters from a CandleActorCritic as a flat f32 buffer.
fn extract_flat_params(policy: &CandleActorCritic) -> Result<Vec<f32>, rlox_nn::NNError> {
    let vars = policy.varmap.all_vars();
    let total: usize = vars.iter().map(|v| v.elem_count()).sum();
    let mut flat = Vec::with_capacity(total);

    for var in &vars {
        let data: Vec<f32> = var
            .flatten_all()
            .map_err(|e| rlox_nn::NNError::Backend(e.to_string()))?
            .to_vec1()
            .map_err(|e| rlox_nn::NNError::Backend(e.to_string()))?;
        flat.extend(data);
    }

    Ok(flat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rlox_nn::ActorCritic;

    #[test]
    fn test_shared_policy_act() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let shared = SharedPolicy::new(ac);
        let policy_ref = shared.clone_ref();

        let obs = TensorData::zeros(vec![8, 4]);
        let p = policy_ref.read().unwrap();
        let result = p.act(&obs).unwrap();
        assert_eq!(result.actions.shape, vec![8]);
    }

    #[test]
    fn test_weight_roundtrip() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let shared = SharedPolicy::new(ac);

        // Extract weights
        let weights = shared.get_weights().unwrap();
        assert!(!weights.is_empty());

        // Sync back (should be a no-op identity)
        shared.sync_weights(&weights).unwrap();

        // Verify still works
        let weights2 = shared.get_weights().unwrap();
        assert_eq!(weights.len(), weights2.len());
        for (a, b) in weights.iter().zip(weights2.iter()) {
            assert!((a - b).abs() < 1e-6, "Weight mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_sync_weights_wrong_size() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let shared = SharedPolicy::new(ac);

        let result = shared.sync_weights(&[1.0, 2.0, 3.0]); // too few
        assert!(result.is_err());
    }

    #[test]
    fn test_make_candle_callbacks() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let shared = SharedPolicy::new(ac);
        let (action_fn, value_fn) = make_candle_callbacks(shared.clone_ref(), 4);

        // 2 envs, obs_dim=4
        let obs = vec![0.0f32; 8];
        let (actions, log_probs) = action_fn(&obs);
        assert_eq!(actions.len(), 2);
        assert_eq!(log_probs.len(), 2);

        let values = value_fn(&obs);
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_candle_callbacks_with_async_collector() {
        use rlox_core::env::builtins::CartPole;
        use rlox_core::env::parallel::VecEnv;
        use rlox_core::env::RLEnv;
        use rlox_core::pipeline::channel::Pipeline;
        use rlox_core::pipeline::collector::AsyncCollector;
        use rlox_core::seed::derive_seed;

        let n_envs = 2;
        let envs: Vec<Box<dyn RLEnv>> = (0..n_envs)
            .map(|i| Box::new(CartPole::new(Some(derive_seed(42, i)))) as Box<dyn RLEnv>)
            .collect();
        let vec_env = Box::new(VecEnv::new(envs).unwrap());

        // Create Candle policy and callbacks
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let shared = SharedPolicy::new(ac);
        let (action_fn, value_fn) = make_candle_callbacks(shared.clone_ref(), 4);

        // Run async collector with Candle callbacks
        let pipe = Pipeline::new(4);
        let tx = pipe.sender();
        let mut collector = AsyncCollector::start(vec_env, 16, 0.99, 0.95, tx, value_fn, action_fn);

        // Should receive a batch with log_probs and values
        let batch = pipe.recv().unwrap();
        assert_eq!(batch.n_steps, 16);
        assert_eq!(batch.n_envs, 2);
        assert_eq!(batch.observations.len(), 16 * 2 * 4);
        assert_eq!(batch.log_probs.len(), 16 * 2);
        assert_eq!(batch.values.len(), 16 * 2);
        assert_eq!(batch.advantages.len(), 16 * 2);

        for &lp in &batch.log_probs {
            assert!(lp.is_finite(), "log_prob must be finite");
        }
        for &v in &batch.values {
            assert!(v.is_finite(), "value must be finite");
        }

        // Weight sync should work while collector is running
        let weights = shared.get_weights().unwrap();
        shared.sync_weights(&weights).unwrap();

        collector.stop();
    }
}
