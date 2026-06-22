use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use rlox_nn::{ActorCritic, PPOStepConfig, TensorData};

type BurnBackend = Autodiff<NdArray>;

use rlox_candle::collector::SharedPolicy;
use rlox_core::env::builtins::CartPole;
use rlox_core::env::parallel::VecEnv;
use rlox_core::env::RLEnv;
use rlox_core::pipeline::channel::{Pipeline, RolloutBatch};
use rlox_core::pipeline::collector::AsyncCollector;
use rlox_core::seed::derive_seed;

/// Convert a `RolloutBatch` into a Python dict of numpy arrays.
fn rollout_batch_to_pydict<'py>(
    py: Python<'py>,
    batch: RolloutBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("observations", PyArray1::from_vec(py, batch.observations))?;
    dict.set_item("actions", PyArray1::from_vec(py, batch.actions))?;
    dict.set_item("rewards", PyArray1::from_vec(py, batch.rewards))?;
    dict.set_item("dones", PyArray1::from_vec(py, batch.dones))?;
    dict.set_item("log_probs", PyArray1::from_vec(py, batch.log_probs))?;
    dict.set_item("values", PyArray1::from_vec(py, batch.values))?;
    dict.set_item("advantages", PyArray1::from_vec(py, batch.advantages))?;
    dict.set_item("returns", PyArray1::from_vec(py, batch.returns))?;
    dict.set_item("obs_dim", batch.obs_dim)?;
    dict.set_item("act_dim", batch.act_dim)?;
    dict.set_item("n_steps", batch.n_steps)?;
    dict.set_item("n_envs", batch.n_envs)?;
    Ok(dict)
}

enum Backend {
    Burn(rlox_burn::actor_critic::BurnActorCritic<BurnBackend>),
    Candle(rlox_candle::actor_critic::CandleActorCritic),
}

/// A discrete actor-critic policy backed by a pure-Rust NN (Burn or Candle).
///
/// Args:
///     backend: ``"burn"`` or ``"candle"``
///     obs_dim: observation dimension
///     n_actions: number of discrete actions
///     hidden: hidden layer width (default 64)
///     lr: learning rate (default 2.5e-4)
///     seed: RNG seed (default 42)
#[pyclass(name = "ActorCritic", unsendable)]
pub struct PyActorCritic {
    inner: Backend,
    obs_dim: usize,
}

#[pymethods]
impl PyActorCritic {
    #[new]
    #[pyo3(signature = (backend, obs_dim, n_actions, hidden = 64, lr = 2.5e-4, seed = 42))]
    fn new(
        backend: &str,
        obs_dim: usize,
        n_actions: usize,
        hidden: usize,
        lr: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = match backend {
            "burn" => {
                let ac = rlox_burn::actor_critic::BurnActorCritic::<BurnBackend>::new(
                    obs_dim,
                    n_actions,
                    hidden,
                    lr as f32,
                    Default::default(),
                    seed,
                );
                Backend::Burn(ac)
            }
            "candle" => {
                let ac = rlox_candle::actor_critic::CandleActorCritic::new(
                    obs_dim,
                    n_actions,
                    hidden,
                    lr,
                    candle_core::Device::Cpu,
                    seed,
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Backend::Candle(ac)
            }
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown backend '{}'. Use 'burn' or 'candle'.",
                    backend
                )));
            }
        };
        Ok(Self { inner, obs_dim })
    }

    /// Sample actions from the policy (no gradient tracking).
    ///
    /// Args:
    ///     obs: flat float32 array of length ``batch * obs_dim``
    ///
    /// Returns:
    ///     ``(actions, log_probs)`` — two 1-D float32 arrays of length ``batch``.
    fn act<'py>(
        &self,
        py: Python<'py>,
        obs: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let obs_slice = obs.as_slice()?;
        let total = obs_slice.len();
        if total % self.obs_dim != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "obs length {} not divisible by obs_dim {}",
                total, self.obs_dim
            )));
        }
        let n = total / self.obs_dim;
        let td = TensorData::new(obs_slice.to_vec(), vec![n, self.obs_dim]);

        let out = match &self.inner {
            Backend::Burn(ac) => ac.act(&td),
            Backend::Candle(ac) => ac.act(&td),
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let actions = PyArray1::from_vec(py, out.actions.data);
        let log_probs = PyArray1::from_vec(py, out.log_probs.data);
        Ok((actions, log_probs))
    }

    /// Compute state values (no gradient tracking).
    ///
    /// Args:
    ///     obs: flat float32 array of length ``batch * obs_dim``
    ///
    /// Returns:
    ///     1-D float32 array of values, length ``batch``.
    fn value<'py>(
        &self,
        py: Python<'py>,
        obs: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let obs_slice = obs.as_slice()?;
        let n = obs_slice.len() / self.obs_dim;
        let td = TensorData::new(obs_slice.to_vec(), vec![n, self.obs_dim]);

        let out = match &self.inner {
            Backend::Burn(ac) => ac.value(&td),
            Backend::Candle(ac) => ac.value(&td),
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyArray1::from_vec(py, out.data))
    }

    /// Perform one PPO gradient step.
    ///
    /// All array arguments are flat float32 of length ``batch``,
    /// except ``obs`` which is ``batch * obs_dim``.
    ///
    /// Returns:
    ///     dict with keys ``policy_loss``, ``value_loss``, ``entropy``,
    ///     ``approx_kl``, ``clip_fraction``.
    #[pyo3(signature = (obs, actions, old_log_probs, advantages, returns, old_values,
                        clip_eps = 0.2, vf_coef = 0.5, ent_coef = 0.01,
                        max_grad_norm = 0.5, clip_vloss = true))]
    fn ppo_step<'py>(
        &mut self,
        py: Python<'py>,
        obs: PyReadonlyArray1<'py, f32>,
        actions: PyReadonlyArray1<'py, f32>,
        old_log_probs: PyReadonlyArray1<'py, f32>,
        advantages: PyReadonlyArray1<'py, f32>,
        returns: PyReadonlyArray1<'py, f32>,
        old_values: PyReadonlyArray1<'py, f32>,
        clip_eps: f32,
        vf_coef: f32,
        ent_coef: f32,
        max_grad_norm: f32,
        clip_vloss: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let obs_slice = obs.as_slice()?;
        let batch_size = obs_slice.len() / self.obs_dim;

        let obs_td = TensorData::new(obs_slice.to_vec(), vec![batch_size, self.obs_dim]);
        let actions_td = TensorData::new(actions.as_slice()?.to_vec(), vec![batch_size]);
        let old_lp_td = TensorData::new(old_log_probs.as_slice()?.to_vec(), vec![batch_size]);
        let adv_td = TensorData::new(advantages.as_slice()?.to_vec(), vec![batch_size]);
        let ret_td = TensorData::new(returns.as_slice()?.to_vec(), vec![batch_size]);
        let old_v_td = TensorData::new(old_values.as_slice()?.to_vec(), vec![batch_size]);

        let config = PPOStepConfig {
            clip_eps,
            vf_coef,
            ent_coef,
            max_grad_norm,
            clip_vloss,
        };

        let metrics = match &mut self.inner {
            Backend::Burn(ac) => ac.ppo_step(
                &obs_td,
                &actions_td,
                &old_lp_td,
                &adv_td,
                &ret_td,
                &old_v_td,
                &config,
            ),
            Backend::Candle(ac) => ac.ppo_step(
                &obs_td,
                &actions_td,
                &old_lp_td,
                &adv_td,
                &ret_td,
                &old_v_td,
                &config,
            ),
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        for (k, v) in &metrics.entries {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// Get the current learning rate.
    #[getter]
    fn learning_rate(&self) -> f32 {
        match &self.inner {
            Backend::Burn(ac) => ac.learning_rate(),
            Backend::Candle(ac) => ac.learning_rate(),
        }
    }

    /// Set the learning rate.
    #[setter]
    fn set_learning_rate(&mut self, lr: f32) {
        match &mut self.inner {
            Backend::Burn(ac) => ac.set_learning_rate(lr),
            Backend::Candle(ac) => ac.set_learning_rate(lr),
        }
    }
}

/// Candle-powered rollout collector.
///
/// Runs policy inference entirely in Rust (no Python calls during collection).
/// The collection loop runs in a background thread:
///   VecEnv.step → Candle.act → Candle.value → GAE → channel.send
///
/// Python receives completed RolloutBatch objects via ``recv()``,
/// runs PyTorch backward + optimizer step, then calls ``sync_weights()``
/// to update the Candle policy.
///
/// Args:
///     env_id: environment ID (currently "CartPole-v1" for native Rust env)
///     n_envs: number of parallel environments
///     obs_dim: observation dimension
///     n_actions: number of discrete actions
///     n_steps: rollout length per batch
///     hidden: hidden layer width (default 64)
///     lr: learning rate (default 2.5e-4)
///     gamma: discount factor (default 0.99)
///     gae_lambda: GAE lambda (default 0.95)
///     seed: random seed (default 42)
///     capacity: pipeline buffer capacity (default 2)
#[pyclass(name = "CandleCollector")]
pub struct PyCandleCollector {
    shared: SharedPolicy,
    pipeline: Pipeline,
    collector: Option<AsyncCollector>,
    #[allow(dead_code)]
    obs_dim: usize,
}

#[pymethods]
impl PyCandleCollector {
    #[new]
    #[pyo3(signature = (env_id, n_envs, obs_dim, n_actions, n_steps, hidden=64, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, seed=42, capacity=2))]
    fn new(
        env_id: &str,
        n_envs: usize,
        obs_dim: usize,
        n_actions: usize,
        n_steps: usize,
        hidden: usize,
        lr: f64,
        gamma: f64,
        gae_lambda: f64,
        seed: u64,
        capacity: usize,
    ) -> PyResult<Self> {
        if env_id != "CartPole-v1" && env_id != "CartPole" {
            return Err(PyValueError::new_err(format!(
                "CandleCollector currently supports CartPole-v1 only (native Rust env). Got '{env_id}'."
            )));
        }

        // Create Candle policy
        let ac = rlox_candle::actor_critic::CandleActorCritic::new(
            obs_dim,
            n_actions,
            hidden,
            lr,
            candle_core::Device::Cpu,
            seed,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let shared = SharedPolicy::new(ac);

        // Create pipeline
        let pipeline = Pipeline::new(capacity);

        // Create VecEnv
        let envs: Vec<Box<dyn RLEnv>> = (0..n_envs)
            .map(|i| Box::new(CartPole::new(Some(derive_seed(seed, i)))) as Box<dyn RLEnv>)
            .collect();
        let vec_env =
            Box::new(VecEnv::new(envs).map_err(|e| PyRuntimeError::new_err(e.to_string()))?);

        // Create Candle callbacks
        let (action_fn, value_fn) =
            rlox_candle::collector::make_candle_callbacks(shared.clone_ref(), obs_dim);

        // Start collector
        let collector = AsyncCollector::start(
            vec_env,
            n_steps,
            gamma,
            gae_lambda,
            pipeline.sender(),
            value_fn,
            action_fn,
        );

        Ok(Self {
            shared,
            pipeline,
            collector: Some(collector),
            obs_dim,
        })
    }

    /// Receive the next rollout batch (blocks until available).
    ///
    /// Returns a dict with numpy arrays: observations, actions, rewards,
    /// dones, log_probs, values, advantages, returns, plus shape metadata.
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let batch = py
            .allow_threads(|| self.pipeline.recv())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        rollout_batch_to_pydict(py, batch)
    }

    /// Try to receive a batch without blocking. Returns None if empty.
    fn try_recv<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.pipeline.try_recv() {
            Some(batch) => Ok(Some(rollout_batch_to_pydict(py, batch)?)),
            None => Ok(None),
        }
    }

    /// Synchronize weights from a flat f32 numpy array.
    ///
    /// Call this after each PyTorch optimizer step to update the Candle
    /// policy used for collection.
    fn sync_weights(&self, py: Python<'_>, weights: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        // Copy to an owned buffer so the write-lock acquisition can run without
        // the GIL (the collection thread holds read locks; never block the GIL
        // on a lock the Rust side needs).
        let w = weights.as_slice()?.to_vec();
        py.allow_threads(|| self.shared.sync_weights(&w))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Extract current Candle weights as a flat f32 numpy array.
    ///
    /// Use this to initialize PyTorch parameters from Candle's random init.
    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let w = self
            .shared
            .get_weights()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyArray1::from_vec(py, w))
    }

    /// Number of batches currently buffered.
    fn __len__(&self) -> usize {
        self.pipeline.len()
    }

    /// Stop the background collection thread.
    fn stop(&mut self, py: Python<'_>) {
        // Release the GIL while joining the collection thread so shutdown can
        // never wedge the interpreter (the deadlock this guards against was
        // root-caused and fixed in AsyncCollector, but holding the GIL across a
        // join is wrong regardless).
        if let Some(mut c) = self.collector.take() {
            py.allow_threads(|| c.stop());
        }
    }
}
