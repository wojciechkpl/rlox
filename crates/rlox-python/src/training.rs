use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rlox_core::pipeline::channel::{Pipeline, RolloutBatch};
use rlox_core::training::augmentation;
use rlox_core::training::cpd;
use rlox_core::training::gae;
use rlox_core::training::normalization::{ExponentialRunningStats, RunningStats, RunningStatsVec};
use rlox_core::training::packing;
use rlox_core::training::reward_shaping;
use rlox_core::training::vtrace;
use rlox_core::training::weight_ops;

/// Compute Generalized Advantage Estimation (GAE).
///
/// Args:
///     rewards: 1-D f64 array of rewards
///     values: 1-D f64 array of value estimates
///     dones: 1-D array (bool or f64; 0.0/False = not done, 1.0/True = done)
///     last_value: bootstrap value for the last step
///     gamma: discount factor
///     lam: GAE lambda parameter
///
/// Returns:
///     (advantages, returns) as a tuple of two numpy f64 arrays
#[pyfunction]
#[pyo3(signature = (rewards, values, dones, last_value, gamma, lam))]
pub fn compute_gae<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    dones: &Bound<'py, pyo3::types::PyAny>,
    last_value: f64,
    gamma: f64,
    lam: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let rewards_owned = rewards.as_slice()?.to_vec();
    let values_owned = values.as_slice()?.to_vec();

    // Accept dones as either f64 array or bool array
    let dones_vec: Vec<f64> = if let Ok(arr) = dones.extract::<PyReadonlyArray1<'py, f64>>() {
        arr.as_slice()?.to_vec()
    } else if let Ok(arr) = dones.extract::<PyReadonlyArray1<'py, bool>>() {
        arr.as_slice()?
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect()
    } else {
        let np_arr: &Bound<'py, pyo3::types::PyAny> = dones;
        let float_arr = np_arr
            .call_method1("astype", ("float64",))
            .map_err(|_| PyTypeError::new_err("dones must be a numpy array of float64 or bool"))?;
        let readonly: PyReadonlyArray1<'py, f64> = float_arr.extract()?;
        readonly.as_slice()?.to_vec()
    };

    let (advantages, returns) = py.allow_threads(|| {
        gae::compute_gae(
            &rewards_owned,
            &values_owned,
            &dones_vec,
            last_value,
            gamma,
            lam,
        )
    });

    Ok((
        PyArray1::from_vec(py, advantages),
        PyArray1::from_vec(py, returns),
    ))
}

/// Batched GAE: compute GAE for multiple environments in a single call.
///
/// All inputs are flat 1-D arrays of length `n_envs * n_steps`, laid out as
/// `[env0_step0, env0_step1, ..., env1_step0, ...]`.
/// `last_values` has length `n_envs`.
///
/// Returns `(advantages, returns)` each of length `n_envs * n_steps`.
#[pyfunction]
#[pyo3(signature = (rewards, values, dones, last_values, n_steps, gamma, lam))]
pub fn compute_gae_batched<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    dones: PyReadonlyArray1<'py, f64>,
    last_values: PyReadonlyArray1<'py, f64>,
    n_steps: usize,
    gamma: f64,
    lam: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let rewards_owned = rewards.as_slice()?.to_vec();
    let values_owned = values.as_slice()?.to_vec();
    let dones_owned = dones.as_slice()?.to_vec();
    let last_values_owned = last_values.as_slice()?.to_vec();

    let (advantages, returns) = py.allow_threads(|| {
        gae::compute_gae_batched(
            &rewards_owned,
            &values_owned,
            &dones_owned,
            &last_values_owned,
            n_steps,
            gamma,
            lam,
        )
    });
    Ok((
        PyArray1::from_vec(py, advantages),
        PyArray1::from_vec(py, returns),
    ))
}

/// Batched GAE in f32 — avoids f64 conversion overhead.
///
/// Same layout as `compute_gae_batched` but operates on f32.
#[pyfunction]
#[pyo3(signature = (rewards, values, dones, last_values, n_steps, gamma, lam))]
pub fn compute_gae_batched_f32<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f32>,
    values: PyReadonlyArray1<'py, f32>,
    dones: PyReadonlyArray1<'py, f32>,
    last_values: PyReadonlyArray1<'py, f32>,
    n_steps: usize,
    gamma: f32,
    lam: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let rewards_owned = rewards.as_slice()?.to_vec();
    let values_owned = values.as_slice()?.to_vec();
    let dones_owned = dones.as_slice()?.to_vec();
    let last_values_owned = last_values.as_slice()?.to_vec();

    let (advantages, returns) = py.allow_threads(|| {
        gae::compute_gae_batched_f32(
            &rewards_owned,
            &values_owned,
            &dones_owned,
            &last_values_owned,
            n_steps,
            gamma,
            lam,
        )
    });
    Ok((
        PyArray1::from_vec(py, advantages),
        PyArray1::from_vec(py, returns),
    ))
}

/// Python-facing RunningStats (Welford's algorithm).
#[pyclass(name = "RunningStats")]
pub struct PyRunningStats {
    inner: RunningStats,
}

#[pymethods]
impl PyRunningStats {
    #[new]
    fn new() -> Self {
        Self {
            inner: RunningStats::new(),
        }
    }

    fn update(&mut self, value: f64) {
        self.inner.update(value);
    }

    fn batch_update(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.inner.batch_update(values.as_slice()?);
        Ok(())
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn var(&self) -> f64 {
        self.inner.var()
    }

    fn std(&self) -> f64 {
        self.inner.std()
    }

    fn normalize(&self, value: f64) -> f64 {
        self.inner.normalize(value)
    }

    fn count(&self) -> u64 {
        self.inner.count()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Python-facing RunningStatsVec (per-dimension Welford's algorithm).
#[pyclass(name = "RunningStatsVec")]
pub struct PyRunningStatsVec {
    inner: RunningStatsVec,
}

#[pymethods]
impl PyRunningStatsVec {
    #[new]
    fn new(dim: usize) -> Self {
        Self {
            inner: RunningStatsVec::new(dim),
        }
    }

    /// Update with a single sample (1-D array of length `dim`).
    fn update(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        if slice.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected {} dimensions, got {}",
                self.inner.dim(),
                slice.len()
            )));
        }
        self.inner.update(slice);
        Ok(())
    }

    /// Update with a flat batch: array of length `batch_size * dim`.
    fn batch_update(&mut self, data: PyReadonlyArray1<'_, f64>, batch_size: usize) -> PyResult<()> {
        let slice = data.as_slice()?;
        if slice.len() != batch_size * self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected {} elements (batch_size={} * dim={}), got {}",
                batch_size * self.inner.dim(),
                batch_size,
                self.inner.dim(),
                slice.len()
            )));
        }
        self.inner.batch_update(slice, batch_size);
        Ok(())
    }

    /// Return the per-dimension mean as a numpy array.
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.mean())
    }

    /// Return the per-dimension population variance as a numpy array.
    fn var<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.var())
    }

    /// Return the per-dimension standard deviation as a numpy array.
    fn std<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.std())
    }

    /// Normalize a single sample: `(values - mean) / max(std, 1e-8)`.
    fn normalize<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = values.as_slice()?;
        if slice.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected {} dimensions, got {}",
                self.inner.dim(),
                slice.len()
            )));
        }
        Ok(PyArray1::from_vec(py, self.inner.normalize(slice)))
    }

    /// Normalize a flat batch: array of length `batch_size * dim`.
    fn normalize_batch<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = data.as_slice()?;
        if slice.len() != batch_size * self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected {} elements (batch_size={} * dim={}), got {}",
                batch_size * self.inner.dim(),
                batch_size,
                self.inner.dim(),
                slice.len()
            )));
        }
        Ok(PyArray1::from_vec(
            py,
            self.inner.normalize_batch(slice, batch_size),
        ))
    }

    fn count(&self) -> u64 {
        self.inner.count()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Pack variable-length sequences into fixed-size bins (first-fit-decreasing).
///
/// Args:
///     sequences: list of 1-D uint32 numpy arrays
///     max_length: maximum bin length
///
/// Returns:
///     list of dicts, each with keys: input_ids, attention_mask, position_ids, sequence_starts
#[pyfunction]
#[pyo3(signature = (sequences, max_length))]
pub fn pack_sequences<'py>(
    py: Python<'py>,
    sequences: Vec<PyReadonlyArray1<'py, u32>>,
    max_length: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let vecs: Vec<Vec<u32>> = sequences
        .iter()
        .map(|arr| arr.as_slice().map(|s| s.to_vec()))
        .collect::<Result<_, _>>()?;
    let slices: Vec<&[u32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let packed = packing::pack_sequences(&slices, max_length)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut result = Vec::with_capacity(packed.len());
    for batch in packed {
        let dict = PyDict::new(py);
        dict.set_item("input_ids", PyArray1::from_vec(py, batch.input_ids))?;
        dict.set_item(
            "attention_mask",
            PyArray1::from_vec(py, batch.attention_mask),
        )?;
        dict.set_item("position_ids", PyArray1::from_vec(py, batch.position_ids))?;
        dict.set_item(
            "sequence_starts",
            PyArray1::from_vec(
                py,
                batch
                    .sequence_starts
                    .into_iter()
                    .map(|s| s as u64)
                    .collect(),
            ),
        )?;
        result.push(dict);
    }
    Ok(result)
}

/// Compute V-trace targets and policy gradient advantages (Espeholt et al. 2018).
///
/// Args:
///     log_rhos: 1-D f32 array of log importance ratios log(pi/mu)
///     rewards: 1-D f32 array of rewards
///     values: 1-D f32 array of value estimates
///     dones: 1-D f32 array of episode termination flags (1.0 = done)
///     bootstrap_value: bootstrap value for the last step
///     gamma: discount factor
///     rho_bar: clipping threshold for importance weights (default 1.0)
///     c_bar: trace cutting threshold (default 1.0)
///
/// Returns:
///     (vs, pg_advantages) as a tuple of two numpy f32 arrays
#[pyfunction]
#[pyo3(signature = (log_rhos, rewards, values, dones, bootstrap_value, gamma, rho_bar=1.0, c_bar=1.0))]
pub fn compute_vtrace<'py>(
    py: Python<'py>,
    log_rhos: PyReadonlyArray1<'py, f32>,
    rewards: PyReadonlyArray1<'py, f32>,
    values: PyReadonlyArray1<'py, f32>,
    dones: PyReadonlyArray1<'py, f32>,
    bootstrap_value: f32,
    gamma: f32,
    rho_bar: f32,
    c_bar: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let log_rhos_owned = log_rhos.as_slice()?.to_vec();
    let rewards_owned = rewards.as_slice()?.to_vec();
    let values_owned = values.as_slice()?.to_vec();
    let dones_owned = dones.as_slice()?.to_vec();

    let (vs, pg_advantages) = py
        .allow_threads(|| {
            vtrace::compute_vtrace(
                &log_rhos_owned,
                &rewards_owned,
                &values_owned,
                &dones_owned,
                bootstrap_value,
                gamma,
                rho_bar,
                c_bar,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray1::from_vec(py, vs),
        PyArray1::from_vec(py, pg_advantages),
    ))
}

/// Python-facing RolloutBatch — a flat batch of rollout data.
///
/// All arrays are 1-D numpy arrays. Shape metadata (obs_dim, act_dim, n_steps,
/// n_envs) is stored as scalar attributes so the Python side can reshape.
#[pyclass(name = "RolloutBatch")]
pub struct PyRolloutBatch {
    inner: RolloutBatch,
}

#[pymethods]
impl PyRolloutBatch {
    /// Create a new RolloutBatch from flat numpy arrays and shape metadata.
    #[new]
    #[pyo3(signature = (observations, actions, rewards, dones, advantages, returns, obs_dim, act_dim, n_steps, n_envs, log_probs=None, values=None))]
    fn new(
        observations: PyReadonlyArray1<'_, f32>,
        actions: PyReadonlyArray1<'_, f32>,
        rewards: PyReadonlyArray1<'_, f64>,
        dones: PyReadonlyArray1<'_, f64>,
        advantages: PyReadonlyArray1<'_, f64>,
        returns: PyReadonlyArray1<'_, f64>,
        obs_dim: usize,
        act_dim: usize,
        n_steps: usize,
        n_envs: usize,
        log_probs: Option<PyReadonlyArray1<'_, f64>>,
        values: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let obs_slice = observations.as_slice()?;
        let act_slice = actions.as_slice()?;
        let rew_slice = rewards.as_slice()?;
        let don_slice = dones.as_slice()?;
        let adv_slice = advantages.as_slice()?;
        let ret_slice = returns.as_slice()?;

        let expected_obs = n_steps * n_envs * obs_dim;
        let expected_act = n_steps * n_envs * act_dim;
        let expected_flat = n_steps * n_envs;

        if obs_slice.len() != expected_obs {
            return Err(PyValueError::new_err(format!(
                "observations length {} != n_steps*n_envs*obs_dim={}",
                obs_slice.len(),
                expected_obs
            )));
        }
        if act_slice.len() != expected_act {
            return Err(PyValueError::new_err(format!(
                "actions length {} != n_steps*n_envs*act_dim={}",
                act_slice.len(),
                expected_act
            )));
        }
        if rew_slice.len() != expected_flat {
            return Err(PyValueError::new_err(format!(
                "rewards length {} != n_steps*n_envs={}",
                rew_slice.len(),
                expected_flat
            )));
        }

        let lp = match log_probs {
            Some(lp) => lp.as_slice()?.to_vec(),
            None => vec![0.0; expected_flat],
        };
        let vals = match values {
            Some(v) => v.as_slice()?.to_vec(),
            None => vec![0.0; expected_flat],
        };

        Ok(Self {
            inner: RolloutBatch {
                observations: obs_slice.to_vec(),
                actions: act_slice.to_vec(),
                rewards: rew_slice.to_vec(),
                dones: don_slice.to_vec(),
                log_probs: lp,
                values: vals,
                advantages: adv_slice.to_vec(),
                returns: ret_slice.to_vec(),
                obs_dim,
                act_dim,
                n_steps,
                n_envs,
            },
        })
    }

    #[getter]
    fn observations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.observations)
    }

    #[getter]
    fn actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.actions)
    }

    #[getter]
    fn rewards<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.rewards)
    }

    #[getter]
    fn dones<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.dones)
    }

    #[getter]
    fn log_probs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.log_probs)
    }

    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.values)
    }

    #[getter]
    fn advantages<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.advantages)
    }

    #[getter]
    fn returns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.returns)
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim
    }

    #[getter]
    fn act_dim(&self) -> usize {
        self.inner.act_dim
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.inner.n_envs
    }
}

/// Bounded experience pipeline for decoupled collection and training.
///
/// Wraps a crossbeam bounded channel. The collector side calls `send()`,
/// the learner side calls `recv()` or `try_recv()`.
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    inner: Pipeline,
}

#[pymethods]
impl PyPipeline {
    /// Create a new pipeline with the given buffer capacity.
    #[new]
    #[pyo3(signature = (capacity=4))]
    fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be >= 1"));
        }
        Ok(Self {
            inner: Pipeline::new(capacity),
        })
    }

    /// Send a batch into the pipeline (blocks if full).
    fn send(&self, batch: &PyRolloutBatch) -> PyResult<()> {
        self.inner
            .send(batch.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Try to receive a batch without blocking. Returns None if empty.
    fn try_recv(&self) -> Option<PyRolloutBatch> {
        self.inner.try_recv().map(|b| PyRolloutBatch { inner: b })
    }

    /// Receive a batch, blocking until one is available.
    fn recv(&self) -> PyResult<PyRolloutBatch> {
        self.inner
            .recv()
            .map(|b| PyRolloutBatch { inner: b })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Number of batches currently buffered in the channel.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the channel is currently empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Wave 2: Image Augmentation
// ---------------------------------------------------------------------------

/// Apply random shift augmentation to a batch of images (DrQ-v2).
///
/// Args:
///     images: flat f32 numpy array of shape (B * C * H * W,)
///     batch_size: number of images
///     channels: number of channels
///     height: image height
///     width: image width
///     pad: padding size (pixels)
///     seed: RNG seed
///
/// Returns:
///     Augmented images as flat f32 numpy array, same shape as input.
#[pyfunction]
#[pyo3(signature = (images, batch_size, channels, height, width, pad, seed))]
pub fn random_shift_batch<'py>(
    py: Python<'py>,
    images: PyReadonlyArray1<'py, f32>,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    pad: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let images_owned = images.as_slice()?.to_vec();
    let result = py
        .allow_threads(|| {
            augmentation::random_shift_batch(
                &images_owned,
                batch_size,
                channels,
                height,
                width,
                pad,
                seed,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

// ---------------------------------------------------------------------------
// Wave 2: Reward Shaping
// ---------------------------------------------------------------------------

/// Compute PBRS shaped rewards: r' = r + gamma * Phi(s') - Phi(s).
///
/// At episode boundaries (dones[i] == 1.0), the potential difference
/// is zeroed out: r'_i = r_i (no shaping across episode boundaries).
#[pyfunction]
#[pyo3(signature = (rewards, potentials_current, potentials_next, gamma, dones))]
pub fn shape_rewards_pbrs<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
    potentials_current: PyReadonlyArray1<'py, f64>,
    potentials_next: PyReadonlyArray1<'py, f64>,
    gamma: f64,
    dones: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r = rewards.as_slice()?.to_vec();
    let pc = potentials_current.as_slice()?.to_vec();
    let pn = potentials_next.as_slice()?.to_vec();
    let d = dones.as_slice()?.to_vec();
    let result = py
        .allow_threads(|| reward_shaping::shape_rewards_pbrs(&r, &pc, &pn, gamma, &d))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Compute goal-distance potentials: Phi(s) = -scale * ||s[goal_slice] - goal||_2.
#[pyfunction]
#[pyo3(signature = (observations, goal, obs_dim, goal_start, goal_dim, scale))]
pub fn compute_goal_distance_potentials<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray1<'py, f64>,
    goal: PyReadonlyArray1<'py, f64>,
    obs_dim: usize,
    goal_start: usize,
    goal_dim: usize,
    scale: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obs = observations.as_slice()?.to_vec();
    let g = goal.as_slice()?.to_vec();
    let result = py
        .allow_threads(|| {
            reward_shaping::compute_goal_distance_potentials(
                &obs, &g, obs_dim, goal_start, goal_dim, scale,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

// ---------------------------------------------------------------------------
// Wave 2: Weight Operations
// ---------------------------------------------------------------------------

/// Reptile weight update: meta_params += lr * (task_params - meta_params).
///
/// Modifies meta_params in-place.
#[pyfunction]
#[pyo3(signature = (meta_params, task_params, meta_lr))]
pub fn reptile_update<'py>(
    _py: Python<'py>,
    meta_params: &Bound<'py, PyArray1<f32>>,
    task_params: PyReadonlyArray1<'py, f32>,
    meta_lr: f32,
) -> PyResult<()> {
    let task_slice = task_params.as_slice()?;
    // Safety: readwrite() ensures exclusive access; as_slice_mut() requires contiguous data
    let mut binding = unsafe { meta_params.as_array_mut() };
    let meta_slice = binding
        .as_slice_mut()
        .ok_or_else(|| PyRuntimeError::new_err("meta_params must be a contiguous array"))?;
    weight_ops::reptile_update(meta_slice, task_slice, meta_lr)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Polyak (EMA) update: target = tau * source + (1-tau) * target.
///
/// Modifies target in-place.
#[pyfunction]
#[pyo3(signature = (target, source, tau))]
pub fn polyak_update<'py>(
    _py: Python<'py>,
    target: &Bound<'py, PyArray1<f32>>,
    source: PyReadonlyArray1<'py, f32>,
    tau: f32,
) -> PyResult<()> {
    let source_slice = source.as_slice()?;
    // Safety: we have exclusive Python-side access via function signature
    let mut binding = unsafe { target.as_array_mut() };
    let target_slice = binding
        .as_slice_mut()
        .ok_or_else(|| PyRuntimeError::new_err("target must be a contiguous array"))?;
    weight_ops::polyak_update(target_slice, source_slice, tau)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Average weight vectors. Returns a new numpy array.
#[pyfunction]
#[pyo3(signature = (vectors,))]
pub fn average_weight_vectors<'py>(
    py: Python<'py>,
    vectors: Vec<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let owned: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .collect::<Result<_, _>>()?;
    let slices: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
    let result = weight_ops::average_weight_vectors(&slices)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

// ---------------------------------------------------------------------------
// EMA Running Statistics (non-stationary RL)
// ---------------------------------------------------------------------------

/// Exponential Moving Average running statistics for non-stationary signals.
///
/// Unlike standard RunningStats (Welford) which weights all observations equally,
/// EMA gives exponentially more weight to recent observations, making it suitable
/// for tracking non-stationary distributions.
///
/// Args:
///     alpha: Smoothing factor in (0, 1). Higher = more responsive.
///
/// Alternative constructors:
///     EmaRunningStats.from_window(N) — alpha = 2/(N+1)
///     EmaRunningStats.from_halflife(h) — weight decays 50% every h steps
#[pyclass(name = "EmaRunningStats")]
pub struct PyEmaRunningStats {
    inner: ExponentialRunningStats,
}

#[pymethods]
impl PyEmaRunningStats {
    #[new]
    fn new(alpha: f64) -> PyResult<Self> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(PyValueError::new_err("alpha must be in (0, 1)"));
        }
        Ok(Self {
            inner: ExponentialRunningStats::new(alpha),
        })
    }

    /// Create from equivalent window size: alpha = 2/(N+1).
    #[staticmethod]
    fn from_window(window: usize) -> PyResult<Self> {
        if window < 1 {
            return Err(PyValueError::new_err("window must be >= 1"));
        }
        Ok(Self {
            inner: ExponentialRunningStats::from_window(window),
        })
    }

    /// Create from half-life (steps for weight to decay by 50%).
    #[staticmethod]
    fn from_halflife(halflife: f64) -> PyResult<Self> {
        if halflife <= 0.0 {
            return Err(PyValueError::new_err("halflife must be > 0"));
        }
        Ok(Self {
            inner: ExponentialRunningStats::from_halflife(halflife),
        })
    }

    /// Update with a single observation.
    fn update(&mut self, value: f64) {
        self.inner.update(value);
    }

    /// Update with a batch of observations.
    fn batch_update(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        self.inner.batch_update(slice);
        Ok(())
    }

    /// Current EMA mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Current EMA variance.
    #[getter]
    fn var(&self) -> f64 {
        self.inner.var()
    }

    /// Current EMA standard deviation.
    #[getter]
    fn std(&self) -> f64 {
        self.inner.std()
    }

    /// Normalize a value using current EMA mean and std.
    fn normalize(&self, value: f64) -> f64 {
        self.inner.normalize(value)
    }

    /// Number of observations seen.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Smoothing factor alpha.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ---------------------------------------------------------------------------
// CUSUM Change-Point Detector (non-stationary RL)
// ---------------------------------------------------------------------------

/// Two-sided CUSUM change-point detector.
///
/// Detects shifts in the mean of a streaming signal. Feed it reward or loss
/// values; when it returns True, the underlying distribution has likely shifted.
///
/// Args:
///     mu_0: Reference level (expected mean under no-change hypothesis)
///     delta: Allowance parameter (minimum shift to detect). Typical: 0.5 * expected_shift
///     h: Detection threshold. Higher = fewer false alarms, slower detection. Typical: 4-8
///
/// Alternative constructor:
///     CusumDetector.with_burnin(burnin, delta, h) — estimates mu_0 from first N samples
#[pyclass(name = "CusumDetector")]
pub struct PyCusumDetector {
    inner: cpd::CusumDetector,
}

#[pymethods]
impl PyCusumDetector {
    #[new]
    fn new(mu_0: f64, delta: f64, h: f64) -> PyResult<Self> {
        if delta < 0.0 {
            return Err(PyValueError::new_err("delta must be non-negative"));
        }
        if h <= 0.0 {
            return Err(PyValueError::new_err("h must be positive"));
        }
        Ok(Self {
            inner: cpd::CusumDetector::new(mu_0, delta, h),
        })
    }

    /// Create a detector that estimates mu_0 from the first `burnin` samples.
    #[staticmethod]
    fn with_burnin(burnin: u64, delta: f64, h: f64) -> PyResult<Self> {
        if burnin == 0 {
            return Err(PyValueError::new_err("burnin must be > 0"));
        }
        if delta < 0.0 {
            return Err(PyValueError::new_err("delta must be non-negative"));
        }
        if h <= 0.0 {
            return Err(PyValueError::new_err("h must be positive"));
        }
        Ok(Self {
            inner: cpd::CusumDetector::with_burnin(burnin, delta, h),
        })
    }

    /// Feed one observation. Returns True if a change-point alarm fires.
    fn update(&mut self, value: f64) -> bool {
        self.inner.update(value)
    }

    /// Feed a batch. Returns the index of the first alarm, or None.
    fn batch_update(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<Option<usize>> {
        let slice = values.as_slice()?;
        Ok(self.inner.batch_update(slice))
    }

    /// Reset CUSUM statistics (call after handling an alarm).
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Reset and re-estimate mu_0 from next burnin samples.
    fn reset_with_burnin(&mut self) {
        self.inner.reset_with_burnin();
    }

    /// Set a new reference level manually.
    fn set_mu(&mut self, mu_0: f64) {
        self.inner.set_mu(mu_0);
    }

    /// Current upward CUSUM statistic.
    #[getter]
    fn s_pos(&self) -> f64 {
        self.inner.s_pos()
    }

    /// Current downward CUSUM statistic.
    #[getter]
    fn s_neg(&self) -> f64 {
        self.inner.s_neg()
    }

    /// Reference level.
    #[getter]
    fn mu_0(&self) -> f64 {
        self.inner.mu_0()
    }

    /// Total observations processed.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Total alarms fired.
    #[getter]
    fn alarm_count(&self) -> u64 {
        self.inner.alarm_count()
    }

    /// Whether burn-in is complete and detection is active.
    #[getter]
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }
}
