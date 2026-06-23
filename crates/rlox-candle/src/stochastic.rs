use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rlox_nn::{Activation, MLPConfig, NNError, StochasticPolicy, TensorData, TrainMetrics};

use crate::convert::*;
use crate::mlp::MLP;

const LOG_STD_MIN: f32 = -20.0;
const LOG_STD_MAX: f32 = 2.0;

pub struct CandleStochasticPolicy {
    shared: MLP,
    mean_head: Linear,
    log_std_head: Linear,
    varmap: VarMap,
    optimizer: candle_nn::AdamW,
    device: Device,
    act_dim: usize,
    lr: f64,
    #[allow(dead_code)]
    rng: ChaCha8Rng,
}

impl CandleStochasticPolicy {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden: usize,
        lr: f64,
        device: Device,
        seed: u64,
    ) -> Result<Self, NNError> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let shared_config = MLPConfig::new(obs_dim, hidden)
            .with_hidden(vec![hidden])
            .with_activation(Activation::ReLU);

        let shared = MLP::new(&shared_config, vb.pp("shared")).nn_err()?;
        let mean_head = linear(hidden, act_dim, vb.pp("mean")).nn_err()?;
        let log_std_head = linear(hidden, act_dim, vb.pp("log_std")).nn_err()?;

        let params = varmap.all_vars();
        let optimizer = candle_nn::AdamW::new(
            params,
            candle_nn::ParamsAdamW {
                lr,
                ..Default::default()
            },
        )
        .nn_err()?;

        Ok(Self {
            shared,
            mean_head,
            log_std_head,
            varmap,
            optimizer,
            device,
            act_dim,
            lr,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    fn forward(&self, obs: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let h = self.shared.forward(obs)?;
        let mean = self.mean_head.forward(&h)?;
        let log_std = self
            .log_std_head
            .forward(&h)?
            .clamp(LOG_STD_MIN, LOG_STD_MAX)?;
        Ok((mean, log_std))
    }
}

impl CandleStochasticPolicy {
    /// SAC actor gradient step with autograd flowing through the critic.
    ///
    /// Takes concrete `CandleTwinQ` to preserve gradient flow from Q-values
    /// back to actor parameters via the reparameterized actions.
    pub fn sac_actor_step(
        &mut self,
        obs: &TensorData,
        alpha: f32,
        critic: &crate::continuous_q::CandleTwinQ,
    ) -> Result<TrainMetrics, NNError> {
        let batch_size = obs.shape[0];
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let (mean, log_std) = self.forward(&obs_t).nn_err()?;
        let std = log_std.exp().nn_err()?;

        let eps = Tensor::randn(0.0_f32, 1.0, (batch_size, self.act_dim), &self.device).nn_err()?;
        let x_t = (&mean + &(&std * &eps).nn_err()?).nn_err()?;
        let y_t = x_t.tanh().nn_err()?;

        // Log-prob (differentiable) — use (x_t - mean) for correct values
        let var = std.sqr().nn_err()?;
        let residual = (&x_t - &mean).nn_err()?;
        let normal_lp = (&residual.sqr().nn_err()? / &var)
            .nn_err()?
            .neg()
            .nn_err()?
            / 2.0;
        let normal_lp = normal_lp.nn_err()?;
        let log_std_term = std.log().nn_err()?.neg().nn_err()?;
        let const_term = -0.5 * (2.0 * std::f32::consts::PI).ln();
        let normal_lp_full =
            (&(&normal_lp + &log_std_term).nn_err()? + const_term as f64).nn_err()?;

        let y_sq = y_t.sqr().nn_err()?;
        let one = Tensor::ones_like(&y_sq).nn_err()?;
        let correction = ((&one - &y_sq).nn_err()? + 1e-6_f64)
            .nn_err()?
            .log()
            .nn_err()?
            .neg()
            .nn_err()?;
        let log_prob = (&normal_lp_full + &correction).nn_err()?.sum(1).nn_err()?;

        // Q-values with full autograd through critic
        let (q1, q2) = critic.twin_q_forward(&obs_t, &y_t).nn_err()?;
        let q1 = q1.squeeze(1).nn_err()?;
        let q2 = q2.squeeze(1).nn_err()?;
        let q_min = q1.minimum(&q2).nn_err()?;

        let actor_loss = (&(&log_prob * alpha as f64).nn_err()? - &q_min)
            .nn_err()?
            .mean_all()
            .nn_err()?;

        self.optimizer.backward_step(&actor_loss).nn_err()?;

        let loss_val: f32 = actor_loss.to_scalar().nn_err()?;

        let mut metrics = TrainMetrics::new();
        metrics.insert("actor_loss", loss_val as f64);
        Ok(metrics)
    }
}

impl StochasticPolicy for CandleStochasticPolicy {
    fn sample_actions(&self, obs: &TensorData) -> Result<(TensorData, TensorData), NNError> {
        let batch_size = obs.shape[0];
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let (mean, log_std) = self.forward(&obs_t).nn_err()?;
        let std = log_std.exp().nn_err()?;

        let eps = Tensor::randn(0.0_f32, 1.0, (batch_size, self.act_dim), &self.device).nn_err()?;
        let x_t = (&mean + &(&std * &eps).nn_err()?).nn_err()?;
        let y_t = x_t.tanh().nn_err()?;

        let mean_data: Vec<f32> = mean.flatten_all().nn_err()?.to_vec1().nn_err()?;
        let std_data: Vec<f32> = std.flatten_all().nn_err()?.to_vec1().nn_err()?;
        let x_data: Vec<f32> = x_t.flatten_all().nn_err()?.to_vec1().nn_err()?;
        let y_data: Vec<f32> = y_t.flatten_all().nn_err()?.to_vec1().nn_err()?;

        let mut actions_flat = Vec::with_capacity(batch_size * self.act_dim);
        let mut log_probs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut lp_sum = 0.0_f32;
            for j in 0..self.act_dim {
                let idx = i * self.act_dim + j;
                let x = x_data[idx];
                let m = mean_data[idx];
                let s = std_data[idx];
                let y = y_data[idx];

                actions_flat.push(y);

                let normal_lp =
                    -0.5 * ((x - m) / s).powi(2) - s.ln() - 0.5 * (2.0 * std::f32::consts::PI).ln();
                let correction = -(1.0 - y * y + 1e-6).ln();
                lp_sum += normal_lp + correction;
            }
            log_probs.push(lp_sum);
        }

        Ok((
            TensorData::new(actions_flat, vec![batch_size, self.act_dim]),
            TensorData::new(log_probs, vec![batch_size]),
        ))
    }

    fn deterministic_action(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let (mean, _) = self.forward(&obs_t).nn_err()?;
        let action = mean.tanh().nn_err()?;
        from_tensor_2d(&action).nn_err()
    }

    fn learning_rate(&self) -> f32 {
        self.lr as f32
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr as f64;
        self.optimizer.set_learning_rate(lr as f64);
    }

    fn save(&self, path: &Path) -> Result<(), NNError> {
        self.varmap
            .save(path)
            .map_err(|e| NNError::Serialization(e.to_string()))
    }

    fn load(&mut self, path: &Path) -> Result<(), NNError> {
        self.varmap
            .load(path)
            .map_err(|e| NNError::Serialization(e.to_string()))
    }
}

unsafe impl Send for CandleStochasticPolicy {}
unsafe impl Sync for CandleStochasticPolicy {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_actions_shape() {
        let policy = CandleStochasticPolicy::new(3, 2, 64, 3e-4, Device::Cpu, 42).unwrap();
        let obs = TensorData::zeros(vec![8, 3]);
        let (actions, log_probs) = policy.sample_actions(&obs).unwrap();
        assert_eq!(actions.shape, vec![8, 2]);
        assert_eq!(log_probs.shape, vec![8]);
    }

    #[test]
    fn test_sample_in_range() {
        let policy = CandleStochasticPolicy::new(3, 1, 64, 3e-4, Device::Cpu, 42).unwrap();
        let obs = TensorData::zeros(vec![100, 3]);
        let (actions, _) = policy.sample_actions(&obs).unwrap();
        for &a in &actions.data {
            assert!((-1.0..=1.0).contains(&a), "out of range: {a}");
        }
    }

    #[test]
    fn test_deterministic_shape() {
        let policy = CandleStochasticPolicy::new(3, 2, 64, 3e-4, Device::Cpu, 42).unwrap();
        let obs = TensorData::zeros(vec![4, 3]);
        let actions = policy.deterministic_action(&obs).unwrap();
        assert_eq!(actions.shape, vec![4, 2]);
    }
}
