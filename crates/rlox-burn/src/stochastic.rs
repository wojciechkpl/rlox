use std::path::Path;

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rlox_nn::{Activation, MLPConfig, NNError, StochasticPolicy, TensorData, TrainMetrics};

use crate::convert::*;
use crate::mlp::{apply_activation, ActivationKind, MLPParams, MLP};

const LOG_STD_MIN: f32 = -20.0;
const LOG_STD_MAX: f32 = 2.0;

/// Learnable parameters for squashed Gaussian — Module-derivable.
#[derive(Module, Debug)]
pub struct SquashedGaussianParams<B: Backend> {
    shared: MLPParams<B>,
    mean_head: burn::nn::Linear<B>,
    log_std_head: burn::nn::Linear<B>,
}

/// Squashed Gaussian policy model for SAC.
#[derive(Debug)]
pub struct SquashedGaussianModel<B: Backend> {
    pub params: SquashedGaussianParams<B>,
    shared_activation: ActivationKind,
}

impl<B: Backend> SquashedGaussianModel<B> {
    pub fn new(obs_dim: usize, act_dim: usize, hidden: usize, device: &B::Device) -> Self {
        let shared_config = MLPConfig::new(obs_dim, hidden)
            .with_hidden(vec![hidden])
            .with_activation(Activation::ReLU);

        let shared_mlp = MLP::new(&shared_config, device);

        Self {
            params: SquashedGaussianParams {
                shared: shared_mlp.params,
                mean_head: burn::nn::LinearConfig::new(hidden, act_dim).init(device),
                log_std_head: burn::nn::LinearConfig::new(hidden, act_dim).init(device),
            },
            shared_activation: shared_config.activation.into(),
        }
    }

    fn shared_forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let layers = &self.params.shared.layers;
        let n = layers.len();
        let mut x = obs;
        for (i, layer) in layers.iter().enumerate() {
            x = layer.forward(x);
            if i < n - 1 {
                x = apply_activation(x, self.shared_activation);
            }
        }
        x
    }

    /// Returns (mean, log_std) tensors.
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.shared_forward(obs);
        let mean = self.params.mean_head.forward(h.clone());
        let log_std = self
            .params
            .log_std_head
            .forward(h)
            .clamp(LOG_STD_MIN, LOG_STD_MAX);
        (mean, log_std)
    }

    pub fn valid(&self) -> SquashedGaussianModel<B::InnerBackend>
    where
        B: AutodiffBackend,
    {
        SquashedGaussianModel {
            params: self.params.valid(),
            shared_activation: self.shared_activation,
        }
    }
}

impl<B: Backend> Clone for SquashedGaussianModel<B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            shared_activation: self.shared_activation,
        }
    }
}

#[allow(dead_code)]
pub struct BurnStochasticPolicy<B: AutodiffBackend> {
    model: SquashedGaussianModel<B>,
    optimizer: OptimizerAdaptor<Adam, SquashedGaussianParams<B>, B>,
    device: B::Device,
    act_dim: usize,
    lr: f32,
    rng: ChaCha8Rng,
}

impl<B: AutodiffBackend> BurnStochasticPolicy<B> {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden: usize,
        lr: f32,
        device: B::Device,
        seed: u64,
    ) -> Self {
        let model = SquashedGaussianModel::new(obs_dim, act_dim, hidden, &device);
        let optimizer = AdamConfig::new().init();

        Self {
            model,
            optimizer,
            device,
            act_dim,
            lr,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl<B: AutodiffBackend> BurnStochasticPolicy<B>
where
    B::Device: Clone,
{
    /// SAC actor gradient step with autograd flowing through the critic.
    ///
    /// This is an inherent method (not on the `StochasticPolicy` trait) because
    /// the gradient from `min(Q1, Q2)` must flow back through the critic's networks
    /// to the actor's parameters via the reparameterized actions. The trait boundary
    /// would sever this autograd chain by converting tensors to `TensorData`.
    pub fn sac_actor_step(
        &mut self,
        obs: &TensorData,
        alpha: f32,
        critic: &crate::continuous_q::BurnTwinQ<B>,
    ) -> Result<TrainMetrics, NNError> {
        let batch_size = obs.shape[0];

        // Forward with autograd
        let obs_t = to_tensor_2d::<B>(obs, &self.device);
        let (mean, log_std) = self.model.forward(obs_t.clone());
        let std = log_std.exp();

        // Reparameterized sample
        let eps = Tensor::<B, 2>::random(
            [batch_size, self.act_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );
        let x_t = mean.clone() + std.clone() * eps;
        let y_t = activation::tanh(x_t.clone());

        // Log-prob (differentiable) — use (x_t - mean) for correct log-prob values
        let residual = x_t - mean;
        let var = std.clone().powf_scalar(2.0);
        let normal_lp = (residual.powf_scalar(2.0) / var / (-2.0))
            - std.log()
            - (2.0 * std::f32::consts::PI).sqrt().ln();
        let tanh_correction =
            -(Tensor::ones_like(&y_t) - y_t.clone().powf_scalar(2.0) + 1e-6).log();
        let log_prob = (normal_lp + tanh_correction).sum_dim(1).squeeze::<1>(1);

        // Q-values with full autograd through critic
        let (q1, q2) = critic.twin_q_forward_ad(obs_t, y_t);
        let q1 = q1.squeeze::<1>(1);
        let q2 = q2.squeeze::<1>(1);
        let q_min = Tensor::min_pair(q1, q2);

        // Actor loss: alpha * log_prob - Q
        let actor_loss = (log_prob * alpha - q_min).mean();

        let grads = actor_loss.clone().backward();
        let grads = GradientsParams::from_grads(grads, &self.model.params);
        self.model.params = self
            .optimizer
            .step(self.lr.into(), self.model.params.clone(), grads);

        let loss_val: f32 = actor_loss.inner().into_data().to_vec::<f32>().unwrap()[0];

        let mut metrics = TrainMetrics::new();
        metrics.insert("actor_loss", loss_val as f64);
        Ok(metrics)
    }
}

impl<B: AutodiffBackend> StochasticPolicy for BurnStochasticPolicy<B>
where
    B::Device: Clone,
{
    fn sample_actions(&self, obs: &TensorData) -> Result<(TensorData, TensorData), NNError> {
        let batch_size = obs.shape[0];
        let dev: <B::InnerBackend as Backend>::Device = self.device.clone().into();
        let obs_t = to_tensor_2d::<B::InnerBackend>(obs, &dev);

        let (mean, log_std) = self.model.valid().forward(obs_t);
        let std = log_std.exp();

        // Reparameterized sample: x = mean + std * eps
        let eps = Tensor::<B::InnerBackend, 2>::random(
            [batch_size, self.act_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &dev,
        );
        let x_t = mean.clone() + std.clone() * eps;
        let y_t = activation::tanh(x_t.clone());

        let mean_data: Vec<f32> = mean.into_data().to_vec().unwrap();
        let std_data: Vec<f32> = std.into_data().to_vec().unwrap();
        let x_data: Vec<f32> = x_t.into_data().to_vec().unwrap();
        let y_data: Vec<f32> = y_t.into_data().to_vec().unwrap();

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

                // Normal log_prob
                let normal_lp =
                    -0.5 * ((x - m) / s).powi(2) - s.ln() - 0.5 * (2.0 * std::f32::consts::PI).ln();
                // Tanh correction
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
        let dev: <B::InnerBackend as Backend>::Device = self.device.clone().into();
        let obs_t = to_tensor_2d::<B::InnerBackend>(obs, &dev);
        let (mean, _) = self.model.valid().forward(obs_t);
        let action = activation::tanh(mean);
        Ok(from_tensor_2d(action))
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn save(&self, path: &Path) -> Result<(), NNError> {
        self.model
            .params
            .clone()
            .save_file(
                path,
                &burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new(),
            )
            .map_err(|e| NNError::Serialization(e.to_string()))
    }

    fn load(&mut self, path: &Path) -> Result<(), NNError> {
        let loaded = self
            .model
            .params
            .clone()
            .load_file(
                path,
                &burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new(),
                &self.device,
            )
            .map_err(|e| NNError::Serialization(e.to_string()))?;
        self.model.params = loaded;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;

    type TestBackend = Autodiff<NdArray>;
    type TestDevice = <NdArray as Backend>::Device;

    fn device() -> TestDevice {
        Default::default()
    }

    #[test]
    fn test_squashed_gaussian_shape() {
        let model = SquashedGaussianModel::<NdArray>::new(3, 1, 64, &device());
        let obs = Tensor::<NdArray, 2>::zeros([4, 3], &device());
        let (mean, log_std) = model.forward(obs);
        assert_eq!(mean.shape().dims, [4, 1]);
        assert_eq!(log_std.shape().dims, [4, 1]);
    }

    #[test]
    fn test_log_std_clamped() {
        let model = SquashedGaussianModel::<NdArray>::new(3, 1, 64, &device());
        let obs = Tensor::<NdArray, 2>::ones([4, 3], &device()) * 100.0;
        let (_, log_std) = model.forward(obs);
        let data: Vec<f32> = log_std.into_data().to_vec().unwrap();
        for &v in &data {
            assert!(
                (LOG_STD_MIN..=LOG_STD_MAX).contains(&v),
                "log_std out of range: {v}"
            );
        }
    }

    #[test]
    fn test_sample_actions_shape() {
        let policy = BurnStochasticPolicy::<TestBackend>::new(3, 2, 64, 3e-4, device().into(), 42);
        let obs = TensorData::zeros(vec![8, 3]);
        let (actions, log_probs) = policy.sample_actions(&obs).unwrap();
        assert_eq!(actions.shape, vec![8, 2]);
        assert_eq!(log_probs.shape, vec![8]);
    }

    #[test]
    fn test_sample_actions_in_range() {
        let policy = BurnStochasticPolicy::<TestBackend>::new(3, 1, 64, 3e-4, device().into(), 42);
        let obs = TensorData::zeros(vec![100, 3]);
        let (actions, _) = policy.sample_actions(&obs).unwrap();
        for &a in &actions.data {
            assert!(
                (-1.0..=1.0).contains(&a),
                "tanh-squashed action should be in [-1, 1]: {a}"
            );
        }
    }

    #[test]
    fn test_deterministic_action_shape() {
        let policy = BurnStochasticPolicy::<TestBackend>::new(3, 2, 64, 3e-4, device().into(), 42);
        let obs = TensorData::zeros(vec![4, 3]);
        let actions = policy.deterministic_action(&obs).unwrap();
        assert_eq!(actions.shape, vec![4, 2]);
    }

    #[test]
    fn test_deterministic_in_range() {
        let policy = BurnStochasticPolicy::<TestBackend>::new(3, 1, 64, 3e-4, device().into(), 42);
        let obs = TensorData::new(
            (0..300).map(|i| (i as f32) * 0.1 - 15.0).collect(),
            vec![100, 3],
        );
        let actions = policy.deterministic_action(&obs).unwrap();
        for &a in &actions.data {
            assert!((-1.0..=1.0).contains(&a), "action out of range: {a}");
        }
    }
}
