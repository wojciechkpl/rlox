use std::cell::RefCell;
use std::path::Path;

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rlox_nn::distributions::{categorical_entropy, categorical_log_prob, categorical_sample};
use rlox_nn::{
    ActionOutput, Activation, EvalOutput, MLPConfig, NNError, PPOStepConfig, TensorData,
    TrainMetrics,
};

use crate::convert::*;
use crate::mlp::{ActivationKind, MLPParams, MLP};

/// Learnable parameters only — Module-derivable.
#[derive(Module, Debug)]
pub struct DiscreteActorCriticParams<B: Backend> {
    actor: MLPParams<B>,
    critic: MLPParams<B>,
}

/// Discrete actor-critic for PPO using Burn.
/// Wraps learnable params with non-Module config (activation kinds).
#[derive(Debug)]
pub struct DiscreteActorCriticModel<B: Backend> {
    pub params: DiscreteActorCriticParams<B>,
    actor_activation: ActivationKind,
    critic_activation: ActivationKind,
}

impl<B: Backend> DiscreteActorCriticModel<B> {
    pub fn new(obs_dim: usize, n_actions: usize, hidden: usize, device: &B::Device) -> Self {
        let actor_config = MLPConfig::new(obs_dim, n_actions)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::Tanh);
        let critic_config = MLPConfig::new(obs_dim, 1)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::Tanh);

        let actor_mlp = MLP::new(&actor_config, device);
        let critic_mlp = MLP::new(&critic_config, device);

        Self {
            params: DiscreteActorCriticParams {
                actor: actor_mlp.params,
                critic: critic_mlp.params,
            },
            actor_activation: actor_config.activation.into(),
            critic_activation: critic_config.activation.into(),
        }
    }

    fn mlp_forward(
        layers: &[burn::nn::Linear<B>],
        input: Tensor<B, 2>,
        act: ActivationKind,
    ) -> Tensor<B, 2> {
        let n = layers.len();
        let mut x = input;
        for (i, layer) in layers.iter().enumerate() {
            x = layer.forward(x);
            if i < n - 1 {
                x = crate::mlp::apply_activation(x, act);
            }
        }
        x
    }

    pub fn actor_forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        Self::mlp_forward(&self.params.actor.layers, obs, self.actor_activation)
    }

    pub fn critic_forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        Self::mlp_forward(&self.params.critic.layers, obs, self.critic_activation)
    }

    pub fn valid(&self) -> DiscreteActorCriticModel<B::InnerBackend>
    where
        B: AutodiffBackend,
    {
        DiscreteActorCriticModel {
            params: self.params.valid(),
            actor_activation: self.actor_activation,
            critic_activation: self.critic_activation,
        }
    }
}

impl<B: Backend> Clone for DiscreteActorCriticModel<B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            actor_activation: self.actor_activation,
            critic_activation: self.critic_activation,
        }
    }
}

pub struct BurnActorCritic<B: AutodiffBackend> {
    model: DiscreteActorCriticModel<B>,
    optimizer: OptimizerAdaptor<Adam, DiscreteActorCriticParams<B>, B>,
    device: B::Device,
    lr: f32,
    n_actions: usize,
    rng: RefCell<ChaCha8Rng>,
}

impl<B: AutodiffBackend> BurnActorCritic<B> {
    pub fn new(
        obs_dim: usize,
        n_actions: usize,
        hidden: usize,
        lr: f32,
        device: B::Device,
        seed: u64,
    ) -> Self {
        let model = DiscreteActorCriticModel::new(obs_dim, n_actions, hidden, &device);
        let optimizer = AdamConfig::new().with_epsilon(1e-5).init();

        Self {
            model,
            optimizer,
            device,
            lr,
            n_actions,
            rng: RefCell::new(ChaCha8Rng::seed_from_u64(seed)),
        }
    }

    fn compute_logits_np(&self, obs: &TensorData) -> Vec<Vec<f32>> {
        let batch_size = obs.shape[0];
        let obs_tensor = to_tensor_2d::<B::InnerBackend>(obs, &self.device.clone().into());
        let logits = self.model.valid().actor_forward(obs_tensor);
        let logits_data: Vec<f32> = logits.into_data().to_vec().unwrap();

        (0..batch_size)
            .map(|i| logits_data[i * self.n_actions..(i + 1) * self.n_actions].to_vec())
            .collect()
    }
}

impl<B: AutodiffBackend> rlox_nn::ActorCritic for BurnActorCritic<B>
where
    B::Device: Clone,
{
    fn act(&self, obs: &TensorData) -> Result<ActionOutput, NNError> {
        if obs.shape.len() != 2 {
            return Err(NNError::ShapeMismatch {
                expected: "2D [batch, obs_dim]".into(),
                got: format!("{:?}", obs.shape),
            });
        }

        let batch_size = obs.shape[0];
        let batch_logits = self.compute_logits_np(obs);

        let mut actions = Vec::with_capacity(batch_size);
        let mut log_probs = Vec::with_capacity(batch_size);

        let mut rng = self.rng.borrow_mut();
        for logits in &batch_logits {
            let u: f32 = rand::Rng::random(&mut *rng);
            let action = categorical_sample(logits, u);
            let lp = categorical_log_prob(logits, action);
            actions.push(action as f32);
            log_probs.push(lp);
        }

        Ok(ActionOutput {
            actions: TensorData::new(actions, vec![batch_size]),
            log_probs: TensorData::new(log_probs, vec![batch_size]),
        })
    }

    fn value(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        if obs.shape.len() != 2 {
            return Err(NNError::ShapeMismatch {
                expected: "2D [batch, obs_dim]".into(),
                got: format!("{:?}", obs.shape),
            });
        }

        let obs_tensor = to_tensor_2d::<B::InnerBackend>(obs, &self.device.clone().into());
        let values = self.model.valid().critic_forward(obs_tensor);
        let values = values.squeeze::<1>(1);
        Ok(from_tensor_1d(values))
    }

    fn evaluate(&self, obs: &TensorData, actions: &TensorData) -> Result<EvalOutput, NNError> {
        let batch_size = obs.shape[0];
        let batch_logits = self.compute_logits_np(obs);

        let mut log_probs = Vec::with_capacity(batch_size);
        let mut entropies = Vec::with_capacity(batch_size);

        for (i, logits) in batch_logits.iter().enumerate() {
            let action = actions.data[i] as usize;
            log_probs.push(categorical_log_prob(logits, action));
            entropies.push(categorical_entropy(logits));
        }

        let values = self.value(obs)?;

        Ok(EvalOutput {
            log_probs: TensorData::new(log_probs, vec![batch_size]),
            entropy: TensorData::new(entropies, vec![batch_size]),
            values,
        })
    }

    fn ppo_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        old_log_probs: &TensorData,
        advantages: &TensorData,
        returns: &TensorData,
        old_values: &TensorData,
        config: &PPOStepConfig,
    ) -> Result<TrainMetrics, NNError> {
        let batch_size = obs.shape[0];

        // Forward pass with autograd — goes through self.model.params directly
        let obs_tensor = to_tensor_2d::<B>(obs, &self.device);

        // Actor forward
        let logits_tensor = self.model.actor_forward(obs_tensor.clone());

        // Compute new log-probs and entropy using softmax on the tensor level
        let log_probs_all = activation::log_softmax(logits_tensor.clone(), 1);

        // Gather log-probs at action indices
        let actions_int = to_int_tensor_1d::<B>(actions, &self.device);
        let actions_2d = actions_int.unsqueeze_dim(1);
        let new_log_probs = log_probs_all.clone().gather(1, actions_2d).squeeze::<1>(1);

        // Entropy: -sum(p * log_p, dim=-1)
        let probs = activation::softmax(logits_tensor, 1);
        let entropy = -(probs * log_probs_all).sum_dim(1).squeeze::<1>(1);

        // Critic forward
        let values_2d = self.model.critic_forward(obs_tensor);
        let new_values = values_2d.squeeze::<1>(1);

        // PPO loss computation
        let old_lp = to_tensor_1d::<B>(old_log_probs, &self.device);
        let adv = to_tensor_1d::<B>(advantages, &self.device);
        let ret = to_tensor_1d::<B>(returns, &self.device);

        let log_ratio = new_log_probs.clone() - old_lp;
        let ratio = log_ratio.clone().exp();

        // Clipped surrogate
        let pg_loss1 = -adv.clone() * ratio.clone();
        let clamped = ratio
            .clone()
            .clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps);
        let pg_loss2 = -adv * clamped;
        let policy_loss = Tensor::max_pair(pg_loss1, pg_loss2).mean();

        // Value loss
        let value_loss = if config.clip_vloss {
            let old_v = to_tensor_1d::<B>(old_values, &self.device);
            let v_clipped = old_v.clone()
                + (new_values.clone() - old_v).clamp(-config.clip_eps, config.clip_eps);
            let vf1 = (new_values - ret.clone()).powf_scalar(2.0);
            let vf2 = (v_clipped - ret).powf_scalar(2.0);
            Tensor::max_pair(vf1, vf2).mean() * 0.5
        } else {
            (new_values - ret).powf_scalar(2.0).mean() * 0.5
        };

        let entropy_loss = entropy.mean();
        let total_loss = policy_loss.clone() + value_loss.clone() * config.vf_coef
            - entropy_loss.clone() * config.ent_coef;

        // Backward + step on the params Module
        let grads = total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model.params);
        self.model.params = self
            .optimizer
            .step(self.lr.into(), self.model.params.clone(), grads);

        // Extract metrics (no_grad)
        let policy_loss_val: f32 = policy_loss.inner().into_data().to_vec::<f32>().unwrap()[0];
        let value_loss_val: f32 = value_loss.inner().into_data().to_vec::<f32>().unwrap()[0];
        let entropy_val: f32 = entropy_loss.inner().into_data().to_vec::<f32>().unwrap()[0];

        // Approx KL and clip fraction from ratio (no grad)
        let ratio_data: Vec<f32> = ratio.inner().into_data().to_vec().unwrap();
        let log_ratio_data: Vec<f32> = log_ratio.inner().into_data().to_vec().unwrap();
        let approx_kl: f32 = ratio_data
            .iter()
            .zip(log_ratio_data.iter())
            .map(|(&r, &lr)| (r - 1.0) - lr)
            .sum::<f32>()
            / batch_size as f32;
        let clip_fraction: f32 = ratio_data
            .iter()
            .filter(|&&r| (r - 1.0).abs() > config.clip_eps)
            .count() as f32
            / batch_size as f32;

        let mut metrics = TrainMetrics::new();
        metrics.insert("policy_loss", policy_loss_val as f64);
        metrics.insert("value_loss", value_loss_val as f64);
        metrics.insert("entropy", entropy_val as f64);
        metrics.insert("approx_kl", approx_kl as f64);
        metrics.insert("clip_fraction", clip_fraction as f64);

        Ok(metrics)
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
    use rlox_nn::ActorCritic;

    type TestBackend = Autodiff<NdArray>;
    type TestDevice = <NdArray as Backend>::Device;

    fn device() -> TestDevice {
        Default::default()
    }

    #[test]
    fn test_model_forward_shapes() {
        let model = DiscreteActorCriticModel::<NdArray>::new(4, 2, 64, &device());
        let obs = Tensor::<NdArray, 2>::zeros([8, 4], &device());

        let logits = model.actor_forward(obs.clone());
        assert_eq!(logits.shape().dims, [8, 2]);

        let values = model.critic_forward(obs);
        assert_eq!(values.shape().dims, [8, 1]);
    }

    #[test]
    fn test_act_returns_valid_actions() {
        let ac = BurnActorCritic::<TestBackend>::new(4, 2, 64, 2.5e-4, device().into(), 42);
        let obs = TensorData::zeros(vec![8, 4]);
        let result = ac.act(&obs).unwrap();

        assert_eq!(result.actions.shape, vec![8]);
        assert_eq!(result.log_probs.shape, vec![8]);

        for &a in &result.actions.data {
            assert!((0.0..2.0).contains(&a), "action out of range: {a}");
        }
        for &lp in &result.log_probs.data {
            assert!(lp <= 0.0, "log_prob should be <= 0: {lp}");
        }
    }

    #[test]
    fn test_value_shape() {
        let ac = BurnActorCritic::<TestBackend>::new(4, 2, 64, 2.5e-4, device().into(), 42);
        let obs = TensorData::zeros(vec![8, 4]);
        let values = ac.value(&obs).unwrap();
        assert_eq!(values.shape, vec![8]);
    }

    #[test]
    fn test_evaluate_shapes() {
        let ac = BurnActorCritic::<TestBackend>::new(4, 2, 64, 2.5e-4, device().into(), 42);
        let obs = TensorData::zeros(vec![4, 4]);
        let actions = TensorData::new(vec![0.0, 1.0, 0.0, 1.0], vec![4]);
        let eval = ac.evaluate(&obs, &actions).unwrap();

        assert_eq!(eval.log_probs.shape, vec![4]);
        assert_eq!(eval.entropy.shape, vec![4]);
        assert_eq!(eval.values.shape, vec![4]);
    }

    #[test]
    fn test_ppo_step_runs() {
        let mut ac = BurnActorCritic::<TestBackend>::new(4, 2, 64, 2.5e-4, device().into(), 42);

        let batch_size = 32;
        let obs = TensorData::zeros(vec![batch_size, 4]);
        let actions = TensorData::new(vec![0.0; batch_size], vec![batch_size]);
        let old_log_probs = TensorData::new(vec![-0.7; batch_size], vec![batch_size]);
        let advantages = TensorData::new(vec![1.0; batch_size], vec![batch_size]);
        let returns = TensorData::new(vec![1.0; batch_size], vec![batch_size]);
        let old_values = TensorData::zeros(vec![batch_size]);

        let config = PPOStepConfig::default();
        let metrics = ac
            .ppo_step(
                &obs,
                &actions,
                &old_log_probs,
                &advantages,
                &returns,
                &old_values,
                &config,
            )
            .unwrap();

        assert!(metrics.get("policy_loss").is_some());
        assert!(metrics.get("value_loss").is_some());
        assert!(metrics.get("entropy").is_some());
        assert!(metrics.get("approx_kl").is_some());
        assert!(metrics.get("clip_fraction").is_some());
    }

    #[test]
    fn test_ppo_step_reduces_loss() {
        let mut ac = BurnActorCritic::<TestBackend>::new(4, 2, 64, 1e-3, device().into(), 42);

        let batch_size = 64;
        let obs = TensorData::new(
            (0..batch_size * 4).map(|i| (i as f32) * 0.01).collect(),
            vec![batch_size, 4],
        );
        let actions = TensorData::new(vec![0.0; batch_size], vec![batch_size]);
        let old_log_probs = TensorData::new(vec![-0.7; batch_size], vec![batch_size]);
        let advantages = TensorData::new(vec![1.0; batch_size], vec![batch_size]);
        let returns = TensorData::new(vec![1.0; batch_size], vec![batch_size]);
        let old_values = TensorData::zeros(vec![batch_size]);

        let config = PPOStepConfig::default();

        let m1 = ac
            .ppo_step(
                &obs,
                &actions,
                &old_log_probs,
                &advantages,
                &returns,
                &old_values,
                &config,
            )
            .unwrap();
        let m2 = ac
            .ppo_step(
                &obs,
                &actions,
                &old_log_probs,
                &advantages,
                &returns,
                &old_values,
                &config,
            )
            .unwrap();

        let vl1 = m1.get("value_loss").unwrap();
        let vl2 = m2.get("value_loss").unwrap();
        assert!(vl2.is_finite(), "value loss should be finite");
        assert!(vl1.is_finite(), "value loss should be finite");
    }

    #[test]
    fn test_lr_get_set() {
        use rlox_nn::ActorCritic;
        let mut ac = BurnActorCritic::<TestBackend>::new(4, 2, 64, 2.5e-4, device().into(), 42);
        assert!((ac.learning_rate() - 2.5e-4).abs() < 1e-8);
        ac.set_learning_rate(1e-3);
        assert!((ac.learning_rate() - 1e-3).abs() < 1e-8);
    }

    #[test]
    fn test_act_invalid_shape() {
        let ac = BurnActorCritic::<TestBackend>::new(4, 2, 64, 2.5e-4, device().into(), 42);
        let obs = TensorData::zeros(vec![4]); // 1D, should fail
        assert!(ac.act(&obs).is_err());
    }

    #[test]
    fn test_act_rng_advances() {
        let ac = BurnActorCritic::<TestBackend>::new(4, 2, 64, 2.5e-4, device().into(), 42);
        let obs = TensorData::zeros(vec![1, 4]);
        let mut seen_different = false;
        let first = ac.act(&obs).unwrap().log_probs.data[0];
        for _ in 0..20 {
            let lp = ac.act(&obs).unwrap().log_probs.data[0];
            if (lp - first).abs() > 1e-6 {
                seen_different = true;
                break;
            }
        }
        assert!(seen_different, "RNG should advance between act() calls");
    }
}
