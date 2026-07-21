use std::path::Path;

use burn::module::AutodiffModule;
use burn::module::Param;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use rlox_nn::{
    Activation, DeterministicPolicy as DeterministicPolicyTrait, MLPConfig, NNError, TensorData,
    TrainMetrics,
};

use crate::convert::*;
use crate::mlp::{apply_activation, ActivationKind, MLPParams, MLP};

/// Learnable parameters for deterministic policy — Module-derivable.
#[derive(Module, Debug)]
pub struct DeterministicParams<B: Backend> {
    net: MLPParams<B>,
}

/// Deterministic policy model for TD3.
#[derive(Debug)]
pub struct DeterministicModel<B: Backend> {
    pub params: DeterministicParams<B>,
    activation: ActivationKind,
    output_activation: Option<ActivationKind>,
    max_action: f32,
}

impl<B: Backend> DeterministicModel<B> {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden: usize,
        max_action: f32,
        device: &B::Device,
    ) -> Self {
        let config = MLPConfig::new(obs_dim, act_dim)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::ReLU)
            .with_output_activation(Activation::Tanh);

        let mlp = MLP::new(&config, device);

        Self {
            params: DeterministicParams { net: mlp.params },
            activation: config.activation.into(),
            output_activation: config.output_activation.map(Into::into),
            max_action,
        }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let layers = &self.params.net.layers;
        let n = layers.len();
        let mut x = obs;
        for (i, layer) in layers.iter().enumerate() {
            x = layer.forward(x);
            if i < n - 1 {
                x = apply_activation(x, self.activation);
            } else if let Some(out_act) = self.output_activation {
                x = apply_activation(x, out_act);
            }
        }
        x * self.max_action
    }

    pub fn valid(&self) -> DeterministicModel<B::InnerBackend>
    where
        B: AutodiffBackend,
    {
        DeterministicModel {
            params: self.params.valid(),
            activation: self.activation,
            output_activation: self.output_activation,
            max_action: self.max_action,
        }
    }
}

impl<B: Backend> Clone for DeterministicModel<B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            activation: self.activation,
            output_activation: self.output_activation,
            max_action: self.max_action,
        }
    }
}

pub struct BurnDeterministicPolicy<B: AutodiffBackend> {
    model: DeterministicModel<B>,
    target: DeterministicModel<B::InnerBackend>,
    optimizer: OptimizerAdaptor<Adam, DeterministicParams<B>, B>,
    device: B::Device,
    lr: f32,
}

impl<B: AutodiffBackend> BurnDeterministicPolicy<B> {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden: usize,
        max_action: f32,
        lr: f32,
        device: B::Device,
    ) -> Self {
        let model = DeterministicModel::new(obs_dim, act_dim, hidden, max_action, &device);
        let target = model.valid();
        let optimizer = AdamConfig::new().init();

        Self {
            model,
            target,
            optimizer,
            device,
            lr,
        }
    }
}

impl<B: AutodiffBackend> BurnDeterministicPolicy<B>
where
    B::Device: Clone,
{
    /// TD3 actor gradient step with autograd flowing through the critic.
    ///
    /// This is an inherent method (not on the `DeterministicPolicy` trait) because
    /// the gradient from Q1(s, a) must flow back through the critic's Q-network
    /// to the actor's parameters via the deterministic actions. The trait boundary
    /// would sever this autograd chain by converting tensors to `TensorData`.
    pub fn td3_actor_step(
        &mut self,
        obs: &TensorData,
        critic: &crate::continuous_q::BurnTwinQ<B>,
    ) -> Result<TrainMetrics, NNError> {
        let obs_t = to_tensor_2d::<B>(obs, &self.device);
        let actions = self.model.forward(obs_t.clone());

        // Q1 with full autograd — gradient flows through critic back to actor
        let q1 = critic.q1_forward_ad(obs_t, actions).squeeze::<1>(1);

        // Actor loss = -Q1.mean()
        let actor_loss = -q1.mean();

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

impl<B: AutodiffBackend> DeterministicPolicyTrait for BurnDeterministicPolicy<B>
where
    B::Device: Clone,
{
    fn act(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let dev: <B::InnerBackend as Backend>::Device = self.device.clone().into();
        let obs_t = to_tensor_2d::<B::InnerBackend>(obs, &dev);
        let actions = self.model.valid().forward(obs_t);
        Ok(from_tensor_2d(actions))
    }

    fn target_act(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let dev: <B::InnerBackend as Backend>::Device = self.device.clone().into();
        let obs_t = to_tensor_2d::<B::InnerBackend>(obs, &dev);
        let actions = self.target.forward(obs_t);
        Ok(from_tensor_2d(actions))
    }

    fn soft_update_target(&mut self, tau: f32) {
        if tau >= 1.0 - 1e-6 {
            self.target = self.model.valid();
        } else {
            let source = self.model.valid();
            let src_record = source.params.net.clone().into_record();
            let mut tgt_record = self.target.params.net.clone().into_record();
            let dev: <B::InnerBackend as Backend>::Device = self.device.clone().into();

            for (src_layer, tgt_layer) in src_record.layers.iter().zip(tgt_record.layers.iter_mut())
            {
                let src_w: Tensor<B::InnerBackend, 2> =
                    Tensor::from_data(src_layer.weight.val().into_data(), &dev);
                let tgt_w: Tensor<B::InnerBackend, 2> =
                    Tensor::from_data(tgt_layer.weight.val().into_data(), &dev);
                let new_w = src_w * tau + tgt_w * (1.0 - tau);
                tgt_layer.weight = Param::from_tensor(new_w);

                if let (Some(src_b), Some(tgt_b)) = (&src_layer.bias, &mut tgt_layer.bias) {
                    let src_bt: Tensor<B::InnerBackend, 1> =
                        Tensor::from_data(src_b.val().into_data(), &dev);
                    let tgt_bt: Tensor<B::InnerBackend, 1> =
                        Tensor::from_data(tgt_b.val().into_data(), &dev);
                    let new_b = src_bt * tau + tgt_bt * (1.0 - tau);
                    *tgt_b = Param::from_tensor(new_b);
                }
            }

            let new_net = self.target.params.net.clone().load_record(tgt_record);
            self.target.params = DeterministicParams { net: new_net };
        }
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
    fn test_deterministic_model_shape() {
        let model = DeterministicModel::<NdArray>::new(3, 1, 64, 1.0, &device());
        let obs = Tensor::<NdArray, 2>::zeros([4, 3], &device());
        let actions = model.forward(obs);
        assert_eq!(actions.shape().dims, [4, 1]);
    }

    #[test]
    fn test_deterministic_action_range() {
        let model = DeterministicModel::<NdArray>::new(3, 1, 64, 2.0, &device());
        let obs = Tensor::<NdArray, 2>::ones([100, 3], &device()) * 10.0;
        let actions = model.forward(obs);
        let data: Vec<f32> = actions.into_data().to_vec().unwrap();
        for &a in &data {
            assert!(
                (-2.0..=2.0).contains(&a),
                "action should be in [-max_action, max_action]: {a}"
            );
        }
    }

    #[test]
    fn test_act_shape() {
        let policy =
            BurnDeterministicPolicy::<TestBackend>::new(3, 1, 64, 1.0, 3e-4, device().into());
        let obs = TensorData::zeros(vec![8, 3]);
        let actions = policy.act(&obs).unwrap();
        assert_eq!(actions.shape, vec![8, 1]);
    }

    #[test]
    fn test_target_act_matches_initially() {
        let policy =
            BurnDeterministicPolicy::<TestBackend>::new(3, 1, 64, 1.0, 3e-4, device().into());
        let obs = TensorData::zeros(vec![4, 3]);
        let act = policy.act(&obs).unwrap();
        let tgt_act = policy.target_act(&obs).unwrap();

        for (a, b) in act.data.iter().zip(tgt_act.data.iter()) {
            assert!((a - b).abs() < 1e-5, "should match initially: {a} vs {b}");
        }
    }

    #[test]
    fn test_soft_update_preserves_equal_weights() {
        let mut policy =
            BurnDeterministicPolicy::<TestBackend>::new(3, 1, 64, 1.0, 3e-4, device().into());
        let obs = TensorData::zeros(vec![4, 3]);

        // Online and target start equal
        let act_before = policy.act(&obs).unwrap();
        let tgt_before = policy.target_act(&obs).unwrap();
        for (a, b) in act_before.data.iter().zip(tgt_before.data.iter()) {
            assert!((a - b).abs() < 1e-5, "should match initially: {a} vs {b}");
        }

        // soft_update with any tau on equal weights should keep them equal
        policy.soft_update_target(0.005);
        let act_after = policy.act(&obs).unwrap();
        let tgt_after = policy.target_act(&obs).unwrap();
        for (a, b) in act_after.data.iter().zip(tgt_after.data.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "soft update on equal weights should preserve equality: {a} vs {b}"
            );
        }

        // Hard update should also preserve equality
        policy.soft_update_target(1.0);
        let act_hard = policy.act(&obs).unwrap();
        let tgt_hard = policy.target_act(&obs).unwrap();
        for (a, b) in act_hard.data.iter().zip(tgt_hard.data.iter()) {
            assert!((a - b).abs() < 1e-5, "hard update should sync: {a} vs {b}");
        }
    }
}
