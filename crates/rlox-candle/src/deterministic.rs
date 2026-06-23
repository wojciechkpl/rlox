use std::path::Path;

use candle_core::Device;
use candle_nn::{Optimizer, VarBuilder, VarMap};

use rlox_nn::{
    Activation, DeterministicPolicy as DeterministicPolicyTrait, MLPConfig, NNError, TensorData,
    TrainMetrics,
};

use crate::convert::*;
use crate::mlp::MLP;

pub struct CandleDeterministicPolicy {
    net: MLP,
    target_net: MLP,
    varmap: VarMap,
    target_varmap: VarMap,
    optimizer: candle_nn::AdamW,
    device: Device,
    max_action: f32,
    lr: f64,
}

impl CandleDeterministicPolicy {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden: usize,
        max_action: f32,
        lr: f64,
        device: Device,
    ) -> Result<Self, NNError> {
        let config = MLPConfig::new(obs_dim, act_dim)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::ReLU)
            .with_output_activation(Activation::Tanh);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let net = MLP::new(&config, vb.pp("actor")).nn_err()?;

        let target_varmap = VarMap::new();
        let tvb = VarBuilder::from_varmap(&target_varmap, candle_core::DType::F32, &device);
        let target_net = MLP::new(&config, tvb.pp("actor")).nn_err()?;

        // Copy weights
        {
            let src = varmap.data().lock().unwrap();
            let tgt = target_varmap.data().lock().unwrap();
            for (name, var) in src.iter() {
                if let Some(tvar) = tgt.get(name) {
                    tvar.set(&var.as_tensor().clone()).unwrap();
                }
            }
        }

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
            net,
            target_net,
            varmap,
            target_varmap,
            optimizer,
            device,
            max_action,
            lr,
        })
    }
}

impl CandleDeterministicPolicy {
    /// TD3 actor gradient step with autograd flowing through the critic.
    ///
    /// Takes concrete `CandleTwinQ` to preserve gradient flow from Q1(s, a)
    /// back to actor parameters via the deterministic actions.
    pub fn td3_actor_step(
        &mut self,
        obs: &TensorData,
        critic: &crate::continuous_q::CandleTwinQ,
    ) -> Result<TrainMetrics, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let actions = self.net.forward(&obs_t).nn_err()?;
        let scaled = (&actions * self.max_action as f64).nn_err()?;

        // Q1 with full autograd — gradient flows through critic back to actor
        let q1 = critic
            .q1_forward(&obs_t, &scaled)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;

        let actor_loss = q1.neg().nn_err()?.mean_all().nn_err()?;

        self.optimizer.backward_step(&actor_loss).nn_err()?;

        let loss_val: f32 = actor_loss.to_scalar().nn_err()?;
        let mut metrics = TrainMetrics::new();
        metrics.insert("actor_loss", loss_val as f64);
        Ok(metrics)
    }
}

impl DeterministicPolicyTrait for CandleDeterministicPolicy {
    fn act(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let actions = self.net.forward(&obs_t).nn_err()?;
        let scaled = (&actions * self.max_action as f64).nn_err()?;
        from_tensor_2d(&scaled).nn_err()
    }

    fn target_act(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let actions = self.target_net.forward(&obs_t).nn_err()?;
        let scaled = (&actions * self.max_action as f64).nn_err()?;
        from_tensor_2d(&scaled).nn_err()
    }

    fn soft_update_target(&mut self, tau: f32) {
        let src = self.varmap.data().lock().unwrap();
        let tgt = self.target_varmap.data().lock().unwrap();
        for (name, var) in src.iter() {
            if let Some(tvar) = tgt.get(name) {
                let src_t = var.as_tensor();
                let tgt_t = tvar.as_tensor();
                let new_val = ((src_t * tau as f64).unwrap()
                    + (tgt_t * (1.0 - tau) as f64).unwrap())
                .unwrap();
                tvar.set(&new_val).unwrap();
            }
        }
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

unsafe impl Send for CandleDeterministicPolicy {}
unsafe impl Sync for CandleDeterministicPolicy {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_act_shape() {
        let policy = CandleDeterministicPolicy::new(3, 1, 64, 1.0, 3e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![8, 3]);
        let actions = policy.act(&obs).unwrap();
        assert_eq!(actions.shape, vec![8, 1]);
    }

    #[test]
    fn test_target_matches_initially() {
        let policy = CandleDeterministicPolicy::new(3, 1, 64, 1.0, 3e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![4, 3]);
        let act = policy.act(&obs).unwrap();
        let tgt = policy.target_act(&obs).unwrap();
        for (a, b) in act.data.iter().zip(tgt.data.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn test_action_range() {
        let policy = CandleDeterministicPolicy::new(3, 1, 64, 2.0, 3e-4, Device::Cpu).unwrap();
        let obs = TensorData::new(
            (0..300).map(|i| (i as f32) * 0.1 - 15.0).collect(),
            vec![100, 3],
        );
        let actions = policy.act(&obs).unwrap();
        for &a in &actions.data {
            assert!(
                (-2.0 - 1e-4..=2.0 + 1e-4).contains(&a),
                "action out of range: {a}"
            );
        }
    }
}
