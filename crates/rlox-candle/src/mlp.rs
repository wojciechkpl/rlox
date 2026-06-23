use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use rlox_nn::{Activation, MLPConfig};

pub struct MLP {
    layers: Vec<Linear>,
    activation: Activation,
    output_activation: Option<Activation>,
}

fn apply_activation(x: &Tensor, act: Activation) -> Result<Tensor> {
    match act {
        Activation::ReLU => x.relu(),
        Activation::Tanh => x.tanh(),
    }
}

impl MLP {
    pub fn new(config: &MLPConfig, vb: VarBuilder) -> Result<Self> {
        let mut dims = Vec::new();
        dims.push(config.input_dim);
        dims.extend_from_slice(&config.hidden_dims);
        dims.push(config.output_dim);

        let layers: Vec<Linear> = dims
            .windows(2)
            .enumerate()
            .map(|(i, w)| linear(w[0], w[1], vb.pp(format!("layer_{i}"))))
            .collect::<Result<_>>()?;

        Ok(Self {
            layers,
            activation: config.activation,
            output_activation: config.output_activation,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let n = self.layers.len();
        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < n - 1 {
                x = apply_activation(&x, self.activation)?;
            } else if let Some(out_act) = self.output_activation {
                x = apply_activation(&x, out_act)?;
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_mlp_forward_shape() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
        let config = MLPConfig::new(4, 2).with_hidden(vec![64, 64]);
        let mlp = MLP::new(&config, vb).unwrap();

        let input = Tensor::zeros((8, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.dims(), &[8, 2]);
    }

    #[test]
    fn test_mlp_single_hidden() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
        let config = MLPConfig::new(3, 1).with_hidden(vec![16]);
        let mlp = MLP::new(&config, vb).unwrap();

        let input = Tensor::zeros((4, 3), candle_core::DType::F32, &Device::Cpu).unwrap();
        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, 1]);
    }

    #[test]
    fn test_mlp_tanh_output() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
        let config = MLPConfig::new(4, 2)
            .with_hidden(vec![32])
            .with_output_activation(Activation::Tanh);
        let mlp = MLP::new(&config, vb).unwrap();

        let input = Tensor::ones((4, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let input = (&input * 100.0).unwrap();
        let output = mlp.forward(&input).unwrap();
        let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for &v in &data {
            assert!((-1.0..=1.0).contains(&v), "tanh output out of range: {v}");
        }
    }
}
