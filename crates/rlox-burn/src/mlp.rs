use burn::module::AutodiffModule;
use burn::nn;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rlox_nn::{Activation, MLPConfig};

#[derive(Debug, Clone, Copy)]
pub enum ActivationKind {
    ReLU,
    Tanh,
}

impl From<Activation> for ActivationKind {
    fn from(a: Activation) -> Self {
        match a {
            Activation::ReLU => ActivationKind::ReLU,
            Activation::Tanh => ActivationKind::Tanh,
        }
    }
}

pub fn apply_activation<B: Backend>(x: Tensor<B, 2>, act: ActivationKind) -> Tensor<B, 2> {
    match act {
        ActivationKind::ReLU => burn::tensor::activation::relu(x),
        ActivationKind::Tanh => burn::tensor::activation::tanh(x),
    }
}

/// Learnable parameters only — Module-derivable.
#[derive(Module, Debug)]
pub struct MLPParams<B: Backend> {
    pub layers: Vec<nn::Linear<B>>,
}

/// Full MLP with activation config stored outside Module.
#[derive(Debug)]
pub struct MLP<B: Backend> {
    pub params: MLPParams<B>,
    activation: ActivationKind,
    output_activation: Option<ActivationKind>,
}

impl<B: Backend> MLP<B> {
    pub fn new(config: &MLPConfig, device: &B::Device) -> Self {
        let mut dims = Vec::new();
        dims.push(config.input_dim);
        dims.extend_from_slice(&config.hidden_dims);
        dims.push(config.output_dim);

        let layers: Vec<nn::Linear<B>> = dims
            .windows(2)
            .map(|w| nn::LinearConfig::new(w[0], w[1]).init(device))
            .collect();

        Self {
            params: MLPParams { layers },
            activation: config.activation.into(),
            output_activation: config.output_activation.map(Into::into),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = self.params.layers.len();
        let mut x = input;
        for (i, layer) in self.params.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < n - 1 {
                x = apply_activation(x, self.activation);
            } else if let Some(out_act) = self.output_activation {
                x = apply_activation(x, out_act);
            }
        }
        x
    }

    /// Get a non-autodiff version for inference.
    pub fn valid(&self) -> MLP<B::InnerBackend>
    where
        B: AutodiffBackend,
    {
        MLP {
            params: self.params.valid(),
            activation: self.activation,
            output_activation: self.output_activation,
        }
    }
}

impl<B: Backend> Clone for MLP<B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            activation: self.activation,
            output_activation: self.output_activation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    #[test]
    fn test_mlp_forward_shape() {
        let config = MLPConfig::new(4, 2).with_hidden(vec![64, 64]);
        let mlp = MLP::<B>::new(&config, &device());
        let input = Tensor::<B, 2>::zeros([8, 4], &device());
        let output = mlp.forward(input);
        assert_eq!(output.shape().dims, [8, 2]);
    }

    #[test]
    fn test_mlp_single_hidden() {
        let config = MLPConfig::new(3, 1).with_hidden(vec![16]);
        let mlp = MLP::<B>::new(&config, &device());
        let input = Tensor::<B, 2>::zeros([4, 3], &device());
        let output = mlp.forward(input);
        assert_eq!(output.shape().dims, [4, 1]);
    }

    #[test]
    fn test_mlp_relu_activation() {
        let config = MLPConfig::new(2, 2)
            .with_hidden(vec![8])
            .with_activation(Activation::ReLU);
        let mlp = MLP::<B>::new(&config, &device());
        let input = Tensor::<B, 2>::zeros([1, 2], &device());
        let output = mlp.forward(input);
        assert_eq!(output.shape().dims, [1, 2]);
    }

    #[test]
    fn test_mlp_with_output_activation() {
        let config = MLPConfig::new(4, 2)
            .with_hidden(vec![32])
            .with_output_activation(Activation::Tanh);
        let mlp = MLP::<B>::new(&config, &device());
        let input = Tensor::<B, 2>::ones([4, 4], &device()) * 100.0;
        let output = mlp.forward(input);
        let data: Vec<f32> = output.into_data().to_vec().unwrap();
        for &v in &data {
            assert!((-1.0..=1.0).contains(&v), "tanh output out of range: {v}");
        }
    }

    #[test]
    fn test_mlp_no_hidden() {
        let config = MLPConfig::new(4, 2).with_hidden(vec![]);
        let mlp = MLP::<B>::new(&config, &device());
        let input = Tensor::<B, 2>::zeros([1, 4], &device());
        let output = mlp.forward(input);
        assert_eq!(output.shape().dims, [1, 2]);
    }
}
