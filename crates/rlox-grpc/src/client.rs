use rlox_core::env::parallel::BatchTransition;
use rlox_core::env::spaces::{Action, Observation};

use crate::error::GrpcError;
use crate::proto::env_service_client::EnvServiceClient;
use crate::proto::{Empty, ResetRequest, StepRequest};

/// Client for connecting to a remote `EnvWorker` over gRPC.
pub struct RemoteEnvClient {
    client: EnvServiceClient<tonic::transport::Channel>,
}

impl RemoteEnvClient {
    /// Connect to a remote environment worker at the given address.
    ///
    /// The address should include the scheme, e.g. `"http://[::1]:50051"`.
    pub async fn connect(addr: &str) -> Result<Self, GrpcError> {
        let client = EnvServiceClient::connect(addr.to_owned()).await?;
        Ok(Self { client })
    }

    /// Step the remote batch of environments with the given actions.
    ///
    /// Returns a `BatchTransition` compatible with the local `VecEnv` interface.
    /// Terminal observations are transmitted over the wire via the response's
    /// `terminal_obs` (flat, zero-padded) + `has_terminal_obs` (mask) fields and
    /// reconstructed here as `Vec<Option<Vec<f32>>>`, so truncation bootstrap
    /// values are preserved. Falls back to `None` per env against older servers
    /// that do not populate those fields.
    pub async fn step_batch(&mut self, actions: &[Action]) -> Result<BatchTransition, GrpcError> {
        let (flat_actions, discrete, act_dim) = flatten_actions(actions)?;

        let request = StepRequest {
            actions: flat_actions,
            discrete,
            act_dim: act_dim as u32,
        };

        let response = self.client.step_batch(request).await?.into_inner();

        let num_envs = response.num_envs as usize;
        let obs_dim = response.obs_dim as usize;

        let obs: Vec<Vec<f32>> = if obs_dim > 0 {
            response
                .obs
                .chunks_exact(obs_dim)
                .map(|c| c.to_vec())
                .collect()
        } else {
            vec![vec![]; num_envs]
        };

        // Reconstruct terminal observations from the flat buffer + mask.
        // The server zero-pads each slot to obs_dim, so we chunk by obs_dim
        // and yield Some(chunk) only where has_terminal_obs[i] is true.
        let terminal_obs: Vec<Option<Vec<f32>>> = if obs_dim > 0
            && !response.terminal_obs.is_empty()
            && !response.has_terminal_obs.is_empty()
        {
            response
                .terminal_obs
                .chunks_exact(obs_dim)
                .zip(response.has_terminal_obs.iter())
                .map(|(chunk, &has)| if has { Some(chunk.to_vec()) } else { None })
                .collect()
        } else {
            vec![None; num_envs]
        };

        Ok(BatchTransition {
            obs,
            obs_flat: Vec::new(),
            obs_dim: 0,
            rewards: response.rewards,
            terminated: response.terminated,
            truncated: response.truncated,
            terminal_obs,
        })
    }

    /// Reset the remote batch of environments.
    pub async fn reset_batch(&mut self, seed: Option<u64>) -> Result<Vec<Observation>, GrpcError> {
        let request = ResetRequest {
            seed: seed.unwrap_or(0),
            has_seed: seed.is_some(),
        };

        let response = self.client.reset_batch(request).await?.into_inner();

        let obs_dim = response.obs_dim as usize;
        let observations: Vec<Observation> = if obs_dim > 0 {
            response
                .obs
                .chunks_exact(obs_dim)
                .map(|c| Observation::Flat(c.to_vec()))
                .collect()
        } else {
            let num_envs = response.num_envs as usize;
            vec![Observation::Flat(vec![]); num_envs]
        };

        Ok(observations)
    }

    /// Query the remote worker for its action/observation spaces and env count.
    pub async fn get_spaces(&mut self) -> Result<SpacesInfo, GrpcError> {
        let response = self.client.get_spaces(Empty {}).await?.into_inner();
        Ok(SpacesInfo {
            action_space_json: response.action_space_json,
            obs_space_json: response.obs_space_json,
            num_envs: response.num_envs as usize,
        })
    }
}

/// Information about a remote worker's environment configuration.
#[derive(Debug, Clone)]
pub struct SpacesInfo {
    pub action_space_json: String,
    pub obs_space_json: String,
    pub num_envs: usize,
}

/// Convert a slice of `Action` values into the flat representation used by the
/// proto `StepRequest`.
fn flatten_actions(actions: &[Action]) -> Result<(Vec<f32>, bool, usize), GrpcError> {
    if actions.is_empty() {
        return Ok((vec![], true, 1));
    }

    match &actions[0] {
        Action::Discrete(_) => {
            let flat: Vec<f32> = actions
                .iter()
                .map(|a| match a {
                    Action::Discrete(v) => Ok(*v as f32),
                    Action::Continuous(_) => Err(GrpcError::MixedActionTypes),
                })
                .collect::<Result<_, _>>()?;
            Ok((flat, true, 1))
        }
        Action::Continuous(v) => {
            let act_dim = v.len();
            let flat: Vec<f32> = actions
                .iter()
                .map(|a| match a {
                    Action::Continuous(v) => Ok(v.clone()),
                    Action::Discrete(_) => Err(GrpcError::MixedActionTypes),
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            Ok((flat, false, act_dim))
        }
    }
}
