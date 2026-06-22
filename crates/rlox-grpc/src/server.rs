use std::sync::Mutex;

use tonic::{Request, Response, Status};

use rlox_core::env::batch::BatchSteppable;
use rlox_core::env::parallel::VecEnv;
use rlox_core::env::spaces::Action;

use crate::proto::env_service_server::EnvService;
use crate::proto::{Empty, ResetRequest, ResetResponse, SpacesResponse, StepRequest, StepResponse};

/// A gRPC service that wraps a `VecEnv` for remote environment stepping.
///
/// The `Mutex` is needed because `BatchSteppable::step_batch` takes `&mut self`,
/// and tonic handlers receive `&self`. Contention is expected to be low since
/// each worker processes one request at a time in the RL loop.
pub struct EnvWorker {
    envs: Mutex<VecEnv>,
}

impl EnvWorker {
    pub fn new(envs: VecEnv) -> Self {
        Self {
            envs: Mutex::new(envs),
        }
    }
}

#[tonic::async_trait]
impl EnvService for EnvWorker {
    async fn step_batch(
        &self,
        request: Request<StepRequest>,
    ) -> Result<Response<StepResponse>, Status> {
        let req = request.into_inner();

        let mut envs = self
            .envs
            .lock()
            .map_err(|e| Status::internal(format!("lock poisoned: {e}")))?;

        let num_envs = envs.num_envs();
        let act_dim = req.act_dim as usize;
        let discrete = req.discrete;

        // Reconstruct actions from the flat f32 array.
        let actions: Vec<Action> = if discrete {
            if req.actions.len() != num_envs {
                return Err(Status::invalid_argument(format!(
                    "expected {} discrete actions, got {}",
                    num_envs,
                    req.actions.len()
                )));
            }
            req.actions
                .iter()
                .map(|&a| Action::Discrete(a as u32))
                .collect()
        } else {
            if act_dim == 0 {
                return Err(Status::invalid_argument(
                    "act_dim must be > 0 for continuous actions",
                ));
            }
            if req.actions.len() != num_envs * act_dim {
                return Err(Status::invalid_argument(format!(
                    "expected {} continuous action floats ({}*{}), got {}",
                    num_envs * act_dim,
                    num_envs,
                    act_dim,
                    req.actions.len()
                )));
            }
            req.actions
                .chunks_exact(act_dim)
                .map(|chunk| Action::Continuous(chunk.to_vec()))
                .collect()
        };

        let batch = envs
            .step_batch(&actions)
            .map_err(|e| Status::internal(format!("step error: {e}")))?;

        let obs_dim = if batch.obs.is_empty() {
            0
        } else {
            batch.obs[0].len()
        };

        let flat_obs: Vec<f32> = batch.obs.into_iter().flatten().collect();

        // Encode terminal observations: flatten into a contiguous buffer
        // (zero-padded for envs without a terminal obs) plus a boolean mask.
        let has_terminal_obs: Vec<bool> = batch
            .terminal_obs
            .iter()
            .map(|t| t.is_some())
            .collect();
        let mut flat_terminal_obs: Vec<f32> = Vec::with_capacity(num_envs * obs_dim);
        for slot in batch.terminal_obs {
            match slot {
                // A present terminal obs must match obs_dim exactly; treat a
                // mismatch as an invariant violation rather than silently
                // truncating or zero-padding real observation data.
                Some(obs) => {
                    if obs.len() != obs_dim {
                        return Err(Status::internal(format!(
                            "terminal_obs has len {} but expected obs_dim {obs_dim}",
                            obs.len()
                        )));
                    }
                    flat_terminal_obs.extend_from_slice(&obs);
                }
                None => flat_terminal_obs.extend(std::iter::repeat(0.0f32).take(obs_dim)),
            }
        }

        Ok(Response::new(StepResponse {
            obs: flat_obs,
            rewards: batch.rewards,
            terminated: batch.terminated,
            truncated: batch.truncated,
            obs_dim: obs_dim as u32,
            num_envs: num_envs as u32,
            terminal_obs: flat_terminal_obs,
            has_terminal_obs,
        }))
    }

    async fn reset_batch(
        &self,
        request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        let req = request.into_inner();
        let seed = if req.has_seed { Some(req.seed) } else { None };

        let mut envs = self
            .envs
            .lock()
            .map_err(|e| Status::internal(format!("lock poisoned: {e}")))?;

        let num_envs = envs.num_envs();

        let observations = envs
            .reset_batch(seed)
            .map_err(|e| Status::internal(format!("reset error: {e}")))?;

        let obs_dim = if observations.is_empty() {
            0
        } else {
            observations[0].as_slice().len()
        };

        let flat_obs: Vec<f32> = observations
            .into_iter()
            .flat_map(|o| o.into_inner())
            .collect();

        Ok(Response::new(ResetResponse {
            obs: flat_obs,
            obs_dim: obs_dim as u32,
            num_envs: num_envs as u32,
        }))
    }

    async fn get_spaces(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<SpacesResponse>, Status> {
        let envs = self
            .envs
            .lock()
            .map_err(|e| Status::internal(format!("lock poisoned: {e}")))?;

        let action_space_json = serde_json::to_string(envs.action_space())
            .map_err(|e| Status::internal(format!("serialize action_space: {e}")))?;
        let obs_space_json = serde_json::to_string(envs.obs_space())
            .map_err(|e| Status::internal(format!("serialize obs_space: {e}")))?;

        Ok(Response::new(SpacesResponse {
            action_space_json,
            obs_space_json,
            num_envs: envs.num_envs() as u32,
        }))
    }
}
