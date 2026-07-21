use rayon::prelude::*;

use crate::env::batch::BatchSteppable;
use crate::env::spaces::{Action, ActionSpace, ObsSpace, Observation};
use crate::env::RLEnv;
use crate::error::RloxError;
use crate::seed::derive_seed;

/// Per-environment raw step result: `(obs_data, reward, terminated, truncated, terminal_obs)`.
type RawStepResult = (Vec<f32>, f64, bool, bool, Option<Vec<f32>>);

/// Columnar batch of transitions from parallel stepping.
#[derive(Debug, Clone)]
pub struct BatchTransition {
    /// Observations: `[num_envs][obs_dim]` — post-reset obs when done
    pub obs: Vec<Vec<f32>>,
    /// Flat observations: `[num_envs * obs_dim]` contiguous layout.
    /// Populated by `step_all_flat`. Empty when using `step_all`.
    pub obs_flat: Vec<f32>,
    /// Observation dimensionality (set by step_all_flat).
    pub obs_dim: usize,
    /// Rewards: `[num_envs]`
    pub rewards: Vec<f64>,
    /// Terminated flags: `[num_envs]`
    pub terminated: Vec<bool>,
    /// Truncated flags: `[num_envs]`
    pub truncated: Vec<bool>,
    /// Terminal observations: `Some` when terminated or truncated, `None` otherwise.
    /// Contains the observation *before* auto-reset, needed for value bootstrapping.
    pub terminal_obs: Vec<Option<Vec<f32>>>,
}

/// A vectorized environment that steps multiple sub-environments in parallel.
pub struct VecEnv {
    envs: Vec<Box<dyn RLEnv>>,
    action_space: ActionSpace,
    obs_space: ObsSpace,
}

impl std::fmt::Debug for VecEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecEnv")
            .field("num_envs", &self.envs.len())
            .field("action_space", &self.action_space)
            .field("obs_space", &self.obs_space)
            .finish()
    }
}

impl VecEnv {
    /// Create a new vectorized environment from a list of sub-environments.
    ///
    /// # Errors
    ///
    /// Returns [`RloxError::EnvError`] if `envs` is empty.
    pub fn new(envs: Vec<Box<dyn RLEnv>>) -> Result<Self, RloxError> {
        if envs.is_empty() {
            return Err(RloxError::EnvError(
                "VecEnv requires at least one environment".into(),
            ));
        }
        let action_space = envs[0].action_space().clone();
        let obs_space = envs[0].obs_space().clone();
        Ok(VecEnv {
            envs,
            action_space,
            obs_space,
        })
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    pub fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    /// Step + auto-reset all environments in parallel. Returns the raw
    /// per-environment results: `(obs_data, reward, terminated, truncated, terminal_obs)`.
    fn step_raw(&mut self, actions: &[Action]) -> Result<Vec<RawStepResult>, RloxError> {
        if actions.len() != self.envs.len() {
            return Err(RloxError::ShapeMismatch {
                expected: format!("{}", self.envs.len()),
                got: format!("{}", actions.len()),
            });
        }

        let results: Vec<Result<RawStepResult, RloxError>> = self
            .envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, action)| {
                let mut transition = env.step(action)?;
                let mut term_obs = None;
                if transition.terminated || transition.truncated {
                    term_obs = Some(transition.obs.clone().into_inner());
                    let new_obs = env.reset(None)?;
                    transition.obs = new_obs;
                }
                let obs_data = transition.obs.into_inner();
                Ok((
                    obs_data,
                    transition.reward,
                    transition.terminated,
                    transition.truncated,
                    term_obs,
                ))
            })
            .collect();

        results.into_iter().collect()
    }

    /// Step all environments in parallel using Rayon.
    ///
    /// If an environment is done after stepping, it is automatically reset
    /// and the returned observation is from the fresh episode.
    pub fn step_all(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError> {
        let raw = self.step_raw(actions)?;
        let n = raw.len();
        let mut obs = Vec::with_capacity(n);
        let mut rewards = Vec::with_capacity(n);
        let mut terminated = Vec::with_capacity(n);
        let mut truncated = Vec::with_capacity(n);
        let mut terminal_obs = Vec::with_capacity(n);

        for (obs_data, reward, term, trunc, tobs) in raw {
            obs.push(obs_data);
            rewards.push(reward);
            terminated.push(term);
            truncated.push(trunc);
            terminal_obs.push(tobs);
        }

        Ok(BatchTransition {
            obs,
            obs_flat: Vec::new(),
            obs_dim: 0,
            rewards,
            terminated,
            truncated,
            terminal_obs,
        })
    }

    /// Step all environments in parallel, returning observations as a flat contiguous buffer.
    ///
    /// Unlike `step_all`, this avoids per-env Vec allocations by collecting
    /// observations directly into `obs_flat: Vec<f32>` of shape `[n_envs * obs_dim]`.
    /// The `obs` field is left empty.
    pub fn step_all_flat(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError> {
        let obs_dim = match &self.obs_space {
            ObsSpace::Discrete(_) => 1,
            ObsSpace::Box { shape, .. } => shape.iter().product(),
            ObsSpace::MultiDiscrete(v) => v.len(),
            ObsSpace::Dict(entries) => entries.iter().map(|(_, d)| d).sum(),
        };

        let raw = self.step_raw(actions)?;
        let n = raw.len();
        let mut obs_flat = vec![0.0f32; n * obs_dim];
        let mut rewards = Vec::with_capacity(n);
        let mut terminated = Vec::with_capacity(n);
        let mut truncated = Vec::with_capacity(n);
        let mut terminal_obs = Vec::with_capacity(n);

        for (i, (obs_data, reward, term, trunc, tobs)) in raw.into_iter().enumerate() {
            obs_flat[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&obs_data);
            rewards.push(reward);
            terminated.push(term);
            truncated.push(trunc);
            terminal_obs.push(tobs);
        }

        Ok(BatchTransition {
            obs: Vec::new(),
            obs_flat,
            obs_dim,
            rewards,
            terminated,
            truncated,
            terminal_obs,
        })
    }

    /// Reset all environments, optionally seeding them deterministically.
    ///
    /// When a master seed is provided, each env `i` gets `derive_seed(master, i)`.
    pub fn reset_all(&mut self, seed: Option<u64>) -> Result<Vec<Observation>, RloxError> {
        self.envs
            .iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let env_seed = seed.map(|s| derive_seed(s, i));
                env.reset(env_seed)
            })
            .collect()
    }
}

impl BatchSteppable for VecEnv {
    fn step_batch(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError> {
        self.step_all(actions)
    }

    fn reset_batch(&mut self, seed: Option<u64>) -> Result<Vec<Observation>, RloxError> {
        self.reset_all(seed)
    }

    fn num_envs(&self) -> usize {
        self.num_envs()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn obs_space(&self) -> &ObsSpace {
        &self.obs_space
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::builtins::CartPole;

    fn make_vec_env(n: usize, seed: u64) -> VecEnv {
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| {
                let s = derive_seed(seed, i);
                Box::new(CartPole::new(Some(s))) as Box<dyn RLEnv>
            })
            .collect();
        VecEnv::new(envs).unwrap()
    }

    #[test]
    fn vec_env_num_envs() {
        let venv = make_vec_env(4, 42);
        assert_eq!(venv.num_envs(), 4);
    }

    #[test]
    fn vec_env_step_all_returns_correct_shapes() {
        let mut venv = make_vec_env(4, 42);
        let actions: Vec<Action> = (0..4).map(|i| Action::Discrete((i % 2) as u32)).collect();
        let batch = venv.step_all(&actions).unwrap();
        assert_eq!(batch.obs.len(), 4);
        assert_eq!(batch.rewards.len(), 4);
        assert_eq!(batch.terminated.len(), 4);
        assert_eq!(batch.truncated.len(), 4);
        for obs in &batch.obs {
            assert_eq!(obs.len(), 4);
        }
    }

    #[test]
    fn vec_env_step_all_flat_returns_contiguous_obs() {
        let mut venv = make_vec_env(4, 42);
        let actions: Vec<Action> = (0..4).map(|i| Action::Discrete((i % 2) as u32)).collect();

        let batch_flat = venv.step_all_flat(&actions).unwrap();
        assert!(
            batch_flat.obs.is_empty(),
            "obs Vec should be empty in flat mode"
        );
        assert_eq!(batch_flat.obs_flat.len(), 4 * 4); // 4 envs * 4 obs_dim (CartPole)
        assert_eq!(batch_flat.obs_dim, 4);
        assert_eq!(batch_flat.rewards.len(), 4);
    }

    #[test]
    fn vec_env_step_all_flat_matches_step_all() {
        let mut venv1 = make_vec_env(4, 42);
        let mut venv2 = make_vec_env(4, 42);
        let actions: Vec<Action> = (0..4).map(|i| Action::Discrete((i % 2) as u32)).collect();

        let batch_vec = venv1.step_all(&actions).unwrap();
        let batch_flat = venv2.step_all_flat(&actions).unwrap();

        // Compare flat obs against per-env obs
        for (i, obs_vec) in batch_vec.obs.iter().enumerate() {
            let flat_slice = &batch_flat.obs_flat[i * 4..(i + 1) * 4];
            assert_eq!(obs_vec, flat_slice, "env {i} obs mismatch");
        }
        assert_eq!(batch_vec.rewards, batch_flat.rewards);
        assert_eq!(batch_vec.terminated, batch_flat.terminated);
    }

    #[test]
    fn vec_env_step_all_wrong_action_count() {
        let mut venv = make_vec_env(4, 42);
        let actions = vec![Action::Discrete(0); 3];
        let result = venv.step_all(&actions);
        assert!(result.is_err());
    }

    #[test]
    fn vec_env_reset_all_deterministic() {
        let mut venv1 = make_vec_env(4, 0);
        let mut venv2 = make_vec_env(4, 0);

        let obs1 = venv1.reset_all(Some(99)).unwrap();
        let obs2 = venv2.reset_all(Some(99)).unwrap();

        for (o1, o2) in obs1.iter().zip(obs2.iter()) {
            assert_eq!(o1.as_slice(), o2.as_slice());
        }
    }

    #[test]
    fn vec_env_large_parallel_stepping() {
        // Validate 256+ envs step correctly in parallel
        let mut venv = make_vec_env(256, 42);
        let actions: Vec<Action> = (0..256).map(|i| Action::Discrete((i % 2) as u32)).collect();
        let batch = venv.step_all(&actions).unwrap();
        assert_eq!(batch.obs.len(), 256);
        assert_eq!(batch.rewards.len(), 256);
        // All rewards should be 1.0 (first step)
        for &r in &batch.rewards {
            assert!((r - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn vec_env_1024_envs_no_panic() {
        // Ensure 1024 parallel envs don't cause thread pool issues
        let mut venv = make_vec_env(1024, 42);
        let actions: Vec<Action> = (0..1024)
            .map(|i| Action::Discrete((i % 2) as u32))
            .collect();
        // Step 10 times
        for _ in 0..10 {
            let batch = venv.step_all(&actions).unwrap();
            assert_eq!(batch.obs.len(), 1024);
        }
    }

    #[test]
    fn vec_env_parallel_determinism() {
        // Parallel stepping must be deterministic across runs
        let run = || {
            let mut venv = make_vec_env(64, 42);
            venv.reset_all(Some(42)).unwrap();
            let actions: Vec<Action> = (0..64).map(|i| Action::Discrete((i % 2) as u32)).collect();
            let mut all_rewards = Vec::new();
            for _ in 0..50 {
                let batch = venv.step_all(&actions).unwrap();
                all_rewards.extend(batch.rewards);
            }
            all_rewards
        };
        let run1 = run();
        let run2 = run();
        assert_eq!(run1, run2);
    }

    #[test]
    fn vec_env_auto_reset_on_done() {
        let mut venv = make_vec_env(2, 42);

        // Step many times - eventually envs will terminate and auto-reset
        for _ in 0..100 {
            let actions: Vec<Action> = (0..2).map(|_| Action::Discrete(1)).collect();
            match venv.step_all(&actions) {
                Ok(_batch) => {} // should always succeed due to auto-reset
                Err(e) => panic!("step_all should not error with auto-reset: {}", e),
            }
        }
    }
}

#[cfg(test)]
mod terminal_obs_tests {
    use super::*;
    use crate::env::builtins::CartPole;
    use crate::seed::derive_seed;

    fn make_vec_env(n: usize, seed: u64) -> VecEnv {
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| Box::new(CartPole::new(Some(derive_seed(seed, i)))) as Box<dyn RLEnv>)
            .collect();
        VecEnv::new(envs).unwrap()
    }

    #[test]
    fn step_result_has_terminal_obs_on_truncation() {
        let mut venv = make_vec_env(4, 42);
        venv.reset_all(Some(42)).unwrap();

        for _ in 0..600 {
            let actions: Vec<Action> = (0..4).map(|_| Action::Discrete(0)).collect();
            let batch = venv.step_all(&actions).unwrap();

            for i in 0..4 {
                if batch.truncated[i] {
                    assert!(
                        batch.terminal_obs[i].is_some(),
                        "terminal_obs must be Some when truncated"
                    );
                }
                if batch.terminated[i] {
                    assert!(
                        batch.terminal_obs[i].is_some(),
                        "terminal_obs must be Some when terminated"
                    );
                }
                if !batch.terminated[i] && !batch.truncated[i] {
                    assert!(
                        batch.terminal_obs[i].is_none(),
                        "terminal_obs must be None when not done"
                    );
                }
            }
        }
    }

    #[test]
    fn terminal_obs_has_correct_dimension() {
        let mut venv = make_vec_env(2, 42);
        venv.reset_all(Some(42)).unwrap();

        for _ in 0..200 {
            let actions: Vec<Action> = vec![Action::Discrete(1); 2];
            let batch = venv.step_all(&actions).unwrap();
            for i in 0..2 {
                if let Some(tobs) = &batch.terminal_obs[i] {
                    assert_eq!(tobs.len(), 4, "CartPole terminal obs must have dim 4");
                }
            }
        }
    }

    #[test]
    fn returned_obs_after_reset_is_fresh_not_terminal() {
        let mut venv = make_vec_env(1, 42);
        venv.reset_all(Some(42)).unwrap();

        for _ in 0..200 {
            let actions = vec![Action::Discrete(1)];
            let batch = venv.step_all(&actions).unwrap();
            if batch.terminated[0] {
                let fresh_obs = &batch.obs[0];
                for &v in fresh_obs {
                    assert!(
                        v.abs() <= 0.06,
                        "post-reset obs should be near zero, got {v}"
                    );
                }
                let tobs = batch.terminal_obs[0]
                    .as_ref()
                    .expect("terminal_obs must exist on termination");
                let max_abs = tobs.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                assert!(
                    max_abs > 0.05,
                    "terminal obs should be out-of-bounds, got max_abs={max_abs}"
                );
                break;
            }
        }
    }
}

#[cfg(test)]
mod pendulum_vec_env_tests {
    use super::*;
    use crate::env::builtins::Pendulum;
    use crate::seed::derive_seed;

    fn make_pendulum_vec_env(n: usize, seed: u64) -> VecEnv {
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| {
                let s = derive_seed(seed, i);
                Box::new(Pendulum::new(Some(s))) as Box<dyn RLEnv>
            })
            .collect();
        VecEnv::new(envs).unwrap()
    }

    #[test]
    fn pendulum_vec_env_step_continuous_actions() {
        let mut venv = make_pendulum_vec_env(4, 42);
        let actions: Vec<Action> = (0..4)
            .map(|i| Action::Continuous(vec![(i as f32 - 1.5) * 0.5]))
            .collect();
        let batch = venv.step_all(&actions).unwrap();
        assert_eq!(batch.obs.len(), 4);
        assert_eq!(batch.rewards.len(), 4);
        for obs in &batch.obs {
            assert_eq!(obs.len(), 3, "Pendulum obs should have 3 dims");
        }
    }

    #[test]
    fn pendulum_vec_env_step_flat() {
        let mut venv = make_pendulum_vec_env(4, 42);
        let actions: Vec<Action> = (0..4).map(|_| Action::Continuous(vec![0.5])).collect();
        let batch = venv.step_all_flat(&actions).unwrap();
        assert!(batch.obs.is_empty());
        assert_eq!(batch.obs_flat.len(), 4 * 3);
        assert_eq!(batch.obs_dim, 3);
    }

    #[test]
    fn pendulum_vec_env_auto_reset() {
        let mut venv = make_pendulum_vec_env(2, 42);
        // Step 300 times — past the 200 truncation limit
        for _ in 0..300 {
            let actions: Vec<Action> = (0..2).map(|_| Action::Continuous(vec![1.0])).collect();
            let batch = venv.step_all(&actions).unwrap();
            assert_eq!(batch.obs.len(), 2);
        }
    }

    #[test]
    fn pendulum_vec_env_action_space() {
        let venv = make_pendulum_vec_env(2, 42);
        match venv.action_space() {
            ActionSpace::Box { low, high, shape } => {
                assert_eq!(shape, &[1]);
                assert_eq!(low, &[-2.0]);
                assert_eq!(high, &[2.0]);
            }
            other => panic!("Expected Box action space, got {:?}", other),
        }
    }

    #[test]
    fn pendulum_vec_env_determinism() {
        let run = || {
            let mut venv = make_pendulum_vec_env(8, 42);
            venv.reset_all(Some(42)).unwrap();
            let actions: Vec<Action> = (0..8)
                .map(|i| Action::Continuous(vec![(i as f32) * 0.25 - 1.0]))
                .collect();
            let mut all_rewards = Vec::new();
            for _ in 0..50 {
                let batch = venv.step_all(&actions).unwrap();
                all_rewards.extend(batch.rewards);
            }
            all_rewards
        };
        let r1 = run();
        let r2 = run();
        assert_eq!(r1, r2);
    }
}
