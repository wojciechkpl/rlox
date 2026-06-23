//! Read-only offline dataset buffer for offline RL algorithms.
//!
//! Unlike [`ReplayBuffer`], this buffer is loaded once from a static dataset
//! and never modified. It supports:
//! - Uniform i.i.d. transition sampling (for TD3+BC, IQL, CQL, BC)
//! - Trajectory subsequence sampling (for Decision Transformer)
//! - Return-conditioned sampling (for return-conditioned methods)
//! - Dataset normalization statistics
//!
//! Designed for D4RL/Minari-scale datasets (1M+ transitions).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;

/// Statistics about the loaded dataset.
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub n_transitions: usize,
    pub n_episodes: usize,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub mean_return: f32,
    pub std_return: f32,
    pub min_return: f32,
    pub max_return: f32,
    pub mean_episode_length: f32,
}

/// A batch of i.i.d. sampled transitions.
#[derive(Debug, Clone)]
pub struct OfflineBatch {
    pub obs: Vec<f32>,       // [batch_size * obs_dim]
    pub next_obs: Vec<f32>,  // [batch_size * obs_dim]
    pub actions: Vec<f32>,   // [batch_size * act_dim]
    pub rewards: Vec<f32>,   // [batch_size]
    pub terminated: Vec<u8>, // [batch_size]
    pub obs_dim: usize,
    pub act_dim: usize,
}

/// A batch of contiguous trajectory subsequences.
#[derive(Debug, Clone)]
pub struct TrajectoryBatch {
    pub obs: Vec<f32>,           // [batch_size * seq_len * obs_dim]
    pub actions: Vec<f32>,       // [batch_size * seq_len * act_dim]
    pub rewards: Vec<f32>,       // [batch_size * seq_len]
    pub returns_to_go: Vec<f32>, // [batch_size * seq_len]
    pub timesteps: Vec<u32>,     // [batch_size * seq_len]
    pub mask: Vec<u8>,           // [batch_size * seq_len] (1 = valid, 0 = padding)
    pub seq_len: usize,
    pub obs_dim: usize,
    pub act_dim: usize,
}

/// Read-only offline dataset buffer.
pub struct OfflineDatasetBuffer {
    obs: Vec<f32>,
    next_obs: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<u8>,
    #[allow(dead_code)]
    truncated: Vec<u8>,

    // Episode boundary tracking
    episode_starts: Vec<usize>,
    episode_lengths: Vec<usize>,
    episode_returns: Vec<f32>,

    obs_dim: usize,
    act_dim: usize,
    len: usize,

    // Normalization (computed lazily)
    obs_mean: Option<Vec<f32>>,
    obs_std: Option<Vec<f32>>,
    reward_mean: Option<f32>,
    reward_std: Option<f32>,
}

impl OfflineDatasetBuffer {
    /// Create from flat arrays.
    ///
    /// Arrays must be row-major: obs has length `n * obs_dim`, etc.
    // Each parameter is a distinct flat array or dimension; grouping into a struct would complicate the Python FFI caller.
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays(
        obs: Vec<f32>,
        next_obs: Vec<f32>,
        actions: Vec<f32>,
        rewards: Vec<f32>,
        terminated: Vec<u8>,
        truncated: Vec<u8>,
        obs_dim: usize,
        act_dim: usize,
    ) -> Result<Self, RloxError> {
        let n = rewards.len();

        if obs.len() != n * obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs length = {} * {} = {}", n, obs_dim, n * obs_dim),
                got: format!("{}", obs.len()),
            });
        }
        if next_obs.len() != n * obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("next_obs length = {}", n * obs_dim),
                got: format!("{}", next_obs.len()),
            });
        }
        if actions.len() != n * act_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("actions length = {} * {} = {}", n, act_dim, n * act_dim),
                got: format!("{}", actions.len()),
            });
        }
        if terminated.len() != n || truncated.len() != n {
            return Err(RloxError::ShapeMismatch {
                expected: format!("terminated/truncated length = {}", n),
                got: format!(
                    "terminated={}, truncated={}",
                    terminated.len(),
                    truncated.len()
                ),
            });
        }

        // Detect episode boundaries
        let mut episode_starts = vec![0usize];
        let mut episode_returns = Vec::new();
        let mut ep_return = 0.0f32;

        for i in 0..n {
            ep_return += rewards[i];
            let done = terminated[i] != 0 || truncated[i] != 0;
            if done || i == n - 1 {
                episode_returns.push(ep_return);
                if i + 1 < n {
                    episode_starts.push(i + 1);
                }
                ep_return = 0.0;
            }
        }

        let episode_lengths: Vec<usize> = episode_starts
            .windows(2)
            .map(|w| w[1] - w[0])
            .chain(std::iter::once(n - episode_starts.last().unwrap_or(&0)))
            .collect();

        Ok(Self {
            obs,
            next_obs,
            actions,
            rewards,
            terminated,
            truncated,
            episode_starts,
            episode_lengths,
            episode_returns,
            obs_dim,
            act_dim,
            len: n,
            obs_mean: None,
            obs_std: None,
            reward_mean: None,
            reward_std: None,
        })
    }

    /// Number of transitions in the dataset.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of episodes in the dataset.
    pub fn n_episodes(&self) -> usize {
        self.episode_starts.len()
    }

    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    pub fn act_dim(&self) -> usize {
        self.act_dim
    }

    /// Compute and cache normalization statistics.
    #[allow(clippy::needless_range_loop)]
    pub fn compute_normalization(&mut self) {
        let n = self.len;
        let d = self.obs_dim;

        // Obs mean and std
        let mut mean = vec![0.0f64; d];
        for i in 0..n {
            for j in 0..d {
                mean[j] += self.obs[i * d + j] as f64;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }

        let mut var = vec![0.0f64; d];
        for i in 0..n {
            for j in 0..d {
                let diff = self.obs[i * d + j] as f64 - mean[j];
                var[j] += diff * diff;
            }
        }
        for v in &mut var {
            *v = (*v / n as f64).sqrt().max(1e-8);
        }

        self.obs_mean = Some(mean.iter().map(|&x| x as f32).collect());
        self.obs_std = Some(var.iter().map(|&x| x as f32).collect());

        // Reward mean and std
        let r_mean = self.rewards.iter().map(|&r| r as f64).sum::<f64>() / n as f64;
        let r_var = self
            .rewards
            .iter()
            .map(|&r| {
                let d = r as f64 - r_mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        self.reward_mean = Some(r_mean as f32);
        self.reward_std = Some((r_var.sqrt().max(1e-8)) as f32);
    }

    /// Sample i.i.d. transitions uniformly.
    pub fn sample(&self, batch_size: usize, seed: u64) -> OfflineBatch {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let d = self.obs_dim;
        let a = self.act_dim;

        let mut obs = Vec::with_capacity(batch_size * d);
        let mut next_obs = Vec::with_capacity(batch_size * d);
        let mut actions = Vec::with_capacity(batch_size * a);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut terminated = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = rng.random_range(0..self.len);

            obs.extend_from_slice(&self.obs[idx * d..(idx + 1) * d]);
            next_obs.extend_from_slice(&self.next_obs[idx * d..(idx + 1) * d]);
            actions.extend_from_slice(&self.actions[idx * a..(idx + 1) * a]);
            rewards.push(self.rewards[idx]);
            terminated.push(self.terminated[idx]);
        }

        // Apply normalization if available
        if let (Some(mean), Some(std)) = (&self.obs_mean, &self.obs_std) {
            for i in 0..batch_size {
                for j in 0..d {
                    obs[i * d + j] = (obs[i * d + j] - mean[j]) / std[j];
                    next_obs[i * d + j] = (next_obs[i * d + j] - mean[j]) / std[j];
                }
            }
        }

        OfflineBatch {
            obs,
            next_obs,
            actions,
            rewards,
            terminated,
            obs_dim: d,
            act_dim: a,
        }
    }

    /// Sample contiguous trajectory subsequences.
    ///
    /// Each sample is a contiguous window of `seq_len` transitions from a
    /// single episode. If the episode is shorter than `seq_len`, the sequence
    /// is right-padded with zeros and the mask indicates valid positions.
    pub fn sample_trajectories(
        &self,
        batch_size: usize,
        seq_len: usize,
        seed: u64,
    ) -> TrajectoryBatch {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let d = self.obs_dim;
        let a = self.act_dim;
        let n_eps = self.n_episodes();

        let total = batch_size * seq_len;
        let mut obs = vec![0.0f32; total * d];
        let mut actions = vec![0.0f32; total * a];
        let mut rewards = vec![0.0f32; total];
        let mut returns_to_go = vec![0.0f32; total];
        let mut timesteps = vec![0u32; total];
        let mut mask = vec![0u8; total];

        for b in 0..batch_size {
            let ep_idx = rng.random_range(0..n_eps);
            let ep_start = self.episode_starts[ep_idx];
            let ep_len = self.episode_lengths[ep_idx];

            // Random start within episode
            let max_start = ep_len.saturating_sub(seq_len);
            let start_offset = rng.random_range(0..=max_start);
            let actual_len = seq_len.min(ep_len - start_offset);

            // Compute returns-to-go for this episode segment
            let mut rtg = vec![0.0f32; actual_len];
            if actual_len > 0 {
                rtg[actual_len - 1] = self.rewards[ep_start + start_offset + actual_len - 1];
                for t in (0..actual_len - 1).rev() {
                    rtg[t] = self.rewards[ep_start + start_offset + t] + rtg[t + 1];
                }
            }

            for (t, rtg_val) in rtg.iter().enumerate() {
                let src_idx = ep_start + start_offset + t;
                let dst_idx = b * seq_len + t;

                obs[dst_idx * d..(dst_idx + 1) * d]
                    .copy_from_slice(&self.obs[src_idx * d..(src_idx + 1) * d]);
                actions[dst_idx * a..(dst_idx + 1) * a]
                    .copy_from_slice(&self.actions[src_idx * a..(src_idx + 1) * a]);
                rewards[dst_idx] = self.rewards[src_idx];
                returns_to_go[dst_idx] = *rtg_val;
                timesteps[dst_idx] = (start_offset + t) as u32;
                mask[dst_idx] = 1;
            }
        }

        TrajectoryBatch {
            obs,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            mask,
            seq_len,
            obs_dim: d,
            act_dim: a,
        }
    }

    /// Get dataset statistics.
    pub fn stats(&self) -> DatasetStats {
        let returns = &self.episode_returns;
        let n_eps = returns.len();

        let mean_return = if n_eps > 0 {
            returns.iter().sum::<f32>() / n_eps as f32
        } else {
            0.0
        };

        let std_return = if n_eps > 1 {
            let var: f32 = returns
                .iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f32>()
                / (n_eps - 1) as f32;
            var.sqrt()
        } else {
            0.0
        };

        let min_return = returns.iter().cloned().reduce(f32::min).unwrap_or(0.0);
        let max_return = returns.iter().cloned().reduce(f32::max).unwrap_or(0.0);

        let mean_ep_len = if n_eps > 0 {
            self.episode_lengths.iter().sum::<usize>() as f32 / n_eps as f32
        } else {
            0.0
        };

        DatasetStats {
            n_transitions: self.len,
            n_episodes: n_eps,
            obs_dim: self.obs_dim,
            act_dim: self.act_dim,
            mean_return,
            std_return,
            min_return,
            max_return,
            mean_episode_length: mean_ep_len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_dataset(
        n: usize,
        obs_dim: usize,
        act_dim: usize,
        ep_len: usize,
    ) -> OfflineDatasetBuffer {
        let rewards = vec![1.0f32; n];
        let mut terminated = vec![0u8; n];
        let truncated = vec![0u8; n];

        // Mark episode boundaries
        for (i, t) in terminated.iter_mut().enumerate().take(n) {
            if (i + 1).is_multiple_of(ep_len) {
                *t = 1;
            }
        }

        OfflineDatasetBuffer::from_arrays(
            vec![0.1f32; n * obs_dim],
            vec![0.2f32; n * obs_dim],
            vec![0.0f32; n * act_dim],
            rewards,
            terminated,
            truncated,
            obs_dim,
            act_dim,
        )
        .unwrap()
    }

    #[test]
    fn test_load_from_arrays() {
        let buf = make_test_dataset(100, 4, 1, 10);
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.obs_dim(), 4);
        assert_eq!(buf.act_dim(), 1);
    }

    #[test]
    fn test_episode_boundary_detection() {
        let buf = make_test_dataset(100, 4, 1, 10);
        assert_eq!(buf.n_episodes(), 10);
        assert_eq!(buf.episode_lengths, vec![10; 10]);
    }

    #[test]
    fn test_episode_returns() {
        let buf = make_test_dataset(100, 4, 1, 10);
        // Each episode has 10 steps with reward 1.0 → return = 10.0
        for &ret in &buf.episode_returns {
            assert!((ret - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sample_uniform_shapes() {
        let buf = make_test_dataset(1000, 4, 2, 100);
        let batch = buf.sample(32, 42);
        assert_eq!(batch.obs.len(), 32 * 4);
        assert_eq!(batch.next_obs.len(), 32 * 4);
        assert_eq!(batch.actions.len(), 32 * 2);
        assert_eq!(batch.rewards.len(), 32);
        assert_eq!(batch.terminated.len(), 32);
    }

    #[test]
    fn test_sample_deterministic() {
        let buf = make_test_dataset(1000, 4, 1, 100);
        let b1 = buf.sample(32, 42);
        let b2 = buf.sample(32, 42);
        assert_eq!(b1.obs, b2.obs);
        assert_eq!(b1.rewards, b2.rewards);
    }

    #[test]
    fn test_sample_different_seeds() {
        // Use varying obs so different indices produce different data
        let n = 1000;
        let obs_dim = 4;
        let obs: Vec<f32> = (0..n * obs_dim).map(|i| i as f32 * 0.001).collect();
        let mut terminated = vec![0u8; n];
        for i in (99..n).step_by(100) {
            terminated[i] = 1;
        }
        let buf = OfflineDatasetBuffer::from_arrays(
            obs.clone(),
            obs,
            vec![0.0; n],
            vec![1.0; n],
            terminated,
            vec![0; n],
            obs_dim,
            1,
        )
        .unwrap();

        let b1 = buf.sample(32, 42);
        let b2 = buf.sample(32, 99);
        assert_ne!(
            b1.obs, b2.obs,
            "Different seeds should produce different samples"
        );
    }

    #[test]
    fn test_normalization() {
        let mut buf = make_test_dataset(1000, 4, 1, 100);
        buf.compute_normalization();
        assert!(buf.obs_mean.is_some());
        assert!(buf.obs_std.is_some());

        let batch = buf.sample(32, 42);
        // Normalized obs should have roughly zero mean
        let mean: f32 = batch.obs.iter().sum::<f32>() / batch.obs.len() as f32;
        assert!(
            mean.abs() < 1.0,
            "Normalized mean should be near 0, got {mean}"
        );
    }

    #[test]
    fn test_sample_trajectories_shapes() {
        let buf = make_test_dataset(1000, 4, 2, 100);
        let batch = buf.sample_trajectories(8, 20, 42);
        assert_eq!(batch.obs.len(), 8 * 20 * 4);
        assert_eq!(batch.actions.len(), 8 * 20 * 2);
        assert_eq!(batch.rewards.len(), 8 * 20);
        assert_eq!(batch.returns_to_go.len(), 8 * 20);
        assert_eq!(batch.timesteps.len(), 8 * 20);
        assert_eq!(batch.mask.len(), 8 * 20);
    }

    #[test]
    fn test_sample_trajectories_mask() {
        // Short episodes → padding
        let buf = make_test_dataset(50, 4, 1, 5); // 10 episodes of length 5
        let batch = buf.sample_trajectories(4, 10, 42); // request seq_len=10

        // Each trajectory comes from ep_len=5 episode, so at most 5 valid
        for b in 0..4 {
            let valid: usize = (0..10).map(|t| batch.mask[b * 10 + t] as usize).sum();
            assert!(
                valid <= 5,
                "Valid mask count should be <= ep_len=5, got {valid}"
            );
            assert!(valid > 0, "Should have at least 1 valid step");
        }
    }

    #[test]
    fn test_sample_trajectories_returns_to_go() {
        let buf = make_test_dataset(100, 4, 1, 10);
        let batch = buf.sample_trajectories(1, 10, 42);

        // Returns-to-go should be decreasing within valid region
        let mut prev_rtg = f32::MAX;
        for t in 0..10 {
            if batch.mask[t] == 1 {
                assert!(
                    batch.returns_to_go[t] <= prev_rtg + 1e-5,
                    "RTG should be non-increasing, got {} after {}",
                    batch.returns_to_go[t],
                    prev_rtg
                );
                prev_rtg = batch.returns_to_go[t];
            }
        }
    }

    #[test]
    fn test_stats() {
        let buf = make_test_dataset(100, 4, 1, 10);
        let stats = buf.stats();
        assert_eq!(stats.n_transitions, 100);
        assert_eq!(stats.n_episodes, 10);
        assert_eq!(stats.obs_dim, 4);
        assert_eq!(stats.act_dim, 1);
        assert!((stats.mean_return - 10.0).abs() < 1e-5);
        assert!((stats.mean_episode_length - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_empty_dataset_error() {
        let result =
            OfflineDatasetBuffer::from_arrays(vec![], vec![], vec![], vec![], vec![], vec![], 4, 1);
        // Empty is technically valid (0 transitions)
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_mismatched_lengths_error() {
        let result = OfflineDatasetBuffer::from_arrays(
            vec![0.0; 40], // 10 * 4
            vec![0.0; 40],
            vec![0.0; 10], // 10 * 1
            vec![0.0; 10],
            vec![0; 5], // WRONG: should be 10
            vec![0; 10],
            4,
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_variable_episode_lengths() {
        // Create dataset with variable-length episodes
        let n = 25; // episodes: 5 + 8 + 12 = 25
        let obs_dim = 2;
        let act_dim = 1;
        let mut terminated = vec![0u8; n];
        terminated[4] = 1; // episode 1: steps 0-4
        terminated[12] = 1; // episode 2: steps 5-12
        terminated[24] = 1; // episode 3: steps 13-24

        let buf = OfflineDatasetBuffer::from_arrays(
            vec![0.0; n * obs_dim],
            vec![0.0; n * obs_dim],
            vec![0.0; n * act_dim],
            vec![1.0; n],
            terminated,
            vec![0; n],
            obs_dim,
            act_dim,
        )
        .unwrap();

        assert_eq!(buf.n_episodes(), 3);
        assert_eq!(buf.episode_lengths, vec![5, 8, 12]);
    }
}
