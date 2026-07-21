//! Hindsight Experience Replay (HER) buffer.
//!
//! Stores transitions with goal information and performs goal relabeling
//! during sampling (Andrychowicz et al., 2017). Supports Final, Future(k),
//! and Episode relabeling strategies.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;

use super::episode::{EpisodeMeta, EpisodeTracker};
use super::ringbuf::{ReplayBuffer, SampledBatch};

/// HER goal relabeling strategy.
#[derive(Debug, Clone, Copy)]
pub enum HERStrategy {
    /// Replace goal with the final state achieved in the episode.
    Final,
    /// Replace goal with a future state sampled uniformly from the remainder.
    Future {
        /// Number of relabeled goals per original transition. Default: 4.
        k: usize,
    },
    /// Replace goal with a random state from the episode.
    Episode,
}

impl Default for HERStrategy {
    fn default() -> Self {
        HERStrategy::Future { k: 4 }
    }
}

/// Hindsight Experience Replay buffer.
///
/// Stores transitions with goal information and performs goal relabeling
/// during sampling. The obs vector layout is:
/// `[obs_core | achieved_goal | desired_goal | ...]`
#[derive(Debug)]
pub struct HERBuffer {
    buffer: ReplayBuffer,
    tracker: EpisodeTracker,
    obs_dim: usize,
    act_dim: usize,
    goal_dim: usize,
    achieved_goal_start: usize,
    desired_goal_start: usize,
    capacity: usize,
    strategy: HERStrategy,
    goal_tolerance: f32,
}

impl HERBuffer {
    /// Create a new HER buffer.
    ///
    /// # Arguments
    /// * `capacity` - maximum transitions
    /// * `obs_dim` - full observation dimension (includes goal components)
    /// * `act_dim` - action dimension
    /// * `goal_dim` - goal vector dimension
    /// * `achieved_goal_start` - index within obs where achieved goal starts
    /// * `desired_goal_start` - index within obs where desired goal starts
    /// * `strategy` - relabeling strategy
    /// * `goal_tolerance` - tolerance for sparse reward computation
    // All 8 params are distinct required dimensions/config for a HER buffer; no natural grouping without breaking the public API.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        capacity: usize,
        obs_dim: usize,
        act_dim: usize,
        goal_dim: usize,
        achieved_goal_start: usize,
        desired_goal_start: usize,
        strategy: HERStrategy,
        goal_tolerance: f32,
    ) -> Self {
        Self {
            buffer: ReplayBuffer::new(capacity, obs_dim, act_dim),
            tracker: EpisodeTracker::new(capacity),
            obs_dim,
            act_dim,
            goal_dim,
            achieved_goal_start,
            desired_goal_start,
            capacity,
            strategy,
            goal_tolerance,
        }
    }

    /// Push a single transition, notifying the episode tracker.
    pub fn push_slices(
        &mut self,
        obs: &[f32],
        next_obs: &[f32],
        action: &[f32],
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> Result<(), RloxError> {
        let write_pos = self.buffer.write_pos();
        let was_full = self.buffer.len() == self.capacity;

        if was_full {
            self.tracker.invalidate_overwritten(write_pos, 1);
        }

        self.buffer
            .push_slices(obs, next_obs, action, reward, terminated, truncated)?;

        let done = terminated || truncated;
        self.tracker.notify_push(write_pos, done);

        Ok(())
    }

    /// Sample a batch with HER relabeling.
    ///
    /// `her_ratio` controls the fraction of samples that get relabeled goals.
    /// The remaining samples use their original goals.
    pub fn sample_with_relabeling(
        &self,
        batch_size: usize,
        her_ratio: f32,
        seed: u64,
    ) -> Result<SampledBatch, RloxError> {
        if self.buffer.is_empty() {
            return Err(RloxError::BufferError("buffer is empty".into()));
        }

        let episodes = self.tracker.episodes();
        let complete: Vec<usize> = episodes
            .iter()
            .enumerate()
            .filter(|(_, ep)| ep.complete)
            .map(|(i, _)| i)
            .collect();

        if complete.is_empty() {
            return Err(RloxError::BufferError(
                "no complete episodes for HER relabeling".into(),
            ));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n_relabeled = ((batch_size as f32) * her_ratio).ceil() as usize;
        let n_original = batch_size - n_relabeled;

        let mut batch = SampledBatch::with_capacity(batch_size, self.obs_dim, self.act_dim);

        // Sample original (unrelabeled) transitions
        if n_original > 0 {
            let original = self.buffer.sample(n_original, rng.random())?;
            batch.observations.extend_from_slice(&original.observations);
            batch
                .next_observations
                .extend_from_slice(&original.next_observations);
            batch.actions.extend_from_slice(&original.actions);
            batch.rewards.extend_from_slice(&original.rewards);
            batch.terminated.extend_from_slice(&original.terminated);
            batch.truncated.extend_from_slice(&original.truncated);
        }

        // Sample relabeled transitions
        for _ in 0..n_relabeled {
            // Pick a random complete episode
            let ep_idx = complete[rng.random_range(0..complete.len())];
            let ep = &episodes[ep_idx];

            // Pick a random transition within the episode
            let trans_offset = rng.random_range(0..ep.length);
            let trans_idx = (ep.start + trans_offset) % self.capacity;

            // Get the original transition
            let (obs, next_obs, action, _reward, terminated, truncated) =
                self.buffer.get(trans_idx);

            // Compute the relabel index based on strategy
            let relabel_offset = match self.strategy {
                HERStrategy::Final => ep.length - 1,
                HERStrategy::Future { .. } => {
                    if trans_offset >= ep.length - 1 {
                        // Already at the end, use the same position
                        trans_offset
                    } else {
                        rng.random_range((trans_offset + 1)..ep.length)
                    }
                }
                HERStrategy::Episode => rng.random_range(0..ep.length),
            };
            let relabel_idx = (ep.start + relabel_offset) % self.capacity;

            // Get the achieved goal from the relabel transition's next_obs
            let (_, relabel_next_obs, _, _, _, _) = self.buffer.get(relabel_idx);
            let new_goal = &relabel_next_obs
                [self.achieved_goal_start..self.achieved_goal_start + self.goal_dim];

            // Create modified observation with new desired goal
            let mut new_obs = obs.to_vec();
            new_obs[self.desired_goal_start..self.desired_goal_start + self.goal_dim]
                .copy_from_slice(new_goal);

            let mut new_next_obs = next_obs.to_vec();
            new_next_obs[self.desired_goal_start..self.desired_goal_start + self.goal_dim]
                .copy_from_slice(new_goal);

            // Compute new reward based on achieved goal in next_obs vs new desired goal
            let achieved_in_next =
                &next_obs[self.achieved_goal_start..self.achieved_goal_start + self.goal_dim];
            let new_reward = sparse_goal_reward(achieved_in_next, new_goal, self.goal_tolerance);

            batch.observations.extend_from_slice(&new_obs);
            batch.next_observations.extend_from_slice(&new_next_obs);
            batch.actions.extend_from_slice(action);
            batch.rewards.push(new_reward);
            batch.terminated.push(terminated);
            batch.truncated.push(truncated);
        }

        batch.batch_size = batch_size;

        Ok(batch)
    }

    /// Compute relabeling indices for a given episode and transition.
    ///
    /// Returns indices (offsets within the episode) to use as substitute goals.
    pub fn compute_relabel_indices(
        &self,
        episode: &EpisodeMeta,
        transition_offset: usize,
        seed: u64,
    ) -> Vec<usize> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self.strategy {
            HERStrategy::Final => vec![episode.length - 1],
            HERStrategy::Future { k } => {
                if transition_offset >= episode.length - 1 {
                    // At the last step, can only relabel with itself
                    vec![transition_offset; k]
                } else {
                    (0..k)
                        .map(|_| rng.random_range((transition_offset + 1)..episode.length))
                        .collect()
                }
            }
            HERStrategy::Episode => {
                vec![rng.random_range(0..episode.length)]
            }
        }
    }

    /// Number of valid transitions currently stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Number of complete episodes currently tracked.
    pub fn num_complete_episodes(&self) -> usize {
        self.tracker.num_complete_episodes()
    }
}

/// Compute sparse goal-conditioned reward.
///
/// Returns `0.0` if `||achieved - desired||_2 < tolerance`, else `-1.0`.
///
/// Uses squared distance comparison to avoid a costly `sqrt`.
#[inline]
pub fn sparse_goal_reward(achieved: &[f32], desired: &[f32], tolerance: f32) -> f32 {
    let dist_sq: f32 = achieved
        .iter()
        .zip(desired.iter())
        .map(|(&a, &d)| (a - d) * (a - d))
        .sum();
    if dist_sq < tolerance * tolerance {
        0.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build an obs vector with embedded achieved and desired goals.
    /// Layout: [core(2) | achieved_goal(goal_dim) | desired_goal(goal_dim)]
    fn make_obs(core: &[f32], achieved: &[f32], desired: &[f32]) -> Vec<f32> {
        let mut obs = Vec::with_capacity(core.len() + achieved.len() + desired.len());
        obs.extend_from_slice(core);
        obs.extend_from_slice(achieved);
        obs.extend_from_slice(desired);
        obs
    }

    fn make_her_buffer(capacity: usize, goal_dim: usize) -> HERBuffer {
        let core_dim = 2;
        let obs_dim = core_dim + goal_dim * 2; // core + achieved + desired
        HERBuffer::new(
            capacity,
            obs_dim,
            1, // act_dim
            goal_dim,
            core_dim,               // achieved_goal_start
            core_dim + goal_dim,    // desired_goal_start
            HERStrategy::default(), // Future { k: 4 }
            0.05,                   // goal_tolerance
        )
    }

    /// Push an episode where the agent moves from origin toward a goal.
    fn push_goal_episode(buf: &mut HERBuffer, length: usize, goal_dim: usize) {
        let desired_goal = vec![10.0; goal_dim];
        for i in 0..length {
            let progress = (i as f32 + 1.0) / length as f32;
            let achieved = vec![10.0 * progress; goal_dim];
            let next_achieved = vec![10.0 * (progress + 1.0 / length as f32).min(1.0); goal_dim];
            let core = vec![progress, progress];

            let obs = make_obs(&core, &achieved, &desired_goal);
            let next_obs = make_obs(
                &[progress + 0.1, progress + 0.1],
                &next_achieved,
                &desired_goal,
            );
            let action = vec![0.0];
            let reward = -1.0; // sparse: not at goal yet
            let done = i == length - 1;

            buf.push_slices(&obs, &next_obs, &action, reward, done, false)
                .unwrap();
        }
    }

    #[test]
    fn test_her_new_is_empty() {
        let buf = make_her_buffer(100, 3);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_her_push_increments() {
        let mut buf = make_her_buffer(100, 3);
        push_goal_episode(&mut buf, 5, 3);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.num_complete_episodes(), 1);
    }

    #[test]
    fn test_final_strategy_uses_last_state() {
        let goal_dim = 2;
        let core_dim = 2;
        let obs_dim = core_dim + goal_dim * 2;
        let mut buf = HERBuffer::new(
            100,
            obs_dim,
            1,
            goal_dim,
            core_dim,
            core_dim + goal_dim,
            HERStrategy::Final,
            0.05,
        );
        push_goal_episode(&mut buf, 5, goal_dim);

        let ep = &buf.tracker.episodes()[0];
        let indices = buf.compute_relabel_indices(ep, 2, 42);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 4); // last step of episode (length 5)
    }

    #[test]
    fn test_future_strategy_picks_future_state() {
        let goal_dim = 2;
        let core_dim = 2;
        let obs_dim = core_dim + goal_dim * 2;
        let mut buf = HERBuffer::new(
            100,
            obs_dim,
            1,
            goal_dim,
            core_dim,
            core_dim + goal_dim,
            HERStrategy::Future { k: 4 },
            0.05,
        );
        push_goal_episode(&mut buf, 10, goal_dim);

        let ep = &buf.tracker.episodes()[0];
        let indices = buf.compute_relabel_indices(ep, 3, 42);
        assert_eq!(indices.len(), 4);
        for &idx in &indices {
            assert!(
                idx > 3,
                "future index {idx} should be > transition offset 3"
            );
            assert!(idx < 10, "future index {idx} should be < episode length 10");
        }
    }

    #[test]
    fn test_episode_strategy_picks_any_state() {
        let goal_dim = 2;
        let core_dim = 2;
        let obs_dim = core_dim + goal_dim * 2;
        let mut buf = HERBuffer::new(
            100,
            obs_dim,
            1,
            goal_dim,
            core_dim,
            core_dim + goal_dim,
            HERStrategy::Episode,
            0.05,
        );
        push_goal_episode(&mut buf, 10, goal_dim);

        let ep = &buf.tracker.episodes()[0];
        let indices = buf.compute_relabel_indices(ep, 5, 42);
        assert_eq!(indices.len(), 1);
        assert!(indices[0] < 10);
    }

    #[test]
    fn test_sparse_goal_reward_achieved() {
        let achieved = [1.0, 2.0, 3.0];
        let desired = [1.0, 2.0, 3.0];
        assert_eq!(sparse_goal_reward(&achieved, &desired, 0.05), 0.0);
    }

    #[test]
    fn test_sparse_goal_reward_not_achieved() {
        let achieved = [1.0, 2.0, 3.0];
        let desired = [10.0, 20.0, 30.0];
        assert_eq!(sparse_goal_reward(&achieved, &desired, 0.05), -1.0);
    }

    #[test]
    fn test_relabel_indices_future_k4() {
        let goal_dim = 2;
        let core_dim = 2;
        let obs_dim = core_dim + goal_dim * 2;
        let mut buf = HERBuffer::new(
            100,
            obs_dim,
            1,
            goal_dim,
            core_dim,
            core_dim + goal_dim,
            HERStrategy::Future { k: 4 },
            0.05,
        );
        push_goal_episode(&mut buf, 10, goal_dim);

        let ep = &buf.tracker.episodes()[0];
        let indices = buf.compute_relabel_indices(ep, 3, 42);
        assert_eq!(indices.len(), 4);
        for &idx in &indices {
            assert!(idx > 3 && idx < 10);
        }
    }

    #[test]
    fn test_relabel_indices_deterministic() {
        let goal_dim = 2;
        let core_dim = 2;
        let obs_dim = core_dim + goal_dim * 2;
        let mut buf = HERBuffer::new(
            100,
            obs_dim,
            1,
            goal_dim,
            core_dim,
            core_dim + goal_dim,
            HERStrategy::Future { k: 4 },
            0.05,
        );
        push_goal_episode(&mut buf, 10, goal_dim);

        let ep = &buf.tracker.episodes()[0];
        let i1 = buf.compute_relabel_indices(ep, 3, 42);
        let i2 = buf.compute_relabel_indices(ep, 3, 42);
        assert_eq!(i1, i2);
    }

    #[test]
    fn test_her_sample_batch_shape() {
        let goal_dim = 3;
        let mut buf = make_her_buffer(100, goal_dim);
        push_goal_episode(&mut buf, 10, goal_dim);

        let batch = buf.sample_with_relabeling(8, 0.8, 42).unwrap();
        let obs_dim = 2 + goal_dim * 2;
        assert_eq!(batch.batch_size, 8);
        assert_eq!(batch.observations.len(), 8 * obs_dim);
        assert_eq!(batch.actions.len(), 8);
        assert_eq!(batch.rewards.len(), 8);
    }

    #[test]
    fn test_her_ratio_controls_relabeling() {
        let goal_dim = 2;
        let mut buf = make_her_buffer(200, goal_dim);
        // Push multiple episodes so we have enough data
        for _ in 0..10 {
            push_goal_episode(&mut buf, 10, goal_dim);
        }

        // With ratio=0.0, no relabeling => all rewards should be -1.0 (original)
        let batch = buf.sample_with_relabeling(32, 0.0, 42).unwrap();
        // All original rewards are -1.0
        for &r in &batch.rewards {
            assert_eq!(
                r, -1.0,
                "with ratio=0, all rewards should be original (-1.0)"
            );
        }
    }

    #[test]
    fn test_her_with_ring_wrap() {
        let goal_dim = 2;
        let mut buf = make_her_buffer(50, goal_dim);
        // Push 100 transitions (wraps around)
        for _ in 0..10 {
            push_goal_episode(&mut buf, 10, goal_dim);
        }
        assert_eq!(buf.len(), 50);
        // Should still be able to sample
        let result = buf.sample_with_relabeling(4, 0.8, 42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_her_empty_buffer_errors() {
        let buf = make_her_buffer(100, 3);
        let result = buf.sample_with_relabeling(4, 0.8, 42);
        assert!(result.is_err());
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_relabel_indices_in_range(
                ep_len in 2usize..20,
                trans_offset in 0usize..19,
            ) {
                let trans_offset = trans_offset.min(ep_len - 1);
                let goal_dim = 2;
                let core_dim = 2;
                let obs_dim = core_dim + goal_dim * 2;
                let buf = HERBuffer::new(
                    100, obs_dim, 1, goal_dim, core_dim, core_dim + goal_dim,
                    HERStrategy::Future { k: 4 }, 0.05,
                );
                let ep = EpisodeMeta { start: 0, length: ep_len, complete: true };
                let indices = buf.compute_relabel_indices(&ep, trans_offset, 42);
                for &idx in &indices {
                    prop_assert!(idx < ep_len, "index {idx} >= episode length {ep_len}");
                }
            }

            #[test]
            fn prop_future_indices_strictly_future(
                ep_len in 3usize..20,
                trans_offset in 0usize..18,
            ) {
                let trans_offset = trans_offset.min(ep_len - 2); // ensure room for future
                let goal_dim = 2;
                let core_dim = 2;
                let obs_dim = core_dim + goal_dim * 2;
                let buf = HERBuffer::new(
                    100, obs_dim, 1, goal_dim, core_dim, core_dim + goal_dim,
                    HERStrategy::Future { k: 4 }, 0.05,
                );
                let ep = EpisodeMeta { start: 0, length: ep_len, complete: true };
                let indices = buf.compute_relabel_indices(&ep, trans_offset, 42);
                for &idx in &indices {
                    prop_assert!(idx > trans_offset,
                        "future index {idx} should be > offset {trans_offset}");
                }
            }

            #[test]
            fn prop_sparse_reward_binary(
                a0 in -10.0f32..10.0,
                a1 in -10.0f32..10.0,
                d0 in -10.0f32..10.0,
                d1 in -10.0f32..10.0,
            ) {
                let r = sparse_goal_reward(&[a0, a1], &[d0, d1], 0.05);
                prop_assert!(r == 0.0 || r == -1.0, "reward should be 0.0 or -1.0, got {r}");
            }
        }
    }
}
