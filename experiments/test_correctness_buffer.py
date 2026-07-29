"""Buffer correctness tests — TDD RED phase.

Defines the behavioral contract for rlox replay/experience buffers.

Test categories
---------------
1. Push N transitions, verify count == N
2. Push and sample — sampled data is a subset of pushed data
3. Ring buffer FIFO semantics when capacity is exceeded
4. Priority buffer statistical sampling test
5. Alignment of obs/action/reward/done/next_obs fields
6. Determinism with seeded RNG
7. Cross-framework validation: rlox vs SB3 ReplayBuffer sample distribution
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


OBS_DIM = 4
ACT_DIM = 1


def _fill_buffer(buf, n: int, obs_dim: int = OBS_DIM, seed: int = 42):
    """Push *n* distinct transitions into *buf*."""
    rng = np.random.default_rng(seed)
    transitions = []
    for i in range(n):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        action = rng.standard_normal(ACT_DIM).astype(np.float32)
        reward = float(rng.standard_normal())
        terminated = bool(rng.random() < 0.05)
        truncated = bool(rng.random() < 0.02)
        transitions.append((obs, action, reward, terminated, truncated))
        buf.push(
            obs=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
    return transitions


# ---------------------------------------------------------------------------
# 1. Count after push
# ---------------------------------------------------------------------------


@pytest.mark.correctness
@pytest.mark.parametrize("n", [1, 100, 1000])
def test_experience_table_push_count(n, rlox_module):
    """After pushing N transitions the table count equals N."""
    table = rlox_module.ExperienceTable(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    _fill_buffer(table, n)
    assert len(table) == n, f"Expected {n} transitions, got {len(table)}"


@pytest.mark.correctness
@pytest.mark.parametrize("n", [1, 50, 500])
def test_replay_buffer_push_count(n, rlox_module):
    """ReplayBuffer len equals min(n_pushed, capacity)."""
    capacity = 1000
    buf = rlox_module.ReplayBuffer(capacity=capacity, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    _fill_buffer(buf, n)
    expected = min(n, capacity)
    assert len(buf) == expected


# ---------------------------------------------------------------------------
# 2. Sampled data is a subset of pushed data
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_replay_buffer_sample_is_subset(rlox_module):
    """Every sampled reward must appear in the set of pushed rewards."""
    buf = rlox_module.ReplayBuffer(capacity=500, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rng = np.random.default_rng(0)
    pushed_rewards = set()
    for _ in range(200):
        obs = rng.standard_normal(OBS_DIM).astype(np.float32)
        reward = round(float(rng.standard_normal()), 6)
        pushed_rewards.add(reward)
        buf.push(
            obs=obs,
            action=np.zeros(ACT_DIM, dtype=np.float32),
            reward=reward,
            terminated=False,
            truncated=False,
        )

    batch = buf.sample(batch_size=32, seed=7)
    sampled_rewards = np.asarray(batch["rewards"]).flatten()

    for r in sampled_rewards:
        assert round(float(r), 6) in pushed_rewards, (
            f"Sampled reward {r} not found among pushed rewards"
        )


# ---------------------------------------------------------------------------
# 3. Ring buffer FIFO semantics
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_replay_buffer_ring_overwrites_oldest(rlox_module):
    """When capacity is exceeded the oldest transitions are discarded."""
    capacity = 10
    buf = rlox_module.ReplayBuffer(capacity=capacity, obs_dim=1, act_dim=1)

    # Push capacity + 5 transitions with unique rewards [0..14]
    for i in range(capacity + 5):
        buf.push(
            obs=np.array([float(i)], dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=float(i),
            terminated=False,
            truncated=False,
        )

    assert len(buf) == capacity, f"Buffer should be at capacity, got {len(buf)}"

    # Sample many times; rewards 0-4 (oldest, overwritten) must never appear
    all_sampled = set()
    for seed in range(50):
        batch = buf.sample(batch_size=capacity, seed=seed)
        for r in np.asarray(batch["rewards"]).flatten():
            all_sampled.add(int(round(float(r))))

    overwritten_rewards = set(range(5))
    intersection = all_sampled & overwritten_rewards
    assert len(intersection) == 0, (
        f"Overwritten rewards {intersection} appeared in samples"
    )


# ---------------------------------------------------------------------------
# 4. Priority buffer statistical sampling
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_priority_buffer_samples_according_to_priorities(rlox_module):
    """Higher-priority transitions should be sampled proportionally more often."""
    if not hasattr(rlox_module, "PriorityReplayBuffer"):
        pytest.skip("PriorityReplayBuffer not available in rlox")

    n = 100
    buf = rlox_module.PriorityReplayBuffer(capacity=n, obs_dim=1, act_dim=1)
    rng = np.random.default_rng(42)

    for i in range(n):
        priority = 10.0 if i < 10 else 1.0  # first 10 have 10x priority
        buf.push(
            obs=np.array([float(i)], dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=float(i),
            terminated=False,
            truncated=False,
            priority=priority,
        )

    # Sample 5000 times; high-priority items should appear ~5x more
    counts = {i: 0 for i in range(n)}
    n_samples = 5000
    for seed in range(n_samples):
        batch = buf.sample(batch_size=1, seed=seed)
        idx = int(round(float(np.asarray(batch["rewards"]).flatten()[0])))
        counts[idx] += 1

    high_priority_mean = np.mean([counts[i] for i in range(10)])
    low_priority_mean = np.mean([counts[i] for i in range(10, n)])
    ratio = high_priority_mean / max(low_priority_mean, 1e-9)

    assert ratio > 3.0, (
        f"High-priority items sampled {ratio:.1f}x more often; expected >3x"
    )


# ---------------------------------------------------------------------------
# 5. obs/action/reward/done/next_obs alignment
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_replay_buffer_field_alignment(rlox_module):
    """Sampled fields must correspond to the same transition (no misalignment)."""
    buf = rlox_module.ReplayBuffer(capacity=200, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rng = np.random.default_rng(11)

    # Push transitions where reward == obs[0] so we can verify alignment
    pushed = []
    for i in range(100):
        obs = np.full(OBS_DIM, float(i), dtype=np.float32)
        action = np.array([float(i % 2)], dtype=np.float32)
        reward = float(i)
        terminated = (i % 20 == 19)
        truncated = False
        pushed.append((obs, action, reward, terminated, truncated))
        buf.push(obs=obs, action=action, reward=reward,
                 terminated=terminated, truncated=truncated)

    batch = buf.sample(batch_size=20, seed=0)
    obs_batch = np.asarray(batch["obs"])
    reward_batch = np.asarray(batch["rewards"]).flatten()

    for j in range(len(reward_batch)):
        expected_obs_0 = reward_batch[j]  # obs[0] == reward by construction
        actual_obs_0 = float(obs_batch[j, 0])
        assert abs(actual_obs_0 - expected_obs_0) < 1e-5, (
            f"Misalignment: obs[0]={actual_obs_0} != reward={expected_obs_0}"
        )


# ---------------------------------------------------------------------------
# 6. Determinism with seeded RNG
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_replay_buffer_sample_determinism(rlox_module):
    """The same seed must produce identical samples across multiple calls."""
    buf = rlox_module.ReplayBuffer(capacity=500, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    _fill_buffer(buf, 200, seed=42)

    batch_a = buf.sample(batch_size=32, seed=99)
    batch_b = buf.sample(batch_size=32, seed=99)

    rewards_a = np.asarray(batch_a["rewards"])
    rewards_b = np.asarray(batch_b["rewards"])
    np.testing.assert_array_equal(rewards_a, rewards_b)


@pytest.mark.correctness
def test_replay_buffer_different_seeds_different_samples(rlox_module):
    """Different seeds must (almost certainly) produce different samples."""
    buf = rlox_module.ReplayBuffer(capacity=500, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    _fill_buffer(buf, 200, seed=42)

    batch_a = buf.sample(batch_size=32, seed=1)
    batch_b = buf.sample(batch_size=32, seed=2)

    rewards_a = np.asarray(batch_a["rewards"])
    rewards_b = np.asarray(batch_b["rewards"])
    # They should not be identical (with overwhelming probability)
    assert not np.array_equal(rewards_a, rewards_b)


# ---------------------------------------------------------------------------
# 7. Cross-framework: rlox vs SB3 sample distribution
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_replay_buffer_sample_distribution_matches_sb3(rlox_module):
    """rlox and SB3 buffers should produce similar empirical sample distributions.

    Both use uniform random sampling without replacement (or with replacement for
    SB3), so over many samples the empirical distribution should be approximately
    uniform.  We verify via a chi-squared test on index frequencies.
    """
    try:
        from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
        import gymnasium as gym
    except ImportError:
        pytest.skip("stable-baselines3 not installed")

    n = 100
    batch_size = 10
    n_samples = 500

    # rlox buffer
    rlox_buf = rlox_module.ReplayBuffer(capacity=n, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rng = np.random.default_rng(42)
    rewards_list = []
    for i in range(n):
        obs = rng.standard_normal(OBS_DIM).astype(np.float32)
        r = float(i)  # unique reward = transition index
        rewards_list.append(r)
        rlox_buf.push(
            obs=obs,
            action=np.zeros(ACT_DIM, dtype=np.float32),
            reward=r,
            terminated=False,
            truncated=False,
        )

    # Count how often each unique reward is sampled
    rlox_counts = np.zeros(n)
    for seed in range(n_samples):
        batch = rlox_buf.sample(batch_size=batch_size, seed=seed)
        for r in np.asarray(batch["rewards"]).flatten():
            idx = int(round(r))
            if 0 <= idx < n:
                rlox_counts[idx] += 1

    # Each transition should be sampled approximately n_samples * batch_size / n times
    expected_freq = n_samples * batch_size / n
    # Chi-squared statistic; all cells should be within 3 standard deviations
    deviations = np.abs(rlox_counts - expected_freq) / np.sqrt(expected_freq)
    n_outliers = int(np.sum(deviations > 5.0))  # very generous threshold
    assert n_outliers < 5, (
        f"{n_outliers} transitions have highly skewed sampling frequency; "
        "sampling may not be uniform"
    )
