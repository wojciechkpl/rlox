"""GAE correctness tests — TDD RED phase.

Write these tests BEFORE the implementation is correct. Each test defines
the behavioral contract that rlox's compute_gae must satisfy.

Test categories
---------------
1. Numerical accuracy vs the reference numpy implementation
2. Episode boundary handling (dones in the middle of trajectories)
3. Edge cases (zeros, ones, negatives, extreme values)
4. Output shape verification
5. Determinism (same input -> same output across repeated calls)
6. Cross-framework parity with SB3's GAE
7. Numerical stability (no NaN/Inf for extreme but finite inputs)
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import reference_gae_numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(n: int, seed: int = 42, done_prob: float = 0.05):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(n).astype(np.float64),   # rewards
        rng.standard_normal(n).astype(np.float64),   # values
        (rng.random(n) < done_prob).astype(np.float64),  # dones
        float(rng.standard_normal()),                  # last_value
    )


# ---------------------------------------------------------------------------
# 1. Numerical accuracy vs reference
# ---------------------------------------------------------------------------


@pytest.mark.correctness
@pytest.mark.parametrize("n_steps", [128, 2048, 32768])
def test_gae_matches_reference_numpy(n_steps, rlox_module):
    """rlox GAE must match the reference numpy loop to rtol=1e-6."""
    rewards, values, dones, last_value = _make_trajectory(n_steps, seed=42)

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, last_value)
    rlox_adv, rlox_ret = rlox_module.compute_gae(
        rewards, values, dones, last_value, 0.99, 0.95
    )

    rlox_adv = np.asarray(rlox_adv, dtype=np.float64)
    rlox_ret = np.asarray(rlox_ret, dtype=np.float64)

    np.testing.assert_allclose(
        rlox_adv, ref_adv, rtol=1e-6,
        err_msg=f"advantages mismatch at n_steps={n_steps}",
    )
    np.testing.assert_allclose(
        rlox_ret, ref_ret, rtol=1e-6,
        err_msg=f"returns mismatch at n_steps={n_steps}",
    )


@pytest.mark.correctness
@pytest.mark.parametrize("gamma,lam", [
    (0.99, 0.95),
    (0.95, 0.90),
    (1.00, 1.00),
    (0.50, 0.50),
])
def test_gae_matches_reference_various_hyperparams(gamma, lam, rlox_module):
    """rlox GAE must agree with reference for diverse gamma/lam combinations."""
    rewards, values, dones, last_value = _make_trajectory(256, seed=7)

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, last_value, gamma, lam)
    rlox_adv, rlox_ret = rlox_module.compute_gae(
        rewards, values, dones, last_value, gamma, lam
    )

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(rlox_ret), ref_ret, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. Episode boundary handling
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_gae_episode_boundary_at_midpoint(rlox_module):
    """An episode done in the middle resets the advantage accumulation."""
    n = 20
    rewards = np.ones(n, dtype=np.float64)
    values = np.zeros(n, dtype=np.float64)
    dones = np.zeros(n, dtype=np.float64)
    dones[9] = 1.0  # episode ends at step 9

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, 0.0)
    rlox_adv, rlox_ret = rlox_module.compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-6)
    # Steps 10+ must be independent of steps 0-9 (boundary cuts the GAE sum)
    assert abs(float(rlox_adv[10])) > 0.0


@pytest.mark.correctness
def test_gae_episode_boundary_at_first_step(rlox_module):
    """Done at step 0 should not propagate any advantage from earlier steps."""
    n = 10
    rewards = np.ones(n, dtype=np.float64)
    values = np.zeros(n, dtype=np.float64)
    dones = np.zeros(n, dtype=np.float64)
    dones[0] = 1.0

    ref_adv, _ = reference_gae_numpy(rewards, values, dones, 0.0)
    rlox_adv, _ = rlox_module.compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-6)


@pytest.mark.correctness
def test_gae_all_episodes_terminated(rlox_module):
    """Every step terminates — each advantage equals just the TD error."""
    n = 8
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    values = np.zeros(n, dtype=np.float64)
    dones = np.ones(n, dtype=np.float64)

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, 0.0)
    rlox_adv, rlox_ret = rlox_module.compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-6)
    # When every step terminates, advantage[t] == reward[t] - value[t] (TD-error only)
    np.testing.assert_allclose(np.asarray(rlox_adv), rewards - values, rtol=1e-6)


@pytest.mark.correctness
def test_gae_multiple_episode_boundaries(rlox_module):
    """Multiple boundaries within a single trajectory are all handled correctly."""
    n = 64
    rewards, values, _, last_value = _make_trajectory(n, seed=99)
    # Force several episode boundaries
    dones = np.zeros(n, dtype=np.float64)
    for idx in [10, 25, 40, 55]:
        dones[idx] = 1.0

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, last_value)
    rlox_adv, rlox_ret = rlox_module.compute_gae(rewards, values, dones, last_value, 0.99, 0.95)

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_gae_all_zeros(rlox_module):
    """All-zero inputs should produce all-zero advantages and returns."""
    n = 64
    z = np.zeros(n, dtype=np.float64)
    rlox_adv, rlox_ret = rlox_module.compute_gae(z, z, z, 0.0, 0.99, 0.95)
    np.testing.assert_allclose(np.asarray(rlox_adv), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rlox_ret), 0.0, atol=1e-15)


@pytest.mark.correctness
def test_gae_all_ones(rlox_module):
    """Constant rewards and zero values with no dones."""
    n = 32
    rewards = np.ones(n, dtype=np.float64)
    values = np.zeros(n, dtype=np.float64)
    dones = np.zeros(n, dtype=np.float64)

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, 0.0)
    rlox_adv, rlox_ret = rlox_module.compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-6)


@pytest.mark.correctness
def test_gae_negative_rewards(rlox_module):
    """Negative rewards (e.g. Pendulum) should work identically to positive."""
    n = 128
    rng = np.random.default_rng(13)
    rewards = rng.uniform(-2.0, 0.0, n).astype(np.float64)
    values = rng.uniform(-1.0, 0.0, n).astype(np.float64)
    dones = (rng.random(n) < 0.05).astype(np.float64)

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, -0.5)
    rlox_adv, rlox_ret = rlox_module.compute_gae(rewards, values, dones, -0.5, 0.99, 0.95)

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-6)


@pytest.mark.correctness
def test_gae_large_values_no_overflow(rlox_module):
    """Large but finite rewards should not overflow to Inf."""
    n = 64
    rewards = np.full(n, 1e6, dtype=np.float64)
    values = np.full(n, 1e6, dtype=np.float64)
    dones = np.zeros(n, dtype=np.float64)

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, 1e6)
    rlox_adv, rlox_ret = rlox_module.compute_gae(rewards, values, dones, 1e6, 0.99, 0.95)

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-5)


@pytest.mark.correctness
def test_gae_single_step(rlox_module):
    """Single-element trajectory: advantage equals the TD error."""
    rewards = np.array([1.0])
    values = np.array([0.5])
    dones = np.array([0.0])
    last_value = 0.8
    gamma, lam = 0.99, 0.95

    ref_adv, ref_ret = reference_gae_numpy(rewards, values, dones, last_value, gamma, lam)
    rlox_adv, rlox_ret = rlox_module.compute_gae(rewards, values, dones, last_value, gamma, lam)

    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. Output shapes
# ---------------------------------------------------------------------------


@pytest.mark.correctness
@pytest.mark.parametrize("n_steps", [1, 7, 128, 1000])
def test_gae_output_shapes(n_steps, rlox_module):
    """Advantages and returns must both have shape (n_steps,)."""
    rewards = np.ones(n_steps, dtype=np.float64)
    values = np.zeros(n_steps, dtype=np.float64)
    dones = np.zeros(n_steps, dtype=np.float64)

    adv, ret = rlox_module.compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)

    assert np.asarray(adv).shape == (n_steps,), f"advantages shape wrong: {np.asarray(adv).shape}"
    assert np.asarray(ret).shape == (n_steps,), f"returns shape wrong: {np.asarray(ret).shape}"


@pytest.mark.correctness
def test_gae_returns_equals_advantages_plus_values(rlox_module):
    """Mathematical identity: returns = advantages + values must hold exactly."""
    rewards, values, dones, last_value = _make_trajectory(256)
    adv, ret = rlox_module.compute_gae(rewards, values, dones, last_value, 0.99, 0.95)

    adv_arr = np.asarray(adv, dtype=np.float64)
    ret_arr = np.asarray(ret, dtype=np.float64)

    np.testing.assert_allclose(ret_arr, adv_arr + values, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_gae_deterministic(rlox_module):
    """Identical inputs must produce bit-identical outputs across 10 calls."""
    rewards, values, dones, last_value = _make_trajectory(512)

    first_adv, first_ret = rlox_module.compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
    first_adv = np.asarray(first_adv, dtype=np.float64).copy()
    first_ret = np.asarray(first_ret, dtype=np.float64).copy()

    for _ in range(9):
        adv, ret = rlox_module.compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
        np.testing.assert_array_equal(np.asarray(adv, dtype=np.float64), first_adv)
        np.testing.assert_array_equal(np.asarray(ret, dtype=np.float64), first_ret)


# ---------------------------------------------------------------------------
# 6. Cross-framework validation: rlox vs SB3
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_gae_matches_sb3(rlox_module):
    """rlox GAE must agree with SB3's internal GAE on identical inputs."""
    try:
        from stable_baselines3.common.buffers import RolloutBuffer
        import gymnasium as gym
        import torch
    except ImportError:
        pytest.skip("stable-baselines3 or gymnasium not installed")

    n_steps = 128
    obs_dim = 4
    rng = np.random.default_rng(42)

    rewards = rng.uniform(0.0, 1.0, n_steps).astype(np.float32)
    values = rng.standard_normal(n_steps).astype(np.float32)
    # dones[t] = 1 means step t is the LAST step of an episode
    dones = (rng.random(n_steps) < 0.05).astype(np.float32)
    last_value_np = 0.0

    # rlox reference
    ref_adv, _ = reference_gae_numpy(
        rewards.astype(np.float64), values.astype(np.float64),
        dones.astype(np.float64), last_value_np,
    )

    # rlox implementation
    rlox_adv, _ = rlox_module.compute_gae(
        rewards.astype(np.float64), values.astype(np.float64),
        dones.astype(np.float64), last_value_np, 0.99, 0.95,
    )

    # SB3 RolloutBuffer.compute_returns_and_advantage
    # SB3 uses episode_start[t] = True when step t is the FIRST step of a
    # new episode. This is the shifted version of dones:
    #   episode_start[0] = True (always)
    #   episode_start[t] = dones[t-1]  for t > 0
    episode_starts = np.zeros(n_steps, dtype=np.float32)
    episode_starts[0] = 1.0
    for t in range(1, n_steps):
        episode_starts[t] = dones[t - 1]

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    act_space = gym.spaces.Discrete(2)
    buf = RolloutBuffer(
        buffer_size=n_steps,
        observation_space=obs_space,
        action_space=act_space,
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=1,
    )
    fake_obs = np.zeros((obs_dim,), dtype=np.float32)
    fake_action = np.array([0])
    fake_log_prob = np.array([0.0])
    for t in range(n_steps):
        buf.add(
            obs=fake_obs[np.newaxis, :],
            action=fake_action[np.newaxis, :],
            reward=rewards[t : t + 1],
            episode_start=episode_starts[t : t + 1],
            value=torch.tensor([[values[t]]]),
            log_prob=torch.tensor(fake_log_prob),
        )
    # last_dones = whether the LAST collected step was a terminal
    buf.compute_returns_and_advantage(
        last_values=torch.tensor([[last_value_np]]),
        dones=np.array([dones[-1]]),
    )
    sb3_adv = buf.advantages.flatten().astype(np.float64)

    # rlox and reference must agree exactly
    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-5)
    # SB3 agreement within relaxed tolerance (float32 accumulation vs float64)
    np.testing.assert_allclose(np.asarray(rlox_adv), sb3_adv, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 7. Numerical stability
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_gae_no_nan_inf_for_extreme_inputs(rlox_module):
    """No NaN or Inf for extreme (but representable) float64 inputs."""
    n = 128
    rng = np.random.default_rng(55)
    # Mix of large positive, large negative, and near-zero values
    rewards = rng.choice([-1e15, -1.0, 0.0, 1.0, 1e15], size=n).astype(np.float64)
    values = rng.choice([-1e10, 0.0, 1e10], size=n).astype(np.float64)
    dones = (rng.random(n) < 0.1).astype(np.float64)

    adv, ret = rlox_module.compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)

    adv_arr = np.asarray(adv)
    ret_arr = np.asarray(ret)
    assert not np.any(np.isnan(adv_arr)), "NaN found in advantages"
    assert not np.any(np.isnan(ret_arr)), "NaN found in returns"
    assert not np.any(np.isinf(adv_arr)), "Inf found in advantages"
    assert not np.any(np.isinf(ret_arr)), "Inf found in returns"


@pytest.mark.correctness
def test_gae_no_nan_uniform_rewards(rlox_module):
    """Uniform rewards (zero std) should not produce NaN."""
    n = 64
    rewards = np.ones(n, dtype=np.float64) * 5.0
    values = np.ones(n, dtype=np.float64) * 3.0
    dones = np.zeros(n, dtype=np.float64)

    adv, ret = rlox_module.compute_gae(rewards, values, dones, 3.0, 0.99, 0.95)
    assert not np.any(np.isnan(np.asarray(adv)))
    assert not np.any(np.isnan(np.asarray(ret)))
