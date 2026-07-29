"""Component performance benchmarks — correctness gated.

IMPORTANT: Each benchmark function first validates correctness against the
reference implementation, then measures timing. Performance results are only
meaningful when correctness is confirmed.

Benchmark coverage
------------------
- GAE at 128, 512, 2048, 8192, 32768 steps (rlox vs numpy vs TorchRL)
- Buffer push 10K transitions (rlox vs SB3 vs TorchRL)
- Buffer sample at batch_size in [32, 64, 256, 1024] (rlox vs SB3 vs TorchRL)
- Env stepping 1-512 envs (rlox vs gymnasium)
- GRPO advantages (rlox vs numpy vs torch)
- Token KL (rlox vs numpy vs torch)

Output: JSON written to results_dir/performance/components.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from utils import (
    bootstrap_ci,
    get_system_info,
    measure_time,
    reference_gae_numpy,
    reference_grpo_numpy,
    reference_token_kl_numpy,
    save_results,
    speedup_ci,
)


# ---------------------------------------------------------------------------
# Correctness guard decorator
# ---------------------------------------------------------------------------


def _assert_gae_correct(rlox_module, rewards, values, dones, last_value):
    """Raise AssertionError if rlox GAE deviates from reference by >1e-5."""
    ref_adv, _ = reference_gae_numpy(rewards, values, dones, last_value)
    rlox_adv, _ = rlox_module.compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
    np.testing.assert_allclose(np.asarray(rlox_adv), ref_adv, rtol=1e-5,
                                err_msg="GAE correctness check failed before benchmark")


def _assert_grpo_correct(rlox_module, rewards):
    ref = reference_grpo_numpy(rewards)
    result = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)
    np.testing.assert_allclose(result, ref, rtol=1e-5,
                                err_msg="GRPO correctness check failed before benchmark")


def _assert_kl_correct(rlox_module, log_p, log_q):
    ref = reference_token_kl_numpy(log_p, log_q)
    result = float(rlox_module.compute_token_kl(log_p, log_q))
    assert abs(result - ref) / (abs(ref) + 1e-10) < 1e-5, (
        f"Token KL correctness check failed: rlox={result}, ref={ref}"
    )


# ---------------------------------------------------------------------------
# GAE benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize("n_steps", [128, 512, 2048, 8192, 32768])
def test_gae_performance(n_steps, rlox_module, results_dir):
    """Benchmark GAE at multiple trajectory lengths after correctness check."""
    rng = np.random.default_rng(42)
    rewards = rng.standard_normal(n_steps).astype(np.float64)
    values = rng.standard_normal(n_steps).astype(np.float64)
    dones = (rng.random(n_steps) < 0.05).astype(np.float64)
    last_value = 0.0

    # --- Correctness guard ---
    _assert_gae_correct(rlox_module, rewards, values, dones, last_value)

    # --- rlox timing ---
    rlox_timing = measure_time(
        lambda: rlox_module.compute_gae(rewards, values, dones, last_value, 0.99, 0.95),
        warmup=10, repeats=100,
    )

    # --- NumPy timing ---
    numpy_timing = measure_time(
        lambda: reference_gae_numpy(rewards, values, dones, last_value),
        warmup=10, repeats=100,
    )

    # --- TorchRL timing (optional) ---
    # Use TorchRL's idiomatic fast path: the *vectorized* GAE kernel on
    # float32 tensors. The scalar `generalized_advantage_estimate` runs a
    # Python-level backward loop and is ~40x slower than this vectorized
    # form on long trajectories; benchmarking against it would overstate
    # rlox's advantage. We also record the scalar/float64 path for
    # reference so the gap is explicit.
    torchrl_timing = None
    torchrl_scalar_timing = None
    try:
        import torch
        from torchrl.objectives.value.functional import (
            generalized_advantage_estimate,
            vec_generalized_advantage_estimate,
        )
        # Idiomatic: vectorized kernel, float32 (PyTorch CPU fast dtype)
        rewards_t = torch.from_numpy(rewards).unsqueeze(-1).float()
        values_t = torch.from_numpy(values).unsqueeze(-1).float()
        nv_t = torch.cat([values_t[1:], torch.zeros(1, 1)])
        dones_t = torch.from_numpy(dones).unsqueeze(-1).bool()
        torchrl_timing = measure_time(
            lambda: vec_generalized_advantage_estimate(
                0.99, 0.95, values_t, nv_t, rewards_t, dones_t, dones_t
            ),
            warmup=10, repeats=100,
        )
        # Reference: the non-idiomatic scalar/float64 path (for context)
        rewards64 = torch.from_numpy(rewards).unsqueeze(-1)
        values64 = torch.from_numpy(values).unsqueeze(-1)
        nv64 = torch.cat([values64[1:], torch.zeros(1, 1, dtype=torch.float64)])
        dones64 = torch.from_numpy(dones).unsqueeze(-1).bool()
        torchrl_scalar_timing = measure_time(
            lambda: generalized_advantage_estimate(
                0.99, 0.95, values64, nv64, rewards64, dones64, dones64
            ),
            warmup=10, repeats=100,
        )
    except ImportError:
        pass

    sp, sp_lo, sp_hi = speedup_ci(rlox_timing["times_ns"], numpy_timing["times_ns"])

    record = {
        "benchmark": f"gae_{n_steps}",
        "n_steps": n_steps,
        "correctness_passed": True,
        "rlox": rlox_timing,
        "numpy": numpy_timing,
        "torchrl": torchrl_timing,  # idiomatic: vectorized GAE, float32
        "torchrl_scalar": torchrl_scalar_timing,  # non-idiomatic ref path
        "speedup_vs_numpy": sp,
        "speedup_ci_95": [sp_lo, sp_hi],
        "significant": sp_lo > 1.0,
    }

    out = results_dir / "performance" / f"gae_{n_steps}.json"
    save_results(record, out)

    print(
        f"\n[GAE n={n_steps}] rlox={rlox_timing['median_ns']/1e3:.1f}us  "
        f"numpy={numpy_timing['median_ns']/1e3:.1f}us  "
        f"speedup={sp:.1f}x [{sp_lo:.1f},{sp_hi:.1f}]"
    )

    # Soft assertion: rlox should not be dramatically slower than numpy
    assert rlox_timing["median_ns"] < numpy_timing["median_ns"] * 50, (
        f"rlox GAE is more than 50x slower than numpy at n_steps={n_steps}"
    )


# ---------------------------------------------------------------------------
# Buffer push benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize("obs_dim", [4, 256])
def test_buffer_push_performance(obs_dim, rlox_module, results_dir):
    """Benchmark push throughput for rlox vs SB3 vs TorchRL after correctness check."""
    n_transitions = 10_000

    # Correctness guard: verify push + len
    table = rlox_module.ExperienceTable(obs_dim=obs_dim, act_dim=1)
    obs = np.zeros(obs_dim, dtype=np.float32)
    for _ in range(10):
        table.push(obs=obs, action=np.zeros(1, dtype=np.float32),
                   reward=1.0, terminated=False, truncated=False)
    assert len(table) == 10, "ExperienceTable push correctness failed"

    # rlox timing
    def rlox_push():
        t = rlox_module.ExperienceTable(obs_dim=obs_dim, act_dim=1)
        for _ in range(n_transitions):
            t.push(obs=obs, action=np.zeros(1, dtype=np.float32),
                   reward=1.0, terminated=False, truncated=False)

    rlox_timing = measure_time(rlox_push, warmup=2, repeats=10)

    # SB3 timing (optional)
    sb3_timing = None
    try:
        from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Discrete(2)
        obs2d = np.zeros((1, obs_dim), dtype=np.float32)
        nobs2d = np.zeros((1, obs_dim), dtype=np.float32)
        act_arr = np.array([[0]])
        rew_arr = np.array([1.0])
        done_arr = np.array([False])

        def sb3_push():
            b = SB3ReplayBuffer(n_transitions + 1000, obs_space, act_space)
            for _ in range(n_transitions):
                b.add(obs2d, nobs2d, act_arr, rew_arr, done_arr, [{}])

        sb3_timing = measure_time(sb3_push, warmup=2, repeats=10)
    except ImportError:
        pass

    record = {
        "benchmark": f"buffer_push_obs{obs_dim}",
        "obs_dim": obs_dim,
        "n_transitions": n_transitions,
        "correctness_passed": True,
        "rlox": rlox_timing,
        "sb3": sb3_timing,
    }
    save_results(record, results_dir / "performance" / f"buffer_push_obs{obs_dim}.json")

    tp = n_transitions / (rlox_timing["median_ns"] / 1e9)
    print(f"\n[BufferPush obs={obs_dim}] rlox={rlox_timing['median_ns']/1e6:.1f}ms  "
          f"({tp:,.0f} trans/s)")


# ---------------------------------------------------------------------------
# Buffer sample benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize("batch_size", [32, 64, 256, 1024])
def test_buffer_sample_performance(batch_size, rlox_module, results_dir):
    """Benchmark sample latency for rlox vs SB3 vs TorchRL."""
    buffer_size = 100_000
    obs_dim = 4

    # Fill rlox buffer
    buf = rlox_module.ReplayBuffer(capacity=buffer_size, obs_dim=obs_dim, act_dim=1)
    obs = np.zeros(obs_dim, dtype=np.float32)
    for _ in range(buffer_size):
        buf.push(obs=obs, action=np.zeros(1, dtype=np.float32),
                 reward=1.0, terminated=False, truncated=False)

    # Correctness guard: sample should return the right batch_size
    batch = buf.sample(batch_size=batch_size, seed=0)
    assert np.asarray(batch["rewards"]).shape[0] == batch_size, (
        "Sample correctness check: wrong batch size returned"
    )

    seed_counter = [0]
    def rlox_sample():
        seed_counter[0] += 1
        buf.sample(batch_size=batch_size, seed=seed_counter[0])

    rlox_timing = measure_time(rlox_sample, warmup=10, repeats=100)

    # SB3 timing (optional)
    sb3_timing = None
    try:
        from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Discrete(2)
        sb3_buf = SB3ReplayBuffer(buffer_size, obs_space, act_space)
        obs2d = np.zeros((1, obs_dim), dtype=np.float32)
        nobs2d = np.zeros((1, obs_dim), dtype=np.float32)
        for _ in range(buffer_size):
            sb3_buf.add(obs2d, nobs2d, np.array([[0]]), np.array([1.0]), np.array([False]), [{}])
        sb3_timing = measure_time(lambda: sb3_buf.sample(batch_size), warmup=10, repeats=100)
    except ImportError:
        pass

    sp = None
    if sb3_timing:
        sp, sp_lo, sp_hi = speedup_ci(rlox_timing["times_ns"], sb3_timing["times_ns"])

    record = {
        "benchmark": f"buffer_sample_b{batch_size}",
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "correctness_passed": True,
        "rlox": rlox_timing,
        "sb3": sb3_timing,
        "speedup_vs_sb3": sp,
    }
    save_results(record, results_dir / "performance" / f"buffer_sample_b{batch_size}.json")

    print(f"\n[BufferSample b={batch_size}] rlox={rlox_timing['median_ns']/1e3:.1f}us  "
          f"(p99={rlox_timing['p99_ns']/1e3:.1f}us)")


# ---------------------------------------------------------------------------
# Env stepping benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize("n_envs", [1, 4, 16, 64, 128, 256, 512])
def test_env_stepping_performance(n_envs, rlox_module, results_dir):
    """Benchmark VecEnv throughput and compare against gymnasium SyncVectorEnv."""
    n_batch_steps = 100

    # Correctness guard: ensure step_all returns correct shape
    env = rlox_module.VecEnv(n=n_envs, seed=42)
    env.reset_all(seed=42)
    res = env.step_all([0] * n_envs)
    assert np.asarray(res["obs"]).shape[0] == n_envs, "VecEnv step_all shape check failed"

    def rlox_step():
        for _ in range(n_batch_steps):
            env.step_all([0] * n_envs)

    rlox_timing = measure_time(rlox_step, warmup=5, repeats=50)

    # Gymnasium SyncVectorEnv (optional)
    gym_timing = None
    try:
        import gymnasium as gym
        from gymnasium.vector import SyncVectorEnv
        gym_env = SyncVectorEnv([lambda: gym.make("CartPole-v1")] * n_envs)
        gym_env.reset(seed=42)
        actions = np.zeros(n_envs, dtype=np.int64)

        def gym_step():
            for _ in range(n_batch_steps):
                gym_env.step(actions)

        gym_timing = measure_time(gym_step, warmup=5, repeats=50)
        gym_env.close()
    except ImportError:
        pass

    sp = None
    if gym_timing:
        sp, sp_lo, sp_hi = speedup_ci(rlox_timing["times_ns"], gym_timing["times_ns"])

    total_steps = n_envs * n_batch_steps
    rlox_sps = total_steps / (rlox_timing["median_ns"] / 1e9)

    record = {
        "benchmark": f"vecenv_{n_envs}",
        "n_envs": n_envs,
        "n_batch_steps": n_batch_steps,
        "correctness_passed": True,
        "rlox": rlox_timing,
        "gymnasium_sync": gym_timing,
        "rlox_steps_per_second": rlox_sps,
        "speedup_vs_gymnasium": sp,
    }
    save_results(record, results_dir / "performance" / f"vecenv_{n_envs}.json")

    print(f"\n[VecEnv n={n_envs}] rlox={rlox_timing['median_ns']/1e6:.2f}ms  "
          f"({rlox_sps:,.0f} steps/s)")


# ---------------------------------------------------------------------------
# GRPO benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize("n_prompts,k", [(16, 4), (64, 8), (256, 16)])
def test_grpo_performance(n_prompts, k, rlox_module, results_dir):
    """Benchmark GRPO advantages computation after correctness check."""
    rng = np.random.default_rng(42)
    groups = [rng.standard_normal(k).astype(np.float64) for _ in range(n_prompts)]

    # Correctness guard
    for g in groups[:3]:
        _assert_grpo_correct(rlox_module, g)

    def rlox_grpo():
        for g in groups:
            rlox_module.compute_group_advantages(g)

    def numpy_grpo():
        for g in groups:
            reference_grpo_numpy(g)

    rlox_timing = measure_time(rlox_grpo, warmup=10, repeats=50)
    numpy_timing = measure_time(numpy_grpo, warmup=10, repeats=50)

    torch_timing = None
    try:
        import torch
        torch_groups = [torch.from_numpy(g) for g in groups]
        def torch_grpo():
            for g in torch_groups:
                m, s = g.mean(), g.std()
                (g - m) / s if s > 1e-8 else torch.zeros_like(g)
        torch_timing = measure_time(torch_grpo, warmup=10, repeats=50)
    except ImportError:
        pass

    sp, sp_lo, sp_hi = speedup_ci(rlox_timing["times_ns"], numpy_timing["times_ns"])

    record = {
        "benchmark": f"grpo_{n_prompts}x{k}",
        "n_prompts": n_prompts, "k": k,
        "correctness_passed": True,
        "rlox": rlox_timing,
        "numpy": numpy_timing,
        "torch": torch_timing,
        "speedup_vs_numpy": sp,
        "speedup_ci_95": [sp_lo, sp_hi],
    }
    save_results(record, results_dir / "performance" / f"grpo_{n_prompts}x{k}.json")

    print(f"\n[GRPO {n_prompts}x{k}] rlox={rlox_timing['median_ns']/1e3:.1f}us  "
          f"numpy={numpy_timing['median_ns']/1e3:.1f}us  speedup={sp:.1f}x")


# ---------------------------------------------------------------------------
# Token KL benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize("seq_len", [128, 512, 2048, 8192])
def test_token_kl_performance(seq_len, rlox_module, results_dir):
    """Benchmark token-level KL divergence after correctness check."""
    rng = np.random.default_rng(42)
    log_p = rng.standard_normal(seq_len).astype(np.float64)
    log_q = rng.standard_normal(seq_len).astype(np.float64)

    # Correctness guard
    _assert_kl_correct(rlox_module, log_p, log_q)

    rlox_timing = measure_time(
        lambda: rlox_module.compute_token_kl(log_p, log_q),
        warmup=10, repeats=100,
    )
    numpy_timing = measure_time(
        lambda: reference_token_kl_numpy(log_p, log_q),
        warmup=10, repeats=100,
    )

    torch_timing = None
    try:
        import torch
        lp_t = torch.from_numpy(log_p)
        lq_t = torch.from_numpy(log_q)
        torch_timing = measure_time(
            lambda: torch.sum(torch.exp(lp_t) * (lp_t - lq_t)).item(),
            warmup=10, repeats=100,
        )
    except ImportError:
        pass

    sp, sp_lo, sp_hi = speedup_ci(rlox_timing["times_ns"], numpy_timing["times_ns"])

    record = {
        "benchmark": f"token_kl_{seq_len}",
        "seq_len": seq_len,
        "correctness_passed": True,
        "rlox": rlox_timing,
        "numpy": numpy_timing,
        "torch": torch_timing,
        "speedup_vs_numpy": sp,
        "speedup_ci_95": [sp_lo, sp_hi],
    }
    save_results(record, results_dir / "performance" / f"token_kl_{seq_len}.json")

    print(f"\n[TokenKL len={seq_len}] rlox={rlox_timing['median_ns']/1e3:.1f}us  "
          f"numpy={numpy_timing['median_ns']/1e3:.1f}us  speedup={sp:.1f}x")
