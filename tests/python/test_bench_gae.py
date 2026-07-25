"""Tests for GAE computation benchmarks (Phase 3)."""

import numpy as np
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))

from harness import BenchmarkResult, ComparisonResult, timed_run


# ---------------------------------------------------------------------------
# Reference implementation for validation
# ---------------------------------------------------------------------------

def reference_gae_numpy(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """CleanRL/SB3-style GAE in pure numpy (Python loop). Serves as both
    correctness reference and performance baseline."""
    n = len(rewards)
    advantages = np.zeros(n)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = last_value
            next_non_terminal = 1.0 - float(dones[t])
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Precondition: rlox GAE exists and is correct
# ---------------------------------------------------------------------------

class TestGAEPreconditions:
    """Validate rlox GAE produces correct results before benchmarking."""

    def test_compute_gae_importable(self):
        from rlox import compute_gae
        advantages, returns = compute_gae(
            rewards=np.array([1.0]),
            values=np.array([0.0]),
            dones=np.array([True]),
            last_value=0.0,
            gamma=0.99,
            lam=0.95,
        )
        assert advantages.shape == (1,)
        assert returns.shape == (1,)

    def test_gae_matches_reference(self):
        """Rust GAE must match reference numpy implementation to within 1e-6."""
        from rlox import compute_gae

        rng = np.random.default_rng(42)
        n = 2048
        rewards = rng.standard_normal(n)
        values = rng.standard_normal(n)
        dones = (rng.random(n) > 0.95).astype(float)  # ~5% done rate
        last_value = rng.standard_normal()

        ref_adv, ref_ret = reference_gae_numpy(
            rewards, values, dones, last_value, 0.99, 0.95
        )
        rust_adv, rust_ret = compute_gae(
            rewards, values, dones, last_value, 0.99, 0.95
        )

        np.testing.assert_allclose(rust_adv, ref_adv, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(rust_ret, ref_ret, rtol=1e-6, atol=1e-10)

    def test_gae_handles_episode_boundaries(self):
        """GAE resets at episode boundaries (dones=True)."""
        from rlox import compute_gae

        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([0.0, 0.0, 0.0])
        dones = np.array([False, True, False])  # boundary at step 1

        ref_adv, _ = reference_gae_numpy(rewards, values, dones, 0.0, 0.99, 0.95)
        rust_adv, _ = compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)

        np.testing.assert_allclose(rust_adv, ref_adv, rtol=1e-6)


# ---------------------------------------------------------------------------
# Benchmark: GAE computation time (H3)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBenchGAEComputation:
    """GAE computation benchmarks."""

    @pytest.mark.parametrize("n_steps", [128, 512, 2048, 8192, 32768])
    def test_rlox_gae_speed(self, n_steps):
        """Rust GAE should be at least 5x faster than numpy loop."""
        from rlox import compute_gae

        rng = np.random.default_rng(42)
        rewards = rng.standard_normal(n_steps)
        values = rng.standard_normal(n_steps)
        dones = (rng.random(n_steps) > 0.95).astype(float)
        last_value = 0.0

        # rlox (Rust)
        rlox_times = timed_run(
            lambda: compute_gae(rewards, values, dones, last_value, 0.99, 0.95),
            n_warmup=10, n_reps=100,
        )

        # NumPy reference (Python loop)
        numpy_times = timed_run(
            lambda: reference_gae_numpy(rewards, values, dones, last_value, 0.99, 0.95),
            n_warmup=10, n_reps=100,
        )

        rlox_result = BenchmarkResult(
            name=f"gae_{n_steps}", category="gae",
            framework="rlox", times_ns=rlox_times,
            params={"n_steps": n_steps},
        )
        numpy_result = BenchmarkResult(
            name=f"gae_{n_steps}", category="gae",
            framework="numpy_loop", times_ns=numpy_times,
            params={"n_steps": n_steps},
        )
        comp = ComparisonResult(
            benchmark_name=f"gae_{n_steps}",
            rlox=rlox_result, baseline=numpy_result,
            baseline_name="numpy_loop",
        )

        # H3: Must be at least 5x faster
        lo, hi = comp.speedup_ci_95
        assert lo > 5.0, (
            f"GAE not fast enough at {n_steps} steps: "
            f"speedup={comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]"
        )

    def test_rlox_gae_batched_across_envs(self):
        """GAE across 128 envs * 2048 steps."""
        from rlox import compute_gae

        rng = np.random.default_rng(42)
        n_envs = 128
        n_steps = 2048

        # Prepare data for all envs
        all_rewards = rng.standard_normal((n_envs, n_steps))
        all_values = rng.standard_normal((n_envs, n_steps))
        all_dones = (rng.random((n_envs, n_steps)) > 0.95).astype(float)

        def rlox_batched():
            for i in range(n_envs):
                compute_gae(
                    all_rewards[i], all_values[i], all_dones[i],
                    0.0, 0.99, 0.95,
                )

        def numpy_batched():
            for i in range(n_envs):
                reference_gae_numpy(
                    all_rewards[i], all_values[i], all_dones[i],
                    0.0, 0.99, 0.95,
                )

        rlox_times = timed_run(rlox_batched, n_warmup=3, n_reps=20)
        numpy_times = timed_run(numpy_batched, n_warmup=3, n_reps=20)

        rlox_result = BenchmarkResult(
            name="gae_batched_128x2048", category="gae",
            framework="rlox", times_ns=rlox_times,
        )
        numpy_result = BenchmarkResult(
            name="gae_batched_128x2048", category="gae",
            framework="numpy_loop", times_ns=numpy_times,
        )
        comp = ComparisonResult(
            benchmark_name="gae_batched_128x2048",
            rlox=rlox_result, baseline=numpy_result,
            baseline_name="numpy_loop",
        )

        # Batched should still be significantly faster
        lo, _ = comp.speedup_ci_95
        assert lo > 3.0, f"Batched GAE speedup too low: {comp.speedup:.1f}x"


# ---------------------------------------------------------------------------
# Benchmark: GAE numpy baseline profiling
# ---------------------------------------------------------------------------

class TestGAENumpyBaseline:
    """Profile the numpy baseline to establish ground truth timing."""

    @pytest.mark.parametrize("n_steps", [128, 2048, 32768])
    def test_numpy_gae_profiling(self, n_steps):
        """Establish numpy GAE baseline timings (always runs, no rlox needed)."""
        rng = np.random.default_rng(42)
        rewards = rng.standard_normal(n_steps)
        values = rng.standard_normal(n_steps)
        dones = (rng.random(n_steps) > 0.95).astype(float)

        times = timed_run(
            lambda: reference_gae_numpy(rewards, values, dones, 0.0, 0.99, 0.95),
            n_warmup=10, n_reps=50,
        )
        result = BenchmarkResult(
            name=f"gae_numpy_{n_steps}", category="gae",
            framework="numpy_loop", times_ns=times,
            params={"n_steps": n_steps},
        )
        # Just verify it's measurable and the result makes sense
        assert result.median_ns > 0
        # GAE on 128 steps should be under 1ms
        if n_steps <= 128:
            assert result.median_ns < 1_000_000
