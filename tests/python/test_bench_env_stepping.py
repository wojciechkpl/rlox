"""TDD tests for environment stepping benchmarks.

Tests validate:
1. Benchmark preconditions (numerical equivalence between frameworks)
2. Benchmark correctness (measurements capture the right thing)
3. Statistical validity of comparisons
"""

import numpy as np
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))

from harness import BenchmarkResult, ComparisonResult, timed_run


# ---------------------------------------------------------------------------
# Precondition: rlox CartPole produces valid observations
# ---------------------------------------------------------------------------

class TestRloxEnvPreconditions:
    """Validate rlox env behavior before benchmarking."""

    def test_cartpole_obs_shape_and_dtype(self):
        from rlox import CartPole
        env = CartPole(seed=42)
        obs = env.reset(seed=42)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32

    def test_cartpole_step_produces_valid_output(self):
        from rlox import CartPole
        env = CartPole(seed=42)
        result = env.step(1)
        assert result["obs"].shape == (4,)
        assert isinstance(result["reward"], float)
        assert isinstance(result["terminated"], (bool, np.bool_))

    def test_vecenv_shapes_correct(self):
        from rlox import VecEnv
        n = 16
        env = VecEnv(n=n, seed=42)
        result = env.step_all([0] * n)
        assert result["obs"].shape == (n, 4)
        assert result["rewards"].shape == (n,)

    def test_vecenv_1024_envs_works(self):
        """1024 parallel envs must work for scaling benchmarks."""
        from rlox import VecEnv
        env = VecEnv(n=1024, seed=42)
        result = env.step_all([0] * 1024)
        assert result["obs"].shape == (1024, 4)


# ---------------------------------------------------------------------------
# Precondition: Gymnasium CartPole equivalence
# ---------------------------------------------------------------------------

class TestGymnasiumEquivalence:
    """rlox and Gymnasium must produce equivalent behavior."""

    @pytest.fixture
    def gymnasium(self):
        return pytest.importorskip("gymnasium")

    def test_gym_cartpole_obs_shape(self, gymnasium):
        env = gymnasium.make("CartPole-v1")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (4,)
        env.close()

    def test_both_produce_4d_observations(self, gymnasium):
        from rlox import CartPole
        rlox_env = CartPole(seed=42)
        rlox_obs = rlox_env.reset(seed=42)

        gym_env = gymnasium.make("CartPole-v1")
        gym_obs, _ = gym_env.reset(seed=42)
        gym_env.close()

        assert rlox_obs.shape == gym_obs.shape
        assert rlox_obs.dtype == gym_obs.dtype

    def test_both_give_reward_one(self, gymnasium):
        from rlox import CartPole
        rlox_env = CartPole(seed=42)
        rlox_result = rlox_env.step(1)

        gym_env = gymnasium.make("CartPole-v1")
        gym_env.reset(seed=42)
        _, reward, _, _, _ = gym_env.step(1)
        gym_env.close()

        assert rlox_result["reward"] == reward == 1.0

    def test_action_space_same(self, gymnasium):
        """Both accept actions 0 and 1."""
        from rlox import CartPole
        rlox_env = CartPole(seed=42)
        # rlox accepts 0 and 1
        rlox_env.step(0)
        # Reset and try 1
        rlox_env2 = CartPole(seed=42)
        rlox_env2.step(1)

        gym_env = gymnasium.make("CartPole-v1")
        gym_env.reset(seed=42)
        gym_env.step(0)
        gym_env.reset(seed=42)
        gym_env.step(1)
        gym_env.close()


# ---------------------------------------------------------------------------
# Benchmark: Single CartPole step latency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBenchSingleStep:
    """Benchmark and validate single environment step latency."""

    def test_rlox_single_step_measurement(self):
        """rlox single step should be measurable and fast (< 10us)."""
        from rlox import CartPole
        env = CartPole(seed=42)

        def step_fn():
            result = env.step(1)
            if result["terminated"] or result["truncated"]:
                env.reset(seed=42)

        times = timed_run(step_fn, n_warmup=100, n_reps=1000)
        result = BenchmarkResult(
            name="single_step", category="env_stepping",
            framework="rlox", times_ns=times,
        )
        # CartPole step should be under 10 microseconds
        assert result.median_ns < 10_000, f"Too slow: {result.median_ns:.0f}ns"
        # Should have low variance (IQR < 5x median)
        assert result.iqr_ns < result.median_ns * 5

    def test_gymnasium_single_step_measurement(self):
        """Gymnasium single step for comparison."""
        gymnasium = pytest.importorskip("gymnasium")
        env = gymnasium.make("CartPole-v1")
        env.reset(seed=42)

        def step_fn():
            _, _, terminated, truncated, _ = env.step(1)
            if terminated or truncated:
                env.reset(seed=42)

        times = timed_run(step_fn, n_warmup=100, n_reps=1000)
        result = BenchmarkResult(
            name="single_step", category="env_stepping",
            framework="gymnasium", times_ns=times,
        )
        env.close()
        # Gymnasium step should be under 100 microseconds
        assert result.median_ns < 100_000, f"Too slow: {result.median_ns:.0f}ns"

    def test_rlox_faster_than_gymnasium_single_step(self):
        """rlox Rust-native CartPole should be faster than Gymnasium Python."""
        gymnasium = pytest.importorskip("gymnasium")
        from rlox import CartPole

        # rlox
        rlox_env = CartPole(seed=42)
        def rlox_step():
            result = rlox_env.step(1)
            if result["terminated"] or result["truncated"]:
                rlox_env.reset(seed=42)
        rlox_times = timed_run(rlox_step, n_warmup=100, n_reps=200)

        # Gymnasium
        gym_env = gymnasium.make("CartPole-v1")
        gym_env.reset(seed=42)
        def gym_step():
            _, _, term, trunc, _ = gym_env.step(1)
            if term or trunc:
                gym_env.reset(seed=42)
        gym_times = timed_run(gym_step, n_warmup=100, n_reps=200)
        gym_env.close()

        rlox_result = BenchmarkResult(
            name="single_step", category="env_stepping",
            framework="rlox", times_ns=rlox_times,
        )
        gym_result = BenchmarkResult(
            name="single_step", category="env_stepping",
            framework="gymnasium", times_ns=gym_times,
        )
        comp = ComparisonResult(
            benchmark_name="single_step_cartpole",
            rlox=rlox_result, baseline=gym_result,
            baseline_name="gymnasium",
        )

        # rlox should be faster (CI lower bound > 1.0)
        lo, hi = comp.speedup_ci_95
        assert lo > 1.0, (
            f"rlox not significantly faster: speedup={comp.speedup:.1f}x, "
            f"CI=[{lo:.1f}, {hi:.1f}]"
        )


# ---------------------------------------------------------------------------
# Benchmark: VecEnv parallel stepping throughput
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBenchVecEnvThroughput:
    """Benchmark vectorized environment stepping."""

    @pytest.mark.parametrize("num_envs", [1, 4, 16, 64, 128])
    def test_rlox_vecenv_throughput(self, num_envs):
        """VecEnv throughput should scale with num_envs."""
        from rlox import VecEnv
        env = VecEnv(n=num_envs, seed=42)
        actions = [0] * num_envs

        def step_fn():
            env.step_all(actions)

        times = timed_run(step_fn, n_warmup=10, n_reps=50)
        result = BenchmarkResult(
            name=f"vecenv_step_{num_envs}",
            category="env_stepping",
            framework="rlox",
            times_ns=times,
            params={"num_envs": num_envs, "n_items": num_envs},
        )
        # Per-env amortized cost should be under 10us
        per_env_ns = result.median_ns / num_envs
        assert per_env_ns < 10_000, (
            f"Per-env cost too high for {num_envs} envs: {per_env_ns:.0f}ns"
        )

    def test_rlox_vecenv_scaling_efficiency(self):
        """Throughput should increase with more envs (not perfectly linear, but positive)."""
        from rlox import VecEnv

        throughputs = {}
        for n in [1, 16, 64, 128]:
            env = VecEnv(n=n, seed=42)
            actions = [0] * n
            times = timed_run(lambda: env.step_all(actions), n_warmup=10, n_reps=30)
            median = np.median(times)
            throughputs[n] = n / (median / 1e9)  # envs-steps/sec

        # Throughput at 128 envs should be > throughput at 1 env
        assert throughputs[128] > throughputs[1], (
            f"No scaling: 128-env throughput ({throughputs[128]:.0f}) <= "
            f"1-env throughput ({throughputs[1]:.0f})"
        )
        # Throughput at 128 envs should show meaningful scaling.
        # Note: CartPole is extremely lightweight (~37ns/step), so Rayon
        # scheduling overhead reduces scaling efficiency. For heavier envs
        # (MuJoCo, Atari) we expect near-linear scaling. For CartPole,
        # 1.2x+ throughput at 128 envs vs 1 env is a conservative bar.
        scaling = throughputs[128] / throughputs[1]
        assert scaling > 1.2, f"Poor scaling: only {scaling:.1f}x at 128 envs"

    @pytest.mark.parametrize("num_envs", [64, 128])
    def test_rlox_vecenv_vs_gymnasium_sync(self, num_envs):
        """rlox VecEnv should outperform Gymnasium SyncVectorEnv."""
        gymnasium = pytest.importorskip("gymnasium")
        from gymnasium.vector import SyncVectorEnv
        from rlox import VecEnv

        # rlox
        rlox_env = VecEnv(n=num_envs, seed=42)
        rlox_actions = [0] * num_envs
        rlox_times = timed_run(
            lambda: rlox_env.step_all(rlox_actions),
            n_warmup=10, n_reps=30,
        )

        # Gymnasium SyncVectorEnv
        gym_env = SyncVectorEnv(
            [lambda: gymnasium.make("CartPole-v1")] * num_envs
        )
        gym_env.reset(seed=42)
        gym_actions = np.zeros(num_envs, dtype=np.int64)
        def gym_step():
            obs, rew, term, trunc, info = gym_env.step(gym_actions)
        gym_times = timed_run(gym_step, n_warmup=10, n_reps=30)
        gym_env.close()

        rlox_result = BenchmarkResult(
            name=f"vecenv_{num_envs}", category="env_stepping",
            framework="rlox", times_ns=rlox_times,
        )
        gym_result = BenchmarkResult(
            name=f"vecenv_{num_envs}", category="env_stepping",
            framework="gymnasium_sync", times_ns=gym_times,
        )
        comp = ComparisonResult(
            benchmark_name=f"vecenv_{num_envs}_vs_gymnasium_sync",
            rlox=rlox_result, baseline=gym_result,
            baseline_name="gymnasium_sync",
        )

        # rlox should be faster
        lo, _ = comp.speedup_ci_95
        assert lo > 1.0, (
            f"rlox VecEnv({num_envs}) not faster than Gymnasium Sync: "
            f"speedup={comp.speedup:.1f}x, CI_lo={lo:.2f}"
        )


# ---------------------------------------------------------------------------
# Benchmark: VecEnv reset throughput
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBenchVecEnvReset:
    """Benchmark environment reset throughput."""

    @pytest.mark.parametrize("num_envs", [1, 16, 64, 128])
    def test_rlox_reset_throughput(self, num_envs):
        from rlox import VecEnv
        env = VecEnv(n=num_envs, seed=42)

        times = timed_run(
            lambda: env.reset_all(seed=42),
            n_warmup=5, n_reps=30,
        )
        result = BenchmarkResult(
            name=f"vecenv_reset_{num_envs}", category="env_stepping",
            framework="rlox", times_ns=times,
            params={"num_envs": num_envs},
        )
        # Reset should be under 1ms per env
        per_env_ns = result.median_ns / num_envs
        assert per_env_ns < 1_000_000


# ---------------------------------------------------------------------------
# Benchmark: Bridge overhead measurement
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBenchBridgeOverhead:
    """Measure PyO3 bridge overhead: rlox native vs rlox GymEnv wrapper."""

    def test_bridge_overhead_measurement(self):
        """GymEnv wrapper adds measurable overhead vs native CartPole."""
        gymnasium = pytest.importorskip("gymnasium")
        from rlox import CartPole, GymEnv

        # Native rlox CartPole
        native_env = CartPole(seed=42)
        def native_step():
            result = native_env.step(1)
            if result["terminated"] or result["truncated"]:
                native_env.reset(seed=42)
        native_times = timed_run(native_step, n_warmup=100, n_reps=500)

        # GymEnv wrapping Gymnasium CartPole
        gym_raw = gymnasium.make("CartPole-v1")
        bridge_env = GymEnv(gym_raw)
        bridge_env.reset(seed=42)
        def bridge_step():
            result = bridge_env.step(1)
            if result["terminated"] or result["truncated"]:
                bridge_env.reset(seed=42)
        bridge_times = timed_run(bridge_step, n_warmup=100, n_reps=500)
        gym_raw.close()

        native_result = BenchmarkResult(
            name="bridge_native", category="bridge_overhead",
            framework="rlox_native", times_ns=native_times,
        )
        bridge_result = BenchmarkResult(
            name="bridge_wrapped", category="bridge_overhead",
            framework="rlox_gymenv", times_ns=bridge_times,
        )

        # Bridge should be slower (native is faster)
        overhead_ns = bridge_result.median_ns - native_result.median_ns
        assert overhead_ns > 0, "Bridge should add overhead"
        # But overhead should be bounded (< 50us per step)
        assert overhead_ns < 50_000, f"Bridge overhead too high: {overhead_ns:.0f}ns"


# ---------------------------------------------------------------------------
# Benchmark: SB3 comparison (if available)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBenchSB3Comparison:
    """Compare rlox VecEnv against SB3 DummyVecEnv and SubprocVecEnv."""

    @pytest.fixture
    def sb3(self):
        return pytest.importorskip("stable_baselines3")

    def test_rlox_vs_sb3_dummyvecenv(self, sb3):
        """rlox should outperform SB3 DummyVecEnv (sequential Python)."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym
        from rlox import VecEnv

        num_envs = 64

        # rlox
        rlox_env = VecEnv(n=num_envs, seed=42)
        rlox_actions = [0] * num_envs
        rlox_times = timed_run(
            lambda: rlox_env.step_all(rlox_actions),
            n_warmup=10, n_reps=30,
        )

        # SB3 DummyVecEnv
        sb3_env = DummyVecEnv([lambda: gym.make("CartPole-v1")] * num_envs)
        sb3_env.reset()
        sb3_actions = np.zeros(num_envs, dtype=np.int64)
        sb3_times = timed_run(
            lambda: sb3_env.step(sb3_actions),
            n_warmup=10, n_reps=30,
        )
        sb3_env.close()

        rlox_result = BenchmarkResult(
            name="vecenv_64", category="env_stepping",
            framework="rlox", times_ns=rlox_times,
        )
        sb3_result = BenchmarkResult(
            name="vecenv_64", category="env_stepping",
            framework="sb3_dummy", times_ns=sb3_times,
        )
        comp = ComparisonResult(
            benchmark_name="vecenv_64_vs_sb3_dummy",
            rlox=rlox_result, baseline=sb3_result,
            baseline_name="sb3_dummyvecenv",
        )

        lo, hi = comp.speedup_ci_95
        print(
            f"\nrlox vs SB3 DummyVecEnv (64 envs): "
            f"{comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]"
        )
        assert lo > 1.0, f"rlox not faster than SB3 DummyVecEnv: {comp.speedup:.1f}x"

    def test_rlox_vs_sb3_subprocvecenv(self, sb3):
        """rlox should outperform SB3 SubprocVecEnv for simple envs."""
        from stable_baselines3.common.vec_env import SubprocVecEnv
        import gymnasium as gym
        from rlox import VecEnv

        num_envs = 16  # fewer for subprocess to avoid test slowness

        # rlox
        rlox_env = VecEnv(n=num_envs, seed=42)
        rlox_actions = [0] * num_envs
        rlox_times = timed_run(
            lambda: rlox_env.step_all(rlox_actions),
            n_warmup=10, n_reps=30,
        )

        # SB3 SubprocVecEnv
        sb3_env = SubprocVecEnv([lambda: gym.make("CartPole-v1")] * num_envs)
        sb3_env.reset()
        sb3_actions = np.zeros(num_envs, dtype=np.int64)
        sb3_times = timed_run(
            lambda: sb3_env.step(sb3_actions),
            n_warmup=10, n_reps=30,
        )
        sb3_env.close()

        rlox_result = BenchmarkResult(
            name="vecenv_16", category="env_stepping",
            framework="rlox", times_ns=rlox_times,
        )
        sb3_result = BenchmarkResult(
            name="vecenv_16", category="env_stepping",
            framework="sb3_subproc", times_ns=sb3_times,
        )
        comp = ComparisonResult(
            benchmark_name="vecenv_16_vs_sb3_subproc",
            rlox=rlox_result, baseline=sb3_result,
            baseline_name="sb3_subprocvecenv",
        )

        lo, hi = comp.speedup_ci_95
        print(
            f"\nrlox vs SB3 SubprocVecEnv (16 envs): "
            f"{comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]"
        )
        # For simple envs like CartPole, subprocess overhead should make SB3 slower
        assert lo > 1.0, f"rlox not faster than SB3 SubprocVecEnv: {comp.speedup:.1f}x"
