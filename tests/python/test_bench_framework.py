import pytest
"""TDD tests for the benchmark framework itself.

These validate that the benchmarking harness works correctly before
trusting any performance numbers.
"""

import numpy as np
import sys
import os

# Add benchmarks dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))

from harness import BenchmarkResult, ComparisonResult, timed_run, system_info


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBenchmarkResult:
    def test_empty_result(self):
        r = BenchmarkResult(name="test", category="unit", framework="rlox")
        assert r.median_ns == 0.0
        assert r.throughput is None
        assert r.summary()["n_samples"] == 0

    def test_median_computation(self):
        r = BenchmarkResult(
            name="test", category="unit", framework="rlox",
            times_ns=[100.0, 200.0, 300.0, 400.0, 500.0],
        )
        assert r.median_ns == 300.0
        assert r.p25_ns == 200.0
        assert r.p75_ns == 400.0
        assert r.iqr_ns == 200.0
        assert r.min_ns == 100.0
        assert r.max_ns == 500.0

    def test_mean_computation(self):
        r = BenchmarkResult(
            name="test", category="unit", framework="rlox",
            times_ns=[100.0, 200.0, 300.0],
        )
        assert abs(r.mean_ns - 200.0) < 1e-6

    def test_p99(self):
        times = list(range(1, 101))  # 1..100
        r = BenchmarkResult(
            name="test", category="unit", framework="rlox",
            times_ns=[float(t) for t in times],
        )
        assert r.p99_ns >= 99.0

    def test_throughput_computation(self):
        r = BenchmarkResult(
            name="test", category="unit", framework="rlox",
            times_ns=[1_000_000.0],  # 1ms
            params={"n_items": 1000},
        )
        # 1000 items in 1ms = 1M items/sec
        assert abs(r.throughput - 1_000_000.0) < 1.0

    def test_summary_dict(self):
        r = BenchmarkResult(
            name="test_bench", category="env", framework="rlox",
            times_ns=[100.0, 200.0, 300.0],
            params={"num_envs": 4},
        )
        s = r.summary()
        assert s["name"] == "test_bench"
        assert s["category"] == "env"
        assert s["framework"] == "rlox"
        assert s["n_samples"] == 3
        assert "median_ns" in s


# ---------------------------------------------------------------------------
# ComparisonResult tests
# ---------------------------------------------------------------------------

class TestComparisonResult:
    def test_speedup_computation(self):
        rlox = BenchmarkResult(
            name="test", category="env", framework="rlox",
            times_ns=[100.0] * 30,
        )
        baseline = BenchmarkResult(
            name="test", category="env", framework="sb3",
            times_ns=[500.0] * 30,
        )
        comp = ComparisonResult(
            benchmark_name="test",
            rlox=rlox, baseline=baseline, baseline_name="sb3",
        )
        assert abs(comp.speedup - 5.0) < 0.01

    def test_speedup_ci_with_clear_winner(self):
        rng = np.random.default_rng(42)
        rlox = BenchmarkResult(
            name="test", category="env", framework="rlox",
            times_ns=(rng.normal(100, 10, 30)).tolist(),
        )
        baseline = BenchmarkResult(
            name="test", category="env", framework="sb3",
            times_ns=(rng.normal(500, 50, 30)).tolist(),
        )
        comp = ComparisonResult(
            benchmark_name="test",
            rlox=rlox, baseline=baseline, baseline_name="sb3",
        )
        lo, hi = comp.speedup_ci_95
        assert lo > 1.0, f"CI lower bound should be > 1.0, got {lo}"
        assert hi > lo

    def test_speedup_ci_with_no_difference(self):
        rng = np.random.default_rng(42)
        times = rng.normal(100, 10, 30).tolist()
        rlox = BenchmarkResult(
            name="test", category="env", framework="rlox",
            times_ns=times[:],
        )
        baseline = BenchmarkResult(
            name="test", category="env", framework="sb3",
            times_ns=times[:],
        )
        comp = ComparisonResult(
            benchmark_name="test",
            rlox=rlox, baseline=baseline, baseline_name="sb3",
        )
        lo, hi = comp.speedup_ci_95
        # CI should straddle 1.0 (no significant difference)
        assert lo < 1.2 and hi > 0.8

    def test_summary_significant_flag(self):
        rlox = BenchmarkResult(
            name="test", category="env", framework="rlox",
            times_ns=[100.0] * 30,
        )
        baseline = BenchmarkResult(
            name="test", category="env", framework="sb3",
            times_ns=[500.0] * 30,
        )
        comp = ComparisonResult(
            benchmark_name="test",
            rlox=rlox, baseline=baseline, baseline_name="sb3",
        )
        s = comp.summary()
        assert s["significant"] is True
        assert s["speedup_ci_95_lo"] > 1.0


# ---------------------------------------------------------------------------
# timed_run tests
# ---------------------------------------------------------------------------

class TestTimedRun:
    def test_returns_correct_count(self):
        times = timed_run(lambda: None, n_warmup=2, n_reps=10)
        assert len(times) == 10

    def test_warmup_not_included(self):
        call_count = [0]
        def fn():
            call_count[0] += 1
        timed_run(fn, n_warmup=5, n_reps=10)
        assert call_count[0] == 15  # 5 warmup + 10 measured

    def test_times_are_positive(self):
        times = timed_run(lambda: sum(range(1000)), n_warmup=2, n_reps=10)
        assert all(t > 0 for t in times)

    def test_times_are_reasonable_for_noop(self):
        times = timed_run(lambda: None, n_warmup=2, n_reps=30)
        median = np.median(times)
        # A no-op should be under 1ms (1_000_000 ns)
        assert median < 1_000_000


# ---------------------------------------------------------------------------
# system_info tests
# ---------------------------------------------------------------------------

class TestSystemInfo:
    def test_has_required_fields(self):
        info = system_info()
        assert "platform" in info
        assert "python_version" in info
        assert "cpu_count" in info
        assert "numpy_version" in info
        assert isinstance(info["cpu_count"], int)
        assert info["cpu_count"] > 0
