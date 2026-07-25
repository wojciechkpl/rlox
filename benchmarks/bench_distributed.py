#!/usr/bin/env python3
"""
Benchmark: Distributed Components

Measures overhead of RemoteEnvPool (mock backend), Pipeline async
collector/learner throughput, and multi-env SPS scaling.

Usage:
    python benchmarks/bench_distributed.py [--output-dir benchmark_results] [--seed 42] [--n-warmup 5] [--n-reps 20]
"""

import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, timed_run, write_report


# ---------------------------------------------------------------------------
# Helper: mock backend for RemoteEnvPool
# ---------------------------------------------------------------------------

def _make_mock_pool(n_envs: int = 8, obs_dim: int = 4) -> "rlox.RemoteEnvPool":
    """Create a RemoteEnvPool with injected mock step/reset to avoid gRPC."""
    import rlox

    pool = rlox.RemoteEnvPool(addresses=["mock:50051"])
    # Bypass connect() -- inject mock implementations directly
    pool._connected = True
    pool._num_envs = n_envs
    pool._obs_dim = obs_dim

    rng = np.random.default_rng(0)

    def mock_reset(seed: int | None = None) -> np.ndarray:
        return rng.standard_normal((n_envs, obs_dim)).astype(np.float32)

    def mock_step(actions: np.ndarray | list) -> dict[str, object]:
        return {
            "obs": rng.standard_normal((n_envs, obs_dim)).astype(np.float32),
            "rewards": rng.standard_normal(n_envs).astype(np.float64),
            "terminated": np.zeros(n_envs, dtype=np.uint8),
            "truncated": np.zeros(n_envs, dtype=np.uint8),
            "terminal_obs": [None] * n_envs,
        }

    pool._reset_impl = mock_reset
    pool._step_impl = mock_step
    return pool


# ---------------------------------------------------------------------------
# bench_remote_env_pool_mock
# ---------------------------------------------------------------------------

def bench_remote_env_pool_mock(
    *,
    n_envs: int = 8,
    obs_dim: int = 4,
    n_warmup: int = 5,
    n_reps: int = 30,
) -> BenchmarkResult:
    """Measure step/reset overhead of RemoteEnvPool with mock backend."""
    pool = _make_mock_pool(n_envs=n_envs, obs_dim=obs_dim)

    actions = np.zeros(n_envs, dtype=np.int64)

    times = timed_run(
        lambda: pool.step_all(actions),
        n_warmup=n_warmup,
        n_reps=n_reps,
    )
    pool.close()

    return BenchmarkResult(
        name="remote_env_pool_mock",
        category="distributed",
        framework="rlox",
        times_ns=times,
        params={"n_envs": n_envs, "obs_dim": obs_dim},
    )


# ---------------------------------------------------------------------------
# bench_pipeline_throughput
# ---------------------------------------------------------------------------

def bench_pipeline_throughput(
    *,
    n_envs: int = 4,
    n_steps: int = 32,
    n_batches: int = 5,
    timeout: float = 30.0,
) -> BenchmarkResult:
    """Measure Pipeline collector/learner throughput on CartPole."""
    from rlox.distributed.pipeline import Pipeline

    pipe = Pipeline(
        env_id="CartPole-v1",
        n_envs=n_envs,
        n_steps=n_steps,
        channel_capacity=4,
        seed=0,
    )

    times: list[float] = []
    total_steps = n_envs * n_steps

    try:
        # Warmup: consume first batch (collector thread startup)
        warmup = pipe.next_batch(timeout=timeout)
        if warmup is None:
            pipe.close()
            return BenchmarkResult(
                name="pipeline_throughput",
                category="distributed",
                framework="rlox",
                times_ns=[],
                params={"n_envs": n_envs, "n_steps": n_steps},
                metadata={"error": "warmup timed out"},
            )

        for _ in range(n_batches):
            start = time.perf_counter_ns()
            batch = pipe.next_batch(timeout=timeout)
            elapsed = time.perf_counter_ns() - start
            if batch is not None:
                times.append(float(elapsed))
    finally:
        pipe.close()

    sps = 0.0
    if times:
        median_s = float(np.median(times)) / 1e9
        if median_s > 0:
            sps = total_steps / median_s

    return BenchmarkResult(
        name="pipeline_throughput",
        category="distributed",
        framework="rlox",
        times_ns=times,
        params={"n_envs": n_envs, "n_steps": n_steps, "n_items": total_steps},
        metadata={"sps": sps},
    )


# ---------------------------------------------------------------------------
# bench_multi_env_scaling
# ---------------------------------------------------------------------------

def bench_multi_env_scaling(
    *,
    env_counts: list[int] | None = None,
    n_steps: int = 128,
    seed: int = 42,
    n_warmup: int = 2,
    n_reps: int = 10,
) -> list[BenchmarkResult]:
    """Measure SPS scaling from 1 to 32 envs for PPO on CartPole."""
    import rlox

    if env_counts is None:
        env_counts = [1, 2, 4, 8, 16, 32]

    results: list[BenchmarkResult] = []

    for n_envs in env_counts:
        env = rlox.VecEnv(n=n_envs, seed=seed, env_id="CartPole-v1")
        obs = env.reset_all()

        def step_loop() -> None:
            nonlocal obs
            for _ in range(n_steps):
                actions = np.random.randint(0, 2, size=n_envs).tolist()
                result = env.step_all(actions)
                obs = result["obs"]

        times = timed_run(step_loop, n_warmup=n_warmup, n_reps=n_reps)

        total_steps = n_envs * n_steps
        median_s = float(np.median(times)) / 1e9
        sps = total_steps / median_s if median_s > 0 else 0.0

        results.append(BenchmarkResult(
            name=f"multi_env_scaling_{n_envs}",
            category="distributed",
            framework="rlox",
            times_ns=times,
            params={"n_envs": n_envs, "n_steps": n_steps, "n_items": total_steps},
            metadata={"sps": sps},
        ))

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    output_dir: str = "benchmark_results",
    *,
    seed: int = 42,
    n_warmup: int = 5,
    n_reps: int = 20,
) -> None:
    print("=" * 70)
    print("Benchmark: Distributed Components")
    print("=" * 70)

    all_results: list[dict] = []

    # RemoteEnvPool mock
    print("\n  RemoteEnvPool (mock backend):")
    res = bench_remote_env_pool_mock(n_warmup=n_warmup, n_reps=n_reps)
    print(f"    step overhead: {res.median_ns / 1e3:>10.1f} us")
    all_results.append(res.summary())

    # Pipeline throughput
    print("\n  Pipeline collector/learner throughput:")
    res = bench_pipeline_throughput()
    if res.times_ns:
        print(f"    batch latency: {res.median_ns / 1e3:>10.1f} us")
        print(f"    SPS:           {res.metadata.get('sps', 0):.0f}")
    else:
        print("    [skip] Pipeline timed out")
    all_results.append(res.summary())

    # Multi-env scaling
    print("\n  Multi-env SPS scaling:")
    scaling_results = bench_multi_env_scaling(seed=seed, n_warmup=n_warmup, n_reps=n_reps)
    for sr in scaling_results:
        print(f"    {sr.params['n_envs']:>3d} envs: {sr.metadata['sps']:>10.0f} SPS")
        all_results.append(sr.summary())

    print()
    path = write_report(all_results, [], output_dir)
    print(f"Report written to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox distributed benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-reps", type=int, default=20)
    args = parser.parse_args()
    run_all(args.output_dir, seed=args.seed, n_warmup=args.n_warmup, n_reps=args.n_reps)
