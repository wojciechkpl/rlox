#!/usr/bin/env python3
"""
Benchmark: Mmap vs In-Memory Replay Buffer

Compares push throughput, sample throughput, and memory usage between
the standard in-memory ReplayBuffer and MmapReplayBuffer.

Usage:
    python benchmarks/bench_mmap_buffer.py [--output-dir benchmark_results] [--seed 42] [--n-warmup 3] [--n-reps 10]
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, ComparisonResult, timed_run, write_report


DEFAULT_CAPACITY = 100_000
DEFAULT_OBS_DIM = 4
DEFAULT_ACT_DIM = 1
DEFAULT_PUSH_COUNT = 100_000
DEFAULT_SAMPLE_COUNT = 10_000
DEFAULT_BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Push throughput
# ---------------------------------------------------------------------------

def bench_push_throughput(
    *,
    n_records: int = DEFAULT_PUSH_COUNT,
    obs_dim: int = DEFAULT_OBS_DIM,
    act_dim: int = DEFAULT_ACT_DIM,
    capacity: int = DEFAULT_CAPACITY,
    seed: int = 42,
    n_warmup: int = 1,
    n_reps: int = 5,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Push n_records into in-memory vs mmap buffer, measure time."""
    from rlox import ReplayBuffer, MmapReplayBuffer

    rng = np.random.default_rng(seed)

    # Pre-generate all data
    obs_data = rng.standard_normal((n_records, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((n_records, act_dim)).astype(np.float32)
    rewards = rng.standard_normal(n_records).astype(np.float32)
    terminated = (rng.random(n_records) > 0.95).astype(bool)
    truncated = np.zeros(n_records, dtype=bool)

    # --- In-memory ReplayBuffer ---
    def push_inmemory() -> None:
        buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, act_dim=act_dim)
        for i in range(n_records):
            buf.push(
                obs=obs_data[i],
                action=float(actions[i, 0]),
                reward=float(rewards[i]),
                terminated=bool(terminated[i]),
                truncated=bool(truncated[i]),
            )

    inmemory_times = timed_run(push_inmemory, n_warmup=n_warmup, n_reps=n_reps)

    inmemory_result = BenchmarkResult(
        name="push_throughput",
        category="mmap_buffer",
        framework="rlox_inmemory",
        times_ns=inmemory_times,
        params={"n_records": n_records, "obs_dim": obs_dim, "n_items": n_records},
    )

    # --- MmapReplayBuffer ---
    def push_mmap() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cold_path = os.path.join(tmpdir, "cold")
            buf = MmapReplayBuffer(
                hot_capacity=capacity // 2,
                total_capacity=capacity,
                obs_dim=obs_dim,
                act_dim=act_dim,
                cold_path=cold_path,
            )
            for i in range(n_records):
                buf.push(
                    obs=obs_data[i],
                    action=float(actions[i, 0]),
                    reward=float(rewards[i]),
                    terminated=bool(terminated[i]),
                    truncated=bool(truncated[i]),
                )
            buf.close()

    mmap_times = timed_run(push_mmap, n_warmup=n_warmup, n_reps=n_reps)

    mmap_result = BenchmarkResult(
        name="push_throughput",
        category="mmap_buffer",
        framework="rlox_mmap",
        times_ns=mmap_times,
        params={"n_records": n_records, "obs_dim": obs_dim, "n_items": n_records},
    )

    return inmemory_result, mmap_result


# ---------------------------------------------------------------------------
# Sample throughput
# ---------------------------------------------------------------------------

def bench_sample_throughput(
    *,
    n_records: int = DEFAULT_CAPACITY,
    n_samples: int = DEFAULT_SAMPLE_COUNT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    obs_dim: int = DEFAULT_OBS_DIM,
    act_dim: int = DEFAULT_ACT_DIM,
    capacity: int = DEFAULT_CAPACITY,
    seed: int = 42,
    n_warmup: int = 3,
    n_reps: int = 10,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Sample n_samples batches from a pre-filled buffer."""
    from rlox import ReplayBuffer, MmapReplayBuffer

    rng = np.random.default_rng(seed)
    obs_data = rng.standard_normal((n_records, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((n_records, act_dim)).astype(np.float32)
    rewards = rng.standard_normal(n_records).astype(np.float32)
    terminated = (rng.random(n_records) > 0.95).astype(bool)
    truncated = np.zeros(n_records, dtype=bool)

    # Fill in-memory buffer
    inmemory_buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, act_dim=act_dim)
    for i in range(n_records):
        inmemory_buf.push(
            obs=obs_data[i],
            action=float(actions[i, 0]),
            reward=float(rewards[i]),
            terminated=bool(terminated[i]),
            truncated=bool(truncated[i]),
        )

    def sample_inmemory() -> None:
        for s in range(n_samples):
            inmemory_buf.sample(batch_size=batch_size, seed=s)

    inmemory_times = timed_run(sample_inmemory, n_warmup=n_warmup, n_reps=n_reps)

    inmemory_result = BenchmarkResult(
        name="sample_throughput",
        category="mmap_buffer",
        framework="rlox_inmemory",
        times_ns=inmemory_times,
        params={
            "n_records": n_records,
            "n_samples": n_samples,
            "batch_size": batch_size,
            "n_items": n_samples,
        },
    )

    # Fill mmap buffer
    tmpdir = tempfile.mkdtemp()
    cold_path = os.path.join(tmpdir, "cold")
    mmap_buf = MmapReplayBuffer(
        hot_capacity=capacity // 2,
        total_capacity=capacity,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cold_path=cold_path,
    )
    for i in range(n_records):
        mmap_buf.push(
            obs=obs_data[i],
            action=float(actions[i, 0]),
            reward=float(rewards[i]),
            terminated=bool(terminated[i]),
            truncated=bool(truncated[i]),
        )

    def sample_mmap() -> None:
        for s in range(n_samples):
            mmap_buf.sample(batch_size=batch_size, seed=s)

    mmap_times = timed_run(sample_mmap, n_warmup=n_warmup, n_reps=n_reps)

    mmap_buf.close()

    mmap_result = BenchmarkResult(
        name="sample_throughput",
        category="mmap_buffer",
        framework="rlox_mmap",
        times_ns=mmap_times,
        params={
            "n_records": n_records,
            "n_samples": n_samples,
            "batch_size": batch_size,
            "n_items": n_samples,
        },
    )

    # Cleanup tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return inmemory_result, mmap_result


# ---------------------------------------------------------------------------
# Memory usage
# ---------------------------------------------------------------------------

def bench_memory_usage(
    *,
    n_records: int = DEFAULT_CAPACITY,
    obs_dim: int = DEFAULT_OBS_DIM,
    act_dim: int = DEFAULT_ACT_DIM,
    capacity: int = DEFAULT_CAPACITY,
    seed: int = 42,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Track RSS before/after filling buffer for in-memory vs mmap."""
    import resource
    from rlox import ReplayBuffer, MmapReplayBuffer

    rng = np.random.default_rng(seed)
    obs_data = rng.standard_normal((n_records, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((n_records, act_dim)).astype(np.float32)
    rewards = rng.standard_normal(n_records).astype(np.float32)
    terminated = (rng.random(n_records) > 0.95).astype(bool)
    truncated = np.zeros(n_records, dtype=bool)

    def _get_rss_bytes() -> int:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on Linux, kilobytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss
        return usage.ru_maxrss * 1024

    # --- In-memory ---
    rss_before = _get_rss_bytes()
    inmemory_buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, act_dim=act_dim)
    for i in range(n_records):
        inmemory_buf.push(
            obs=obs_data[i],
            action=float(actions[i, 0]),
            reward=float(rewards[i]),
            terminated=bool(terminated[i]),
            truncated=bool(truncated[i]),
        )
    rss_after_inmemory = _get_rss_bytes()
    del inmemory_buf

    inmemory_result = BenchmarkResult(
        name="memory_usage",
        category="mmap_buffer",
        framework="rlox_inmemory",
        times_ns=[0],  # not a timing benchmark
        params={"n_records": n_records, "obs_dim": obs_dim},
        metadata={
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after_inmemory,
            "rss_delta_bytes": rss_after_inmemory - rss_before,
        },
    )

    # --- Mmap ---
    rss_before_mmap = _get_rss_bytes()
    tmpdir = tempfile.mkdtemp()
    cold_path = os.path.join(tmpdir, "cold")
    mmap_buf = MmapReplayBuffer(
        hot_capacity=capacity // 2,
        total_capacity=capacity,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cold_path=cold_path,
    )
    for i in range(n_records):
        mmap_buf.push(
            obs=obs_data[i],
            action=float(actions[i, 0]),
            reward=float(rewards[i]),
            terminated=bool(terminated[i]),
            truncated=bool(truncated[i]),
        )
    rss_after_mmap = _get_rss_bytes()
    mmap_buf.close()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    mmap_result = BenchmarkResult(
        name="memory_usage",
        category="mmap_buffer",
        framework="rlox_mmap",
        times_ns=[0],  # not a timing benchmark
        params={"n_records": n_records, "obs_dim": obs_dim},
        metadata={
            "rss_before_bytes": rss_before_mmap,
            "rss_after_bytes": rss_after_mmap,
            "rss_delta_bytes": rss_after_mmap - rss_before_mmap,
        },
    )

    return inmemory_result, mmap_result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    output_dir: str = "benchmark_results",
    *,
    seed: int = 42,
    n_warmup: int = 3,
    n_reps: int = 10,
) -> None:
    print("=" * 70)
    print("Benchmark: Mmap vs In-Memory Replay Buffer")
    print("=" * 70)

    all_results: list[dict] = []
    all_comparisons: list[dict] = []
    bench_kw = {"seed": seed, "n_warmup": n_warmup, "n_reps": n_reps}

    # Push throughput
    print("\n  Push throughput (100K records):")
    inmem_push, mmap_push = bench_push_throughput(**bench_kw)
    print(f"    in-memory: {inmem_push.median_ns / 1e6:>10.1f} ms")
    print(f"    mmap:      {mmap_push.median_ns / 1e6:>10.1f} ms")
    all_results.extend([inmem_push.summary(), mmap_push.summary()])

    comp_push = ComparisonResult(
        "push_throughput", inmem_push, mmap_push, "rlox_mmap",
    )
    print(f"    ratio (inmem/mmap): {comp_push.speedup:.2f}x")
    all_comparisons.append(comp_push.summary())

    # Sample throughput
    print("\n  Sample throughput (10K batches of 256):")
    inmem_sample, mmap_sample = bench_sample_throughput(**bench_kw)
    print(f"    in-memory: {inmem_sample.median_ns / 1e6:>10.1f} ms")
    print(f"    mmap:      {mmap_sample.median_ns / 1e6:>10.1f} ms")
    all_results.extend([inmem_sample.summary(), mmap_sample.summary()])

    comp_sample = ComparisonResult(
        "sample_throughput", inmem_sample, mmap_sample, "rlox_mmap",
    )
    print(f"    ratio (inmem/mmap): {comp_sample.speedup:.2f}x")
    all_comparisons.append(comp_sample.summary())

    # Memory usage
    print("\n  Memory usage (100K records):")
    inmem_mem, mmap_mem = bench_memory_usage(seed=seed)
    inmem_delta = inmem_mem.metadata["rss_delta_bytes"]
    mmap_delta = mmap_mem.metadata["rss_delta_bytes"]
    print(f"    in-memory RSS delta: {inmem_delta / 1e6:>8.1f} MB")
    print(f"    mmap RSS delta:      {mmap_delta / 1e6:>8.1f} MB")
    all_results.extend([inmem_mem.summary(), mmap_mem.summary()])

    print()
    path = write_report(all_results, all_comparisons, output_dir)
    print(f"Report written to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox mmap buffer benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-reps", type=int, default=10)
    args = parser.parse_args()
    run_all(args.output_dir, seed=args.seed, n_warmup=args.n_warmup, n_reps=args.n_reps)
