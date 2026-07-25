#!/usr/bin/env python3
"""
Benchmark: Buffer Operations (Push Throughput & Sample Latency)

Compares rlox vs TorchRL vs Stable-Baselines3 replay/experience buffers.

Usage:
    python benchmarks/bench_buffer_ops.py [--output-dir benchmark_results]
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, ComparisonResult, timed_run, write_report

# ---------------------------------------------------------------------------
# Benchmark repetition defaults
# ---------------------------------------------------------------------------
DEFAULT_N_WARMUP_PUSH = 2
DEFAULT_N_REPS_PUSH = 10
DEFAULT_N_WARMUP_SAMPLE = 10
DEFAULT_N_REPS_SAMPLE = 100


# ---------------------------------------------------------------------------
# rlox buffer benchmarks
# ---------------------------------------------------------------------------

def bench_rlox_push(obs_dim: int, n_transitions: int = 10_000, *, n_warmup: int = DEFAULT_N_WARMUP_PUSH, n_reps: int = DEFAULT_N_REPS_PUSH) -> BenchmarkResult:
    from rlox import ExperienceTable

    table = ExperienceTable(obs_dim=obs_dim, act_dim=1)
    obs = np.zeros(obs_dim, dtype=np.float32)

    def push_batch():
        for _ in range(n_transitions):
            table.push(obs=obs, action=np.array([0.0], dtype=np.float32), reward=1.0,
                       terminated=False, truncated=False)

    times = timed_run(push_batch, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"push_obs{obs_dim}", category="buffer_ops",
        framework="rlox", times_ns=times,
        params={"obs_dim": obs_dim, "n_items": n_transitions},
    )


def bench_rlox_sample(batch_size: int, buffer_size: int = 100_000, *, n_warmup: int = DEFAULT_N_WARMUP_SAMPLE, n_reps: int = DEFAULT_N_REPS_SAMPLE) -> BenchmarkResult:
    from rlox import ReplayBuffer

    buf = ReplayBuffer(capacity=buffer_size, obs_dim=4, act_dim=1)
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(buffer_size):
        buf.push(obs=obs, action=np.array([0.0], dtype=np.float32), reward=1.0,
                 terminated=False, truncated=False)

    seed_counter = [0]

    def sample():
        seed_counter[0] += 1
        buf.sample(batch_size=batch_size, seed=seed_counter[0])

    times = timed_run(sample, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"sample_b{batch_size}", category="buffer_ops",
        framework="rlox", times_ns=times,
        params={"batch_size": batch_size, "buffer_size": buffer_size},
    )


# ---------------------------------------------------------------------------
# TorchRL buffer benchmarks
# ---------------------------------------------------------------------------

def bench_torchrl_push(obs_dim: int, n_transitions: int = 10_000, *, n_warmup: int = DEFAULT_N_WARMUP_PUSH, n_reps: int = DEFAULT_N_REPS_PUSH) -> BenchmarkResult | None:
    try:
        import torch
        from tensordict import TensorDict
        from torchrl.data import ReplayBuffer, LazyTensorStorage
    except ImportError:
        print(f"  [skip] torchrl not installed")
        return None

    rb = ReplayBuffer(storage=LazyTensorStorage(max_size=n_transitions + 1000))

    obs = torch.zeros(obs_dim, dtype=torch.float32)
    td = TensorDict({
        "obs": obs,
        "action": torch.tensor(0.0),
        "reward": torch.tensor(1.0),
        "terminated": torch.tensor(False),
        "truncated": torch.tensor(False),
    })

    def push_batch():
        for _ in range(n_transitions):
            rb.add(td)

    times = timed_run(push_batch, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"push_obs{obs_dim}", category="buffer_ops",
        framework="torchrl", times_ns=times,
        params={"obs_dim": obs_dim, "n_items": n_transitions},
    )


def bench_torchrl_sample(batch_size: int, buffer_size: int = 100_000, *, n_warmup: int = DEFAULT_N_WARMUP_SAMPLE, n_reps: int = DEFAULT_N_REPS_SAMPLE) -> BenchmarkResult | None:
    try:
        import torch
        from tensordict import TensorDict
        from torchrl.data import ReplayBuffer, LazyTensorStorage
    except ImportError:
        return None

    rb = ReplayBuffer(storage=LazyTensorStorage(max_size=buffer_size), batch_size=batch_size)

    obs = torch.zeros(4, dtype=torch.float32)
    td = TensorDict({
        "obs": obs,
        "action": torch.tensor(0.0),
        "reward": torch.tensor(1.0),
        "terminated": torch.tensor(False),
        "truncated": torch.tensor(False),
    })
    for _ in range(buffer_size):
        rb.add(td)

    def sample():
        rb.sample()

    times = timed_run(sample, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"sample_b{batch_size}", category="buffer_ops",
        framework="torchrl", times_ns=times,
        params={"batch_size": batch_size, "buffer_size": buffer_size},
    )


# ---------------------------------------------------------------------------
# SB3 buffer benchmarks
# ---------------------------------------------------------------------------

def bench_sb3_push(obs_dim: int, n_transitions: int = 10_000, *, n_warmup: int = DEFAULT_N_WARMUP_PUSH, n_reps: int = DEFAULT_N_REPS_PUSH) -> BenchmarkResult | None:
    try:
        from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
        import gymnasium as gym
    except ImportError:
        print(f"  [skip] stable-baselines3 not installed")
        return None

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)
    buf = SB3ReplayBuffer(
        buffer_size=n_transitions + 1000,
        observation_space=obs_space,
        action_space=act_space,
    )

    obs = np.zeros((1, obs_dim), dtype=np.float32)
    next_obs = np.zeros((1, obs_dim), dtype=np.float32)
    action = np.array([[0]])
    reward = np.array([1.0])
    done = np.array([False])
    infos = [{}]

    def push_batch():
        for _ in range(n_transitions):
            buf.add(obs, next_obs, action, reward, done, infos)

    times = timed_run(push_batch, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"push_obs{obs_dim}", category="buffer_ops",
        framework="sb3", times_ns=times,
        params={"obs_dim": obs_dim, "n_items": n_transitions},
    )


def bench_sb3_sample(batch_size: int, buffer_size: int = 100_000, *, n_warmup: int = DEFAULT_N_WARMUP_SAMPLE, n_reps: int = DEFAULT_N_REPS_SAMPLE) -> BenchmarkResult | None:
    try:
        from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
        import gymnasium as gym
    except ImportError:
        return None

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)
    buf = SB3ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
    )

    obs = np.zeros((1, 4), dtype=np.float32)
    next_obs = np.zeros((1, 4), dtype=np.float32)
    action = np.array([[0]])
    reward = np.array([1.0])
    done = np.array([False])
    infos = [{}]

    for _ in range(buffer_size):
        buf.add(obs, next_obs, action, reward, done, infos)

    def sample():
        buf.sample(batch_size)

    times = timed_run(sample, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"sample_b{batch_size}", category="buffer_ops",
        framework="sb3", times_ns=times,
        params={"batch_size": batch_size, "buffer_size": buffer_size},
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    output_dir: str = "benchmark_results",
    *,
    n_warmup_push: int = DEFAULT_N_WARMUP_PUSH,
    n_reps_push: int = DEFAULT_N_REPS_PUSH,
    n_warmup_sample: int = DEFAULT_N_WARMUP_SAMPLE,
    n_reps_sample: int = DEFAULT_N_REPS_SAMPLE,
):
    print("=" * 70)
    print("Benchmark: Buffer Operations (rlox vs TorchRL vs SB3)")
    print("=" * 70)

    all_results = []
    all_comparisons = []

    # --- Push throughput ---
    print("\nPush Throughput")
    print("-" * 40)

    for obs_dim in [4, 28224]:
        print(f"\n  obs_dim = {obs_dim}:")

        rlox_push = bench_rlox_push(obs_dim, n_warmup=n_warmup_push, n_reps=n_reps_push)
        tp = rlox_push.throughput
        print(f"    rlox:     {rlox_push.median_ns/1e6:>8.2f} ms  ({tp:>12,.0f} trans/s)")
        all_results.append(rlox_push.summary())

        torchrl_push = bench_torchrl_push(obs_dim, n_warmup=n_warmup_push, n_reps=n_reps_push)
        if torchrl_push:
            tp_t = torchrl_push.throughput
            print(f"    torchrl:  {torchrl_push.median_ns/1e6:>8.2f} ms  ({tp_t:>12,.0f} trans/s)")
            all_results.append(torchrl_push.summary())
            comp = ComparisonResult(f"push_obs{obs_dim}", rlox_push, torchrl_push, "torchrl")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs torchrl: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

        sb3_push = bench_sb3_push(obs_dim, n_warmup=n_warmup_push, n_reps=n_reps_push)
        if sb3_push:
            tp_s = sb3_push.throughput
            print(f"    sb3:      {sb3_push.median_ns/1e6:>8.2f} ms  ({tp_s:>12,.0f} trans/s)")
            all_results.append(sb3_push.summary())
            comp = ComparisonResult(f"push_obs{obs_dim}", rlox_push, sb3_push, "sb3")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs sb3:     {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

    # --- Sample latency ---
    print("\n\nSample Latency (buffer_size=100K, obs_dim=4)")
    print("-" * 40)

    for batch_size in [32, 64, 256, 1024]:
        print(f"\n  batch_size = {batch_size}:")

        rlox_samp = bench_rlox_sample(batch_size, n_warmup=n_warmup_sample, n_reps=n_reps_sample)
        print(f"    rlox:     {rlox_samp.median_ns/1e3:>8.1f} us  (p99: {rlox_samp.p99_ns/1e3:.1f} us)")
        all_results.append(rlox_samp.summary())

        torchrl_samp = bench_torchrl_sample(batch_size, n_warmup=n_warmup_sample, n_reps=n_reps_sample)
        if torchrl_samp:
            print(f"    torchrl:  {torchrl_samp.median_ns/1e3:>8.1f} us  (p99: {torchrl_samp.p99_ns/1e3:.1f} us)")
            all_results.append(torchrl_samp.summary())
            comp = ComparisonResult(f"sample_b{batch_size}", rlox_samp, torchrl_samp, "torchrl")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs torchrl: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

        sb3_samp = bench_sb3_sample(batch_size, n_warmup=n_warmup_sample, n_reps=n_reps_sample)
        if sb3_samp:
            print(f"    sb3:      {sb3_samp.median_ns/1e3:>8.1f} us  (p99: {sb3_samp.p99_ns/1e3:.1f} us)")
            all_results.append(sb3_samp.summary())
            comp = ComparisonResult(f"sample_b{batch_size}", rlox_samp, sb3_samp, "sb3")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs sb3:     {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

    print()
    path = write_report(all_results, all_comparisons, output_dir)
    print(f"Report written to: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Buffer Operations")
    print("=" * 70)
    for c in all_comparisons:
        sig = "***" if c["significant"] else ""
        print(
            f"  {c['benchmark']:<25s} "
            f"rlox={c['rlox_median_ns']/1e3:>8.1f}us  "
            f"{c['baseline_framework']:<10s}={c['baseline_median_ns']/1e3:>8.1f}us  "
            f"speedup={c['speedup']:>5.1f}x [{c['speedup_ci_95_lo']:.1f},{c['speedup_ci_95_hi']:.1f}] {sig}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox buffer operation benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--n-warmup-push", type=int, default=DEFAULT_N_WARMUP_PUSH)
    parser.add_argument("--n-reps-push", type=int, default=DEFAULT_N_REPS_PUSH)
    parser.add_argument("--n-warmup-sample", type=int, default=DEFAULT_N_WARMUP_SAMPLE)
    parser.add_argument("--n-reps-sample", type=int, default=DEFAULT_N_REPS_SAMPLE)
    args = parser.parse_args()
    run_all(
        args.output_dir,
        n_warmup_push=args.n_warmup_push,
        n_reps_push=args.n_reps_push,
        n_warmup_sample=args.n_warmup_sample,
        n_reps_sample=args.n_reps_sample,
    )
