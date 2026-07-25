#!/usr/bin/env python3
"""
Benchmark: GAE Computation

Compares rlox Rust GAE vs TorchRL C++ GAE vs Python loop (SB3/CleanRL baseline).

Usage:
    python benchmarks/bench_gae.py [--output-dir benchmark_results] [--seed 42] [--n-warmup 10] [--n-reps 100]
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, ComparisonResult, timed_run, write_report


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def reference_gae_numpy(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """CleanRL/SB3-style GAE in pure Python loop. Correctness baseline."""
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
# rlox GAE
# ---------------------------------------------------------------------------

def bench_rlox_gae(n_steps: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    from rlox import compute_gae

    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal(n_steps)
    values = rng.standard_normal(n_steps)
    dones = (rng.random(n_steps) > 0.95).astype(float)

    times = timed_run(
        lambda: compute_gae(rewards, values, dones, 0.0, 0.99, 0.95),
        n_warmup=n_warmup, n_reps=n_reps,
    )
    return BenchmarkResult(
        name=f"gae_{n_steps}", category="gae",
        framework="rlox", times_ns=times,
        params={"n_steps": n_steps},
    )


# ---------------------------------------------------------------------------
# TorchRL GAE
# ---------------------------------------------------------------------------

def bench_torchrl_gae(n_steps: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult | None:
    try:
        import torch
        from torchrl.objectives.value.functional import generalized_advantage_estimate
    except ImportError:
        print("  [skip] torchrl not installed")
        return None

    rng = np.random.default_rng(seed)
    rewards_np = rng.standard_normal(n_steps)
    values_np = rng.standard_normal(n_steps)
    dones_np = (rng.random(n_steps) > 0.95).astype(float)

    # TorchRL expects: gamma, lam, state_value, next_state_value, reward, done, terminated
    # Shapes: [T, 1] for values/rewards, [T, 1] for done
    gamma = 0.99
    lam = 0.95

    rewards_t = torch.from_numpy(rewards_np).unsqueeze(-1).double()
    # state_value: values[0..T], next_state_value: values[1..T] + last_value
    values_t = torch.from_numpy(values_np).unsqueeze(-1).double()
    next_values_t = torch.cat([values_t[1:], torch.zeros(1, 1, dtype=torch.float64)])
    dones_t = torch.from_numpy(dones_np).unsqueeze(-1).bool()
    terminated_t = dones_t.clone()

    def run_gae():
        generalized_advantage_estimate(
            gamma, lam, values_t, next_values_t, rewards_t, dones_t, terminated_t,
        )

    times = timed_run(run_gae, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"gae_{n_steps}", category="gae",
        framework="torchrl", times_ns=times,
        params={"n_steps": n_steps},
    )


# ---------------------------------------------------------------------------
# NumPy/Python loop GAE (SB3/CleanRL equivalent)
# ---------------------------------------------------------------------------

def bench_numpy_gae(n_steps: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal(n_steps)
    values = rng.standard_normal(n_steps)
    dones = (rng.random(n_steps) > 0.95).astype(float)

    times = timed_run(
        lambda: reference_gae_numpy(rewards, values, dones, 0.0, 0.99, 0.95),
        n_warmup=n_warmup, n_reps=n_reps,
    )
    return BenchmarkResult(
        name=f"gae_{n_steps}", category="gae",
        framework="numpy_loop", times_ns=times,
        params={"n_steps": n_steps},
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(output_dir: str = "benchmark_results", *, seed: int, n_warmup: int, n_reps: int):
    print("=" * 70)
    print("Benchmark: GAE Computation (rlox vs TorchRL vs NumPy loop)")
    print("=" * 70)

    all_results = []
    all_comparisons = []

    step_counts = [128, 512, 2048, 8192, 32768]

    bench_kw = {"seed": seed, "n_warmup": n_warmup, "n_reps": n_reps}

    for n_steps in step_counts:
        print(f"\n  n_steps = {n_steps}:")

        rlox_gae = bench_rlox_gae(n_steps, **bench_kw)
        print(f"    rlox:       {rlox_gae.median_ns/1e3:>10.1f} us")
        all_results.append(rlox_gae.summary())

        numpy_gae = bench_numpy_gae(n_steps, **bench_kw)
        print(f"    numpy loop: {numpy_gae.median_ns/1e3:>10.1f} us")
        all_results.append(numpy_gae.summary())

        comp_np = ComparisonResult(f"gae_{n_steps}", rlox_gae, numpy_gae, "numpy_loop")
        lo, hi = comp_np.speedup_ci_95
        print(f"    -> vs numpy: {comp_np.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
        all_comparisons.append(comp_np.summary())

        torchrl_gae = bench_torchrl_gae(n_steps, **bench_kw)
        if torchrl_gae:
            print(f"    torchrl:    {torchrl_gae.median_ns/1e3:>10.1f} us")
            all_results.append(torchrl_gae.summary())
            comp_trl = ComparisonResult(f"gae_{n_steps}", rlox_gae, torchrl_gae, "torchrl")
            lo, hi = comp_trl.speedup_ci_95
            print(f"    -> vs torchrl: {comp_trl.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp_trl.summary())

    print()
    path = write_report(all_results, all_comparisons, output_dir)
    print(f"Report written to: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — GAE Computation")
    print("=" * 70)
    for c in all_comparisons:
        sig = "***" if c["significant"] else ""
        print(
            f"  {c['benchmark']:<25s} "
            f"rlox={c['rlox_median_ns']/1e3:>8.1f}us  "
            f"{c['baseline_framework']:<12s}={c['baseline_median_ns']/1e3:>8.1f}us  "
            f"speedup={c['speedup']:>5.1f}x [{c['speedup_ci_95_lo']:.1f},{c['speedup_ci_95_hi']:.1f}] {sig}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox GAE computation benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-reps", type=int, default=100)
    args = parser.parse_args()
    run_all(args.output_dir, seed=args.seed, n_warmup=args.n_warmup, n_reps=args.n_reps)
