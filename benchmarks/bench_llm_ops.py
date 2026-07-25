#!/usr/bin/env python3
"""
Benchmark: LLM Post-Training Operations

GRPO group advantages and token-level KL divergence.
rlox vs NumPy vs PyTorch (no TorchRL/SB3 equivalents).

Usage:
    python benchmarks/bench_llm_ops.py [--output-dir benchmark_results]
        [--seed 42] [--n-warmup 10] [--n-reps-grpo 50] [--n-reps-kl 100]
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

def numpy_group_advantages(rewards: np.ndarray) -> np.ndarray:
    mean = rewards.mean()
    std = rewards.std()
    if std < 1e-8:
        return np.zeros_like(rewards)
    return (rewards - mean) / std


def numpy_token_kl(log_p: np.ndarray, log_q: np.ndarray) -> float:
    return float(np.sum(np.exp(log_p) * (log_p - log_q)))


# ---------------------------------------------------------------------------
# GRPO advantages
# ---------------------------------------------------------------------------

def bench_rlox_grpo(n_prompts: int, k: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    from rlox import compute_group_advantages

    rng = np.random.default_rng(seed)
    groups = [rng.standard_normal(k) for _ in range(n_prompts)]

    def compute():
        for g in groups:
            compute_group_advantages(g)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"grpo_{n_prompts}x{k}", category="llm",
        framework="rlox", times_ns=times,
    )


def bench_numpy_grpo(n_prompts: int, k: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    rng = np.random.default_rng(seed)
    groups = [rng.standard_normal(k) for _ in range(n_prompts)]

    def compute():
        for g in groups:
            numpy_group_advantages(g)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"grpo_{n_prompts}x{k}", category="llm",
        framework="numpy", times_ns=times,
    )


def bench_torch_grpo(n_prompts: int, k: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult | None:
    try:
        import torch
    except ImportError:
        return None

    rng = np.random.default_rng(seed)
    groups = [torch.from_numpy(rng.standard_normal(k)) for _ in range(n_prompts)]

    def compute():
        for g in groups:
            mean = g.mean()
            std = g.std()
            if std < 1e-8:
                torch.zeros_like(g)
            else:
                (g - mean) / std

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"grpo_{n_prompts}x{k}", category="llm",
        framework="pytorch", times_ns=times,
    )


# ---------------------------------------------------------------------------
# Token KL
# ---------------------------------------------------------------------------

def bench_rlox_kl(seq_len: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    from rlox import compute_token_kl

    rng = np.random.default_rng(seed)
    log_p = rng.standard_normal(seq_len)
    log_q = rng.standard_normal(seq_len)

    times = timed_run(
        lambda: compute_token_kl(log_p, log_q),
        n_warmup=n_warmup, n_reps=n_reps,
    )
    return BenchmarkResult(
        name=f"token_kl_{seq_len}", category="llm",
        framework="rlox", times_ns=times,
    )


def bench_numpy_kl(seq_len: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    rng = np.random.default_rng(seed)
    log_p = rng.standard_normal(seq_len)
    log_q = rng.standard_normal(seq_len)

    times = timed_run(
        lambda: numpy_token_kl(log_p, log_q),
        n_warmup=n_warmup, n_reps=n_reps,
    )
    return BenchmarkResult(
        name=f"token_kl_{seq_len}", category="llm",
        framework="numpy", times_ns=times,
    )


def bench_torch_kl(seq_len: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult | None:
    try:
        import torch
    except ImportError:
        return None

    rng = np.random.default_rng(seed)
    log_p = torch.from_numpy(rng.standard_normal(seq_len))
    log_q = torch.from_numpy(rng.standard_normal(seq_len))

    def compute():
        torch.sum(torch.exp(log_p) * (log_p - log_q)).item()

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"token_kl_{seq_len}", category="llm",
        framework="pytorch", times_ns=times,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    output_dir: str = "benchmark_results",
    *,
    seed: int = 42,
    n_warmup: int = 10,
    n_reps_grpo: int = 50,
    n_reps_kl: int = 100,
):
    print("=" * 70)
    print("Benchmark: LLM Operations (rlox vs NumPy vs PyTorch)")
    print("=" * 70)

    all_results = []
    all_comparisons = []

    # --- GRPO ---
    print("\nGRPO Group Advantages")
    print("-" * 40)

    for n_prompts, k in [(16, 4), (64, 8), (256, 16)]:
        print(f"\n  {n_prompts} prompts × {k} completions:")

        grpo_kw = dict(seed=seed, n_warmup=n_warmup, n_reps=n_reps_grpo)

        rlox_res = bench_rlox_grpo(n_prompts, k, **grpo_kw)
        print(f"    rlox:    {rlox_res.median_ns/1e3:>8.1f} us")
        all_results.append(rlox_res.summary())

        numpy_res = bench_numpy_grpo(n_prompts, k, **grpo_kw)
        print(f"    numpy:   {numpy_res.median_ns/1e3:>8.1f} us")
        all_results.append(numpy_res.summary())
        comp = ComparisonResult(f"grpo_{n_prompts}x{k}", rlox_res, numpy_res, "numpy")
        lo, hi = comp.speedup_ci_95
        print(f"    -> vs numpy: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
        all_comparisons.append(comp.summary())

        torch_res = bench_torch_grpo(n_prompts, k, **grpo_kw)
        if torch_res:
            print(f"    pytorch: {torch_res.median_ns/1e3:>8.1f} us")
            all_results.append(torch_res.summary())
            comp = ComparisonResult(f"grpo_{n_prompts}x{k}", rlox_res, torch_res, "pytorch")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs pytorch: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

    # --- Token KL ---
    print("\n\nToken-Level KL Divergence")
    print("-" * 40)

    for seq_len in [128, 512, 2048, 8192]:
        print(f"\n  seq_len = {seq_len}:")

        kl_kw = dict(seed=seed, n_warmup=n_warmup, n_reps=n_reps_kl)

        rlox_res = bench_rlox_kl(seq_len, **kl_kw)
        print(f"    rlox:    {rlox_res.median_ns/1e3:>8.1f} us")
        all_results.append(rlox_res.summary())

        numpy_res = bench_numpy_kl(seq_len, **kl_kw)
        print(f"    numpy:   {numpy_res.median_ns/1e3:>8.1f} us")
        all_results.append(numpy_res.summary())
        comp = ComparisonResult(f"token_kl_{seq_len}", rlox_res, numpy_res, "numpy")
        lo, hi = comp.speedup_ci_95
        print(f"    -> vs numpy: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
        all_comparisons.append(comp.summary())

        torch_res = bench_torch_kl(seq_len, **kl_kw)
        if torch_res:
            print(f"    pytorch: {torch_res.median_ns/1e3:>8.1f} us")
            all_results.append(torch_res.summary())
            comp = ComparisonResult(f"token_kl_{seq_len}", rlox_res, torch_res, "pytorch")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs pytorch: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

    print()
    path = write_report(all_results, all_comparisons, output_dir)
    print(f"Report written to: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — LLM Operations")
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
    parser = argparse.ArgumentParser(description="rlox LLM operation benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-reps-grpo", type=int, default=50)
    parser.add_argument("--n-reps-kl", type=int, default=100)
    args = parser.parse_args()
    run_all(
        args.output_dir,
        seed=args.seed,
        n_warmup=args.n_warmup,
        n_reps_grpo=args.n_reps_grpo,
        n_reps_kl=args.n_reps_kl,
    )
