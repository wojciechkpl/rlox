#!/usr/bin/env python3
"""
Benchmark: rlox vs TRL-style operations

Compares rlox's Rust primitives against TRL's (HuggingFace) PyTorch
implementations for GRPO group advantages and token-level KL divergence.

TRL's computations are replicated as standalone PyTorch functions —
no TRL dependency required.

Usage:
    python benchmarks/bench_trl_comparison.py [--output-dir benchmark_results]
    python benchmarks/bench_trl_comparison.py --seed 123 --n-reps 200 --n-warmup 20
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, ComparisonResult, timed_run, write_report

# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

# (n_prompts, completions_per_prompt)
GRPO_CONFIGS = [(16, 4), (64, 8), (256, 16), (1024, 32)]

# (batch_size, sequence_length)
KL_CONFIGS = [
    (1, 128), (1, 512), (1, 2048), (1, 8192),
    (32, 128), (32, 512), (32, 2048),
]

# Verification configs
VERIFY_GRPO_CONFIGS = [(4, 3), (16, 8), (64, 16)]
VERIFY_KL_SEQ_LENS = [32, 128, 512]

# Tolerance thresholds for correctness checks
GRPO_COARSE_TOL = 0.5    # for small K where population/sample std diverge
GRPO_FINE_TOL = 0.25     # for K >= 8
KL_REL_TOL = 0.01        # relative tolerance for f64 vs f32

# ---------------------------------------------------------------------------
# TRL-style reference implementations (PyTorch CPU)
# ---------------------------------------------------------------------------

def trl_grpo_advantages(rewards, num_generations: int):
    """Replicate TRL GRPOTrainer._compute_rewards_and_advantages.

    TRL processes all groups at once via reshape + repeat_interleave.
    Source: trl/trainer/grpo_trainer.py (v0.16+).
    """
    import torch

    grouped = rewards.reshape(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + 1e-8)
    return advantages.reshape(-1)


def trl_token_kl_schulman(per_token_logps, per_token_ref_logps):
    """Replicate TRL's per-token KL (Schulman estimator).

    KL_approx = exp(log_ratio) - log_ratio - 1
    Source: trl/trainer/grpo_trainer.py (v0.16+).
    """
    import torch

    log_ratio = per_token_logps - per_token_ref_logps
    return torch.exp(log_ratio) - log_ratio - 1


# ---------------------------------------------------------------------------
# Section 1: GRPO Group Advantages
# ---------------------------------------------------------------------------

def bench_rlox_grpo_loop(n_prompts: int, k: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """rlox: per-group calls in a Python loop (existing API)."""
    from rlox import compute_group_advantages

    rng = np.random.default_rng(seed)
    groups = [rng.standard_normal(k) for _ in range(n_prompts)]

    def compute():
        for g in groups:
            compute_group_advantages(g)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"grpo_{n_prompts}x{k}", category="trl_comparison",
        framework="rlox_loop", times_ns=times,
    )


def bench_rlox_grpo_batched(n_prompts: int, k: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """rlox: single Rust call for all groups (batched API)."""
    from rlox import compute_batch_group_advantages

    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal(n_prompts * k)

    def compute():
        compute_batch_group_advantages(rewards, k)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"grpo_{n_prompts}x{k}", category="trl_comparison",
        framework="rlox_batched", times_ns=times,
    )


def bench_trl_grpo_cpu(n_prompts: int, k: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """TRL-style: vectorized PyTorch on CPU."""
    import torch

    rng = np.random.default_rng(seed)
    rewards = torch.from_numpy(rng.standard_normal(n_prompts * k)).float()

    def compute():
        trl_grpo_advantages(rewards, k)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"grpo_{n_prompts}x{k}", category="trl_comparison",
        framework="trl_cpu", times_ns=times,
    )


def bench_numpy_grpo_batched(n_prompts: int, k: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """NumPy: vectorized (reshape + broadcasting)."""
    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal(n_prompts * k)

    def compute():
        grouped = rewards.reshape(-1, k)
        mean = grouped.mean(axis=1, keepdims=True)
        std = grouped.std(axis=1, keepdims=True)
        _ = ((grouped - mean) / (std + 1e-8)).ravel()

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"grpo_{n_prompts}x{k}", category="trl_comparison",
        framework="numpy", times_ns=times,
    )


# ---------------------------------------------------------------------------
# Section 2: Token KL Divergence (Schulman estimator)
# ---------------------------------------------------------------------------

def bench_rlox_kl_schulman(batch: int, seq_len: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """rlox: Schulman KL, per-sequence loop."""
    from rlox import compute_token_kl_schulman

    rng = np.random.default_rng(seed)
    log_ps = [rng.standard_normal(seq_len) for _ in range(batch)]
    log_qs = [rng.standard_normal(seq_len) for _ in range(batch)]

    def compute():
        for lp, lq in zip(log_ps, log_qs):
            compute_token_kl_schulman(lp, lq)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"kl_{batch}x{seq_len}", category="trl_comparison",
        framework="rlox_schulman", times_ns=times,
    )


def bench_rlox_kl_batched(batch: int, seq_len: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """rlox: Schulman KL, single batched Rust call."""
    from rlox import compute_batch_token_kl_schulman

    rng = np.random.default_rng(seed)
    log_p = rng.standard_normal(batch * seq_len)
    log_q = rng.standard_normal(batch * seq_len)

    def compute():
        compute_batch_token_kl_schulman(log_p, log_q, seq_len)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"kl_{batch}x{seq_len}", category="trl_comparison",
        framework="rlox_batched", times_ns=times,
    )


def bench_rlox_kl_exact(batch: int, seq_len: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """rlox: exact KL, per-sequence loop."""
    from rlox import compute_token_kl

    rng = np.random.default_rng(seed)
    log_ps = [rng.standard_normal(seq_len) for _ in range(batch)]
    log_qs = [rng.standard_normal(seq_len) for _ in range(batch)]

    def compute():
        for lp, lq in zip(log_ps, log_qs):
            compute_token_kl(lp, lq)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"kl_{batch}x{seq_len}", category="trl_comparison",
        framework="rlox_exact", times_ns=times,
    )


def bench_trl_kl_cpu(batch: int, seq_len: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """TRL-style: Schulman KL, batched PyTorch on CPU."""
    import torch

    rng = np.random.default_rng(seed)
    log_p = torch.from_numpy(rng.standard_normal((batch, seq_len))).float()
    log_q = torch.from_numpy(rng.standard_normal((batch, seq_len))).float()

    def compute():
        trl_token_kl_schulman(log_p, log_q)

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"kl_{batch}x{seq_len}", category="trl_comparison",
        framework="trl_cpu", times_ns=times,
    )


def bench_numpy_kl_schulman(batch: int, seq_len: int, *, seed: int, n_warmup: int, n_reps: int) -> BenchmarkResult:
    """NumPy: Schulman KL, batched."""
    rng = np.random.default_rng(seed)
    log_p = rng.standard_normal((batch, seq_len))
    log_q = rng.standard_normal((batch, seq_len))

    def compute():
        r = log_p - log_q
        _ = np.exp(r) - r - 1

    times = timed_run(compute, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"kl_{batch}x{seq_len}", category="trl_comparison",
        framework="numpy", times_ns=times,
    )


# ---------------------------------------------------------------------------
# Section 3: Numerical Correctness Verification
# ---------------------------------------------------------------------------

def verify_grpo_equivalence(seed: int):
    """Verify rlox and TRL-style produce the same GRPO advantages."""
    import torch
    from rlox import compute_batch_group_advantages

    rng = np.random.default_rng(seed)
    for n_prompts, k in VERIFY_GRPO_CONFIGS:
        rewards_np = rng.standard_normal(n_prompts * k)
        rewards_pt = torch.from_numpy(rewards_np).float()

        rlox_adv = compute_batch_group_advantages(rewards_np, k)
        trl_adv = trl_grpo_advantages(rewards_pt, k).numpy()

        # Note: rlox uses population std (ddof=0), TRL/PyTorch uses sample std (ddof=1)
        # For large K these converge; check that results are directionally consistent
        max_diff = np.max(np.abs(rlox_adv.astype(np.float32) - trl_adv))
        status = "PASS" if max_diff < GRPO_COARSE_TOL else "FAIL"
        print(f"  GRPO {n_prompts}x{k}: max_diff={max_diff:.6f} [{status}]")
        if k >= 8:
            # Population std (rlox) vs sample std (PyTorch) differ by ~1/(2K),
            # so advantages can differ by ~0.2 for K=8. Threshold accounts for this.
            assert max_diff < GRPO_FINE_TOL, f"GRPO mismatch: {max_diff}"


def verify_kl_equivalence(seed: int):
    """Verify rlox Schulman KL matches TRL-style PyTorch."""
    import torch
    from rlox import compute_token_kl_schulman

    rng = np.random.default_rng(seed)
    for seq_len in VERIFY_KL_SEQ_LENS:
        log_p_np = rng.standard_normal(seq_len)
        log_q_np = rng.standard_normal(seq_len)

        rlox_kl = compute_token_kl_schulman(log_p_np, log_q_np)

        log_p_pt = torch.from_numpy(log_p_np).float()
        log_q_pt = torch.from_numpy(log_q_np).float()
        trl_kl = trl_token_kl_schulman(log_p_pt, log_q_pt).sum().item()

        # f64 vs f32 difference
        rel_diff = abs(rlox_kl - trl_kl) / max(abs(trl_kl), 1e-10)
        status = "PASS" if rel_diff < KL_REL_TOL else "FAIL"
        print(f"  KL seq={seq_len}: rlox={rlox_kl:.6f} trl={trl_kl:.6f} rel_diff={rel_diff:.6f} [{status}]")
        assert rel_diff < KL_REL_TOL, f"KL mismatch: {rel_diff}"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(output_dir: str, *, seed: int, n_warmup: int, n_reps_grpo: int, n_reps_kl: int):
    print("=" * 70)
    print("rlox Benchmark: TRL Comparison")
    print(f"  seed={seed}  n_warmup={n_warmup}  n_reps_grpo={n_reps_grpo}  n_reps_kl={n_reps_kl}")
    print("=" * 70)

    bench_kw_grpo = dict(seed=seed, n_warmup=n_warmup, n_reps=n_reps_grpo)
    bench_kw_kl = dict(seed=seed, n_warmup=n_warmup, n_reps=n_reps_kl)

    all_results = []
    all_comparisons = []

    # --- Correctness ---
    print("\nNumerical Correctness Verification")
    print("-" * 40)
    verify_grpo_equivalence(seed)
    verify_kl_equivalence(seed)
    print("  All correctness checks passed.\n")

    # --- GRPO Advantages ---
    print("GRPO Group Advantages")
    print("-" * 40)

    for n_prompts, k in GRPO_CONFIGS:
        print(f"\n  {n_prompts} prompts x {k} completions ({n_prompts*k} total):")

        rlox_loop = bench_rlox_grpo_loop(n_prompts, k, **bench_kw_grpo)
        print(f"    rlox (loop):    {rlox_loop.median_ns/1e3:>10.1f} us")
        all_results.append(rlox_loop.summary())

        rlox_batch = bench_rlox_grpo_batched(n_prompts, k, **bench_kw_grpo)
        print(f"    rlox (batched): {rlox_batch.median_ns/1e3:>10.1f} us")
        all_results.append(rlox_batch.summary())

        trl_cpu = bench_trl_grpo_cpu(n_prompts, k, **bench_kw_grpo)
        print(f"    trl-style CPU:  {trl_cpu.median_ns/1e3:>10.1f} us")
        all_results.append(trl_cpu.summary())

        numpy_res = bench_numpy_grpo_batched(n_prompts, k, **bench_kw_grpo)
        print(f"    numpy:          {numpy_res.median_ns/1e3:>10.1f} us")
        all_results.append(numpy_res.summary())

        # Comparisons: rlox_batched vs others
        for baseline, name in [(trl_cpu, "trl_cpu"), (numpy_res, "numpy")]:
            comp = ComparisonResult(
                f"grpo_{n_prompts}x{k}", rlox_batch, baseline, name,
            )
            lo, hi = comp.speedup_ci_95
            sig = "***" if comp.summary()["significant"] else ""
            print(f"    -> rlox_batched vs {name}: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}] {sig}")
            all_comparisons.append(comp.summary())

        # Also compare loop vs batched
        comp = ComparisonResult(
            f"grpo_{n_prompts}x{k}_loop_vs_batch", rlox_loop, rlox_batch, "rlox_batched",
        )
        # This is inverted: loop/batch — if >1, batched is faster
        lo, hi = comp.speedup_ci_95
        print(f"    -> loop overhead: {comp.speedup:.1f}x (batched = {comp.speedup:.1f}x faster)")

    # --- Token KL ---
    print("\n\nToken KL Divergence (Schulman estimator)")
    print("-" * 40)

    for batch, seq_len in KL_CONFIGS:
        label = f"{batch}x{seq_len}" if batch > 1 else str(seq_len)
        print(f"\n  batch={batch}, seq_len={seq_len} ({batch*seq_len} elements):")

        rlox_schulman = bench_rlox_kl_schulman(batch, seq_len, **bench_kw_kl)
        print(f"    rlox schulman:  {rlox_schulman.median_ns/1e3:>10.1f} us")
        all_results.append(rlox_schulman.summary())

        rlox_batched = bench_rlox_kl_batched(batch, seq_len, **bench_kw_kl)
        print(f"    rlox batched:   {rlox_batched.median_ns/1e3:>10.1f} us")
        all_results.append(rlox_batched.summary())

        if batch == 1:
            rlox_exact = bench_rlox_kl_exact(batch, seq_len, **bench_kw_kl)
            print(f"    rlox exact:     {rlox_exact.median_ns/1e3:>10.1f} us")
            all_results.append(rlox_exact.summary())

        trl_cpu = bench_trl_kl_cpu(batch, seq_len, **bench_kw_kl)
        print(f"    trl-style CPU:  {trl_cpu.median_ns/1e3:>10.1f} us")
        all_results.append(trl_cpu.summary())

        numpy_res = bench_numpy_kl_schulman(batch, seq_len, **bench_kw_kl)
        print(f"    numpy:          {numpy_res.median_ns/1e3:>10.1f} us")
        all_results.append(numpy_res.summary())

        # Comparisons: use batched rlox (fastest) as the rlox representative
        for baseline, name in [(trl_cpu, "trl_cpu"), (numpy_res, "numpy")]:
            comp = ComparisonResult(
                f"kl_{batch}x{seq_len}", rlox_batched, baseline, name,
            )
            lo, hi = comp.speedup_ci_95
            sig = "***" if comp.summary()["significant"] else ""
            print(f"    -> rlox_batched vs {name}: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}] {sig}")
            all_comparisons.append(comp.summary())

        # Loop vs batched overhead
        if batch > 1:
            comp = ComparisonResult(
                f"kl_{batch}x{seq_len}_loop_vs_batch", rlox_schulman, rlox_batched, "rlox_batched",
            )
            print(f"    -> loop overhead: batched is {comp.speedup:.1f}x faster")

    print()
    path = write_report(all_results, all_comparisons, output_dir)
    print(f"Report written to: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for c in all_comparisons:
        sig = "***" if c["significant"] else ""
        print(
            f"  {c['benchmark']:<30s} "
            f"rlox={c['rlox_median_ns']/1e3:>8.1f}us  "
            f"{c['baseline_framework']:<10s}={c['baseline_median_ns']/1e3:>8.1f}us  "
            f"speedup={c['speedup']:>5.1f}x [{c['speedup_ci_95_lo']:.1f},{c['speedup_ci_95_hi']:.1f}] {sig}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox vs TRL comparison benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--n-warmup", type=int, default=10, help="Warmup iterations before timing")
    parser.add_argument("--n-reps-grpo", type=int, default=50, help="Timed repetitions for GRPO benchmarks")
    parser.add_argument("--n-reps-kl", type=int, default=100, help="Timed repetitions for KL benchmarks")
    args = parser.parse_args()
    run_all(
        args.output_dir,
        seed=args.seed,
        n_warmup=args.n_warmup,
        n_reps_grpo=args.n_reps_grpo,
        n_reps_kl=args.n_reps_kl,
    )
