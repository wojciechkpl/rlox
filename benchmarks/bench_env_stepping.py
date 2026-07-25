#!/usr/bin/env python3
"""
Benchmark 3.1: Environment Stepping

Measures:
- Single-step latency (rlox vs Gymnasium vs EnvPool)
- Vectorized throughput scaling (1 to 1024 envs)
- Bridge overhead (native vs GymEnv wrapper)

Usage:
    python benchmarks/bench_env_stepping.py [--output-dir benchmark_results]
"""

import argparse
import json
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, ComparisonResult, timed_run, system_info, write_report

# ---------------------------------------------------------------------------
# Default repetition / warmup constants
# ---------------------------------------------------------------------------
DEFAULT_N_WARMUP_SINGLE = 100
DEFAULT_N_REPS_SINGLE = 1000
DEFAULT_N_WARMUP_BATCH = 5
DEFAULT_N_REPS_BATCH = 50
DEFAULT_N_WARMUP_HEAVY = 3
DEFAULT_N_REPS_HEAVY = 20
DEFAULT_N_WARMUP_NATIVE = 100
DEFAULT_N_REPS_NATIVE = 500
DEFAULT_N_BATCH_STEPS = 50
DEFAULT_N_REPS_FRAMEWORK = 10


# ---------------------------------------------------------------------------
# 3.1.1 Single environment step latency
# ---------------------------------------------------------------------------

def bench_rlox_single_step(n_reps: int = DEFAULT_N_REPS_SINGLE) -> BenchmarkResult:
    from rlox import CartPole
    env = CartPole(seed=42)

    def step():
        result = env.step(1)
        if result["terminated"] or result["truncated"]:
            env.reset(seed=42)

    times = timed_run(step, n_warmup=DEFAULT_N_WARMUP_SINGLE, n_reps=n_reps)
    return BenchmarkResult(
        name="single_step", category="env_stepping",
        framework="rlox", times_ns=times,
    )


def bench_gymnasium_single_step(n_reps: int = DEFAULT_N_REPS_SINGLE) -> BenchmarkResult | None:
    try:
        import gymnasium as gym
    except ImportError:
        print("  [skip] gymnasium not installed")
        return None

    env = gym.make("CartPole-v1")
    env.reset(seed=42)

    def step():
        _, _, term, trunc, _ = env.step(1)
        if term or trunc:
            env.reset(seed=42)

    times = timed_run(step, n_warmup=DEFAULT_N_WARMUP_SINGLE, n_reps=n_reps)
    env.close()
    return BenchmarkResult(
        name="single_step", category="env_stepping",
        framework="gymnasium", times_ns=times,
    )


# ---------------------------------------------------------------------------
# 3.1.2 Vectorized environment throughput
# ---------------------------------------------------------------------------

def bench_rlox_vecenv(num_envs: int, n_batch_steps: int = DEFAULT_N_BATCH_STEPS, n_reps: int = DEFAULT_N_REPS_BATCH) -> BenchmarkResult:
    from rlox import VecEnv
    env = VecEnv(n=num_envs, seed=42)
    actions = [0] * num_envs

    def step_batch():
        for _ in range(n_batch_steps):
            env.step_all(actions)

    times = timed_run(step_batch, n_warmup=DEFAULT_N_WARMUP_BATCH, n_reps=n_reps)
    total_steps_per_call = num_envs * n_batch_steps
    return BenchmarkResult(
        name=f"vecenv_{num_envs}", category="env_stepping",
        framework="rlox", times_ns=times,
        params={"num_envs": num_envs, "batch_steps": n_batch_steps, "n_items": total_steps_per_call},
    )


def bench_gymnasium_sync_vecenv(num_envs: int, n_batch_steps: int = DEFAULT_N_BATCH_STEPS, n_reps: int = DEFAULT_N_REPS_BATCH) -> BenchmarkResult | None:
    try:
        import gymnasium as gym
        from gymnasium.vector import SyncVectorEnv
    except ImportError:
        return None

    env = SyncVectorEnv([lambda: gym.make("CartPole-v1")] * num_envs)
    env.reset(seed=42)
    actions = np.zeros(num_envs, dtype=np.int64)

    def step_batch():
        for _ in range(n_batch_steps):
            env.step(actions)

    times = timed_run(step_batch, n_warmup=DEFAULT_N_WARMUP_BATCH, n_reps=n_reps)
    env.close()
    total_steps_per_call = num_envs * n_batch_steps
    return BenchmarkResult(
        name=f"vecenv_{num_envs}", category="env_stepping",
        framework="gymnasium_sync", times_ns=times,
        params={"num_envs": num_envs, "batch_steps": n_batch_steps, "n_items": total_steps_per_call},
    )


def bench_gymnasium_async_vecenv(num_envs: int, n_batch_steps: int = DEFAULT_N_BATCH_STEPS, n_reps: int = DEFAULT_N_REPS_HEAVY) -> BenchmarkResult | None:
    try:
        import gymnasium as gym
        from gymnasium.vector import AsyncVectorEnv
    except ImportError:
        return None

    env = AsyncVectorEnv([lambda: gym.make("CartPole-v1")] * num_envs)
    env.reset(seed=42)
    actions = np.zeros(num_envs, dtype=np.int64)

    def step_batch():
        for _ in range(n_batch_steps):
            env.step(actions)

    times = timed_run(step_batch, n_warmup=DEFAULT_N_WARMUP_HEAVY, n_reps=n_reps)
    env.close()
    total_steps_per_call = num_envs * n_batch_steps
    return BenchmarkResult(
        name=f"vecenv_{num_envs}", category="env_stepping",
        framework="gymnasium_async", times_ns=times,
        params={"num_envs": num_envs, "batch_steps": n_batch_steps, "n_items": total_steps_per_call},
    )


def bench_sb3_dummyvecenv(num_envs: int, n_batch_steps: int = DEFAULT_N_BATCH_STEPS, n_reps: int = DEFAULT_N_REPS_BATCH) -> BenchmarkResult | None:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym
    except ImportError:
        return None

    env = DummyVecEnv([lambda: gym.make("CartPole-v1")] * num_envs)
    env.reset()
    actions = np.zeros(num_envs, dtype=np.int64)

    def step_batch():
        for _ in range(n_batch_steps):
            env.step(actions)

    times = timed_run(step_batch, n_warmup=DEFAULT_N_WARMUP_BATCH, n_reps=n_reps)
    env.close()
    total_steps_per_call = num_envs * n_batch_steps
    return BenchmarkResult(
        name=f"vecenv_{num_envs}", category="env_stepping",
        framework="sb3_dummy", times_ns=times,
        params={"num_envs": num_envs, "batch_steps": n_batch_steps, "n_items": total_steps_per_call},
    )


def bench_sb3_subprocvecenv(num_envs: int, n_batch_steps: int = DEFAULT_N_BATCH_STEPS, n_reps: int = DEFAULT_N_REPS_HEAVY) -> BenchmarkResult | None:
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        import gymnasium as gym
    except ImportError:
        return None

    env = SubprocVecEnv([lambda: gym.make("CartPole-v1")] * num_envs)
    env.reset()
    actions = np.zeros(num_envs, dtype=np.int64)

    def step_batch():
        for _ in range(n_batch_steps):
            env.step(actions)

    times = timed_run(step_batch, n_warmup=DEFAULT_N_WARMUP_HEAVY, n_reps=n_reps)
    env.close()
    total_steps_per_call = num_envs * n_batch_steps
    return BenchmarkResult(
        name=f"vecenv_{num_envs}", category="env_stepping",
        framework="sb3_subproc", times_ns=times,
        params={"num_envs": num_envs, "batch_steps": n_batch_steps, "n_items": total_steps_per_call},
    )


# ---------------------------------------------------------------------------
# 3.1.2b TorchRL environment stepping
# ---------------------------------------------------------------------------

def bench_torchrl_single_step(n_reps: int = DEFAULT_N_REPS_SINGLE) -> BenchmarkResult | None:
    try:
        from torchrl.envs import GymEnv
        import torch
    except ImportError:
        print("  [skip] torchrl not installed")
        return None

    env = GymEnv("CartPole-v1", device="cpu")
    td = env.reset()

    def step():
        nonlocal td
        td.set("action", torch.tensor(1))
        td = env.step(td)
        next_td = td.get("next")
        if next_td.get("done").item():
            td = env.reset()
        else:
            td = next_td

    times = timed_run(step, n_warmup=DEFAULT_N_WARMUP_SINGLE, n_reps=n_reps)
    env.close()
    return BenchmarkResult(
        name="single_step", category="env_stepping",
        framework="torchrl", times_ns=times,
    )


def bench_torchrl_serial_vecenv(num_envs: int, n_batch_steps: int = DEFAULT_N_BATCH_STEPS, n_reps: int = DEFAULT_N_REPS_BATCH) -> BenchmarkResult | None:
    try:
        from torchrl.envs import GymEnv, SerialEnv
        import torch
    except ImportError:
        return None

    try:
        env = SerialEnv(num_envs, lambda: GymEnv("CartPole-v1", device="cpu"))
        td = env.reset()

        def step_batch():
            nonlocal td
            for _ in range(n_batch_steps):
                td.set("action", torch.zeros(num_envs, dtype=torch.long))
                td = env.step(td)
                td = td.get("next")

        times = timed_run(step_batch, n_warmup=DEFAULT_N_WARMUP_BATCH, n_reps=n_reps)
        env.close()
    except Exception as e:
        print(f"    [skip] torchrl SerialEnv failed ({num_envs} envs): {e}")
        return None

    total_steps_per_call = num_envs * n_batch_steps
    return BenchmarkResult(
        name=f"vecenv_{num_envs}", category="env_stepping",
        framework="torchrl_serial", times_ns=times,
        params={"num_envs": num_envs, "batch_steps": n_batch_steps, "n_items": total_steps_per_call},
    )


def bench_torchrl_parallel_vecenv(num_envs: int, n_batch_steps: int = DEFAULT_N_BATCH_STEPS, n_reps: int = DEFAULT_N_REPS_HEAVY) -> BenchmarkResult | None:
    try:
        from torchrl.envs import GymEnv, ParallelEnv
        import torch
    except ImportError:
        return None

    try:
        env = ParallelEnv(num_envs, lambda: GymEnv("CartPole-v1", device="cpu"))
        td = env.reset()

        def step_batch():
            nonlocal td
            for _ in range(n_batch_steps):
                td.set("action", torch.zeros(num_envs, dtype=torch.long))
                td = env.step(td)
                td = td.get("next")

        times = timed_run(step_batch, n_warmup=DEFAULT_N_WARMUP_HEAVY, n_reps=n_reps)
        env.close()
    except Exception as e:
        print(f"    [skip] torchrl ParallelEnv failed ({num_envs} envs): {e}")
        return None

    total_steps_per_call = num_envs * n_batch_steps
    return BenchmarkResult(
        name=f"vecenv_{num_envs}", category="env_stepping",
        framework="torchrl_parallel", times_ns=times,
        params={"num_envs": num_envs, "batch_steps": n_batch_steps, "n_items": total_steps_per_call},
    )


# ---------------------------------------------------------------------------
# 3.1.3 Bridge overhead
# ---------------------------------------------------------------------------

def bench_bridge_overhead(n_reps: int = DEFAULT_N_REPS_NATIVE) -> tuple[BenchmarkResult, BenchmarkResult | None]:
    from rlox import CartPole

    native_env = CartPole(seed=42)
    def native_step():
        result = native_env.step(1)
        if result["terminated"] or result["truncated"]:
            native_env.reset(seed=42)

    native_times = timed_run(native_step, n_warmup=DEFAULT_N_WARMUP_NATIVE, n_reps=n_reps)
    native_result = BenchmarkResult(
        name="bridge_overhead", category="env_stepping",
        framework="rlox_native", times_ns=native_times,
    )

    bridge_result = None
    try:
        import gymnasium as gym
        from rlox import GymEnv

        gym_raw = gym.make("CartPole-v1")
        bridge_env = GymEnv(gym_raw)
        bridge_env.reset(seed=42)

        def bridge_step():
            result = bridge_env.step(1)
            if result["terminated"] or result["truncated"]:
                bridge_env.reset(seed=42)

        bridge_times = timed_run(bridge_step, n_warmup=DEFAULT_N_WARMUP_NATIVE, n_reps=n_reps)
        gym_raw.close()
        bridge_result = BenchmarkResult(
            name="bridge_overhead", category="env_stepping",
            framework="rlox_gymenv_bridge", times_ns=bridge_times,
        )
    except ImportError:
        pass

    return native_result, bridge_result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _scale(n: int, scale: float) -> int:
    """Scale a rep count by *scale*, clamping to at least 1."""
    return max(1, int(n * scale))


def run_all(output_dir: str = "benchmark_results", reps_scale: float = 1.0):
    print("=" * 70)
    print("rlox Benchmark 3.1: Environment Stepping")
    print("=" * 70)
    print(f"\nSystem: {sys.platform}, Python {sys.version.split()[0]}")
    print(f"CPU cores: {os.cpu_count()}")
    if reps_scale != 1.0:
        print(f"Reps scale: {reps_scale:.2f}x")
    print()

    # Pre-compute scaled rep counts
    n_reps_single = _scale(DEFAULT_N_REPS_SINGLE, reps_scale)
    n_reps_batch = _scale(DEFAULT_N_REPS_BATCH, reps_scale)
    n_reps_heavy = _scale(DEFAULT_N_REPS_HEAVY, reps_scale)
    n_reps_native = _scale(DEFAULT_N_REPS_NATIVE, reps_scale)
    n_reps_framework = _scale(DEFAULT_N_REPS_FRAMEWORK, reps_scale)

    all_results = []
    all_comparisons = []

    # --- 3.1.1 Single step ---
    print("3.1.1 Single Step Latency")
    print("-" * 40)

    rlox_single = bench_rlox_single_step(n_reps=n_reps_single)
    print(f"  rlox:       {rlox_single.median_ns:>10.0f} ns (IQR: {rlox_single.iqr_ns:.0f})")
    all_results.append(rlox_single.summary())

    gym_single = bench_gymnasium_single_step(n_reps=n_reps_single)
    if gym_single:
        print(f"  gymnasium:  {gym_single.median_ns:>10.0f} ns (IQR: {gym_single.iqr_ns:.0f})")
        all_results.append(gym_single.summary())
        comp = ComparisonResult("single_step", rlox_single, gym_single, "gymnasium")
        lo, hi = comp.speedup_ci_95
        print(f"  -> rlox speedup: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
        all_comparisons.append(comp.summary())

    torchrl_single = bench_torchrl_single_step(n_reps=n_reps_single)
    if torchrl_single:
        print(f"  torchrl:    {torchrl_single.median_ns:>10.0f} ns (IQR: {torchrl_single.iqr_ns:.0f})")
        all_results.append(torchrl_single.summary())
        comp = ComparisonResult("single_step", rlox_single, torchrl_single, "torchrl")
        lo, hi = comp.speedup_ci_95
        print(f"  -> vs torchrl: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
        all_comparisons.append(comp.summary())
    print()

    # --- 3.1.2 Vectorized throughput ---
    print("3.1.2 Vectorized Environment Throughput")
    print("-" * 40)

    env_counts = [1, 4, 16, 64, 128, 256, 512]

    for n in env_counts:
        print(f"\n  num_envs = {n}:")

        rlox_vec = bench_rlox_vecenv(n, n_reps=n_reps_batch)
        throughput = rlox_vec.throughput
        print(f"    rlox:            {rlox_vec.median_ns/1e6:>8.2f} ms  ({throughput:>12,.0f} steps/s)")
        all_results.append(rlox_vec.summary())

        gym_sync = bench_gymnasium_sync_vecenv(n, n_reps=n_reps_batch)
        if gym_sync:
            throughput_gs = gym_sync.throughput
            print(f"    gym sync:        {gym_sync.median_ns/1e6:>8.2f} ms  ({throughput_gs:>12,.0f} steps/s)")
            all_results.append(gym_sync.summary())
            comp = ComparisonResult(f"vecenv_{n}", rlox_vec, gym_sync, "gymnasium_sync")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs gym sync:  {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

        # Only run async/subproc for moderate env counts (too slow otherwise)
        if n <= 128:
            gym_async = bench_gymnasium_async_vecenv(n, n_batch_steps=DEFAULT_N_BATCH_STEPS, n_reps=n_reps_framework)
            if gym_async:
                throughput_ga = gym_async.throughput
                print(f"    gym async:       {gym_async.median_ns/1e6:>8.2f} ms  ({throughput_ga:>12,.0f} steps/s)")
                all_results.append(gym_async.summary())
                comp = ComparisonResult(f"vecenv_{n}", rlox_vec, gym_async, "gymnasium_async")
                lo, hi = comp.speedup_ci_95
                print(f"    -> vs gym async: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
                all_comparisons.append(comp.summary())

            sb3_dummy = bench_sb3_dummyvecenv(n, n_reps=n_reps_batch)
            if sb3_dummy:
                throughput_sd = sb3_dummy.throughput
                print(f"    sb3 dummy:       {sb3_dummy.median_ns/1e6:>8.2f} ms  ({throughput_sd:>12,.0f} steps/s)")
                all_results.append(sb3_dummy.summary())
                comp = ComparisonResult(f"vecenv_{n}", rlox_vec, sb3_dummy, "sb3_dummyvecenv")
                lo, hi = comp.speedup_ci_95
                print(f"    -> vs sb3 dummy: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
                all_comparisons.append(comp.summary())

        # TorchRL serial
        if n <= 256:
            torchrl_serial = bench_torchrl_serial_vecenv(n, n_reps=n_reps_batch)
            if torchrl_serial:
                throughput_ts = torchrl_serial.throughput
                print(f"    torchrl serial:  {torchrl_serial.median_ns/1e6:>8.2f} ms  ({throughput_ts:>12,.0f} steps/s)")
                all_results.append(torchrl_serial.summary())
                comp = ComparisonResult(f"vecenv_{n}", rlox_vec, torchrl_serial, "torchrl_serial")
                lo, hi = comp.speedup_ci_95
                print(f"    -> vs torchrl serial: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
                all_comparisons.append(comp.summary())

        # TorchRL parallel (multiprocess)
        if n <= 64:
            torchrl_par = bench_torchrl_parallel_vecenv(n, n_batch_steps=DEFAULT_N_BATCH_STEPS, n_reps=n_reps_framework)
            if torchrl_par:
                throughput_tp = torchrl_par.throughput
                print(f"    torchrl parallel: {torchrl_par.median_ns/1e6:>8.2f} ms  ({throughput_tp:>12,.0f} steps/s)")
                all_results.append(torchrl_par.summary())
                comp = ComparisonResult(f"vecenv_{n}", rlox_vec, torchrl_par, "torchrl_parallel")
                lo, hi = comp.speedup_ci_95
                print(f"    -> vs torchrl parallel: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
                all_comparisons.append(comp.summary())

        if n <= 64:
            sb3_subproc = bench_sb3_subprocvecenv(n, n_batch_steps=DEFAULT_N_BATCH_STEPS, n_reps=n_reps_framework)
            if sb3_subproc:
                throughput_ss = sb3_subproc.throughput
                print(f"    sb3 subproc:     {sb3_subproc.median_ns/1e6:>8.2f} ms  ({throughput_ss:>12,.0f} steps/s)")
                all_results.append(sb3_subproc.summary())
                comp = ComparisonResult(f"vecenv_{n}", rlox_vec, sb3_subproc, "sb3_subprocvecenv")
                lo, hi = comp.speedup_ci_95
                print(f"    -> vs sb3 subproc: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
                all_comparisons.append(comp.summary())

    print()

    # --- 3.1.3 Bridge overhead ---
    print("3.1.3 Bridge Overhead")
    print("-" * 40)

    native, bridge = bench_bridge_overhead(n_reps=n_reps_native)
    print(f"  native:  {native.median_ns:>10.0f} ns")
    all_results.append(native.summary())

    if bridge:
        overhead = bridge.median_ns - native.median_ns
        print(f"  bridge:  {bridge.median_ns:>10.0f} ns")
        print(f"  overhead: {overhead:>9.0f} ns ({overhead/1000:.1f} us)")
        all_results.append(bridge.summary())

    print()

    # --- Write report ---
    path = write_report(all_results, all_comparisons, output_dir)
    print(f"Report written to: {path}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for c in all_comparisons:
        sig = "***" if c["significant"] else ""
        print(
            f"  {c['benchmark']:<35s} "
            f"rlox={c['rlox_median_ns']/1e3:>8.1f}us  "
            f"{c['baseline_framework']:<20s}={c['baseline_median_ns']/1e3:>8.1f}us  "
            f"speedup={c['speedup']:>5.1f}x [{c['speedup_ci_95_lo']:.1f},{c['speedup_ci_95_hi']:.1f}] {sig}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox env stepping benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument(
        "--n-reps-scale", type=float, default=1.0,
        help="Multiplier for all rep counts (e.g. 0.1 for a quick smoke run, 2.0 for thorough)",
    )
    args = parser.parse_args()
    run_all(args.output_dir, reps_scale=args.n_reps_scale)
