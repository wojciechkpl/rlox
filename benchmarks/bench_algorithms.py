#!/usr/bin/env python3
"""
Benchmark: Algorithm Wall-Clock Performance

Measures wall-clock time and SPS for PPO, A2C, DQN, and IMPALA
on CartPole-v1 over a fixed number of timesteps.

Usage:
    python benchmarks/bench_algorithms.py [--output-dir benchmark_results] [--seed 42] [--total-timesteps 10000]
"""

import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, write_report


DEFAULT_TOTAL_TIMESTEPS = 10_000


# ---------------------------------------------------------------------------
# Individual algorithm benchmarks
# ---------------------------------------------------------------------------

def _bench_trainer(
    trainer_cls: type,
    name: str,
    *,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    seed: int = 42,
    config: dict | None = None,
) -> BenchmarkResult:
    """Run a single trainer and measure wall-clock time + SPS."""
    trainer = trainer_cls(env="CartPole-v1", seed=seed, **({"config": config} if config else {}))

    start = time.perf_counter_ns()
    metrics = trainer.train(total_timesteps=total_timesteps)
    elapsed_ns = float(time.perf_counter_ns() - start)

    elapsed_s = elapsed_ns / 1e9
    sps = total_timesteps / elapsed_s if elapsed_s > 0 else 0.0

    return BenchmarkResult(
        name=name,
        category="algorithms",
        framework="rlox",
        times_ns=[elapsed_ns],
        params={"total_timesteps": total_timesteps, "n_items": total_timesteps},
        metadata={
            "sps": sps,
            "wall_clock_s": elapsed_s,
            "train_metrics": {
                k: v for k, v in (metrics or {}).items()
                if isinstance(v, (int, float))
            },
        },
    )


def bench_ppo_cartpole(
    *,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    seed: int = 42,
) -> BenchmarkResult:
    """PPO wall-clock + SPS on CartPole-v1."""
    from rlox import PPOTrainer
    return _bench_trainer(
        PPOTrainer, "ppo_cartpole",
        total_timesteps=total_timesteps, seed=seed,
    )


def bench_a2c_cartpole(
    *,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    seed: int = 42,
) -> BenchmarkResult:
    """A2C wall-clock + SPS on CartPole-v1."""
    from rlox import A2CTrainer
    return _bench_trainer(
        A2CTrainer, "a2c_cartpole",
        total_timesteps=total_timesteps, seed=seed,
    )


def bench_dqn_cartpole(
    *,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    seed: int = 42,
) -> BenchmarkResult:
    """DQN wall-clock + SPS on CartPole-v1."""
    from rlox import DQNTrainer
    return _bench_trainer(
        DQNTrainer, "dqn_cartpole",
        total_timesteps=total_timesteps, seed=seed,
    )


def bench_impala_cartpole(
    *,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    seed: int = 42,
) -> BenchmarkResult:
    """IMPALA wall-clock + SPS on CartPole-v1."""
    from rlox import IMPALATrainer
    return _bench_trainer(
        IMPALATrainer, "impala_cartpole",
        total_timesteps=total_timesteps, seed=seed,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    output_dir: str = "benchmark_results",
    *,
    seed: int = 42,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
) -> None:
    print("=" * 70)
    print("Benchmark: Algorithm Wall-Clock Performance (CartPole-v1)")
    print("=" * 70)

    all_results: list[dict] = []
    bench_kw = {"total_timesteps": total_timesteps, "seed": seed}

    benchmarks = [
        ("PPO", bench_ppo_cartpole),
        ("A2C", bench_a2c_cartpole),
        ("DQN", bench_dqn_cartpole),
        ("IMPALA", bench_impala_cartpole),
    ]

    for label, bench_fn in benchmarks:
        print(f"\n  {label}:")
        try:
            res = bench_fn(**bench_kw)
            print(f"    wall-clock: {res.metadata['wall_clock_s']:>8.2f} s")
            print(f"    SPS:        {res.metadata['sps']:>8.0f}")
            all_results.append(res.summary())
        except Exception as e:
            print(f"    [FAILED] {e}")

    print()
    path = write_report(all_results, [], output_dir)
    print(f"Report written to: {path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY -- Algorithm Performance")
    print("=" * 70)
    for r in all_results:
        name = r["name"]
        sps = r.get("sps", 0)
        wall = r.get("wall_clock_s", 0)
        print(f"  {name:<20s}  wall={wall:>7.2f}s  SPS={sps:>8.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox algorithm benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    args = parser.parse_args()
    run_all(
        args.output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
    )
