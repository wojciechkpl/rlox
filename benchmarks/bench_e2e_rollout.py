#!/usr/bin/env python3
"""
Benchmark: End-to-End Rollout Collection

Full pipeline: reset envs → step N times → store transitions → compute GAE.
Compares rlox vs SB3 vs TorchRL.

Usage:
    python benchmarks/bench_e2e_rollout.py [--output-dir benchmark_results]
                                           [--n-warmup 1] [--n-reps 10]
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, ComparisonResult, timed_run, write_report


# ---------------------------------------------------------------------------
# rlox end-to-end rollout
# ---------------------------------------------------------------------------

def bench_rlox_rollout(n_envs: int, n_steps: int, *, n_warmup: int, n_reps: int) -> BenchmarkResult:
    from rlox import VecEnv, ExperienceTable, compute_gae

    vec_env = VecEnv(n=n_envs, seed=42)
    total_transitions = n_envs * n_steps

    def rollout():
        table = ExperienceTable(obs_dim=4, act_dim=1)
        obs_batch = vec_env.reset_all(seed=42)
        all_rewards = []
        all_values = []
        all_dones = []

        for _ in range(n_steps):
            actions = [0] * n_envs  # dummy policy
            result = vec_env.step_all(actions)
            obs_array = result["obs"]
            rewards = result["rewards"]
            terminateds = result["terminated"]
            truncateds = result["truncated"]

            for j in range(n_envs):
                table.push(
                    obs=obs_array[j],
                    action=np.array([0.0], dtype=np.float32),
                    reward=float(rewards[j]),
                    terminated=bool(terminateds[j]),
                    truncated=bool(truncateds[j]),
                )

            all_rewards.extend(rewards)
            all_values.extend([0.0] * n_envs)  # dummy values
            all_dones.extend([bool(terminateds[j] or truncateds[j]) for j in range(n_envs)])

        # Compute GAE
        rewards_arr = np.array(all_rewards, dtype=np.float64)
        values_arr = np.array(all_values, dtype=np.float64)
        dones_arr = np.array(all_dones, dtype=np.float64)
        compute_gae(rewards_arr, values_arr, dones_arr, 0.0, 0.99, 0.95)

    times = timed_run(rollout, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"e2e_{n_envs}x{n_steps}", category="e2e_rollout",
        framework="rlox", times_ns=times,
        params={"n_envs": n_envs, "n_steps": n_steps, "n_items": total_transitions},
    )


# ---------------------------------------------------------------------------
# SB3 end-to-end rollout
# ---------------------------------------------------------------------------

def bench_sb3_rollout(n_envs: int, n_steps: int, *, n_warmup: int, n_reps: int) -> BenchmarkResult | None:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym
    except ImportError:
        print("  [skip] stable-baselines3 not installed")
        return None

    total_transitions = n_envs * n_steps

    def rollout():
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")] * n_envs)
        obs = env.reset()

        all_obs = []
        all_rewards = []
        all_dones = []

        for _ in range(n_steps):
            actions = np.zeros(n_envs, dtype=np.int64)
            obs, rewards, dones, infos = env.step(actions)
            all_obs.append(obs.copy())
            all_rewards.extend(rewards)
            all_dones.extend(dones)

        # GAE (Python loop, same as SB3 internals)
        rewards_arr = np.array(all_rewards, dtype=np.float64)
        values_arr = np.zeros_like(rewards_arr)
        dones_arr = np.array(all_dones, dtype=np.float64)

        # Python GAE loop (what SB3 does internally)
        n = len(rewards_arr)
        advantages = np.zeros(n)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = 0.0
            else:
                next_val = values_arr[t + 1]
            nonterminal = 1.0 - dones_arr[t]
            delta = rewards_arr[t] + 0.99 * next_val * nonterminal - values_arr[t]
            last_gae = delta + 0.99 * 0.95 * nonterminal * last_gae
            advantages[t] = last_gae

        env.close()

    times = timed_run(rollout, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"e2e_{n_envs}x{n_steps}", category="e2e_rollout",
        framework="sb3", times_ns=times,
        params={"n_envs": n_envs, "n_steps": n_steps, "n_items": total_transitions},
    )


# ---------------------------------------------------------------------------
# TorchRL end-to-end rollout
# ---------------------------------------------------------------------------

def bench_torchrl_rollout(n_envs: int, n_steps: int, *, n_warmup: int, n_reps: int) -> BenchmarkResult | None:
    try:
        import torch
        from torchrl.envs import GymEnv, SerialEnv
        from torchrl.objectives.value.functional import generalized_advantage_estimate
    except ImportError:
        print("  [skip] torchrl not installed")
        return None

    total_transitions = n_envs * n_steps

    def rollout():
        env = SerialEnv(n_envs, lambda: GymEnv("CartPole-v1", device="cpu"))
        td = env.reset()

        all_rewards = []
        all_dones = []

        for _ in range(n_steps):
            # Dummy action
            td.set("action", torch.zeros(n_envs, dtype=torch.long))
            td = env.step(td)
            next_td = td.get("next")
            all_rewards.append(next_td.get("reward").clone())
            all_dones.append(next_td.get("done").clone())
            td = next_td

        # GAE via TorchRL
        rewards_t = torch.stack(all_rewards).unsqueeze(-1).double()
        dones_t = torch.stack(all_dones).unsqueeze(-1)
        values_t = torch.zeros_like(rewards_t)
        next_values_t = torch.zeros_like(rewards_t)
        terminated_t = dones_t.clone()

        generalized_advantage_estimate(
            0.99, 0.95, values_t, next_values_t, rewards_t, dones_t, terminated_t,
        )
        env.close()

    times = timed_run(rollout, n_warmup=n_warmup, n_reps=n_reps)
    return BenchmarkResult(
        name=f"e2e_{n_envs}x{n_steps}", category="e2e_rollout",
        framework="torchrl", times_ns=times,
        params={"n_envs": n_envs, "n_steps": n_steps, "n_items": total_transitions},
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(output_dir: str = "benchmark_results", *, n_warmup: int = 1, n_reps: int = 10):
    print("=" * 70)
    print("Benchmark: End-to-End Rollout Collection")
    print("=" * 70)

    all_results = []
    all_comparisons = []

    configs = [
        (16, 128),
        (64, 512),
        (256, 2048),
    ]

    for n_envs, n_steps in configs:
        total = n_envs * n_steps
        print(f"\n  {n_envs} envs × {n_steps} steps = {total:,} transitions:")

        rlox_res = bench_rlox_rollout(n_envs, n_steps, n_warmup=n_warmup, n_reps=n_reps)
        tp = rlox_res.throughput
        print(f"    rlox:    {rlox_res.median_ns/1e6:>10.1f} ms  ({tp:>12,.0f} trans/s)")
        all_results.append(rlox_res.summary())

        sb3_res = bench_sb3_rollout(n_envs, n_steps, n_warmup=n_warmup, n_reps=n_reps)
        if sb3_res:
            tp_s = sb3_res.throughput
            print(f"    sb3:     {sb3_res.median_ns/1e6:>10.1f} ms  ({tp_s:>12,.0f} trans/s)")
            all_results.append(sb3_res.summary())
            comp = ComparisonResult(f"e2e_{n_envs}x{n_steps}", rlox_res, sb3_res, "sb3")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs sb3:     {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

        torchrl_res = bench_torchrl_rollout(n_envs, n_steps, n_warmup=n_warmup, n_reps=n_reps)
        if torchrl_res:
            tp_t = torchrl_res.throughput
            print(f"    torchrl: {torchrl_res.median_ns/1e6:>10.1f} ms  ({tp_t:>12,.0f} trans/s)")
            all_results.append(torchrl_res.summary())
            comp = ComparisonResult(f"e2e_{n_envs}x{n_steps}", rlox_res, torchrl_res, "torchrl")
            lo, hi = comp.speedup_ci_95
            print(f"    -> vs torchrl: {comp.speedup:.1f}x [{lo:.1f}, {hi:.1f}]")
            all_comparisons.append(comp.summary())

    print()
    path = write_report(all_results, all_comparisons, output_dir)
    print(f"Report written to: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — End-to-End Rollout")
    print("=" * 70)
    for c in all_comparisons:
        sig = "***" if c["significant"] else ""
        print(
            f"  {c['benchmark']:<25s} "
            f"rlox={c['rlox_median_ns']/1e6:>8.1f}ms  "
            f"{c['baseline_framework']:<10s}={c['baseline_median_ns']/1e6:>8.1f}ms  "
            f"speedup={c['speedup']:>5.1f}x [{c['speedup_ci_95_lo']:.1f},{c['speedup_ci_95_hi']:.1f}] {sig}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox end-to-end rollout benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-reps", type=int, default=10)
    args = parser.parse_args()
    run_all(args.output_dir, n_warmup=args.n_warmup, n_reps=args.n_reps)
