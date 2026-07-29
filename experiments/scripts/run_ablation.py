"""Ablation experiment: replace Rust components with Python one-by-one.

This is the key experiment for the paper. It measures the marginal
contribution of each Rust component to end-to-end training speed.

Setup:
1. Full Rust (baseline): rlox VecEnv + rlox buffer + rlox GAE
2. Python GAE only: rlox VecEnv + rlox buffer + numpy GAE
3. Python buffer only: rlox VecEnv + python buffer + rlox GAE
4. Python env only: gymnasium VecEnv + rlox buffer + rlox GAE
5. All Python (control): gymnasium VecEnv + python buffer + numpy GAE

For each config, run PPO on CartPole-v1 and Pendulum-v1, measure:
- Wall-clock time for N rollouts
- Steps per second
- Marginal slowdown vs full Rust

Usage:
    python experiments/scripts/run_ablation.py --n-rollouts 100 --seed 42
"""

from __future__ import annotations

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch


def numpy_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Pure Python/NumPy GAE for ablation comparison."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 1.0 - dones[t]
        next_value = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def run_ablation(n_rollouts: int, seed: int, output_dir: str):
    """Run the full ablation experiment."""
    import rlox
    from rlox.gym_vec_env import GymVecEnv
    from rlox.policies import DiscretePolicy

    env_id = "CartPole-v1"
    n_envs = 8
    n_steps = 128
    gamma = 0.99
    gae_lambda = 0.95

    policy = DiscretePolicy(obs_dim=4, n_actions=2)

    configs = {
        "full_rust": {"env": "rust", "buffer": "rust", "gae": "rust"},
        "python_gae": {"env": "rust", "buffer": "rust", "gae": "python"},
        "python_env": {"env": "python", "buffer": "rust", "gae": "rust"},
        "all_python": {"env": "python", "buffer": "rust", "gae": "python"},
    }

    results = {}

    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {name} ({config})")
        print(f"{'='*60}")

        # Create environment
        if config["env"] == "rust":
            env = rlox.VecEnv(n=n_envs, seed=seed, env_id=env_id)
        else:
            env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)

        obs = env.reset_all()

        start = time.perf_counter()

        for rollout in range(n_rollouts):
            all_rewards = []
            all_values = []
            all_dones = []

            for step in range(n_steps):
                obs_tensor = torch.as_tensor(
                    np.array(obs) if not isinstance(obs, np.ndarray) else obs,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    actions, _ = policy.get_action_and_logprob(obs_tensor)
                    values = policy.get_value(obs_tensor)

                if config["env"] == "rust":
                    actions_list = actions.cpu().numpy().astype(np.uint32).tolist()
                    step_result = env.step_all(actions_list)
                else:
                    actions_np = actions.cpu().numpy().astype(np.int64)
                    step_result = env.step_all(actions_np)

                all_rewards.append(step_result["rewards"].astype(np.float64))
                all_values.append(values.cpu().numpy().astype(np.float64))
                terminated = step_result["terminated"].astype(bool)
                truncated = step_result["truncated"].astype(bool)
                dones = (terminated | truncated).astype(np.float64)
                all_dones.append(dones)
                obs = step_result["obs"]

            # Compute GAE per environment
            for env_idx in range(n_envs):
                rewards_env = np.array([r[env_idx] for r in all_rewards])
                values_env = np.array([v[env_idx] for v in all_values])
                dones_env = np.array([d[env_idx] for d in all_dones])

                if config["gae"] == "rust":
                    rlox.compute_gae(
                        rewards_env, values_env, dones_env,
                        0.0, gamma, gae_lambda,
                    )
                else:
                    numpy_gae(
                        rewards_env, values_env, dones_env,
                        0.0, gamma, gae_lambda,
                    )

        elapsed = time.perf_counter() - start
        total_steps = n_rollouts * n_envs * n_steps
        sps = total_steps / elapsed

        results[name] = {
            "elapsed": elapsed,
            "total_steps": total_steps,
            "sps": sps,
            "config": config,
        }
        print(f"  Time: {elapsed:.2f}s | SPS: {sps:,.0f}")

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    baseline_sps = results["full_rust"]["sps"]
    for name, r in results.items():
        slowdown = baseline_sps / r["sps"]
        print(f"  {name:20s}  SPS={r['sps']:>10,.0f}  slowdown={slowdown:.2f}x")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path / 'ablation_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/raw")
    args = parser.parse_args()
    run_ablation(args.n_rollouts, args.seed, args.output_dir)
