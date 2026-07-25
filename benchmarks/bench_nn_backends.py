#!/usr/bin/env python3
"""
Benchmark: Neural Network Backend Comparison

Measures PyTorch nn.Module forward/backward speed for the same architectures
used by rlox's Burn and Candle backends. Compare these results against the
Rust criterion benchmarks in crates/rlox-bench/benches/nn_backends.rs.

Usage:
    python benchmarks/bench_nn_backends.py [--output-dir benchmark_results]
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from harness import BenchmarkResult, ComparisonResult, timed_run, write_report

# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------

DEFAULT_N_WARMUP_INFERENCE = 50
DEFAULT_N_REPS_INFERENCE = 200
DEFAULT_N_WARMUP_TRAINING = 20
DEFAULT_N_REPS_TRAINING = 100


# ---------------------------------------------------------------------------
# PyTorch helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int, output_dim: int, hidden: int, activation="tanh"):
    import torch.nn as nn
    act = nn.Tanh if activation == "tanh" else nn.ReLU
    return nn.Sequential(
        nn.Linear(input_dim, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, output_dim),
    )


def _random_obs(batch: int, dim: int):
    import torch
    return torch.randn(batch, dim)


# ---------------------------------------------------------------------------
# ActorCritic (PPO) inference
# ---------------------------------------------------------------------------

def bench_pytorch_actor_critic_act(batch_size: int, n_reps: int = DEFAULT_N_REPS_INFERENCE) -> BenchmarkResult:
    import torch

    obs_dim, n_actions, hidden = 4, 2, 64
    actor = _make_mlp(obs_dim, n_actions, hidden, "tanh")
    critic = _make_mlp(obs_dim, 1, hidden, "tanh")
    obs = _random_obs(batch_size, obs_dim)

    def run():
        with torch.no_grad():
            logits = actor(obs)
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
            values = critic(obs).squeeze(-1)
        return actions, log_probs, values

    times = timed_run(run, n_warmup=DEFAULT_N_WARMUP_INFERENCE, n_reps=n_reps)
    return BenchmarkResult(
        name=f"actor_critic_act_{batch_size}", category="nn_backends",
        framework="pytorch", times_ns=times,
        params={"batch_size": batch_size, "obs_dim": obs_dim, "n_actions": n_actions},
    )


# ---------------------------------------------------------------------------
# PPO training step
# ---------------------------------------------------------------------------

def bench_pytorch_ppo_step(batch_size: int, n_reps: int = DEFAULT_N_REPS_TRAINING) -> BenchmarkResult:
    import torch
    import torch.nn as nn

    obs_dim, n_actions, hidden = 4, 2, 64
    actor = _make_mlp(obs_dim, n_actions, hidden, "tanh")
    critic = _make_mlp(obs_dim, 1, hidden, "tanh")
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4, eps=1e-5)

    obs = _random_obs(batch_size, obs_dim)
    actions = torch.zeros(batch_size, dtype=torch.long)
    old_log_probs = torch.full((batch_size,), -0.7)
    advantages = torch.ones(batch_size)
    returns = torch.ones(batch_size)
    old_values = torch.zeros(batch_size)
    clip_eps = 0.2

    def run():
        logits = actor(obs)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        new_lp = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        new_values = critic(obs).squeeze(-1)

        ratio = (new_lp - old_log_probs).exp()
        pg1 = -advantages * ratio
        pg2 = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = torch.max(pg1, pg2).mean()

        value_loss = 0.5 * ((new_values - returns) ** 2).mean()
        entropy_loss = entropy.mean()
        total = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(params, 0.5)
        optimizer.step()

    times = timed_run(run, n_warmup=DEFAULT_N_WARMUP_TRAINING, n_reps=n_reps)
    return BenchmarkResult(
        name=f"ppo_step_{batch_size}", category="nn_backends",
        framework="pytorch", times_ns=times,
        params={"batch_size": batch_size},
    )


# ---------------------------------------------------------------------------
# DQN Q-values (inference)
# ---------------------------------------------------------------------------

def bench_pytorch_dqn_q_values(batch_size: int, n_reps: int = DEFAULT_N_REPS_INFERENCE) -> BenchmarkResult:
    import torch

    obs_dim, n_actions, hidden = 4, 2, 64
    q_net = _make_mlp(obs_dim, n_actions, hidden, "relu")
    obs = _random_obs(batch_size, obs_dim)

    def run():
        with torch.no_grad():
            return q_net(obs)

    times = timed_run(run, n_warmup=DEFAULT_N_WARMUP_INFERENCE, n_reps=n_reps)
    return BenchmarkResult(
        name=f"dqn_q_values_{batch_size}", category="nn_backends",
        framework="pytorch", times_ns=times,
        params={"batch_size": batch_size},
    )


# ---------------------------------------------------------------------------
# DQN TD step (training)
# ---------------------------------------------------------------------------

def bench_pytorch_dqn_td_step(batch_size: int, n_reps: int = DEFAULT_N_REPS_TRAINING) -> BenchmarkResult:
    import torch
    import torch.nn.functional as F

    obs_dim, n_actions, hidden = 4, 2, 64
    q_net = _make_mlp(obs_dim, n_actions, hidden, "relu")
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)

    obs = _random_obs(batch_size, obs_dim)
    actions = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.ones(batch_size)

    def run():
        q_all = q_net(obs)
        q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    times = timed_run(run, n_warmup=DEFAULT_N_WARMUP_TRAINING, n_reps=n_reps)
    return BenchmarkResult(
        name=f"dqn_td_step_{batch_size}", category="nn_backends",
        framework="pytorch", times_ns=times,
        params={"batch_size": batch_size},
    )


# ---------------------------------------------------------------------------
# SAC stochastic policy (sample actions)
# ---------------------------------------------------------------------------

def bench_pytorch_sac_sample(batch_size: int, n_reps: int = DEFAULT_N_REPS_INFERENCE) -> BenchmarkResult:
    import torch
    import torch.nn as nn

    obs_dim, act_dim, hidden = 17, 6, 256

    class SquashedGaussian(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.mean_head = nn.Linear(hidden, act_dim)
            self.log_std_head = nn.Linear(hidden, act_dim)

        def forward(self, obs):
            h = self.shared(obs)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(-20, 2)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            x_t = dist.rsample()
            y_t = torch.tanh(x_t)
            log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            return y_t, log_prob

    policy = SquashedGaussian()
    obs = _random_obs(batch_size, obs_dim)

    def run():
        with torch.no_grad():
            return policy(obs)

    times = timed_run(run, n_warmup=DEFAULT_N_WARMUP_INFERENCE, n_reps=n_reps)
    return BenchmarkResult(
        name=f"sac_sample_{batch_size}", category="nn_backends",
        framework="pytorch", times_ns=times,
        params={"batch_size": batch_size},
    )


# ---------------------------------------------------------------------------
# TD3 deterministic policy (act)
# ---------------------------------------------------------------------------

def bench_pytorch_td3_act(batch_size: int, n_reps: int = DEFAULT_N_REPS_INFERENCE) -> BenchmarkResult:
    import torch
    import torch.nn as nn

    obs_dim, act_dim, hidden = 17, 6, 256
    actor = nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, act_dim), nn.Tanh(),
    )
    obs = _random_obs(batch_size, obs_dim)

    def run():
        with torch.no_grad():
            return actor(obs)

    times = timed_run(run, n_warmup=DEFAULT_N_WARMUP_INFERENCE, n_reps=n_reps)
    return BenchmarkResult(
        name=f"td3_act_{batch_size}", category="nn_backends",
        framework="pytorch", times_ns=times,
        params={"batch_size": batch_size},
    )


# ---------------------------------------------------------------------------
# Twin-Q critic (forward)
# ---------------------------------------------------------------------------

def bench_pytorch_twin_q(batch_size: int, n_reps: int = DEFAULT_N_REPS_INFERENCE) -> BenchmarkResult:
    import torch
    import torch.nn as nn

    obs_dim, act_dim, hidden = 17, 6, 256

    class TwinQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.q1 = nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            self.q2 = nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, obs, actions):
            x = torch.cat([obs, actions], dim=-1)
            return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    critic = TwinQ()
    obs = _random_obs(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)

    def run():
        with torch.no_grad():
            return critic(obs, actions)

    times = timed_run(run, n_warmup=DEFAULT_N_WARMUP_INFERENCE, n_reps=n_reps)
    return BenchmarkResult(
        name=f"twin_q_{batch_size}", category="nn_backends",
        framework="pytorch", times_ns=times,
        params={"batch_size": batch_size},
    )


# ---------------------------------------------------------------------------
# Twin-Q critic training step
# ---------------------------------------------------------------------------

def bench_pytorch_critic_step(batch_size: int, n_reps: int = DEFAULT_N_REPS_TRAINING) -> BenchmarkResult:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    obs_dim, act_dim, hidden = 17, 6, 256

    class TwinQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.q1 = nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            self.q2 = nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, obs, actions):
            x = torch.cat([obs, actions], dim=-1)
            return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    critic = TwinQ()
    optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    obs = _random_obs(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)
    targets = torch.ones(batch_size)

    def run():
        q1, q2 = critic(obs, actions)
        loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    times = timed_run(run, n_warmup=DEFAULT_N_WARMUP_TRAINING, n_reps=n_reps)
    return BenchmarkResult(
        name=f"critic_step_{batch_size}", category="nn_backends",
        framework="pytorch", times_ns=times,
        params={"batch_size": batch_size},
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    output_dir: str = "benchmark_results",
    n_reps_inference: int = DEFAULT_N_REPS_INFERENCE,
    n_reps_training: int = DEFAULT_N_REPS_TRAINING,
):
    try:
        import torch
    except ImportError:
        print("PyTorch not installed, skipping nn_backends benchmark")
        return

    print("=" * 70)
    print("rlox Benchmark: Neural Network Backends (PyTorch baseline)")
    print("=" * 70)
    print(f"PyTorch {torch.__version__}, CPU only")
    print()

    all_results = []
    all_comparisons = []

    batch_sizes_inference = [1, 32, 256]
    batch_sizes_training = [64, 256]

    # --- ActorCritic act ---
    print("ActorCritic (PPO) Inference")
    print("-" * 40)
    for bs in batch_sizes_inference:
        r = bench_pytorch_actor_critic_act(bs, n_reps=n_reps_inference)
        print(f"  batch={bs:>4d}:  {r.median_ns/1e3:>8.1f} us (IQR: {r.iqr_ns/1e3:.1f})")
        all_results.append(r.summary())
    print()

    # --- PPO step ---
    print("PPO Training Step")
    print("-" * 40)
    for bs in batch_sizes_training:
        r = bench_pytorch_ppo_step(bs, n_reps=n_reps_training)
        print(f"  batch={bs:>4d}:  {r.median_ns/1e3:>8.1f} us (IQR: {r.iqr_ns/1e3:.1f})")
        all_results.append(r.summary())
    print()

    # --- DQN q_values ---
    print("DQN Q-Values (Inference)")
    print("-" * 40)
    for bs in batch_sizes_inference:
        r = bench_pytorch_dqn_q_values(bs, n_reps=n_reps_inference)
        print(f"  batch={bs:>4d}:  {r.median_ns/1e3:>8.1f} us (IQR: {r.iqr_ns/1e3:.1f})")
        all_results.append(r.summary())
    print()

    # --- DQN td_step ---
    print("DQN TD Step (Training)")
    print("-" * 40)
    for bs in batch_sizes_training:
        r = bench_pytorch_dqn_td_step(bs, n_reps=n_reps_training)
        print(f"  batch={bs:>4d}:  {r.median_ns/1e3:>8.1f} us (IQR: {r.iqr_ns/1e3:.1f})")
        all_results.append(r.summary())
    print()

    # --- SAC sample ---
    print("SAC Sample Actions (Inference)")
    print("-" * 40)
    for bs in batch_sizes_inference:
        r = bench_pytorch_sac_sample(bs, n_reps=n_reps_inference)
        print(f"  batch={bs:>4d}:  {r.median_ns/1e3:>8.1f} us (IQR: {r.iqr_ns/1e3:.1f})")
        all_results.append(r.summary())
    print()

    # --- TD3 act ---
    print("TD3 Deterministic Action (Inference)")
    print("-" * 40)
    for bs in batch_sizes_inference:
        r = bench_pytorch_td3_act(bs, n_reps=n_reps_inference)
        print(f"  batch={bs:>4d}:  {r.median_ns/1e3:>8.1f} us (IQR: {r.iqr_ns/1e3:.1f})")
        all_results.append(r.summary())
    print()

    # --- Twin-Q forward ---
    print("Twin-Q Forward (Inference)")
    print("-" * 40)
    for bs in batch_sizes_inference:
        r = bench_pytorch_twin_q(bs, n_reps=n_reps_inference)
        print(f"  batch={bs:>4d}:  {r.median_ns/1e3:>8.1f} us (IQR: {r.iqr_ns/1e3:.1f})")
        all_results.append(r.summary())
    print()

    # --- Critic step ---
    print("Twin-Q Critic Step (Training)")
    print("-" * 40)
    for bs in batch_sizes_training:
        r = bench_pytorch_critic_step(bs, n_reps=n_reps_training)
        print(f"  batch={bs:>4d}:  {r.median_ns/1e3:>8.1f} us (IQR: {r.iqr_ns/1e3:.1f})")
        all_results.append(r.summary())
    print()

    # --- Write report ---
    path = write_report(all_results, all_comparisons, output_dir)
    print(f"Report written to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rlox nn backend benchmarks (PyTorch baseline)")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--n-reps-inference", type=int, default=DEFAULT_N_REPS_INFERENCE,
                        help="repetitions per inference benchmark (default: %(default)s)")
    parser.add_argument("--n-reps-training", type=int, default=DEFAULT_N_REPS_TRAINING,
                        help="repetitions per training benchmark (default: %(default)s)")
    args = parser.parse_args()
    run_all(args.output_dir, n_reps_inference=args.n_reps_inference,
            n_reps_training=args.n_reps_training)
