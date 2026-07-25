<p align="center">
  <img src="assets/rlox_logo.png" alt="rlox logo" width="500">
</p>

<h1 align="center">rlox</h1>

<p align="center">Rust-accelerated reinforcement learning — the Polars architecture pattern applied to RL.</p>

<p align="center">
  <a href="https://wojciechkpl.github.io/rlox/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-7c4dff.svg" alt="Documentation"></a>
  <a href="https://crates.io/crates/rlox-core"><img src="https://img.shields.io/crates/v/rlox-core.svg" alt="crates.io"></a>
  <a href="https://pypi.org/project/rlox/"><img src="https://img.shields.io/pypi/v/rlox.svg" alt="PyPI"></a>
  <a href="https://github.com/wojciechkpl/rlox/actions/workflows/ci.yml"><img src="https://github.com/wojciechkpl/rlox/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://deepwiki.com/wojciechkpl/rlox"><img src="https://img.shields.io/badge/DeepWiki-rlox-blue.svg" alt="DeepWiki"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License"></a>
  <a href="https://pepy.tech/project/rlox"><img src="https://static.pepy.tech/badge/rlox" alt="Downloads"></a>
</p>

<p align="center">
  <a href="https://wojciechkpl.github.io/rlox/"><b>Documentation</b></a> &nbsp;|&nbsp;
  <a href="https://wojciechkpl.github.io/rlox/python/learning-path/"><b>Learning Path</b></a> &nbsp;|&nbsp;
  <a href="https://wojciechkpl.github.io/rlox/python/rl-introduction/"><b>RL Introduction</b></a> &nbsp;|&nbsp;
  <a href="https://wojciechkpl.github.io/rlox/python/algorithms/"><b>22 Algorithms</b></a> &nbsp;|&nbsp;
  <a href="https://wojciechkpl.github.io/rlox/python/examples/"><b>Examples</b></a>
</p>

> **New to Reinforcement Learning?** Start with our [RL Introduction](https://wojciechkpl.github.io/rlox/python/rl-introduction/) to learn the fundamentals, then follow the [Learning Path](https://wojciechkpl.github.io/rlox/python/learning-path/) from beginner to production.

## Why rlox?

RL frameworks like Stable-Baselines3 and TorchRL do everything in Python — environment stepping, buffer storage, advantage computation. This works, but Python interpreter overhead becomes the bottleneck long before your GPU does.

rlox applies the **Polars architecture pattern** to RL: a **Rust data plane** handles the compute-heavy, latency-sensitive work (env stepping, buffers, GAE) while a **Python control plane** stays in charge of training logic, configs, and neural networks via PyTorch. The two connect through PyO3 with zero-copy where possible.

The result: **3-50x faster** than SB3/TorchRL on data-plane operations, with the same Python training API you're used to.

## Quick Start

### Prerequisites

- **[uv](https://docs.astral.sh/uv/)** -- fast Python package & environment manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Python 3.10-3.13** -- uv can install one for you: `uv python install 3.12`
- **Rust 1.75+** (only needed to build from source) -- install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Optional**: `uv pip install "gymnasium[mujoco]"` for MuJoCo environments
- **Optional**: `uv pip install pettingzoo` for multi-agent environments

### Installation

```bash
uv venv                # create a virtual environment (.venv)
uv pip install rlox
```

Or add rlox to a uv-managed project:

```bash
uv add rlox
```

Then activate the environment (`source .venv/bin/activate`) or prefix commands
with `uv run` (e.g. `uv run python -m rlox train --config config.yaml`).

Or build from source:

```bash
uv venv
uv pip install maturin numpy gymnasium torch
uv run maturin develop --release
```

**Train PPO on CartPole in 3 lines:**

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

**Train SAC on Pendulum:**

```python
from rlox import Trainer

trainer = Trainer("sac", env="Pendulum-v1", config={"learning_starts": 500})
metrics = trainer.train(total_timesteps=20_000)
```

> **Note:** Per-algorithm trainers (`PPOTrainer`, `SACTrainer`, etc.) are deprecated. Use the unified `Trainer("algo", ...)` API instead.

**Config-driven training (YAML):**

```bash
python -m rlox train --config config.yaml
```

```python
from rlox import TrainingConfig, train_from_config

config = TrainingConfig.from_yaml("config.yaml")
metrics = train_from_config(config)
```

**Use Rust primitives directly:**

```python
import rlox

# 140x faster GAE than Python loops
advantages, returns = rlox.compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95)

# 35x faster GRPO advantages
advantages = rlox.compute_batch_group_advantages(rewards, group_size=4)

# Parallel env stepping (2.7M steps/s at 512 envs)
env = rlox.VecEnv(n=256, seed=42, env_id="CartPole-v1")
result = env.step_all(actions)
```

> More examples in [`examples/`](examples/) — PPO, SAC, GRPO custom rewards, fast GAE, VecEnv throughput.

## Documentation

| Resource | Link |
|----------|------|
| **Full Documentation** | [wojciechkpl.github.io/rlox](https://wojciechkpl.github.io/rlox/) |
| Getting Started | [Tutorial](https://wojciechkpl.github.io/rlox/python/getting-started/) |
| Python API Guide | [User Guide](https://wojciechkpl.github.io/rlox/python/) |
| Examples | [Code Examples](https://wojciechkpl.github.io/rlox/python/examples/) |
| Rust API | [cargo doc](https://wojciechkpl.github.io/rlox/rust/rlox_core/) |
| Migrating from SB3 | [Migration Guide](https://wojciechkpl.github.io/rlox/python/tutorials/migration-sb3/) |
| API Reference | [Autodoc](https://wojciechkpl.github.io/rlox/python/api/) |

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Python (control plane)                          │
│  PPO, SAC, DQN, TD3, A2C, MAPPO, DreamerV3,     │
│  IMPALA, GRPO, DPO                               │
│  GymVecEnv, VecNormalize, callbacks,             │
│  YAML/TOML configs, trainers, checkpointing,     │
│  diagnostics dashboard                           │
│  vLLM/TGI/SGLang backends, multi-GPU (DDP)       │
├────────────── PyO3 boundary ─────────────────────┤
│  Rust (data plane)                               │
│  rlox-core:   envs (CartPole, Pendulum,           │
│               NonStationaryCartPole),            │
│               Rayon parallel stepping,           │
│               buffers (ring, mmap, priority),    │
│               GAE, V-trace, GRPO, pipeline,      │
│               EMA stats, CUSUM change detection  │
│  rlox-nn:     RL algorithm traits                │
│  rlox-burn:   Burn backend (NdArray)             │
│  rlox-candle: Candle backend (CPU)               │
│  rlox-python: PyO3 bindings                      │
└──────────────────────────────────────────────────┘
```

Multi-crate workspace ([crates.io](https://crates.io/crates/rlox-core)):
- **rlox-core** — pure Rust: environments, buffers (ring, mmap, priority), GAE, V-trace, GRPO, pipeline
- **rlox-nn** — RL algorithm traits (`ActorCritic`, `QFunction`, `StochasticPolicy`, etc.)
- **rlox-burn** — Burn `Autodiff<NdArray>` implementations
- **rlox-candle** — Candle CPU implementations
- **rlox-python** — PyO3 bindings exposing `rlox-core` to Python

> For a deep-dive into the architecture, module relationships, and API reference, see the [DeepWiki](https://deepwiki.com/wojciechkpl/rlox).

## Benchmark Highlights

Measured 2026-04-08 on Apple M3 Pro (ARM, 12-core), CPU only, against
stable-baselines3 2.7.1 and torchrl 0.11.1. All results statistically
significant (bootstrap 95% CI lower bound > 1.0).

| Component | vs NumPy / SB3 | vs TorchRL | Details |
|-----------|--------|------------|---------|
| GAE (32K steps) | **135x vs NumPy** | **1,588x** | [docs/benchmark/gae.md](docs/benchmark/gae.md) |
| Replay buffer push (obs_dim=4, 10K) | **4.6x vs SB3** | **70x** | [docs/benchmark/buffer-ops.md](docs/benchmark/buffer-ops.md) |
| Replay buffer sample (batch=32) | **9.7x vs SB3** | **9x** | [docs/benchmark/buffer-ops.md](docs/benchmark/buffer-ops.md) |
| E2E rollout (256×2048) | **3.1x vs SB3** | **42x** | [docs/benchmark/e2e-rollout.md](docs/benchmark/e2e-rollout.md) |
| GRPO advantages (64×8) | **41x vs NumPy** | **42x vs PyTorch** | [docs/benchmark/llm-ops.md](docs/benchmark/llm-ops.md) |
| Tokenwise KL (seq=128) | **5x vs NumPy** | **9x vs PyTorch** | [docs/benchmark/llm-ops.md](docs/benchmark/llm-ops.md) |

> **Full methodology, raw data, and reproducibility instructions**: [docs/benchmark/](docs/benchmark/)

### Performance

Key numbers at a glance (Apple M3 Pro, CPU only, median of 100–200
timed iterations):

| Operation | rlox | Baseline | Speedup |
|-----------|------|-----------|---------|
| GAE (32K steps) | 69 µs | 9,376 µs (NumPy loop) | **135x** |
| Replay buffer push (10K, obs_dim=4) | 3.2 ms | 14.7 ms (SB3) | **4.6x** |
| Replay buffer sample (batch=32) | 2.0 µs | 19.3 µs (SB3) | **9.7x** |
| E2E rollout (256×2048 trans) | 669 ms | 2,057 ms (SB3) | **3.1x** |
| GRPO advantages (64×8 groups) | 7.8 µs | 318 µs (NumPy) | **41x** |
| Tokenwise KL (seq=128) | 0.3 µs | 3.0 µs (PyTorch) | **9x** |

### Convergence (rlox vs SB3)

Same hyperparameters (rl-zoo3 defaults), same evaluation harness, 5 seeds per cell, IQM + bootstrap 95% CI per Agarwal et al. 2021. 10 cells, convergence parity on every one.

| Algo | Environment | rlox IQM | rlox CI | SB3 IQM | SB3 CI |
|---|---|---:|---|---:|---|
| PPO | CartPole-v1 | 450.8 | [440.5, 454.2] | 438.2 | [389.7, 500.0] |
| PPO | Acrobot-v1 | -86.0 | [-89.7, -83.0] | -83.7 | [-97.0, -77.4] |
| PPO | Hopper-v4 | 932.8 | [706.0, 2190.4] | 1173.1 | [719.4, 1578.8] |
| PPO | HalfCheetah-v4 | 1854.6 | [1381.3, 2598.8] | 1568.7 | [1516.9, 3094.3] |
| SAC | Pendulum-v1 | -152.1 | [-173.9, -129.5] | — | — |
| SAC | HalfCheetah-v4 | 10871.9 | [10294.9, 11293.1] | 10795.5 | [10499.7, 11542.2] |
| TD3 | Pendulum-v1 | -149.1 | [-171.7, -134.2] | — | — |
| TD3 | HalfCheetah-v4 | 10880.1 | [7584.4, 11299.1] | — | — |
| DQN | CartPole-v1 | 500.0 | [195.8, 500.0] | 500.0 | [217.6, 500.0] |
| A2C | CartPole-v1 | 417.8 | [82.5, 500.0] | 491.6 | [167.5, 500.0] |

SAC HalfCheetah: rlox 10872 vs SB3 10796 — statistically identical, both beat the zoo reference (9656) by ~12%. TD3 HalfCheetah: rlox 10880 beats the zoo reference (9709) by 12%. DQN CartPole: both frameworks hit 500 (perfect). A2C CartPole: rlox 418 vs SB3 492, CIs overlap. The PPO MuJoCo "gap" vs zoo references is a protocol-and-version artifact (v4 vs v3, different eval protocol), not a framework deficit — both rlox and SB3 show the same gap when measured in the same harness.

![SPS Comparison](docs/benchmark/convergence/sps_comparison.png)

> Full convergence results, learning curves, and performance profiles: [docs/benchmark/convergence-results.md](docs/benchmark/convergence-results.md)

## Features

- **22 Algorithms**: PPO, SAC, DQN, TD3, A2C, VPG, TRPO, MAPPO, DreamerV3, IMPALA, and more (+ GRPO, DPO for LLM)
- **Trainers**: Each algorithm has a high-level `Trainer` with `train()`, `save()`, `from_checkpoint()`, `predict()`
- **Environments**: Gymnasium-compatible, Rayon-parallel VecEnv, CartPole and Pendulum-v1 built-in
- **Visual RL wrappers**: `FrameStack`, `ImagePreprocess`, `AtariWrapper`, `DMControlWrapper` for pixel-based RL
- **Language RL wrappers**: `LanguageWrapper`, `GoalConditionedWrapper` for language-grounded tasks
- **Plugin ecosystem**: `ENV_REGISTRY`, `BUFFER_REGISTRY`, `REWARD_REGISTRY`, `discover_plugins` for extensibility
- **Model zoo**: `ModelZoo.register`, `ModelZoo.load` for sharing and reusing pretrained agents
- **VecNormalize**: Obs/reward normalization at the environment boundary (SB3-compatible)
- **Buffers**: ring, mmap, priority replay — all in Rust with zero-copy Python access
- **Config-driven training**: YAML/TOML configs via `TrainingConfig` and `python -m rlox train --config config.yaml`
- **Diagnostics dashboard**: `TerminalDashboard`, `HTMLReport`, entropy/KL/gradient monitoring
- **LLM post-training**: GRPO, DPO, token KL, sequence packing, vLLM/TGI/SGLang backends
- **Cloud deploy**: Dockerfile generator, Kubernetes manifest generator, SageMaker integration
- **Distributed**: pipeline parallelism (crossbeam), gRPC workers, multi-GPU (DDP)
- **Evaluation**: `Trainer.evaluate()`, `Trainer.enjoy()`, `VideoRecordingCallback`, score normalization
- **Non-stationary RL**: EMA running stats, CUSUM change-point detection, sliding window replay, dynamic regret metrics
- **Asymmetric actor-critic**: `AsymmetricPolicy` for privileged critic observations (sim-to-real)
- **Production**: callbacks, checkpointing, eval toolkit (IQM, bootstrap CI, performance profiles)
- **NN backends**: Burn (NdArray) and Candle (CPU) for pure-Rust inference, PyTorch for training
- **Agentic-RL sandbox benchmark**: Hard-isolated sandbox (`rlox-sandbox`) for executing untrusted agent code with zero contagion. Validated on GRPO post-training: 30-step sweep shows Treatment (sandbox) survives 3/3 runs vs Baseline 1/3 at 10% adversarial injection, with quality-parity guardrail (reward 0.917 = 0.917). Includes reward-integrity protocol, seccomp+cgroup+namespace containment, and adversarial corpus regression tests.
- **~670 Rust tests, ~1100+ Python tests** — comprehensive coverage across core, sandbox, and the agentic harness

## Tutorials & Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first training run, basic API |
| [Agentic-RL Sandbox Benchmark](docs/tutorials/agentic-sandbox-benchmark.md) | Hard-isolation sandbox for untrusted agent code, P3 containment study, running the sweep |
| [Custom Rewards & Training Loops](docs/tutorials/custom-rewards-and-training-loops.md) | Reward shaping, GRPO reward functions, custom algorithms |
| [Python Guide](docs/python-guide.md) | Python API reference and patterns |
| [Rust Guide](docs/rust-guide.md) | Rust crate architecture and extending in Rust |
| [Math Reference](docs/math-reference.md) | GAE, V-trace, GRPO, DPO derivations |
| [Benchmark Details](docs/benchmark/) | Full methodology, per-benchmark analysis, reproducibility |
| [DeepWiki](https://deepwiki.com/wojciechkpl/rlox) | Auto-generated architecture docs and API reference |

## Running Tests

```bash
# Everything CI checks, in one command (recommended before pushing).
# Scope with `rust` / `python`; add WK=1 to include the Linux sandbox suite.
bash scripts/check-ci-local.sh

# Rust tests across all crates. --no-fail-fast matters: without it cargo stops at
# the first failing test *binary* and later binaries' failures stay hidden.
# On macOS add `--exclude rlox-sandbox` — that crate is Linux-only.
cargo test --workspace --no-fail-fast

# Python tests (build the extension into the venv, then run pytest).
# Run `tests/` — `tests/python/` alone skips tests/agentic/ and the hygiene guards.
uv run maturin develop --release
uv run pytest tests/ -q

# Quick smoke test (skip slow tests)
uv run pytest tests/ -m "not slow" -q

# Single crate
cargo test --package rlox-core
cargo test --package rlox-sandbox   # Linux only

# Fast inner loop (rlox-core + pytest; not a CI-parity check)
./scripts/test.sh

# Agentic-RL benchmark (single-GPU GRPO on Linux with cgroup v2)
bash benchmarks/agentic/repro.sh --setup-only

# Full benchmark suite (rlox vs TorchRL vs SB3)
.venv/bin/python benchmarks/run_all.py
```

## Project Layout

```
crates/
  rlox-core/       Pure Rust: envs, buffers (ring, mmap, priority), GAE,
                   V-trace, GRPO, pipeline (crossbeam), sequence packing
  rlox-rl-ops/     Estimator-agnostic advantage + token-KL ops (GRPO, DAPO)
  rlox-sandbox/    Hard-isolation sandbox for untrusted code (agentic-benchmark)
  rlox-nn/         RL algorithm traits (ActorCritic, QFunction, etc.)
  rlox-burn/       Burn backend (Autodiff<NdArray>)
  rlox-candle/     Candle backend (CPU)
  rlox-python/     PyO3 bindings
  rlox-bench/      Criterion benchmarks (env stepping, NN backends)
python/rlox/
  algorithms/      PPO, SAC, DQN, TD3, A2C, GRPO, DPO, MAPPO, DreamerV3, IMPALA
  distributed/     Pipeline, vLLM/TGI/SGLang backends, multi-GPU (DDP)
  llm/             LLM environment, reward model serving
  *.py             Collectors, configs, callbacks, policies, trainers,
                   evaluation toolkit, diagnostics, checkpointing
benchmarks/
  convergence/     Multi-seed RL convergence suite (5 seeds per cell, IQM + CI)
  agentic/         GRPO agentic-RL validation benchmark (P3 containment study)
tests/python/      Python integration & benchmark TDD tests
docs/              Guides, tutorials, benchmark methodology
```

## Citation

If you use rlox in your research, please cite:

```bibtex
@software{kowalinski2026rlox,
  author       = {Kowalinski, Wojciech},
  title        = {rlox: Rust-Accelerated Reinforcement Learning},
  year         = {2026},
  url          = {https://github.com/wojciechkpl/rlox},
  version      = {1.0.0},
  license      = {MIT OR Apache-2.0}
}
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
