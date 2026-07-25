# Getting Started with rlox

This tutorial walks you through installing rlox, training your first agent, and understanding the architecture so you can build on it.

## Prerequisites

- Python 3.10+ (tested on 3.10, 3.11, 3.12, 3.13)
- Rust toolchain (`rustup` — install from https://rustup.rs)
- macOS (ARM/Intel), Linux, or Windows

## Installation

```bash
# Clone the repository
git clone https://github.com/wojciechkpl/rlox.git
cd rlox

# Create a Python virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install build dependencies
pip install maturin numpy gymnasium torch

# Build the Rust extension and install in development mode
maturin develop --release

# Verify the installation
python -c "from rlox import CartPole; print('rlox ready')"
```

> **Tip**: Always use `--release` with maturin. Debug builds are 10-50x slower and will make benchmarks meaningless.

## Step 1: Your First CartPole Agent

The fastest way to train an RL agent with rlox:

```python
from rlox import Trainer

# Train PPO on CartPole-v1 with default hyperparameters
trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)

print(f"Mean reward: {metrics['mean_reward']:.1f}")
# Expected: ~400-500 (CartPole max is 500)
```

That's it — 3 lines to a trained agent. Under the hood, rlox uses:
- Rust `VecEnv` for parallel environment stepping (8 envs by default)
- Rust `compute_gae` for advantage estimation (140x faster than Python)
- PyTorch for policy network and SGD updates

## Step 1b: Evaluating and Watching Your Agent

After training, use `evaluate()` to get episode statistics and `enjoy()` to watch the trained policy:

```python
# Evaluate over 10 episodes (deterministic policy)
results = trainer.evaluate(n_episodes=10, seed=0)
print(f"Mean reward: {results['mean_reward']:.1f} +/- {results['std_reward']:.1f}")
print(f"Min: {results['min_reward']:.1f}, Max: {results['max_reward']:.1f}")
print(f"Mean length: {results['mean_length']:.1f}")

# Watch the agent play (opens a render window)
trainer.enjoy(n_episodes=1, seed=0)
```

### Recording Videos

Use `VideoRecordingCallback` to save mp4 recordings during training:

```python
from rlox import Trainer
from rlox.callbacks import VideoRecordingCallback

trainer = Trainer("ppo",
    env="CartPole-v1",
    callbacks=[VideoRecordingCallback(
        video_folder="videos",
        record_freq=50_000,  # record every 50K steps
        n_episodes=1,
    )],
    seed=42,
)
metrics = trainer.train(total_timesteps=200_000)
# Videos saved to videos/ directory
```

## Step 2: Understanding the Architecture

rlox follows the **Polars pattern** — a Rust data plane for heavy computation, with Python in control:

```
┌─────────────────────────────────────────┐
│  Python (control plane)                 │
│  Training loops, policies (PyTorch),    │
│  config, logging, callbacks             │
├────────── PyO3 boundary ────────────────┤
│  Rust (data plane)                      │
│  Environments, parallel stepping,       │
│  buffers, GAE, GRPO, KL divergence      │
└─────────────────────────────────────────┘
```

**Why this split?** RL training has two bottlenecks:
1. **Data collection** — stepping environments and storing transitions. This is embarrassingly parallel and benefits enormously from Rust's zero-cost abstractions.
2. **Gradient computation** — neural network forward/backward passes. PyTorch/CUDA already handles this well.

rlox accelerates (1) while leaving (2) to PyTorch.

## Step 3: Using the Low-Level API

The trainer is convenient, but the component API gives you full control:

```python
import rlox
from rlox import RolloutCollector, PPOLoss, RolloutBatch
from rlox.policies import DiscretePolicy
import torch

# 1. Create a policy network
policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=64)
optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4)

# 2. Create a rollout collector (Rust VecEnv + GAE)
collector = RolloutCollector(
    env_id="CartPole-v1",
    n_envs=8,
    seed=0,
    gamma=0.99,
    gae_lambda=0.95,
)

# 3. Create the PPO loss function
loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)

# 4. Training loop
for update in range(100):
    # Collect 128 steps from each of 8 envs → 1024 transitions
    batch = collector.collect(policy, n_steps=128)

    # PPO: multiple SGD passes over the same data
    for epoch in range(4):
        for mb in batch.sample_minibatches(batch_size=256):
            # Normalise advantages
            adv = (mb.advantages - mb.advantages.mean()) / (mb.advantages.std() + 1e-8)

            loss, metrics = loss_fn(
                policy, mb.obs, mb.actions, mb.log_probs,
                adv, mb.returns, mb.values,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    if update % 10 == 0:
        print(f"Update {update}: reward={batch.rewards.sum().item()/8:.1f}, "
              f"entropy={metrics['entropy']:.3f}")
```

## Step 4: Using Rust Primitives Directly

You can use rlox's Rust components independently:

### GAE Computation

```python
import numpy as np
import rlox

# Compute advantages for a single trajectory
rewards = np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float64)
values  = np.array([0.5, 0.6, 0.7, 0.3, 0.8], dtype=np.float64)
dones   = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

advantages, returns = rlox.compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    last_value=0.9,  # V(s_T+1) bootstrap
    gamma=0.99,
    lam=0.95,
)
```

### Replay Buffer (for off-policy algorithms)

```python
import numpy as np
import rlox

# Create a buffer for an environment with obs_dim=4, act_dim=1
buffer = rlox.ReplayBuffer(capacity=10_000, obs_dim=4, act_dim=1)

# Store transitions
for i in range(100):
    obs = np.random.randn(4).astype(np.float32)
    buffer.push(obs, action=0.5, reward=1.0, terminated=False, truncated=False)

# Sample a batch
batch = buffer.sample(batch_size=32, seed=0)
print(batch["obs"].shape)      # (32, 4)
print(batch["rewards"].shape)  # (32,)
```

### LLM Post-Training (GRPO)

```python
import numpy as np
import rlox

# Group-relative advantages for 16 completions of one prompt
rewards = np.random.randn(16).astype(np.float64)
advantages = rlox.compute_group_advantages(rewards)
# advantages = (rewards - mean) / std

# Token-level KL divergence for regularisation
log_p = np.random.randn(128).astype(np.float64)  # policy log-probs
log_q = np.random.randn(128).astype(np.float64)  # reference log-probs
kl = rlox.compute_token_kl(log_p, log_q)
```

## Step 5: Off-Policy Algorithms

For continuous control with SAC or TD3:

```python
from rlox import Trainer

trainer = Trainer("sac",
    env="Pendulum-v1",
    config={
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "train_freq": 1,        # env steps between gradient updates
        "gradient_steps": 1,    # SGD steps per update
        "ent_coef": "auto",     # learned entropy coefficient (or a float to pin it)
    },
    seed=42,
)
metrics = trainer.train(total_timesteps=20_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

For discrete control with DQN (with Rainbow extensions):

```python
from rlox import Trainer

trainer = Trainer("dqn",
    env="CartPole-v1",
    config={
        "double_dqn": True,
        "dueling": True,
        "learning_rate": 1e-4,
        "train_freq": 1,        # env steps between gradient updates
        "gradient_steps": 1,    # SGD steps per update
    },
    seed=42,
)
metrics = trainer.train(total_timesteps=50_000)
```

> **Tip**: For sparse-reward environments like MountainCar, set `train_freq=16, gradient_steps=8` to avoid over-training on uninformative early transitions.

## Step 6: Custom Environments

All off-policy algorithms (SAC, TD3, DQN) accept either a string env ID or a pre-constructed Gymnasium environment:

```python
import gymnasium as gym
from rlox.algorithms.sac import SAC

# Pass a custom env with modified parameters
env = gym.make("Pendulum-v1", g=5.0)
sac = SAC(env_id=env, learning_starts=1000)
sac.train(total_timesteps=50_000)
action = sac.predict(obs, deterministic=True)
```

For on-policy algorithms (PPO, A2C), rlox's native `VecEnv` supports CartPole. For other environments,
use Gymnasium directly with rlox's Rust primitives for the heavy lifting:

```python
import gymnasium
import numpy as np
import rlox

# Create a gymnasium vectorized environment
vec_env = gymnasium.vector.SyncVectorEnv(
    [lambda: gymnasium.make("Acrobot-v1") for _ in range(8)]
)
obs, _ = vec_env.reset(seed=42)

# Use rlox for storage and GAE
buffer = rlox.ExperienceTable(obs_dim=6, act_dim=1)

# Collect experience with gymnasium, compute advantages with rlox
rewards_list = np.array([1.0] * 128, dtype=np.float64)
values_list = np.array([0.5] * 128, dtype=np.float64)
dones_list = np.zeros(128, dtype=np.float64)
advantages, returns = rlox.compute_gae(
    rewards_list, values_list, dones_list,
    last_value=0.5, gamma=0.99, lam=0.95,
)
```

## Step 7: Logging and Callbacks

### Weights & Biases

```python
from rlox import Trainer
from rlox.logging import WandbLogger

logger = WandbLogger(project="rlox-experiments", name="ppo-cartpole")
trainer = Trainer("ppo", env="CartPole-v1", logger=logger)
trainer.train(total_timesteps=100_000)
```

### TensorBoard

```python
from rlox.logging import TensorBoardLogger

logger = TensorBoardLogger(log_dir="runs/ppo-cartpole")
trainer = Trainer("ppo", env="CartPole-v1", logger=logger)
trainer.train(total_timesteps=100_000)
# Then: tensorboard --logdir runs/
```

### Early Stopping

```python
from rlox.callbacks import EarlyStoppingCallback
from rlox import Trainer

trainer = Trainer("ppo", 
    env="CartPole-v1",
    callbacks=[EarlyStoppingCallback(patience=20)],
)
trainer.train(total_timesteps=100_000)
```

## Running Tests

```bash
# Rust unit tests
cargo test --package rlox-core

# Python integration tests (`tests/`, not `tests/python/` — the latter skips
# tests/agentic/ and the repo-hygiene guards)
.venv/bin/python -m pytest tests/ -q

# Every gate CI runs (recommended before pushing)
bash scripts/check-ci-local.sh
```

## API Reference Summary

### Rust Primitives (imported from `rlox`)

| Component | Purpose |
|-----------|---------|
| `CartPole` | Native CartPole-v1 environment |
| `VecEnv` | Parallel CartPole stepping (Rayon) |
| `GymEnv` | Gymnasium environment wrapper |
| `ExperienceTable` | Append-only columnar storage |
| `ReplayBuffer` | Ring buffer with uniform sampling |
| `PrioritizedReplayBuffer` | Sum-tree prioritised sampling |
| `VarLenStore` | Variable-length sequence storage |
| `compute_gae` | Generalised Advantage Estimation |
| `compute_vtrace` | V-trace (for IMPALA) |
| `compute_group_advantages` | GRPO group-relative advantages |
| `compute_token_kl` | Token-level KL divergence |
| `DPOPair` | DPO preference pair container |
| `RunningStats` | Online mean/variance (Welford) |
| `RunningStatsVec` | Per-dimension online mean/variance (for observation normalisation) |
| `EmaRunningStats` | Exponential moving average stats (non-stationary signals) |
| `CusumDetector` | Two-sided CUSUM change-point detection |
| `pack_sequences` | LLM sequence packing |
| `Pendulum` | Native Pendulum-v1 environment |
| `FrameStack` | Stack consecutive observation frames (visual RL) |
| `ImagePreprocess` | Resize, grayscale, normalize pixel observations |
| `AtariWrapper` | Standard Atari preprocessing pipeline |
| `DMControlWrapper` | DeepMind Control Suite environment wrapper |

### Python Layer (from `rlox.*`)

| Component | Module | Purpose |
|-----------|--------|---------|
| `Trainer("ppo", ...)` | `rlox` | Unified PPO trainer |
| `Trainer("sac", ...)` | `rlox` | Unified SAC trainer |
| `Trainer("dqn", ...)` | `rlox` | Unified DQN trainer |
| `A2CTrainer` | `rlox.trainers` | High-level A2C trainer |
| `TD3Trainer` | `rlox.trainers` | High-level TD3 trainer |
| `Trainer("mappo", ...)` | `rlox` | Unified multi-agent PPO trainer |
| `DreamerV3Trainer` | `rlox.trainers` | World-model-based trainer (DreamerV3) |
| `IMPALATrainer` | `rlox.trainers` | Distributed actor-learner trainer |
| `PPO`, `A2C` | `rlox.algorithms` | On-policy algorithms |
| `SAC`, `TD3`, `DQN` | `rlox.algorithms` | Off-policy algorithms (with `predict()`) |
| `GRPO`, `DPO`, `OnlineDPO` | `rlox.algorithms` | LLM post-training |
| `IMPALA`, `MAPPO`, `DreamerV3` | `rlox.algorithms` | Advanced algorithms |
| `BestOfN` | `rlox.algorithms` | Inference-time rejection sampling |
| `DiscretePolicy` | `rlox.policies` | Actor-critic for discrete actions |
| `AsymmetricPolicy` | `rlox.policies` | Asymmetric actor-critic (privileged critic) |
| `RolloutBatch` | `rlox.batch` | Flat tensor container |
| `RolloutCollector` | `rlox.collectors` | VecEnv + batched GAE collection |
| `PPOLoss` | `rlox.losses` | Clipped surrogate objective |
| `VecNormalize` | `rlox.wrappers` | Running observation/reward normalisation for VecEnv |
| `TrainingConfig` | `rlox.config` | Unified YAML/dataclass config for any algorithm |
| `train_from_config` | `rlox.runner` | Launch training from a `TrainingConfig` or YAML file |
| `PPOConfig`, `SACConfig`, `DQNConfig` | `rlox.config` | Validated configs |
| `ConsoleLogger`, `WandbLogger`, `TensorBoardLogger` | `rlox.logging` | Training loggers |
| `TerminalDashboard` | `rlox.dashboard` | Live terminal dashboard (Rich-based) |
| `HTMLReport` | `rlox.dashboard` | Static HTML training report |
| `MetricsCollector` | `rlox.dashboard` | In-memory metrics aggregation for dashboards |
| `ProgressBarCallback`, `TimingCallback` | `rlox.callbacks` | Progress & profiling |
| `Callback`, `EvalCallback` | `rlox.callbacks` | Training hooks |
| `VideoRecordingCallback` | `rlox.callbacks` | Record eval episodes to mp4 |
| `compile_policy` | `rlox.compile` | torch.compile integration |
| `MmapReplayBuffer` | `rlox` | Disk-spilling replay for large obs |
| `ENV_REGISTRY` | `rlox.plugins` | Plugin registry for custom environments |
| `BUFFER_REGISTRY` | `rlox.plugins` | Plugin registry for custom buffers |
| `REWARD_REGISTRY` | `rlox.plugins` | Plugin registry for custom reward functions |
| `discover_plugins` | `rlox.plugins` | Auto-discover plugins from installed packages |
| `ModelZoo` | `rlox.model_zoo` | Registry of pretrained agents |
| `FrameStack`, `AtariWrapper` | `rlox.wrappers.visual` | Visual RL observation wrappers |
| `LanguageWrapper` | `rlox.wrappers.language` | Language-grounded RL wrapper |
| `generate_dockerfile` | `rlox.deploy` | Generate Dockerfile for model serving |
| `generate_k8s_job` | `rlox.deploy` | Generate Kubernetes job manifest |

## Next Steps

- **[Python User Guide](python-guide.md)** — comprehensive API documentation
- **[Examples](examples.md)** — code examples for all algorithms and primitives
- **[LLM Post-Training](llm-post-training.md)** — DPO, GRPO, OnlineDPO guide
- **[Benchmark Results](benchmark/README.md)** — performance comparison vs SB3 and TRL
- **CLI**: Run `python -m rlox train --algo ppo --env CartPole-v1 --timesteps 100000`
