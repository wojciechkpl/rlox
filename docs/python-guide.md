# rlox Python User Guide

rlox provides three levels of API for reinforcement learning in Python. Each level gives more control at the cost of more code.

| Level | What you write | What rlox handles |
|-------|----------------|-------------------|
| **High-level** (Trainer) | `Trainer("ppo", env="CartPole-v1").train(50_000)` | Everything |
| **Mid-level** (Algorithm) | Training loop, hyperparams | Network creation, collection, loss |
| **Low-level** (Primitives) | Full loop, custom networks | Fast env stepping, GAE, buffers |

---

## Installation

```bash
# Prerequisites: Python 3.10+, Rust toolchain
python3 -m venv .venv
source .venv/bin/activate

pip install maturin numpy gymnasium torch

# Build the Rust extension
maturin develop --release

# Verify
python -c "from rlox import CartPole; print('rlox ready')"
```

> Always use `--release` with maturin. Debug builds are 10-50x slower.

### CLI Quick Start

```bash
# Train from the command line (no Python script needed)
python -m rlox train --algo ppo --env CartPole-v1 --timesteps 100000
python -m rlox train --algo sac --env Pendulum-v1 --timesteps 50000 --save model.pt
python -m rlox eval --algo ppo --checkpoint model.pt --env CartPole-v1 --episodes 10
```

---

## High-Level: Trainer API

Three lines to a trained agent:

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

### Available Trainers

```python
from rlox import Trainer
```

**Trainer("ppo", ...)** -- On-policy, discrete or continuous actions.

```python
trainer = Trainer("ppo", 
    env="CartPole-v1",
    config={"n_envs": 16, "n_steps": 256, "learning_rate": 3e-4},
    seed=42,
)
```

**Trainer("sac", ...)** -- Off-policy, continuous actions (e.g. Pendulum, MuJoCo).

```python
trainer = Trainer("sac", 
    env="Pendulum-v1",
    config={"learning_rate": 3e-4, "buffer_size": 100_000},
    seed=42,
)
```

**Trainer("dqn", ...)** -- Off-policy, discrete actions with Rainbow extensions.

```python
trainer = Trainer("dqn", 
    env="CartPole-v1",
    config={"double_dqn": True, "dueling": True},
    seed=42,
)
```

**Trainer("a2c", ...)** -- On-policy, single gradient step per rollout.

```python
trainer = Trainer("a2c",
    env="CartPole-v1",
    config={"n_envs": 8, "learning_rate": 7e-4},
    seed=42,
)
```

**Trainer("td3", ...)** -- Off-policy, continuous actions with delayed policy updates.

```python
trainer = Trainer("td3",
    env="Pendulum-v1",
    config={"policy_delay": 2, "target_noise": 0.2},
    seed=42,
)
```

**Trainer("mappo", ...)** -- Multi-agent PPO with centralised critic and per-agent actors.

```python
from rlox import Trainer

trainer = MATrainer("ppo", 
    env="spread_v3",   # PettingZoo environment
    n_agents=3,
    seed=42,
)
metrics = trainer.train(total_timesteps=500_000)
```

**Trainer("dreamer", ...)** -- World-model-based training (learns a latent dynamics model, trains the policy inside the learned world model).

```python
from rlox import Trainer

trainer = DreamerV3Trainer(
    env="Pendulum-v1",
    seed=42,
)
metrics = trainer.train(total_timesteps=200_000)
```

**Trainer("impala", ...)** -- Distributed actor-learner architecture with V-trace off-policy correction. Scales to many actors across machines via gRPC.

```python
from rlox import Trainer

trainer = IMPALATrainer(
    env="CartPole-v1",
    n_actors=8,
    seed=42,
)
metrics = trainer.train(total_timesteps=1_000_000)
```

### Callbacks

```python
from rlox.callbacks import (
    EarlyStoppingCallback,
    ProgressBarCallback,
    TimingCallback,
)

trainer = Trainer("ppo", 
    env="CartPole-v1",
    callbacks=[
        EarlyStoppingCallback(patience=20, min_delta=1.0),
        ProgressBarCallback(),   # tqdm progress bar
        TimingCallback(),         # phase-level profiling
    ],
)
metrics = trainer.train(total_timesteps=100_000)

# After training, see where time was spent
timing = trainer.callbacks[2]  # TimingCallback
print(timing.summary())
# {'env_step': 42.1, 'gae_compute': 8.3, 'gradient_update': 49.6}
```

| Callback | Purpose |
|----------|---------|
| `EarlyStoppingCallback` | Stop when reward plateaus for `patience` steps |
| `ProgressBarCallback` | tqdm progress bar with live reward display |
| `TimingCallback` | Wall-clock profiling of each training phase |
| `EvalCallback` | Periodic evaluation on a separate environment |
| `CheckpointCallback` | Save model weights at regular intervals |
| `VideoRecordingCallback` | Record evaluation episodes to mp4 videos |
| `Callback` | Base class for custom callbacks |

### Logging

```python
from rlox.logging import ConsoleLogger, WandbLogger, TensorBoardLogger

# Simple console output (no dependencies)
logger = ConsoleLogger(log_interval=500)
# Prints: step=500 | SPS=1234 | reward=45.20

# Weights & Biases
logger = WandbLogger(project="rlox-experiments", name="ppo-cartpole")

# TensorBoard
logger = TensorBoardLogger(log_dir="runs/ppo-cartpole")

trainer = Trainer("ppo", env="CartPole-v1", logger=logger)
trainer.train(total_timesteps=100_000)
```

Extend `LoggerCallback` for custom logging backends (CSV, MLflow, etc.):

```python
from rlox.logging import LoggerCallback

class CSVLogger(LoggerCallback):
    def on_train_step(self, step, metrics):
        # Write metrics to CSV
        ...
```

---

## Mid-Level: Algorithm API

The algorithm classes give you control over the training loop while handling network creation and loss computation:

### On-Policy (PPO, A2C)

```python
from rlox.algorithms import PPO, A2C

# PPO with custom hyperparameters
ppo = PPO(
    env_id="CartPole-v1",
    n_envs=8,
    seed=42,
    n_steps=128,
    n_epochs=4,
    clip_eps=0.2,
    learning_rate=2.5e-4,
)
metrics = ppo.train(total_timesteps=50_000)
```

```python
# A2C: single gradient step per rollout, shorter n_steps
a2c = A2C(
    env_id="CartPole-v1",
    n_envs=8,
    n_steps=5,
    learning_rate=7e-4,
    gae_lambda=1.0,  # full Monte Carlo returns
)
metrics = a2c.train(total_timesteps=50_000)
```

Both PPO and A2C use:
- `rlox.VecEnv` for parallel environment stepping
- `rlox.compute_gae` for advantage computation
- `RolloutCollector` for the collect-then-compute pattern

### Off-Policy (SAC, TD3, DQN)

```python
from rlox.algorithms import SAC, TD3, DQN

# SAC with automatic entropy tuning
sac = SAC(
    env_id="Pendulum-v1",
    buffer_size=1_000_000,
    learning_rate=3e-4,
    tau=0.005,
    gamma=0.99,
    auto_entropy=True,
    train_freq=1,        # gradient updates every N env steps
    gradient_steps=1,    # SGD steps per update
    ent_coef="auto",     # learned alpha; pass a float to pin it
)
metrics = sac.train(total_timesteps=20_000)
```

```python
# TD3 with delayed policy updates
td3 = TD3(
    env_id="Pendulum-v1",
    policy_delay=2,
    target_noise=0.2,
    noise_clip=0.5,
    exploration_noise=0.1,
    train_freq=1,              # gradient updates every N env steps
    gradient_steps=1,          # SGD steps per update
    target_policy_noise=None,  # alias for target_noise (SB3 compat)
    target_noise_clip=None,    # alias for noise_clip (SB3 compat)
)
metrics = td3.train(total_timesteps=20_000)
```

```python
# DQN with Rainbow extensions
dqn = DQN(
    env_id="CartPole-v1",
    double_dqn=True,
    dueling=True,
    n_step=3,
    prioritized=True,
    alpha=0.6,
    beta_start=0.4,
    train_freq=1,        # gradient updates every N env steps
    gradient_steps=1,    # SGD steps per update
    max_grad_norm=10.0,  # gradient clipping (default: inf = no clipping)
)
metrics = dqn.train(total_timesteps=50_000)
```

Off-policy algorithms use `rlox.ReplayBuffer` (or `PrioritizedReplayBuffer`) for storage, with Gymnasium for environment stepping.

### Multi-Environment Collection

All off-policy algorithms support parallel data collection via `OffPolicyCollector`. Use `n_envs` for automatic setup, or inject a custom collector:

```python
# Automatic: pass n_envs to any off-policy algorithm
sac = SAC(env_id="Pendulum-v1", n_envs=4, learning_starts=5000)
sac.train(total_timesteps=100_000)  # 4x collection throughput

td3 = TD3(env_id="Pendulum-v1", n_envs=4, learning_starts=5000)
dqn = DQN(env_id="CartPole-v1", n_envs=8, learning_starts=1000)
```

```python
# Manual: create and inject your own collector
from rlox.off_policy_collector import OffPolicyCollector
from rlox.exploration import GaussianNoise

buf = rlox.ReplayBuffer(1_000_000, obs_dim=3, act_dim=1)
collector = OffPolicyCollector(
    env_id="Pendulum-v1",
    n_envs=4,
    buffer=buf,
    exploration=GaussianNoise(sigma=0.1),
)
sac = SAC(env_id="Pendulum-v1", buffer=buf, collector=collector)
sac.train(total_timesteps=100_000)
```

The collector uses `GymVecEnv` internally and batch-inserts transitions via `push_batch` for efficiency. When `n_envs=1` (default), algorithms use the original single-env loop with zero overhead.

### Offline RL (TD3+BC, IQL, CQL, BC)

Train from static datasets without environment interaction. All offline algorithms
use `OfflineDatasetBuffer` (Rust-accelerated) and extend `OfflineAlgorithm` base class.

```python
import rlox
from rlox.algorithms.td3_bc import TD3BC

# Load dataset (D4RL, Minari, or custom numpy arrays)
buf = rlox.OfflineDatasetBuffer(
    obs.ravel(), next_obs.ravel(), actions.ravel(),
    rewards, terminated, truncated, normalize=True,
)
print(buf.stats())  # {'n_transitions': ..., 'n_episodes': ..., 'mean_return': ...}

# TD3+BC: TD3 with behavioral cloning regularization
algo = TD3BC(dataset=buf, obs_dim=17, act_dim=6, alpha=2.5)
algo.train(n_gradient_steps=100_000)
```

```python
# IQL: Implicit Q-Learning (avoids OOD action queries)
from rlox.algorithms.iql import IQL
algo = IQL(dataset=buf, obs_dim=17, act_dim=6, expectile=0.7)
```

```python
# CQL: Conservative Q-Learning (penalizes OOD Q-values)
from rlox.algorithms.cql import CQL
algo = CQL(dataset=buf, obs_dim=17, act_dim=6, cql_alpha=5.0)
```

```python
# BC: Behavioral Cloning (supervised learning on demonstrations)
from rlox.algorithms.bc import BC
algo = BC(dataset=buf, obs_dim=17, act_dim=6)
```

### Candle Hybrid Collection

`HybridPPO` runs policy inference entirely in Rust using Candle — zero Python
dispatch overhead during data collection. Collection takes only ~27% of wall
time vs ~50-60% with standard PyTorch inference.

```python
from rlox.algorithms.hybrid_ppo import HybridPPO

ppo = HybridPPO(env_id="CartPole-v1", n_envs=16, hidden=64)
metrics = ppo.train(total_timesteps=100_000)
print(ppo.timing_summary())
# {'collection_pct': 27.0, 'training_pct': 73.0}
```

### Inference with `predict()`

All algorithms provide a `predict()` method for evaluation. This includes PPO, A2C, VPG, TRPO, SAC, TD3, DQN, IMPALA, MAPPO, and all other trainers:

```python
# On-policy (PPO, A2C, VPG, TRPO): returns numpy action
action = ppo.predict(obs, deterministic=True)

# SAC/TD3: returns numpy action array (scaled to env range)
action = sac.predict(obs, deterministic=True)

# DQN: returns int action
action = dqn.predict(obs)

# IMPALA/MAPPO: also support predict()
action = impala.predict(obs)
```

### Custom Environments

Pass a pre-constructed Gymnasium env instead of an ID string:

```python
import gymnasium as gym

env = gym.make("Pendulum-v1", g=5.0)  # custom gravity
sac = SAC(env_id=env, learning_starts=1000)
sac.train(total_timesteps=50_000)
```

### LLM Post-Training (GRPO, DPO)

```python
from rlox.algorithms import GRPO, DPO

# GRPO: group-relative policy optimization
grpo = GRPO(
    model=my_lm,
    ref_model=ref_lm,
    reward_fn=reward_function,
    group_size=4,
    kl_coef=0.1,
    max_new_tokens=64,
)
metrics = grpo.train_step(prompt_batch)
```

```python
# DPO: direct preference optimization
dpo = DPO(
    model=my_lm,
    ref_model=ref_lm,
    beta=0.1,
)
loss, metrics = dpo.compute_loss(prompt, chosen, rejected)
```

---

## Low-Level: Rust Primitives

Import Rust primitives directly from `rlox`:

```python
import rlox
```

### Environment Stepping

```python
# Single CartPole
env = rlox.CartPole(seed=42)
obs = env.reset()                # shape: (4,)
result = env.step(1)             # push right
obs, reward = result["obs"], result["reward"]

# Vectorized CartPole (Rayon parallel)
vec = rlox.VecEnv(n=64, seed=0)
obs = vec.reset_all()            # shape: (64, 4)
result = vec.step_all([1] * 64)
next_obs = result["obs"]         # shape: (64, 4)
rewards = result["rewards"]      # shape: (64,)
terminated = result["terminated"] # shape: (64,), bool
truncated = result["truncated"]   # shape: (64,), bool

# Gymnasium wrapper
import gymnasium
gym_env = gymnasium.make("Acrobot-v1")
wrapped = rlox.GymEnv(gym_env)
```

### GAE Computation

```python
import numpy as np
import rlox

rewards = np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float64)
values  = np.array([0.5, 0.6, 0.7, 0.3, 0.8], dtype=np.float64)
dones   = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

advantages, returns = rlox.compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    last_value=0.9,
    gamma=0.99,
    lam=0.95,
)
# advantages.shape == (5,), returns.shape == (5,)
# Invariant: returns == advantages + values

# Batched GAE: all environments in one call (Rayon-parallel)
rewards_flat = np.random.randn(8 * 2048)  # env-major: [env0_step0, env0_step1, ...]
values_flat = np.random.randn(8 * 2048)
dones_flat = np.zeros(8 * 2048)
last_vals = np.random.randn(8)

adv, ret = rlox.compute_gae_batched(
    rewards_flat, values_flat, dones_flat, last_vals,
    n_steps=2048, gamma=0.99, lam=0.95,
)

# f32 variant (1.5x faster at 64+ envs, avoids f64 conversion)
adv_f32, ret_f32 = rlox.compute_gae_batched_f32(
    rewards_flat.astype(np.float32), values_flat.astype(np.float32),
    dones_flat.astype(np.float32), last_vals.astype(np.float32),
    n_steps=2048, gamma=0.99, lam=0.95,
)
```

### V-trace

```python
log_rhos = np.array([0.2, -0.3, 0.8], dtype=np.float32)
rewards  = np.array([1.0, 2.0, 3.0], dtype=np.float32)
values   = np.array([0.5, 1.0, 1.5], dtype=np.float32)
dones    = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # episode boundaries

vs, pg_advantages = rlox.compute_vtrace(
    log_rhos=log_rhos,
    rewards=rewards,
    values=values,
    dones=dones,            # zeroes discount at episode boundaries
    bootstrap_value=2.0,
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
)
```

### Replay Buffers

```python
# Uniform replay buffer (zero-copy push via Rust push_slices)
buf = rlox.ReplayBuffer(capacity=100_000, obs_dim=4, act_dim=1)
obs = np.zeros(4, dtype=np.float32)
next_obs = np.ones(4, dtype=np.float32)
buf.push(obs, action=np.array([0.5], dtype=np.float32), reward=1.0,
         terminated=False, truncated=False, next_obs=next_obs)
batch = buf.sample(batch_size=32, seed=0)
# batch keys: "obs", "next_obs", "actions", "rewards", "terminated", "truncated"

# Prioritized replay buffer (O(1) min via augmented min-tree)
pbuf = rlox.PrioritizedReplayBuffer(
    capacity=100_000, obs_dim=4, act_dim=1, alpha=0.6, beta=0.4
)
pbuf.push(obs, action=np.array([0.5], dtype=np.float32), reward=1.0,
          terminated=False, truncated=False, next_obs=next_obs, priority=1.0)
batch = pbuf.sample(batch_size=32, seed=0)
# Additional keys: "weights" (IS weights), "indices" (for priority update)
pbuf.update_priorities(batch["indices"], new_td_errors)
pbuf.set_beta(0.7)  # anneal toward 1.0

# Memory-mapped buffer (for Atari-scale observations)
mmap_buf = rlox.MmapReplayBuffer(
    hot_capacity=10_000,       # kept in memory
    total_capacity=1_000_000,  # overflow spills to disk
    obs_dim=84*84*4,
    act_dim=1,
    cold_path="/tmp/replay_cold.bin",
)
# Same push/sample API as ReplayBuffer
```

### LLM Operations

```python
# GRPO group-relative advantages (single group)
rewards = np.random.randn(16).astype(np.float64)
advantages = rlox.compute_group_advantages(rewards)

# Batched GRPO (Rayon-parallel for large batches)
all_rewards = np.random.randn(1024 * 8).astype(np.float64)  # 1024 prompts x 8 completions
all_advantages = rlox.compute_batch_group_advantages(all_rewards, group_size=8)

# Token-level KL divergence (single sequence)
log_p = np.random.randn(128).astype(np.float64)
log_q = np.random.randn(128).astype(np.float64)
kl = rlox.compute_token_kl(log_p, log_q)

# Batched KL (single Rust call for all sequences, Rayon-parallel)
log_p_flat = np.random.randn(32 * 2048).astype(np.float32)
log_q_flat = np.random.randn(32 * 2048).astype(np.float32)
kl_per_seq = rlox.compute_batch_token_kl_schulman_f32(log_p_flat, log_q_flat, seq_len=2048)
# kl_per_seq: (32,) array — 2-9x faster than TRL

# DPO preference pair
pair = rlox.DPOPair(
    prompt_tokens=np.array([1, 2, 3], dtype=np.uint32),
    chosen_tokens=np.array([4, 5], dtype=np.uint32),
    rejected_tokens=np.array([6, 7, 8], dtype=np.uint32),
)

# Variable-length sequence storage
store = rlox.VarLenStore()
store.push(np.array([1, 2, 3], dtype=np.uint32))
store.push(np.array([4, 5], dtype=np.uint32))
seq = store.get(0)  # array([1, 2, 3])

# Sequence packing for transformers
packed = rlox.pack_sequences(
    sequences=[np.array([1,2,3], dtype=np.uint32),
               np.array([4,5], dtype=np.uint32)],
    max_length=8,
)
```

### RunningStats

```python
stats = rlox.RunningStats()
stats.batch_update(np.array([1.0, 2.0, 3.0]))
print(stats.mean())   # 2.0
print(stats.std())     # ~0.816
print(stats.count())   # 3
```

---

## Configuration

Typed configuration with validation, merging, and serialisation:

```python
from rlox.config import PPOConfig, SACConfig, DQNConfig

# Create with defaults (CleanRL-matching)
cfg = PPOConfig()

# Create from dict (ignores unknown keys)
cfg = PPOConfig.from_dict({"n_envs": 16, "clip_eps": 0.1, "unknown_key": 42})

# Merge overrides into existing config
cfg2 = cfg.merge({"learning_rate": 1e-3})

# Serialise for logging
d = cfg.to_dict()

# Validation happens in __post_init__
try:
    PPOConfig(learning_rate=-1)  # raises ValueError
except ValueError as e:
    print(e)
```

### PPOConfig Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_envs` | 8 | Parallel environments |
| `n_steps` | 128 | Rollout length per env |
| `n_epochs` | 4 | SGD passes per rollout |
| `batch_size` | 256 | Minibatch size |
| `learning_rate` | 2.5e-4 | Adam LR |
| `clip_eps` | 0.2 | PPO clip range |
| `vf_coef` | 0.5 | Value loss coefficient |
| `ent_coef` | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `normalize_advantages` | True | Per-minibatch normalisation |
| `clip_vloss` | True | Clipped value loss |
| `anneal_lr` | True | Linear LR annealing |

---

## Config-Driven Training

Define your entire experiment in a YAML file and launch with `train_from_config`:

```yaml
# experiment.yaml
algorithm: ppo
env: CartPole-v1
total_timesteps: 100_000
seed: 42
config:
  n_envs: 16
  learning_rate: 3e-4
  n_steps: 128
  n_epochs: 4
logger:
  type: wandb
  project: rlox-experiments
```

```python
from rlox.runner import train_from_config
from rlox.config import TrainingConfig

# From a YAML file
metrics = train_from_config("experiment.yaml")

# Or build programmatically
cfg = TrainingConfig(
    algorithm="ppo",
    env="CartPole-v1",
    total_timesteps=100_000,
    seed=42,
    config={"n_envs": 16, "learning_rate": 3e-4},
)
metrics = train_from_config(cfg)
```

---

## VecNormalize

`VecNormalize` wraps a vectorised environment to apply running normalisation to observations and rewards. It uses `RunningStatsVec` (Rust) for efficient per-dimension statistics.

```python
from rlox import Trainer
from rlox.wrappers import VecNormalize

trainer = Trainer("ppo", 
    env="CartPole-v1",
    wrappers=[VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)],
    seed=42,
)
metrics = trainer.train(total_timesteps=100_000)
```

`VecNormalize` is especially useful for environments with large or variable observation scales (MuJoCo, robotics).

---

## Diagnostics Dashboard

`MetricsCollector` aggregates training metrics in memory and feeds them to visualisation backends.

```python
from rlox.dashboard import MetricsCollector, HTMLReport, TerminalDashboard

# Collect metrics during training
collector = MetricsCollector()

from rlox import Trainer
trainer = Trainer("ppo", 
    env="CartPole-v1",
    callbacks=[collector],
    seed=42,
)
trainer.train(total_timesteps=50_000)

# Generate a static HTML report
report = HTMLReport(collector)
report.save("training_report.html")

# Or use the live terminal dashboard (Rich-based)
# Pass TerminalDashboard as a callback for real-time display:
from rlox import Trainer
trainer = Trainer("ppo", 
    env="CartPole-v1",
    callbacks=[TerminalDashboard()],
    seed=42,
)
trainer.train(total_timesteps=50_000)
```

---

## Custom Policies

### Discrete Actions (PPO/A2C)

```python
from rlox.policies import DiscretePolicy

policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=64)

# Required interface (called by PPOLoss / RolloutCollector):
actions, log_probs = policy.get_action_and_logprob(obs_tensor)
values = policy.get_value(obs_tensor)
log_probs, entropy = policy.get_logprob_and_entropy(obs_tensor, actions_tensor)
```

Architecture: separate actor and critic MLPs with orthogonal initialisation, Tanh activations, and reduced gain (0.01) on the policy head.

### Continuous Actions (SAC/TD3)

```python
from rlox.networks import SquashedGaussianPolicy, DeterministicPolicy, QNetwork

# SAC: squashed Gaussian policy
actor = SquashedGaussianPolicy(obs_dim=3, act_dim=1, hidden=256)
action, log_prob = actor.sample(obs_tensor)       # reparameterised
det_action = actor.deterministic(obs_tensor)       # mean through tanh

# TD3: deterministic policy
actor = DeterministicPolicy(obs_dim=3, act_dim=1, hidden=256, max_action=2.0)
action = actor(obs_tensor)  # scaled by max_action

# Shared Q-network for SAC/TD3
critic = QNetwork(obs_dim=3, act_dim=1, hidden=256)
q_value = critic(obs_tensor, action_tensor)  # scalar
```

### Discrete Q-Networks (DQN)

```python
from rlox.networks import SimpleQNetwork, DuelingQNetwork

# Standard DQN
q_net = SimpleQNetwork(obs_dim=4, act_dim=2, hidden=256)
q_values = q_net(obs_tensor)  # (B, n_actions)

# Dueling architecture: V(s) + A(s,a) - mean(A)
q_net = DuelingQNetwork(obs_dim=4, act_dim=2, hidden=256)
q_values = q_net(obs_tensor)  # same interface
```

---

## RolloutBatch and RolloutCollector

The collector orchestrates on-policy data collection:

```python
from rlox.collectors import RolloutCollector
from rlox.policies import DiscretePolicy

collector = RolloutCollector(
    env_id="CartPole-v1",
    n_envs=8,
    seed=0,
    gamma=0.99,
    gae_lambda=0.95,
    normalize_rewards=False,
    normalize_obs=False,
)

policy = DiscretePolicy(obs_dim=4, n_actions=2)
batch = collector.collect(policy, n_steps=128)

# batch is a RolloutBatch with shape (n_envs * n_steps, ...)
batch.obs.shape        # (1024, 4)
batch.actions.shape    # (1024,)
batch.advantages.shape # (1024,)
batch.returns.shape    # (1024,)
```

The collection pipeline:
1. Steps `n_envs` environments for `n_steps` using `rlox.VecEnv` or `GymVecEnv`
2. Evaluates the policy at each step (forward pass only)
3. Computes GAE using `rlox.compute_gae_batched` (single Rust call, Rayon-parallel)
4. Flattens and returns a `RolloutBatch`

### Minibatch Iteration

```python
for epoch in range(4):
    for mb in batch.sample_minibatches(batch_size=256, shuffle=True):
        # mb is a RolloutBatch with shape (256, ...)
        loss = compute_loss(mb)
        loss.backward()
```

---

## PPOLoss

Stateless loss calculator implementing the clipped surrogate objective:

```python
from rlox.losses import PPOLoss

loss_fn = PPOLoss(
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    clip_vloss=True,
)

total_loss, metrics = loss_fn(
    policy, obs, actions, old_log_probs,
    advantages, returns, old_values,
)
# metrics: policy_loss, value_loss, entropy, approx_kl, clip_fraction

total_loss.backward()
```

---

## Statistical Evaluation

Following Agarwal et al. (2021) for reliable deep RL evaluation:

```python
from rlox.evaluation import interquartile_mean, performance_profiles, stratified_bootstrap_ci

# IQM: robust central tendency (discards top/bottom 25%)
scores = [450, 480, 500, 200, 490]
iqm = interquartile_mean(scores)

# Bootstrap confidence interval
lower, upper = stratified_bootstrap_ci(scores, n_bootstrap=10_000, ci=0.95)

# Performance profiles: fraction of runs above threshold
profiles = performance_profiles(
    {"rlox": [450, 480, 500], "baseline": [300, 350, 400]},
    thresholds=[100, 200, 300, 400, 500],
)
```

---

## Evaluation & Rendering

After training, use `trainer.evaluate()` for deterministic evaluation and `trainer.enjoy()` for visual inspection:

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
trainer.train(total_timesteps=50_000)

# Deterministic evaluation over 10 episodes
results = trainer.evaluate(n_episodes=10, seed=0)
# Returns: {mean_reward, std_reward, min_reward, max_reward, mean_length, n_episodes}
print(f"Mean: {results['mean_reward']:.1f} +/- {results['std_reward']:.1f}")

# Watch the policy play (opens render window)
trainer.enjoy(n_episodes=1, seed=0)
```

### Video Recording

`VideoRecordingCallback` records evaluation episodes to mp4 at regular intervals during training:

```python
from rlox import Trainer
from rlox.callbacks import VideoRecordingCallback

trainer = Trainer("ppo",
    env="CartPole-v1",
    callbacks=[VideoRecordingCallback(
        video_folder="videos",    # output directory
        record_freq=50_000,       # record every 50K steps
        n_episodes=1,             # episodes per recording
    )],
    seed=42,
)
trainer.train(total_timesteps=200_000)
```

### Episode Statistics

`RolloutCollector` and `GymVecEnv` track completed episode statistics automatically:

```python
from rlox.collectors import RolloutCollector
from rlox.policies import DiscretePolicy

collector = RolloutCollector(env_id="CartPole-v1", n_envs=8, seed=0)
policy = DiscretePolicy(obs_dim=4, n_actions=2)
batch = collector.collect(policy, n_steps=128)

# Access completed episode stats
print(collector.episode_rewards)   # list[float]
print(collector.episode_lengths)   # list[int]
```

---

## Score Normalization

Normalize raw environment returns to [0, 1] using random and expert baselines (for cross-environment comparison following rliable conventions):

```python
from rlox.evaluation import normalize_score, normalize_scores, SCORE_BASELINES

# Single score
normalized = normalize_score(450.0, env_id="CartPole-v1")
# (450 - 22) / (500 - 22) = 0.895

# Batch of scores
import numpy as np
scores = [450, 480, 500, 200, 490]
normalized_arr = normalize_scores(scores, env_id="CartPole-v1")

# Custom baselines (for envs not in SCORE_BASELINES)
normalized = normalize_score(1500.0, env_id="MyCustomEnv-v0",
                             random_score=0.0, expert_score=2000.0)

# See all built-in baselines
print(sorted(SCORE_BASELINES.keys()))
# CartPole-v1, Acrobot-v1, Pendulum-v1, HalfCheetah-v4, Hopper-v4, ...
```

---

## AsymmetricPolicy

`AsymmetricPolicy` implements the asymmetric actor-critic pattern where the critic sees privileged state during training (e.g. ground-truth positions, velocities) while the actor only sees deployment-time observations (e.g. sensor readings). Common in sim-to-real transfer and Isaac Gym workflows.

```python
from rlox.policies import AsymmetricPolicy

# Discrete actions: actor sees 10-dim obs, critic sees 20-dim privileged state
policy = AsymmetricPolicy(
    obs_dim=10,
    critic_obs_dim=20,
    n_actions=4,
    hidden=64,
)

# Continuous actions: actor sees 10-dim obs, critic sees 30-dim privileged state
policy = AsymmetricPolicy(
    obs_dim=10,
    critic_obs_dim=30,
    act_dim=3,
    hidden=128,
)

# Interface (same as DiscretePolicy / ContinuousPolicy):
# actions, log_probs = policy.get_action_and_logprob(actor_obs)
# values = policy.get_value(critic_obs)  # uses privileged observations
# log_probs, entropy = policy.get_logprob_and_entropy(actor_obs, actions)
```

Exactly one of `n_actions` (discrete) or `act_dim` (continuous) must be provided.

---

## Non-Stationary RL Tools

rlox includes Rust-accelerated primitives for non-stationary RL settings where the environment dynamics change over time.

### EMA Running Stats

`EmaRunningStats` tracks exponential moving average statistics, giving more weight to recent observations. Use it instead of `RunningStats` when the underlying distribution drifts.

```python
from rlox._rlox_core import EmaRunningStats
import numpy as np

# Create with explicit smoothing factor (higher = more responsive)
ema = EmaRunningStats(alpha=0.1)

# Or from equivalent window size / half-life
ema = EmaRunningStats.from_window(20)       # alpha = 2/(20+1) ~ 0.095
ema = EmaRunningStats.from_halflife(10.0)   # weight decays 50% every 10 steps

# Update with observations
ema.update(1.0)
ema.update(2.0)
ema.batch_update(np.array([3.0, 4.0, 5.0]))

# Read current statistics
print(ema.mean)    # EMA mean
print(ema.std)     # EMA standard deviation
print(ema.var)     # EMA variance
print(ema.count)   # total observations seen

# Normalize a value using current EMA stats
normalized = ema.normalize(5.0)  # (5.0 - mean) / std
```

### CUSUM Change-Point Detection

`CusumDetector` implements two-sided CUSUM (cumulative sum) for detecting distributional shifts in a streaming signal. Feed it reward or loss values; it fires an alarm when the mean has shifted.

```python
from rlox._rlox_core import CusumDetector
import numpy as np

# Known reference level
detector = CusumDetector(mu_0=1.0, delta=0.5, h=5.0)
# mu_0: expected mean under no-change hypothesis
# delta: allowance (minimum shift to detect). Typical: 0.5 * expected_shift
# h: threshold (higher = fewer false alarms). Typical: 4-8

# Auto-estimate reference level from first N samples
detector = CusumDetector.with_burnin(burnin=100, delta=0.5, h=5.0)

# Feed observations one at a time
alarm = detector.update(1.2)   # returns True if change detected

# Or feed a batch (returns index of first alarm, or None)
values = np.array([1.0, 1.1, 0.9, 5.0, 5.2, 5.1])
alarm_idx = detector.batch_update(values)
if alarm_idx is not None:
    print(f"Change detected at index {alarm_idx}")

# Reset after handling an alarm
detector.reset()
detector.reset_with_burnin()   # re-estimate mu_0
detector.set_mu(2.0)           # set new reference manually
```

### Sliding Window Replay

`ReplayBuffer.sample_recent()` samples uniformly from only the most recent transitions, discarding stale experience from previous regimes:

```python
import numpy as np
import rlox

buffer = rlox.ReplayBuffer(capacity=100_000, obs_dim=4, act_dim=1)
# ... push transitions ...

# Sample from only the last 1000 transitions
batch = buffer.sample_recent(batch_size=32, window_size=1000, seed=0)
# Same keys as buffer.sample(): obs, next_obs, actions, rewards, terminated, truncated
```

### Dynamic Regret Metrics

Evaluation metrics for non-stationary settings (in `rlox.evaluation`):

```python
from rlox.evaluation import dynamic_regret, adaptation_latency, forgetting_ratio

# Dynamic regret: cumulative gap between agent and time-varying optimal
dr = dynamic_regret(
    agent_rewards=[1.0, 0.8, 0.5, 0.9, 1.0],
    optimal_rewards=[1.0, 1.0, 1.0, 1.0, 1.0],
)
# dr = 0.0 + 0.2 + 0.5 + 0.1 + 0.0 = 0.8

# Adaptation latency: steps to recover after a change-point
latency = adaptation_latency(
    rewards=[1.0]*50 + [0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95],
    change_point=50,
    pre_change_mean=1.0,
    recovery_fraction=0.9,  # recover to 90% of pre-change performance
)
# returns number of steps after change_point, or None if never recovered

# Forgetting ratio: performance retained on old task after training on new one
fr = forgetting_ratio(
    reward_on_task_before=100.0,
    reward_on_task_after=85.0,
)
# fr = 0.85 (retained 85% performance). 1.0 = no forgetting
```

### NonStationaryCartPole (Rust only)

A CartPole variant with configurable parameter drift, available in the Rust crate `rlox-core`. Physical parameters (gravity, pole length, cart mass, force magnitude) can drift independently via `DriftMode::None`, `DriftMode::Linear`, `DriftMode::Sinusoidal`, or `DriftMode::Step`. Not yet exposed to Python via PyO3.

---

## Using rlox with Non-CartPole Environments

`rlox.VecEnv` currently only supports CartPole natively. For other environments, use Gymnasium for stepping and rlox for the compute-heavy parts:

```python
import gymnasium
import numpy as np
import rlox

# Gymnasium for stepping
vec_env = gymnasium.vector.SyncVectorEnv(
    [lambda: gymnasium.make("Acrobot-v1") for _ in range(8)]
)
obs, _ = vec_env.reset(seed=42)

# rlox for storage
buffer = rlox.ExperienceTable(obs_dim=6, act_dim=1)

# rlox for GAE
rewards = np.ones(128, dtype=np.float64)
values = np.ones(128, dtype=np.float64) * 0.5
dones = np.zeros(128, dtype=np.float64)
advantages, returns = rlox.compute_gae(
    rewards, values, dones,
    last_value=0.5, gamma=0.99, lam=0.95,
)
```

See `benchmarks/convergence/rlox_runner.py` for a complete example of this pattern.

---

## Running Tests

```bash
# Rust tests
cargo test --package rlox-core

# Python tests (`tests/`, not `tests/python/` — the latter skips tests/agentic/
# and the repo-hygiene guards)
.venv/bin/python -m pytest tests/ -q

# Every gate CI runs (recommended before pushing)
bash scripts/check-ci-local.sh
```

---

## torch.compile

Accelerate neural network inference with `torch.compile`:

```python
from rlox.compile import compile_policy

sac = SAC(env_id="Pendulum-v1")
compile_policy(sac)  # compiles actor, critic1, critic2
sac.train(total_timesteps=50_000)

# For on-policy policies (PPO/A2C), individual methods are compiled:
# get_action_and_logprob, get_value, get_logprob_and_entropy
ppo = PPO(env_id="CartPole-v1")
compile_policy(ppo)
```

---

## Plugin Ecosystem

rlox provides a plugin system for registering custom environments, buffers, and reward functions. Third-party packages can expose plugins via entry points that are discovered automatically.

### Registries

```python
from rlox.plugins import ENV_REGISTRY, BUFFER_REGISTRY, REWARD_REGISTRY

# Register a custom environment factory
ENV_REGISTRY.register("my-env-v0", lambda: MyCustomEnv())

# Register a custom buffer class
BUFFER_REGISTRY.register("my-buffer", MyBufferClass)

# Register a custom reward function
REWARD_REGISTRY.register("shaped-reward", my_reward_fn)
```

### Using registered components

```python
from rlox import Trainer

# Use a registered custom environment by name
trainer = Trainer("ppo", env="my-env-v0", seed=42)
metrics = trainer.train(total_timesteps=50_000)
```

### Plugin discovery

Plugins from installed packages are discovered automatically via Python entry points:

```python
from rlox.plugins import discover_plugins

# Scan installed packages for rlox plugins
discover_plugins()

# Plugins register themselves into ENV_REGISTRY, BUFFER_REGISTRY, etc.
```

To make your package discoverable, add an entry point in your `pyproject.toml`:

```toml
[project.entry-points."rlox.plugins"]
my_plugin = "my_package.plugin:register"
```

---

## Model Zoo

The model zoo provides a registry of pretrained agents with metadata (algorithm, environment, hyperparameters, performance).

```python
from rlox.model_zoo import ModelZoo, ModelCard

# Register a trained model
card = ModelCard(
    name="ppo-cartpole-v1",
    algorithm="ppo",
    env="CartPole-v1",
    mean_reward=500.0,
    description="PPO agent trained to solve CartPole-v1",
)
ModelZoo.register(card, checkpoint_path="checkpoints/ppo_cartpole.pt")

# List available models
for card in ModelZoo.list():
    print(f"{card.name}: {card.mean_reward:.1f}")

# Load a pretrained model
trainer = ModelZoo.load("ppo-cartpole-v1")
action = trainer.predict(obs, deterministic=True)
```

---

## Visual RL Wrappers

Wrappers for pixel-based reinforcement learning, providing standard preprocessing for image observations.

```python
from rlox.wrappers.visual import FrameStack, ImagePreprocess, AtariWrapper, DMControlWrapper

# FrameStack: stack N consecutive frames along the channel axis
env = FrameStack(env, n_frames=4)

# ImagePreprocess: resize, grayscale, normalize pixel values
env = ImagePreprocess(env, width=84, height=84, grayscale=True)

# AtariWrapper: combines standard Atari preprocessing
# (NoopReset, MaxAndSkip, EpisodicLife, FireReset, ClipReward, FrameStack)
env = AtariWrapper(env, frame_stack=4)

# DMControlWrapper: wraps DeepMind Control Suite environments
env = DMControlWrapper(domain="cartpole", task="swingup")
```

---

## Language RL Wrappers

Wrappers for language-grounded and goal-conditioned reinforcement learning.

```python
from rlox.wrappers.language import LanguageWrapper, GoalConditionedWrapper

# LanguageWrapper: adds language instructions to observations
env = LanguageWrapper(env, instruction_fn=lambda obs: "move to the red block")

# GoalConditionedWrapper: adds goal specifications to the observation space
env = GoalConditionedWrapper(env, goal_fn=sample_goal, reward_fn=goal_reward)
```

---

## Cloud Deploy

Generate deployment artifacts for trained agents: Dockerfiles, Kubernetes job manifests, and SageMaker configurations.

```python
from rlox.deploy import generate_dockerfile, generate_k8s_job, generate_sagemaker_config

# Generate a Dockerfile for serving a trained model
dockerfile = generate_dockerfile(
    checkpoint_path="checkpoints/ppo_cartpole.pt",
    algorithm="ppo",
    env="CartPole-v1",
    base_image="python:3.12-slim",
)
with open("Dockerfile", "w") as f:
    f.write(dockerfile)

# Generate a Kubernetes job manifest
k8s_manifest = generate_k8s_job(
    name="rlox-training",
    image="my-registry/rlox-agent:latest",
    gpu=1,
    memory="8Gi",
)
with open("k8s-job.yaml", "w") as f:
    f.write(k8s_manifest)

# Generate SageMaker training config
sagemaker_cfg = generate_sagemaker_config(
    algorithm="ppo",
    env="CartPole-v1",
    instance_type="ml.g4dn.xlarge",
)
```

> **Note:** The deploy module validates all inputs (checkpoint paths, image names, resource specifications) before generating artifacts.

---

## Checkpoint Security

All checkpoint loading uses `weights_only=True` by default via `safe_torch_load()`. This prevents arbitrary code execution from untrusted checkpoint files:

```python
from rlox.checkpointing import safe_torch_load

# Safe by default — only loads tensor data, not arbitrary Python objects
state_dict = safe_torch_load("model.pt")

# All Trainer.from_checkpoint() and algorithm.load() calls use safe_torch_load internally
trainer = Trainer.from_checkpoint("model.pt", algorithm="ppo", env="CartPole-v1")
```

---

## Cross-References

- [Examples](examples.md) -- comprehensive code examples
- [LLM Post-Training Guide](llm-post-training.md) -- DPO, GRPO, OnlineDPO
- [Rust User Guide](rust-guide.md) -- using `rlox-core` directly from Rust
- [Mathematical Reference](math-reference.md) -- algorithm derivations
- [References](references.md) -- academic papers
- [Getting Started](getting-started.md) -- tutorial walkthrough
