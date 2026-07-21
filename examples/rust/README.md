# Rust Examples

Standalone Rust programs demonstrating rlox-core without Python.

## Running

```bash
cd examples/rust
cargo run --bin cartpole        # CartPole with random actions
cargo run --bin pendulum        # Pendulum with continuous actions
cargo run --bin vec_env         # 64 parallel envs benchmark
cargo run --bin compute_gae     # GAE advantage computation
cargo run --bin replay_buffer   # Uniform + prioritized buffers
cargo run --bin reward_shaping  # PBRS + goal-distance potentials
cargo run --bin grpo_advantages # GRPO advantage estimation (agentic)
```

## Examples

| Binary | What it demonstrates |
|--------|---------------------|
| `cartpole` | `RLEnv` trait, `CartPole`, discrete actions, episode tracking |
| `pendulum` | `Pendulum`, continuous actions (`Action::Continuous`) |
| `vec_env` | `VecEnv` parallel stepping with Rayon, throughput benchmark |
| `compute_gae` | `compute_gae` with known trajectory, invariant verification |
| `replay_buffer` | `ReplayBuffer` + `PrioritizedReplayBuffer`, push/sample |
| `reward_shaping` | `shape_rewards_pbrs`, `compute_goal_distance_potentials` |

## Agentic sandbox

These examples target the agentic-RL benchmark components.

| Binary | What it demonstrates | Platform |
|--------|---------------------|----------|
| `grpo_advantages` | `GroupRelativeEstimator`, `AdvantageEstimator` trait, multi-group z-score normalisation, error path | all |
| `sandbox_verify` | `run_sandboxed`: benign job → `Clean`/`reward > 0`; fork-bomb → `Timeout`/contained | Linux only |

### `grpo_advantages` (all platforms)

```bash
cd examples/rust
cargo run --bin grpo_advantages
```

### `sandbox_verify` (Linux only — requires cgroup v2 delegation)

The sandbox needs to run inside a `systemd` scope with `Delegate=yes` so that
the child process can self-migrate into its leaf cgroup.

Via the repo helper (handles delegation automatically):

```bash
bash scripts/wk-sync-test.sh \
  'cargo run --manifest-path examples/rust/Cargo.toml --bin sandbox_verify'
```

Manually on Linux:

```bash
systemd-run --user --scope --slice=rlox.slice -p Delegate=yes \
  cargo run --manifest-path examples/rust/Cargo.toml --bin sandbox_verify
```

On macOS, `cargo build --bin sandbox_verify` still compiles — the binary
prints a short note and exits cleanly (no Linux APIs are referenced outside
the `#[cfg(target_os = "linux")]` block).

## Using rlox-core in your project

```toml
# Cargo.toml
[dependencies]
rlox-core = { git = "https://github.com/wojciechkpl/rlox", package = "rlox-core" }
```

Or with a local checkout:

```toml
[dependencies]
rlox-core = { path = "path/to/rlox/crates/rlox-core" }
```
