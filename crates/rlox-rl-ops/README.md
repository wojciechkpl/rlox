# rlox-rl-ops

Estimator-agnostic advantage and token-KL operations for rlox.

This crate has **no dependency on `rlox-core`** — no environment physics, replay buffers, or GAE. It is intentionally slim so that `rlox-sandbox` can depend on it without pulling in the entire training data-plane.

## Purpose

`rlox-rl-ops` decouples advantage estimation (e.g., GRPO, DAPO) and token-level KL divergence computation from the larger training framework. This enables:

- **Pluggable advantage estimators** — implement the `AdvantageEstimator` trait and swap algorithms without touching the crate's core
- **Lightweight sandbox usage** — `rlox-sandbox` uses GRPO advantages for reward computation without depending on full `rlox-core`
- **Back-compat re-exports** — `rlox-core` re-exports these ops for existing code

## Key Types

### `AdvantageEstimator` trait

```rust
pub trait AdvantageEstimator: Send + Sync {
    fn compute(
        &self,
        rewards: &[f32],
        group_size: usize,
    ) -> Result<Vec<f32>, RlOpsError>;
}
```

Compute per-rollout advantages from a flat slice of scalar rewards. `rewards` is length `n_groups * group_size`; returns a `Vec` of the same length.

Implementations MUST be:
- **Deterministic**: same inputs always produce identical outputs.
- **Autograd-free**: this crate has no tensor/gradient dependencies.
- **Thread-safe**: `Send + Sync` so the estimator can live inside `Arc<dyn AdvantageEstimator>`.

### `GroupRelativeEstimator` (GRPO)

```rust
pub struct GroupRelativeEstimator;

impl AdvantageEstimator for GroupRelativeEstimator { ... }
```

Normalises rewards within each group using z-score: `z = (r - mean) / std`.
Returns zeros when `std < 1e-8` (constant-reward group — no variance to normalize).

Uses rayon parallel dispatch when `rewards.len() >= 4096` elements.

### `compute_single_group_advantages(rewards: &[f32]) -> Vec<f32>`

Convenience free function for single-group advantage computation. Equivalent to:
```rust
let estimator = GroupRelativeEstimator;
estimator.compute(rewards, rewards.len()).unwrap()
```

### Error: `RlOpsError`

```rust
pub enum RlOpsError {
    ShapeMismatch { reason: String },
}
```

Returned when `rewards.len() % group_size != 0` (non-divisible) or `group_size == 0`.

## Token-level KL ops

Module `kl` provides exact and approximate token-wise KL divergence:

- **`f64_ops`** — double-precision, f64 vectors
- **`f32_ops`** — single-precision, f32 vectors
- **`compute_kl_exact(logp_old, logp_new)`** — exact: sum of log(p_old / p_new)
- **`compute_kl_schulman_2020(logp_old, logp_new)`** — Schulman 2020 approximation for numerical stability

All re-exported at crate root for ergonomic access.

## Example: GRPO Advantages

```rust
use rlox_rl_ops::{AdvantageEstimator, GroupRelativeEstimator};

// Four samples from two rollouts (groups): [1, 0, 1, 0] | [5, 5, 5, 5]
let rewards = [1.0f32, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0, 5.0];

let estimator = GroupRelativeEstimator;
let advantages = estimator.compute(&rewards, 4)?;

// Group 0: mean=0.5, std=0.5 → [+1.0, -1.0, +1.0, -1.0]
// Group 1: mean=5.0, std=0.0 → [0.0, 0.0, 0.0, 0.0] (no variance)
assert_eq!(advantages[0], 1.0);
assert_eq!(advantages[1], -1.0);
assert_eq!(advantages[4], 0.0); // Constant group
```

## Example: Non-divisible Group Size (Error)

```rust
use rlox_rl_ops::{AdvantageEstimator, GroupRelativeEstimator};

let estimator = GroupRelativeEstimator;
let rewards = [1.0f32, 0.0, 1.0]; // 3 elements
let result = estimator.compute(&rewards, 2); // 3 % 2 != 0

assert!(result.is_err(), "must return Err for non-divisible group_size");
```

## Dependencies

- **`libc`** — raw memory utilities
- **`rayon`** — parallel iteration for large batches (≥4096 elements)
- **`thiserror`** — error type definition

No PyO3, no tensor libraries, no autograd.

## Test Coverage

30 unit tests covering:
- Core advantage computation (z-score normalization, constant groups)
- Multiple groups (independent normalization per group)
- Error handling (non-divisible shapes, zero group_size)
- Rayon parallelism threshold
- Trait object storage (`Arc<dyn AdvantageEstimator>`)
- Token-KL exact and approximate formulas

Run tests with:
```bash
cargo test -p rlox-rl-ops
```

## See Also

- `rlox-sandbox` — uses `rlox-rl-ops::GroupRelativeEstimator` for reward computation
- `rlox-core` — re-exports these ops for backward compatibility
- `benchmarks/agentic/` — uses GRPO advantages in the validation benchmark

## License

Dual-licensed under MIT or Apache 2.0 (workspace standard).
