# rlox — agent quick reference

> Read this FIRST before any code-writing or review task. It replaces ~8
> discovery calls (Grep/Glob/Read of layout files) with one read. If a fact
> here is stale, fix the fact before writing new code.
>
> This file is **machine-oriented** — terse, factual, no marketing.
> For a human-facing overview, read `README.md`.

---

## What this is

Rust-accelerated reinforcement learning framework. Python control plane +
Rust data plane. The Polars pattern: **any data transform without autograd
belongs in Rust**. Autograd (policies, losses, optimizers) lives in PyTorch.

22 RL algorithms share a unified `Trainer('algo', env=..., config=...)` API.
Convergence parity with Stable-Baselines3 on the core six (PPO, SAC, TD3,
DQN, A2C, TRPO).

---

## Critical commands

| purpose | command |
|---|---|
| Install (editable, rebuild Rust) | `maturin develop --release` |
| Run unit tests | `./.venv/bin/python -m pytest tests/ -q` (use `tests/`, not `tests/python/` — the latter collects 1583 of 2233 and skips `tests/agentic/`) |
| Run slow / integration tests | `./.venv/bin/python -m pytest -m slow` |
| **Verify everything CI checks** | `bash scripts/check-ci-local.sh` (`rust` / `python` to scope, `WK=1` to add the Linux sandbox suite) |
| Run Rust tests | `cargo test --workspace --no-fail-fast` |
| Python lint | `ruff check python/ tests/` |
| Rust lint (host) | `cargo clippy --workspace --exclude rlox-sandbox --all-targets -- -D warnings` |
| Rust lint (Linux-only crate) | `cargo clippy -p rlox-sandbox --all-targets --target x86_64-unknown-linux-gnu -- -D warnings` |
| Format | `ruff format python/ tests/ && cargo fmt --all` |
| Run a single benchmark cell | `./.venv/bin/python benchmarks/multi_seed_runner.py --algo ppo --env Hopper-v4 --timesteps 100000 --seeds 1` |
| Multi-seed convergence (local) | `cd ../rlox-priv && bash scripts/run-multi-seed.sh --local` |
| Multi-seed convergence (GCP) | `cd ../rlox-priv && bash scripts/run-multi-seed.sh --gcp` |
| Aggregate historic results | `./.venv/bin/python scripts/inspect_results.py` |

Python interpreter: always invoke via `./.venv/bin/python`, never bare
`python3` (the venv has the editable `rlox` install and MuJoCo). Bare `python3`
is also 3.14 on this machine, which pyo3 0.23 does not support — it hard-fails
the `pyo3-ffi` build script and takes the whole workspace lint down with it.

Two traps worth knowing before trusting a local green run:

- `cargo clippy --workspace` **does not compile on macOS** — `rlox-sandbox` is
  Linux-only (namespaces, seccomp, cgroup v2) and its `seccompiler` dep fails
  against macOS libc. Lints inside `#[cfg(target_os = "linux")]` are invisible
  unless you lint against a Linux target (see the table above).
- `cargo test` stops at the first failing test *binary*, so one failure hides
  every later binary's. Always `--no-fail-fast`.

`scripts/check-ci-local.sh` handles both, plus toolchain pinning
(`rust-toolchain.toml`) so local clippy matches CI's lint set exactly.

---

## Directory map (1-line each — keep stable)

### Python control plane (`python/rlox/`)
- `trainer.py`                  — unified `Trainer('ppo'|'sac'|...)` entrypoint + `ALGORITHM_REGISTRY` + `evaluate()` + `enjoy()`
- `config.py`                   — `PPOConfig` / `SACConfig` / ... dataclasses with YAML alias support
- `losses.py`                   — `PPOLoss` (SB3-aligned, see "Non-obvious facts")
- `collectors.py`               — `RolloutCollector` (on-policy rollouts, GAE via Rust, episode stats tracking)
- `vec_normalize.py`            — SB3-compatible running obs/reward normalization
- `gym_vec_env.py`              — Gymnasium `SyncVectorEnv` wrapper matching `rlox.VecEnv` interface + RecordEpisodeStatistics
- `policies.py`                 — `DiscretePolicy`, `ContinuousPolicy`, `AsymmetricPolicy` (orthogonal init, Tanh)
- `checkpoint.py`               — `safe_torch_load` with `allow_unsafe` opt-in
- `algorithms/`                 — 22 self-contained algo files (`ppo.py`, `sac.py`, `dqn.py`, ...)
- `evaluation.py`               — IQM, CI, score normalization, dynamic regret, adaptation latency, forgetting ratio
- `callbacks.py`, `logging.py`  — callback protocol + console/TB/MLflow loggers + `VideoRecordingCallback`

### Rust data plane (`crates/`)
- `rlox-core/`      — GAE, replay buffer, running stats (Welford + EMA), CUSUM/PageHinkley CPD, env physics (incl. NonStationaryCartPole); pure Rust, no PyO3
- `rlox-python/`    — PyO3 bindings (`rlox._rlox_core`)
- `rlox-candle/`    — Candle NN backend (experimental)
- `rlox-burn/`      — Burn NN backend (experimental)
- `rlox-grpc/`      — distributed actor/learner via tonic
- `rlox-bench/`     — microbenchmarks (criterion)
- `rlox-nn/`        — shared NN primitives

### Benchmarks, tests, docs
- `benchmarks/multi_seed_runner.py`    — 5-seed IQM + bootstrap CI harness (auto-resolves preset YAMLs)
- `benchmarks/multi_seed_runner_sb3.py` — SB3-in-our-harness runner (same eval protocol, identical seeds)
- `benchmarks/convergence/configs/`    — per-(algo, env) YAML presets (`ppo_hopper.yaml`, ...)
- `benchmarks/convergence/rlox_runner.py` — legacy single-seed runner (kept for equivalence tests)
- `benchmarks/ablation_component.py`   — component ablation (Rust GAE vs Python, Rust buffer vs SB3, etc.)
- `benchmarks/ablation_n_envs.py`      — scaling ablation: SPS vs number of parallel envs
- `benchmarks/ablation_obs_dim.py`     — scaling ablation: SPS vs observation dimensionality
- `tests/python/`                      — pytest suite, `-m slow` for long integration runs
- `docs/plans/`                        — live plan + review markdown (read these before proposing changes)
- `scripts/inspect_results.py`         — aggregator across v5/v6/v7/v8 single-seed JSON logs

### Companion repository
- `../rlox-priv/`                      — GCP launch scripts, non-public benchmark data, paper drafts
- `../rlox-priv/scripts/run-multi-seed.sh` — GCP sweep launcher (`--local` or `--gcp`)
- `../rlox-priv/results/`              — archived benchmark runs

---

## Conventions (enforced)

- **TDD**: Red-Green-Refactor. No new code without a test. No test, no commit.
- **Never attribute code changes to Claude** in commit messages or file headers.
- **No mocking the database** in integration tests (we got burned once).
- **Type hints**: PEP 484/604/695. No bare `Any` except at FFI boundaries.
- **Async**: Tokio in Rust, `asyncio` in Python. No blocking calls inside async functions.
- **Error handling**: `thiserror` per-module in Rust, typed exceptions in Python.
  Never use bare `except:`.
- **Data plane rule**: if it's data transformation without autograd, it belongs in Rust.
- **Configurability**: prefer configurable parameters over hardcoded seeds, rep counts, magic numbers.

---

## Non-obvious facts (common gotchas)

- **`PPOLoss` HAS an inner 0.5 factor** on the value loss. We follow CleanRL's
  `0.5 * MSE` convention, not SB3's plain `F.mse_loss`. `vf_coef=0.5` here
  produces an effective weight of `0.25 * MSE`. An earlier attempt to remove
  this factor (aligning with SB3) regressed Hopper-v4 by 57% at 1M steps
  and was reverted. See `docs/plans/benchmark-comparison-inconsistencies.md`.
- **`PPOConfig.clip_vloss` default is `True`** (CleanRL max-of-clipped
  formulation). Set to `False` for plain MSE (closer to SB3's
  `clip_range_vf=None`).
- **DQN defaults `train_freq=1, gradient_steps=1`** (1 grad step per env step).
  Set `train_freq=16, gradient_steps=8` for MountainCar or the agent over-trains
  on early transitions and never explores to the goal. See commit `17cdfeb`.
- **`Pendulum-v1` and `CartPole-v1` have native Rust envs** (see `_NATIVE_ENV_IDS`
  in `collectors.py`). When wrapped with `VecNormalize`, discreteness MUST be
  detected from the actual `action_space` object, not the env_id — see commit
  `1d882e8`.
- **`multi_seed_runner.py` auto-resolves per-(algo, env) preset YAMLs** from
  `benchmarks/convergence/configs/<algo>_<env_short>.yaml`. Without this,
  Trainer defaults (CleanRL CartPole-tuned) silently ran on MuJoCo.
- **Eval episode seeding**: use `seed=base+ep`, not `seed=base`, in eval loops.
  Identical resets across episodes inflate pseudo-precision of per-seed std.
- **Truncation bootstrap**: `RolloutCollector` adds `gamma * V(terminal_obs)`
  to the reward on truncation (time-limit), passing only `terminated` (not
  `done`) to GAE. This matters a lot for HalfCheetah (always truncates at 1000).
- **Checkpoints are secure by default**: `safe_torch_load(path)` uses
  `weights_only=True`. Pass `allow_unsafe=True` explicitly for legacy checkpoints.
- **`EmaRunningStats` and `CusumDetector`** are exposed via `rlox._rlox_core`
  (PyO3 bindings), not re-exported in `rlox.__init__`. Import as
  `from rlox._rlox_core import EmaRunningStats, CusumDetector`.
- **`ReplayBuffer.sample_recent(batch_size, window_size, seed)`** samples
  uniformly from only the most recent `window_size` transitions. Use for
  non-stationary RL where stale experience hurts.
- **`NonStationaryCartPole`** is Rust-only (no PyO3 binding yet). Use it from
  `rlox-core` Rust tests or extend `rlox-python/src/env.rs` to expose it.
- **`Trainer.evaluate()` freezes VecNormalize** stats automatically and
  restores them after evaluation. Seeds episodes as `seed + ep` to avoid
  inflated pseudo-precision.
- **Algorithm maturity is explicit.** `rlox.trainer.ALGORITHM_STATUS` maps every
  registered algo to `"validated"` (PPO/SAC/TD3/DQN/A2C — multi-seed SB3 parity)
  or `"experimental"` (the other 13). `Trainer.status` exposes it, `repr(trainer)`
  shows it, and constructing an experimental algo by name emits a `UserWarning`.
  A test (`tests/python/test_algorithm_status.py`) enforces
  `set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY)`, so a new algo must declare a
  status. Add new validated algos to `_VALIDATED` in `trainer.py`; everything else
  defaults to experimental.
- **Don't "deduplicate" the per-algo `_RUST_NATIVE_ENVS = {"CartPole-v1",
  "CartPole"}`** in `ppo.py`/`trpo.py`/`vpg.py` against
  `collectors._NATIVE_ENV_IDS`. They differ on purpose: the collector set also
  includes Pendulum, but those on-policy algos only route the *discrete* CartPole
  through the native Rust `VecEnv`. Merging them would send Pendulum down the Rust
  path and change behavior.

---

## Where "the rules" live

| scope | file |
|---|---|
| User global prefs | `~/.claude/CLAUDE.md` |
| Workspace-wide | `/Users/wojciechkowalinski/Sync/work/rlox-workspace/CLAUDE.md` |
| Auto-memory index | `~/.claude/projects/-Users-wojciechkowalinski-Sync-work-rlox-workspace/memory/MEMORY.md` |
| Live plans | `docs/plans/*.md` (read before proposing changes) |

---

## Load-bearing plan docs

Read these before proposing architectural or experimental changes:

- `docs/plans/results-inspection-2026-04-06.md` — current experimental state, SB3 alignment evidence
- `docs/plans/convergence-gap-investigation.md` — PPO vs SB3 MuJoCo history
- `docs/plans/master-improvement-plan.md` — the north-star roadmap
- `docs/plans/paper-implementation.md` — paper-track work queue
- `../rlox-priv/docs/plans/multi-seed-pre-flight-review-2026-04-06.md` — latest methodology critique

---

## Current state snapshot (update when it changes)

- **All 10 multi-seed cells complete** (2026-04-09). PPO/SAC/TD3/A2C/DQN converge with parity on every cell.
- **DQN MountainCar** requires `train_freq=16, gradient_steps=8`; not included in the 10-cell multi-seed sweep (DQN CartPole used instead -- both hit 500).
- **Multi-seed GCP sweep** runs via `rlox-priv/scripts/run-multi-seed.sh --gcp`. Auto-resolves preset YAMLs.
- **Ablation scripts** added: `benchmarks/ablation_component.py`, `ablation_n_envs.py`, `ablation_obs_dim.py`.
- **Trainer↔runner PPO equivalence** pinned by `tests/python/test_ppo_path_equivalence.py` (slow marker, Hopper 24k, 100-reward tolerance).
