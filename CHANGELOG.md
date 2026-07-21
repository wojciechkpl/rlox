# Changelog

All notable changes to rlox are documented here.

## [Unreleased]

### Added
- **`rlox-sandbox` crate** — unprivileged Linux sandbox for executing untrusted code with hard isolation (Component 1 of agentic-benchmark MVP). Combines Linux user+pid+net+mnt namespaces, seccomp-BPF allowlist filter (blocks network, CLONE_NEWUSER, ptrace, etc.), and cgroup v2 resource limits (memory.max, pids.max, cpu.weight) with kill-safety via cgroup freeze + `cgroup.kill` (Linux 5.14+). Memory bombs are contained at the cap (no swap-thrash) via `memory.swap.max=0` with authoritative OOM detection through `memory.events`. Includes 26 regression tests covering benign execution, timeouts, and adversarial containment (fork bombs, memory bombs, stdout flooding, nested-userns denial, cgroup-base validation).
- **Adversarial corpus v1** (`benchmarks/agentic/corpus/`) — fixed, versioned, SHA-256-integrity-checked stimulus for the benchmark's P3 reliability claim. Covers all six categories (infinite loop, fork bomb, memory bomb, unkillable thread, blocking network, fd exhaustion); every sample is verified contained by `rlox-sandbox` (bounded time-to-contain, no survivors).
- **Rollout server** (`rlox-sandbox::server`) — async axum `/rollout` service (Component 2 of agentic-benchmark MVP): calls a vLLM `/v1/completions` endpoint, runs each completion through the sandbox for a verifiable reward, computes group-relative advantages via the canonical `rlox-core` op, and returns trajectories plus first-class `BackendStats` telemetry (P1 throughput + P3 containment counters: `adversarial_contained`, `contagion_events`, `setup_error_events`, `time_to_contain_secs`, cgroup/oom events). vLLM failures surface as HTTP 502 (no silent zero-reward degradation); sandbox concurrency is bounded; `cgroup_base` is validated at startup (server must run inside a systemd-delegated scope). Python-side `rlox.agentic.stats.BackendStats` dataclass mirrors the wire contract.
- **Verification endpoint** (`rlox-sandbox::server` `POST /verify`) — sandbox-only seam for reward-level hosts (prime-rl/verifiers): takes already-generated `{code, tests, is_adversarial}`, runs one sandbox execution (no vLLM), returns `{reward, backend_stats}`. Complements `/rollout` (full generate+verify, for AgentLoop-style hosts).
- **Verifiers adapter** (`rlox.agentic.verifiers_adapter`, Component 4 of agentic-benchmark MVP) — implements prime-rl's `verifiers` `load_environment` seam with a one-key Baseline↔Treatment swap (`rollout_backend`): `"in_loop"` runs code in-process (the unsafe Baseline), `"rlox"` POSTs to the sandbox `/verify` endpoint (the isolated Treatment); identical config keys and identical adversarial injection in both. Wires `group_size → env.sampling_args["n"]`.
- **Adversarial injector** (`rlox.agentic.adversarial_corpus`) — loads + SHA-256-integrity-checks the corpus and injects adversarial samples at a configurable fraction via an instance-local seeded PRNG (deterministic, backend-independent). Canonical digest convention is consistent across Rust (serde_json) and Python (`json.dumps`).
- **Benchmark harness** (`rlox.agentic`, Component 6/7/8 of agentic-benchmark MVP) — `MetricCollector` (injectable GPU sampler, per-step JSONL, warm-up-excluded summary; AC-4), `ContagionDetector` (VmRSS-spike / FD-leak / server-reported rules with rolling baselines; AC-5 P3 hard line), `BenchmarkConfig` + `validate_config` (refuses to run on unset locked constants or version-pin mismatch; AC-2), and the `run_sweep` driver (2 conditions × seeds × adversarial fractions, per-run survival + metric-store persistence, exception-resilient). All TDD-tested on a light env (no GPU).
- **OQ-3 pilot** (`benchmarks/agentic/oq3_pilot.py`) — validates that the unprotected Baseline code-exec path degrades on the adversarial corpus (6/6 categories stall or exhaust), gating the pre-registered sweep; runs each sample in a resource-capped scope so it cannot wedge the host.
- **Reporting + go/no-go** (`rlox.agentic.reporting`, Step 7 of agentic-benchmark MVP) — numpy percentile-`bootstrap_ci` (deterministic), `ci_overlap_check` quality-parity guardrail (≥2/3 seed pairs; AC-9), `write_summary` (machine-readable JSON + CSV; AC-7), and `assess_go_no_go` encoding the pre-registered P1/P3/guardrail thresholds verbatim.
- **`repro.sh`** (`benchmarks/agentic/repro.sh`, AC-8) — one-command, idempotent reproduction: Rust toolchain + uv venv with pinned deps + release build of `rlox-sandbox` + adversarial-corpus SHA-256 integrity check + launch the sweep under a resource-capped systemd scope. `--setup-only` / `--dry-run` modes; the full 24-run grid (2×3 seeds×4 fractions) previews via `run_benchmark.py --dry-run`.
- **`rlox-rl-ops` crate** — extracted GRPO/token-KL ops from `rlox-core` into a slim, zero-dependency-on-core crate (estimator-agnostic `AdvantageEstimator` trait for GRPO↔DAPO pluggability, `GroupRelativeEstimator` z-score normalisation with rayon parallel dispatch ≥4096 elems, single-group convenience functions, token-KL ops f32+f64 exact and Schulman 2020 approximate). `rlox-core` re-exports for back-compat.
- **`rlox_agent` Python package** — the standalone, torch-free agentic harness (`adversarial_corpus`, `verifiers_adapter`, `stats`, `metric_collector`, `contagion_detector`, `config`, `reporting`) split out of `rlox.agentic`, which now re-exports as thin shims. Import as `from rlox_agent.<module>` (canonical); `rlox.agentic.<module>` remains for backward-compat but is deprecated.
- **`/verify` reward integrity** — nonce-authenticated trusted-runner protocol in `rlox-sandbox::server` so model code cannot forge reward via `sys.exit(0)`/`os._exit()`/monkeypatch; runner seals return value before child process sees it.
- **Security hardening** — seccomp fail-closed (denies by default), `clone3` blocked, `CLONE_NEWUSER` masked (conditional BPF rule), no persistent `/tmp` per sandbox, proc-scoped environment. Regression-tested in `crates/rlox-sandbox/tests/security_isolation.rs`. Honest caveat: filesystem/`/proc` info-isolation is best-effort under AppArmor `unprivileged_userns` (enforced at the Python-runtime level; a raw-syscall adversary could bypass it — a kernel-level guarantee requires the AppArmor profile to be relaxed).
- **TRL single-GPU GRPO runner** (`benchmarks/agentic/trl_grpo_run.py`) — Qwen3-4B-Instruct-2507 + LoRA on unit-test-verified coding task; `--backend {in_loop,rlox}` and `--adversarial-fraction` flags for Baseline↔Treatment A/B study. 30-step canonical sweep: Treatment (rlox) survives 3/3 runs at every injection fraction (0/1/5/10 %) vs Baseline 3/3·3/3·2/3·1/3; quality-parity guardrail MET (final reward 0.917 = 0.917). P1 (multi-GPU throughput) deferred; rlox's novel contribution is P3 (containment) + reward integrity.
- **Test coverage** — rlox-rl-ops 30 tests, rlox-core 472, rlox-sandbox 60+doctest, rlox_verify 11, benchmarks/agentic 567.
- **Per-algorithm maturity status** — `rlox.trainer.ALGORITHM_STATUS` maps every registered algorithm to `"validated"` (convergence-tested with SB3 parity: PPO, SAC, TD3, DQN, A2C) or `"experimental"` (implemented but not convergence-validated: the other 13 registered algorithms). Surfaced via the new read-only `Trainer.status` property, included in `repr(Trainer(...))`, and a `UserWarning` is emitted when an experimental algorithm is constructed by name. Helper `algorithm_status(name)` does a case-insensitive lookup. A completeness invariant (`set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY)`) forces every newly registered algorithm to declare a status.
- **`rlox-grpc` end-to-end test coverage** — first integration tests for the distributed env-worker gRPC layer (previously zero Rust tests): in-process server↔client round-trips for `reset_batch`/`step_batch` shape correctness and terminal-observation transmission.
- **TRPO promoted to `validated`** — added a convergence config (`benchmarks/convergence/configs/trpo_cartpole.yaml`) and ran a 5-seed sweep: CartPole-v1 **IQM = 500.0** (4/5 seeds ≥ 494.9, threshold 475), and confirmed learning on continuous control (Hopper-v4, 223 vs ~15 random at 100k). TRPO needs a large rollout per update (`n_steps=2048`) for a stable Fisher/KL estimate; smaller batches collapse convergence. `ALGORITHM_STATUS["trpo"]` is now `"validated"`; a full MuJoCo multi-seed parity sweep is the tracked follow-up.
- **PQN algorithm** (`rlox.algorithms.PQN`, Parallelised Q-Network — Gallici et al., arXiv:2407.04811) — value-based RL with **no replay buffer and no target network**: a LayerNorm-regularised Q-network (`LayerNormQNetwork`) plus ℓ² regularisation stabilise TD learning, with λ-returns computed over parallel rollouts. Reuses the existing Rust `compute_gae_batched` op for the Q(λ) targets (no new Rust) and rlox's Rayon `VecEnv` for the parallel env loop. Discrete action spaces; correct truncation bootstrapping (adds `γ·max_a Q(terminal_obs,a)` on time-limit truncation, mirroring `RolloutCollector`). Registered as `Trainer("pqn", ...)`, status `experimental` (learns CartPole; multi-seed validation pending). `PQNConfig` with validated ε schedule.

### Fixed
- **AWR `predict()`** — raised `AttributeError` on every inference call (`self.policy.actor` referenced a non-existent attribute; the class holds `self.actor`/`self.critic`). Now branches on `self.discrete` and returns an int action (discrete) or a `numpy.ndarray` (continuous), matching the `train()` path. Regression tests added.
- **`rlox-grpc` dropped terminal observations** — `RemoteEnvClient` hardcoded `terminal_obs: vec![None; num_envs]`, so truncation-bootstrap observations were never transmitted, silently corrupting value targets for distributed IMPALA on truncating environments (e.g. all MuJoCo tasks). The `StepResponse` now carries `terminal_obs` (flat, zero-padded) plus a `has_terminal_obs` mask; the server errors on an obs-dim mismatch instead of silently truncating, and the client reconstructs `Vec<Option<Vec<f32>>>` (with a backward-compatible fallback against older servers).

## [1.2.0] - 2026-05-05

### Added
- **`Trainer.evaluate(n_episodes, seed, render)`** -- deterministic evaluation returning mean/std/min/max reward and episode lengths
- **`Trainer.enjoy(n_episodes, seed)`** -- render the trained policy for visual inspection
- **`VideoRecordingCallback`** -- records evaluation episodes to mp4 at configurable intervals during training
- **`AsymmetricPolicy`** -- actor-critic with separate observation spaces (actor sees deployment obs, critic sees privileged state). Supports both discrete and continuous actions
- **Episode statistics tracking** -- `RolloutCollector` and `GymVecEnv` now expose `episode_rewards` and `episode_lengths` properties for completed episodes
- **`RecordEpisodeStatistics`** auto-wrapping in `GymVecEnv`
- **Score normalization** -- `normalize_score()`, `normalize_scores()`, and `SCORE_BASELINES` dict for mapping raw returns to [0, 1] using random/expert baselines (14 environments)
- **Bootstrap CI bands** on learning curves in `multi_seed_runner.py` via `--eval-freq` flag
- **`EmaRunningStats`** (Rust + PyO3) -- exponential moving average mean/variance for non-stationary signals. Constructors: `EmaRunningStats(alpha)`, `.from_window(N)`, `.from_halflife(h)`
- **`CusumDetector`** (Rust + PyO3) -- two-sided CUSUM change-point detection with optional burn-in period for automatic reference level estimation
- **`PageHinkleyDetector`** (Rust only) -- Page-Hinkley change-point detection
- **`NonStationaryCartPole`** (Rust only) -- CartPole with configurable parameter drift (gravity, pole length, cart mass, force magnitude) via `DriftMode` (None, Linear, Sinusoidal, Step)
- **`ReplayBuffer.sample_recent(batch_size, window_size, seed)`** -- sliding window replay for non-stationary RL, sampling only from recent transitions
- **Dynamic regret metrics** in `evaluation.py`: `dynamic_regret()`, `adaptation_latency()`, `forgetting_ratio()` for non-stationary RL evaluation

### Fixed
- CI: override `target-cpu=native` to prevent SIGILL from stale cache on different hardware

## [1.1.0] - 2026-03-29

### Added
- **VPG algorithm** -- Vanilla Policy Gradient with GAE support
- **Plugin ecosystem** -- `ENV_REGISTRY`, `BUFFER_REGISTRY`, `REWARD_REGISTRY` for registering custom components; `discover_plugins()` for auto-discovery via Python entry points
- **Model zoo** -- `ModelZoo.register`, `ModelZoo.load`, `ModelCard` for sharing and reusing pretrained agents
- **Visual RL wrappers** -- `FrameStack`, `ImagePreprocess`, `AtariWrapper`, `DMControlWrapper` for pixel-based RL
- **Language RL wrappers** -- `LanguageWrapper`, `GoalConditionedWrapper` for language-grounded tasks
- **Cloud deploy** -- `generate_dockerfile`, `generate_k8s_job`, `generate_sagemaker_config` for deployment artifact generation
- **`predict()` method** added to TRPO, IMPALA, MAPPO, A2C, VPG (all algorithms now support `predict()`)
- **22 algorithm documentation pages** completed

### Changed
- `safe_torch_load()` -- all checkpoint loading now uses `weights_only=True` by default for security
- `VecEnv::new` now returns `Result<VecEnv, RloxError>` instead of panicking on invalid input
- `Transition.info` changed from `HashMap<String, f64>` to `Option<HashMap<String, f64>>` (None when no metadata)
- PBT (Population-Based Training) is now fully reproducible with seeded RNG
- Docker deploy module validates all inputs (checkpoint paths, image names, resource specs) before generating artifacts
- 12+ core types now derive `Debug` and `Clone` for better ergonomics and debuggability

### Fixed
- Checkpoint security: prevented potential arbitrary code execution from untrusted checkpoints via `weights_only=True`

### Test Suite
- 444 Rust tests (was 409)
- ~1094 Python tests (was 869)

## [1.0.0] - 2026-03-29

### API 1.0 Freeze

This release marks the first stable API. All public exports are frozen and
covered by stability tests. Semver guarantees apply from this version onward.

### Added
- **A2CTrainer** -- high-level trainer wrapping A2C with callback/logger integration
- **TD3Trainer** -- high-level trainer wrapping TD3 with callback/logger integration
- **MAPPOTrainer** -- high-level trainer wrapping MAPPO for multi-agent environments
- **DreamerV3Trainer** -- high-level trainer wrapping DreamerV3 world-model-based RL
- **IMPALATrainer** -- high-level trainer wrapping IMPALA actor-learner architecture
- All trainers expose `train(total_timesteps)`, `save(path)`, and `from_checkpoint(path)`
- `train_from_config` dispatch extended to all 8 algorithms: ppo, sac, dqn, a2c, td3, mappo, dreamer, impala
- Complete `__all__` exports: trainers, configs, protocols, exploration, builders, losses, distributed, dashboard
- Distributed components exported at top level: `MultiGPUTrainer`, `RemoteEnvPool`, `launch_elastic`
- API stability test suite expanded: all 8 trainers, all 8 configs, distributed symbols, runner dispatch

### Changed
- `_VALID_ALGORITHMS` in config now includes `mappo`, `dreamer`, `impala`
- Runner dispatch uses Trainer wrappers for all algorithms (no more raw algo class fallback)
- `pyproject.toml` classifier updated to `Development Status :: 5 - Production/Stable`
- Version bumped to 1.0.0 in `__init__.py` and `pyproject.toml`

## [0.3.0] - 2026-03-29

### Added
- **VecNormalize environment wrapper** — obs/reward normalization at the
  environment boundary (SB3 architecture), replacing collector-level normalization
- **RunningStatsVec** — per-dimension Welford statistics in Rust (PyO3 exposed)
- **Native Pendulum-v1** — Rust environment with continuous action space
- **Polymorphic VecEnv.step_all** — accepts discrete (Vec<u32>) and continuous
  (ndarray float32) actions
- **VecEnv.action_space property** — typed dict for Python-side detection
- **VecEnv protocol** — formal protocol in `protocols.py`
- **A2CConfig, TD3Config** — dataclass configs with validation and YAML support
- **Offline RL**: TD3+BC, IQL, CQL, BC algorithms with `OfflineDatasetBuffer` (Rust)
- **Candle Hybrid Collection**: `CandleCollector` (180K SPS on CartPole), `HybridPPO` trainer
- **OffPolicyCollector**: Reusable multi-env collection for SAC, TD3, DQN (`n_envs` parameter)
- **`OfflineAlgorithm` base class** with `OfflineDataset` protocol for extensible offline RL
- **`SharedPolicy`** + weight sync for Candle/PyTorch interop
- **`RolloutBatch`** extended with `log_probs` and `values` fields
- **SB3 migration guide** at `docs/tutorials/migration-sb3.md`
- **API reference** pages with mkdocstrings autodoc
- **CONTRIBUTING.md** with development setup and guidelines
- **Cross-navigation** header across all documentation components
- Python 3.13 added to CI test matrix

### Fixed
- **Truncation bootstrap** — truncated episodes now bootstrap V(terminal_obs)
  instead of treating as deaths (value=0). Critical for MuJoCo time limits.
- **Per-dimension obs normalization** — replaced scalar mean/std with per-dim
  tracking, preserving observation structure across different scales
- **Return-based reward normalization** — std of discounted returns (SB3
  convention) instead of std of raw rewards
- **Train/collect obs mismatch** — consistent normalization during collection
  and training
- **A2C advantage normalization default** — changed to False, preventing gradient
  explosion with small batches (n_steps=5, batch=40)
- **log_std init** — 0.0 (std=1.0) matching SB3, was -0.5
- **GCS upload path** — absolute paths in convergence benchmark scripts
- IMPALA: V-trace now uses computed bootstrap value instead of hardcoded 0.0
- IMPALA: Auto-detects continuous envs, falls back to GymVecEnv for non-CartPole
- DreamerV3: World model frozen during actor-critic training (prevents gradient leakage)
- DreamerV3: Gradient clipping added to both world model and actor-critic updates
- MAPPO: NotImplementedError for n_agents > 1 (prevents silent dimension mismatch)
- MAPPO: Simplified critic input for single-agent case

### Changed
- Normalization moved from `RolloutCollector` to `VecNormalize` wrapper
- PPO auto-wraps env with VecNormalize when normalize flags set
- EvalCallback freezes normalization stats during evaluation

### Improved
- Landing page redesigned with quickstart, benchmarks, comparison table, algorithm grid
- Rust crate descriptions and lib.rs doc comments updated
- 80+ new Python tests (convergence fixes, VecNormalize, Pendulum, offline RL)
- 30+ new Rust tests (RunningStatsVec, Pendulum, OfflineDatasetBuffer)

## [0.2.0] - 2026-03-16

### Added
- **Phase 7: Algorithm Completeness**
  - `GymVecEnv` wrapper for arbitrary Gymnasium environments (AutoresetMode.SAME_STEP)
  - `ContinuousPolicy` (Gaussian, orthogonal init) for on-policy continuous control
  - `BatchSteppable` trait for environment abstraction
  - Auto env detection: PPO/A2C auto-select Discrete/Continuous policy from action space
  - `reward_fn` parameter on `RolloutCollector` for reward shaping
  - Callbacks wired into all 7 algorithms (PPO, SAC, DQN, TD3, A2C, GRPO, DPO)
  - `save()`/`from_checkpoint()` on PPO, SAC, DQN, TD3, GRPO, DPO
  - `from_yaml()`/`to_yaml()` on PPOConfig, SACConfig, DQNConfig
  - GRPO batched advantages (eliminates Python loop, uses `compute_batch_group_advantages`)

- **Phase 8: Production Hardening**
  - Statistical evaluation toolkit: IQM, bootstrap CI, performance profiles, P(improvement)
  - `TrainingDiagnostics` callback: entropy collapse, KL spike, gradient explosion detection
  - Memory-mapped replay buffer (`MmapReplayBuffer`) for hot/cold architecture
  - CI workflows: GitHub Actions for tests + maturin wheel builds (4 platforms)
  - Experiment metadata capture + `save_experiment()`

- **Phase 9: Distributed & Scale**
  - Decoupled collection/training pipeline (crossbeam channels, `AsyncCollector`)
  - gRPC distributed env workers (`rlox-grpc` crate with tonic)
  - Multi-GPU training composition (PyTorch DDP wrapper)
  - vLLM, TGI, SGLang inference backends with factory
  - `RemoteEnvPool` Python client for gRPC workers
  - Transition provenance (`TransitionMeta` with serialize/deserialize)
  - MAPPO, DreamerV3, IMPALA algorithms with env auto-detection
  - API 1.0 freeze: comprehensive `__all__`, stability tests

- **Buffer Extensions**
  - Typed extra columns (`register_column`/`push_extra`) with O(1) ColumnHandle access
  - Dict observation space (`Observation::Dict`, `ObsSpace::Dict`)
  - `BatchDictBuilder` for deduplicated PyO3 dict construction

- **Infrastructure**
  - MIT OR Apache-2.0 dual license
  - Published to crates.io: rlox-core, rlox-nn, rlox-burn, rlox-candle
  - Tutorial: custom rewards and training loops (1,480 lines)
  - Logo and citation info (CITATION.cff)

### Fixed
- **Critical**: `PyVecEnv` silently fell back to CartPole for unknown env_ids — now raises `ValueError`
- **Critical**: Replay buffer missing `next_obs` — off-policy algorithms (SAC, TD3, DQN) computed wrong Bellman targets
- SAC: action scaling now multiplies by `act_high` (was only clipping)
- TD3: critic target updates moved outside `policy_delay` gate
- DQN: n-step flush uses actual termination flags (was hardcoded `terminated=True`)
- Config consolidation: single validated `PPOConfig` (was duplicated)

### Test Suite
- 313 Rust tests at v0.2.0 (was 255)
- 382 Python tests at v0.2.0 (was 85)
- Zero benchmark regressions

## [0.1.0] - 2026-03-14

### Added
- Phases 0-6: core Rust engine, environment stepping, buffers, GAE, V-trace
- LLM post-training: GRPO, DPO, token KL, sequence packing
- NN backend abstraction: rlox-nn traits, rlox-burn, rlox-candle
- Three-framework benchmark suite (rlox vs TorchRL vs SB3)
- Convergence benchmarks (rlox vs SB3 on Classic Control)
- 255 Rust tests, 85 Python tests
