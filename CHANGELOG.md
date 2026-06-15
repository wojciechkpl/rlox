# Changelog

All notable changes to rlox are documented here.

## [Unreleased]

### Added
- **`rlox-sandbox` crate** — unprivileged Linux sandbox for executing untrusted code with hard isolation (Component 1 of agentic-benchmark MVP). Combines Linux user+pid+net+mnt namespaces, seccomp-BPF allowlist filter (blocks network, CLONE_NEWUSER, ptrace, etc.), and cgroup v2 resource limits (memory.max, pids.max, cpu.weight) with kill-safety via cgroup freeze + `cgroup.kill` (Linux 5.14+). Memory bombs are contained at the cap (no swap-thrash) via `memory.swap.max=0` with authoritative OOM detection through `memory.events`. Includes 26 regression tests covering benign execution, timeouts, and adversarial containment (fork bombs, memory bombs, stdout flooding, nested-userns denial, cgroup-base validation).
- **Adversarial corpus v1** (`benchmarks/agentic/corpus/`) — fixed, versioned, SHA-256-integrity-checked stimulus for the benchmark's P3 reliability claim. Covers all six categories (infinite loop, fork bomb, memory bomb, unkillable thread, blocking network, fd exhaustion); every sample is verified contained by `rlox-sandbox` (bounded time-to-contain, no survivors).

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
