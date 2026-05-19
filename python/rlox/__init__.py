"""rlox -- Rust-accelerated reinforcement learning.

The Polars architecture pattern applied to RL: Rust data plane for
environments, buffers, and advantage computation; Python control plane
for training logic, policies, and neural networks via PyTorch.

Core algorithms: PPO, SAC, DQN, TD3, A2C, TRPO (+ 16 more via submodules).

Rust primitives (via PyO3)
--------------------------
- :class:`CartPole`, :class:`VecEnv`, :class:`GymEnv` -- environment stepping
- :class:`ExperienceTable`, :class:`ReplayBuffer`, :class:`PrioritizedReplayBuffer`,
  :class:`MmapReplayBuffer`, :class:`OfflineDatasetBuffer` -- storage
- :func:`compute_gae`, :func:`compute_vtrace` -- advantage estimation
- :func:`compute_group_advantages`, :func:`compute_token_kl` -- LLM post-training
- :class:`DPOPair`, :class:`VarLenStore`, :func:`pack_sequences` -- sequence handling
- :class:`RunningStats`, :class:`RunningStatsVec` -- online mean/variance
- :class:`EmaRunningStats` -- exponential moving average stats (non-stationary signals)
- :class:`CusumDetector` -- two-sided CUSUM change-point detection
- :class:`ActorCritic`, :class:`CandleCollector` -- NN backend

Python layer
------------
- :class:`VecNormalize` -- obs/reward normalization wrapper (SB3-compatible)
- :class:`RolloutBatch` -- flat-tensor container for on-policy data
- :class:`RolloutCollector`, :class:`OffPolicyCollector` -- data collection
- :class:`GymVecEnv` -- gymnasium wrapper with VecEnv interface
- :class:`ContinuousPolicy`, :class:`DiscretePolicy`, :class:`AsymmetricPolicy` -- policy networks
- :class:`TrainingConfig` -- YAML/TOML config-driven training
- :class:`TerminalDashboard`, :class:`HTMLReport` -- diagnostics dashboard

Unified Trainer
---------------
- :class:`Trainer` -- single entry point for all 8 algorithms
- Algorithms: ``ppo``, ``sac``, ``dqn``, ``td3``, ``a2c``, ``mappo``, ``dreamer``, ``impala``
- :data:`ALGORITHM_REGISTRY` -- algorithm name -> class mapping

All algorithms expose ``train(total_timesteps)``, ``save(path)``,
``from_checkpoint(path)``, ``evaluate(n_episodes)``, and ``enjoy()``
via the unified :class:`Trainer`.

For algorithm implementations, see :mod:`rlox.algorithms`.
For config-driven training, see :func:`train_from_config`.

Quick start::

    from rlox import Trainer

    trainer = Trainer("ppo", env="CartPole-v1", seed=42)
    metrics = trainer.train(total_timesteps=50_000)
    print(f"Mean reward: {metrics['mean_reward']:.1f}")

Config-driven::

    from rlox import TrainingConfig, train_from_config

    config = TrainingConfig.from_yaml("config.yaml")
    metrics = train_from_config(config)
"""

# -- Rust primitives (PyO3) ---------------------------------------------------
from rlox._rlox_core import (
    CartPole,
    VecEnv,
    GymEnv,
    ExperienceTable,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    MmapReplayBuffer,
    OfflineDatasetBuffer,
    VarLenStore,
    compute_gae,
    compute_gae_batched,
    compute_gae_batched_f32,
    compute_vtrace,
    compute_group_advantages,
    compute_batch_group_advantages,
    compute_token_kl,
    compute_token_kl_schulman,
    compute_batch_token_kl,
    compute_batch_token_kl_schulman,
    compute_token_kl_f32,
    compute_token_kl_schulman_f32,
    compute_batch_token_kl_f32,
    compute_batch_token_kl_schulman_f32,
    DPOPair,
    RunningStats,
    RunningStatsVec,
    pack_sequences,
    ActorCritic,
    CandleCollector,
    # Wave 2/3: new Rust bindings
    random_shift_batch,
    shape_rewards_pbrs,
    compute_goal_distance_potentials,
    reptile_update,
    average_weight_vectors,
    SequenceReplayBuffer,
    HERBuffer,
    py_sample_mixed,
)

# -- Core Python imports (eager — these are in __all__) -----------------------
from rlox.gym_vec_env import GymVecEnv
from rlox.vec_normalize import VecNormalize
from rlox.policies import ContinuousPolicy, DiscretePolicy, AsymmetricPolicy
from rlox.config import PPOConfig, SACConfig, DQNConfig, A2CConfig, TD3Config, TRPOConfig, TrainingConfig
from rlox.trainer import Trainer, ALGORITHM_REGISTRY
from rlox.runner import train_from_config
from rlox.callbacks import Callback, EvalCallback, VideoRecordingCallback
from rlox.logging import ConsoleLogger

# ---------------------------------------------------------------------------
# Lazy imports for non-core modules (loaded on first access)
# This keeps `import rlox` fast (~0.2s instead of ~0.7s).
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """Lazy import for non-core symbols."""
    _LAZY_MAP = {
        # Protocols
        "OnPolicyActor": "rlox.protocols",
        "StochasticActor": "rlox.protocols",
        "DeterministicActor": "rlox.protocols",
        "QFunction": "rlox.protocols",
        "DiscreteQFunction": "rlox.protocols",
        "ExplorationStrategy": "rlox.protocols",
        "ReplayBufferProtocol": "rlox.protocols",
        # VecEnvProtocol is an alias for rlox.protocols.VecEnv
        "VecEnvProtocol": "rlox.protocols",  # handled specially below
        "Augmentation": "rlox.protocols",
        "RewardShaper": "rlox.protocols",
        "IntrinsicMotivation": "rlox.protocols",
        "MetaLearner": "rlox.protocols",
        # Exploration
        "GaussianNoise": "rlox.exploration",
        "EpsilonGreedy": "rlox.exploration",
        "OUNoise": "rlox.exploration",
        "GoExplore": "rlox.exploration",
        # Wrappers
        "FrameStack": "rlox.wrappers",
        "ImagePreprocess": "rlox.wrappers",
        "AtariWrapper": "rlox.wrappers",
        "DMControlWrapper": "rlox.wrappers",
        "LanguageWrapper": "rlox.wrappers",
        "GoalConditionedWrapper": "rlox.wrappers",
        # Advanced
        "RandomShift": "rlox.augmentation",
        "PotentialShaping": "rlox.reward_shaping",
        "GoalDistanceShaping": "rlox.reward_shaping",
        "RND": "rlox.intrinsic",
        "ICM": "rlox.intrinsic",
        "Reptile": "rlox.meta",
        "PBT": "rlox.pbt",
        "SelfPlay": "rlox.self_play",
        "OfflineToOnline": "rlox.offline_to_online",
        # Logging
        "LoggerCallback": "rlox.logging",
        "WandbLogger": "rlox.logging",
        "TensorBoardLogger": "rlox.logging",
        # Callbacks (non-core)
        "CallbackList": "rlox.callbacks",
        "EarlyStoppingCallback": "rlox.callbacks",
        "CheckpointCallback": "rlox.callbacks",
        "ProgressBarCallback": "rlox.callbacks",
        "TimingCallback": "rlox.callbacks",
        # Evaluation
        "interquartile_mean": "rlox.evaluation",
        "performance_profiles": "rlox.evaluation",
        "stratified_bootstrap_ci": "rlox.evaluation",
        "aggregate_metrics": "rlox.evaluation",
        "probability_of_improvement": "rlox.evaluation",
        # Dashboard
        "MetricsCollector": "rlox.dashboard",
        "TerminalDashboard": "rlox.dashboard",
        "HTMLReport": "rlox.dashboard",
        # Diagnostics
        "TrainingDiagnostics": "rlox.diagnostics",
        # Checkpoint
        "Checkpoint": "rlox.checkpoint",
        # Hub
        "push_to_hub": "rlox.hub",
        "load_from_hub": "rlox.hub",
        # Plugins
        "ENV_REGISTRY": "rlox.plugins",
        "BUFFER_REGISTRY": "rlox.plugins",
        "REWARD_REGISTRY": "rlox.plugins",
        "register_env": "rlox.plugins",
        "register_buffer": "rlox.plugins",
        "register_reward": "rlox.plugins",
        "discover_plugins": "rlox.plugins",
        "list_registered": "rlox.plugins",
        # Zoo
        "ModelZoo": "rlox.zoo",
        "ModelCard": "rlox.zoo",
        # Compile
        "compile_policy": "rlox.compile",
        "apply_spectral_norm": "rlox.networks",
        # Distributed
        "MultiGPUTrainer": "rlox.distributed",
        "RemoteEnvPool": "rlox.distributed",
        "launch_elastic": "rlox.distributed",
        # Deploy
        "generate_dockerfile": "rlox.deploy",
        "generate_k8s_job": "rlox.deploy",
        "SageMakerEstimator": "rlox.deploy",
        # Builders
        "PPOBuilder": "rlox.builders",
        "SACBuilder": "rlox.builders",
        "DQNBuilder": "rlox.builders",
        # Losses
        "LossComponent": "rlox.losses",
        "CompositeLoss": "rlox.losses",
        "PPOLoss": "rlox.losses",
        # Collectors
        "OffPolicyCollector": "rlox.off_policy_collector",
        "CollectorProtocol": "rlox.off_policy_collector",
        # Batch
        "RolloutBatch": "rlox.batch",
        # Non-core configs
        "MAPPOConfig": "rlox.config",
        "DreamerV3Config": "rlox.config",
        "IMPALAConfig": "rlox.config",
        "DecisionTransformerConfig": "rlox.config",
        "QMIXConfig": "rlox.config",
        "CalQLConfig": "rlox.config",
        "MPOConfig": "rlox.config",
        "PBTConfig": "rlox.config",
        "SelfPlayConfig": "rlox.config",
        "GoExploreConfig": "rlox.config",
        "VPGConfig": "rlox.config",
        "DiffusionPolicyConfig": "rlox.config",
        "DTPConfig": "rlox.config",
        # Non-core Rust bindings
        "RolloutCollector": "rlox.collectors",
        # Deprecated trainers
        "PPOTrainer": "rlox.trainers",
        "SACTrainer": "rlox.trainers",
        "DQNTrainer": "rlox.trainers",
        "A2CTrainer": "rlox.trainers",
        "TD3Trainer": "rlox.trainers",
        "MAPPOTrainer": "rlox.trainers",
        "DreamerV3Trainer": "rlox.trainers",
        "IMPALATrainer": "rlox.trainers",
    }
    # Aliases where the attribute name differs from the module name
    _ALIASES = {"VecEnvProtocol": ("rlox.protocols", "VecEnv")}

    if name in _ALIASES:
        import importlib
        mod_path, attr_name = _ALIASES[name]
        module = importlib.import_module(mod_path)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    if name in _LAZY_MAP:
        import importlib
        module = importlib.import_module(_LAZY_MAP[name])
        attr = getattr(module, name)
        globals()[name] = attr  # cache for next access
        return attr
    raise AttributeError(f"module 'rlox' has no attribute {name!r}")

__version__ = "1.2.0"

# ---------------------------------------------------------------------------
# Public API (__all__)
#
# Only the essentials are exported at top level. Everything else is available
# via submodule imports (e.g. ``from rlox.callbacks import EvalCallback``).
# This keeps ``dir(rlox)`` focused and discoverable.
# ---------------------------------------------------------------------------

__all__ = [
    # ---- Core (what 90% of users need) ----
    "Trainer",                  # Single entry point for all algorithms
    "TrainingConfig",           # YAML/TOML config-driven training
    "train_from_config",        # Run from config file
    "ALGORITHM_REGISTRY",       # Algorithm name → class map

    # ---- Rust primitives (data plane) ----
    "CartPole",
    "VecEnv",                   # Parallel env stepping (Rust)
    "GymEnv",
    "GymVecEnv",                # Gymnasium env wrapper
    "VecNormalize",             # Obs/reward normalization
    "ExperienceTable",
    "ReplayBuffer",             # Off-policy storage
    "PrioritizedReplayBuffer",
    "MmapReplayBuffer",
    "OfflineDatasetBuffer",
    "VarLenStore",
    "SequenceReplayBuffer",
    "HERBuffer",
    "compute_gae",              # Generalized Advantage Estimation (135x)
    "compute_gae_batched",
    "compute_gae_batched_f32",
    "compute_vtrace",
    "compute_group_advantages",
    "compute_batch_group_advantages",
    "compute_token_kl",
    "compute_token_kl_schulman",
    "compute_batch_token_kl",
    "compute_batch_token_kl_schulman",
    "compute_token_kl_f32",
    "compute_token_kl_schulman_f32",
    "compute_batch_token_kl_f32",
    "compute_batch_token_kl_schulman_f32",
    "DPOPair",
    "RunningStats",
    "RunningStatsVec",
    "pack_sequences",
    "ActorCritic",
    "CandleCollector",
    "random_shift_batch",
    "shape_rewards_pbrs",
    "compute_goal_distance_potentials",
    "reptile_update",
    "average_weight_vectors",
    "py_sample_mixed",

    # ---- Configs (6 core algorithms) ----
    "PPOConfig",
    "SACConfig",
    "DQNConfig",
    "TD3Config",
    "A2CConfig",
    "TRPOConfig",

    # ---- Policies ----
    "ContinuousPolicy",
    "DiscretePolicy",
    "AsymmetricPolicy",

    # ---- Callbacks ----
    "Callback",
    "EvalCallback",
    "VideoRecordingCallback",

    # ---- Logging ----
    "ConsoleLogger",

    # ---- Version ----
    "__version__",
]

# Everything else remains importable via submodules:
#   from rlox.algorithms import PPO, SAC, DQN
#   from rlox.callbacks import CheckpointCallback, ProgressBarCallback
#   from rlox.config import MAPPOConfig, DreamerV3Config, IMPALAConfig
#   from rlox.exploration import GaussianNoise, EpsilonGreedy
#   from rlox.intrinsic import RND, ICM
#   from rlox.meta import Reptile
#   from rlox.plugins import register_env, register_buffer
#   from rlox.wrappers import FrameStack, AtariWrapper
#   from rlox.deploy import generate_dockerfile
#   from rlox.dashboard import MetricsCollector, HTMLReport
#   from rlox.zoo import ModelZoo
#   from rlox.distributed import MultiGPUTrainer
