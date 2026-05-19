"""API 1.0 stability tests.

Verify that all public symbols are importable, that the version is a string,
and that key interfaces (configs, trainers, callbacks) expose the expected
methods.
"""

from __future__ import annotations

import importlib

import pytest

import rlox


# =========================================================================
# Public API surface
# =========================================================================


class TestAllPublicSymbolsInAll:
    """Every symbol listed in rlox.__all__ must actually be importable."""

    def test_all_public_symbols_in_all(self):
        missing = []
        for name in rlox.__all__:
            obj = getattr(rlox, name, None)
            if obj is None:
                missing.append(name)
        assert missing == [], f"Symbols in __all__ but not importable: {missing}"

    def test_no_duplicates_in_all(self):
        seen: set[str] = set()
        dupes: list[str] = []
        for name in rlox.__all__:
            if name in seen:
                dupes.append(name)
            seen.add(name)
        assert dupes == [], f"Duplicate entries in __all__: {dupes}"


class TestVersionIsString:
    def test_version_is_string(self):
        assert isinstance(rlox.__version__, str)

    def test_version_is_semver_like(self):
        parts = rlox.__version__.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_version_is_1_2_0(self):
        assert rlox.__version__ == "1.2.0"


# =========================================================================
# Rust primitives
# =========================================================================


class TestRustPrimitivesImportable:
    """All Rust bindings should import correctly from rlox."""

    RUST_SYMBOLS = [
        "CartPole",
        "VecEnv",
        "GymEnv",
        "ExperienceTable",
        "ReplayBuffer",
        "PrioritizedReplayBuffer",
        "MmapReplayBuffer",
        "OfflineDatasetBuffer",
        "VarLenStore",
        "compute_gae",
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
    ]

    @pytest.mark.parametrize("symbol", RUST_SYMBOLS)
    def test_rust_primitives_importable(self, symbol: str):
        obj = getattr(rlox, symbol, None)
        assert obj is not None, f"Rust primitive '{symbol}' not importable from rlox"


# =========================================================================
# Python layer
# =========================================================================


class TestPythonLayerImportable:
    """All Python layer classes should import correctly from rlox."""

    PYTHON_SYMBOLS = [
        "RolloutBatch",
        "RolloutCollector",
        "GymVecEnv",
        "VecNormalize",
        "PPOLoss",
        "ContinuousPolicy",
        "DiscretePolicy",
        "PPOConfig",
        "SACConfig",
        "DQNConfig",
        "A2CConfig",
        "TD3Config",
        "MAPPOConfig",
        "DreamerV3Config",
        "IMPALAConfig",
        "TrainingConfig",
        "train_from_config",
        "Callback",
        "CallbackList",
        "EvalCallback",
        "EarlyStoppingCallback",
        "CheckpointCallback",
        "ProgressBarCallback",
        "TimingCallback",
        "LoggerCallback",
        "WandbLogger",
        "TensorBoardLogger",
        "ConsoleLogger",
        "interquartile_mean",
        "performance_profiles",
        "stratified_bootstrap_ci",
        "aggregate_metrics",
        "probability_of_improvement",
        "TrainingDiagnostics",
        "MetricsCollector",
        "TerminalDashboard",
        "HTMLReport",
        "Checkpoint",
        "push_to_hub",
        "load_from_hub",
        "compile_policy",
        "VecEnvProtocol",
        # Protocols
        "OnPolicyActor",
        "StochasticActor",
        "DeterministicActor",
        "QFunction",
        "DiscreteQFunction",
        "ExplorationStrategy",
        "ReplayBufferProtocol",
        # Exploration
        "GaussianNoise",
        "EpsilonGreedy",
        "OUNoise",
        # Off-policy
        "OffPolicyCollector",
        "CollectorProtocol",
        # Builders
        "PPOBuilder",
        "SACBuilder",
        "DQNBuilder",
        # Losses
        "LossComponent",
        "CompositeLoss",
        # Distributed
        "MultiGPUTrainer",
        "RemoteEnvPool",
        "launch_elastic",
    ]

    @pytest.mark.parametrize("symbol", PYTHON_SYMBOLS)
    def test_python_layer_importable(self, symbol: str):
        obj = getattr(rlox, symbol, None)
        assert obj is not None, f"Python symbol '{symbol}' not importable from rlox"


# =========================================================================
# Algorithms
# =========================================================================


class TestAlgorithmsImportable:
    """All algorithm classes should import from rlox.algorithms."""

    ALGORITHM_NAMES = [
        "PPO",
        "A2C",
        "SAC",
        "TD3",
        "DQN",
        "GRPO",
        "DPO",
        "OnlineDPO",
        "BestOfN",
        "MAPPO",
        "DreamerV3",
        "IMPALA",
    ]

    @pytest.mark.parametrize("name", ALGORITHM_NAMES)
    def test_algorithms_importable(self, name: str):
        mod = importlib.import_module("rlox.algorithms")
        obj = getattr(mod, name, None)
        assert obj is not None, f"Algorithm '{name}' not importable from rlox.algorithms"


# =========================================================================
# Configs
# =========================================================================


class TestConfigsHaveFromDictAndToDict:
    """All config classes must have from_dict, to_dict, merge,
    from_yaml, and to_yaml."""

    CONFIG_CLASSES = [
        rlox.PPOConfig,
        rlox.SACConfig,
        rlox.DQNConfig,
        rlox.A2CConfig,
        rlox.TD3Config,
        rlox.MAPPOConfig,
        rlox.DreamerV3Config,
        rlox.IMPALAConfig,
    ]

    REQUIRED_METHODS = ["from_dict", "to_dict", "merge", "from_yaml", "to_yaml"]

    @pytest.mark.parametrize("cls", CONFIG_CLASSES, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("method", REQUIRED_METHODS)
    def test_configs_have_from_dict_and_to_dict(self, cls: type, method: str):
        assert hasattr(cls, method), f"{cls.__name__} missing method: {method}"
        assert callable(getattr(cls, method))

    @pytest.mark.parametrize("cls", CONFIG_CLASSES, ids=lambda c: c.__name__)
    def test_config_roundtrip(self, cls: type):
        """from_dict(to_dict()) should produce an equivalent config."""
        original = cls() if cls is not rlox.MAPPOConfig else cls(n_agents=2)
        d = original.to_dict()
        restored = cls.from_dict(d)
        assert original.to_dict() == restored.to_dict()


# =========================================================================
# Trainers
# =========================================================================


class TestTrainersHaveTrainMethod:
    """All trainers must have train() and be importable."""

    def _get_trainer_class(self, name: str) -> type:
        mod = importlib.import_module("rlox.trainers")
        return getattr(mod, name)

    TRAINER_NAMES = [
        "PPOTrainer",
        "SACTrainer",
        "DQNTrainer",
        "A2CTrainer",
        "TD3Trainer",
        "MAPPOTrainer",
        "DreamerV3Trainer",
        "IMPALATrainer",
    ]

    @pytest.mark.parametrize("name", TRAINER_NAMES)
    def test_trainers_have_train_method(self, name: str):
        cls = self._get_trainer_class(name)
        assert hasattr(cls, "train"), f"{name} missing train() method"
        assert callable(getattr(cls, "train"))

    @pytest.mark.parametrize("name", TRAINER_NAMES)
    def test_trainers_importable(self, name: str):
        cls = self._get_trainer_class(name)
        assert cls is not None

    @pytest.mark.parametrize("name", TRAINER_NAMES)
    def test_trainers_have_save_method(self, name: str):
        cls = self._get_trainer_class(name)
        assert hasattr(cls, "save"), f"{name} missing save() method"

    @pytest.mark.parametrize("name", TRAINER_NAMES)
    def test_trainers_have_from_checkpoint(self, name: str):
        cls = self._get_trainer_class(name)
        assert hasattr(cls, "from_checkpoint"), f"{name} missing from_checkpoint() classmethod"

    # Deprecated trainers are importable from rlox.trainers but NOT in __all__
    TRAINER_EXPORTS = [
        "PPOTrainer",
        "SACTrainer",
        "DQNTrainer",
        "A2CTrainer",
        "TD3Trainer",
        "MAPPOTrainer",
        "DreamerV3Trainer",
        "IMPALATrainer",
    ]

    @pytest.mark.parametrize("name", TRAINER_EXPORTS)
    def test_deprecated_trainers_importable_from_submodule(self, name: str):
        """Deprecated trainers are still importable from rlox.trainers."""
        import importlib
        mod = importlib.import_module("rlox.trainers")
        assert hasattr(mod, name), f"{name} not importable from rlox.trainers"

    def test_trainer_in_rlox_all(self):
        """The unified Trainer is the only trainer in __all__."""
        assert "Trainer" in rlox.__all__
        # Deprecated per-algorithm trainers should NOT be in __all__
        for name in self.TRAINER_EXPORTS:
            assert name not in rlox.__all__, f"{name} should not be in __all__ (deprecated)"


# =========================================================================
# Callback protocol
# =========================================================================


class TestCallbackProtocol:
    """All callback classes must have on_training_start, on_step,
    on_training_end."""

    CALLBACK_CLASSES = [
        rlox.Callback,
        rlox.EvalCallback,
        rlox.EarlyStoppingCallback,
        rlox.CheckpointCallback,
        rlox.TrainingDiagnostics,
    ]

    REQUIRED_METHODS = ["on_training_start", "on_step", "on_training_end"]

    @pytest.mark.parametrize(
        "cls", CALLBACK_CLASSES, ids=lambda c: c.__name__
    )
    @pytest.mark.parametrize("method", REQUIRED_METHODS)
    def test_callback_protocol(self, cls: type, method: str):
        assert hasattr(cls, method), f"{cls.__name__} missing method: {method}"
        assert callable(getattr(cls, method))

    @pytest.mark.parametrize(
        "cls", CALLBACK_CLASSES, ids=lambda c: c.__name__
    )
    def test_callback_on_step_returns_bool(self, cls: type):
        """on_step() should return a bool (True = continue)."""
        instance = cls()
        result = instance.on_step()
        assert isinstance(result, bool)


# =========================================================================
# Distributed
# =========================================================================


class TestDistributedImportable:
    """Distributed components must be importable from rlox."""

    DISTRIBUTED_SYMBOLS = [
        "MultiGPUTrainer",
        "RemoteEnvPool",
        "launch_elastic",
    ]

    @pytest.mark.parametrize("symbol", DISTRIBUTED_SYMBOLS)
    def test_distributed_importable(self, symbol: str):
        obj = getattr(rlox, symbol, None)
        assert obj is not None, f"Distributed symbol '{symbol}' not importable from rlox"


class TestRemoteEnvPool:
    def test_importable(self):
        from rlox.distributed import RemoteEnvPool
        assert RemoteEnvPool is not None

    def test_empty_addresses_raises(self):
        from rlox.distributed import RemoteEnvPool
        with pytest.raises(ValueError):
            RemoteEnvPool(addresses=[])


# =========================================================================
# Runner dispatch completeness
# =========================================================================


class TestRunnerAlgoMap:
    """train_from_config runner must support all algorithms."""

    EXPECTED_ALGOS = ["ppo", "sac", "dqn", "a2c", "td3", "mappo", "dreamer", "impala"]

    @pytest.mark.parametrize("algo", EXPECTED_ALGOS)
    def test_runner_knows_algo(self, algo: str):
        from rlox.trainer import ALGORITHM_REGISTRY
        assert algo in ALGORITHM_REGISTRY, (
            f"Algorithm '{algo}' missing from ALGORITHM_REGISTRY"
        )
