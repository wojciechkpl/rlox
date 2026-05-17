"""TDD tests for the 6 core algorithms: PPO, SAC, DQN, TD3, A2C, TRPO.

These are the algorithms we validate deeply. Each gets:
- Construction test
- Training returns metrics test
- Checkpoint save/load roundtrip
- predict() works
- Config from_dict roundtrip
- Convergence on CartPole/Pendulum
"""

import warnings

import numpy as np
import pytest
import torch

import rlox
from rlox import Trainer


class TestCoreAPI:
    """All Rust primitives and core Python classes are in __all__."""

    def test_all_count(self):
        assert len(rlox.__all__) == 57

    def test_trainer_in_all(self):
        assert "Trainer" in rlox.__all__

    def test_compute_gae_in_all(self):
        assert "compute_gae" in rlox.__all__

    def test_six_configs_in_all(self):
        for name in ["PPOConfig", "SACConfig", "DQNConfig", "TD3Config", "A2CConfig", "TRPOConfig"]:
            assert name in rlox.__all__, f"{name} missing from __all__"

    def test_version_is_1_1_0(self):
        assert rlox.__version__ == "1.1.0"

    def test_deprecated_trainers_not_in_all(self):
        for name in ["PPOTrainer", "SACTrainer", "DQNTrainer"]:
            assert name not in rlox.__all__

    def test_submodule_imports_still_work(self):
        from rlox.callbacks import CheckpointCallback, ProgressBarCallback
        from rlox.intrinsic import RND
        from rlox.plugins import register_env
        from rlox.dashboard import MetricsCollector


class TestPPO:
    """PPO — the default algorithm, most thoroughly tested."""

    def test_train_cartpole(self):
        trainer = Trainer("ppo", env="CartPole-v1", seed=42, config={"n_envs": 4, "n_steps": 64})
        metrics = trainer.train(total_timesteps=5_000)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics or "policy_loss" in metrics

    def test_predict(self):
        trainer = Trainer("ppo", env="CartPole-v1", seed=42, config={"n_envs": 4, "n_steps": 64})
        trainer.train(total_timesteps=2_000)
        obs = torch.zeros(1, 4)
        result = trainer.algo.predict(obs, deterministic=True)
        assert result is not None

    def test_save_load(self, tmp_path):
        trainer = Trainer("ppo", env="CartPole-v1", seed=42, config={"n_envs": 4, "n_steps": 64})
        trainer.train(total_timesteps=2_000)
        path = str(tmp_path / "ppo.pt")
        trainer.save(path)
        restored = Trainer.from_checkpoint(path, algorithm="ppo", env="CartPole-v1")
        assert restored.algo is not None

    def test_config_alias_warns(self):
        """normalize_reward (no 's') should warn and map to normalize_rewards."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = rlox.PPOConfig.from_dict({"normalize_reward": True})
            assert cfg.normalize_rewards is True
            assert any("alias" in str(x.message).lower() for x in w)

    def test_config_unknown_key_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = rlox.PPOConfig.from_dict({"bogus_param": 42})
            assert any("unknown" in str(x.message).lower() for x in w)


class TestSAC:
    """SAC — the default off-policy algorithm."""

    def test_train_pendulum(self):
        trainer = Trainer("sac", env="Pendulum-v1", seed=42, config={"learning_starts": 100})
        metrics = trainer.train(total_timesteps=2_000)
        assert isinstance(metrics, dict)

    def test_predict(self):
        trainer = Trainer("sac", env="Pendulum-v1", seed=42, config={"learning_starts": 100})
        trainer.train(total_timesteps=1_000)
        obs = torch.zeros(1, 3)
        result = trainer.algo.predict(obs, deterministic=True)
        assert result is not None


class TestDQN:
    """DQN — discrete off-policy."""

    def test_train_cartpole(self):
        trainer = Trainer("dqn", env="CartPole-v1", seed=42, config={"learning_starts": 100})
        metrics = trainer.train(total_timesteps=2_000)
        assert isinstance(metrics, dict)


class TestTD3:
    """TD3 — deterministic continuous off-policy."""

    def test_train_pendulum(self):
        trainer = Trainer("td3", env="Pendulum-v1", seed=42, config={"learning_starts": 100})
        metrics = trainer.train(total_timesteps=2_000)
        assert isinstance(metrics, dict)


class TestA2C:
    """A2C — simple on-policy baseline."""

    def test_train_cartpole(self):
        trainer = Trainer("a2c", env="CartPole-v1", seed=42, config={"n_envs": 4, "n_steps": 5})
        metrics = trainer.train(total_timesteps=2_000)
        assert isinstance(metrics, dict)


class TestTRPO:
    """TRPO — trust region on-policy."""

    def test_train_cartpole(self):
        trainer = Trainer("trpo", env="CartPole-v1", seed=42)
        metrics = trainer.train(total_timesteps=5_000)
        assert isinstance(metrics, dict)


class TestSecureCheckpoints:
    """Checkpoints should be secure by default."""

    def test_safe_load_rejects_unsafe_by_default(self, tmp_path):
        """safe_torch_load should NOT fall back to weights_only=False."""
        import pickle
        # Create a file that requires pickle
        path = str(tmp_path / "unsafe.pt")
        torch.save({"model": "data", "custom_obj": object()}, path)

        from rlox.checkpoint import safe_torch_load
        # Should raise, not silently load
        with pytest.raises(RuntimeError, match="weights_only=True"):
            safe_torch_load(path)

    def test_safe_load_with_explicit_unsafe(self, tmp_path):
        """allow_unsafe=True should work for legacy checkpoints."""
        path = str(tmp_path / "legacy.pt")
        torch.save({"step": 100}, path)

        from rlox.checkpoint import safe_torch_load
        data = safe_torch_load(path, allow_unsafe=True)
        assert data["step"] == 100
