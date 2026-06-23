"""Tests for Parallelised Q-Network (PQN).

TDD red phase: these tests are written FIRST, before any implementation.
The feature (rlox.algorithms.pqn) does not exist yet; every test here is
expected to FAIL with ImportError / AttributeError / AssertionError until
the implementation lands.

Contract (from docs/plans/pqn-design-2026-06-21.md):

  - ``from rlox.algorithms.pqn import PQN``
  - ``PQN(env_id, ...)`` — discrete action spaces only
  - ``PQN.train(total_timesteps) -> dict[str, float]`` — finite values
  - ``PQN.predict(obs, deterministic=True) -> int``
  - NO target network, NO replay buffer (the defining PQN simplification)
  - Q-network contains at least one ``torch.nn.LayerNorm`` module
  - ε decays from eps_start toward eps_end over training
  - Q(λ) targets via the existing ``rlox.compute_gae_batched`` Rust op
  - ``Trainer("pqn", env="CartPole-v1")`` resolves; ``.status == "experimental"``

Interface assumptions (the implementer MUST honour):
  - ``PQN.__init__`` accepts the signature in the design doc, plus ``**kwargs``
    for forward compatibility.
  - The Q-network is stored at ``self.q_network`` (attribute name as in DQN).
  - The current ε value is readable via ``self.epsilon`` (a float attribute
    that is updated by the training loop).
  - ``train()`` returns a dict containing at least a ``"loss"`` key with a
    finite float value.
  - ``predict(obs, deterministic=True)`` returns an ``int`` that is a valid
    action in the CartPole action space (0 or 1).
  - Constructing PQN on a continuous env (``Pendulum-v1``) raises
    ``ValueError`` or ``TypeError``.
  - ``n_envs * n_steps`` rollouts are used for Q(λ) target computation, and
    the training metrics reflect a return/target computation over that many
    transitions.

Additional interface assumptions introduced by the code-review RED tests
(TestPQNComputeTargetsDirect, TestPQNEpsGreedyReproducibility,
TestPQNConfigEpsValidation):

  - ``PQN._compute_targets(obs_arr, actions_arr, rewards_arr, terminated_arr,
        truncated_arr, terminal_obs_list, last_obs)``
    is a public-for-testing helper that replicates the rollout-to-returns
    computation inside ``train()``.  It returns
    ``(obs_flat_t, actions_flat_t, returns_t)`` — all torch.Tensors, where
    ``returns_t`` is the flat (n_steps*n_envs,) Q(λ) return array (step-major).
    See TestPQNTruncationBootstrap for exact argument shapes.

  - ``PQN._rng`` is the instance-level NumPy Generator
    (``np.random.default_rng(seed)``) used for all ε-greedy randomness inside
    ``train()`` and ``predict(deterministic=False)``.  It must be serialised
    and restored by ``save()`` / ``from_checkpoint()``.

  - ``PQNConfig.__post_init__`` must validate:
    - ``eps_start > eps_end`` (a non-decreasing schedule is nonsensical —
      raise ``ValueError``).
    - ``0.0 < exploration_fraction <= 1.0`` (raise ``ValueError`` if > 1.0).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TestPQNConstruction
# ---------------------------------------------------------------------------


class TestPQNConstruction:
    """PQN can be instantiated with canonical hyperparameters."""

    def test_pqn_constructs_with_defaults(self):
        """PQN is instantiable on CartPole-v1 with default hyperparameters."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1")
        assert pqn is not None
        assert pqn.env_id == "CartPole-v1"

    def test_pqn_constructs_with_all_hyperparams(self):
        """PQN accepts the full hyperparameter set from the design doc."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(
            env_id="CartPole-v1",
            n_envs=4,
            n_steps=16,
            learning_rate=2.5e-4,
            gamma=0.99,
            q_lambda=0.65,
            num_epochs=2,
            num_minibatches=2,
            max_grad_norm=10.0,
            weight_decay=0.0,
            hidden=64,
            eps_start=1.0,
            eps_end=0.05,
            exploration_fraction=0.5,
            seed=7,
        )
        assert pqn is not None
        assert pqn.env_id == "CartPole-v1"

    def test_pqn_env_id_stored(self):
        """PQN stores env_id as a public attribute."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert pqn.env_id == "CartPole-v1"

    def test_pqn_discrete_only_raises_on_continuous_env(self):
        """PQN must raise ValueError or TypeError when given a continuous action space.

        Assumption: Pendulum-v1 has a continuous (Box) action space, so PQN
        must detect this at construction time and raise an informative error.
        """
        from rlox.algorithms.pqn import PQN

        with pytest.raises((ValueError, TypeError)):
            PQN(env_id="Pendulum-v1")

    def test_pqn_error_message_mentions_discrete(self):
        """The error message for a continuous env should mention 'discrete'."""
        from rlox.algorithms.pqn import PQN

        with pytest.raises((ValueError, TypeError), match="[Dd]iscrete"):
            PQN(env_id="Pendulum-v1")


# ---------------------------------------------------------------------------
# TestPQNProtocol
# ---------------------------------------------------------------------------


class TestPQNProtocol:
    """PQN exposes the rlox algorithm protocol (train, predict, save, from_checkpoint)."""

    def test_pqn_has_train_method(self):
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert callable(getattr(pqn, "train", None))

    def test_pqn_has_predict_method(self):
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert callable(getattr(pqn, "predict", None))

    def test_pqn_has_save_method(self):
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert callable(getattr(pqn, "save", None))

    def test_pqn_has_from_checkpoint_classmethod(self):
        from rlox.algorithms.pqn import PQN

        assert callable(getattr(PQN, "from_checkpoint", None))


# ---------------------------------------------------------------------------
# TestPQNNoPolicySimplifications — THE heart of the PQN contract
# ---------------------------------------------------------------------------


class TestPQNNoPolicySimplifications:
    """PQN's defining simplification: no target network, no replay buffer.

    These are the invariants that make PQN *different* from DQN.  They pin
    the algorithm contract so that the implementation cannot accidentally
    drift back toward DQN.
    """

    def test_pqn_has_no_target_network_attribute(self):
        """PQN must NOT have a ``target_network`` attribute (no target net)."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert not hasattr(pqn, "target_network"), (
            "PQN must not have a target_network; LayerNorm stabilises "
            "TD learning instead."
        )

    def test_pqn_has_no_target_q_attribute(self):
        """PQN must NOT have a ``target_q`` attribute (no target net, any name)."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert not hasattr(pqn, "target_q"), (
            "PQN must not have a target_q attribute."
        )

    def test_pqn_has_no_buffer_attribute(self):
        """PQN must NOT have a ``buffer`` attribute (no replay buffer)."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert not hasattr(pqn, "buffer"), (
            "PQN must not have a buffer attribute; it is on-policy "
            "with no experience replay."
        )

    def test_pqn_has_no_replay_buffer_attribute(self):
        """PQN must NOT have a ``replay_buffer`` attribute."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert not hasattr(pqn, "replay_buffer"), (
            "PQN must not have a replay_buffer attribute."
        )


# ---------------------------------------------------------------------------
# TestPQNLayerNorm — Q-network architecture contract
# ---------------------------------------------------------------------------


class TestPQNLayerNorm:
    """The Q-network must contain at least one nn.LayerNorm module.

    LayerNorm is the load-bearing ingredient that lets PQN remove the
    target network.  Its absence would be a silent algorithmic regression.
    """

    def test_q_network_attribute_exists(self):
        """PQN exposes ``self.q_network`` as a public attribute."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert hasattr(pqn, "q_network"), (
            "PQN must expose its Q-network at self.q_network"
        )

    def test_q_network_is_nn_module(self):
        """``self.q_network`` must be a torch.nn.Module."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8)
        assert isinstance(pqn.q_network, nn.Module)

    def test_q_network_contains_layer_norm(self):
        """The Q-network must contain at least one ``nn.LayerNorm`` layer.

        This is the structural invariant that distinguishes PQN's architecture
        from a plain DQN MLP.
        """
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, hidden=64)
        has_layer_norm = any(
            isinstance(m, nn.LayerNorm) for m in pqn.q_network.modules()
        )
        assert has_layer_norm, (
            "PQN's Q-network must contain at least one nn.LayerNorm module. "
            "LayerNorm is PQN's stability mechanism in lieu of a target network."
        )


# ---------------------------------------------------------------------------
# TestPQNEpsilonSchedule — ε-greedy exploration decay
# ---------------------------------------------------------------------------


class TestPQNEpsilonSchedule:
    """ε starts at eps_start and decays toward eps_end over training."""

    def test_epsilon_attribute_exists_before_training(self):
        """PQN exposes ``self.epsilon`` as a public float attribute."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, eps_start=1.0)
        assert hasattr(pqn, "epsilon"), (
            "PQN must expose the current exploration rate as self.epsilon"
        )

    def test_epsilon_initialised_to_eps_start(self):
        """Before any training, epsilon should equal eps_start."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, eps_start=0.8)
        assert pqn.epsilon == pytest.approx(0.8, abs=1e-6), (
            f"Expected initial epsilon == eps_start=0.8, got {pqn.epsilon}"
        )

    def test_epsilon_decays_after_training(self):
        """After a non-trivial training run, epsilon has decreased from eps_start.

        We use exploration_fraction=0.9 so exploration covers most of the budget,
        ensuring the schedule has had meaningful time to decay.
        """
        from rlox.algorithms.pqn import PQN

        eps_start = 1.0
        eps_end = 0.05
        pqn = PQN(
            env_id="CartPole-v1",
            n_envs=4,
            n_steps=16,
            eps_start=eps_start,
            eps_end=eps_end,
            exploration_fraction=0.9,
            seed=42,
        )
        pqn.train(total_timesteps=4 * 16 * 4)  # 4 rollouts

        assert pqn.epsilon < eps_start, (
            f"epsilon should have decayed below eps_start={eps_start}, "
            f"got epsilon={pqn.epsilon}"
        )

    def test_epsilon_never_goes_below_eps_end(self):
        """Even after extensive training, epsilon must not drop below eps_end."""
        from rlox.algorithms.pqn import PQN

        eps_end = 0.05
        pqn = PQN(
            env_id="CartPole-v1",
            n_envs=4,
            n_steps=16,
            eps_start=1.0,
            eps_end=eps_end,
            exploration_fraction=0.1,  # fast decay
            seed=42,
        )
        # Train for much longer than the exploration_fraction suggests
        pqn.train(total_timesteps=4 * 16 * 20)

        assert pqn.epsilon >= eps_end - 1e-6, (
            f"epsilon must not go below eps_end={eps_end}, "
            f"got epsilon={pqn.epsilon}"
        )


# ---------------------------------------------------------------------------
# TestPQNTraining — train() contract
# ---------------------------------------------------------------------------


class TestPQNTraining:
    """train() runs without error and returns a well-formed metrics dict."""

    def test_train_returns_dict(self):
        """train() returns a dict."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=16, seed=0)
        result = pqn.train(total_timesteps=64)
        assert isinstance(result, dict)

    def test_train_metrics_contains_loss_key(self):
        """The metrics dict must contain a 'loss' key (Q-value MSE loss)."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=16, seed=0)
        metrics = pqn.train(total_timesteps=64)
        assert "loss" in metrics, (
            f"Expected 'loss' in metrics, got keys: {list(metrics.keys())}"
        )

    def test_train_loss_is_finite(self):
        """The returned loss value must be finite (not NaN or Inf)."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=16, seed=0)
        metrics = pqn.train(total_timesteps=64)
        loss = metrics["loss"]
        assert math.isfinite(loss), (
            f"Expected finite loss, got {loss}"
        )

    def test_train_all_metric_values_are_finite(self):
        """Every value in the returned metrics dict must be finite."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=16, seed=0)
        metrics = pqn.train(total_timesteps=64)
        non_finite = {
            k: v for k, v in metrics.items()
            if isinstance(v, float) and not math.isfinite(v)
        }
        assert not non_finite, f"Non-finite metric values: {non_finite}"

    def test_train_completes_multiple_rollouts(self):
        """train() with total_timesteps > n_envs*n_steps completes without error."""
        from rlox.algorithms.pqn import PQN

        # 3 rollouts worth of data
        n_envs, n_steps = 4, 16
        pqn = PQN(env_id="CartPole-v1", n_envs=n_envs, n_steps=n_steps, seed=1)
        metrics = pqn.train(total_timesteps=n_envs * n_steps * 3)
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# TestPQNPredict — predict() contract
# ---------------------------------------------------------------------------


class TestPQNPredict:
    """predict() returns a valid discrete action."""

    def test_predict_returns_int(self):
        """predict() returns an integer action."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=16, seed=0)
        obs = np.zeros(4, dtype=np.float32)  # CartPole obs dim = 4
        action = pqn.predict(obs, deterministic=True)
        assert isinstance(action, (int, np.integer)), (
            f"predict() should return an int, got {type(action)}"
        )

    def test_predict_action_in_action_space(self):
        """predict() returns an action in {0, 1} for CartPole."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=16, seed=0)
        obs = np.zeros(4, dtype=np.float32)
        action = pqn.predict(obs, deterministic=True)
        assert action in (0, 1), (
            f"CartPole action must be 0 or 1, got {action}"
        )

    def test_predict_deterministic_is_reproducible(self):
        """deterministic=True gives the same action for the same observation."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=16, seed=42)
        obs = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float32)
        action1 = pqn.predict(obs, deterministic=True)
        action2 = pqn.predict(obs, deterministic=True)
        assert action1 == action2, (
            "deterministic=True must return identical actions for identical obs"
        )

    def test_predict_works_after_training(self):
        """predict() is valid after a training run."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=16, seed=0)
        pqn.train(total_timesteps=64)
        obs = np.zeros(4, dtype=np.float32)
        action = pqn.predict(obs, deterministic=True)
        assert action in (0, 1)


# ---------------------------------------------------------------------------
# TestPQNQLambdaTargets — Q(λ) return computation
# ---------------------------------------------------------------------------


class TestPQNQLambdaTargets:
    """Q(λ) targets are computed correctly over the parallel rollout.

    The design doc specifies that PQN reuses ``rlox.compute_gae_batched``
    for the TD(λ) returns.  We verify the target shape and finiteness through
    the training metrics rather than coupling to internal rollout tensors.

    Assumption: the implementer stores the last computed returns (or exposes
    their count) so we can verify n_envs * n_steps targets were produced.
    We use a proxy: a short train() run with known geometry must succeed and
    report a finite loss, which implies finite targets were computed.
    """

    def test_qlambda_targets_finite_after_train(self):
        """A single-rollout train() step (n_envs*n_steps transitions) produces
        a finite Q(λ) loss — confirming target computation did not produce NaN/Inf."""
        from rlox.algorithms.pqn import PQN

        n_envs, n_steps = 4, 16
        pqn = PQN(
            env_id="CartPole-v1",
            n_envs=n_envs,
            n_steps=n_steps,
            q_lambda=0.65,
            gamma=0.99,
            seed=42,
        )
        metrics = pqn.train(total_timesteps=n_envs * n_steps)
        assert math.isfinite(metrics.get("loss", float("nan"))), (
            "Q(λ) target computation produced non-finite loss; "
            "check compute_gae_batched call and returned returns."
        )

    def test_qlambda_lambda_zero_approaches_one_step_td(self):
        """With q_lambda=0 the return degenerates to one-step TD (r + gamma*V).

        Both q_lambda=0 and q_lambda=0.65 must produce finite losses.
        We verify both run without numerical error.
        """
        from rlox.algorithms.pqn import PQN

        for lam in (0.0, 0.65, 1.0):
            pqn = PQN(
                env_id="CartPole-v1",
                n_envs=4,
                n_steps=16,
                q_lambda=lam,
                seed=0,
            )
            metrics = pqn.train(total_timesteps=64)
            assert math.isfinite(metrics.get("loss", float("nan"))), (
                f"Non-finite loss for q_lambda={lam}"
            )


# ---------------------------------------------------------------------------
# TestPQNRegistryAndStatus — Trainer integration
# ---------------------------------------------------------------------------


class TestPQNRegistryAndStatus:
    """PQN is registered in the Trainer registry with 'experimental' status."""

    def test_trainer_pqn_resolves(self):
        """Trainer('pqn', env='CartPole-v1') does not raise a ValueError."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("pqn", env="CartPole-v1")
        assert trainer is not None

    def test_trainer_pqn_status_is_experimental(self):
        """Trainer('pqn', ...).status == 'experimental'."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("pqn", env="CartPole-v1")
        assert trainer.status == "experimental", (
            f"Expected 'experimental', got {trainer.status!r}"
        )

    def test_trainer_pqn_emits_experimental_warning(self):
        """Constructing Trainer('pqn', ...) fires a UserWarning about 'experimental'."""
        from rlox.trainer import Trainer

        with pytest.warns(UserWarning, match="experimental"):
            Trainer("pqn", env="CartPole-v1")

    def test_algorithm_status_dict_contains_pqn(self):
        """ALGORITHM_STATUS must have a 'pqn' entry."""
        from rlox.trainer import ALGORITHM_STATUS

        assert "pqn" in ALGORITHM_STATUS, (
            "ALGORITHM_STATUS must include 'pqn'. "
            "Add it to _register_builtins() in trainer.py."
        )

    def test_algorithm_registry_contains_pqn(self):
        """ALGORITHM_REGISTRY must have a 'pqn' entry."""
        from rlox.trainer import ALGORITHM_REGISTRY

        assert "pqn" in ALGORITHM_REGISTRY, (
            "ALGORITHM_REGISTRY must include 'pqn'. "
            "Add it to _register_builtins() in trainer.py."
        )

    def test_algorithm_status_completeness_invariant_holds_and_pqn_present(self):
        """set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY) holds AND both contain 'pqn'.

        Fails until 'pqn' is added to both dicts AND they remain in sync.
        This combines the new-entry requirement with the completeness invariant.
        """
        from rlox.trainer import ALGORITHM_REGISTRY, ALGORITHM_STATUS

        # Both collections must include pqn (will fail before implementation)
        assert "pqn" in ALGORITHM_STATUS, (
            "'pqn' missing from ALGORITHM_STATUS — add it in trainer.py."
        )
        assert "pqn" in ALGORITHM_REGISTRY, (
            "'pqn' missing from ALGORITHM_REGISTRY — add it in _register_builtins()."
        )
        # The completeness invariant must still hold
        assert set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY), (
            f"Status/registry mismatch after pqn was added. "
            f"Missing from STATUS: {set(ALGORITHM_REGISTRY) - set(ALGORITHM_STATUS)}. "
            f"Extra in STATUS: {set(ALGORITHM_STATUS) - set(ALGORITHM_REGISTRY)}."
        )

    def test_trainer_pqn_can_train(self):
        """Trainer('pqn', ...).train() runs without error (short budget)."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer(
                "pqn",
                env="CartPole-v1",
                config={"n_envs": 2, "n_steps": 16, "seed": 0},
            )
        metrics = trainer.train(total_timesteps=64)
        assert isinstance(metrics, dict)
        assert math.isfinite(metrics.get("loss", float("nan")))


# ---------------------------------------------------------------------------
# TestPQNConfig — PQNConfig dataclass
# ---------------------------------------------------------------------------


class TestPQNConfig:
    """PQNConfig dataclass is importable and serialisable."""

    def test_pqn_config_importable(self):
        """PQNConfig is importable from rlox.config."""
        from rlox.config import PQNConfig  # noqa: F401

    def test_pqn_config_has_expected_fields(self):
        """PQNConfig carries the canonical PQN hyperparameters."""
        from rlox.config import PQNConfig

        cfg = PQNConfig()
        for field in (
            "learning_rate",
            "gamma",
            "q_lambda",
            "n_envs",
            "n_steps",
            "num_epochs",
            "num_minibatches",
            "max_grad_norm",
            "weight_decay",
            "hidden",
            "eps_start",
            "eps_end",
            "exploration_fraction",
        ):
            assert hasattr(cfg, field), (
                f"PQNConfig missing expected field: {field!r}"
            )

    def test_pqn_config_defaults(self):
        """PQNConfig default values match the design doc."""
        from rlox.config import PQNConfig

        cfg = PQNConfig()
        assert cfg.q_lambda == pytest.approx(0.65)
        assert cfg.eps_start == pytest.approx(1.0)
        assert cfg.eps_end == pytest.approx(0.05)
        assert cfg.exploration_fraction == pytest.approx(0.5)
        assert cfg.n_envs == 8
        assert cfg.n_steps == 32
        assert cfg.gamma == pytest.approx(0.99)
        assert cfg.learning_rate == pytest.approx(2.5e-4)

    def test_pqn_config_roundtrip(self):
        """PQNConfig serialises to dict and back without loss."""
        from rlox.config import PQNConfig

        cfg = PQNConfig(learning_rate=1e-3, q_lambda=0.8, hidden=256)
        d = cfg.to_dict()
        cfg2 = PQNConfig.from_dict(d)
        assert cfg2.learning_rate == pytest.approx(1e-3)
        assert cfg2.q_lambda == pytest.approx(0.8)
        assert cfg2.hidden == 256


# ---------------------------------------------------------------------------
# TestPQNSaveLoad — checkpoint contract
# ---------------------------------------------------------------------------


class TestPQNSaveLoad:
    """PQN.save() / from_checkpoint() round-trip."""

    def test_pqn_save_and_load(self, tmp_path):
        """PQN can be saved and restored; env_id is preserved."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, seed=0)
        pqn.train(total_timesteps=32)

        ckpt = str(tmp_path / "pqn.pt")
        pqn.save(ckpt)
        pqn2 = PQN.from_checkpoint(ckpt, env_id="CartPole-v1")
        assert pqn2.env_id == "CartPole-v1"

    def test_loaded_pqn_can_predict(self, tmp_path):
        """A PQN restored from checkpoint can call predict() without error."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, seed=0)
        pqn.train(total_timesteps=32)

        ckpt = str(tmp_path / "pqn2.pt")
        pqn.save(ckpt)
        pqn2 = PQN.from_checkpoint(ckpt, env_id="CartPole-v1")

        obs = np.zeros(4, dtype=np.float32)
        action = pqn2.predict(obs, deterministic=True)
        assert action in (0, 1)


# ---------------------------------------------------------------------------
# TestPQNConvergence — marked slow, not run in the fast suite
# ---------------------------------------------------------------------------


class TestPQNConvergence:
    """PQN learns CartPole above random level (smoke convergence).

    These tests are expensive; mark with @pytest.mark.slow so they are
    excluded from the default fast suite.
    """

    @pytest.mark.slow
    def test_pqn_learns_above_random_cartpole(self):
        """After a modest training budget PQN achieves mean_reward > 20 on CartPole.

        CartPole random baseline ≈ 10–20 steps. Threshold of 50 is a low bar
        that proves the Q-values are doing something useful.
        """
        from rlox.algorithms.pqn import PQN

        pqn = PQN(
            env_id="CartPole-v1",
            n_envs=8,
            n_steps=32,
            learning_rate=2.5e-4,
            q_lambda=0.65,
            eps_start=1.0,
            eps_end=0.05,
            exploration_fraction=0.5,
            seed=42,
        )
        metrics = pqn.train(total_timesteps=50_000)
        mean_reward = metrics.get("mean_reward", 0.0)
        assert mean_reward > 50, (
            f"Expected mean_reward > 50 after 50k steps, got {mean_reward:.1f}. "
            "PQN may not be learning on CartPole."
        )


# ---------------------------------------------------------------------------
# Helpers for truncation bootstrap tests
# ---------------------------------------------------------------------------


class _AlwaysTruncatesEnv(gym.Env):
    """Minimal Gymnasium env that always truncates after ``max_steps`` steps.

    Contract:
    - Discrete action space (2 actions) so PQN accepts it.
    - Observation is a constant non-trivial vector so V(s) can be non-zero.
    - Always truncates (time-limit); NEVER terminates.
    - Constant positive reward of +1.0 per step so the discounted bootstrap
      contribution is guaranteed to be strictly positive when gamma > 0.
    """

    metadata: dict[str, Any] = {}

    def __init__(self, max_steps: int = 4):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self._max_steps = max_steps
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = np.array([0.5, -0.5, 0.3, -0.3], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.array([0.5, -0.5, 0.3, -0.3], dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = self._step_count >= self._max_steps
        if truncated:
            self._step_count = 0  # auto-reset semantics handled by VecEnv wrapper
        return obs, reward, terminated, truncated, {}


def _make_always_truncates_env(max_steps: int = 4):
    """Factory thunk for gymnasium.vector.SyncVectorEnv."""
    def _thunk():
        return _AlwaysTruncatesEnv(max_steps=max_steps)
    return _thunk


# ---------------------------------------------------------------------------
# TestPQNTruncationBootstrap — replaced by TestPQNComputeTargetsDirect
# ---------------------------------------------------------------------------

# NOTE: The original TestPQNTruncationBootstrap class used a _StubVecEnv that
# wraps SyncVectorEnv without AutoresetMode.SAME_STEP.  That wrapper never
# provides terminal_obs (SyncVectorEnv puts final_observation in infos, not
# final_obs, and only under gymnasium >= 0.26 autoresets), so terminal_obs_list
# is always [None, None, ...] in those tests.  The augmentation block in
# _compute_targets is therefore never reached — the tests would pass even if the
# entire ``gamma * V(terminal_obs)`` block were deleted.
#
# TestPQNComputeTargetsDirect replaces them with direct, robust tests of the
# _compute_targets seam using hand-crafted inputs.  The helper class
# _AlwaysTruncatesEnv is retained because it is still used elsewhere.


class TestPQNComputeTargetsDirect:
    """Direct tests of PQN._compute_targets with hand-crafted inputs.

    These tests bypass the VecEnv entirely and call _compute_targets with
    numpy arrays constructed in the test body, so terminal_obs values are
    explicit Python ndarrays — not None — wherever the augmentation must fire.

    The expected target values are computed by calling pqn.q_network directly
    under torch.no_grad(), making every assertion exact and init-independent.
    The tests remain valid regardless of whether the implementer removes the
    hardcoded optimistic bias-init (nn.init.constant_(final_linear.bias, 0.5)).

    Interface assumptions honoured here (implementer MUST preserve):
      - _compute_targets(obs_arr, actions_arr, rewards_arr, terminated_arr,
            truncated_arr, terminal_obs_list, last_obs) -> (obs_flat_t,
            actions_flat_t, returns_t)
      - obs_arr      : (n_steps, n_envs, obs_dim) float32 ndarray
      - rewards_arr  : (n_steps, n_envs) float64 ndarray
      - terminated_arr, truncated_arr : (n_steps, n_envs) bool ndarray
      - terminal_obs_list[t][i] : ndarray or None
      - returns_t shape: (n_steps * n_envs,) step-major
        (step t, env i -> index t*n_envs + i)
    """

    # ------------------------------------------------------------------
    # Shared geometry: CartPole-v1, obs_dim=4, n_actions=2.
    # Small rollout: n_steps=4, n_envs=2.
    # ------------------------------------------------------------------

    _OBS_DIM = 4
    _N_ENVS = 2
    _N_STEPS = 4
    _GAMMA = 0.99
    _TERMINAL_OBS = np.array([0.5, -0.5, 0.3, -0.3], dtype=np.float32)

    def _make_pqn(self, seed: int = 0, gamma: float = 0.99, q_lambda: float = 0.65):
        """Construct a small PQN on CartPole-v1 with known geometry."""
        from rlox.algorithms.pqn import PQN

        return PQN(
            env_id="CartPole-v1",
            n_envs=self._N_ENVS,
            n_steps=self._N_STEPS,
            gamma=gamma,
            q_lambda=q_lambda,
            seed=seed,
        )

    def _make_rollout_arrays(
        self,
        *,
        n_steps: int | None = None,
        n_envs: int | None = None,
        rewards: float = 1.0,
    ):
        """Return zeroed-out rollout arrays with constant reward."""
        n_steps = n_steps or self._N_STEPS
        n_envs = n_envs or self._N_ENVS
        obs_arr = np.zeros((n_steps, n_envs, self._OBS_DIM), dtype=np.float32)
        actions_arr = np.zeros((n_steps, n_envs), dtype=np.int64)
        rewards_arr = np.full((n_steps, n_envs), rewards, dtype=np.float64)
        terminated_arr = np.zeros((n_steps, n_envs), dtype=bool)
        truncated_arr = np.zeros((n_steps, n_envs), dtype=bool)
        terminal_obs_list = [[None] * n_envs for _ in range(n_steps)]
        last_obs = np.zeros((n_envs, self._OBS_DIM), dtype=np.float32)
        return (
            obs_arr, actions_arr, rewards_arr,
            terminated_arr, truncated_arr, terminal_obs_list, last_obs,
        )

    def _q_max(self, pqn, obs: np.ndarray) -> float:
        """Return max_a Q(obs, a) from the live q_network — no grad."""
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            return pqn.q_network(obs_t).max(dim=-1).values.item()

    # ------------------------------------------------------------------
    # Test 1: augmentation is applied when terminal_obs is a real array
    # ------------------------------------------------------------------

    def test_truncation_augmentation_applied_when_terminal_obs_present(self):
        """With a real terminal_obs, the return at the truncated step is
        exactly ``base_return + gamma * max_a Q(terminal_obs, a)``.

        The test runs _compute_targets twice:
          - ``with_obs``: terminal_obs_list[t][i] = TERMINAL_OBS (ndarray)
          - ``without_obs``: same rollout but terminal_obs_list[t][i] = None

        It asserts:
          returns_with[idx] - returns_without[idx] == gamma * Q(terminal_obs)

        The difference must be strictly positive (i.e. the augmentation
        contributes a non-zero amount).  We skip if Q(terminal_obs) is
        negligibly close to zero (astronomically unlikely with random init;
        noted here so the implementer knows it is not a logic flaw if it
        triggers after removing optimistic bias-init).

        This test FAILS if the ``gamma * V(terminal_obs)`` augmentation block
        in _compute_targets is deleted.
        """
        from rlox.algorithms.pqn import PQN

        # Truncation at step t=2, env 0.  All other cells are normal.
        trunc_t, trunc_e = 2, 0
        pqn = self._make_pqn(seed=0, gamma=self._GAMMA, q_lambda=0.65)

        (
            obs_arr, actions_arr, rewards_arr,
            terminated_arr, truncated_arr, terminal_obs_list, last_obs,
        ) = self._make_rollout_arrays()

        truncated_arr[trunc_t, trunc_e] = True

        # Build two terminal_obs_list variants: one with the real obs, one with None.
        terminal_obs_with = [row[:] for row in terminal_obs_list]  # shallow copy rows
        terminal_obs_with[trunc_t][trunc_e] = self._TERMINAL_OBS.copy()

        terminal_obs_without = [row[:] for row in terminal_obs_list]  # all-None

        # Compute expected augmentation directly from the Q-network.
        q_terminal = self._q_max(pqn, self._TERMINAL_OBS)
        expected_aug = self._GAMMA * q_terminal

        if abs(expected_aug) < 1e-8:
            pytest.skip(
                f"gamma * Q(terminal_obs) = {expected_aug:.2e} is negligibly "
                "close to zero for this random init — augmentation contribution "
                "cannot be distinguished numerically.  Re-run with a different "
                "seed if this triggers unexpectedly."
            )

        _, _, returns_with = pqn._compute_targets(
            obs_arr=obs_arr,
            actions_arr=actions_arr,
            rewards_arr=rewards_arr,
            terminated_arr=terminated_arr,
            truncated_arr=truncated_arr,
            terminal_obs_list=terminal_obs_with,
            last_obs=last_obs,
        )
        _, _, returns_without = pqn._compute_targets(
            obs_arr=obs_arr,
            actions_arr=actions_arr,
            rewards_arr=rewards_arr,
            terminated_arr=terminated_arr,
            truncated_arr=truncated_arr,
            terminal_obs_list=terminal_obs_without,
            last_obs=last_obs,
        )

        # Step-major index: t * n_envs + e
        idx = trunc_t * self._N_ENVS + trunc_e
        val_with = returns_with[idx].item()
        val_without = returns_without[idx].item()
        actual_diff = val_with - val_without

        assert actual_diff == pytest.approx(expected_aug, abs=1e-4), (
            f"_compute_targets did not apply the truncation augmentation. "
            f"Expected returns_with[{idx}] - returns_without[{idx}] = "
            f"gamma * Q(terminal_obs) = {self._GAMMA} * {q_terminal:.6f} "
            f"= {expected_aug:.6f}, but got diff = {actual_diff:.6f}. "
            "Check that the augmentation block adds "
            "``gamma * max_a Q(terminal_obs, a)`` to rewards_aug[t, i] "
            "when truncated_arr[t, i] and not terminated_arr[t, i] "
            "and terminal_obs_list[t][i] is not None."
        )

        # Strictly positive: the with-terminal_obs return must be larger
        # (assuming expected_aug > 0; skip handles the degenerate case above).
        if expected_aug > 0:
            assert val_with > val_without, (
                f"returns_with[{idx}]={val_with:.6f} must be strictly greater "
                f"than returns_without[{idx}]={val_without:.6f}."
            )

    def test_truncation_augmentation_propagates_to_earlier_steps(self):
        """The augmentation at a truncated step affects earlier steps via λ-returns.

        When truncation occurs at step t=2, the augmented reward is folded
        into the multi-step return via compute_gae_batched.  Steps t=0 and t=1
        in the same environment therefore also carry a larger return in the
        with-terminal_obs case.

        Assert: for env i=0, returns_with[t * n_envs + 0] >
                              returns_without[t * n_envs + 0]
        for all t < trunc_t (steps before the truncation).
        """
        from rlox.algorithms.pqn import PQN

        trunc_t, trunc_e = 2, 0
        pqn = self._make_pqn(seed=0, gamma=self._GAMMA, q_lambda=0.65)

        (
            obs_arr, actions_arr, rewards_arr,
            terminated_arr, truncated_arr, terminal_obs_list, last_obs,
        ) = self._make_rollout_arrays()

        truncated_arr[trunc_t, trunc_e] = True
        terminal_obs_with = [row[:] for row in terminal_obs_list]
        terminal_obs_with[trunc_t][trunc_e] = self._TERMINAL_OBS.copy()
        terminal_obs_without = [row[:] for row in terminal_obs_list]

        q_terminal = self._q_max(pqn, self._TERMINAL_OBS)
        if abs(self._GAMMA * q_terminal) < 1e-8:
            pytest.skip("Q(terminal_obs) negligibly small; propagation undetectable.")

        _, _, returns_with = pqn._compute_targets(
            obs_arr, actions_arr, rewards_arr, terminated_arr, truncated_arr,
            terminal_obs_with, last_obs,
        )
        _, _, returns_without = pqn._compute_targets(
            obs_arr, actions_arr, rewards_arr, terminated_arr, truncated_arr,
            terminal_obs_without, last_obs,
        )

        # Earlier steps of the same env must also show a larger return.
        for t in range(trunc_t):
            idx = t * self._N_ENVS + trunc_e
            val_with = returns_with[idx].item()
            val_without = returns_without[idx].item()
            # Direction of inequality matches sign of gamma * Q(terminal_obs).
            if self._GAMMA * q_terminal > 0:
                assert val_with > val_without, (
                    f"At step t={t}, env {trunc_e}: expected returns_with[{idx}] "
                    f"({val_with:.6f}) > returns_without[{idx}] ({val_without:.6f}) "
                    "because truncation augmentation at t=2 should propagate "
                    "backward through the lambda-return."
                )
            else:
                assert val_with < val_without, (
                    f"At step t={t}, env {trunc_e}: expected returns_with[{idx}] "
                    f"({val_with:.6f}) < returns_without[{idx}] ({val_without:.6f}) "
                    "because truncation augmentation (negative Q) propagates backward."
                )

    # ------------------------------------------------------------------
    # Test 2: terminated steps do NOT receive the bootstrap augmentation
    # ------------------------------------------------------------------

    def test_terminated_step_does_not_get_truncation_augmentation(self):
        """terminated=True must NOT trigger the gamma*V(terminal_obs) augmentation.

        Even when terminal_obs_list[t][i] is a real ndarray and truncated_arr[t, i]
        is True, if terminated_arr[t, i] is ALSO True the augmentation must be
        suppressed (the episode ended by natural termination — there is no
        continuation state to bootstrap from).

        The contract: augmentation fires only when
          ``truncated AND NOT terminated AND terminal_obs is not None``.

        Observable consequence: for a step that is both truncated and terminated,
        the return must equal the return of an otherwise-identical rollout where
        terminal_obs is None (because the augmentation is blocked by terminated=True).
        """
        from rlox.algorithms.pqn import PQN

        term_t, term_e = 1, 0
        pqn = self._make_pqn(seed=0, gamma=self._GAMMA, q_lambda=0.65)

        (
            obs_arr, actions_arr, rewards_arr,
            terminated_arr, truncated_arr, terminal_obs_list, last_obs,
        ) = self._make_rollout_arrays()

        # Both flags set: episode is terminated (natural end), also truncated
        # in the sense of time-limit, but terminated takes priority.
        terminated_arr[term_t, term_e] = True
        truncated_arr[term_t, term_e] = True  # set truncated too — augmentation must NOT fire

        # Provide a real terminal_obs — if the augmentation fires incorrectly it
        # will change the return, which we detect below.
        terminal_obs_list_real = [row[:] for row in terminal_obs_list]
        terminal_obs_list_real[term_t][term_e] = self._TERMINAL_OBS.copy()

        # Baseline: no terminal_obs at all.
        terminal_obs_list_none = [row[:] for row in terminal_obs_list]

        _, _, returns_real = pqn._compute_targets(
            obs_arr, actions_arr, rewards_arr, terminated_arr, truncated_arr,
            terminal_obs_list_real, last_obs,
        )
        _, _, returns_none = pqn._compute_targets(
            obs_arr, actions_arr, rewards_arr, terminated_arr, truncated_arr,
            terminal_obs_list_none, last_obs,
        )

        idx = term_t * self._N_ENVS + term_e
        assert returns_real[idx].item() == pytest.approx(
            returns_none[idx].item(), abs=1e-6
        ), (
            f"_compute_targets applied the truncation augmentation at "
            f"(t={term_t}, e={term_e}) even though terminated=True. "
            f"returns_real[{idx}]={returns_real[idx].item():.6f}, "
            f"returns_none[{idx}]={returns_none[idx].item():.6f}. "
            "The augmentation must be suppressed when terminated_arr[t,i] is True, "
            "regardless of truncated_arr[t,i] or terminal_obs."
        )

    def test_terminated_step_return_equals_immediate_reward_with_no_future(self):
        """A step that terminates with no continuation has return == reward.

        For a single-step rollout (n_steps=1) where the only step terminates
        with reward R:
          - The GAE backend receives done=1, zeroing the bootstrap.
          - No future steps exist to propagate backward.
          - Therefore returns_t[0] must equal R exactly.

        This test pins the behaviour that terminated episodes have a
        self-contained return (no bootstrap from any source).
        """
        from rlox.algorithms.pqn import PQN

        reward_val = 7.0
        pqn = self._make_pqn(seed=3, gamma=self._GAMMA, q_lambda=0.65)

        n_steps, n_envs = 1, 1
        obs_arr = np.zeros((n_steps, n_envs, self._OBS_DIM), dtype=np.float32)
        actions_arr = np.zeros((n_steps, n_envs), dtype=np.int64)
        rewards_arr = np.array([[reward_val]], dtype=np.float64)
        terminated_arr = np.array([[True]])
        truncated_arr = np.array([[False]])
        terminal_obs_list = [[None]]
        last_obs = np.zeros((n_envs, self._OBS_DIM), dtype=np.float32)

        _, _, returns_t = pqn._compute_targets(
            obs_arr, actions_arr, rewards_arr, terminated_arr, truncated_arr,
            terminal_obs_list, last_obs,
        )

        assert returns_t.shape == (1,), (
            f"Expected shape (1,), got {returns_t.shape}"
        )
        assert returns_t[0].item() == pytest.approx(reward_val, abs=1e-5), (
            f"Terminated step with no continuation: return must equal the "
            f"immediate reward {reward_val}. Got {returns_t[0].item():.6f}. "
            "Check that done=1 is passed to GAE for terminated steps, "
            "zeroing the bootstrap value."
        )

    # ------------------------------------------------------------------
    # Test 3: truncated (no termination) propagates future return unlike
    #         a terminated step (distinguishing the two flags)
    # ------------------------------------------------------------------

    def test_truncated_not_terminated_propagates_future_return(self):
        """A truncated-but-not-terminated step propagates the future return;
        a terminated step does not.

        Setup: 2-step rollout, 1 env.  Step t=0 has reward=1.0.  Step t=1
        has reward=2.0.  last_obs = zeros.

        Case A (terminated at t=0): GAE sees done=1 at t=0, so the return at
          t=0 is effectively isolated from future rewards.
          return_A[0] ≈ 1.0  (immediate reward, no bootstrap from t=1).

        Case B (truncated at t=0, no terminal_obs): GAE sees done=0 at t=0
          (only terminated passes to GAE as dones).  The λ-return at t=0
          folds in the future value from t=1.
          return_B[0] > return_A[0].

        The strict inequality return_B[0] > return_A[0] is the observable
        contract that distinguishes the two flags in _compute_targets.
        """
        from rlox.algorithms.pqn import PQN

        # Use 1 env so the step-major index equals the step index.
        pqn = self._make_pqn(seed=7, gamma=self._GAMMA, q_lambda=1.0)
        n_steps, n_envs = 2, 1

        obs_arr = np.zeros((n_steps, n_envs, self._OBS_DIM), dtype=np.float32)
        actions_arr = np.zeros((n_steps, n_envs), dtype=np.int64)
        rewards_arr = np.array([[1.0], [2.0]], dtype=np.float64)
        last_obs = np.zeros((n_envs, self._OBS_DIM), dtype=np.float32)
        terminal_obs_none = [[None], [None]]

        # Case A: terminated at t=0.
        term_A = np.array([[True], [False]])
        trunc_A = np.array([[False], [False]])
        _, _, returns_A = pqn._compute_targets(
            obs_arr, actions_arr, rewards_arr, term_A, trunc_A, terminal_obs_none, last_obs,
        )

        # Case B: truncated (not terminated) at t=0.
        term_B = np.array([[False], [False]])
        trunc_B = np.array([[True], [False]])
        _, _, returns_B = pqn._compute_targets(
            obs_arr, actions_arr, rewards_arr, term_B, trunc_B, terminal_obs_none, last_obs,
        )

        ret_A_t0 = returns_A[0].item()
        ret_B_t0 = returns_B[0].item()

        # Terminated step: return should equal immediate reward (1.0).
        assert ret_A_t0 == pytest.approx(1.0, abs=1e-5), (
            f"Terminated step t=0 return must be 1.0 (immediate reward only). "
            f"Got {ret_A_t0:.6f}."
        )

        # Truncated step: return must fold in future reward (> 1.0).
        assert ret_B_t0 > ret_A_t0, (
            f"Truncated-not-terminated step t=0 return ({ret_B_t0:.6f}) must "
            f"exceed terminated step return ({ret_A_t0:.6f}).  With terminated "
            "the GAE zeroes the bootstrap; with truncated it does not (since "
            "_compute_targets passes only `terminated` as dones to GAE, not "
            "`terminated | truncated`)."
        )

    # ------------------------------------------------------------------
    # Sanity: helper method exists and returns correct tensor shapes
    # ------------------------------------------------------------------

    def test_compute_targets_helper_exists(self):
        """PQN must expose _compute_targets as a callable method."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=4, seed=0)
        assert callable(getattr(pqn, "_compute_targets", None)), (
            "PQN must expose ``_compute_targets(obs_arr, actions_arr, "
            "rewards_arr, terminated_arr, truncated_arr, terminal_obs_list, "
            "last_obs)`` as a testable seam for the Q(λ) target computation."
        )

    def test_compute_targets_returns_correct_shapes(self):
        """_compute_targets returns tensors with the right shapes."""
        from rlox.algorithms.pqn import PQN

        n_steps, n_envs = 3, 2
        obs_dim = 4
        pqn = PQN(env_id="CartPole-v1", n_envs=n_envs, n_steps=n_steps, seed=0)

        obs_arr = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        actions_arr = np.zeros((n_steps, n_envs), dtype=np.int64)
        rewards_arr = np.ones((n_steps, n_envs), dtype=np.float64)
        terminated_arr = np.zeros((n_steps, n_envs), dtype=bool)
        truncated_arr = np.zeros((n_steps, n_envs), dtype=bool)
        terminal_obs_list = [[None] * n_envs for _ in range(n_steps)]
        last_obs = np.zeros((n_envs, obs_dim), dtype=np.float32)

        obs_t, actions_t, returns_t = pqn._compute_targets(
            obs_arr, actions_arr, rewards_arr, terminated_arr, truncated_arr,
            terminal_obs_list, last_obs,
        )

        assert obs_t.shape == (n_steps * n_envs, obs_dim), (
            f"obs_flat_t shape: expected ({n_steps * n_envs}, {obs_dim}), "
            f"got {tuple(obs_t.shape)}"
        )
        assert actions_t.shape == (n_steps * n_envs,), (
            f"actions_flat_t shape: expected ({n_steps * n_envs},), "
            f"got {tuple(actions_t.shape)}"
        )
        assert returns_t.shape == (n_steps * n_envs,), (
            f"returns_t shape: expected ({n_steps * n_envs},), "
            f"got {tuple(returns_t.shape)}"
        )
        assert torch.isfinite(returns_t).all(), (
            f"returns_t contains non-finite values: {returns_t}"
        )


# ---------------------------------------------------------------------------
# TestPQNEpsGreedyReproducibility
# ---------------------------------------------------------------------------


class TestPQNEpsGreedyReproducibility:
    """ε-greedy draws are seed-deterministic and survive checkpoint round-trips.

    Root cause of the current non-reproducibility:
        pqn.py uses ``np.random.random(n_envs)`` and
        ``np.random.randint(0, self.n_actions, size=(n_envs,))`` — the module-
        level global NumPy legacy RNG.  This RNG state is not bound to ``seed``
        and not serialised by ``save()``.  Two PQN instances with the same seed
        will diverge as soon as the global RNG state differs between runs.

    Required fix (implementer must honour):
        - Replace global ``np.random`` calls in the training loop and
          ``predict(deterministic=False)`` with ``self._rng`` where
          ``self._rng = np.random.default_rng(seed)`` is set in ``__init__``.
        - ``save()`` must persist ``self._rng.bit_generator.state`` (a plain
          dict, JSON-serialisable).
        - ``from_checkpoint()`` must restore ``self._rng`` from that state so
          that subsequent draws continue the identical stream.

    These tests pin those three obligations.
    """

    @staticmethod
    def _collect_explore_decisions(pqn, n_steps: int) -> list[bool]:
        """Simulate ``n_steps`` ε-greedy decisions using the PQN RNG.

        We isolate the ε-greedy decision stream from any stochasticity in the
        environment or Q-network by calling the RNG directly via ``self._rng``
        (the seam the implementer must provide).  This lets us verify the RNG
        stream independently of environment or network output.

        Returns a list of booleans: True = explore (random action), False = exploit.
        """
        # Access the seeded RNG instance directly.
        # If _rng doesn't exist the test fails with AttributeError (correct RED).
        rng = pqn._rng
        # Replicate the ε-greedy mask logic: explore if random() < epsilon.
        # Use the PQN's current epsilon (may have been set by training).
        epsilon = pqn.epsilon
        return [bool(rng.random() < epsilon) for _ in range(n_steps)]

    def test_pqn_has_rng_attribute(self):
        """PQN must have a ``self._rng`` attribute (np.random.Generator).

        This is the seam that isolates ε-greedy randomness from the global
        NumPy RNG, enabling per-instance seed-determinism and checkpoint
        restoration.
        """
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, seed=7)
        assert hasattr(pqn, "_rng"), (
            "PQN must have a ``self._rng`` attribute: "
            "``self._rng = np.random.default_rng(seed)`` in ``__init__``. "
            "This is required for seed-deterministic ε-greedy exploration."
        )

    def test_rng_is_numpy_generator(self):
        """``self._rng`` must be a ``numpy.random.Generator`` instance."""
        from rlox.algorithms.pqn import PQN

        pqn = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, seed=7)
        assert isinstance(pqn._rng, np.random.Generator), (
            f"Expected self._rng to be np.random.Generator, "
            f"got {type(pqn._rng)}"
        )

    def test_two_instances_same_seed_produce_identical_rng_stream(self):
        """Two PQN instances initialised with the same seed have identical RNG streams.

        We advance both RNGs by the same number of draws and compare the
        results.  If both use ``np.random.default_rng(seed)``, they start from
        the same state and produce the same stream.  If they share/use the
        global RNG they will diverge.
        """
        from rlox.algorithms.pqn import PQN

        seed = 42
        n_draws = 20
        pqn_a = PQN(env_id="CartPole-v1", n_envs=4, n_steps=8, seed=seed)
        pqn_b = PQN(env_id="CartPole-v1", n_envs=4, n_steps=8, seed=seed)

        stream_a = [pqn_a._rng.random() for _ in range(n_draws)]
        stream_b = [pqn_b._rng.random() for _ in range(n_draws)]

        assert stream_a == stream_b, (
            "Two PQN instances with the same seed produced different RNG streams. "
            "Each instance must initialise its own ``np.random.default_rng(seed)`` "
            "and never share or fall back to the global RNG."
        )

    def test_different_seeds_produce_different_rng_streams(self):
        """Two PQN instances with different seeds produce different RNG streams."""
        from rlox.algorithms.pqn import PQN

        pqn_a = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, seed=1)
        pqn_b = PQN(env_id="CartPole-v1", n_envs=2, n_steps=8, seed=2)

        stream_a = [pqn_a._rng.random() for _ in range(10)]
        stream_b = [pqn_b._rng.random() for _ in range(10)]

        assert stream_a != stream_b, (
            "PQN instances with different seeds must produce different RNG streams."
        )

    def test_checkpoint_round_trip_restores_rng_stream(self, tmp_path):
        """After ``save()`` → ``from_checkpoint()``, the RNG stream is restored.

        Protocol:
        1. Create PQN with seed=S, run a short training step (advances RNG).
        2. Save checkpoint; record the next N draws from the live instance (_rng_a).
        3. Restore from checkpoint (fresh PQN instance).
        4. Advance restored RNG by the same N draws (_rng_b).
        5. Assert _rng_a == _rng_b.

        The test fails until ``save()`` persists ``_rng.bit_generator.state``
        and ``from_checkpoint()`` restores it.
        """
        from rlox.algorithms.pqn import PQN

        n_envs = 2
        n_steps = 8
        seed = 13
        pqn_orig = PQN(
            env_id="CartPole-v1", n_envs=n_envs, n_steps=n_steps, seed=seed
        )
        pqn_orig.train(total_timesteps=n_envs * n_steps)  # advance the RNG

        ckpt = str(tmp_path / "pqn_rng_test.pt")
        pqn_orig.save(ckpt)

        # Record the next 20 draws from the live (unsaved) continuation.
        n_probe = 20
        stream_orig = [pqn_orig._rng.random() for _ in range(n_probe)]

        # Restore and record the same 20 draws.
        pqn_restored = PQN.from_checkpoint(ckpt, env_id="CartPole-v1")
        stream_restored = [pqn_restored._rng.random() for _ in range(n_probe)]

        assert stream_orig == stream_restored, (
            "RNG stream after checkpoint restore does not match the original. "
            "``save()`` must persist ``self._rng.bit_generator.state`` and "
            "``from_checkpoint()`` must restore it so the ε-greedy draw "
            "sequence is identical to a non-interrupted run."
        )

    def test_train_epsilon_greedy_is_seed_reproducible(self):
        """Two identical short training runs on the same seed produce the same epsilon.

        This is a weaker but end-to-end check: because ε is only a function of
        elapsed steps and fixed hyperparams it is always deterministic already.
        The more important check is the RNG stream above.  We include this as a
        regression guard for the exploration_fraction arithmetic.
        """
        from rlox.algorithms.pqn import PQN

        n_envs = 2
        n_steps = 8
        total = n_envs * n_steps * 2  # 2 rollouts

        pqn_a = PQN(
            env_id="CartPole-v1",
            n_envs=n_envs,
            n_steps=n_steps,
            seed=99,
            eps_start=1.0,
            eps_end=0.1,
            exploration_fraction=0.5,
        )
        pqn_b = PQN(
            env_id="CartPole-v1",
            n_envs=n_envs,
            n_steps=n_steps,
            seed=99,
            eps_start=1.0,
            eps_end=0.1,
            exploration_fraction=0.5,
        )

        pqn_a.train(total_timesteps=total)
        pqn_b.train(total_timesteps=total)

        assert pqn_a.epsilon == pytest.approx(pqn_b.epsilon, abs=1e-8), (
            f"Epsilon after training must be identical for same-seed instances. "
            f"Got eps_a={pqn_a.epsilon}, eps_b={pqn_b.epsilon}."
        )


# ---------------------------------------------------------------------------
# TestPQNConfigEpsValidation
# ---------------------------------------------------------------------------


class TestPQNConfigEpsValidation:
    """PQNConfig must validate the ε schedule and exploration_fraction.

    The current ``__post_init__`` does not check:
    - ``eps_start > eps_end`` (inverted schedule — meaningless or harmful)
    - ``exploration_fraction <= 1.0`` (>1.0 is nonsensical: "explore for 150%
      of training" — the schedule never reaches eps_end)

    Adding these guards prevents silent misconfiguration that would produce
    a rising ε schedule or a schedule that never finishes decaying.
    """

    def test_pqn_config_raises_when_eps_start_less_than_eps_end(self):
        """PQNConfig(eps_start=0.05, eps_end=1.0) must raise ValueError.

        eps_start < eps_end means ε would have to *increase* over training,
        which is the opposite of annealing exploration.  This is always a
        misconfiguration and must be caught eagerly at construction time.
        """
        from rlox.config import PQNConfig

        with pytest.raises(ValueError, match="eps"):
            PQNConfig(eps_start=0.05, eps_end=1.0)

    def test_pqn_config_raises_when_eps_start_equal_eps_end_zero(self):
        """PQNConfig(eps_start=0.0, eps_end=0.0) must raise ValueError.

        A completely degenerate schedule (constant ε=0) is likely a mistake.
        This pins that eps_start must be strictly greater than eps_end (or
        at minimum eps_start >= eps_end > 0).  The test uses the extreme case
        of both zero to ensure the validator fires.

        Assumption: the validator checks ``eps_start > eps_end``; since 0.0 is
        not greater than 0.0 this raises.  If the implementer only checks
        strict ordering, this test is satisfied by that check.
        """
        from rlox.config import PQNConfig

        with pytest.raises(ValueError, match="eps"):
            PQNConfig(eps_start=0.0, eps_end=0.0)

    def test_pqn_config_raises_when_exploration_fraction_above_one(self):
        """PQNConfig(exploration_fraction=1.5) must raise ValueError.

        exploration_fraction > 1.0 is nonsensical: it would mean exploring for
        150% of the training budget, so ε never reaches eps_end within the run.
        """
        from rlox.config import PQNConfig

        with pytest.raises(ValueError, match="exploration_fraction"):
            PQNConfig(exploration_fraction=1.5)

    def test_pqn_config_raises_when_exploration_fraction_zero(self):
        """PQNConfig(exploration_fraction=0.0) must raise ValueError.

        exploration_fraction=0 means no exploration budget at all — ε would
        jump to eps_end immediately, which discards the entire exploration phase.
        The contract requires a strictly positive fraction.
        """
        from rlox.config import PQNConfig

        with pytest.raises(ValueError, match="exploration_fraction"):
            PQNConfig(exploration_fraction=0.0)

    def test_pqn_config_raises_when_exploration_fraction_negative(self):
        """PQNConfig(exploration_fraction=-0.1) must raise ValueError."""
        from rlox.config import PQNConfig

        with pytest.raises(ValueError, match="exploration_fraction"):
            PQNConfig(exploration_fraction=-0.1)

    def test_pqn_config_valid_eps_schedule_does_not_raise(self):
        """PQNConfig with eps_start > eps_end and 0 < exploration_fraction <= 1
        must NOT raise — this is the common case."""
        from rlox.config import PQNConfig

        # These are all valid configurations; none should raise.
        PQNConfig(eps_start=1.0, eps_end=0.05, exploration_fraction=0.5)
        PQNConfig(eps_start=0.5, eps_end=0.1, exploration_fraction=1.0)
        PQNConfig(eps_start=0.2, eps_end=0.01, exploration_fraction=0.01)

    def test_pqn_construction_forwards_config_validation(self):
        """PQN(eps_start=0.05, eps_end=1.0) must also raise ValueError.

        PQN.__init__ builds a PQNConfig internally; the validation in
        PQNConfig.__post_init__ must therefore propagate to the PQN constructor.
        This test ensures the end-to-end path raises, not just the config alone.
        """
        from rlox.algorithms.pqn import PQN

        with pytest.raises(ValueError, match="eps"):
            PQN(env_id="CartPole-v1", eps_start=0.05, eps_end=1.0)

    def test_pqn_construction_forwards_exploration_fraction_validation(self):
        """PQN(exploration_fraction=2.0) must raise ValueError."""
        from rlox.algorithms.pqn import PQN

        with pytest.raises(ValueError, match="exploration_fraction"):
            PQN(env_id="CartPole-v1", exploration_fraction=2.0)
