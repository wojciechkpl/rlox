"""Tests for CrossQ (ICLR 2024) — SAC minus target networks, plus BatchRenorm critics.

TDD red phase: these tests are written FIRST, before any implementation.
The feature (rlox.algorithms.crossq) does not exist yet; every test here is
expected to FAIL with ImportError / AttributeError / AssertionError until
the implementation lands.

Contract (from docs/plans/crossq-design-2026-06-23.md):

  - ``from rlox.algorithms.crossq import CrossQ``
  - ``CrossQ(env_id, ...)`` — continuous action spaces ONLY
  - ``CrossQ.train(total_timesteps) -> dict[str, float]`` — finite values
  - ``CrossQ.predict(obs, deterministic=True) -> np.ndarray`` — in bounds
  - NO target critics, NO tau/polyak (defining CrossQ simplification)
  - Critic networks contain ``rlox.networks.BatchRenorm1d`` modules
  - ``policy_delay`` (default 3) actually delays actor updates relative to critic
  - ``Trainer("crossq", env="Pendulum-v1").status == "experimental"``
  - Constructing on discrete env raises ``ValueError`` or ``TypeError``

Interface assumptions the implementer MUST honour:

  - ``CrossQ.__init__`` accepts the full signature from the design doc:
    ``(env_id, learning_rate=1e-3, gamma=0.99, batch_size=256,
      learning_starts=1000, train_freq=1, gradient_steps=1, policy_delay=3,
      hidden=256, auto_entropy=True, ent_coef="auto", bn_momentum=0.01,
      bn_eps=1e-3, renorm_warmup_steps=100_000, seed=42, ...)``
  - Critics are stored at ``self.critic1`` and ``self.critic2`` (no ``*_target``
    counterparts — the NO-target invariant).
  - ``tau`` must NOT be set as an attribute on CrossQ.
  - ``BatchRenorm1d`` lives at ``rlox.networks.BatchRenorm1d``.
  - Actor update counter is exposed as ``self._n_actor_updates`` (int).
  - Critic update counter is exposed as ``self._n_updates`` (int).
  - Both counters increment during ``train()``.
  - ``Trainer("crossq", ...)`` routes through the standard registry and emits
    a ``UserWarning`` because status is ``"experimental"``.
"""

from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TestCrossQConstruction
# ---------------------------------------------------------------------------


class TestCrossQConstruction:
    """CrossQ can be instantiated with canonical hyperparameters."""

    def test_crossq_constructs_with_defaults(self):
        """CrossQ is instantiable on Pendulum-v1 with default hyperparameters."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert crossq is not None

    def test_crossq_stores_env_id(self):
        """CrossQ stores env_id as a public attribute."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert crossq.env_id == "Pendulum-v1"

    def test_crossq_constructs_with_all_hyperparams(self):
        """CrossQ accepts the full hyperparameter set from the design doc."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1",
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=64,
            learning_starts=100,
            train_freq=1,
            gradient_steps=1,
            policy_delay=3,
            hidden=64,
            auto_entropy=True,
            ent_coef="auto",
            bn_momentum=0.01,
            bn_eps=1e-3,
            renorm_warmup_steps=100_000,
            seed=7,
        )
        assert crossq is not None
        assert crossq.env_id == "Pendulum-v1"

    def test_crossq_continuous_only_raises_on_discrete_env(self):
        """CrossQ must raise ValueError or TypeError when given a discrete action space.

        CartPole-v1 has a Discrete action space.  CrossQ requires continuous.
        """
        from rlox.algorithms.crossq import CrossQ

        with pytest.raises((ValueError, TypeError)):
            CrossQ(env_id="CartPole-v1")

    def test_crossq_discrete_env_error_mentions_continuous_or_discrete(self):
        """The error message for a discrete env should mention 'continuous' or 'discrete'."""
        from rlox.algorithms.crossq import CrossQ

        with pytest.raises((ValueError, TypeError), match="[Cc]ontinuous|[Dd]iscrete"):
            CrossQ(env_id="CartPole-v1")

    def test_crossq_has_train_method(self):
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert callable(getattr(crossq, "train", None))

    def test_crossq_has_predict_method(self):
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert callable(getattr(crossq, "predict", None))

    def test_crossq_has_save_method(self):
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert callable(getattr(crossq, "save", None))

    def test_crossq_has_from_checkpoint_classmethod(self):
        from rlox.algorithms.crossq import CrossQ

        assert callable(getattr(CrossQ, "from_checkpoint", None))


# ---------------------------------------------------------------------------
# TestCrossQNoTargetNetworks — THE heart of the CrossQ contract
# ---------------------------------------------------------------------------


class TestCrossQNoTargetNetworks:
    """CrossQ's defining simplification: no target critics, no polyak/tau.

    These tests pin the invariants that separate CrossQ from SAC.  If the
    implementation accidentally adds target networks, these tests catch it.
    """

    def test_crossq_has_no_critic1_target_attribute(self):
        """CrossQ must NOT have a ``critic1_target`` attribute."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert not hasattr(crossq, "critic1_target"), (
            "CrossQ must not have critic1_target: CrossQ removes target networks entirely. "
            "BatchRenorm stabilises TD learning without polyak targets."
        )

    def test_crossq_has_no_critic2_target_attribute(self):
        """CrossQ must NOT have a ``critic2_target`` attribute."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert not hasattr(crossq, "critic2_target"), (
            "CrossQ must not have critic2_target."
        )

    def test_crossq_has_no_critic_target_attribute(self):
        """CrossQ must NOT have a generic ``critic_target`` attribute."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert not hasattr(crossq, "critic_target"), (
            "CrossQ must not have critic_target."
        )

    def test_crossq_has_no_tau_attribute(self):
        """CrossQ must NOT have a ``tau`` (polyak) attribute.

        Tau controls the soft target-network update rate in SAC.  CrossQ
        abolishes target networks entirely, so tau has no meaning.
        """
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert not hasattr(crossq, "tau"), (
            "CrossQ must not have a tau attribute: polyak soft-updates are "
            "SAC-specific and do not apply to CrossQ."
        )

    def test_crossq_has_live_critic1_and_critic2(self):
        """CrossQ does have ``self.critic1`` and ``self.critic2`` (the live critics)."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert hasattr(crossq, "critic1"), "CrossQ must expose self.critic1"
        assert hasattr(crossq, "critic2"), "CrossQ must expose self.critic2"

    def test_crossq_critics_are_nn_modules(self):
        """Both critics must be ``torch.nn.Module`` instances."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert isinstance(crossq.critic1, nn.Module)
        assert isinstance(crossq.critic2, nn.Module)

    def test_sac_has_target_critics_to_confirm_crossq_removes_them(self):
        """SAC (the baseline) DOES have critic targets — confirms the negative assertion
        in CrossQ tests is testing something real."""
        from rlox.algorithms.sac import SAC

        sac = SAC(env_id="Pendulum-v1")
        assert hasattr(sac, "critic1_target"), (
            "SAC must have critic1_target; this confirms CrossQ's absence is intentional."
        )
        assert hasattr(sac, "tau"), (
            "SAC must have tau; this confirms CrossQ's absence is intentional."
        )


# ---------------------------------------------------------------------------
# TestCrossQBatchRenorm — critic must contain BatchRenorm1d
# ---------------------------------------------------------------------------


class TestCrossQBatchRenorm:
    """CrossQ's critic networks contain BatchRenorm1d modules.

    This is the load-bearing ingredient: BatchRenorm replaces the target
    network as the TD-stability mechanism.  Its absence is an algorithmic
    regression.
    """

    def test_critic1_contains_batch_renorm1d(self):
        """CrossQ's critic1 must contain at least one ``BatchRenorm1d`` module."""
        from rlox.algorithms.crossq import CrossQ
        from rlox.networks import BatchRenorm1d

        crossq = CrossQ(env_id="Pendulum-v1", hidden=64)
        has_renorm = any(
            isinstance(m, BatchRenorm1d) for m in crossq.critic1.modules()
        )
        assert has_renorm, (
            "CrossQ critic1 must contain at least one BatchRenorm1d module. "
            "BatchRenorm is CrossQ's stability mechanism replacing target networks."
        )

    def test_critic2_contains_batch_renorm1d(self):
        """CrossQ's critic2 must contain at least one ``BatchRenorm1d`` module."""
        from rlox.algorithms.crossq import CrossQ
        from rlox.networks import BatchRenorm1d

        crossq = CrossQ(env_id="Pendulum-v1", hidden=64)
        has_renorm = any(
            isinstance(m, BatchRenorm1d) for m in crossq.critic2.modules()
        )
        assert has_renorm, (
            "CrossQ critic2 must contain at least one BatchRenorm1d module."
        )

    def test_plain_sac_qnetwork_has_no_batch_renorm1d(self):
        """A plain SAC QNetwork does NOT contain BatchRenorm1d.

        This confirms the positive assertion in the CrossQ tests is meaningful —
        adding BatchRenorm is a real structural change from the SAC baseline.
        """
        from rlox.networks import BatchRenorm1d, QNetwork

        # Observe the plain QNetwork from SAC: it uses ReLU activations, no BN.
        qnet = QNetwork(obs_dim=3, act_dim=1, hidden=64)
        has_renorm = any(isinstance(m, BatchRenorm1d) for m in qnet.modules())
        assert not has_renorm, (
            "Plain SAC QNetwork should NOT have BatchRenorm1d — "
            "confirming that CrossQ's BatchRenorm is a genuine addition."
        )

    def test_batch_renorm1d_is_importable(self):
        """``BatchRenorm1d`` is importable from ``rlox.networks``."""
        from rlox.networks import BatchRenorm1d  # noqa: F401

    def test_batch_renorm1d_is_nn_module(self):
        """``BatchRenorm1d`` is a subclass of ``torch.nn.Module``."""
        from rlox.networks import BatchRenorm1d

        assert issubclass(BatchRenorm1d, nn.Module), (
            "BatchRenorm1d must be a torch.nn.Module subclass."
        )


# ---------------------------------------------------------------------------
# TestBatchRenorm1dUnit — standalone unit tests for BatchRenorm1d behaviour
# ---------------------------------------------------------------------------


class TestBatchRenorm1dUnit:
    """Unit tests for ``rlox.networks.BatchRenorm1d``.

    Contract (Ioffe 2017 + design doc):
      - ``BatchRenorm1d(num_features, momentum=0.01, eps=1e-3,
          warmup_steps=100_000)``
      - ``train()`` mode: normalises using BATCH statistics; updates running
        mean/var after each forward pass.
      - ``eval()`` mode: uses FROZEN running mean/var (does NOT update them).
      - Output shape == input shape (no spatial collapse).
      - During warmup (step_count < warmup_steps): behaves like standard
        BatchNorm (r=1, d=0 corrections).
    """

    def _make_br(
        self,
        num_features: int = 8,
        momentum: float = 0.01,
        eps: float = 1e-3,
        warmup_steps: int = 100_000,
    ):
        from rlox.networks import BatchRenorm1d

        return BatchRenorm1d(
            num_features,
            momentum=momentum,
            eps=eps,
            warmup_steps=warmup_steps,
        )

    def test_construction(self):
        """BatchRenorm1d is constructible with required ``num_features`` arg."""
        br = self._make_br(num_features=16)
        assert br is not None

    def test_output_shape_matches_input_shape_2d(self):
        """Output shape == input shape for a 2-D (batch, features) tensor."""
        br = self._make_br(num_features=8)
        x = torch.randn(32, 8)
        y = br(x)
        assert y.shape == x.shape, (
            f"Expected output shape {x.shape}, got {y.shape}"
        )

    def test_output_shape_matches_input_shape_different_batch(self):
        """Output shape is preserved for varying batch sizes."""
        br = self._make_br(num_features=4)
        for batch_size in (1, 8, 128):
            x = torch.randn(batch_size, 4)
            y = br(x)
            assert y.shape == x.shape, (
                f"Batch size {batch_size}: expected {x.shape}, got {y.shape}"
            )

    def test_train_mode_output_is_finite(self):
        """Forward pass in train mode produces finite outputs."""
        br = self._make_br(num_features=8)
        br.train()
        x = torch.randn(32, 8)
        y = br(x)
        assert torch.isfinite(y).all(), "BatchRenorm1d train-mode output contains NaN/Inf"

    def test_eval_mode_output_is_finite(self):
        """Forward pass in eval mode produces finite outputs (after warmup running stats)."""
        br = self._make_br(num_features=8)
        # Warm up running stats with a few train-mode passes.
        br.train()
        for _ in range(10):
            br(torch.randn(32, 8))

        br.eval()
        x = torch.randn(32, 8)
        y = br(x)
        assert torch.isfinite(y).all(), "BatchRenorm1d eval-mode output contains NaN/Inf"

    def test_running_mean_updates_in_train_mode(self):
        """Running mean changes after several train-mode forward passes."""
        br = self._make_br(num_features=4, momentum=0.1)
        br.train()

        # Record initial running mean.
        initial_mean = br.running_mean.clone()

        # Feed batches with a nonzero mean.
        x = torch.ones(64, 4) * 5.0  # constant input far from zero
        for _ in range(5):
            br(x)

        assert not torch.allclose(br.running_mean, initial_mean, atol=1e-6), (
            "Running mean must update after train-mode forward passes. "
            "Expected it to shift toward 5.0 from the initial value."
        )

    def test_running_mean_frozen_in_eval_mode(self):
        """Running mean does NOT change during eval-mode forward passes."""
        br = self._make_br(num_features=4, momentum=0.1)
        # Warm up running stats.
        br.train()
        for _ in range(5):
            br(torch.randn(64, 4))

        br.eval()
        mean_before = br.running_mean.clone()

        # Several eval-mode passes with a shifted distribution.
        for _ in range(10):
            br(torch.ones(64, 4) * 99.0)

        assert torch.allclose(br.running_mean, mean_before, atol=1e-9), (
            "Running mean must NOT update in eval mode. "
            f"Before: {mean_before.tolist()}, After: {br.running_mean.tolist()}"
        )

    def test_train_vs_eval_outputs_differ_after_warmup(self):
        """Train-mode and eval-mode outputs differ for the same input.

        After running stats have been updated (several train-mode passes with
        a non-unit-normal distribution), train mode uses batch statistics and
        eval mode uses running statistics.  For an off-distribution input the
        two normalisation paths diverge.
        """
        br = self._make_br(num_features=4, momentum=0.5)
        br.train()

        # Train on a highly shifted distribution to set running stats far from
        # the evaluation input.
        for _ in range(20):
            br(torch.ones(128, 4) * 50.0)

        # Now evaluate with a different distribution.
        x_eval = torch.randn(64, 4)  # mean ~0 — very different from running mean ~50

        br.train()
        out_train = br(x_eval).detach()

        br.eval()
        out_eval = br(x_eval).detach()

        assert not torch.allclose(out_train, out_eval, atol=1e-4), (
            "Train-mode and eval-mode outputs must differ when batch statistics "
            "differ from running statistics. Got identical outputs — check that "
            "eval mode uses running_mean/running_var, not batch statistics."
        )

    def test_warmup_behaves_like_batchnorm(self):
        """During warmup (step_count very small), output matches nn.BatchNorm1d.

        When warmup_steps is very large (no renorm corrections yet), the
        module should behave identically to standard BatchNorm1d in train mode.
        We set warmup_steps=1_000_000 to ensure we are firmly in the warmup
        regime after just a few forward passes.
        """
        from rlox.networks import BatchRenorm1d

        num_features = 4
        eps = 1e-5
        momentum = 0.1

        br = BatchRenorm1d(
            num_features, momentum=momentum, eps=eps, warmup_steps=1_000_000
        )
        bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

        # Copy weights/biases from BatchRenorm1d to BatchNorm1d for fair comparison.
        with torch.no_grad():
            bn.weight.copy_(br.weight)
            bn.bias.copy_(br.bias)

        br.train()
        bn.train()

        x = torch.randn(64, num_features)
        out_br = br(x)
        out_bn = bn(x)

        assert torch.allclose(out_br, out_bn, atol=1e-5), (
            "During warmup, BatchRenorm1d must produce the same output as "
            "standard BatchNorm1d (r=1, d=0 corrections). "
            f"Max abs diff: {(out_br - out_bn).abs().max().item():.2e}"
        )

    def test_running_var_attribute_exists(self):
        """BatchRenorm1d exposes ``running_mean`` and ``running_var`` as buffers."""
        from rlox.networks import BatchRenorm1d

        br = BatchRenorm1d(8)
        assert hasattr(br, "running_mean"), "BatchRenorm1d must expose running_mean"
        assert hasattr(br, "running_var"), "BatchRenorm1d must expose running_var"


# ---------------------------------------------------------------------------
# TestCrossQPolicyDelay — actor update cadence
# ---------------------------------------------------------------------------


class TestCrossQPolicyDelay:
    """``policy_delay`` actually delays actor updates relative to critic updates.

    Contract:
      - Over N critic updates, the actor optimizer steps ~ N // policy_delay times.
      - The implementer MUST expose:
          - ``self._n_updates``        (int): total critic update steps taken
          - ``self._n_actor_updates``  (int): total actor optimizer steps taken
      - After a train() call that causes K critic updates with policy_delay=D:
          floor(K / D) <= _n_actor_updates <= ceil(K / D)
    """

    def test_n_updates_counter_exists(self):
        """CrossQ exposes ``self._n_updates`` as an integer counter."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1", learning_starts=50, seed=0)
        assert hasattr(crossq, "_n_updates"), (
            "CrossQ must expose self._n_updates (int) counting critic update steps."
        )

    def test_n_actor_updates_counter_exists(self):
        """CrossQ exposes ``self._n_actor_updates`` as an integer counter."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1", learning_starts=50, seed=0)
        assert hasattr(crossq, "_n_actor_updates"), (
            "CrossQ must expose self._n_actor_updates (int) counting actor optimizer steps."
        )

    def test_counters_start_at_zero(self):
        """Both counters are zero before training."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1", learning_starts=50, seed=0)
        assert crossq._n_updates == 0, (
            f"Expected _n_updates=0 before training, got {crossq._n_updates}"
        )
        assert crossq._n_actor_updates == 0, (
            f"Expected _n_actor_updates=0 before training, got {crossq._n_actor_updates}"
        )

    def test_policy_delay_3_actor_updates_less_than_critic_updates(self):
        """With policy_delay=3, actor updates must be substantially fewer than critic.

        Strategy: run a short train() so that multiple critic updates happen.
        Then assert _n_actor_updates < _n_updates, and the ratio is
        approximately 1/policy_delay.
        """
        from rlox.algorithms.crossq import CrossQ

        policy_delay = 3
        crossq = CrossQ(
            env_id="Pendulum-v1",
            learning_starts=100,
            batch_size=32,
            policy_delay=policy_delay,
            seed=0,
        )
        crossq.train(total_timesteps=300)

        n_critic = crossq._n_updates
        n_actor = crossq._n_actor_updates

        # Sanity: at least some updates must have happened.
        assert n_critic > 0, (
            f"Expected >0 critic updates after training, got _n_updates={n_critic}"
        )

        # Actor must update LESS than critic (it is delayed by policy_delay).
        assert n_actor < n_critic, (
            f"Actor updates ({n_actor}) must be fewer than critic updates ({n_critic}) "
            f"when policy_delay={policy_delay}."
        )

        # Actor steps should be approximately n_critic // policy_delay.
        expected_actor = n_critic // policy_delay
        # Allow ±1 tolerance for off-by-one at boundaries.
        assert abs(n_actor - expected_actor) <= 1, (
            f"With policy_delay={policy_delay} and {n_critic} critic updates, "
            f"expected ~{expected_actor} actor updates (±1). "
            f"Got _n_actor_updates={n_actor}."
        )

    def test_policy_delay_1_actor_updates_equal_critic_updates(self):
        """With policy_delay=1, actor updates == critic updates (no delay)."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1",
            learning_starts=100,
            batch_size=32,
            policy_delay=1,
            seed=0,
        )
        crossq.train(total_timesteps=250)

        n_critic = crossq._n_updates
        n_actor = crossq._n_actor_updates

        assert n_critic > 0, "Expected >0 critic updates."
        assert n_actor == n_critic, (
            f"With policy_delay=1, actor and critic update counts must be equal. "
            f"Got _n_updates={n_critic}, _n_actor_updates={n_actor}."
        )


# ---------------------------------------------------------------------------
# TestCrossQTraining — train() contract
# ---------------------------------------------------------------------------


class TestCrossQTraining:
    """train() runs without error and returns a well-formed metrics dict."""

    def test_train_returns_dict(self):
        """train() returns a dict."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        result = crossq.train(total_timesteps=200)
        assert isinstance(result, dict)

    def test_train_metrics_contains_critic_loss_key(self):
        """The metrics dict must contain a 'critic_loss' key."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        metrics = crossq.train(total_timesteps=200)
        assert "critic_loss" in metrics, (
            f"Expected 'critic_loss' in metrics, got keys: {list(metrics.keys())}"
        )

    def test_train_metrics_contains_actor_loss_key(self):
        """The metrics dict must contain an 'actor_loss' key."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        metrics = crossq.train(total_timesteps=200)
        assert "actor_loss" in metrics, (
            f"Expected 'actor_loss' in metrics, got keys: {list(metrics.keys())}"
        )

    def test_train_critic_loss_is_finite(self):
        """The returned critic_loss must be finite (not NaN or Inf)."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        metrics = crossq.train(total_timesteps=200)
        assert math.isfinite(metrics["critic_loss"]), (
            f"Expected finite critic_loss, got {metrics['critic_loss']}"
        )

    def test_train_actor_loss_is_finite(self):
        """The returned actor_loss must be finite."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        metrics = crossq.train(total_timesteps=200)
        assert math.isfinite(metrics["actor_loss"]), (
            f"Expected finite actor_loss, got {metrics['actor_loss']}"
        )

    def test_train_all_metric_values_are_finite(self):
        """Every float value in the returned metrics dict must be finite."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        metrics = crossq.train(total_timesteps=200)
        non_finite = {
            k: v for k, v in metrics.items()
            if isinstance(v, float) and not math.isfinite(v)
        }
        assert not non_finite, f"Non-finite metric values: {non_finite}"

    def test_train_completes_with_small_learning_starts(self):
        """train() with learning_starts=100 and 500 steps completes without error."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1",
            learning_starts=100,
            batch_size=32,
            policy_delay=3,
            seed=1,
        )
        metrics = crossq.train(total_timesteps=500)
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# TestCrossQPredict — predict() contract
# ---------------------------------------------------------------------------


class TestCrossQPredict:
    """predict() returns a valid continuous action within env bounds."""

    def test_predict_returns_numpy_array(self):
        """predict() returns an np.ndarray."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1", seed=0)
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        action = crossq.predict(obs, deterministic=True)
        assert isinstance(action, np.ndarray), (
            f"predict() should return np.ndarray, got {type(action)}"
        )

    def test_predict_action_within_bounds(self):
        """predict() returns an action within the environment's action bounds."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1", seed=0)
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        action = crossq.predict(obs, deterministic=True)
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(action >= low - 1e-6) and np.all(action <= high + 1e-6), (
            f"Action {action} outside bounds [{low}, {high}]"
        )

    def test_predict_deterministic_is_reproducible(self):
        """deterministic=True gives the same action for the same observation."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1", seed=42)
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset(seed=0)
        action1 = crossq.predict(obs, deterministic=True)
        action2 = crossq.predict(obs, deterministic=True)
        np.testing.assert_array_equal(
            action1, action2,
            err_msg="deterministic=True must return identical actions for the same obs"
        )

    def test_predict_works_after_training(self):
        """predict() is valid after a training run."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        crossq.train(total_timesteps=200)

        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        action = crossq.predict(obs, deterministic=True)
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(action >= low - 1e-6) and np.all(action <= high + 1e-6)


# ---------------------------------------------------------------------------
# TestCrossQRegistryAndStatus — Trainer integration
# ---------------------------------------------------------------------------


class TestCrossQRegistryAndStatus:
    """CrossQ is registered in the Trainer registry with 'experimental' status."""

    def test_trainer_crossq_resolves(self):
        """Trainer('crossq', env='Pendulum-v1') does not raise a ValueError."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("crossq", env="Pendulum-v1")
        assert trainer is not None

    def test_trainer_crossq_status_is_experimental(self):
        """Trainer('crossq', ...).status == 'experimental'."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("crossq", env="Pendulum-v1")
        assert trainer.status == "experimental", (
            f"Expected 'experimental', got {trainer.status!r}"
        )

    def test_trainer_crossq_emits_experimental_warning(self):
        """Constructing Trainer('crossq', ...) fires a UserWarning about 'experimental'."""
        from rlox.trainer import Trainer

        with pytest.warns(UserWarning, match="experimental"):
            Trainer("crossq", env="Pendulum-v1")

    def test_algorithm_status_dict_contains_crossq(self):
        """ALGORITHM_STATUS must have a 'crossq' entry."""
        from rlox.trainer import ALGORITHM_STATUS

        assert "crossq" in ALGORITHM_STATUS, (
            "ALGORITHM_STATUS must include 'crossq'. "
            "Add it to _register_builtins() in trainer.py."
        )

    def test_algorithm_registry_contains_crossq(self):
        """ALGORITHM_REGISTRY must have a 'crossq' entry."""
        from rlox.trainer import ALGORITHM_REGISTRY

        assert "crossq" in ALGORITHM_REGISTRY, (
            "ALGORITHM_REGISTRY must include 'crossq'. "
            "Add it to _register_builtins() in trainer.py."
        )

    def test_algorithm_status_crossq_is_experimental(self):
        """ALGORITHM_STATUS['crossq'] == 'experimental'."""
        from rlox.trainer import ALGORITHM_STATUS

        status = ALGORITHM_STATUS.get("crossq")
        assert status == "experimental", (
            f"Expected ALGORITHM_STATUS['crossq'] == 'experimental', got {status!r}"
        )

    def test_algorithm_status_completeness_invariant_holds_with_crossq(self):
        """set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY) still holds after crossq is added."""
        from rlox.trainer import ALGORITHM_REGISTRY, ALGORITHM_STATUS

        assert "crossq" in ALGORITHM_STATUS, (
            "'crossq' missing from ALGORITHM_STATUS."
        )
        assert "crossq" in ALGORITHM_REGISTRY, (
            "'crossq' missing from ALGORITHM_REGISTRY."
        )
        assert set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY), (
            f"Status/registry mismatch. "
            f"Missing from STATUS: {set(ALGORITHM_REGISTRY) - set(ALGORITHM_STATUS)}. "
            f"Extra in STATUS: {set(ALGORITHM_STATUS) - set(ALGORITHM_REGISTRY)}."
        )

    def test_trainer_crossq_can_train_short_run(self):
        """Trainer('crossq', ...).train() runs without error (short budget)."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer(
                "crossq",
                env="Pendulum-v1",
                config={"learning_starts": 100, "batch_size": 32, "seed": 0},
            )
        metrics = trainer.train(total_timesteps=200)
        assert isinstance(metrics, dict)
        assert math.isfinite(metrics.get("critic_loss", float("nan")))


# ---------------------------------------------------------------------------
# TestCrossQSaveLoad — checkpoint contract
# ---------------------------------------------------------------------------


class TestCrossQSaveLoad:
    """CrossQ.save() / from_checkpoint() round-trip."""

    def test_crossq_save_and_load(self, tmp_path):
        """CrossQ can be saved and restored; env_id is preserved."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        crossq.train(total_timesteps=200)

        ckpt = str(tmp_path / "crossq.pt")
        crossq.save(ckpt)
        crossq2 = CrossQ.from_checkpoint(ckpt, env_id="Pendulum-v1")
        assert crossq2.env_id == "Pendulum-v1"

    def test_loaded_crossq_can_predict(self, tmp_path):
        """A CrossQ restored from checkpoint can call predict() without error."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        crossq.train(total_timesteps=200)

        ckpt = str(tmp_path / "crossq2.pt")
        crossq.save(ckpt)
        crossq2 = CrossQ.from_checkpoint(ckpt, env_id="Pendulum-v1")

        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        action = crossq2.predict(obs, deterministic=True)
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(action >= low - 1e-6) and np.all(action <= high + 1e-6)

    def test_loaded_crossq_has_no_target_critics(self, tmp_path):
        """A CrossQ restored from checkpoint STILL has no target critic attributes."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0
        )
        crossq.train(total_timesteps=200)

        ckpt = str(tmp_path / "crossq3.pt")
        crossq.save(ckpt)
        crossq2 = CrossQ.from_checkpoint(ckpt, env_id="Pendulum-v1")

        assert not hasattr(crossq2, "critic1_target"), (
            "Restored CrossQ must not have critic1_target."
        )
        assert not hasattr(crossq2, "tau"), (
            "Restored CrossQ must not have tau."
        )


# ---------------------------------------------------------------------------
# TestCrossQConvergence — marked slow, not run in the fast suite
# ---------------------------------------------------------------------------


class TestCrossQConvergence:
    """CrossQ learns Pendulum-v1 well above random level.

    Pendulum-v1 random baseline: mean reward ≈ −1200.
    Threshold of −400 is a low bar that confirms the agent is doing something.
    These tests are expensive; mark with @pytest.mark.slow.
    """

    @pytest.mark.slow
    def test_crossq_learns_pendulum_above_random(self):
        """After 20k steps CrossQ achieves mean_reward > −400 on Pendulum-v1.

        Random policy on Pendulum-v1: ≈ −1200.  A threshold of −400 is a
        conservative bar that proves CrossQ's joint forward pass + BatchRenorm
        is actually learning.
        """
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1",
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=256,
            learning_starts=1000,
            train_freq=1,
            gradient_steps=1,
            policy_delay=3,
            hidden=256,
            auto_entropy=True,
            ent_coef="auto",
            bn_momentum=0.01,
            bn_eps=1e-3,
            renorm_warmup_steps=100_000,
            seed=42,
        )
        metrics = crossq.train(total_timesteps=20_000)
        mean_reward = metrics.get("mean_reward", -9999.0)
        assert mean_reward > -400, (
            f"Expected mean_reward > -400 after 20k steps on Pendulum-v1, "
            f"got {mean_reward:.1f}. "
            "CrossQ may not be learning — check the joint forward pass and "
            "BatchRenorm correctness."
        )
