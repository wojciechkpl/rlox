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

Convergence-fix contract (see docs/plans/crossq-convergence-fix-2026-07-18.md
for the two controlled experiments that diagnosed these):

  - ``CrossQ.__init__`` accepts an ``adam_betas`` kwarg, default
    ``(0.5, 0.999)`` (NOT torch's ``(0.9, 0.999)`` default). It configures
    ALL four Adam optimizers: ``actor_optimizer``, ``critic1_optimizer``,
    ``critic2_optimizer``, and ``alpha_optimizer`` (when ``auto_entropy``).
    ``CrossQConfig`` must expose the same field name (``adam_betas``) so
    ``from_checkpoint()`` -- which rebuilds via ``cls(env_id=eid, **config)``
    -- round-trips it correctly.
  - ``seed`` must actually control torch/np/env RNG (currently a no-op: it
    is stored as ``self.seed`` but never applied). Two ``CrossQ`` instances
    built (and, for a short identical training budget, trained) with the
    same ``seed`` must be bit-identical; different seeds must diverge. This
    is required both for reproducibility and for the project's multi-seed
    IQM validation methodology.
  - The joint forward pass's ``[:B]``/``[B:]`` split (current vs next half)
    and the ``.detach()`` on the TD target are already correct and must
    stay that way -- ``TestCrossQJointPassValueAlignment`` is a FAST,
    mutation-checked guard for this (previously only the slow convergence
    test would have caught a regression here).
  - The ``@pytest.mark.slow`` convergence test asserts on a **greedy
    (deterministic) evaluation** mean reward, not the training-loop
    ``mean_reward`` (which is dragged down by the random-exploration
    ``learning_starts`` phase and stochastic action sampling).
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
# TestCrossQAdamBetas — Adam beta1 must default to 0.5, not torch's 0.9
# ---------------------------------------------------------------------------


class TestCrossQAdamBetas:
    """CrossQ's Adam optimizers must use ``betas=(0.5, 0.999)`` by default.

    Root cause (docs/plans/crossq-convergence-fix-2026-07-18.md): a
    controlled ablation holding everything else fixed showed torch's default
    ``betas=(0.9, 0.999)`` fails to converge on Pendulum-v1 (greedy eval
    -716) while ``betas=(0.5, 0.999)`` (paper / SB3-contrib value) solves it
    (greedy eval -166). BatchNorm-based critics destabilise under high Adam
    first-moment momentum, so CrossQ needs a lower beta1 than SAC/TD3.

    All four optimizers built by ``CrossQ.__init__`` (actor, critic1,
    critic2, and alpha when ``auto_entropy=True``) must use these betas.
    ``adam_betas`` must also be a configurable constructor kwarg (project
    convention: no hardcoded magic numbers).
    """

    def test_actor_optimizer_default_betas_is_0_5_0_999(self):
        """The actor Adam optimizer defaults to betas=(0.5, 0.999)."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        betas = crossq.actor_optimizer.param_groups[0]["betas"]
        assert betas == pytest.approx((0.5, 0.999)), (
            f"Expected default Adam betas=(0.5, 0.999) for actor_optimizer "
            f"(CrossQ needs beta1=0.5 for BatchNorm stability -- torch's "
            f"default beta1=0.9 fails to converge), got {betas}."
        )

    def test_critic1_optimizer_default_betas_is_0_5_0_999(self):
        """The critic1 Adam optimizer defaults to betas=(0.5, 0.999)."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        betas = crossq.critic1_optimizer.param_groups[0]["betas"]
        assert betas == pytest.approx((0.5, 0.999)), (
            f"Expected default Adam betas=(0.5, 0.999) for critic1_optimizer, "
            f"got {betas}."
        )

    def test_critic2_optimizer_default_betas_is_0_5_0_999(self):
        """The critic2 Adam optimizer defaults to betas=(0.5, 0.999)."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        betas = crossq.critic2_optimizer.param_groups[0]["betas"]
        assert betas == pytest.approx((0.5, 0.999)), (
            f"Expected default Adam betas=(0.5, 0.999) for critic2_optimizer, "
            f"got {betas}."
        )

    def test_alpha_optimizer_default_betas_is_0_5_0_999_when_auto_entropy(self):
        """The alpha (entropy coefficient) optimizer also defaults to betas=(0.5, 0.999)."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1", auto_entropy=True)
        assert hasattr(crossq, "alpha_optimizer"), (
            "CrossQ with auto_entropy=True must expose self.alpha_optimizer."
        )
        betas = crossq.alpha_optimizer.param_groups[0]["betas"]
        assert betas == pytest.approx((0.5, 0.999)), (
            f"Expected default Adam betas=(0.5, 0.999) for alpha_optimizer, "
            f"got {betas}."
        )

    def test_config_stores_default_adam_betas(self):
        """CrossQ.config (a CrossQConfig) records adam_betas=(0.5, 0.999) by default.

        Required for checkpoint round-tripping: from_checkpoint() rebuilds
        CrossQ via ``cls(env_id=eid, **config)``, so the CrossQConfig field
        name must match the __init__ kwarg name exactly.
        """
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1")
        assert crossq.config.adam_betas == pytest.approx((0.5, 0.999)), (
            f"Expected crossq.config.adam_betas == (0.5, 0.999), "
            f"got {crossq.config.adam_betas!r}."
        )

    def test_adam_betas_kwarg_configures_actor_and_critic_optimizers(self):
        """A custom adam_betas kwarg overrides the default for actor/critic1/critic2."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(env_id="Pendulum-v1", adam_betas=(0.7, 0.99))
        for name, opt in (
            ("actor_optimizer", crossq.actor_optimizer),
            ("critic1_optimizer", crossq.critic1_optimizer),
            ("critic2_optimizer", crossq.critic2_optimizer),
        ):
            betas = opt.param_groups[0]["betas"]
            assert betas == pytest.approx((0.7, 0.99)), (
                f"Expected {name} betas == (0.7, 0.99) after passing "
                f"adam_betas=(0.7, 0.99), got {betas}."
            )

    def test_adam_betas_kwarg_configures_alpha_optimizer(self):
        """A custom adam_betas kwarg also overrides the alpha optimizer's betas."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1", adam_betas=(0.7, 0.99), auto_entropy=True
        )
        betas = crossq.alpha_optimizer.param_groups[0]["betas"]
        assert betas == pytest.approx((0.7, 0.99)), (
            f"Expected alpha_optimizer betas == (0.7, 0.99) after passing "
            f"adam_betas=(0.7, 0.99), got {betas}."
        )

    def test_custom_adam_betas_round_trips_through_checkpoint(self, tmp_path):
        """A non-default adam_betas survives a save()/from_checkpoint() round-trip."""
        from rlox.algorithms.crossq import CrossQ

        crossq = CrossQ(
            env_id="Pendulum-v1",
            adam_betas=(0.7, 0.99),
            learning_starts=50,
            batch_size=32,
            seed=0,
        )
        crossq.train(total_timesteps=100)

        ckpt = str(tmp_path / "crossq_betas.pt")
        crossq.save(ckpt)
        crossq2 = CrossQ.from_checkpoint(ckpt, env_id="Pendulum-v1")

        betas = crossq2.actor_optimizer.param_groups[0]["betas"]
        assert betas == pytest.approx((0.7, 0.99)), (
            f"Expected adam_betas=(0.7, 0.99) to survive a checkpoint "
            f"round-trip, got actor_optimizer betas={betas}."
        )


# ---------------------------------------------------------------------------
# TestCrossQSeedReproducibility — `seed` must actually control RNG
# ---------------------------------------------------------------------------


class TestCrossQSeedReproducibility:
    """The ``seed`` constructor kwarg must control torch/np/env RNG.

    Root cause (docs/plans/crossq-convergence-fix-2026-07-18.md):
    ``crossq.py`` stores ``self.seed = seed`` but never applies it, so every
    construction/training run draws from whatever state the process-global
    RNGs happen to be in. This breaks reproducibility of a single run AND
    the project's multi-seed IQM validation methodology (different `seed`
    values must produce controlled, comparable, but *different* runs).

    Interface assumption: fixing this requires applying `seed` to torch
    (network init and stochastic policy sampling) and -- since Pendulum-v1's
    reset state and the random-exploration ``action_space.sample()`` calls
    are also sources of randomness -- to numpy/the env as well. These tests
    assert only the externally observable consequence (same seed -> same
    outcome; different seed -> different outcome), not which specific RNG
    call the implementer seeds.
    """

    _FIXED_OBS = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _assert_state_dicts_equal(sd_a, sd_b, label):
        assert sd_a.keys() == sd_b.keys(), (
            f"{label}: state_dict keys differ: {list(sd_a.keys())} vs {list(sd_b.keys())}"
        )
        for key in sd_a:
            assert torch.equal(sd_a[key], sd_b[key]), (
                f"{label}: tensor '{key}' differs between the two instances."
            )

    @staticmethod
    def _assert_state_dicts_differ(sd_a, sd_b, label):
        assert sd_a.keys() == sd_b.keys(), (
            f"{label}: state_dict keys differ: {list(sd_a.keys())} vs {list(sd_b.keys())}"
        )
        all_equal = all(torch.equal(sd_a[key], sd_b[key]) for key in sd_a)
        assert not all_equal, (
            f"{label}: every tensor is identical between the two instances; "
            f"expected at least one to differ."
        )

    # -- construction-time determinism (no training) -----------------------

    def test_same_seed_produces_identical_initial_actor_parameters(self):
        """Two CrossQ(seed=123) instances must have bit-identical actor init."""
        from rlox.algorithms.crossq import CrossQ

        a = CrossQ(env_id="Pendulum-v1", seed=123)
        b = CrossQ(env_id="Pendulum-v1", seed=123)
        self._assert_state_dicts_equal(
            a.actor.state_dict(), b.actor.state_dict(),
            "same seed=123, actor init",
        )

    def test_same_seed_produces_identical_initial_critic_parameters(self):
        """Two CrossQ(seed=123) instances must have bit-identical critic1/critic2 init."""
        from rlox.algorithms.crossq import CrossQ

        a = CrossQ(env_id="Pendulum-v1", seed=123)
        b = CrossQ(env_id="Pendulum-v1", seed=123)
        self._assert_state_dicts_equal(
            a.critic1.state_dict(), b.critic1.state_dict(),
            "same seed=123, critic1 init",
        )
        self._assert_state_dicts_equal(
            a.critic2.state_dict(), b.critic2.state_dict(),
            "same seed=123, critic2 init",
        )

    def test_different_seeds_produce_different_initial_actor_parameters(self):
        """CrossQ(seed=123) and CrossQ(seed=456) must NOT have identical actor init.

        Guards against a degenerate 'fix' that calls torch.manual_seed with a
        hardcoded constant instead of the `seed` argument, which would make
        the identical-init test above pass without `seed` actually doing
        anything useful.
        """
        from rlox.algorithms.crossq import CrossQ

        a = CrossQ(env_id="Pendulum-v1", seed=123)
        b = CrossQ(env_id="Pendulum-v1", seed=456)
        self._assert_state_dicts_differ(
            a.actor.state_dict(), b.actor.state_dict(),
            "seed=123 vs seed=456, actor init",
        )

    # -- short-training determinism -----------------------------------------

    @pytest.fixture(scope="class")
    def same_seed_pair(self):
        """Two CrossQ(seed=123) instances trained for an identical short budget."""
        from rlox.algorithms.crossq import CrossQ

        kwargs = dict(
            env_id="Pendulum-v1",
            hidden=32,
            batch_size=32,
            learning_starts=50,
            policy_delay=3,
            seed=123,
        )
        a = CrossQ(**kwargs)
        a.train(total_timesteps=800)
        b = CrossQ(**kwargs)
        b.train(total_timesteps=800)
        return a, b

    @pytest.fixture(scope="class")
    def different_seed_pair(self):
        """Two CrossQ instances (seed=123 vs seed=456), same short training budget."""
        from rlox.algorithms.crossq import CrossQ

        a = CrossQ(
            env_id="Pendulum-v1", hidden=32, batch_size=32,
            learning_starts=50, policy_delay=3, seed=123,
        )
        a.train(total_timesteps=800)
        b = CrossQ(
            env_id="Pendulum-v1", hidden=32, batch_size=32,
            learning_starts=50, policy_delay=3, seed=456,
        )
        b.train(total_timesteps=800)
        return a, b

    def test_same_seed_short_training_produces_identical_predict_output(
        self, same_seed_pair
    ):
        """Same seed -> identical deterministic predict() after ~800 training steps."""
        a, b = same_seed_pair
        action_a = a.predict(self._FIXED_OBS, deterministic=True)
        action_b = b.predict(self._FIXED_OBS, deterministic=True)
        np.testing.assert_array_equal(
            action_a, action_b,
            err_msg=(
                "Two CrossQ(seed=123) instances trained for an identical "
                "800-step budget must produce identical predict() output on "
                "the same observation. `seed` must control torch/np/env RNG."
            ),
        )

    def test_same_seed_short_training_produces_identical_critic_weights(
        self, same_seed_pair
    ):
        """Same seed -> identical critic1 weights after ~800 training steps."""
        a, b = same_seed_pair
        self._assert_state_dicts_equal(
            a.critic1.state_dict(), b.critic1.state_dict(),
            "same seed=123, critic1 after 800 training steps",
        )

    def test_different_seeds_short_training_produce_different_predict_output(
        self, different_seed_pair
    ):
        """Different seeds -> different predict() output after ~800 training steps.

        Guards against a degenerate 'fix' where seeding is applied but the
        actual `seed` value is ignored (e.g. hardcoded to a constant).
        """
        a, b = different_seed_pair
        action_a = a.predict(self._FIXED_OBS, deterministic=True)
        action_b = b.predict(self._FIXED_OBS, deterministic=True)
        assert not np.array_equal(action_a, action_b), (
            "CrossQ(seed=123) and CrossQ(seed=456) trained for an identical "
            "800-step budget produced IDENTICAL predict() output -- "
            "different seed values must produce different (controlled) runs."
        )


# ---------------------------------------------------------------------------
# TestCrossQJointPassValueAlignment — fast guard for the joint forward pass
# ---------------------------------------------------------------------------


class TestCrossQJointPassValueAlignment:
    """Fast, mutation-checked guard for the joint critic forward pass.

    The joint pass concatenates (obs, next_obs) and (actions, next_act) into
    ONE batch, forwards it through the live critic ONCE (so BatchRenorm sees
    consistent statistics), then splits the result back into a "current"
    half (``[:B]``, used in the critic loss) and a "next"/bootstrap half
    (``[B:]``, used -- detached -- in the TD target). This guards that split
    against a future split/detach regression with a FAST test (previously
    only the ``@pytest.mark.slow`` convergence test would have caught it).

    Technique: replace critic1/critic2 with a deterministic stub whose
    output is a known function of its input (``sum(obs) + sum(act)``), push
    transitions whose obs/next_obs sums are numerically far apart (~3 vs
    ~60), and inspect what ``F.mse_loss`` is actually called with. If the
    ``[:B]``/``[B:]`` split were reversed, or the TD target were not
    detached, this test would fail (verified by mutation -- see the RED
    phase report).
    """

    class _SumCritic(nn.Module):
        """Deterministic stub: Q(obs, act) = sum(obs) + sum(act) + bias(=0).

        ``bias`` is a learnable (zero-initialised) parameter purely so the
        computation graph has a leaf with ``requires_grad=True`` -- without
        it, ``critic_loss.backward()`` inside the real ``_update()`` would
        raise (nothing in the graph would require grad, since obs/act
        tensors sampled from the replay buffer carry no grad of their own).
        Because ``bias`` starts at exactly 0, the numeric VALUE this stub
        produces is unaffected -- it is still exactly ``sum(obs) + sum(act)``.
        """

        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, obs, act):
            return (
                obs.sum(dim=-1, keepdim=True)
                + act.sum(dim=-1, keepdim=True)
                + self.bias
            )

    def test_joint_pass_current_next_split_matches_concat_order_and_target_is_detached(
        self, monkeypatch
    ):
        """The [:B]/[B:] split matches concat order; the TD target is detached."""
        import rlox.algorithms.crossq as crossq_module
        from rlox.algorithms.crossq import CrossQ

        # policy_delay=1000 guarantees the (irrelevant, for this test) actor
        # update branch cannot fire on the single _update() call below.
        crossq = CrossQ(
            env_id="Pendulum-v1",
            hidden=8,
            batch_size=16,
            policy_delay=1000,
            seed=0,
        )
        crossq.critic1 = self._SumCritic()
        crossq.critic2 = self._SumCritic()

        # "Current" cluster (obs) sums to ~3; "next" cluster (next_obs) sums
        # to ~60 -- far enough apart that a reversed split is unmistakable,
        # but small enough to stay in-distribution for the (untouched, real)
        # actor network, which also consumes next_obs internally.
        n_transitions = 16
        for i in range(n_transitions):
            obs_i = np.array([1.0, 1.0, 1.0], dtype=np.float32) + i * 1e-3
            next_obs_i = np.array([20.0, 20.0, 20.0], dtype=np.float32) + i * 1e-3
            action_i = np.array([0.0], dtype=np.float32)
            crossq.buffer.push(obs_i, action_i, 1.0, False, False, next_obs_i)

        recorded_calls = []
        real_mse_loss = crossq_module.F.mse_loss

        def _recording_mse_loss(input, target, *args, **kwargs):
            recorded_calls.append(
                (input.detach().clone(), target, bool(target.requires_grad))
            )
            return real_mse_loss(input, target, *args, **kwargs)

        monkeypatch.setattr(crossq_module.F, "mse_loss", _recording_mse_loss)

        crossq._update(step=0)

        assert len(recorded_calls) == 2, (
            f"Expected exactly 2 F.mse_loss calls (critic1_loss, critic2_loss) "
            f"from one _update() call with the actor branch inactive; got "
            f"{len(recorded_calls)}."
        )

        for i, (q_current, target, target_requires_grad) in enumerate(recorded_calls):
            current_mean = q_current.mean().item()
            target_mean = target.mean().item()

            # (a) split alignment: the value passed as the *prediction* to
            # F.mse_loss must come from the CURRENT half ([:B], obs+actions,
            # sum ~3), not the NEXT half ([B:], next_obs+next_act, sum ~60).
            assert current_mean < 15.0, (
                f"F.mse_loss call #{i}: prediction Q-value mean="
                f"{current_mean:.3f} looks like the NEXT half (obs~20 "
                f"cluster), not the CURRENT half (obs~1 cluster). The "
                f"[:B]/[B:] split in _update() may be reversed."
            )
            # And the TD target must be built from the NEXT half (~60,
            # scaled by gamma and offset by reward/entropy terms).
            assert target_mean > 15.0, (
                f"F.mse_loss call #{i}: target mean={target_mean:.3f} looks "
                f"like it was built from the CURRENT half, not the NEXT "
                f"half. The [:B]/[B:] split in _update() may be reversed."
            )

            # (b) the TD target must be detached -- no gradient path back
            # into the critic through the bootstrap ([B:]) computation.
            assert target_requires_grad is False, (
                f"F.mse_loss call #{i}: target.requires_grad is True. The "
                f"TD target must be `.detach()`-ed before use in the critic "
                f"loss, or gradients leak through the next-half computation "
                f"back into critic1/critic2's parameters."
            )


# ---------------------------------------------------------------------------
# TestCrossQConvergence — marked slow, not run in the fast suite
# ---------------------------------------------------------------------------


class TestCrossQConvergence:
    """CrossQ solves Pendulum-v1 under greedy (deterministic) evaluation.

    Pendulum-v1 random baseline: mean reward ≈ −1200. A working CrossQ
    (correct joint forward pass, correct BatchRenorm, β1=0.5 Adam betas)
    reaches roughly −166 to −174 (docs/plans/crossq-convergence-fix-2026-07-18.md).
    The threshold here (−250) sits comfortably below those measured values
    but well above the random baseline, giving margin for run-to-run
    variation while still only being clearable by a genuinely converged
    policy. The *old* version of this test used a −400 bar on the
    training-loop ``mean_reward`` (not a greedy eval) and was both flaky
    (the seed bug meant every run was an uncontrolled fresh draw) and too
    coarse (it only proved the agent was "doing something," not that it had
    converged).

    This test asserts on a **greedy eval** (``predict(deterministic=True)``
    averaged over several episodes with a fresh env), NOT the training-loop
    ``mean_reward`` returned by ``train()`` -- that figure is dragged down by
    the random-exploration ``learning_starts`` phase and by stochastic
    (non-deterministic) action sampling throughout training, so it
    systematically underestimates the learned policy's actual quality.

    Relies on ``seed`` actually controlling torch/np/env RNG (see
    ``TestCrossQSeedReproducibility``) to be deterministic run-to-run.
    Expensive; marked with @pytest.mark.slow.
    """

    @pytest.mark.slow
    def test_crossq_greedy_eval_solves_pendulum(self):
        """After 20k training steps, a 10-episode greedy eval reaches > −250.

        Random policy on Pendulum-v1: ≈ −1200. A trained-but-broken CrossQ
        (default Adam betas) plateaus around −700 to −800. A correctly
        configured CrossQ (β1=0.5) reaches ≈ −166 to −174. −250 is a
        conservative bar in between that only a genuinely converged policy
        clears.
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
        crossq.train(total_timesteps=20_000)

        # Greedy evaluation on a fresh env -- deterministic actions, no
        # exploration noise. Per-episode seeds follow the project's
        # eval-seeding convention (`seed=base+ep`, not a fixed `seed=base`
        # every episode -- see PROJECT_QUICK_REFERENCE.md "Non-obvious
        # facts") so the 10 episodes probe genuinely different initial
        # states rather than repeating one.
        n_eval_episodes = 10
        eval_env = gym.make("Pendulum-v1")
        episode_rewards = []
        for ep in range(n_eval_episodes):
            obs, _ = eval_env.reset(seed=42 + ep)
            terminated = truncated = False
            ep_reward = 0.0
            while not (terminated or truncated):
                action = crossq.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_reward += float(reward)
            episode_rewards.append(ep_reward)

        mean_eval_reward = float(np.mean(episode_rewards))
        assert mean_eval_reward > -250, (
            f"Expected greedy-eval mean_reward > -250 after 20k steps on "
            f"Pendulum-v1 (seed=42), got {mean_eval_reward:.1f} over "
            f"{n_eval_episodes} episodes: {episode_rewards}. "
            "CrossQ's Adam betas and/or seed handling may still be broken "
            "-- see docs/plans/crossq-convergence-fix-2026-07-18.md."
        )
