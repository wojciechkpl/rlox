"""Tests for TQC (Truncated Quantile Critics) — Kuznetsov et al., ICML 2020.

arXiv:2005.04269, "Controlling Overestimation Bias with Truncated Mixture of
Continuous Distributional Quantile Critics".

TDD red phase: these tests are written FIRST, before any implementation. The
feature (rlox.algorithms.tqc) does not exist yet; every test here is expected
to FAIL with ImportError / AttributeError / AssertionError until the
implementation lands.

Contract:

  - ``from rlox.algorithms.tqc import TQC, TQCConfig, QuantileQNetwork,
    truncate_quantiles, quantile_huber_loss``
  - ``TQC(env_id, ...)`` — continuous (Box) action spaces ONLY
  - ``TQC.train(total_timesteps) -> dict[str, float]`` — finite values
  - ``TQC.predict(obs, deterministic=True) -> np.ndarray`` — in bounds
  - Ensemble of ``n_critics`` (default 5) ``QuantileQNetwork`` critics, each
    predicting ``n_quantiles`` (default 25) quantiles of the return
    distribution from ``(obs, action)``.
  - Target computation: sample next action from the actor, evaluate all
    ``n_critics * n_quantiles`` next-quantiles via the TARGET critic
    ensemble, pool + sort them, then drop the top
    ``top_quantiles_to_drop_per_net * n_critics`` (largest / most
    optimistic) before bootstrapping — the overestimation-control
    mechanism that replaces SAC/TD3's ``min`` of two point estimates.
  - Critic loss: pairwise quantile Huber (pinball) loss between each
    critic's predicted quantiles and the SAME shared truncated target set.
  - Actor loss: maximise the mean of ALL (untruncated) critics' quantiles
    at the actor's sampled action, minus ``alpha * log_pi`` (standard SAC
    entropy term). Reuses ``SquashedGaussianPolicy``, ``rlox.ReplayBuffer``,
    and SAC-style automatic entropy tuning.
  - ``seed`` must control torch/np/env RNG from construction (applied
    BEFORE network construction, precedent: crossq.py).

This module is tested directly (``from rlox.algorithms.tqc import TQC``),
NOT via ``Trainer("tqc")`` — Trainer-registry wiring (``ALGORITHM_REGISTRY``,
``ALGORITHM_STATUS``, ``config.py``) is added later during orchestrator
reconciliation and is intentionally out of scope here.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TestQuantileQNetwork — the per-critic MLP: (obs, act) -> n_quantiles
# ---------------------------------------------------------------------------


class TestQuantileQNetwork:
    """``QuantileQNetwork`` maps (obs, action) to ``n_quantiles`` quantiles."""

    def test_constructs(self):
        from rlox.algorithms.tqc import QuantileQNetwork

        net = QuantileQNetwork(obs_dim=3, act_dim=1, hidden=32, n_quantiles=25)
        assert net is not None

    def test_is_nn_module(self):
        from rlox.algorithms.tqc import QuantileQNetwork

        assert issubclass(QuantileQNetwork, nn.Module)

    def test_output_shape_is_batch_by_n_quantiles(self):
        """forward(obs, act) -> (batch, n_quantiles), NOT (batch, 1)."""
        from rlox.algorithms.tqc import QuantileQNetwork

        net = QuantileQNetwork(obs_dim=3, act_dim=1, hidden=32, n_quantiles=25)
        obs = torch.randn(16, 3)
        act = torch.randn(16, 1)
        out = net(obs, act)
        assert out.shape == (16, 25), (
            f"Expected QuantileQNetwork output shape (16, 25), got {tuple(out.shape)}"
        )

    @pytest.mark.parametrize("n_quantiles", [1, 4, 25, 32])
    def test_output_width_matches_n_quantiles_param(self, n_quantiles):
        from rlox.algorithms.tqc import QuantileQNetwork

        net = QuantileQNetwork(obs_dim=5, act_dim=2, hidden=16, n_quantiles=n_quantiles)
        out = net(torch.randn(4, 5), torch.randn(4, 2))
        assert out.shape == (4, n_quantiles)

    def test_output_is_finite(self):
        from rlox.algorithms.tqc import QuantileQNetwork

        net = QuantileQNetwork(obs_dim=3, act_dim=1, hidden=32, n_quantiles=25)
        out = net(torch.randn(8, 3), torch.randn(8, 1))
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# TestTruncateQuantiles — the overestimation-control truncation
# ---------------------------------------------------------------------------


class TestTruncateQuantiles:
    """``truncate_quantiles`` sorts ascending and drops the largest N values.

    This is the core of TQC's overestimation control: dropping the top
    (most optimistic) pooled quantiles before bootstrapping.
    """

    def test_drops_correct_count(self):
        from rlox.algorithms.tqc import truncate_quantiles

        pooled = torch.tensor([[5.0, 1.0, 3.0, 2.0, 4.0]])
        out = truncate_quantiles(pooled, n_dropped=2)
        assert out.shape == (1, 3), f"Expected shape (1, 3), got {tuple(out.shape)}"

    def test_drops_the_largest_values_not_smallest(self):
        """Dropping must remove the TOP (largest) values -- the pessimism trick.

        A bug that drops the smallest values instead would defeat TQC's
        entire overestimation-control mechanism.
        """
        from rlox.algorithms.tqc import truncate_quantiles

        pooled = torch.tensor([[5.0, 1.0, 3.0, 2.0, 4.0]])
        out = truncate_quantiles(pooled, n_dropped=2)
        # Sorted ascending: [1, 2, 3, 4, 5]; drop top 2 (4, 5) -> [1, 2, 3].
        assert out.tolist() == [[1.0, 2.0, 3.0]], (
            f"Expected the 3 SMALLEST values [1, 2, 3] to remain (largest "
            f"dropped), got {out.tolist()}"
        )

    def test_result_is_sorted_ascending(self):
        from rlox.algorithms.tqc import truncate_quantiles

        pooled = torch.tensor([[9.0, -3.0, 0.5, 7.2, -1.0, 2.0]])
        out = truncate_quantiles(pooled, n_dropped=1)
        values = out.squeeze(0).tolist()
        assert values == sorted(values), "Result must be sorted ascending"

    def test_n_dropped_zero_keeps_everything(self):
        from rlox.algorithms.tqc import truncate_quantiles

        pooled = torch.tensor([[3.0, 1.0, 2.0]])
        out = truncate_quantiles(pooled, n_dropped=0)
        assert out.shape == (1, 3)
        assert out.tolist() == [[1.0, 2.0, 3.0]]

    def test_preserves_batch_dimension(self):
        from rlox.algorithms.tqc import truncate_quantiles

        pooled = torch.randn(32, 125)  # e.g. 5 critics x 25 quantiles
        out = truncate_quantiles(pooled, n_dropped=10)
        assert out.shape == (32, 115)

    def test_raises_on_negative_n_dropped(self):
        from rlox.algorithms.tqc import truncate_quantiles

        pooled = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            truncate_quantiles(pooled, n_dropped=-1)

    def test_raises_when_n_dropped_equals_or_exceeds_total(self):
        """Dropping >= all quantiles would leave an empty (meaningless) target."""
        from rlox.algorithms.tqc import truncate_quantiles

        pooled = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            truncate_quantiles(pooled, n_dropped=3)


# ---------------------------------------------------------------------------
# TestQuantileHuberLoss — shape / sign sanity for the critic loss
# ---------------------------------------------------------------------------


class TestQuantileHuberLoss:
    """``quantile_huber_loss`` — pairwise pinball/Huber loss (QR-DQN / TQC).

    Contract: quantile Huber loss uses the pinball/quantile loss with a
    Huber(kappa=1) core and per-quantile fractions tau_i = (i+0.5)/n; the
    target quantiles are ALWAYS detached internally.
    """

    def test_returns_scalar(self):
        from rlox.algorithms.tqc import quantile_huber_loss

        predicted = torch.randn(8, 25, requires_grad=True)
        target = torch.randn(8, 20)
        tau = (torch.arange(25, dtype=torch.float32) + 0.5) / 25
        loss = quantile_huber_loss(predicted, target, tau)
        assert loss.dim() == 0, f"Expected a 0-dim scalar loss, got shape {loss.shape}"

    def test_zero_for_perfect_prediction(self):
        """predicted == target (broadcast) everywhere -> zero loss."""
        from rlox.algorithms.tqc import quantile_huber_loss

        predicted = torch.full((4, 3), 2.5)
        target = torch.full((4, 3), 2.5)
        tau = (torch.arange(3, dtype=torch.float32) + 0.5) / 3
        loss = quantile_huber_loss(predicted, target, tau)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negative_for_random_inputs(self):
        from rlox.algorithms.tqc import quantile_huber_loss

        torch.manual_seed(0)
        for _ in range(5):
            predicted = torch.randn(8, 10)
            target = torch.randn(8, 12)
            tau = (torch.arange(10, dtype=torch.float32) + 0.5) / 10
            loss = quantile_huber_loss(predicted, target, tau)
            assert loss.item() >= 0.0

    def test_larger_error_gives_larger_loss(self):
        """Within the quadratic (|delta| < kappa) Huber regime, loss grows with |error|."""
        from rlox.algorithms.tqc import quantile_huber_loss

        tau = torch.tensor([0.5])
        predicted = torch.zeros(1, 1)
        small_err_loss = quantile_huber_loss(predicted, torch.full((1, 1), 0.1), tau)
        large_err_loss = quantile_huber_loss(predicted, torch.full((1, 1), 0.5), tau)
        assert large_err_loss.item() > small_err_loss.item()

    def test_underprediction_penalized_more_at_high_tau_than_low_tau(self):
        """Standard pinball asymmetry: under-prediction hurts more at high tau.

        For a fixed positive error (target > predicted), the quantile weight
        is ``tau`` (since the indicator term is 0), so a higher tau produces
        a strictly larger loss than a lower tau on the SAME error.
        """
        from rlox.algorithms.tqc import quantile_huber_loss

        predicted = torch.zeros(1, 1)
        target = torch.full((1, 1), 1.0)  # under-prediction: target > predicted
        loss_low_tau = quantile_huber_loss(predicted, target, torch.tensor([0.25]))
        loss_high_tau = quantile_huber_loss(predicted, target, torch.tensor([0.75]))
        assert loss_high_tau.item() > loss_low_tau.item(), (
            "Under-prediction (target > predicted) at tau=0.75 must be "
            "penalized more than at tau=0.25 (standard pinball-loss "
            "asymmetry). Check the sign of the indicator/weight term."
        )

    def test_overprediction_penalized_more_at_low_tau_than_high_tau(self):
        """The mirror-image asymmetry: over-prediction hurts more at low tau."""
        from rlox.algorithms.tqc import quantile_huber_loss

        predicted = torch.full((1, 1), 1.0)
        target = torch.zeros(1, 1)  # over-prediction: predicted > target
        loss_low_tau = quantile_huber_loss(predicted, target, torch.tensor([0.25]))
        loss_high_tau = quantile_huber_loss(predicted, target, torch.tensor([0.75]))
        assert loss_low_tau.item() > loss_high_tau.item(), (
            "Over-prediction (predicted > target) at tau=0.25 must be "
            "penalized more than at tau=0.75."
        )

    def test_gradient_flows_into_predicted(self):
        from rlox.algorithms.tqc import quantile_huber_loss

        predicted = torch.randn(4, 5, requires_grad=True)
        target = torch.randn(4, 6)
        tau = (torch.arange(5, dtype=torch.float32) + 0.5) / 5
        loss = quantile_huber_loss(predicted, target, tau)
        loss.backward()
        assert predicted.grad is not None
        assert torch.isfinite(predicted.grad).all()

    def test_target_is_detached_no_gradient_flows_into_it(self):
        """The target quantiles must be detached even if the caller forgot to.

        This guards the explicit correctness requirement: "the target
        quantiles are detached."
        """
        from rlox.algorithms.tqc import quantile_huber_loss

        predicted = torch.randn(4, 5, requires_grad=True)
        target = torch.randn(4, 6, requires_grad=True)  # deliberately NOT detached
        tau = (torch.arange(5, dtype=torch.float32) + 0.5) / 5
        loss = quantile_huber_loss(predicted, target, tau)
        loss.backward()
        assert target.grad is None, (
            "Gradients must NOT flow into `target` -- quantile_huber_loss "
            "must detach the target internally regardless of the caller."
        )


# ---------------------------------------------------------------------------
# TestTQCConfig
# ---------------------------------------------------------------------------


class TestTQCConfig:
    """``TQCConfig`` — locally-defined config dataclass (not in config.py)."""

    def test_constructs_with_defaults(self):
        from rlox.algorithms.tqc import TQCConfig

        cfg = TQCConfig()
        assert cfg.n_critics == 5
        assert cfg.n_quantiles == 25

    def test_to_dict_from_dict_round_trip(self):
        from rlox.algorithms.tqc import TQCConfig

        cfg = TQCConfig(n_critics=3, n_quantiles=10, top_quantiles_to_drop_per_net=1)
        d = cfg.to_dict()
        cfg2 = TQCConfig.from_dict(d)
        assert cfg2.n_critics == 3
        assert cfg2.n_quantiles == 10
        assert cfg2.top_quantiles_to_drop_per_net == 1

    def test_rejects_zero_n_critics(self):
        from rlox.algorithms.tqc import TQCConfig

        with pytest.raises(ValueError):
            TQCConfig(n_critics=0)

    def test_rejects_zero_n_quantiles(self):
        from rlox.algorithms.tqc import TQCConfig

        with pytest.raises(ValueError):
            TQCConfig(n_quantiles=0)

    def test_rejects_top_quantiles_to_drop_per_net_equal_to_n_quantiles(self):
        """Dropping >= n_quantiles per net would zero out every critic's contribution."""
        from rlox.algorithms.tqc import TQCConfig

        with pytest.raises(ValueError):
            TQCConfig(n_quantiles=25, top_quantiles_to_drop_per_net=25)

    def test_accepts_top_quantiles_to_drop_per_net_one_less_than_n_quantiles(self):
        from rlox.algorithms.tqc import TQCConfig

        cfg = TQCConfig(n_quantiles=25, top_quantiles_to_drop_per_net=24)
        assert cfg.top_quantiles_to_drop_per_net == 24

    def test_rejects_non_positive_learning_rate(self):
        from rlox.algorithms.tqc import TQCConfig

        with pytest.raises(ValueError):
            TQCConfig(learning_rate=0.0)


# ---------------------------------------------------------------------------
# TestTQCConstruction
# ---------------------------------------------------------------------------


class TestTQCConstruction:
    """TQC can be instantiated with canonical hyperparameters."""

    def test_constructs_with_defaults(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1")
        assert tqc is not None

    def test_stores_env_id(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1")
        assert tqc.env_id == "Pendulum-v1"

    def test_default_n_critics_is_5(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1")
        assert tqc.n_critics == 5

    def test_default_n_quantiles_is_25(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1")
        assert tqc.n_quantiles == 25

    def test_constructs_with_full_hyperparameter_set(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(
            env_id="Pendulum-v1",
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=64,
            learning_starts=100,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            hidden=64,
            n_critics=3,
            n_quantiles=10,
            top_quantiles_to_drop_per_net=1,
            huber_kappa=1.0,
            auto_entropy=True,
            ent_coef="auto",
            seed=7,
        )
        assert tqc is not None
        assert tqc.n_critics == 3
        assert tqc.n_quantiles == 10

    def test_continuous_only_raises_on_discrete_env(self):
        from rlox.algorithms.tqc import TQC

        with pytest.raises((ValueError, TypeError)):
            TQC(env_id="CartPole-v1")

    def test_discrete_env_error_mentions_continuous_or_discrete(self):
        from rlox.algorithms.tqc import TQC

        with pytest.raises((ValueError, TypeError), match="[Cc]ontinuous|[Dd]iscrete"):
            TQC(env_id="CartPole-v1")

    def test_rejects_n_critics_zero(self):
        from rlox.algorithms.tqc import TQC

        with pytest.raises(ValueError):
            TQC(env_id="Pendulum-v1", n_critics=0)

    def test_has_train_method(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1")
        assert callable(getattr(tqc, "train", None))

    def test_has_predict_method(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1")
        assert callable(getattr(tqc, "predict", None))

    def test_has_save_method(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1")
        assert callable(getattr(tqc, "save", None))

    def test_has_from_checkpoint_classmethod(self):
        from rlox.algorithms.tqc import TQC

        assert callable(getattr(TQC, "from_checkpoint", None))

    def test_critics_is_module_list_of_length_n_critics(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", n_critics=4, hidden=16)
        assert isinstance(tqc.critics, nn.ModuleList)
        assert len(tqc.critics) == 4

    def test_critic_targets_is_module_list_of_length_n_critics(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", n_critics=4, hidden=16)
        assert isinstance(tqc.critic_targets, nn.ModuleList)
        assert len(tqc.critic_targets) == 4

    def test_each_critic_is_quantile_q_network(self):
        from rlox.algorithms.tqc import TQC, QuantileQNetwork

        tqc = TQC(env_id="Pendulum-v1", n_critics=3, hidden=16)
        for critic in tqc.critics:
            assert isinstance(critic, QuantileQNetwork)

    def test_n_target_quantiles_matches_formula(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(
            env_id="Pendulum-v1",
            n_critics=5,
            n_quantiles=25,
            top_quantiles_to_drop_per_net=2,
            hidden=16,
        )
        assert tqc.n_target_quantiles == 5 * 25 - 2 * 5


# ---------------------------------------------------------------------------
# TestTQCCriticOutputShape — "the critic outputs n_quantiles per (obs,act)"
# ---------------------------------------------------------------------------


class TestTQCCriticOutputShape:
    """Each critic in the TQC ensemble outputs exactly ``n_quantiles`` values."""

    def test_single_critic_forward_shape(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", n_critics=3, n_quantiles=10, hidden=16)
        obs = torch.randn(6, tqc.obs_dim)
        act = torch.randn(6, tqc.act_dim)
        out = tqc.critics[0](obs, act)
        assert out.shape == (6, 10)

    def test_all_ensemble_members_share_output_width(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", n_critics=5, n_quantiles=25, hidden=16)
        obs = torch.randn(4, tqc.obs_dim)
        act = torch.randn(4, tqc.act_dim)
        for critic in tqc.critics:
            assert critic(obs, act).shape == (4, 25)

    def test_target_ensemble_shares_output_width(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", n_critics=5, n_quantiles=25, hidden=16)
        obs = torch.randn(4, tqc.obs_dim)
        act = torch.randn(4, tqc.act_dim)
        for target in tqc.critic_targets:
            assert target(obs, act).shape == (4, 25)


# ---------------------------------------------------------------------------
# TestTQCTruncationWiring — "the truncation actually drops the right count"
# ---------------------------------------------------------------------------


class TestTQCTruncationWiring:
    """``_update()`` invokes ``truncate_quantiles`` with the correct pool
    width and drop count derived from the configured hyperparameters."""

    def test_update_calls_truncate_quantiles_with_correct_pool_and_drop_count(
        self, monkeypatch
    ):
        import rlox.algorithms.tqc as tqc_module
        from rlox.algorithms.tqc import TQC

        n_critics = 3
        n_quantiles = 4
        top_drop = 1
        tqc = TQC(
            env_id="Pendulum-v1",
            hidden=8,
            batch_size=8,
            n_critics=n_critics,
            n_quantiles=n_quantiles,
            top_quantiles_to_drop_per_net=top_drop,
            seed=0,
        )

        for i in range(16):
            obs_i = np.array([1.0, 0.0, 0.0], dtype=np.float32) + i * 1e-3
            next_obs_i = np.array([0.9, 0.1, 0.1], dtype=np.float32) + i * 1e-3
            action_i = np.array([0.0], dtype=np.float32)
            tqc.buffer.push(obs_i, action_i, 1.0, False, False, next_obs_i)

        recorded_calls = []
        real_truncate = tqc_module.truncate_quantiles

        def _recording_truncate(pooled, n_dropped):
            recorded_calls.append((tuple(pooled.shape), n_dropped))
            return real_truncate(pooled, n_dropped)

        monkeypatch.setattr(tqc_module, "truncate_quantiles", _recording_truncate)

        tqc._update(step=0)

        assert len(recorded_calls) == 1, (
            f"Expected exactly 1 call to truncate_quantiles per _update(), "
            f"got {len(recorded_calls)}."
        )
        pooled_shape, n_dropped = recorded_calls[0]
        expected_pool_width = n_critics * n_quantiles
        expected_n_dropped = top_drop * n_critics

        assert pooled_shape[-1] == expected_pool_width, (
            f"Expected pooled quantile width {expected_pool_width} "
            f"(n_critics * n_quantiles), got {pooled_shape[-1]}."
        )
        assert n_dropped == expected_n_dropped, (
            f"Expected n_dropped={expected_n_dropped} "
            f"(top_quantiles_to_drop_per_net * n_critics), got {n_dropped}."
        )

    def test_truncated_target_has_n_target_quantiles_width(self, monkeypatch):
        """The truncated target actually used downstream has width n_target_quantiles."""
        import rlox.algorithms.tqc as tqc_module
        from rlox.algorithms.tqc import TQC

        tqc = TQC(
            env_id="Pendulum-v1",
            hidden=8,
            batch_size=8,
            n_critics=3,
            n_quantiles=4,
            top_quantiles_to_drop_per_net=1,
            seed=0,
        )
        for i in range(16):
            obs_i = np.array([1.0, 0.0, 0.0], dtype=np.float32) + i * 1e-3
            next_obs_i = np.array([0.9, 0.1, 0.1], dtype=np.float32) + i * 1e-3
            action_i = np.array([0.0], dtype=np.float32)
            tqc.buffer.push(obs_i, action_i, 1.0, False, False, next_obs_i)

        recorded_outputs = []
        real_huber = tqc_module.quantile_huber_loss

        def _recording_huber(predicted, target, tau, kappa=1.0):
            recorded_outputs.append(target.shape[-1])
            return real_huber(predicted, target, tau, kappa)

        monkeypatch.setattr(tqc_module, "quantile_huber_loss", _recording_huber)

        tqc._update(step=0)

        assert recorded_outputs, "quantile_huber_loss was never called"
        for width in recorded_outputs:
            assert width == tqc.n_target_quantiles, (
                f"Expected every critic's target width == "
                f"n_target_quantiles={tqc.n_target_quantiles}, got {width}."
            )


# ---------------------------------------------------------------------------
# TestTQCTraining — train() contract
# ---------------------------------------------------------------------------


class TestTQCTraining:
    def test_train_returns_dict(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0)
        result = tqc.train(total_timesteps=200)
        assert isinstance(result, dict)

    def test_train_metrics_contains_critic_loss_key(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0)
        metrics = tqc.train(total_timesteps=200)
        assert "critic_loss" in metrics

    def test_train_metrics_contains_actor_loss_key(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0)
        metrics = tqc.train(total_timesteps=200)
        assert "actor_loss" in metrics

    def test_train_critic_loss_is_finite(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0)
        metrics = tqc.train(total_timesteps=200)
        assert math.isfinite(metrics["critic_loss"]), (
            f"Expected finite critic_loss, got {metrics['critic_loss']}"
        )

    def test_train_critic_loss_is_non_negative(self):
        """Quantile Huber loss is always >= 0 by construction."""
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0)
        metrics = tqc.train(total_timesteps=200)
        assert metrics["critic_loss"] >= 0.0

    def test_train_actor_loss_is_finite(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0)
        metrics = tqc.train(total_timesteps=200)
        assert math.isfinite(metrics["actor_loss"])

    def test_train_all_metric_values_are_finite(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0)
        metrics = tqc.train(total_timesteps=200)
        non_finite = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, float) and not math.isfinite(v)
        }
        assert not non_finite, f"Non-finite metric values: {non_finite}"

    def test_train_completes_with_small_learning_starts(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(
            env_id="Pendulum-v1",
            learning_starts=100,
            batch_size=32,
            n_critics=3,
            n_quantiles=8,
            seed=1,
        )
        metrics = tqc.train(total_timesteps=500)
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# TestTQCPredict — predict() contract
# ---------------------------------------------------------------------------


class TestTQCPredict:
    def test_predict_returns_numpy_array(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", seed=0)
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        action = tqc.predict(obs, deterministic=True)
        assert isinstance(action, np.ndarray)

    def test_predict_action_within_bounds(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", seed=0)
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        action = tqc.predict(obs, deterministic=True)
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(action >= low - 1e-6) and np.all(action <= high + 1e-6), (
            f"Action {action} outside bounds [{low}, {high}]"
        )

    def test_predict_deterministic_is_reproducible(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", seed=42)
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset(seed=0)
        action1 = tqc.predict(obs, deterministic=True)
        action2 = tqc.predict(obs, deterministic=True)
        np.testing.assert_array_equal(action1, action2)

    def test_predict_works_after_training(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(env_id="Pendulum-v1", learning_starts=100, batch_size=32, seed=0)
        tqc.train(total_timesteps=200)

        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        action = tqc.predict(obs, deterministic=True)
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(action >= low - 1e-6) and np.all(action <= high + 1e-6)


# ---------------------------------------------------------------------------
# TestTQCSeedReproducibility — seed must control torch/np/env RNG
# ---------------------------------------------------------------------------


class TestTQCSeedReproducibility:
    """``seed`` must be applied before network construction (crossq.py precedent)."""

    @staticmethod
    def _assert_state_dicts_equal(sd_a, sd_b):
        assert sd_a.keys() == sd_b.keys()
        for key in sd_a:
            assert torch.equal(sd_a[key], sd_b[key]), f"tensor '{key}' differs"

    @staticmethod
    def _assert_state_dicts_differ(sd_a, sd_b):
        assert sd_a.keys() == sd_b.keys()
        all_equal = all(torch.equal(sd_a[key], sd_b[key]) for key in sd_a)
        assert not all_equal, "expected at least one tensor to differ"

    def test_same_seed_produces_identical_actor_init(self):
        from rlox.algorithms.tqc import TQC

        a = TQC(env_id="Pendulum-v1", seed=123, hidden=16)
        b = TQC(env_id="Pendulum-v1", seed=123, hidden=16)
        self._assert_state_dicts_equal(a.actor.state_dict(), b.actor.state_dict())

    def test_same_seed_produces_identical_critic_ensemble_init(self):
        from rlox.algorithms.tqc import TQC

        a = TQC(env_id="Pendulum-v1", seed=123, hidden=16, n_critics=3)
        b = TQC(env_id="Pendulum-v1", seed=123, hidden=16, n_critics=3)
        self._assert_state_dicts_equal(a.critics.state_dict(), b.critics.state_dict())

    def test_different_seeds_produce_different_actor_init(self):
        from rlox.algorithms.tqc import TQC

        a = TQC(env_id="Pendulum-v1", seed=123, hidden=16)
        b = TQC(env_id="Pendulum-v1", seed=456, hidden=16)
        self._assert_state_dicts_differ(a.actor.state_dict(), b.actor.state_dict())


# ---------------------------------------------------------------------------
# TestTQCSaveLoad — checkpoint round-trip
# ---------------------------------------------------------------------------


class TestTQCSaveLoad:
    def test_save_and_load_round_trip_preserves_env_id(self, tmp_path):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(
            env_id="Pendulum-v1",
            learning_starts=100,
            batch_size=32,
            n_critics=3,
            n_quantiles=8,
            seed=0,
        )
        tqc.train(total_timesteps=200)

        ckpt = str(tmp_path / "tqc.pt")
        tqc.save(ckpt)
        tqc2 = TQC.from_checkpoint(ckpt, env_id="Pendulum-v1")
        assert tqc2.env_id == "Pendulum-v1"
        assert tqc2.n_critics == 3
        assert tqc2.n_quantiles == 8

    def test_loaded_model_can_predict_in_bounds(self, tmp_path):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(
            env_id="Pendulum-v1",
            learning_starts=100,
            batch_size=32,
            n_critics=3,
            n_quantiles=8,
            seed=0,
        )
        tqc.train(total_timesteps=200)

        ckpt = str(tmp_path / "tqc2.pt")
        tqc.save(ckpt)
        tqc2 = TQC.from_checkpoint(ckpt, env_id="Pendulum-v1")

        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        action = tqc2.predict(obs, deterministic=True)
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(action >= low - 1e-6) and np.all(action <= high + 1e-6)


# ---------------------------------------------------------------------------
# TestTQCConvergence — marked slow, not run in the fast suite
# ---------------------------------------------------------------------------


class TestTQCConvergence:
    """TQC solves Pendulum-v1 under greedy (deterministic) evaluation.

    Pendulum-v1 random baseline: mean reward ~ -1200. This test asserts on a
    greedy eval (predict(deterministic=True), averaged over several
    episodes with a fresh env), NOT the training-loop mean_reward -- that
    figure is dragged down by the random-exploration learning_starts phase
    and stochastic action sampling throughout training.

    Threshold matches the project's CrossQ precedent (docs/plans/
    crossq-convergence-fix-2026-07-18.md): -250 sits comfortably below a
    converged policy's expected performance (~-150 to -200) but well above
    the random baseline.
    """

    @pytest.mark.slow
    # TQC trains an ensemble of n_critics=5 quantile networks (25 quantiles each),
    # so ~19k gradient steps here cost far more than SAC's single critic pair —
    # and the slow job's global --timeout=600 was calibrated for SAC. Measured
    # 351 s locally (Apple silicon); GitHub runners are ~2x slower on these envs,
    # which put it just over 600 s and failed the job at 40 min in. 1200 s is
    # ~3.4x the local time, leaving headroom for runner variance without relaxing
    # the budget for every other slow test. Prefer this over cutting
    # total_timesteps: the -250 threshold below is calibrated to 20k steps, so a
    # smaller budget would mean weakening the assertion.
    @pytest.mark.timeout(1200)
    def test_tqc_greedy_eval_solves_pendulum(self):
        from rlox.algorithms.tqc import TQC

        tqc = TQC(
            env_id="Pendulum-v1",
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=256,
            learning_starts=1000,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            hidden=256,
            n_critics=5,
            n_quantiles=25,
            top_quantiles_to_drop_per_net=2,
            auto_entropy=True,
            ent_coef="auto",
            seed=42,
        )
        tqc.train(total_timesteps=20_000)

        # Greedy evaluation on a fresh env -- deterministic actions, no
        # exploration noise. Per-episode seeds follow the project's
        # eval-seeding convention (seed=base+ep, not a fixed seed=base every
        # episode -- see PROJECT_QUICK_REFERENCE.md "Non-obvious facts").
        n_eval_episodes = 10
        eval_env = gym.make("Pendulum-v1")
        episode_rewards = []
        for ep in range(n_eval_episodes):
            obs, _ = eval_env.reset(seed=42 + ep)
            terminated = truncated = False
            ep_reward = 0.0
            while not (terminated or truncated):
                action = tqc.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_reward += float(reward)
            episode_rewards.append(ep_reward)

        mean_eval_reward = float(np.mean(episode_rewards))
        assert mean_eval_reward > -250, (
            f"Expected greedy-eval mean_reward > -250 after 20k steps on "
            f"Pendulum-v1 (seed=42), got {mean_eval_reward:.1f} over "
            f"{n_eval_episodes} episodes: {episode_rewards}."
        )
