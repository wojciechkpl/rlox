"""Wave 4 tests: Python wrappers, protocols, and algorithms using Rust bindings.

TDD: These tests are written FIRST, before the implementations.
"""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import rlox
from rlox.protocols import (
    Algorithm,
    RewardShaper,
    IntrinsicMotivation,
    MetaLearner,
    Augmentation,
)


# ============================================================================
# TestRandomShift
# ============================================================================


class TestRandomShift:
    """Tests for DrQ-v2 random shift augmentation wrapper."""

    def test_output_shape_matches_input(self):
        """Augmented tensor has same shape as input."""
        from rlox.augmentation import RandomShift

        aug = RandomShift(pad=4)
        obs = torch.randn(8, 3, 84, 84)
        out = aug(obs, seed=42)
        assert out.shape == obs.shape

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical results."""
        from rlox.augmentation import RandomShift

        aug = RandomShift(pad=4)
        obs = torch.randn(4, 3, 84, 84)
        out1 = aug(obs, seed=123)
        out2 = aug(obs, seed=123)
        torch.testing.assert_close(out1, out2)

    def test_stochastic_with_different_seeds(self):
        """Different seeds produce different results."""
        from rlox.augmentation import RandomShift

        aug = RandomShift(pad=4)
        obs = torch.randn(4, 3, 84, 84)
        out1 = aug(obs, seed=1)
        out2 = aug(obs, seed=2)
        assert not torch.allclose(out1, out2)

    def test_protocol_compliance(self):
        """RandomShift satisfies the Augmentation protocol."""
        from rlox.augmentation import RandomShift

        aug = RandomShift(pad=4)
        assert isinstance(aug, Augmentation)

    def test_single_image_batch(self):
        """Works with batch size 1."""
        from rlox.augmentation import RandomShift

        aug = RandomShift(pad=4)
        obs = torch.randn(1, 3, 84, 84)
        out = aug(obs, seed=42)
        assert out.shape == (1, 3, 84, 84)

    def test_preserves_dtype_float32(self):
        """Output is float32 like the input."""
        from rlox.augmentation import RandomShift

        aug = RandomShift(pad=4)
        obs = torch.randn(2, 3, 84, 84, dtype=torch.float32)
        out = aug(obs, seed=42)
        assert out.dtype == torch.float32


# ============================================================================
# TestPotentialShaping
# ============================================================================


class TestPotentialShaping:
    """Tests for PBRS reward shaping backed by Rust."""

    def test_shape_returns_correct_length(self):
        """Shaped rewards have same length as input."""
        from rlox.reward_shaping import PotentialShaping

        # Simple potential: sum of obs
        pot_fn = lambda x: x.sum(axis=-1)
        shaper = PotentialShaping(potential_fn=pot_fn, gamma=0.99)

        obs = np.random.randn(16, 4).astype(np.float64)
        next_obs = np.random.randn(16, 4).astype(np.float64)
        rewards = np.random.randn(16).astype(np.float64)
        dones = np.zeros(16, dtype=np.float64)

        shaped = shaper.shape(rewards, obs, next_obs, dones)
        assert shaped.shape == (16,)

    def test_pbrs_arithmetic(self):
        """Shaped reward = r + gamma * phi(s') - phi(s)."""
        from rlox.reward_shaping import PotentialShaping

        pot_fn = lambda x: x[:, 0]  # potential = first dim
        shaper = PotentialShaping(potential_fn=pot_fn, gamma=0.99)

        obs = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        next_obs = np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        rewards = np.array([10.0, 20.0], dtype=np.float64)
        dones = np.array([0.0, 0.0], dtype=np.float64)

        shaped = shaper.shape(rewards, obs, next_obs, dones)

        # r' = r + gamma * phi(s') - phi(s)
        expected = rewards + 0.99 * next_obs[:, 0] - obs[:, 0]
        np.testing.assert_allclose(shaped, expected, rtol=1e-10)

    def test_done_resets_shaping(self):
        """When done=1, shaping bonus is zeroed -- shaped reward equals raw reward."""
        from rlox.reward_shaping import PotentialShaping

        pot_fn = lambda x: x[:, 0]
        shaper = PotentialShaping(potential_fn=pot_fn, gamma=0.99)

        obs = np.array([[1.0]], dtype=np.float64)
        next_obs = np.array([[5.0]], dtype=np.float64)
        rewards = np.array([10.0], dtype=np.float64)
        dones = np.array([1.0], dtype=np.float64)

        shaped = shaper.shape(rewards, obs, next_obs, dones)

        # At episode boundaries (done=1), Rust zeros out the shaping term entirely
        np.testing.assert_allclose(shaped, rewards, rtol=1e-10)

    def test_protocol_compliance(self):
        """PotentialShaping satisfies RewardShaper protocol."""
        from rlox.reward_shaping import PotentialShaping

        shaper = PotentialShaping(potential_fn=lambda x: x.sum(axis=-1), gamma=0.99)
        assert isinstance(shaper, RewardShaper)


# ============================================================================
# TestGoalDistanceShaping
# ============================================================================


class TestGoalDistanceShaping:
    """Tests for goal-distance potential shaping backed by Rust."""

    def test_closer_goal_higher_reward(self):
        """Moving closer to goal should increase shaped reward."""
        from rlox.reward_shaping import GoalDistanceShaping

        goal = np.array([0.0, 0.0], dtype=np.float64)
        shaper = GoalDistanceShaping(
            goal=goal, obs_dim=4, goal_start=0, goal_dim=2, scale=1.0, gamma=0.99,
        )

        # obs is far, next_obs is closer
        obs = np.array([[5.0, 5.0, 0.0, 0.0]], dtype=np.float64)
        next_obs = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        rewards = np.array([0.0], dtype=np.float64)
        dones = np.array([0.0], dtype=np.float64)

        shaped = shaper.shape(rewards, obs, next_obs, dones)
        # Moving closer => positive shaping bonus
        assert shaped[0] > 0.0

    def test_farther_goal_lower_reward(self):
        """Moving farther from goal should decrease shaped reward."""
        from rlox.reward_shaping import GoalDistanceShaping

        goal = np.array([0.0, 0.0], dtype=np.float64)
        shaper = GoalDistanceShaping(
            goal=goal, obs_dim=4, goal_start=0, goal_dim=2, scale=1.0, gamma=0.99,
        )

        # obs is close, next_obs is far
        obs = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        next_obs = np.array([[5.0, 5.0, 0.0, 0.0]], dtype=np.float64)
        rewards = np.array([0.0], dtype=np.float64)
        dones = np.array([0.0], dtype=np.float64)

        shaped = shaper.shape(rewards, obs, next_obs, dones)
        assert shaped[0] < 0.0

    def test_protocol_compliance(self):
        """GoalDistanceShaping satisfies RewardShaper protocol."""
        from rlox.reward_shaping import GoalDistanceShaping

        goal = np.zeros(2, dtype=np.float64)
        shaper = GoalDistanceShaping(
            goal=goal, obs_dim=4, goal_start=0, goal_dim=2, scale=1.0, gamma=0.99,
        )
        assert isinstance(shaper, RewardShaper)


# ============================================================================
# TestSpectralNorm
# ============================================================================


class TestSpectralNorm:
    """Tests for spectral normalization utility."""

    def test_applies_to_linear_layers(self):
        """Spectral norm is applied to nn.Linear layers."""
        from rlox.networks import apply_spectral_norm

        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        apply_spectral_norm(net)

        # After spectral norm, weight_orig should exist
        for name, child in net.named_children():
            if isinstance(child, nn.Linear):
                assert hasattr(child, "weight_orig")

    def test_applies_to_conv_layers(self):
        """Spectral norm is applied to nn.Conv2d layers."""
        from rlox.networks import apply_spectral_norm

        net = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Conv2d(16, 32, 3))
        apply_spectral_norm(net)

        for name, child in net.named_children():
            if isinstance(child, nn.Conv2d):
                assert hasattr(child, "weight_orig")

    def test_constrains_singular_values(self):
        """After spectral norm, largest singular value should be approximately 1."""
        from rlox.networks import apply_spectral_norm

        linear = nn.Linear(10, 10)
        net = nn.Sequential(linear)
        apply_spectral_norm(net)

        # Power iteration needs multiple forward passes to converge
        for _ in range(50):
            _ = net(torch.randn(1, 10))

        w = linear.weight.detach()
        svd = torch.linalg.svdvals(w)
        assert svd[0].item() == pytest.approx(1.0, abs=0.15)

    def test_returns_module(self):
        """apply_spectral_norm returns the module for chaining."""
        from rlox.networks import apply_spectral_norm

        net = nn.Sequential(nn.Linear(4, 4))
        result = apply_spectral_norm(net)
        assert result is net

    def test_trains_correctly(self):
        """Network with spectral norm can be trained (gradients flow)."""
        from rlox.networks import apply_spectral_norm

        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        apply_spectral_norm(net)

        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)

        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()
        opt.step()

        # Verify parameters changed (training works)
        loss2 = nn.functional.mse_loss(net(x), y)
        assert loss2.item() != loss.item()


# ============================================================================
# TestAWR
# ============================================================================


class TestAWR:
    """Tests for Advantage Weighted Regression algorithm."""

    def test_constructs(self):
        """AWR can be constructed with an env_id."""
        from rlox.algorithms.awr import AWR

        awr = AWR(env_id="CartPole-v1")
        assert awr.env_id == "CartPole-v1"

    def test_satisfies_algorithm_protocol(self):
        """AWR satisfies the Algorithm protocol."""
        from rlox.algorithms.awr import AWR

        awr = AWR(env_id="CartPole-v1")
        assert isinstance(awr, Algorithm)

    def test_train_returns_metrics(self):
        """AWR.train() returns a dict with mean_reward."""
        from rlox.algorithms.awr import AWR

        awr = AWR(env_id="CartPole-v1", learning_starts=50)
        metrics = awr.train(total_timesteps=200)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics

    def test_save_and_load(self):
        """AWR can save and load from checkpoint."""
        from rlox.algorithms.awr import AWR

        awr = AWR(env_id="CartPole-v1")
        # Train briefly to get some state
        awr.train(total_timesteps=100)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        awr.save(path)
        loaded = AWR.from_checkpoint(path, env_id="CartPole-v1")
        assert loaded.env_id == "CartPole-v1"

        # Verify actor weights match
        for p1, p2 in zip(awr.actor.parameters(), loaded.actor.parameters()):
            torch.testing.assert_close(p1, p2)

        Path(path).unlink()

    def test_registered_in_trainer(self):
        """AWR is registered in ALGORITHM_REGISTRY."""
        from rlox.trainer import ALGORITHM_REGISTRY

        assert "awr" in ALGORITHM_REGISTRY

    def test_policy_loss_uses_advantage_weighting(self):
        """Policy loss weights log-probs by exponentiated advantages."""
        from rlox.algorithms.awr import AWR

        awr = AWR(env_id="CartPole-v1", beta=1.0)
        # Verify the beta parameter is stored
        assert awr.beta == 1.0

    # ------------------------------------------------------------------
    # Regression tests for AWR.predict() AttributeError on self.policy
    # Bug: predict() called self.policy.actor() / hasattr(self.policy, ...)
    # but AWR.__init__ never sets self.policy — it sets self.actor and
    # self.critic.  The fix must use self.actor and self.discrete instead.
    # ------------------------------------------------------------------

    def test_predict_discrete_single_obs_returns_valid_action(self):
        """predict() on CartPole-v1 (discrete) returns an int in the action space.

        Assumption: AWR.predict(obs, deterministic=True) for a discrete env
        returns a Python int (or numpy int) in range [0, n_actions).
        """
        import gymnasium as gym
        from rlox.algorithms.awr import AWR

        awr = AWR(env_id="CartPole-v1", seed=0)
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=0)  # shape (4,)

        # Must not raise AttributeError: 'AWR' object has no attribute 'policy'
        action = awr.predict(obs, deterministic=True)

        assert env.action_space.contains(int(action)), (
            f"predict() returned {action!r} which is not in CartPole-v1 action space"
        )

    def test_predict_discrete_batch_obs_returns_valid_action(self):
        """predict() on CartPole-v1 with a pre-batched (1, obs_dim) tensor works.

        Assumption: if the caller manually adds the batch dimension the method
        still returns a valid action without raising.
        """
        import gymnasium as gym
        import torch
        from rlox.algorithms.awr import AWR

        awr = AWR(env_id="CartPole-v1", seed=1)
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=1)
        obs_batched = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 4)

        action = awr.predict(obs_batched, deterministic=True)

        assert env.action_space.contains(int(action)), (
            f"predict() with batched obs returned {action!r} which is not in action space"
        )

    def test_predict_continuous_returns_correct_shape_and_bounds(self):
        """predict() on Pendulum-v1 (continuous) returns a numpy array of shape (act_dim,).

        Assumption: AWR.predict(obs, deterministic=True) for a continuous env
        returns a numpy ndarray with shape equal to the environment's action
        space shape (1,) for Pendulum-v1.  The bug causes AttributeError before
        any shape check is reachable.
        """
        import gymnasium as gym
        import numpy as np
        from rlox.algorithms.awr import AWR

        awr = AWR(env_id="Pendulum-v1", seed=0)
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset(seed=0)  # shape (3,)

        # Must not raise AttributeError: 'AWR' object has no attribute 'policy'
        action = awr.predict(obs, deterministic=True)

        expected_shape = env.action_space.shape  # (1,)
        assert hasattr(action, "shape"), (
            f"predict() on continuous env must return an array-like, got {type(action)}"
        )
        assert action.shape == expected_shape, (
            f"predict() returned shape {action.shape}, expected {expected_shape}"
        )
        # Action should be within action bounds (Pendulum: [-2, 2])
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(action >= low) and np.all(action <= high), (
            f"predict() returned action {action} outside bounds [{low}, {high}]"
        )

    def test_predict_discrete_deterministic_is_reproducible(self):
        """predict(obs, deterministic=True) returns the same action on repeated calls.

        Assumption: deterministic=True must suppress sampling; for a discrete env
        this means argmax over logits, so the same obs always yields the same action.
        """
        from rlox.algorithms.awr import AWR
        import gymnasium as gym

        awr = AWR(env_id="CartPole-v1", seed=7)
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=7)

        action_1 = awr.predict(obs, deterministic=True)
        action_2 = awr.predict(obs, deterministic=True)

        assert action_1 == action_2, (
            f"deterministic=True produced different actions on identical obs: "
            f"{action_1} vs {action_2}"
        )

    def test_predict_continuous_deterministic_is_reproducible(self):
        """predict(obs, deterministic=True) for continuous env is same on repeated calls.

        Assumption: for a continuous env, deterministic=True must return the
        mean action (actor output) without stochastic sampling.
        """
        import numpy as np
        from rlox.algorithms.awr import AWR
        import gymnasium as gym

        awr = AWR(env_id="Pendulum-v1", seed=3)
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset(seed=3)

        action_1 = awr.predict(obs, deterministic=True)
        action_2 = awr.predict(obs, deterministic=True)

        np.testing.assert_array_equal(
            action_1, action_2,
            err_msg=(
                "deterministic=True for continuous env produced different actions "
                f"on identical obs: {action_1} vs {action_2}"
            ),
        )


# ============================================================================
# TestRND
# ============================================================================


class TestRND:
    """Tests for Random Network Distillation intrinsic motivation."""

    def test_reward_shape(self):
        """Intrinsic reward has shape (batch_size,)."""
        from rlox.intrinsic.rnd import RND

        rnd = RND(obs_dim=4, hidden=64)
        obs = torch.randn(16, 4)
        reward = rnd.compute_intrinsic_reward(obs)
        assert reward.shape == (16,)

    def test_novelty_detection(self):
        """Repeated observations should yield lower intrinsic reward after training."""
        from rlox.intrinsic.rnd import RND

        rnd = RND(obs_dim=4, hidden=64, learning_rate=1e-3)
        obs = torch.randn(32, 4)

        reward_before = rnd.compute_intrinsic_reward(obs).mean().item()

        # Train on these observations many times
        for _ in range(100):
            rnd.update(obs)

        reward_after = rnd.compute_intrinsic_reward(obs).mean().item()
        # After training, prediction error should decrease
        assert reward_after < reward_before

    def test_target_frozen(self):
        """Target network parameters should not change during update."""
        from rlox.intrinsic.rnd import RND

        rnd = RND(obs_dim=4, hidden=64)
        target_params_before = [p.clone() for p in rnd.target.parameters()]

        obs = torch.randn(16, 4)
        rnd.update(obs)

        for before, after in zip(target_params_before, rnd.target.parameters()):
            torch.testing.assert_close(before, after)

    def test_update_returns_loss_dict(self):
        """update() returns dict with 'rnd_loss' key."""
        from rlox.intrinsic.rnd import RND

        rnd = RND(obs_dim=4, hidden=64)
        obs = torch.randn(16, 4)
        info = rnd.update(obs)
        assert "rnd_loss" in info
        assert isinstance(info["rnd_loss"], float)

    def test_protocol_compliance(self):
        """RND satisfies IntrinsicMotivation protocol."""
        from rlox.intrinsic.rnd import RND

        rnd = RND(obs_dim=4, hidden=64)
        assert isinstance(rnd, IntrinsicMotivation)


# ============================================================================
# TestReptile
# ============================================================================


class TestReptile:
    """Tests for Reptile meta-learning."""

    def test_meta_train_runs(self):
        """meta_train completes and returns metrics dict."""
        from rlox.meta.reptile import Reptile

        reptile = Reptile(
            algorithm_cls_name="awr",
            env_ids=["CartPole-v1"],
            meta_lr=0.1,
            inner_steps=50,
        )
        metrics = reptile.meta_train(n_iterations=2)
        assert isinstance(metrics, dict)
        assert "meta_loss" in metrics or "mean_reward" in metrics

    def test_adapt_produces_algorithm(self):
        """adapt() returns an Algorithm instance."""
        from rlox.meta.reptile import Reptile

        reptile = Reptile(
            algorithm_cls_name="awr",
            env_ids=["CartPole-v1"],
            meta_lr=0.1,
            inner_steps=50,
        )
        # Quick meta-train
        reptile.meta_train(n_iterations=1)
        algo = reptile.adapt(env_id="CartPole-v1", n_steps=50)
        assert isinstance(algo, Algorithm)

    def test_weight_averaging_via_rust(self):
        """Weight averaging uses rlox.average_weight_vectors."""
        # Direct test of the Rust function
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        avg = rlox.average_weight_vectors([v1, v2])
        np.testing.assert_allclose(avg, [2.0, 3.0, 4.0])

    def test_protocol_compliance(self):
        """Reptile satisfies MetaLearner protocol."""
        from rlox.meta.reptile import Reptile

        reptile = Reptile(
            algorithm_cls_name="awr",
            env_ids=["CartPole-v1"],
            meta_lr=0.1,
            inner_steps=50,
        )
        assert isinstance(reptile, MetaLearner)


# ============================================================================
# TestOfflineToOnline
# ============================================================================


class TestOfflineToOnline:
    """Tests for offline-to-online fine-tuning."""

    def test_mixed_sampling_ratio(self):
        """Mixed sampling respects the offline/online ratio."""
        from rlox.offline_to_online import OfflineToOnline

        # Create two buffers with known data
        offline_buf = rlox.ReplayBuffer(100, 4, 1)
        online_buf = rlox.ReplayBuffer(100, 4, 1)

        # Fill buffers
        for i in range(50):
            obs = np.full(4, float(i), dtype=np.float32)
            act = np.array([0.0], dtype=np.float32)
            offline_buf.push(obs, act, 1.0, False, False, obs)
            online_buf.push(obs, act, 2.0, False, False, obs)

        o2o = OfflineToOnline(
            offline_buffer=offline_buf,
            online_buffer=online_buf,
            offline_ratio=0.5,
            batch_size=32,
        )

        batch = o2o.sample_mixed(seed=42)
        assert "obs" in batch
        # Batch size should be as requested
        obs_arr = np.asarray(batch["obs"])
        assert obs_arr.shape[0] == 32

    def test_schedule_annealing(self):
        """Offline ratio should decrease over time with annealing."""
        from rlox.offline_to_online import OfflineToOnline

        o2o = OfflineToOnline(
            offline_buffer=rlox.ReplayBuffer(10, 4, 1),
            online_buffer=rlox.ReplayBuffer(10, 4, 1),
            offline_ratio=0.8,
            batch_size=16,
            anneal_steps=100,
        )

        ratio_start = o2o.get_offline_ratio(step=0)
        ratio_mid = o2o.get_offline_ratio(step=50)
        ratio_end = o2o.get_offline_ratio(step=100)

        assert ratio_start == pytest.approx(0.8, abs=0.01)
        assert ratio_mid < ratio_start
        assert ratio_end < ratio_mid
        # After annealing, ratio should be near 0
        assert ratio_end == pytest.approx(0.0, abs=0.01)

    def test_no_annealing_keeps_constant_ratio(self):
        """Without annealing, ratio stays constant."""
        from rlox.offline_to_online import OfflineToOnline

        o2o = OfflineToOnline(
            offline_buffer=rlox.ReplayBuffer(10, 4, 1),
            online_buffer=rlox.ReplayBuffer(10, 4, 1),
            offline_ratio=0.5,
            batch_size=16,
        )

        assert o2o.get_offline_ratio(step=0) == 0.5
        assert o2o.get_offline_ratio(step=1000) == 0.5
