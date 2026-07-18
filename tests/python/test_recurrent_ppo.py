"""Tests for Recurrent PPO (LSTM actor-critic).

Covers, in order:
- construction, config, and the discrete-only guard
- the ``LSTMActorCritic`` hidden-state contract (shapes, episode-boundary
  reset, and the "no gradient flows across `done`" BPTT property)
- segment reconstruction (splitting a rollout into per-episode sequences)
- padding/masking correctness (padded steps must not influence the loss)
- ``train()`` / ``predict()`` end-to-end behaviour
- a slow, seeded CartPole-v1 convergence check

Only ``tests/python/test_recurrent_ppo.py`` should be run for this module
(``./.venv/bin/python -m pytest tests/python/test_recurrent_ppo.py -v``) —
other agents may be running the rest of the suite concurrently.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from rlox.algorithms.recurrent_ppo import (
    LSTMActorCritic,
    RecurrentPPO,
    RecurrentPPOConfig,
    _pad_segments,
    _RecurrentPPOLoss,
    _Rollout,
    _split_into_segments,
)

CARTPOLE_OBS_DIM = 4
CARTPOLE_N_ACTIONS = 2


# ---------------------------------------------------------------------------
# Construction / config / discrete-only guard
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_instantiation_builds_expected_attributes(self):
        ppo = RecurrentPPO(env_id="CartPole-v1", n_envs=2, seed=0, n_steps=8)
        assert isinstance(ppo.config, RecurrentPPOConfig)
        assert isinstance(ppo.policy, LSTMActorCritic)
        assert isinstance(ppo.optimizer, torch.optim.Adam)
        assert ppo.config.n_envs == 2
        assert ppo.config.n_steps == 8

    def test_continuous_action_space_raises_value_error(self):
        with pytest.raises(ValueError, match="discrete"):
            RecurrentPPO(env_id="Pendulum-v1", n_envs=2, seed=0, n_steps=8)

    def test_reproducible_construction(self):
        """Seeding torch/np/env in __init__ must make two instances identical."""
        ppo1 = RecurrentPPO(env_id="CartPole-v1", n_envs=2, seed=123, n_steps=8)
        ppo2 = RecurrentPPO(env_id="CartPole-v1", n_envs=2, seed=123, n_steps=8)
        for p1, p2 in zip(ppo1.policy.parameters(), ppo2.policy.parameters()):
            assert torch.allclose(p1, p2)
        assert np.allclose(ppo1._obs, ppo2._obs)

    def test_config_validates_n_envs(self):
        with pytest.raises(ValueError):
            RecurrentPPOConfig(n_envs=0)


# ---------------------------------------------------------------------------
# LSTMActorCritic hidden-state contract
# ---------------------------------------------------------------------------


class TestLSTMActorCriticHiddenState:
    def test_initial_state_shape(self):
        policy = LSTMActorCritic(
            obs_dim=CARTPOLE_OBS_DIM, n_actions=CARTPOLE_N_ACTIONS, lstm_hidden=16
        )
        h, c = policy.initial_state(n_envs=5)
        assert h.shape == (1, 5, 16)
        assert c.shape == (1, 5, 16)

    def test_act_returns_new_state_of_correct_shape(self):
        policy = LSTMActorCritic(
            obs_dim=CARTPOLE_OBS_DIM, n_actions=CARTPOLE_N_ACTIONS, lstm_hidden=16
        )
        state = policy.initial_state(n_envs=3)
        obs = torch.randn(1, 3, CARTPOLE_OBS_DIM)
        episode_starts = torch.zeros(1, 3)

        action, log_prob, value, new_state = policy.act(obs, state, episode_starts)

        assert action.shape == (1, 3)
        assert log_prob.shape == (1, 3)
        assert value.shape == (1, 3)
        assert new_state[0].shape == (1, 3, 16)
        assert new_state[1].shape == (1, 3, 16)

    def test_hidden_state_resets_at_episode_boundary(self):
        """After episode_starts[t]=True, the trajectory from t onward must
        equal replaying from t with a fresh zero initial state."""
        torch.manual_seed(0)
        policy = LSTMActorCritic(
            obs_dim=CARTPOLE_OBS_DIM, n_actions=CARTPOLE_N_ACTIONS, lstm_hidden=8
        )
        T, B = 6, 1
        obs = torch.randn(T, B, CARTPOLE_OBS_DIM)
        episode_starts = torch.zeros(T, B)
        episode_starts[3, 0] = 1.0  # reset immediately before processing t=3

        # Deliberately non-zero initial state to prove it gets wiped at the reset.
        h0 = torch.randn(1, B, 8)
        c0 = torch.randn(1, B, 8)
        full_features, _ = policy.get_states(obs, (h0, c0), episode_starts)

        suffix_obs = obs[3:]
        suffix_starts = torch.zeros(T - 3, B)
        zero_state = policy.initial_state(n_envs=B)
        suffix_features, _ = policy.get_states(suffix_obs, zero_state, suffix_starts)

        assert torch.allclose(full_features[3:], suffix_features, atol=1e-6)
        # Sanity: the pre-reset run must actually have differed (else the
        # comparison above would be vacuous).
        assert not torch.allclose(full_features[0], suffix_features[0])

    def test_no_gradient_flows_across_episode_boundary(self):
        """A loss computed from a post-`done` timestep must not backprop
        into observations that occurred before the reset."""
        torch.manual_seed(0)
        policy = LSTMActorCritic(
            obs_dim=CARTPOLE_OBS_DIM, n_actions=CARTPOLE_N_ACTIONS, lstm_hidden=8
        )
        T, B = 6, 1
        obs = torch.randn(T, B, CARTPOLE_OBS_DIM, requires_grad=True)
        episode_starts = torch.zeros(T, B)
        episode_starts[3, 0] = 1.0

        state = policy.initial_state(n_envs=B)
        features, _ = policy.get_states(obs, state, episode_starts)
        value = policy.critic(features).squeeze(-1)  # (T, B)

        value[-1, 0].backward()

        assert obs.grad is not None
        assert torch.all(obs.grad[:3] == 0.0), (
            "gradient leaked across the done boundary"
        )
        assert torch.any(obs.grad[3:] != 0.0), "post-reset obs should receive gradient"


# ---------------------------------------------------------------------------
# Segment reconstruction (episode-boundary splitting for BPTT)
# ---------------------------------------------------------------------------


class TestSegmentReconstruction:
    def test_splits_at_episode_boundaries_per_env(self):
        # 2 envs, 5 steps. Env 0 resets at t=2 and t=4. Env 1 never resets
        # again after the rollout-start reset at t=0.
        episode_starts = np.array(
            [
                [1, 1],
                [0, 0],
                [1, 0],
                [0, 0],
                [1, 0],
            ],
            dtype=np.float32,
        )
        segments = _split_into_segments(episode_starts)

        env0 = sorted((s.start, s.end) for s in segments if s.env_idx == 0)
        env1 = sorted((s.start, s.end) for s in segments if s.env_idx == 1)
        assert env0 == [(0, 1), (2, 3), (4, 4)]
        assert env1 == [(0, 4)]

    def test_no_boundary_yields_single_segment_per_env(self):
        episode_starts = np.zeros((4, 3), dtype=np.float32)
        episode_starts[0, :] = 1.0  # rollout-start reset only
        segments = _split_into_segments(episode_starts)
        assert len(segments) == 3
        for seg in segments:
            assert (seg.start, seg.end) == (0, 3)

    def test_continuing_episode_segment_seeds_from_initial_state(self):
        """A segment starting at t=0 with episode_starts[0]=False (episode
        was already in progress) must be seeded from the rollout's carried
        initial hidden state rather than zero."""
        n_envs, lstm_hidden = 1, 4
        episode_starts = np.zeros((3, n_envs), dtype=np.float32)  # never resets
        segments = _split_into_segments(episode_starts)
        assert len(segments) == 1

        rollout = _Rollout(
            obs=torch.randn(3, n_envs, CARTPOLE_OBS_DIM),
            actions=torch.zeros(3, n_envs, dtype=torch.long),
            log_probs=torch.zeros(3, n_envs),
            values=torch.zeros(3, n_envs),
            advantages=torch.zeros(3, n_envs),
            returns=torch.zeros(3, n_envs),
            episode_starts=torch.as_tensor(episode_starts),
            initial_lstm_state=(
                torch.full((1, n_envs, lstm_hidden), 7.0),
                torch.full((1, n_envs, lstm_hidden), -3.0),
            ),
        )
        batch = _pad_segments(segments, rollout, lstm_hidden=lstm_hidden)
        h0, c0 = batch.lstm_state
        assert torch.allclose(h0[0, 0], torch.full((lstm_hidden,), 7.0))
        assert torch.allclose(c0[0, 0], torch.full((lstm_hidden,), -3.0))


# ---------------------------------------------------------------------------
# Padding / masking correctness
# ---------------------------------------------------------------------------


class TestPaddingMask:
    @staticmethod
    def _toy_rollout() -> _Rollout:
        T, n_envs = 4, 2
        obs = torch.randn(T, n_envs, CARTPOLE_OBS_DIM)
        actions = torch.zeros(T, n_envs, dtype=torch.long)
        log_probs = torch.zeros(T, n_envs)
        values = torch.zeros(T, n_envs)
        advantages = torch.randn(T, n_envs)
        returns = torch.randn(T, n_envs)
        episode_starts = torch.zeros(T, n_envs)
        episode_starts[0, :] = 1.0
        episode_starts[2, 0] = (
            1.0  # env 0: two 2-step segments; env 1: one 4-step segment
        )
        initial_state = (torch.zeros(1, n_envs, 8), torch.zeros(1, n_envs, 8))
        return _Rollout(
            obs,
            actions,
            log_probs,
            values,
            advantages,
            returns,
            episode_starts,
            initial_state,
        )

    def test_segments_have_different_lengths_so_padding_is_exercised(self):
        rollout = self._toy_rollout()
        segments = _split_into_segments(rollout.episode_starts.numpy())
        batch = _pad_segments(segments, rollout, lstm_hidden=8)
        assert (batch.mask == 0).any(), "fixture must actually produce padding"

    def test_loss_is_invariant_to_padded_region_content(self):
        torch.manual_seed(0)
        rollout = self._toy_rollout()
        segments = _split_into_segments(rollout.episode_starts.numpy())
        policy = LSTMActorCritic(
            obs_dim=CARTPOLE_OBS_DIM, n_actions=CARTPOLE_N_ACTIONS, lstm_hidden=8
        )
        loss_fn = _RecurrentPPOLoss(
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, clip_vloss=True
        )

        batch_a = _pad_segments(segments, rollout, lstm_hidden=8)
        loss_a, _ = loss_fn(policy, batch_a, batch_a.advantages)

        batch_b = _pad_segments(segments, rollout, lstm_hidden=8)
        pad_positions = batch_b.mask == 0
        assert pad_positions.any()
        corrupted_obs = batch_b.obs.clone()
        corrupted_obs[pad_positions] = torch.randn_like(corrupted_obs[pad_positions])
        # batch is frozen — swap in the corrupted obs via a fresh instance.
        batch_b = dataclasses.replace(batch_b, obs=corrupted_obs)
        loss_b, _ = loss_fn(policy, batch_b, batch_b.advantages)

        assert torch.allclose(loss_a, loss_b, atol=1e-6)


# ---------------------------------------------------------------------------
# train() / predict()
# ---------------------------------------------------------------------------


class TestTrainAndPredict:
    def test_train_short_run_returns_finite_metrics(self):
        ppo = RecurrentPPO(
            env_id="CartPole-v1",
            n_envs=4,
            seed=0,
            n_steps=16,
            n_epochs=2,
            n_minibatches=2,
            hidden=16,
            lstm_hidden=16,
        )
        metrics = ppo.train(total_timesteps=128)

        assert "mean_reward" in metrics
        for key, value in metrics.items():
            assert math.isfinite(value), f"metric {key}={value} is not finite"

    def test_predict_returns_valid_action_and_updates_state(self):
        ppo = RecurrentPPO(
            env_id="CartPole-v1", n_envs=2, seed=0, n_steps=8, hidden=16, lstm_hidden=16
        )
        ppo.reset_predict_state()
        obs = np.zeros(CARTPOLE_OBS_DIM, dtype=np.float32)

        state_before = tuple(t.clone() for t in ppo._predict_lstm_state)
        action = ppo.predict(obs, deterministic=False)
        assert action in (0, 1)

        state_after = ppo._predict_lstm_state
        assert not torch.allclose(state_before[0], state_after[0])

        # A second call continues from the updated (non-reset) state.
        action2 = ppo.predict(obs, deterministic=True)
        assert action2 in (0, 1)

    def test_reset_predict_state_zeros_the_state(self):
        ppo = RecurrentPPO(
            env_id="CartPole-v1", n_envs=2, seed=0, n_steps=8, hidden=16, lstm_hidden=16
        )
        obs = np.zeros(CARTPOLE_OBS_DIM, dtype=np.float32)
        ppo.reset_predict_state()
        ppo.predict(obs)

        ppo.reset_predict_state()
        h, c = ppo._predict_lstm_state
        assert torch.all(h == 0.0)
        assert torch.all(c == 0.0)


# ---------------------------------------------------------------------------
# Slow convergence check
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCartPoleConvergence:
    def test_recurrent_ppo_solves_cartpole(self):
        """Seeded end-to-end convergence check.

        CartPole-v1 is fully observable, so a recurrent policy should still
        solve it; the LSTM must not prevent convergence.
        """
        ppo = RecurrentPPO(
            env_id="CartPole-v1",
            n_envs=8,
            seed=42,
            n_steps=32,
            n_epochs=4,
            n_minibatches=4,
            learning_rate=1e-3,
            hidden=64,
            lstm_hidden=64,
        )
        ppo.train(total_timesteps=300_000)

        recent = ppo.episode_rewards[-20:]
        mean_recent = float(np.mean(recent)) if recent else 0.0
        print(
            f"\nRecurrentPPO CartPole-v1: mean reward over last "
            f"{len(recent)} episodes = {mean_recent:.1f}"
        )
        assert mean_recent > 400, f"Expected > 400, got {mean_recent:.1f}"
