"""Tests for tier-1 feature improvements.

Covers:
- trainer.evaluate() and trainer.enjoy()
- VideoRecordingCallback
- Episode stats tracking in RolloutCollector
- RecordEpisodeStatistics auto-wrap in GymVecEnv
- Score normalization (normalize_score, normalize_scores, SCORE_BASELINES)
- AsymmetricPolicy
- CI bands on learning curves (_compute_learning_curve_ci)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ===========================================================================
# trainer.evaluate() and trainer.enjoy()
# ===========================================================================


class TestEvaluateOnRealEnvironments:
    """`evaluate()` against real Gymnasium envs — no mocking.

    The mocked tests below exercise evaluate()'s bookkeeping, but a MagicMock
    env accepts any action type, so they cannot catch the thing that actually
    broke: algorithms return actions in whatever type suits them (PPO yields a
    `torch.Tensor`), and a real `Discrete` space rejects anything that is not an
    int. `evaluate()` passed the policy output straight to `env.step()`, so it
    raised `AssertionError: tensor([0]) invalid` on *every* discrete Gymnasium
    environment — CartPole-v1 included, which is the first example in the docs.

    These tests use real envs so that failure mode cannot hide again.
    """

    @pytest.mark.parametrize(
        "env_id,max_return",
        [
            ("CartPole-v1", 500.0),  # Discrete(2)
            ("Pendulum-v1", 0.0),  # Box(1,) — returns are negative
        ],
    )
    def test_evaluate_runs_and_returns_episode_returns(self, env_id, max_return):
        from rlox.trainer import Trainer

        trainer = Trainer(
            "ppo", env=env_id, config={"n_envs": 1, "n_steps": 64}, seed=42
        )
        trainer.train(total_timesteps=128)

        result = trainer.evaluate(n_episodes=2, seed=0)

        assert result["n_episodes"] == 2
        assert result["min_reward"] <= result["mean_reward"] <= result["max_reward"]
        # An *episode return*, not a rollout reward-sum. The distinction matters:
        # train()'s "mean_reward" is sum(rewards)/n_envs over a rollout, so it
        # scales with n_steps and can exceed the environment's episode maximum.
        assert result["mean_reward"] <= max_return, (
            f"{env_id}: mean_reward={result['mean_reward']} exceeds the maximum "
            f"attainable episode return ({max_return}) — this looks like a "
            f"rollout sum rather than an episode return"
        )
        assert result["mean_length"] > 0

    def test_evaluate_is_comparable_to_a_manual_rollout(self):
        """evaluate()'s mean_reward must match a hand-rolled greedy rollout.

        This pins the semantics the paper's onboarding listing relies on: that
        `trainer.evaluate(...)["mean_reward"]` means the same thing as SB3's
        `evaluate_policy(...)` return, so the two are directly comparable.
        """
        import gymnasium as gym

        from rlox.trainer import Trainer

        trainer = Trainer(
            "ppo", env="CartPole-v1", config={"n_envs": 1, "n_steps": 64}, seed=42
        )
        trainer.train(total_timesteps=128)

        got = trainer.evaluate(n_episodes=3, seed=0)["mean_reward"]

        env = gym.make("CartPole-v1")
        returns = []
        for ep in range(3):
            obs, _ = env.reset(seed=0 + ep)
            done, total = False, 0.0
            while not done:
                action = trainer.predict(obs, deterministic=True)
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                obs, r, term, trunc, _ = env.step(int(np.asarray(action).reshape(-1)[0]))
                total += float(r)
                done = term or trunc
            returns.append(total)
        env.close()

        assert got == pytest.approx(float(np.mean(returns))), (
            "evaluate() must report the mean greedy episode return"
        )


def _make_mock_env(n_steps_per_episode: int = 5):
    """Build a mock gymnasium Env that runs for a fixed number of steps.

    The mock env returns obs of shape (4,), reward=1.0 per step, and
    terminates after ``n_steps_per_episode`` steps.
    """
    obs_arr = np.zeros(4, dtype=np.float32)
    call_counts = {"steps": 0}

    mock_env = MagicMock()
    mock_env.reset.return_value = (obs_arr.copy(), {})

    def _step(_action):
        call_counts["steps"] += 1
        terminated = call_counts["steps"] >= n_steps_per_episode
        if terminated:
            call_counts["steps"] = 0
        return obs_arr.copy(), 1.0, terminated, False, {}

    mock_env.step.side_effect = _step
    mock_env.close = MagicMock()
    return mock_env


class TestTrainerEvaluate:
    """Tests for Trainer.evaluate() bookkeeping, against a mock env.

    Mocking keeps these fast and lets them pin exact episode counts and lengths.
    It does NOT exercise action/observation marshalling: a MagicMock env accepts
    any action type, which is precisely how a crash on every real discrete env
    went unnoticed. `TestEvaluateOnRealEnvironments` above covers that; keep both.
    """

    def _make_trainer(self):
        """Build a minimally-trained PPO trainer, no actual env calls needed."""
        from rlox.trainer import Trainer

        trainer = Trainer(
            "ppo",
            env="CartPole-v1",
            config={"n_envs": 1, "n_steps": 8},
            seed=42,
        )
        trainer.train(total_timesteps=64)
        return trainer

    def test_evaluate_returns_required_keys(self):
        """evaluate() must return a dict with all six required keys."""
        trainer = self._make_trainer()
        mock_env = _make_mock_env(n_steps_per_episode=3)
        with patch("gymnasium.make", return_value=mock_env):
            result = trainer.evaluate(n_episodes=3, seed=0)

        required = {"mean_reward", "std_reward", "min_reward", "max_reward",
                    "mean_length", "n_episodes"}
        assert required == set(result.keys())

    def test_evaluate_n_episodes_reflected(self):
        """n_episodes in the result must equal the argument passed."""
        trainer = self._make_trainer()
        mock_env = _make_mock_env(n_steps_per_episode=2)
        with patch("gymnasium.make", return_value=mock_env):
            n = 5
            result = trainer.evaluate(n_episodes=n, seed=0)
        assert result["n_episodes"] == n

    def test_evaluate_min_max_ordering(self):
        """min_reward <= mean_reward <= max_reward must hold."""
        trainer = self._make_trainer()
        mock_env = _make_mock_env(n_steps_per_episode=3)
        with patch("gymnasium.make", return_value=mock_env):
            result = trainer.evaluate(n_episodes=5, seed=1)
        assert result["min_reward"] <= result["mean_reward"]
        assert result["mean_reward"] <= result["max_reward"]

    def test_evaluate_std_non_negative(self):
        """std_reward must be >= 0."""
        trainer = self._make_trainer()
        mock_env = _make_mock_env(n_steps_per_episode=2)
        with patch("gymnasium.make", return_value=mock_env):
            result = trainer.evaluate(n_episodes=4, seed=7)
        assert result["std_reward"] >= 0.0

    def test_evaluate_mean_length_positive(self):
        """mean_length must be > 0 (at least one step per episode)."""
        trainer = self._make_trainer()
        mock_env = _make_mock_env(n_steps_per_episode=4)
        with patch("gymnasium.make", return_value=mock_env):
            result = trainer.evaluate(n_episodes=3, seed=0)
        assert result["mean_length"] > 0.0

    def test_evaluate_all_values_are_float(self):
        """All values in the returned dict must be Python floats or ints."""
        trainer = self._make_trainer()
        mock_env = _make_mock_env(n_steps_per_episode=2)
        with patch("gymnasium.make", return_value=mock_env):
            result = trainer.evaluate(n_episodes=2, seed=0)
        for key, val in result.items():
            assert isinstance(val, (float, int)), (
                f"Expected float/int for key {key!r}, got {type(val)}"
            )

    def test_evaluate_reward_equals_steps_per_episode(self):
        """With reward=1.0/step and fixed episode length, mean_reward == n_steps."""
        trainer = self._make_trainer()
        n_steps = 7
        mock_env = _make_mock_env(n_steps_per_episode=n_steps)
        with patch("gymnasium.make", return_value=mock_env):
            result = trainer.evaluate(n_episodes=4, seed=0)
        assert result["mean_reward"] == pytest.approx(float(n_steps))
        assert result["mean_length"] == pytest.approx(float(n_steps))

    def test_evaluate_freezes_and_restores_vec_normalize(self):
        """vec_normalize.training must be restored to True after evaluate()."""
        from rlox.trainer import Trainer

        trainer = Trainer(
            "ppo",
            env="CartPole-v1",
            config={"n_envs": 1, "n_steps": 8},
            seed=0,
        )
        trainer.train(total_timesteps=64)

        mock_vn = MagicMock()
        mock_vn.training = True
        # normalize_obs must return a flat numpy array matching obs_dim
        mock_vn.normalize_obs.return_value = np.zeros((1, 4), dtype=np.float32)
        trainer.algo.vec_normalize = mock_vn

        mock_env = _make_mock_env(n_steps_per_episode=2)
        with patch("gymnasium.make", return_value=mock_env):
            trainer.evaluate(n_episodes=1, seed=0)

        # Training mode must be restored regardless of exception path
        assert mock_vn.training is True

    def test_evaluate_env_closed_after_call(self):
        """evaluate() must close the environment it created."""
        trainer = self._make_trainer()
        mock_env = _make_mock_env(n_steps_per_episode=2)
        with patch("gymnasium.make", return_value=mock_env):
            trainer.evaluate(n_episodes=2, seed=0)
        mock_env.close.assert_called_once()

    def test_evaluate_without_vec_normalize(self):
        """evaluate() works when algo has no vec_normalize attribute."""
        from rlox.trainer import Trainer

        trainer = Trainer(
            "ppo",
            env="CartPole-v1",
            config={"n_envs": 1, "n_steps": 8},
            seed=3,
        )
        trainer.train(total_timesteps=64)

        if hasattr(trainer.algo, "vec_normalize"):
            del trainer.algo.vec_normalize

        mock_env = _make_mock_env(n_steps_per_episode=2)
        with patch("gymnasium.make", return_value=mock_env):
            result = trainer.evaluate(n_episodes=2, seed=0)
        assert "mean_reward" in result


class TestTrainerEnjoy:
    """Tests for Trainer.enjoy()."""

    def _make_trainer(self):
        from rlox.trainer import Trainer

        trainer = Trainer(
            "ppo",
            env="CartPole-v1",
            config={"n_envs": 1, "n_steps": 8},
            seed=0,
        )
        trainer.train(total_timesteps=64)
        return trainer

    def test_enjoy_delegates_to_evaluate_with_render(self):
        """enjoy() must call evaluate() with render=True."""
        trainer = self._make_trainer()

        captured_kwargs: dict = {}

        def spy_evaluate(**kwargs):
            captured_kwargs.update(kwargs)
            # Intercept; don't call real evaluate (it needs a real env)
            return {"mean_reward": 0.0, "std_reward": 0.0, "min_reward": 0.0,
                    "max_reward": 0.0, "mean_length": 1.0, "n_episodes": 2}

        trainer.evaluate = spy_evaluate
        trainer.enjoy(n_episodes=2, seed=5)

        assert captured_kwargs.get("n_episodes") == 2
        assert captured_kwargs.get("seed") == 5
        assert captured_kwargs.get("render") is True

    def test_enjoy_default_args(self):
        """enjoy() defaults: n_episodes=1, seed=0."""
        trainer = self._make_trainer()

        captured_kwargs: dict = {}

        def spy_evaluate(**kwargs):
            captured_kwargs.update(kwargs)
            return {"mean_reward": 0.0, "std_reward": 0.0, "min_reward": 0.0,
                    "max_reward": 0.0, "mean_length": 1.0, "n_episodes": 1}

        trainer.evaluate = spy_evaluate
        trainer.enjoy()

        assert captured_kwargs.get("n_episodes") == 1
        assert captured_kwargs.get("seed") == 0

    def test_enjoy_returns_none(self):
        """enjoy() has no return value (returns None)."""
        trainer = self._make_trainer()

        mock_env = _make_mock_env(n_steps_per_episode=2)
        with patch("gymnasium.make", return_value=mock_env):
            result = trainer.enjoy(n_episodes=1, seed=0)

        assert result is None


# ===========================================================================
# VideoRecordingCallback
# ===========================================================================


class TestVideoRecordingCallback:
    """Tests for VideoRecordingCallback."""

    def test_init_defaults(self):
        """Constructor defaults match documented values."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback()
        assert cb.video_folder == "videos"
        assert cb.record_freq == 50000
        assert cb.n_episodes == 1
        assert cb.verbose is True

    def test_init_custom(self):
        """Constructor accepts custom parameters."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(
            video_folder="/tmp/myvids",
            record_freq=10000,
            n_episodes=3,
            verbose=False,
        )
        assert cb.video_folder == "/tmp/myvids"
        assert cb.record_freq == 10000
        assert cb.n_episodes == 3
        assert cb.verbose is False

    def test_on_step_returns_true(self):
        """on_step must always return True (never stops training)."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(record_freq=1000)
        result = cb.on_step(step=1)
        assert result is True

    def test_on_step_before_record_freq_no_recording(self):
        """No recording should happen before the first record_freq boundary."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(record_freq=50000, video_folder="/tmp/novid")

        mock_algo = MagicMock()
        mock_algo.predict = MagicMock(return_value=0)
        mock_algo.env_id = "CartPole-v1"

        with patch("gymnasium.make") as mock_make:
            for step in range(1, 1000):
                cb.on_step(step=step, algo=mock_algo)
            mock_make.assert_not_called()

    def test_on_step_without_algo_skips_recording(self):
        """If algo kwarg is missing, on_step should skip silently."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(record_freq=1)
        with patch("gymnasium.make") as mock_make:
            result = cb.on_step(step=1)
        assert result is True
        mock_make.assert_not_called()

    def test_on_step_without_env_id_skips_recording(self):
        """If algo has no env_id, on_step should skip silently."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(record_freq=1)
        mock_algo = MagicMock(spec=[])  # no attributes
        mock_algo.predict = MagicMock(return_value=0)

        with patch("gymnasium.make") as mock_make:
            result = cb.on_step(step=1, algo=mock_algo)
        assert result is True
        mock_make.assert_not_called()

    def test_step_counter_uses_kwarg_when_provided(self):
        """When 'step' kwarg is passed, it should override the internal counter."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(record_freq=100)
        # Pass step=99 explicitly -- counter should be at 99, not trigger at 100 yet
        cb.on_step(step=99)
        assert cb._step_count == 99

    def test_step_counter_increments_without_kwarg(self):
        """When 'step' kwarg is absent, internal counter should increment."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(record_freq=10000)
        for _ in range(5):
            cb.on_step()
        assert cb._step_count == 5

    def test_on_step_triggers_recording_at_record_freq(self, tmp_path):
        """Recording should trigger exactly at the record_freq boundary."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(
            record_freq=100,
            n_episodes=1,
            video_folder=str(tmp_path / "vids"),
            verbose=False,
        )

        mock_algo = MagicMock()
        mock_algo.env_id = "CartPole-v1"
        mock_algo.vec_normalize = None
        mock_algo.predict = MagicMock(return_value=0)
        del mock_algo.vec_normalize  # ensure getattr returns None

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(4, dtype=np.float32), {})
        # step returns: obs, reward, terminated=True, truncated=False, info
        mock_env.step.return_value = (
            np.zeros(4, dtype=np.float32), 1.0, True, False, {}
        )

        with patch("gymnasium.make", return_value=mock_env), \
             patch("gymnasium.wrappers.RecordVideo", return_value=mock_env):
            # Steps 1–99: no recording
            for step in range(1, 100):
                cb.on_step(step=step, algo=mock_algo)

            # Step 100: recording should trigger
            import gymnasium as gym
            with patch.object(gym, "make", return_value=mock_env) as gm:
                # Reset mock call count
                gm.reset_mock()
                cb.on_step(step=100, algo=mock_algo)
                # gymnasium.make should have been called to create record env
                assert gm.called

    def test_video_recording_callback_is_callback_subclass(self):
        """VideoRecordingCallback must inherit from Callback."""
        from rlox.callbacks import Callback, VideoRecordingCallback

        assert issubclass(VideoRecordingCallback, Callback)

    def test_vec_normalize_restored_after_recording(self):
        """vec_normalize.training must be True after a recording episode."""
        from rlox.callbacks import VideoRecordingCallback

        cb = VideoRecordingCallback(record_freq=10, n_episodes=1, verbose=False)

        mock_vn = MagicMock()
        mock_vn.training = True
        mock_vn.normalize_obs = MagicMock(
            return_value=np.zeros((1, 4), dtype=np.float32)
        )

        mock_algo = MagicMock()
        mock_algo.env_id = "CartPole-v1"
        mock_algo.vec_normalize = mock_vn
        mock_algo.predict = MagicMock(return_value=0)

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(4, dtype=np.float32), {})
        mock_env.step.return_value = (
            np.zeros(4, dtype=np.float32), 1.0, True, False, {}
        )

        with patch("gymnasium.make", return_value=mock_env), \
             patch("gymnasium.wrappers.RecordVideo", return_value=mock_env):
            cb.on_step(step=10, algo=mock_algo)

        assert mock_vn.training is True


# ===========================================================================
# Episode stats tracking in RolloutCollector
# ===========================================================================


class TestRolloutCollectorEpisodeStats:
    """Tests for RolloutCollector.episode_rewards and episode_lengths."""

    def _make_collector(self, n_envs: int = 2):
        from rlox.collectors import RolloutCollector

        return RolloutCollector(
            env_id="CartPole-v1",
            n_envs=n_envs,
            seed=0,
            device="cpu",
            gamma=0.99,
            gae_lambda=0.95,
        )

    def test_episode_rewards_initially_empty(self):
        """Before any collect(), episode_rewards must be an empty list."""
        collector = self._make_collector()
        assert collector.episode_rewards == []

    def test_episode_lengths_initially_empty(self):
        """Before any collect(), episode_lengths must be an empty list."""
        collector = self._make_collector()
        assert collector.episode_lengths == []

    def test_episode_rewards_populated_after_collect(self):
        """After enough steps, at least one episode should complete."""
        from rlox.policies import DiscretePolicy

        collector = self._make_collector(n_envs=4)
        policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=16)

        # 200 steps * 4 envs = 800 transitions — CartPole episodes are short
        collector.collect(policy, n_steps=200)

        assert len(collector.episode_rewards) > 0, (
            "Expected at least one completed episode after 800 total transitions"
        )

    def test_episode_lengths_match_rewards_count(self):
        """episode_rewards and episode_lengths must have the same length."""
        from rlox.policies import DiscretePolicy

        collector = self._make_collector(n_envs=4)
        policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=16)
        collector.collect(policy, n_steps=200)

        assert len(collector.episode_rewards) == len(collector.episode_lengths)

    def test_episode_rewards_are_floats(self):
        """All entries in episode_rewards must be Python floats."""
        from rlox.policies import DiscretePolicy

        collector = self._make_collector(n_envs=4)
        policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=16)
        collector.collect(policy, n_steps=200)

        for r in collector.episode_rewards:
            assert isinstance(r, float), f"Expected float, got {type(r)}"

    def test_episode_lengths_are_ints(self):
        """All entries in episode_lengths must be Python ints."""
        from rlox.policies import DiscretePolicy

        collector = self._make_collector(n_envs=4)
        policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=16)
        collector.collect(policy, n_steps=200)

        for length in collector.episode_lengths:
            assert isinstance(length, int), f"Expected int, got {type(length)}"

    def test_episode_lengths_positive(self):
        """All episode lengths must be >= 1."""
        from rlox.policies import DiscretePolicy

        collector = self._make_collector(n_envs=4)
        policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=16)
        collector.collect(policy, n_steps=200)

        for length in collector.episode_lengths:
            assert length >= 1, f"Episode length must be >= 1, got {length}"

    def test_episode_stats_accumulate_across_collect_calls(self):
        """Multiple collect() calls should accumulate stats in the same lists."""
        from rlox.policies import DiscretePolicy

        collector = self._make_collector(n_envs=4)
        policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=16)

        collector.collect(policy, n_steps=100)
        count_after_first = len(collector.episode_rewards)

        collector.collect(policy, n_steps=100)
        count_after_second = len(collector.episode_rewards)

        assert count_after_second >= count_after_first, (
            "episode_rewards should accumulate across multiple collect() calls"
        )

    def test_episode_rewards_property_returns_list(self):
        """episode_rewards must be a list (not a numpy array or other type)."""
        collector = self._make_collector()
        assert isinstance(collector.episode_rewards, list)

    def test_episode_lengths_property_returns_list(self):
        """episode_lengths must be a list (not a numpy array or other type)."""
        collector = self._make_collector()
        assert isinstance(collector.episode_lengths, list)


# ===========================================================================
# RecordEpisodeStatistics auto-wrap in GymVecEnv
# ===========================================================================


class TestGymVecEnvEpisodeStats:
    """Tests for GymVecEnv.episode_rewards and episode_lengths."""

    def _make_env(self, n_envs: int = 4, record: bool = True) -> Any:
        from rlox.gym_vec_env import GymVecEnv

        return GymVecEnv("CartPole-v1", n_envs=n_envs, seed=42,
                         record_episode_stats=record)

    def test_episode_rewards_initially_empty(self):
        """episode_rewards is empty before any steps."""
        env = self._make_env()
        assert env.episode_rewards == []

    def test_episode_lengths_initially_empty(self):
        """episode_lengths is empty before any steps."""
        env = self._make_env()
        assert env.episode_lengths == []

    @pytest.mark.xfail(
        reason=(
            "GymVecEnv.step_all() looks for 'episode' in infos but gymnasium "
            "1.2+ with AutoresetMode.SAME_STEP places it under "
            "infos['final_info']['episode']. The property always returns [] "
            "until the info-parsing is updated."
        ),
        strict=True,
    )
    def test_episode_rewards_populated_after_episodes(self):
        """After enough steps, episode_rewards should contain completed episodes."""
        env = self._make_env(n_envs=4)
        env.reset_all()

        for _ in range(500):
            actions = np.array(
                [env.action_space.sample() for _ in range(4)], dtype=np.int64
            )
            env.step_all(actions)
            if env.episode_rewards:
                break

        assert len(env.episode_rewards) > 0, (
            "Expected at least one completed episode after 500 steps"
        )

    def test_episode_lengths_match_rewards(self):
        """episode_rewards and episode_lengths must have the same count."""
        env = self._make_env(n_envs=4)
        env.reset_all()

        for _ in range(500):
            actions = np.array(
                [env.action_space.sample() for _ in range(4)], dtype=np.int64
            )
            env.step_all(actions)

        assert len(env.episode_rewards) == len(env.episode_lengths)

    def test_episode_rewards_are_floats(self):
        """Rewards in episode_rewards should be Python floats."""
        env = self._make_env(n_envs=4)
        env.reset_all()

        for _ in range(500):
            actions = np.array(
                [env.action_space.sample() for _ in range(4)], dtype=np.int64
            )
            env.step_all(actions)

        for r in env.episode_rewards:
            assert isinstance(r, float), f"Expected float, got {type(r)}"

    def test_episode_lengths_are_ints(self):
        """Lengths in episode_lengths should be Python ints."""
        env = self._make_env(n_envs=4)
        env.reset_all()

        for _ in range(500):
            actions = np.array(
                [env.action_space.sample() for _ in range(4)], dtype=np.int64
            )
            env.step_all(actions)

        for length in env.episode_lengths:
            assert isinstance(length, int), f"Expected int, got {type(length)}"

    def test_no_stats_when_record_episode_stats_false(self):
        """When record_episode_stats=False, episode_rewards should stay empty."""
        env = self._make_env(n_envs=4, record=False)
        env.reset_all()

        for _ in range(500):
            actions = np.array(
                [env.action_space.sample() for _ in range(4)], dtype=np.int64
            )
            env.step_all(actions)

        assert env.episode_rewards == [], (
            "episode_rewards should remain empty when record_episode_stats=False"
        )

    def test_episode_stats_properties_are_lists(self):
        """episode_rewards and episode_lengths must be lists."""
        env = self._make_env()
        assert isinstance(env.episode_rewards, list)
        assert isinstance(env.episode_lengths, list)


# ===========================================================================
# Score normalization
# ===========================================================================


class TestNormalizeScore:
    """Tests for normalize_score()."""

    def test_random_score_normalizes_to_zero(self):
        """Normalizing the random-level score should give 0.0."""
        from rlox.evaluation import normalize_score, SCORE_BASELINES

        random_score, _ = SCORE_BASELINES["CartPole-v1"]
        result = normalize_score(random_score, "CartPole-v1")
        assert result == pytest.approx(0.0)

    def test_expert_score_normalizes_to_one(self):
        """Normalizing the expert-level score should give 1.0."""
        from rlox.evaluation import normalize_score, SCORE_BASELINES

        _, expert_score = SCORE_BASELINES["CartPole-v1"]
        result = normalize_score(expert_score, "CartPole-v1")
        assert result == pytest.approx(1.0)

    def test_midpoint_normalizes_to_half(self):
        """A score halfway between random and expert should give 0.5."""
        from rlox.evaluation import normalize_score, SCORE_BASELINES

        random_score, expert_score = SCORE_BASELINES["CartPole-v1"]
        midpoint = (random_score + expert_score) / 2.0
        result = normalize_score(midpoint, "CartPole-v1")
        assert result == pytest.approx(0.5)

    def test_above_expert_exceeds_one(self):
        """A score above expert level should normalize to > 1.0."""
        from rlox.evaluation import normalize_score, SCORE_BASELINES

        _, expert_score = SCORE_BASELINES["CartPole-v1"]
        result = normalize_score(expert_score * 1.1, "CartPole-v1")
        assert result > 1.0

    def test_unknown_env_raises_value_error(self):
        """normalize_score should raise ValueError for unknown env without overrides."""
        from rlox.evaluation import normalize_score

        with pytest.raises(ValueError, match="No baseline scores"):
            normalize_score(100.0, "UnknownEnv-v99")

    def test_unknown_env_with_overrides_works(self):
        """Providing random_score and expert_score bypasses the lookup table."""
        from rlox.evaluation import normalize_score

        result = normalize_score(50.0, "UnknownEnv-v99",
                                 random_score=0.0, expert_score=100.0)
        assert result == pytest.approx(0.5)

    def test_manual_overrides_take_precedence(self):
        """Explicit random/expert overrides should override the table."""
        from rlox.evaluation import normalize_score

        # Use custom overrides, not CartPole defaults (22.0, 500.0)
        result = normalize_score(50.0, "CartPole-v1",
                                 random_score=0.0, expert_score=100.0)
        assert result == pytest.approx(0.5)

    def test_degenerate_zero_denominator_returns_zero(self):
        """When random == expert, normalize_score should return 0.0, not divide by zero."""
        from rlox.evaluation import normalize_score

        result = normalize_score(42.0, "UnknownEnv-v99",
                                 random_score=100.0, expert_score=100.0)
        assert result == pytest.approx(0.0)

    def test_mujoco_env_lookup(self):
        """Spot-check that a MuJoCo env is in SCORE_BASELINES."""
        from rlox.evaluation import normalize_score

        # Hopper random baseline is 18.0, expert is 2500.0
        result = normalize_score(18.0, "Hopper-v4")
        assert result == pytest.approx(0.0)

    def test_negative_random_score_env(self):
        """Environments with negative random baselines (e.g. Pendulum) work correctly."""
        from rlox.evaluation import normalize_score, SCORE_BASELINES

        random_score, expert_score = SCORE_BASELINES["Pendulum-v1"]
        assert random_score < 0
        midpoint = (random_score + expert_score) / 2.0
        result = normalize_score(midpoint, "Pendulum-v1")
        assert result == pytest.approx(0.5)


class TestNormalizeScores:
    """Tests for normalize_scores() — the vectorized version."""

    def test_returns_numpy_array(self):
        """normalize_scores must return a numpy array."""
        from rlox.evaluation import normalize_scores

        result = normalize_scores([100.0, 200.0, 300.0], "CartPole-v1",
                                  random_score=0.0, expert_score=500.0)
        assert isinstance(result, np.ndarray)

    def test_dtype_is_float64(self):
        """Output array should have dtype float64."""
        from rlox.evaluation import normalize_scores

        result = normalize_scores([0.0, 250.0, 500.0], "CartPole-v1",
                                  random_score=0.0, expert_score=500.0)
        assert result.dtype == np.float64

    def test_shape_matches_input(self):
        """Output length should match input length."""
        from rlox.evaluation import normalize_scores

        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_scores(scores, "CartPole-v1",
                                  random_score=0.0, expert_score=10.0)
        assert result.shape == (len(scores),)

    def test_values_match_scalar_normalize(self):
        """Each element must equal the scalar normalize_score() result."""
        from rlox.evaluation import normalize_score, normalize_scores

        scores = [22.0, 100.0, 261.0, 500.0]
        arr = normalize_scores(scores, "CartPole-v1")
        for i, s in enumerate(scores):
            expected = normalize_score(s, "CartPole-v1")
            assert arr[i] == pytest.approx(expected)

    def test_accepts_numpy_array_input(self):
        """normalize_scores should accept a numpy array as input."""
        from rlox.evaluation import normalize_scores

        scores = np.array([0.0, 50.0, 100.0])
        result = normalize_scores(scores, "CartPole-v1",
                                  random_score=0.0, expert_score=100.0)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_empty_input_returns_empty_array(self):
        """An empty list should return an empty numpy array."""
        from rlox.evaluation import normalize_scores

        result = normalize_scores([], "CartPole-v1",
                                  random_score=0.0, expert_score=500.0)
        assert result.shape == (0,)


class TestScoreBaselines:
    """Tests for the SCORE_BASELINES constant."""

    def test_cartpole_present(self):
        """CartPole-v1 must be in SCORE_BASELINES."""
        from rlox.evaluation import SCORE_BASELINES

        assert "CartPole-v1" in SCORE_BASELINES

    def test_mujoco_envs_present(self):
        """Core MuJoCo environments must all be in SCORE_BASELINES."""
        from rlox.evaluation import SCORE_BASELINES

        mujoco_envs = [
            "HalfCheetah-v4", "Walker2d-v4", "Hopper-v4",
            "Ant-v4", "Humanoid-v4",
        ]
        for env in mujoco_envs:
            assert env in SCORE_BASELINES, f"{env} missing from SCORE_BASELINES"

    def test_baselines_are_tuples_of_two_floats(self):
        """Each entry must be a (random_score, expert_score) tuple of numbers."""
        from rlox.evaluation import SCORE_BASELINES

        for env_id, baseline in SCORE_BASELINES.items():
            assert len(baseline) == 2, f"{env_id}: expected tuple of length 2"
            random_score, expert_score = baseline
            assert isinstance(random_score, (int, float)), (
                f"{env_id}: random_score not numeric"
            )
            assert isinstance(expert_score, (int, float)), (
                f"{env_id}: expert_score not numeric"
            )

    def test_expert_differs_from_random(self):
        """Expert score and random score must differ (non-degenerate baselines)."""
        from rlox.evaluation import SCORE_BASELINES

        for env_id, (random_score, expert_score) in SCORE_BASELINES.items():
            assert random_score != expert_score, (
                f"{env_id}: random_score == expert_score (degenerate baseline)"
            )


# ===========================================================================
# AsymmetricPolicy
# ===========================================================================


class TestAsymmetricPolicyDiscrete:
    """Tests for AsymmetricPolicy with a discrete actor."""

    @pytest.fixture
    def policy(self):
        from rlox.policies import AsymmetricPolicy

        # Actor sees 4 dims, critic sees 6 dims (4 actor obs + 2 privileged)
        return AsymmetricPolicy(
            obs_dim=4,
            critic_obs_dim=6,
            n_actions=2,
            hidden=32,
        )

    def test_exactly_one_of_n_actions_or_act_dim_required(self):
        """Providing both or neither must raise ValueError."""
        from rlox.policies import AsymmetricPolicy

        with pytest.raises(ValueError, match="Exactly one"):
            AsymmetricPolicy(obs_dim=4, critic_obs_dim=6)

        with pytest.raises(ValueError, match="Exactly one"):
            AsymmetricPolicy(obs_dim=4, critic_obs_dim=6, n_actions=2, act_dim=1)

    def test_is_nn_module(self, policy):
        """AsymmetricPolicy must be a torch.nn.Module."""
        assert isinstance(policy, torch.nn.Module)

    def test_action_shape_discrete(self, policy):
        """get_action_and_logprob: actions shape == (batch,) for discrete."""
        obs = torch.randn(8, 6)
        actions, log_probs = policy.get_action_and_logprob(obs)
        assert actions.shape == (8,)
        assert log_probs.shape == (8,)

    def test_value_shape(self, policy):
        """get_value: output must be (batch,)."""
        obs = torch.randn(8, 6)
        values = policy.get_value(obs)
        assert values.shape == (8,)

    def test_get_action_value_shapes_discrete(self, policy):
        """get_action_value: all three tensors must have shape (batch,)."""
        obs = torch.randn(8, 6)
        actions, log_probs, values = policy.get_action_value(obs)
        assert actions.shape == (8,)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)

    def test_get_logprob_and_entropy_shapes_discrete(self, policy):
        """get_logprob_and_entropy: both outputs must be (batch,)."""
        obs = torch.randn(8, 6)
        actions = torch.randint(0, 2, (8,))
        log_probs, entropy = policy.get_logprob_and_entropy(obs, actions)
        assert log_probs.shape == (8,)
        assert entropy.shape == (8,)

    def test_actor_only_uses_obs_dim_columns(self, policy):
        """Critic columns beyond obs_dim must not affect the actor's logits."""
        base_obs = torch.randn(4, 6)

        # Modify the privileged columns (indices 4 and 5)
        modified_obs = base_obs.clone()
        modified_obs[:, 4:] = modified_obs[:, 4:] * 100.0

        with torch.no_grad():
            # Compare raw actor logits, not sampled log_probs (sampling is stochastic)
            logits_base = policy.actor(base_obs[:, :policy.obs_dim])
            logits_mod = policy.actor(modified_obs[:, :policy.obs_dim])

        # Logits must be identical since actor only receives the first obs_dim cols
        torch.testing.assert_close(logits_base, logits_mod)

    def test_critic_uses_full_obs(self, policy):
        """Changing privileged columns must change critic value estimate."""
        torch.manual_seed(9)
        base_obs = torch.randn(4, 6)
        modified_obs = base_obs.clone()
        modified_obs[:, 4:] = modified_obs[:, 4:] + 100.0  # large perturbation

        with torch.no_grad():
            v_base = policy.get_value(base_obs)
            v_mod = policy.get_value(modified_obs)

        assert not torch.allclose(v_base, v_mod), (
            "Critic should respond to changes in privileged observations"
        )

    def test_single_obs_batch(self, policy):
        """Works with batch_size=1."""
        obs = torch.randn(1, 6)
        actions, log_probs, values = policy.get_action_value(obs)
        assert actions.shape == (1,)
        assert log_probs.shape == (1,)
        assert values.shape == (1,)


class TestAsymmetricPolicyContinuous:
    """Tests for AsymmetricPolicy with a continuous actor."""

    @pytest.fixture
    def policy(self):
        from rlox.policies import AsymmetricPolicy

        return AsymmetricPolicy(
            obs_dim=3,
            critic_obs_dim=5,
            act_dim=2,
            hidden=32,
        )

    def test_action_shape_continuous(self, policy):
        """get_action_and_logprob: actions shape == (batch, act_dim)."""
        obs = torch.randn(8, 5)
        actions, log_probs = policy.get_action_and_logprob(obs)
        assert actions.shape == (8, 2)
        assert log_probs.shape == (8,)

    def test_get_action_value_shapes_continuous(self, policy):
        """get_action_value: actions (batch, act_dim), log_probs/values (batch,)."""
        obs = torch.randn(8, 5)
        actions, log_probs, values = policy.get_action_value(obs)
        assert actions.shape == (8, 2)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)

    def test_get_logprob_and_entropy_shapes_continuous(self, policy):
        """get_logprob_and_entropy: both outputs (batch,)."""
        obs = torch.randn(8, 5)
        actions = torch.randn(8, 2)
        log_probs, entropy = policy.get_logprob_and_entropy(obs, actions)
        assert log_probs.shape == (8,)
        assert entropy.shape == (8,)

    def test_log_std_parameter_present(self, policy):
        """Continuous policy must have a log_std nn.Parameter."""
        assert hasattr(policy, "log_std")
        assert isinstance(policy.log_std, torch.nn.Parameter)
        assert policy.log_std.shape == (2,)

    def test_no_log_std_for_discrete(self):
        """Discrete AsymmetricPolicy must not have a log_std attribute."""
        from rlox.policies import AsymmetricPolicy

        policy = AsymmetricPolicy(obs_dim=4, critic_obs_dim=6, n_actions=2, hidden=16)
        assert not hasattr(policy, "log_std")

    def test_actor_ignores_privileged_columns(self, policy):
        """Continuous actor only uses the first obs_dim columns."""
        base_obs = torch.randn(4, 5)
        modified_obs = base_obs.clone()
        modified_obs[:, 3:] = modified_obs[:, 3:] * 50.0  # change privileged cols

        with torch.no_grad():
            # Compare actor mean output directly — sampling is stochastic so
            # log_probs from get_action_and_logprob would differ due to noise.
            mean_base = policy.actor(base_obs[:, :policy.obs_dim])
            mean_mod = policy.actor(modified_obs[:, :policy.obs_dim])

        torch.testing.assert_close(mean_base, mean_mod)

    def test_gradient_flows_through_actor(self, policy):
        """Gradient must flow through the actor branch."""
        obs = torch.randn(4, 5)
        _, log_probs = policy.get_action_and_logprob(obs)
        (-log_probs.mean()).backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.actor.parameters()
        )
        assert has_grad, "No gradient flowed through actor parameters"

    def test_gradient_flows_through_critic(self, policy):
        """Gradient must flow through the critic branch."""
        obs = torch.randn(4, 5)
        values = policy.get_value(obs)
        values.mean().backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.critic.parameters()
        )
        assert has_grad, "No gradient flowed through critic parameters"


# ===========================================================================
# CI bands on learning curves (_compute_learning_curve_ci)
# ===========================================================================


class TestComputeLearningCurveCI:
    """Tests for _compute_learning_curve_ci in multi_seed_runner."""

    def _make_results(self, n_seeds: int = 3, steps: list[int] | None = None):
        """Build synthetic seed results dicts with learning curves."""
        if steps is None:
            steps = [10000, 20000, 30000]
        rng = np.random.default_rng(0)
        results = []
        for seed_i in range(n_seeds):
            curve = [
                {"step": s, "reward": float(rng.uniform(100, 400))}
                for s in steps
            ]
            results.append({"seed": seed_i, "mean_reward": 200.0, "learning_curve": curve})
        return results

    def test_returns_list(self):
        """_compute_learning_curve_ci must return a list."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        results = self._make_results()
        ci = _compute_learning_curve_ci(results, n_bootstrap=100)
        assert isinstance(ci, list)

    def test_length_matches_common_steps(self):
        """Output length must equal the number of common checkpoint steps."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        steps = [10000, 20000, 30000, 40000]
        results = self._make_results(n_seeds=3, steps=steps)
        ci = _compute_learning_curve_ci(results, n_bootstrap=100)
        assert len(ci) == len(steps)

    def test_output_keys(self):
        """Each dict in the output must have step, mean, ci_low, ci_high."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        results = self._make_results()
        ci = _compute_learning_curve_ci(results, n_bootstrap=200)

        for entry in ci:
            assert "step" in entry
            assert "mean" in entry
            assert "ci_low" in entry
            assert "ci_high" in entry

    def test_ci_ordering(self):
        """ci_low <= mean <= ci_high must hold at every checkpoint."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        results = self._make_results(n_seeds=5)
        ci = _compute_learning_curve_ci(results, n_bootstrap=500)

        for entry in ci:
            assert entry["ci_low"] <= entry["mean"], (
                f"ci_low={entry['ci_low']} > mean={entry['mean']} at step {entry['step']}"
            )
            assert entry["mean"] <= entry["ci_high"], (
                f"mean={entry['mean']} > ci_high={entry['ci_high']} at step {entry['step']}"
            )

    def test_steps_are_sorted_ascending(self):
        """Steps in the output must be in ascending order."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        results = self._make_results(n_seeds=3, steps=[30000, 10000, 20000])
        ci = _compute_learning_curve_ci(results, n_bootstrap=100)
        steps = [entry["step"] for entry in ci]
        assert steps == sorted(steps)

    def test_steps_are_int(self):
        """step values must be Python ints, not floats."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        results = self._make_results()
        ci = _compute_learning_curve_ci(results, n_bootstrap=100)
        for entry in ci:
            assert isinstance(entry["step"], int), (
                f"step should be int, got {type(entry['step'])}"
            )

    def test_empty_results_returns_empty(self):
        """No results → empty output."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        ci = _compute_learning_curve_ci([])
        assert ci == []

    def test_results_without_learning_curve_returns_empty(self):
        """Results with no 'learning_curve' key → empty output."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        results = [{"seed": 0, "mean_reward": 200.0}]
        ci = _compute_learning_curve_ci(results)
        assert ci == []

    def test_mismatched_steps_uses_intersection(self):
        """When seeds have different checkpoint steps, only common steps are returned."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        results = [
            {"learning_curve": [{"step": 1000, "reward": 100.0},
                                  {"step": 2000, "reward": 200.0}]},
            {"learning_curve": [{"step": 2000, "reward": 150.0},
                                  {"step": 3000, "reward": 250.0}]},
        ]
        ci = _compute_learning_curve_ci(results, n_bootstrap=100)
        # Only step=2000 is common to both seeds
        assert len(ci) == 1
        assert ci[0]["step"] == 2000

    def test_wider_ci_for_more_variable_data(self):
        """More variable reward data should produce a wider confidence interval."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        # Low variance: all seeds same reward
        low_var = [
            {"learning_curve": [{"step": 1000, "reward": 200.0}]}
            for _ in range(10)
        ]
        # High variance: seeds span a wide range
        high_var = [
            {"learning_curve": [{"step": 1000, "reward": float(r)}]}
            for r in np.linspace(0, 1000, 10)
        ]

        ci_low_var = _compute_learning_curve_ci(low_var, n_bootstrap=500)
        ci_high_var = _compute_learning_curve_ci(high_var, n_bootstrap=500)

        width_low = ci_low_var[0]["ci_high"] - ci_low_var[0]["ci_low"]
        width_high = ci_high_var[0]["ci_high"] - ci_high_var[0]["ci_low"]

        assert width_high > width_low, (
            "High variance data should produce a wider CI band"
        )

    def test_single_seed_ci_is_tight(self):
        """With a single seed, bootstrap CI should be very narrow (zero variance)."""
        from benchmarks.multi_seed_runner import _compute_learning_curve_ci

        results = [{"learning_curve": [{"step": 1000, "reward": 300.0}]}]
        ci = _compute_learning_curve_ci(results, n_bootstrap=200)
        entry = ci[0]
        width = entry["ci_high"] - entry["ci_low"]
        # With one seed, all bootstrap resamples are the same value
        assert width == pytest.approx(0.0), (
            f"Expected zero-width CI for single seed, got width={width}"
        )
