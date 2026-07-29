"""Convergence correctness tests.

These tests train full agents and verify that they reach minimum reward
thresholds.  They are SLOW (minutes each) and are only run when explicitly
selected with:

    pytest -m convergence

or

    pytest -m slow

Test inventory
--------------
PPO on CartPole-v1   >= 400 within 50K steps  (rlox)
PPO on CartPole-v1   >= 400 within 50K steps  (SB3)
rlox PPO and SB3 PPO statistically similar final rewards
A2C on CartPole-v1   >= 350 within 50K steps  (rlox)
DQN on CartPole-v1   >= 400 within 100K steps (rlox)
SAC on Pendulum-v1   >= -300 within 20K steps (rlox)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from utils import bootstrap_ci, get_system_info, save_results


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------


def evaluate_policy(get_action_fn, env_id: str, n_episodes: int = 20, seed: int = 1000):
    """Evaluate *get_action_fn* for *n_episodes* in a fresh gymnasium env.

    Parameters
    ----------
    get_action_fn:
        Callable that takes a numpy observation and returns an action.
    env_id:
        Gymnasium environment ID.
    n_episodes:
        Number of evaluation episodes.
    seed:
        Evaluation seed base.

    Returns
    -------
    (mean_return, std_return, episode_returns)
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise RuntimeError("gymnasium must be installed for convergence tests")

    env = gym.make(env_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            action = get_action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    return mean_r, std_r, returns


# ---------------------------------------------------------------------------
# PPO on CartPole (rlox)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.convergence
def test_ppo_cartpole_converges_rlox(results_dir):
    """rlox PPO must reach mean eval reward >= 400 on CartPole within 50K steps."""
    try:
        from rlox.algorithms.ppo import PPO
        import torch
    except ImportError:
        pytest.skip("rlox.algorithms.ppo not available")

    torch.manual_seed(1)
    np.random.seed(1)

    agent = PPO(
        env_id="CartPole-v1",
        n_envs=4,
        seed=1,
        n_steps=128,
        n_epochs=4,
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    t0 = time.perf_counter()
    metrics = agent.train(total_timesteps=50_000)
    elapsed = time.perf_counter() - t0

    def get_action(obs):
        import torch as th
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            action, _ = agent.policy.get_action_and_logprob(obs_t)
            return int(action.item())

    mean_r, std_r, returns = evaluate_policy(get_action, "CartPole-v1")

    record = {
        "algorithm": "PPO",
        "framework": "rlox",
        "env": "CartPole-v1",
        "total_timesteps": 50_000,
        "reward_threshold": 400.0,
        "mean_eval_reward": mean_r,
        "std_eval_reward": std_r,
        "episode_rewards": returns,
        "elapsed_s": elapsed,
        "passed": mean_r >= 400.0,
        "system": get_system_info(),
    }
    save_results(record, results_dir / "convergence" / "ppo_cartpole_rlox.json")

    assert mean_r >= 400.0, (
        f"rlox PPO on CartPole: mean eval reward {mean_r:.1f} < threshold 400.0"
    )


# ---------------------------------------------------------------------------
# PPO on CartPole (SB3)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.convergence
def test_ppo_cartpole_converges_sb3(results_dir):
    """SB3 PPO must reach mean eval reward >= 400 on CartPole within 50K steps."""
    try:
        from stable_baselines3 import PPO as SB3PPO
        import gymnasium as gym
    except ImportError:
        pytest.skip("stable-baselines3 not installed")

    import torch
    torch.manual_seed(1)

    model = SB3PPO(
        "MlpPolicy",
        "CartPole-v1",
        n_steps=128,
        n_epochs=4,
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=1,
        verbose=0,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=50_000)
    elapsed = time.perf_counter() - t0

    def get_action(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    mean_r, std_r, returns = evaluate_policy(get_action, "CartPole-v1")

    record = {
        "algorithm": "PPO",
        "framework": "sb3",
        "env": "CartPole-v1",
        "total_timesteps": 50_000,
        "reward_threshold": 400.0,
        "mean_eval_reward": mean_r,
        "std_eval_reward": std_r,
        "episode_rewards": returns,
        "elapsed_s": elapsed,
        "passed": mean_r >= 400.0,
        "system": get_system_info(),
    }
    save_results(record, results_dir / "convergence" / "ppo_cartpole_sb3.json")

    assert mean_r >= 400.0, (
        f"SB3 PPO on CartPole: mean eval reward {mean_r:.1f} < threshold 400.0"
    )


# ---------------------------------------------------------------------------
# Statistical parity: rlox PPO vs SB3 PPO
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.convergence
def test_ppo_rlox_and_sb3_statistically_similar(results_dir):
    """rlox and SB3 PPO final rewards must have overlapping 95% bootstrap CIs."""
    try:
        from rlox.algorithms.ppo import PPO as RloxPPO
        from stable_baselines3 import PPO as SB3PPO
        import torch
    except ImportError:
        pytest.skip("rlox.algorithms.ppo or stable-baselines3 not available")

    torch.manual_seed(42)
    np.random.seed(42)

    # Train rlox PPO
    rlox_agent = RloxPPO(
        env_id="CartPole-v1", n_envs=4, seed=42, n_steps=128,
        n_epochs=4, batch_size=64, learning_rate=2.5e-4,
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
    )
    rlox_agent.train(total_timesteps=50_000)

    def rlox_action(obs):
        import torch as th
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            a, _ = rlox_agent.policy.get_action_and_logprob(obs_t)
            return int(a.item())

    _, _, rlox_returns = evaluate_policy(rlox_action, "CartPole-v1", n_episodes=50)

    # Train SB3 PPO
    sb3_agent = SB3PPO(
        "MlpPolicy", "CartPole-v1",
        n_steps=128, n_epochs=4, batch_size=64, learning_rate=2.5e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        seed=42, verbose=0,
    )
    sb3_agent.learn(total_timesteps=50_000)

    def sb3_action(obs):
        a, _ = sb3_agent.predict(obs, deterministic=True)
        return int(a)

    _, _, sb3_returns = evaluate_policy(sb3_action, "CartPole-v1", n_episodes=50)

    # Bootstrap CIs
    rlox_lo, rlox_hi = bootstrap_ci(rlox_returns, n_resamples=10_000)
    sb3_lo, sb3_hi = bootstrap_ci(sb3_returns, n_resamples=10_000)

    # CIs must overlap
    overlap = (rlox_lo <= sb3_hi) and (sb3_lo <= rlox_hi)

    record = {
        "test": "ppo_rlox_vs_sb3_parity",
        "rlox_mean": float(np.mean(rlox_returns)),
        "sb3_mean": float(np.mean(sb3_returns)),
        "rlox_ci_95": [rlox_lo, rlox_hi],
        "sb3_ci_95": [sb3_lo, sb3_hi],
        "cis_overlap": overlap,
        "rlox_returns": rlox_returns,
        "sb3_returns": sb3_returns,
    }
    save_results(record, results_dir / "convergence" / "ppo_rlox_vs_sb3.json")

    assert overlap, (
        f"rlox PPO CI=[{rlox_lo:.1f},{rlox_hi:.1f}] and "
        f"SB3 PPO CI=[{sb3_lo:.1f},{sb3_hi:.1f}] do not overlap"
    )


# ---------------------------------------------------------------------------
# A2C on CartPole (rlox)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.convergence
def test_a2c_cartpole_converges_rlox(results_dir):
    """rlox A2C must reach mean eval reward >= 350 on CartPole within 50K steps."""
    try:
        from rlox.algorithms.a2c import A2C
        import torch
    except ImportError:
        pytest.skip("rlox.algorithms.a2c not available")

    torch.manual_seed(1)
    np.random.seed(1)

    agent = A2C(
        env_id="CartPole-v1",
        n_envs=4,
        seed=1,
        n_steps=5,
        learning_rate=7e-4,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
    )

    t0 = time.perf_counter()
    agent.train(total_timesteps=50_000)
    elapsed = time.perf_counter() - t0

    def get_action(obs):
        import torch as th
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            action, _ = agent.policy.get_action_and_logprob(obs_t)
            return int(action.item())

    mean_r, std_r, returns = evaluate_policy(get_action, "CartPole-v1")

    record = {
        "algorithm": "A2C",
        "framework": "rlox",
        "env": "CartPole-v1",
        "total_timesteps": 50_000,
        "reward_threshold": 350.0,
        "mean_eval_reward": mean_r,
        "std_eval_reward": std_r,
        "elapsed_s": elapsed,
        "passed": mean_r >= 350.0,
    }
    save_results(record, results_dir / "convergence" / "a2c_cartpole_rlox.json")

    assert mean_r >= 350.0, (
        f"rlox A2C on CartPole: mean eval reward {mean_r:.1f} < threshold 350.0"
    )


# ---------------------------------------------------------------------------
# DQN on CartPole (rlox)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.convergence
def test_dqn_cartpole_converges_rlox(results_dir):
    """rlox DQN must reach mean eval reward >= 400 on CartPole within 100K steps."""
    try:
        from rlox.algorithms.dqn import DQN
        import torch
    except ImportError:
        pytest.skip("rlox.algorithms.dqn not available")

    torch.manual_seed(1)
    np.random.seed(1)

    agent = DQN(
        env_id="CartPole-v1",
        buffer_size=50_000,
        learning_rate=1e-3,
        batch_size=64,
        gamma=0.99,
        target_update_freq=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=1000,
        double_dqn=True,
        dueling=False,
        n_step=1,
        hidden=128,
        seed=1,
    )

    # Manual training loop (mirrors verify_convergence.py)
    import gymnasium as gym
    import torch as th

    obs, _ = agent.env.reset(seed=1)
    total_timesteps = 100_000
    t0 = time.perf_counter()

    for step in range(total_timesteps):
        eps = agent._get_epsilon(step, total_timesteps)
        if np.random.random() < eps or step < agent.learning_starts:
            action = int(agent.env.action_space.sample())
        else:
            with th.no_grad():
                obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
                action = int(agent.q_network(obs_t).argmax(dim=-1).item())

        next_obs, reward, terminated, truncated, _ = agent.env.step(action)
        agent._store_transition(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs

        if terminated or truncated:
            # Flush n-step buffer
            while agent._n_step_buffer:
                R = 0.0
                for i in reversed(range(len(agent._n_step_buffer))):
                    _, _, r, _, d, tr = agent._n_step_buffer[i]
                    R = r + agent.gamma * R * (1.0 - float(d or tr))
                fo, fa, _, _, _, _ = agent._n_step_buffer[0]
                _, _, _, lo, ld, ltr = agent._n_step_buffer[-1]
                agent.buffer.push(
                    np.asarray(fo, dtype=np.float32),
                    np.array([float(fa)], dtype=np.float32),
                    float(R), bool(ld), bool(ltr),
                    np.asarray(lo, dtype=np.float32),
                )
                agent._n_step_buffer.pop(0)
            obs, _ = agent.env.reset()

        if step >= agent.learning_starts and len(agent.buffer) >= agent.batch_size:
            agent._update(step, total_timesteps)
        if step % agent.target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

    elapsed = time.perf_counter() - t0

    def get_action(obs):
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            return int(agent.q_network(obs_t).argmax(dim=-1).item())

    mean_r, std_r, returns = evaluate_policy(get_action, "CartPole-v1")

    record = {
        "algorithm": "DQN",
        "framework": "rlox",
        "env": "CartPole-v1",
        "total_timesteps": total_timesteps,
        "reward_threshold": 400.0,
        "mean_eval_reward": mean_r,
        "std_eval_reward": std_r,
        "elapsed_s": elapsed,
        "passed": mean_r >= 400.0,
    }
    save_results(record, results_dir / "convergence" / "dqn_cartpole_rlox.json")

    assert mean_r >= 400.0, (
        f"rlox DQN on CartPole: mean eval reward {mean_r:.1f} < threshold 400.0"
    )


# ---------------------------------------------------------------------------
# SAC on Pendulum (rlox)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.convergence
def test_sac_pendulum_converges_rlox(results_dir):
    """rlox SAC must reach mean eval reward >= -300 on Pendulum within 20K steps."""
    try:
        from rlox.algorithms.sac import SAC
        import torch as th
    except ImportError:
        pytest.skip("rlox.algorithms.sac not available")

    th.manual_seed(1)
    np.random.seed(1)

    agent = SAC(
        env_id="Pendulum-v1",
        buffer_size=50_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1000,
        hidden=256,
        seed=1,
        auto_entropy=True,
    )

    obs, _ = agent.env.reset(seed=1)
    total_timesteps = 20_000
    t0 = time.perf_counter()

    for step in range(total_timesteps):
        if step < agent.learning_starts:
            action = agent.env.action_space.sample()
        else:
            with th.no_grad():
                obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
                action_t, _ = agent.actor.sample(obs_t)
                action = (action_t.squeeze(0).numpy() * agent.act_high)

        next_obs, reward, terminated, truncated, _ = agent.env.step(action)
        agent.buffer.push(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward), bool(terminated), bool(truncated),
            np.asarray(next_obs, dtype=np.float32),
        )
        obs = next_obs
        if terminated or truncated:
            obs, _ = agent.env.reset()

        if step >= agent.learning_starts and len(agent.buffer) >= agent.batch_size:
            agent._update(step)

    elapsed = time.perf_counter() - t0

    def get_action(obs):
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            action = agent.actor.deterministic(obs_t).squeeze(0).numpy()
            return action * agent.act_high

    mean_r, std_r, returns = evaluate_policy(get_action, "Pendulum-v1")

    record = {
        "algorithm": "SAC",
        "framework": "rlox",
        "env": "Pendulum-v1",
        "total_timesteps": total_timesteps,
        "reward_threshold": -300.0,
        "mean_eval_reward": mean_r,
        "std_eval_reward": std_r,
        "elapsed_s": elapsed,
        "passed": mean_r >= -300.0,
    }
    save_results(record, results_dir / "convergence" / "sac_pendulum_rlox.json")

    assert mean_r >= -300.0, (
        f"rlox SAC on Pendulum: mean eval reward {mean_r:.1f} < threshold -300.0"
    )
