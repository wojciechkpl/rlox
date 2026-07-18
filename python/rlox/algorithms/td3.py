"""Twin Delayed DDPG (TD3)."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.logging import LoggerCallback
from rlox.networks import QNetwork, DeterministicPolicy, polyak_update


class TD3:
    """Twin Delayed DDPG.

    Deterministic policy, target policy smoothing, delayed policy updates.
    Uses rlox.ReplayBuffer for storage.
    """

    def __init__(
        self,
        env_id: str,
        buffer_size: int = 1_000_000,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        learning_starts: int = 1000,
        hidden: int = 256,
        seed: int = 42,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        train_freq: int = 1,
        gradient_steps: int = 1,
        # SB3 name aliases — accepted so preset YAMLs don't crash.
        target_policy_noise: float | None = None,
        target_noise_clip: float | None = None,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
        compile: bool = False,
        actor: nn.Module | None = None,
        critic: nn.Module | None = None,
        buffer: object | None = None,
        collector: object | None = None,
        n_envs: int = 1,
    ):
        if isinstance(env_id, str):
            self.env = gym.make(env_id)
            self.env_id = env_id
        else:
            self.env = env_id
            self.env_id = (
                getattr(env_id.spec, "id", "custom")
                if hasattr(env_id, "spec") and env_id.spec
                else "custom"
            )
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_delay = policy_delay
        # Accept SB3 name aliases from preset YAMLs.
        self.target_noise = target_policy_noise if target_policy_noise is not None else target_noise
        self.noise_clip = target_noise_clip if target_noise_clip is not None else noise_clip
        self.exploration_noise = exploration_noise
        self.train_freq = max(1, int(train_freq))
        self.gradient_steps = max(1, int(gradient_steps))
        self.learning_rate = learning_rate
        self.hidden = hidden
        self.buffer_size = buffer_size

        # Seed torch/np/env RNG for reproducibility.  This MUST happen
        # before the actor/critic networks are constructed below so that
        # their weight initialisation is deterministic for a given seed
        # (precedent: pqn.py:118, crossq.py:141).  Gymnasium does not
        # propagate `env.reset(seed=...)` to the action space's RNG, and
        # the `learning_starts` exploration phase in `train()` calls
        # `action_space.sample()`, so the action space needs its own
        # `.seed()` call here too.
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.action_space.seed(seed)

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        act_high = float(self.env.action_space.high[0])

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_high = act_high

        # Networks — use custom if provided, otherwise default MLP
        self.actor = (
            actor
            if actor is not None
            else DeterministicPolicy(obs_dim, act_dim, hidden, act_high)
        )
        self.actor_target = copy.deepcopy(self.actor)
        if critic is not None:
            self.critic1 = critic
            self.critic2 = copy.deepcopy(critic)
        else:
            self.critic1 = QNetwork(obs_dim, act_dim, hidden)
            self.critic2 = QNetwork(obs_dim, act_dim, hidden)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=learning_rate
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=learning_rate
        )

        # Replay buffer — use custom if provided
        self.buffer = (
            buffer
            if buffer is not None
            else rlox.ReplayBuffer(buffer_size, obs_dim, act_dim)
        )

        # Off-policy collector — enables multi-env collection
        self.collector = collector
        self.n_envs = n_envs

        # Callbacks and logger
        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

        if compile:
            from rlox.compile import compile_policy

            compile_policy(self)

    def _get_config_dict(self) -> dict[str, Any]:
        """Return a serialisable config dict for checkpointing."""
        return {
            "buffer_size": self.buffer_size,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "learning_starts": self.learning_starts,
            "hidden": self.hidden,
            "policy_delay": self.policy_delay,
            "target_noise": self.target_noise,
            "noise_clip": self.noise_clip,
            "exploration_noise": self.exploration_noise,
        }

    def train(self, total_timesteps: int) -> dict[str, float]:
        # Use OffPolicyCollector for multi-env, or default single-env loop
        if self.n_envs > 1 and self.collector is None:
            from rlox.off_policy_collector import OffPolicyCollector

            self.collector = OffPolicyCollector(
                env_id=self.env_id,
                n_envs=self.n_envs,
                buffer=self.buffer,
                act_high=self.act_high,
                seed=getattr(self, "seed", 42),
            )

        if self.collector is not None:
            return self._train_with_collector(total_timesteps)

        obs, _ = self.env.reset(seed=self.seed)
        episode_rewards: list[float] = []
        ep_reward = 0.0
        metrics: dict[str, float] = {}
        update_count = 0

        self.callbacks.on_training_start()

        for step in range(total_timesteps):
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = self.actor(obs_t).squeeze(0).numpy()
                    noise = np.random.randn(self.act_dim) * self.exploration_noise
                    action = np.clip(action + noise, -self.act_high, self.act_high)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            ep_reward += float(reward)

            self.buffer.push(
                np.asarray(obs, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                bool(terminated),
                bool(truncated),
                np.asarray(next_obs, dtype=np.float32),
            )

            obs = next_obs
            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

            # Callback: on_step
            self._global_step += 1
            should_continue = self.callbacks.on_step(
                reward=ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            # Update — only every ``train_freq`` env steps, doing
            # ``gradient_steps`` SGD steps per round (mirrors SB3).
            if (
                step >= self.learning_starts
                and len(self.buffer) >= self.batch_size
                and step % self.train_freq == 0
            ):
                for _ in range(self.gradient_steps):
                    update_count += 1
                    metrics = self._update(step, update_count)
                    self.callbacks.on_train_batch(**metrics)

                # Logger
                if self.logger is not None and self._global_step % 1000 == 0:
                    self.logger.on_train_step(self._global_step, metrics)

        self.callbacks.on_training_end()

        metrics["mean_reward"] = (
            float(np.mean(episode_rewards)) if episode_rewards else 0.0
        )
        return metrics

    def _update(self, step: int, update_count: int) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size, step)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32)

        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)

        with torch.no_grad():
            # Target policy smoothing
            noise = torch.randn_like(actions) * self.target_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(
                -self.act_high, self.act_high
            )

            q1_next = self.critic1_target(next_obs, next_actions).squeeze(-1)
            q2_next = self.critic2_target(next_obs, next_actions).squeeze(-1)
            target_q = rewards + self.gamma * (1.0 - terminated) * torch.min(
                q1_next, q2_next
            )

        # Critic updates
        q1 = self.critic1(obs, actions).squeeze(-1)
        q2 = self.critic2(obs, actions).squeeze(-1)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        critic_loss = critic1_loss + critic2_loss
        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Every step: update critic targets
        polyak_update(self.critic1, self.critic1_target, self.tau)
        polyak_update(self.critic2, self.critic2_target, self.tau)

        actor_loss_val = 0.0

        # Delayed policy update
        if update_count % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            polyak_update(self.actor, self.actor_target, self.tau)

        return {
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "actor_loss": actor_loss_val,
        }

    def _train_with_collector(self, total_timesteps: int) -> dict[str, float]:
        """Train using OffPolicyCollector for multi-env data collection."""
        collector = self.collector
        collector.reset()

        metrics: dict[str, float] = {}
        update_count = 0

        self.callbacks.on_training_start()

        def get_action(obs_batch: np.ndarray) -> np.ndarray:
            if self._global_step < self.learning_starts:
                return (
                    np.random.randn(obs_batch.shape[0], self.act_dim) * self.act_high
                ).astype(np.float32)
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
                actions = self.actor(obs_t).numpy()
                noise = np.random.randn(*actions.shape) * self.exploration_noise
                return np.clip(actions + noise, -self.act_high, self.act_high).astype(
                    np.float32
                )

        for step in range(total_timesteps):
            _, _, mean_ep_reward = collector.collect_step(
                get_action, step, total_timesteps
            )

            self._global_step += 1
            should_continue = self.callbacks.on_step(
                reward=mean_ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            if (
                step >= self.learning_starts
                and len(self.buffer) >= self.batch_size
                and step % self.train_freq == 0
            ):
                for _ in range(self.gradient_steps):
                    update_count += 1
                    metrics = self._update(step, update_count)
                    self.callbacks.on_train_batch(**metrics)

                if self.logger is not None and self._global_step % 1000 == 0:
                    self.logger.on_train_step(self._global_step, metrics)

        self.callbacks.on_training_end()

        completed = collector._completed_rewards
        metrics["mean_reward"] = float(np.mean(completed)) if completed else 0.0
        return metrics

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from the policy (always deterministic for TD3)."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.actor(obs_t).squeeze(0).numpy()

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        data: dict[str, Any] = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "step": self._global_step,
            "config": self._get_config_dict(),
            "env_id": self.env_id,
            "torch_rng_state": torch.random.get_rng_state(),
        }
        torch.save(data, path)

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> TD3:
        """Restore TD3 from a checkpoint."""
        from rlox.checkpoint import safe_torch_load

        data = safe_torch_load(path)
        config = data["config"]
        eid = env_id or data.get("env_id", "Pendulum-v1")

        td3 = cls(env_id=eid, **config)
        td3.actor.load_state_dict(data["actor_state_dict"])
        td3.actor_target.load_state_dict(data["actor_target_state_dict"])
        td3.critic1.load_state_dict(data["critic1_state_dict"])
        td3.critic2.load_state_dict(data["critic2_state_dict"])
        td3.critic1_target.load_state_dict(data["critic1_target_state_dict"])
        td3.critic2_target.load_state_dict(data["critic2_target_state_dict"])
        td3.actor_optimizer.load_state_dict(data["actor_optimizer_state_dict"])
        td3.critic1_optimizer.load_state_dict(data["critic1_optimizer_state_dict"])
        td3.critic2_optimizer.load_state_dict(data["critic2_optimizer_state_dict"])
        td3._global_step = data.get("step", 0)

        if "torch_rng_state" in data:
            torch.random.set_rng_state(data["torch_rng_state"])

        return td3
