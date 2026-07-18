"""Advantage Weighted Regression (AWR) algorithm.

AWR (Peng et al., 2019) is an off-policy actor-critic that avoids
importance sampling by weighting log-probabilities with exponentiated
advantages: L_policy = -E[exp(A / beta) * log pi(a|s)].

Uses the standard ReplayBuffer and a simple actor-critic architecture.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar
    Self = TypeVar("Self")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.logging import LoggerCallback
from rlox.trainer import register_algorithm


@register_algorithm("awr")
class AWR:
    """Advantage Weighted Regression.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    beta : float
        Temperature for advantage weighting (default 1.0).
    learning_rate : float
        Learning rate for actor and critic (default 3e-4).
    gamma : float
        Discount factor (default 0.99).
    batch_size : int
        Minibatch size for SGD (default 256).
    buffer_size : int
        Replay buffer capacity (default 100_000).
    hidden : int
        Hidden layer size (default 256).
    learning_starts : int
        Number of random steps before training (default 1000).
    n_critic_updates : int
        Number of critic updates per training step (default 1).
    max_advantage : float
        Clip exponentiated advantage to prevent overflow (default 20.0).
    seed : int
        Random seed (default 42).
    callbacks : list[Callback], optional
        Training callbacks.
    logger : LoggerCallback, optional
        Logger for metrics.
    """

    def __init__(
        self,
        env_id: str,
        beta: float = 1.0,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        hidden: int = 256,
        learning_starts: int = 1000,
        n_critic_updates: int = 1,
        max_advantage: float = 20.0,
        seed: int = 42,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ) -> None:
        if isinstance(env_id, str):
            self.env = gym.make(env_id)
            self.env_id = env_id
        else:
            self.env = env_id
            self.env_id = getattr(getattr(env_id, "spec", None), "id", "custom")

        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.n_critic_updates = n_critic_updates
        self.max_advantage = max_advantage
        self.seed = seed

        # Seed torch/np/env RNG for reproducibility.  This MUST happen
        # before the actor/critic networks are constructed below so that
        # their weight initialisation is deterministic for a given seed
        # (precedent: pqn.py:118, crossq.py:141).  Gymnasium does not
        # propagate `env.reset(seed=...)` to the action space's RNG, and
        # the `learning_starts` exploration phase in `train()` calls
        # `action_space.sample()`, so the action space needs its own
        # `.seed()` call here too.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.action_space.seed(seed)

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_space = self.env.action_space
        self.obs_dim = obs_dim
        self.discrete = isinstance(act_space, gym.spaces.Discrete)

        if self.discrete:
            self.n_actions = int(act_space.n)
            self.act_dim = 1
        else:
            self.n_actions = 0
            self.act_dim = int(np.prod(act_space.shape))

        # Actor: outputs action distribution
        if self.discrete:
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.n_actions),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.act_dim),
            )

        # Critic: state value function
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.buffer = rlox.ReplayBuffer(buffer_size, obs_dim, self.act_dim)

        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0
        self._learning_rate = learning_rate
        self._hidden = hidden
        self._buffer_size = buffer_size

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run AWR training loop.

        Parameters
        ----------
        total_timesteps : int
            Number of environment steps.

        Returns
        -------
        metrics : dict with at least 'mean_reward'
        """
        obs, _ = self.env.reset(seed=self.seed)
        episode_rewards: list[float] = []
        ep_reward = 0.0
        metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for step in range(total_timesteps):
            # Collect experience
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    if self.discrete:
                        logits = self.actor(obs_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()
                    else:
                        action = self.actor(obs_t).squeeze(0).numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            ep_reward += float(reward)

            act_arr = np.array([action], dtype=np.float32) if self.discrete else np.asarray(action, dtype=np.float32)
            self.buffer.push(
                np.asarray(obs, dtype=np.float32),
                act_arr,
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

            self._global_step += 1

            # Update
            if step >= self.learning_starts and len(self.buffer) >= self.batch_size:
                metrics = self._update(step)

                if self.logger is not None and self._global_step % 1000 == 0:
                    self.logger.on_train_step(self._global_step, metrics)

        self.callbacks.on_training_end()

        metrics["mean_reward"] = (
            float(np.mean(episode_rewards)) if episode_rewards else 0.0
        )
        return metrics

    def _update(self, step: int) -> dict[str, float]:
        """Single AWR update step."""
        batch = self.buffer.sample(self.batch_size, step)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)

        # -- Critic update --
        with torch.no_grad():
            next_values = self.critic(next_obs).squeeze(-1)
            targets = rewards + self.gamma * (1.0 - terminated) * next_values

        for _ in range(self.n_critic_updates):
            values = self.critic(obs).squeeze(-1)
            critic_loss = F.mse_loss(values, targets)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()

        # -- Actor update (AWR) --
        with torch.no_grad():
            values = self.critic(obs).squeeze(-1)
            advantages = targets - values
            # Exponentiated advantage weights
            weights = torch.exp(advantages / self.beta)
            weights = weights.clamp(max=self.max_advantage)

        if self.discrete:
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            actions_int = actions.squeeze(-1).long()
            log_probs = dist.log_prob(actions_int)
        else:
            pred_actions = self.actor(obs)
            # Gaussian log-prob with unit variance
            log_probs = -0.5 * ((actions - pred_actions) ** 2).sum(dim=-1)

        actor_loss = -(weights * log_probs).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray | int:
        """Return action for given observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            if self.discrete:
                logits = self.actor(obs)
                return logits.argmax(dim=-1).item()
            else:
                return self.actor(obs).squeeze(0).numpy()

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "step": self._global_step,
                "env_id": self.env_id,
                "config": {
                    "beta": self.beta,
                    "learning_rate": self._learning_rate,
                    "gamma": self.gamma,
                    "batch_size": self.batch_size,
                    "buffer_size": self._buffer_size,
                    "hidden": self._hidden,
                    "learning_starts": self.learning_starts,
                    "n_critic_updates": self.n_critic_updates,
                    "max_advantage": self.max_advantage,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> Self:
        """Restore AWR from a checkpoint."""
        from rlox.checkpoint import safe_torch_load

        data = safe_torch_load(path)
        eid = env_id or data.get("env_id", "CartPole-v1")
        config = data["config"]

        awr = cls(env_id=eid, **config)
        awr.actor.load_state_dict(data["actor_state_dict"])
        awr.critic.load_state_dict(data["critic_state_dict"])
        awr.actor_optimizer.load_state_dict(data["actor_optimizer_state_dict"])
        awr.critic_optimizer.load_state_dict(data["critic_optimizer_state_dict"])
        awr._global_step = data.get("step", 0)
        return awr
