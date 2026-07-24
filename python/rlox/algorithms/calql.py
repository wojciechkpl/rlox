"""Calibrated Conservative Q-Learning (Cal-QL).

Extends CQL with an adaptive, calibrated penalty that scales the
conservative regularisation based on the gap between current Q-values
and offline return estimates. This avoids the overly pessimistic
behaviour of vanilla CQL on near-distribution actions.

Reference:
    N. Nakamoto, S. Zhai, A. Singh, M. S. Mark, Y. Ma, C. Finn,
    A. Kumar, S. Levine,
    "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online
    Fine-Tuning," NeurIPS, 2023.
    https://arxiv.org/abs/2303.05479
"""

from __future__ import annotations

import copy
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar
    Self = TypeVar("Self")

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rlox.callbacks import Callback, CallbackList
from rlox.logging import LoggerCallback
from rlox.networks import QNetwork, SquashedGaussianPolicy, polyak_update
from rlox.trainer import register_algorithm


@register_algorithm("calql")
class CalQL:
    """Calibrated Conservative Q-Learning -- CQL with adaptive penalty.

    SAC-style actor-critic backbone with a calibrated CQL penalty:
    the standard CQL term is scaled by max(Q(s,a) - Q_calibration, 0),
    where Q_calibration is derived from offline return estimates.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    learning_rate : float
        Learning rate for all optimisers (default 3e-4).
    buffer_size : int
        Replay buffer capacity (default 100_000).
    batch_size : int
        Minibatch size (default 256).
    gamma : float
        Discount factor (default 0.99).
    tau : float
        Polyak averaging coefficient (default 0.005).
    cql_alpha : float
        CQL penalty weight (default 5.0).
    calibration_tau : float
        Quantile for calibration threshold from offline returns (default 0.5).
    auto_alpha : bool
        Whether to auto-tune cql_alpha via dual gradient descent (default False).
    hidden : int
        Hidden layer width (default 256).
    n_random_actions : int
        Random actions for CQL penalty estimation (default 10).
    warmup_steps : int
        Random exploration steps before training (default 1000).
    seed : int
        Random seed (default 42).
    callbacks : list[Callback], optional
    logger : LoggerCallback, optional
    """

    def __init__(
        self,
        env_id: str,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        cql_alpha: float = 5.0,
        calibration_tau: float = 0.5,
        auto_alpha: bool = False,
        hidden: int = 256,
        n_random_actions: int = 10,
        warmup_steps: int = 1000,
        seed: int = 42,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ) -> None:
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.seed = seed

        # Seed torch/np/env RNG for reproducibility.  This MUST happen
        # before the actor/critic networks are constructed below so that
        # their weight initialisation is deterministic for a given seed
        # (precedent: pqn.py:118, crossq.py:141).  Gymnasium does not
        # propagate `env.reset(seed=...)` to the action space's RNG, and
        # the `warmup_steps` random-exploration phase in `train()` calls
        # `action_space.sample()`, so the action space needs its own
        # `.seed()` call here too.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.action_space.seed(seed)

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_space = self.env.action_space
        self._obs_dim = obs_dim
        self.discrete = isinstance(act_space, gym.spaces.Discrete)

        if self.discrete:
            raise ValueError(
                "Cal-QL uses a SAC actor-critic backbone (SquashedGaussianPolicy) "
                "and requires a continuous action space. "
                "For discrete envs, consider CQL or DQN-based offline RL."
            )

        self.act_dim = int(np.prod(act_space.shape))

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.cql_alpha = cql_alpha
        self._calibration_tau = calibration_tau
        self.auto_cql_alpha = auto_alpha
        self.n_random_actions = n_random_actions
        self.warmup_steps = warmup_steps
        self._learning_rate = learning_rate
        self._hidden = hidden
        self._buffer_size = buffer_size

        # SAC-style networks
        self.actor = SquashedGaussianPolicy(obs_dim, self.act_dim, hidden)
        self.critic1 = QNetwork(obs_dim, self.act_dim, hidden)
        self.critic2 = QNetwork(obs_dim, self.act_dim, hidden)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate,
        )

        # Entropy auto-tuning (SAC-style)
        self.target_entropy = -float(self.act_dim)
        self.log_alpha_entropy = torch.zeros(1, requires_grad=True)
        self.alpha_entropy_optimizer = torch.optim.Adam(
            [self.log_alpha_entropy], lr=learning_rate,
        )
        self.alpha_entropy = self.log_alpha_entropy.exp().item()

        # CQL alpha auto-tuning (Lagrangian dual)
        if auto_alpha:
            self.log_cql_alpha = torch.zeros(1, requires_grad=True)
            self.cql_alpha_optimizer = torch.optim.Adam(
                [self.log_cql_alpha], lr=learning_rate,
            )

        # Calibration: running estimate of offline return distribution
        self._calibration_value = 0.0
        self._return_buffer: list[float] = []

        # Simple replay buffer
        self._buffer_obs: list[np.ndarray] = []
        self._buffer_next: list[np.ndarray] = []
        self._buffer_act: list[np.ndarray] = []
        self._buffer_rew: list[float] = []
        self._buffer_done: list[bool] = []
        self._buffer_max = buffer_size
        self._buffer_pos = 0

        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

        self._config = {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
            "cql_alpha": cql_alpha,
            "calibration_tau": calibration_tau,
            "auto_alpha": auto_alpha,
            "hidden": hidden,
            "n_random_actions": n_random_actions,
            "warmup_steps": warmup_steps,
            "seed": seed,
        }

    # -- Buffer operations -----------------------------------------------------

    def _push(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        if len(self._buffer_obs) < self._buffer_max:
            self._buffer_obs.append(obs)
            self._buffer_next.append(next_obs)
            self._buffer_act.append(act)
            self._buffer_rew.append(reward)
            self._buffer_done.append(done)
        else:
            self._buffer_obs[self._buffer_pos] = obs
            self._buffer_next[self._buffer_pos] = next_obs
            self._buffer_act[self._buffer_pos] = act
            self._buffer_rew[self._buffer_pos] = reward
            self._buffer_done[self._buffer_pos] = done
        self._buffer_pos = (self._buffer_pos + 1) % self._buffer_max

    def _sample(
        self, rng: np.random.Generator,
    ) -> dict[str, torch.Tensor]:
        n = len(self._buffer_obs)
        indices = rng.integers(0, n, size=self.batch_size)
        return {
            "obs": torch.as_tensor(
                np.stack([self._buffer_obs[i] for i in indices]), dtype=torch.float32,
            ),
            "next_obs": torch.as_tensor(
                np.stack([self._buffer_next[i] for i in indices]), dtype=torch.float32,
            ),
            "actions": torch.as_tensor(
                np.stack([self._buffer_act[i] for i in indices]), dtype=torch.float32,
            ),
            "rewards": torch.as_tensor(
                np.array([self._buffer_rew[i] for i in indices]), dtype=torch.float32,
            ),
            "terminated": torch.as_tensor(
                np.array([float(self._buffer_done[i]) for i in indices]),
                dtype=torch.float32,
            ),
        }

    # -- Calibration -----------------------------------------------------------

    def _update_calibration(self, episode_return: float) -> None:
        """Update the calibration threshold from episode returns."""
        self._return_buffer.append(episode_return)
        if self._return_buffer:
            self._calibration_value = float(
                np.quantile(self._return_buffer, self._calibration_tau)
            )

    def _compute_calibrated_penalty(
        self,
        q_current: torch.Tensor,
        q_offline: torch.Tensor,
    ) -> torch.Tensor:
        """Compute calibrated CQL penalty.

        penalty = max(Q(s,a) - Q_calibration, 0) * cql_alpha
        scaled by calibration_tau.
        """
        gap = q_current - self._calibration_value
        scale = torch.clamp(gap, min=0.0) * self._calibration_tau
        raw_penalty = q_current - q_offline
        return (scale * raw_penalty).mean()

    # -- Training --------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run Cal-QL training loop.

        Collects data online and applies calibrated CQL updates.

        Returns
        -------
        dict with 'mean_reward', 'critic_loss', 'actor_loss', 'cql_loss'.
        """
        rng = np.random.default_rng(self.seed)
        obs, _ = self.env.reset(seed=self.seed)
        episode_rewards: list[float] = []
        ep_reward = 0.0
        metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for step in range(total_timesteps):
            # Collect experience
            if step < self.warmup_steps:
                if self.discrete:
                    action = self.env.action_space.sample()
                    act_arr = np.array([action], dtype=np.float32)
                else:
                    action = self.env.action_space.sample()
                    act_arr = np.asarray(action, dtype=np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action_t, _ = self.actor.sample(obs_t)
                    act_arr = action_t.squeeze(0).numpy()
                    if self.discrete:
                        action = int(np.clip(np.round(act_arr[0]), 0, self.env.action_space.n - 1))
                    else:
                        action = act_arr

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            ep_reward += float(reward)

            self._push(
                np.asarray(obs, dtype=np.float32),
                act_arr if not self.discrete else np.array([action], dtype=np.float32),
                float(reward),
                bool(terminated),
                np.asarray(next_obs, dtype=np.float32),
            )

            obs = next_obs
            if terminated or truncated:
                episode_rewards.append(ep_reward)
                self._update_calibration(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

            # Update
            if step >= self.warmup_steps and len(self._buffer_obs) >= self.batch_size:
                metrics = self._update(rng)

            self._global_step += 1

        self.callbacks.on_training_end()

        metrics["mean_reward"] = (
            float(np.mean(episode_rewards)) if episode_rewards else 0.0
        )
        return metrics

    def _update(self, rng: np.random.Generator) -> dict[str, float]:
        """Single Cal-QL update step."""
        batch = self._sample(rng)
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminated = batch["terminated"]

        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        bs = obs.shape[0]

        # -- Critic update --
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            q1_next = self.critic1_target(next_obs, next_actions).squeeze(-1)
            q2_next = self.critic2_target(next_obs, next_actions).squeeze(-1)
            q_next = torch.min(q1_next, q2_next) - self.alpha_entropy * next_log_prob
            target_q = rewards + self.gamma * (1.0 - terminated) * q_next

        q1_data = self.critic1(obs, actions).squeeze(-1)
        q2_data = self.critic2(obs, actions).squeeze(-1)
        bellman_loss = F.mse_loss(q1_data, target_q) + F.mse_loss(q2_data, target_q)

        # CQL penalty with calibration
        random_actions = torch.FloatTensor(bs * self.n_random_actions, self.act_dim).uniform_(-1, 1)
        obs_rep = obs.unsqueeze(1).expand(-1, self.n_random_actions, -1).reshape(-1, obs.shape[-1])

        q1_rand = self.critic1(obs_rep, random_actions).reshape(bs, self.n_random_actions)
        q2_rand = self.critic2(obs_rep, random_actions).reshape(bs, self.n_random_actions)

        with torch.no_grad():
            policy_actions, _ = self.actor.sample(obs_rep)
        q1_policy = self.critic1(obs_rep, policy_actions).reshape(bs, self.n_random_actions)
        q2_policy = self.critic2(obs_rep, policy_actions).reshape(bs, self.n_random_actions)

        cat_q1 = torch.cat([q1_rand, q1_policy], dim=1)
        cat_q2 = torch.cat([q2_rand, q2_policy], dim=1)

        # Standard CQL: logsumexp(Q_ood) - Q_data
        cql_q1 = torch.logsumexp(cat_q1, dim=1).mean() - q1_data.mean()
        cql_q2 = torch.logsumexp(cat_q2, dim=1).mean() - q2_data.mean()

        # Calibrated penalty: scale by max(Q - Q_cal, 0)
        cal_penalty = self._compute_calibrated_penalty(
            torch.logsumexp(cat_q1, dim=1),
            q1_data,
        )

        cql_loss = cql_q1 + cql_q2 + cal_penalty

        # Auto-tune CQL alpha
        if self.auto_cql_alpha:
            cql_alpha_val = self.log_cql_alpha.exp()
            alpha_loss = cql_alpha_val * (cql_loss.detach() - 1.0)
            self.cql_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.cql_alpha_optimizer.step()
            self.cql_alpha = cql_alpha_val.item()

        critic_loss = bellman_loss + self.cql_alpha * cql_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # -- Actor update (SAC-style) --
        new_actions, log_prob = self.actor.sample(obs)
        q1_new = self.critic1(obs, new_actions).squeeze(-1)
        q2_new = self.critic2(obs, new_actions).squeeze(-1)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha_entropy * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Entropy alpha update
        alpha_ent_loss = -(
            self.log_alpha_entropy * (log_prob.detach() + self.target_entropy)
        ).mean()
        self.alpha_entropy_optimizer.zero_grad(set_to_none=True)
        alpha_ent_loss.backward()
        self.alpha_entropy_optimizer.step()
        self.alpha_entropy = self.log_alpha_entropy.exp().item()

        # Soft target update
        polyak_update(self.critic1, self.critic1_target, self.tau)
        polyak_update(self.critic2, self.critic2_target, self.tau)

        return {
            "critic_loss": bellman_loss.item() / 2,
            "cql_loss": cql_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_entropy": self.alpha_entropy,
            "cql_alpha": self.cql_alpha,
            "calibration_value": self._calibration_value,
        }

    # -- Inference -------------------------------------------------------------

    def predict(
        self, obs: np.ndarray, deterministic: bool = True,
    ) -> np.ndarray:
        """Get action from the policy."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                return self.actor.deterministic(obs_t).squeeze(0).numpy()
            action, _ = self.actor.sample(obs_t)
            return action.squeeze(0).numpy()

    # -- Serialization ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Save checkpoint."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic1_target": self.critic1_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "step": self._global_step,
                "env_id": self.env_id,
                "config": self._config,
                "calibration_value": self._calibration_value,
                "return_buffer": self._return_buffer,
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> Self:
        """Restore from checkpoint."""
        from rlox.checkpoint import safe_torch_load

        data = safe_torch_load(path)
        eid = env_id or data.get("env_id", "CartPole-v1")
        config = data["config"]

        agent = cls(env_id=eid, **config)
        agent.actor.load_state_dict(data["actor"])
        agent.critic1.load_state_dict(data["critic1"])
        agent.critic2.load_state_dict(data["critic2"])
        agent.critic1_target.load_state_dict(data["critic1_target"])
        agent.critic2_target.load_state_dict(data["critic2_target"])
        agent.actor_optimizer.load_state_dict(data["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(data["critic_optimizer"])
        agent._global_step = data.get("step", 0)
        agent._calibration_value = data.get("calibration_value", 0.0)
        agent._return_buffer = data.get("return_buffer", [])
        return agent
