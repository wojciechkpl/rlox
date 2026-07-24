"""Maximum a Posteriori Policy Optimization (MPO).

Abdolmaleki et al., 2018. Off-policy algorithm that decouples policy
improvement into an E-step (compute action weights from Q-values) and
an M-step (fit the parametric policy via KL-constrained supervised
learning). Dual variables auto-tune the temperature and KL constraint.
"""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.checkpoint import Checkpoint
from rlox.config import MPOConfig
from rlox.networks import QNetwork, SquashedGaussianPolicy, polyak_update
from rlox.trainer import register_algorithm


@register_algorithm("mpo")
class MPO:
    """Maximum a Posteriori Policy Optimization.

    Off-policy algorithm with E-step/M-step decomposition and dual
    variable optimization for temperature and KL constraints.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    seed : int
        Random seed (default 42).
    **config_kwargs
        Override any MPOConfig fields.
    """

    def __init__(
        self,
        env_id: str,
        seed: int = 42,
        logger: Any | None = None,
        callbacks: list[Callback] | None = None,
        **config_kwargs: Any,
    ) -> None:
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

        cfg_fields = {f.name for f in MPOConfig.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in config_kwargs.items() if k in cfg_fields}
        self.config = MPOConfig(**cfg_dict)

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        act_high = float(self.env.action_space.high[0])

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_high = act_high
        hidden = self.config.hidden

        # Actor: squashed Gaussian policy
        self.actor = SquashedGaussianPolicy(obs_dim, act_dim, hidden)

        # Twin Q-networks
        self.critic1 = QNetwork(obs_dim, act_dim, hidden)
        self.critic2 = QNetwork(obs_dim, act_dim, hidden)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.config.learning_rate,
        )

        # Dual variables (log-space for positivity)
        # eta: temperature for E-step
        self.log_eta = torch.zeros(1, requires_grad=True)
        self.dual_optimizer = torch.optim.Adam(
            [self.log_eta], lr=self.config.dual_lr
        )

        # Replay buffer
        self.buffer = rlox.ReplayBuffer(self.config.buffer_size, obs_dim, act_dim)

        # Callbacks and logger
        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

    # ------------------------------------------------------------------
    # E-step: compute non-parametric action distribution from Q-values
    # ------------------------------------------------------------------

    def compute_e_step_weights(
        self,
        q_values: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute normalized weights for the E-step.

        q(a|s) proportional to exp(Q(s,a) / eta)

        Parameters
        ----------
        q_values : (N,) tensor
            Q-values for sampled actions.
        temperature : float
            Temperature parameter (eta).

        Returns
        -------
        weights : (N,) tensor
            Normalized action weights summing to 1.
        """
        log_weights = q_values / max(temperature, 1e-8)
        # Numerically stable softmax
        log_weights = log_weights - log_weights.max()
        weights = torch.exp(log_weights)
        weights = weights / weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Dual variable update
    # ------------------------------------------------------------------

    def update_dual(
        self,
        q_values: torch.Tensor,
        target_epsilon: float | None = None,
    ) -> dict[str, float]:
        """Update the temperature dual variable via gradient descent.

        Minimizes: eta * epsilon + eta * log(1/N * sum(exp(Q/eta)))

        Parameters
        ----------
        q_values : (N,) tensor
            Q-values for sampled actions.
        target_epsilon : float, optional
            KL constraint. Defaults to config.epsilon.

        Returns
        -------
        info : dict with 'eta', 'dual_loss'.
        """
        eps = target_epsilon if target_epsilon is not None else self.config.epsilon
        eta = self.log_eta.exp()
        n = q_values.shape[0]

        # Dual loss: eta * eps + eta * log(1/N * sum(exp(Q / eta)))
        scaled_q = q_values.detach() / eta
        log_sum_exp = torch.logsumexp(scaled_q, dim=0)
        dual_loss = eta * eps + eta * (log_sum_exp - np.log(n))

        self.dual_optimizer.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.dual_optimizer.step()

        return {"eta": eta.item(), "dual_loss": dual_loss.item()}

    # ------------------------------------------------------------------
    # M-step: fit policy to E-step distribution
    # ------------------------------------------------------------------

    def _m_step(
        self,
        obs: torch.Tensor,
        sampled_actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> dict[str, float]:
        """Fit the parametric policy to match the E-step distribution.

        Minimizes weighted negative log-likelihood (KL(q || pi_theta)).

        Parameters
        ----------
        obs : (B, obs_dim) tensor
        sampled_actions : (B, N, act_dim) tensor
        weights : (B, N) tensor -- E-step weights

        Returns
        -------
        info : dict with 'actor_loss'.
        """
        b, n, _ = sampled_actions.shape

        # Expand obs to match action samples: (B, N, obs_dim)
        obs_expanded = obs.unsqueeze(1).expand(b, n, self.obs_dim)
        obs_flat = obs_expanded.reshape(b * n, self.obs_dim)
        actions_flat = sampled_actions.reshape(b * n, self.act_dim)

        # Get log-probs under current policy
        mean, log_std = self.actor(obs_flat)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # Inverse tanh to get pre-squash values
        actions_clamped = actions_flat.clamp(-0.999, 0.999)
        pre_tanh = torch.atanh(actions_clamped)
        log_probs = dist.log_prob(pre_tanh)
        # Tanh correction
        log_probs = log_probs - torch.log(1.0 - actions_flat.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1).reshape(b, n)

        # Weighted negative log-likelihood
        weights_detached = weights.detach()
        actor_loss = -(weights_detached * log_probs).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item()}

    # ------------------------------------------------------------------
    # Critic update
    # ------------------------------------------------------------------

    def _update_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        """Update twin Q-networks using clipped double Q-learning."""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_target = self.critic1_target(next_obs, next_actions)
            q2_target = self.critic2_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target).squeeze(-1)
            target = rewards + self.config.gamma * (1.0 - dones) * q_target

        q1 = self.critic1(obs, actions).squeeze(-1)
        q2 = self.critic2(obs, actions).squeeze(-1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return {"critic_loss": critic_loss.item()}

    # ------------------------------------------------------------------
    # Full update step
    # ------------------------------------------------------------------

    def _update(self, batch_size: int, step: int = 0) -> dict[str, float]:
        """Perform a single MPO update step from replay buffer."""
        cfg = self.config

        # Sample batch
        batch = self.buffer.sample(batch_size, step)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["terminated"], dtype=torch.float32)

        # 1. Critic update
        critic_info = self._update_critic(obs, actions, rewards, next_obs, dones)

        # 2. E-step: sample actions and compute Q-values
        n_samples = cfg.n_action_samples
        b = obs.shape[0]
        with torch.no_grad():
            obs_expanded = obs.unsqueeze(1).expand(b, n_samples, self.obs_dim)
            obs_flat = obs_expanded.reshape(b * n_samples, self.obs_dim)
            sampled_actions, _ = self.actor.sample(obs_flat)
            sampled_actions = sampled_actions.reshape(b, n_samples, self.act_dim)

            # Q-values for each sampled action
            q1 = self.critic1(
                obs_flat,
                sampled_actions.reshape(b * n_samples, self.act_dim),
            ).reshape(b, n_samples)
            q2 = self.critic2(
                obs_flat,
                sampled_actions.reshape(b * n_samples, self.act_dim),
            ).reshape(b, n_samples)
            q_vals = torch.min(q1, q2)

        # Compute E-step weights (vectorized softmax over action samples)
        eta = self.log_eta.exp().item()
        log_w = q_vals / max(eta, 1e-8)
        log_w = log_w - log_w.max(dim=-1, keepdim=True).values
        weights = torch.softmax(log_w, dim=-1)

        # 3. Dual update (use mean Q-values across batch)
        q_flat = q_vals.reshape(-1)
        dual_info = self.update_dual(q_flat)

        # 4. M-step: fit policy
        actor_info = self._m_step(obs, sampled_actions, weights)

        # 5. Target network update
        polyak_update(self.critic1, self.critic1_target, cfg.tau)
        polyak_update(self.critic2, self.critic2_target, cfg.tau)

        return {**critic_info, **dual_info, **actor_info}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run MPO training and return final metrics.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps.

        Returns
        -------
        metrics : dict
        """
        cfg = self.config
        obs, _ = self.env.reset(seed=self.seed)
        episode_rewards: list[float] = []
        ep_reward = 0.0
        metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for step in range(total_timesteps):
            if step < cfg.learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action_t, _ = self.actor.sample(obs_t)
                    action = action_t.squeeze(0).numpy()
                    action = np.clip(action, -self.act_high, self.act_high)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
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
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()

            # Update after enough experience
            if step >= cfg.learning_starts and len(self.buffer) >= cfg.batch_size:
                metrics = self._update(cfg.batch_size, step=step)

            self._global_step += 1

            should_continue = self.callbacks.on_step(
                reward=ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

        self.callbacks.on_training_end()

        metrics["mean_reward"] = (
            float(np.mean(episode_rewards)) if episode_rewards else 0.0
        )
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        Checkpoint.save(
            path,
            model=self.actor,
            optimizer=self.actor_optimizer,
            step=self._global_step,
            config=self.config.to_dict(),
        )

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> MPO:
        """Restore MPO from a checkpoint."""
        data = Checkpoint.load(path)
        config = data["config"]
        eid = env_id or config.get("env_id", "Pendulum-v1")
        mpo = cls(env_id=eid, **config)
        mpo.actor.load_state_dict(data["model_state_dict"])
        mpo.actor_optimizer.load_state_dict(data["optimizer_state_dict"])
        mpo._global_step = data.get("step", 0)
        return mpo

    def predict(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        """Get action from the trained policy."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).numpy()
