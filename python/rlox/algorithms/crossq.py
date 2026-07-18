"""CrossQ: Batch Normalization in Deep RL for Greater Sample Efficiency.

ICLR 2024 — Bhatt, Palenicek et al. (arXiv:1902.05605).

CrossQ is SAC with two changes:
  1. **No target networks** — BatchRenorm stabilises TD learning instead.
  2. **Joint forward pass** — current and next observations pass through the
     live critic *together* so that BN statistics are consistent between
     Q(s,a) (trained) and Q(s',a') (bootstrap, detached).

Actor update is delayed by ``policy_delay`` (default 3) critic steps.
Everything else — replay buffer, SquashedGaussianPolicy, auto-entropy — is
unchanged from rlox SAC.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.config import CrossQConfig
from rlox.logging import LoggerCallback
from rlox.networks import BNQNetwork, SquashedGaussianPolicy


class CrossQ:
    """CrossQ — SAC minus target networks, plus BatchRenorm critics.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.  Must have a continuous (Box) action space.
    learning_rate : float
        Adam learning rate for all optimisers (default 1e-3).
    adam_betas : tuple[float, float]
        Adam ``(beta1, beta2)`` coefficients for all four optimisers
        (default ``(0.5, 0.999)``).  CrossQ's BatchRenorm critics need a
        lower ``beta1`` than torch's default of 0.9 -- a controlled
        ablation showed ``beta1=0.9`` fails to converge on Pendulum-v1
        while ``beta1=0.5`` (the paper / SB3-contrib value) solves it. See
        docs/plans/crossq-convergence-fix-2026-07-18.md.
    gamma : float
        Discount factor (default 0.99).
    batch_size : int
        Minibatch size (default 256).
    learning_starts : int
        Random exploration steps before training begins (default 1000).
    train_freq : int
        Critic update frequency in env steps (default 1).
    gradient_steps : int
        Gradient steps per training round (default 1).
    policy_delay : int
        Actor update every ``policy_delay`` critic updates (default 3).
    hidden : int
        Hidden layer width (default 256).
    auto_entropy : bool
        Whether to automatically tune the entropy coefficient (default True).
    target_entropy : float or None
        Target entropy for auto-tuning.  None → ``-dim(action_space)``.
    ent_coef : str or float
        ``"auto"`` for learned alpha, or a fixed float value.
    bn_momentum : float
        EMA momentum for BatchRenorm1d running stats (default 0.01).
    bn_eps : float
        Numerical stability term for BatchRenorm1d (default 1e-3).
    renorm_warmup_steps : int
        Training steps before renorm corrections are applied (default 100_000).
    seed : int
        Random seed (default 42).
    callbacks : list[Callback] or None
        Optional training callbacks.
    logger : LoggerCallback or None
        Optional logger.
    """

    def __init__(
        self,
        env_id: str,
        buffer_size: int = 1_000_000,
        learning_rate: float = 1e-3,
        adam_betas: tuple[float, float] = (0.5, 0.999),
        gamma: float = 0.99,
        batch_size: int = 256,
        learning_starts: int = 1000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy_delay: int = 3,
        hidden: int = 256,
        auto_entropy: bool = True,
        target_entropy: float | None = None,
        ent_coef: str | float = "auto",
        bn_momentum: float = 0.01,
        bn_eps: float = 1e-3,
        renorm_warmup_steps: int = 100_000,
        seed: int = 42,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
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

        # CrossQ is continuous-only.
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise ValueError(
                "CrossQ requires a continuous (Box) action space. "
                f"Got a discrete/unsupported action space: "
                f"{type(self.env.action_space).__name__}. "
                "Use SAC or TD3 for continuous envs; DQN/PPO for discrete envs."
            )

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = max(1, int(train_freq))
        self.gradient_steps = max(1, int(gradient_steps))
        self.policy_delay = max(1, int(policy_delay))
        self.seed = seed

        # Seed torch/np/env RNG for reproducibility.  This MUST happen
        # before the actor/critic networks are constructed below so that
        # their weight initialisation is deterministic for a given seed
        # (precedent: pqn.py:118).  Gymnasium does not propagate
        # `env.reset(seed=...)` to the action space's RNG, and the
        # `learning_starts` exploration phase in `train()` calls
        # `action_space.sample()`, so the action space needs its own
        # `.seed()` call here too.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.action_space.seed(seed)

        # ent_coef handling — mirrors SAC.
        if isinstance(ent_coef, (int, float)):
            auto_entropy = False
            self._fixed_alpha = float(ent_coef)
        else:
            self._fixed_alpha = None  # "auto" → learned

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        act_high = float(self.env.action_space.high[0])

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_high = act_high

        self.config = CrossQConfig(
            learning_rate=learning_rate,
            adam_betas=adam_betas,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            target_entropy=target_entropy,
            auto_entropy=auto_entropy,
            learning_starts=learning_starts,
            hidden=hidden,
            policy_delay=policy_delay,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
            renorm_warmup_steps=renorm_warmup_steps,
        )

        # Actor: reuse SAC's SquashedGaussianPolicy unchanged.
        self.actor = SquashedGaussianPolicy(obs_dim, act_dim, hidden)

        # Critics: BNQNetwork with BatchRenorm1d.  NO target critics.
        self.critic1 = BNQNetwork(
            obs_dim,
            act_dim,
            hidden=hidden,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
            renorm_warmup_steps=renorm_warmup_steps,
        )
        self.critic2 = BNQNetwork(
            obs_dim,
            act_dim,
            hidden=hidden,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
            renorm_warmup_steps=renorm_warmup_steps,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate, betas=adam_betas
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=learning_rate, betas=adam_betas
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=learning_rate, betas=adam_betas
        )

        # Entropy tuning.
        self.auto_entropy = auto_entropy
        self.target_entropy = (
            -float(act_dim) if target_entropy is None else target_entropy
        )

        if auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=learning_rate, betas=adam_betas
            )
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self._fixed_alpha if self._fixed_alpha is not None else 0.2

        # Replay buffer.
        self.buffer = rlox.ReplayBuffer(buffer_size, obs_dim, act_dim)

        # Callbacks / logger.
        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

        # Update counters (required by tests).
        self._n_updates = 0
        self._n_actor_updates = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run training loop and return a metrics dict."""
        obs, _ = self.env.reset(seed=self.seed)
        episode_rewards: list[float] = []
        ep_reward = 0.0
        metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for step in range(total_timesteps):
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action_t, _ = self.actor.sample(obs_t)
                    action = action_t.squeeze(0).numpy()
                action = action * self.act_high

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
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

            self._global_step += 1
            should_continue = self.callbacks.on_step(
                reward=ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            if (
                step >= self.learning_starts
                and len(self.buffer) >= self.batch_size
                and step % self.train_freq == 0
            ):
                for _ in range(self.gradient_steps):
                    metrics = self._update(step)
                    self.callbacks.on_train_batch(**metrics)

                if self.logger is not None and self._global_step % 1000 == 0:
                    self.logger.on_train_step(self._global_step, metrics)

        self.callbacks.on_training_end()

        metrics["mean_reward"] = (
            float(np.mean(episode_rewards)) if episode_rewards else 0.0
        )
        return metrics

    def _update(self, step: int) -> dict[str, float]:
        """One gradient step: joint critic forward pass, delayed actor update."""
        batch = self.buffer.sample(self.batch_size, step)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)

        # ------------------------------------------------------------------
        # Critic update — joint forward pass (the CrossQ crux).
        #
        # Current and next transitions pass through the live critic in ONE
        # concatenated batch so that BatchRenorm statistics are consistent
        # between Q(s,a) and Q(s',a').  The bootstrap half is detached;
        # NO target network is used.
        # ------------------------------------------------------------------
        self.critic1.train()
        self.critic2.train()

        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs)
            next_act = next_act * self.act_high

        B = obs.shape[0]
        cat_obs = torch.cat([obs, next_obs], dim=0)  # (2B, obs_dim)
        cat_act = torch.cat([actions, next_act], dim=0)  # (2B, act_dim)

        q1_both = self.critic1(cat_obs, cat_act)  # (2B, 1)
        q2_both = self.critic2(cat_obs, cat_act)  # (2B, 1)

        q1 = q1_both[:B].squeeze(-1)
        q1_next = q1_both[B:].squeeze(-1)
        q2 = q2_both[:B].squeeze(-1)
        q2_next = q2_both[B:].squeeze(-1)

        # Bootstrap target — fully detached, no grad through the target path.
        q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
        target = (rewards + self.gamma * (1.0 - terminated) * q_next).detach()

        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        self._n_updates += 1

        # ------------------------------------------------------------------
        # Actor update — delayed by policy_delay critic updates.
        #
        # Put critics in eval() so that the actor's forward pass does NOT
        # update BatchRenorm running statistics (which would corrupt the BN
        # estimates with the policy's action distribution, not the replay
        # distribution).  Restore train() afterwards so the next critic
        # update runs in the correct mode.
        # ------------------------------------------------------------------
        actor_loss_val = 0.0
        alpha_loss_val = 0.0

        if self._n_updates % self.policy_delay == 0:
            self.critic1.eval()
            self.critic2.eval()

            new_actions, log_prob = self.actor.sample(obs)
            new_actions = new_actions * self.act_high
            q1_new = self.critic1(obs, new_actions).squeeze(-1)
            q2_new = self.critic2(obs, new_actions).squeeze(-1)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_val = actor_loss.item()

            # Alpha / entropy-coefficient update.
            if self.auto_entropy:
                alpha_loss = -(
                    self.log_alpha * (log_prob.detach() + self.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
                alpha_loss_val = alpha_loss.item()

            self._n_actor_updates += 1

            # Restore critic train mode for the next critic update.
            self.critic1.train()
            self.critic2.train()

        return {
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2.0,
            "actor_loss": actor_loss_val,
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_val,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return an action for the given observation."""
        # Critics must be in eval mode for inference so that BN running stats
        # are used and no stats are accidentally updated.
        self.critic1.eval()
        self.critic2.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                action = self.actor.deterministic(obs_t).squeeze(0).numpy()
            else:
                action, _ = self.actor.sample(obs_t)
                action = action.squeeze(0).numpy()
        return action * self.act_high

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save a training checkpoint to *path*."""
        data: dict[str, Any] = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "step": self._global_step,
            "n_updates": self._n_updates,
            "n_actor_updates": self._n_actor_updates,
            "config": self.config.to_dict(),
            "env_id": self.env_id,
            "torch_rng_state": torch.random.get_rng_state(),
        }
        if self.auto_entropy:
            data["log_alpha"] = self.log_alpha.detach().clone()
            data["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        torch.save(data, path)

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> CrossQ:
        """Restore CrossQ from a checkpoint produced by :meth:`save`."""
        from rlox.checkpoint import safe_torch_load

        data = safe_torch_load(path)
        config = data["config"]
        eid = env_id or data.get("env_id", "Pendulum-v1")

        crossq = cls(env_id=eid, **config)
        crossq.actor.load_state_dict(data["actor_state_dict"])
        crossq.critic1.load_state_dict(data["critic1_state_dict"])
        crossq.critic2.load_state_dict(data["critic2_state_dict"])
        crossq.actor_optimizer.load_state_dict(data["actor_optimizer_state_dict"])
        crossq.critic1_optimizer.load_state_dict(data["critic1_optimizer_state_dict"])
        crossq.critic2_optimizer.load_state_dict(data["critic2_optimizer_state_dict"])
        crossq._global_step = data.get("step", 0)
        crossq._n_updates = data.get("n_updates", 0)
        crossq._n_actor_updates = data.get("n_actor_updates", 0)

        if crossq.auto_entropy and "log_alpha" in data:
            crossq.log_alpha.data.copy_(data["log_alpha"])
            crossq.alpha = crossq.log_alpha.exp().item()
            if "alpha_optimizer_state_dict" in data:
                crossq.alpha_optimizer.load_state_dict(
                    data["alpha_optimizer_state_dict"]
                )

        if "torch_rng_state" in data:
            torch.random.set_rng_state(data["torch_rng_state"])

        # Ensure critics are ready for inference after loading.
        crossq.critic1.eval()
        crossq.critic2.eval()

        return crossq
