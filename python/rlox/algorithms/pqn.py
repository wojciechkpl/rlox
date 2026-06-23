"""Parallelised Q-Network (PQN) — Gallici et al., arXiv:2407.04811.

PQN is a value-based algorithm that exploits many parallel environments to
collect on-policy rollouts and computes Q(λ) targets via the Rust GAE op,
with **no target network and no replay buffer**.  LayerNorm in the Q-network
replaces the target network as the stability mechanism.

Key differences from DQN
-------------------------
- No target network (LayerNorm stabilises TD learning instead).
- No replay buffer (on-policy rollouts from ``n_envs`` parallel envs).
- Q(λ) multi-step returns via ``rlox.compute_gae_batched`` (Rust).
- Multiple SGD epochs over each rollout (similar cadence to PPO).
- Discrete action spaces only.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.checkpoint import safe_torch_load
from rlox.config import PQNConfig
from rlox.gym_vec_env import GymVecEnv
from rlox.logging import LoggerCallback
from rlox.networks import LayerNormQNetwork

# CartPole has a native Rust VecEnv — matches the convention used by PPO/A2C/TRPO
# for discrete envs only (not the broader _NATIVE_ENV_IDS set in collectors.py
# which also includes Pendulum; PQN is discrete-only so Pendulum is irrelevant).
_RUST_NATIVE_ENVS: frozenset[str] = frozenset({"CartPole-v1", "CartPole"})


class PQN:
    """Parallelised Q-Network (PQN).

    Implements arXiv:2407.04811: value-based RL with parallel envs,
    LayerNorm Q-net, Q(λ) returns, no target network, no replay buffer.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.  Must have a Discrete action space.
    n_envs : int
        Number of parallel environments (default 8).
    n_steps : int
        Rollout length per environment per update (default 32).
    learning_rate : float
        Adam learning rate (default 2.5e-4).
    gamma : float
        Discount factor (default 0.99).
    q_lambda : float
        Lambda for Q(λ) / TD(λ) returns (default 0.65).
    num_epochs : int
        SGD epochs over each rollout (default 4).
    num_minibatches : int
        Number of minibatches per epoch (default 4).
    max_grad_norm : float
        Gradient clipping threshold (default 10.0).
    weight_decay : float
        Adam weight-decay (ℓ² regularisation, default 0.0).
    hidden : int
        Hidden layer width for the Q-network (default 128).
    eps_start : float
        Initial ε for ε-greedy exploration (default 1.0).
    eps_end : float
        Final ε after the exploration schedule (default 0.05).
    exploration_fraction : float
        Fraction of ``total_timesteps`` over which ε is linearly annealed
        (default 0.5).
    seed : int
        Random seed (default 42).
    q_network : nn.Module or None
        Custom Q-network.  If None, a ``LayerNormQNetwork`` is built.
    callbacks : list[Callback] or None
        Training callbacks.
    logger : LoggerCallback or None
        Logger for metrics.
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 8,
        n_steps: int = 32,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        q_lambda: float = 0.65,
        num_epochs: int = 4,
        num_minibatches: int = 4,
        max_grad_norm: float = 10.0,
        weight_decay: float = 0.0,
        hidden: int = 128,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        exploration_fraction: float = 0.5,
        seed: int = 42,
        q_network: nn.Module | None = None,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
        **kwargs: Any,
    ):
        self.env_id = env_id
        self.seed = seed

        # Per-instance seeded RNG for all ε-greedy draws (Fix 2).
        self._rng = np.random.default_rng(seed)

        # Seed PyTorch so that network weight initialisation is reproducible
        # with the given seed.  This ensures training trajectories are
        # deterministic when the same seed is used across runs.
        torch.manual_seed(seed)

        # Detect action space — PQN is discrete-only.
        tmp = gym.make(env_id)
        try:
            if not isinstance(tmp.action_space, gym.spaces.Discrete):
                raise ValueError(
                    f"PQN supports discrete action spaces only. "
                    f"Environment {env_id!r} has a "
                    f"{type(tmp.action_space).__name__} action space. "
                    f"Use a discrete environment."
                )
            obs_dim = int(np.prod(tmp.observation_space.shape))
            n_actions = int(tmp.action_space.n)
        finally:
            tmp.close()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Build config (stores all hyperparams for checkpoint serialisation).
        self.config = PQNConfig(
            n_envs=n_envs,
            n_steps=n_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            q_lambda=q_lambda,
            num_epochs=num_epochs,
            num_minibatches=num_minibatches,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            hidden=hidden,
            eps_start=eps_start,
            eps_end=eps_end,
            exploration_fraction=exploration_fraction,
        )

        # Q-network: LayerNorm MLP (the core PQN ingredient).
        if q_network is not None:
            self.q_network: nn.Module = q_network
        else:
            self.q_network = LayerNormQNetwork(obs_dim, n_actions, hidden=hidden)

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # ε-greedy state
        self.epsilon: float = eps_start

        # Vectorised environment — Rust native for CartPole, GymVecEnv otherwise.
        if env_id in _RUST_NATIVE_ENVS:
            try:
                self._vec_env = rlox.VecEnv(n=n_envs, seed=seed, env_id=env_id)
                self._is_rust_env = True
            except (ValueError, RuntimeError):
                self._vec_env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)
                self._is_rust_env = False
        else:
            self._vec_env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)
            self._is_rust_env = False

        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Public training interface
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run PQN training and return a metrics dict.

        The returned dict always contains a ``"loss"`` key with the mean
        Q-MSE loss over the final rollout's SGD updates, and a
        ``"mean_reward"`` key with the mean completed-episode reward.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to collect.

        Returns
        -------
        dict[str, float]
        """
        cfg = self.config
        n_envs = cfg.n_envs
        n_steps = cfg.n_steps
        steps_per_rollout = n_envs * n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)
        minibatch_size = max(1, steps_per_rollout // cfg.num_minibatches)

        # Exploration schedule — total exploration budget in env steps.
        exploration_steps = max(1, int(total_timesteps * cfg.exploration_fraction))

        obs = self._vec_env.reset_all()  # (n_envs, obs_dim)

        completed_rewards: list[float] = []
        ep_rewards = np.zeros(n_envs, dtype=np.float64)

        last_metrics: dict[str, float] = {"loss": 0.0}

        self.callbacks.on_training_start()

        for update in range(n_updates):
            # ----------------------------------------------------------
            # Phase 1: collect n_steps rollout with ε-greedy policy.
            # ----------------------------------------------------------
            rollout_obs: list[np.ndarray] = []
            rollout_actions: list[np.ndarray] = []
            rollout_rewards: list[np.ndarray] = []
            rollout_terminated: list[np.ndarray] = []
            rollout_truncated: list[np.ndarray] = []
            # terminal_obs_list[t][i]: ndarray or None — obs at end of truncated ep.
            rollout_terminal_obs: list[list[np.ndarray | None]] = []

            for _ in range(n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)

                with torch.no_grad():
                    q_vals = self.q_network(obs_t)  # (n_envs, n_actions)
                    greedy_actions = q_vals.argmax(dim=-1).numpy()  # (n_envs,)

                # ε-greedy action selection — use per-instance RNG (Fix 2).
                random_mask = self._rng.random(n_envs) < self.epsilon
                random_actions = self._rng.integers(0, self.n_actions, size=(n_envs,))
                actions = np.where(random_mask, random_actions, greedy_actions)

                # Step the vectorised environment.
                if self._is_rust_env:
                    step_result = self._vec_env.step_all(
                        actions.astype(np.uint32).tolist()
                    )
                else:
                    step_result = self._vec_env.step_all(actions)

                rewards = step_result["rewards"].astype(np.float64)
                terminated = step_result["terminated"].astype(bool)
                truncated = step_result["truncated"].astype(bool)
                dones = terminated | truncated

                rollout_obs.append(obs.copy())
                rollout_actions.append(actions.copy())
                rollout_rewards.append(rewards)
                rollout_terminated.append(terminated)
                rollout_truncated.append(truncated)
                # Capture terminal obs provided by the VecEnv wrapper (Fix 1).
                raw_terminal = step_result.get("terminal_obs")
                if raw_terminal is None:
                    raw_terminal = [None] * n_envs
                rollout_terminal_obs.append(list(raw_terminal))

                # Episode stats
                ep_rewards += rewards
                for i in range(n_envs):
                    if dones[i]:
                        completed_rewards.append(float(ep_rewards[i]))
                        ep_rewards[i] = 0.0

                obs = step_result["obs"]
                self._global_step += n_envs

                self.callbacks.on_step(
                    reward=float(rewards.mean()),
                    step=self._global_step,
                    algo=self,
                )

            # ----------------------------------------------------------
            # Phase 2: compute Q(λ) targets with truncation bootstrap (Fix 1).
            # ----------------------------------------------------------
            obs_arr = np.stack(rollout_obs)           # (n_steps, n_envs, obs_dim)
            actions_arr = np.stack(rollout_actions)   # (n_steps, n_envs)
            rewards_arr = np.stack(rollout_rewards)   # (n_steps, n_envs)
            terminated_arr = np.stack(rollout_terminated)  # (n_steps, n_envs)
            truncated_arr = np.stack(rollout_truncated)    # (n_steps, n_envs)

            obs_sgd, actions_sgd, returns_t = self._compute_targets(
                obs_arr=obs_arr,
                actions_arr=actions_arr,
                rewards_arr=rewards_arr,
                terminated_arr=terminated_arr,
                truncated_arr=truncated_arr,
                terminal_obs_list=rollout_terminal_obs,
                last_obs=obs,
            )

            # ----------------------------------------------------------
            # Phase 3: num_epochs × num_minibatches SGD updates.
            # ----------------------------------------------------------
            total_samples = n_steps * n_envs
            epoch_losses: list[float] = []

            for _epoch in range(cfg.num_epochs):
                perm = torch.randperm(total_samples)
                for start in range(0, total_samples, minibatch_size):
                    idx = perm[start : start + minibatch_size]
                    mb_obs = obs_sgd[idx]
                    mb_actions = actions_sgd[idx]
                    mb_returns = returns_t[idx].detach()

                    q_values = self.q_network(mb_obs)  # (mb, n_actions)
                    q_taken = q_values.gather(
                        1, mb_actions.unsqueeze(1)
                    ).squeeze(1)  # (mb,)

                    loss = nn.functional.mse_loss(q_taken, mb_returns)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.q_network.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    epoch_losses.append(loss.item())

                    self.callbacks.on_train_batch(loss=loss.item())

            # ----------------------------------------------------------
            # Phase 4: decay ε linearly.
            # ----------------------------------------------------------
            elapsed = (update + 1) * steps_per_rollout
            fraction = min(1.0, elapsed / exploration_steps)
            self.epsilon = cfg.eps_start + fraction * (cfg.eps_end - cfg.eps_start)

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            last_metrics = {"loss": mean_loss}

            if self.logger is not None:
                self.logger.on_train_step(
                    self._global_step,
                    {**last_metrics, "epsilon": self.epsilon},
                )

        self.callbacks.on_training_end()

        last_metrics["mean_reward"] = (
            float(np.mean(completed_rewards)) if completed_rewards else 0.0
        )
        last_metrics["epsilon"] = self.epsilon
        return last_metrics

    # ------------------------------------------------------------------
    # Target computation (testable seam — Fix 1)
    # ------------------------------------------------------------------

    def _compute_targets(
        self,
        obs_arr: np.ndarray,
        actions_arr: np.ndarray,
        rewards_arr: np.ndarray,
        terminated_arr: np.ndarray,
        truncated_arr: np.ndarray,
        terminal_obs_list: list[list[np.ndarray | None]],
        last_obs: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q(λ) targets with correct truncation bootstrap.

        On steps that are truncated but not terminated, add
        ``gamma * max_a Q(terminal_obs, a)`` to the reward **before** passing
        only ``terminated`` (not ``truncated``) to ``compute_gae_batched``.
        This mirrors the ``RolloutCollector`` pattern in ``collectors.py``.

        Parameters
        ----------
        obs_arr : ndarray, shape (n_steps, n_envs, obs_dim)
        actions_arr : ndarray, shape (n_steps, n_envs) — int actions
        rewards_arr : ndarray, shape (n_steps, n_envs) — float64
        terminated_arr : ndarray, shape (n_steps, n_envs) — bool
        truncated_arr : ndarray, shape (n_steps, n_envs) — bool
        terminal_obs_list : list of length n_steps, each element is a list
            of length n_envs where terminal_obs_list[t][i] is either an
            ndarray of shape (obs_dim,) or None.
        last_obs : ndarray, shape (n_envs, obs_dim) — next obs after rollout

        Returns
        -------
        obs_flat_t : Tensor, shape (n_steps * n_envs, obs_dim)  step-major
        actions_flat_t : Tensor, shape (n_steps * n_envs,)  step-major
        returns_t : Tensor, shape (n_steps * n_envs,)  step-major
        """
        cfg = self.config
        n_steps, n_envs = obs_arr.shape[:2]

        # Work with a mutable copy so we don't modify the caller's array.
        rewards_aug = rewards_arr.copy()  # (n_steps, n_envs) float64

        # Truncation bootstrap: for each truncated-but-not-terminated step,
        # add gamma * V(terminal_obs) to the reward.
        for t in range(n_steps):
            step_terminal_obs = terminal_obs_list[t]
            for i in range(n_envs):
                if (
                    truncated_arr[t, i]
                    and not terminated_arr[t, i]
                    and step_terminal_obs[i] is not None
                ):
                    term_obs_t = torch.as_tensor(
                        np.asarray(step_terminal_obs[i], dtype=np.float32),
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        term_v = self.q_network(term_obs_t).max(dim=-1).values.item()
                    rewards_aug[t, i] += cfg.gamma * term_v

        # Per-step state values: V_t = max_a Q(s_t, a) — no grad.
        obs_flat_t = torch.as_tensor(
            obs_arr.reshape(n_steps * n_envs, self.obs_dim), dtype=torch.float32
        )
        with torch.no_grad():
            q_all = self.q_network(obs_flat_t)  # (n_steps*n_envs, n_actions)
            values_flat = q_all.max(dim=-1).values.numpy()  # (n_steps*n_envs,)

        # Bootstrap value at s_{n_steps}.
        last_obs_t = torch.as_tensor(last_obs, dtype=torch.float32)
        with torch.no_grad():
            last_q = self.q_network(last_obs_t)  # (n_envs, n_actions)
            last_values = last_q.max(dim=-1).values.numpy()  # (n_envs,)

        # Env-major flattening required by compute_gae_batched:
        # transpose (n_steps, n_envs) -> (n_envs, n_steps), then ravel.
        rewards_flat = rewards_aug.T.ravel().astype(np.float64)
        values_env_major = (
            values_flat.reshape(n_steps, n_envs).T.ravel().astype(np.float64)
        )
        # Pass only terminated — NOT (terminated | truncated) — to GAE (Fix 1).
        dones_flat = terminated_arr.T.ravel().astype(np.float64)

        _adv_flat, ret_flat = rlox.compute_gae_batched(
            rewards=rewards_flat,
            values=values_env_major,
            dones=dones_flat,
            last_values=last_values.astype(np.float64),
            n_steps=n_steps,
            gamma=cfg.gamma,
            lam=cfg.q_lambda,
        )

        # ret_flat is env-major (n_envs * n_steps); convert to step-major.
        returns_step_major = ret_flat.reshape(n_envs, n_steps).T  # (n_steps, n_envs)
        returns_t = torch.as_tensor(
            returns_step_major.ravel(), dtype=torch.float32
        )  # (n_steps*n_envs,) step-major

        actions_flat_t = torch.as_tensor(
            actions_arr.ravel(), dtype=torch.long
        )  # (n_steps*n_envs,) step-major

        return obs_flat_t, actions_flat_t, returns_t

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Return an action for the given observation.

        Parameters
        ----------
        obs : np.ndarray
            Single observation of shape ``(obs_dim,)``.
        deterministic : bool
            If True, return argmax Q (greedy). If False, apply ε-greedy.

        Returns
        -------
        int
        """
        if not deterministic and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))

        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_network(obs_t).argmax(dim=-1).item())

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save a training checkpoint to *path*."""
        data: dict[str, Any] = {
            "q_network_state_dict": self.q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self._global_step,
            "config": self.config.to_dict(),
            "env_id": self.env_id,
            "epsilon": self.epsilon,
            "torch_rng_state": torch.random.get_rng_state(),
            "rng_state": self._rng.bit_generator.state,
        }
        torch.save(data, path)

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> PQN:
        """Restore PQN from a saved checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file written by :meth:`save`.
        env_id : str, optional
            Override the environment ID stored in the checkpoint.

        Returns
        -------
        PQN
        """
        data = safe_torch_load(path)
        config: dict[str, Any] = data["config"]
        eid = env_id or data.get("env_id", "CartPole-v1")

        pqn = cls(env_id=eid, **config)
        pqn.q_network.load_state_dict(data["q_network_state_dict"])
        pqn.optimizer.load_state_dict(data["optimizer_state_dict"])
        pqn._global_step = data.get("step", 0)
        pqn.epsilon = data.get("epsilon", pqn.epsilon)

        if "torch_rng_state" in data:
            torch.random.set_rng_state(data["torch_rng_state"])

        if "rng_state" in data:
            pqn._rng.bit_generator.state = data["rng_state"]

        return pqn
