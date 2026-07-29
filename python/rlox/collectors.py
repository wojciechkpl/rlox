"""RolloutCollector: gather on-policy experience using Rust VecEnv + GAE.

This module provides the bridge between environment stepping (Rust) and
policy evaluation (PyTorch). It collects ``n_steps`` of experience from
``n_envs`` parallel environments and computes GAE advantages, returning
a flat :class:`~rlox.batch.RolloutBatch` ready for SGD.

For Rust-native environments (CartPole, Pendulum), the collector uses ``rlox.VecEnv``
for maximum throughput. For all other Gymnasium environments, it falls back
to :class:`~rlox.gym_vec_env.GymVecEnv`.

Observation and reward normalization is handled by :class:`~rlox.vec_normalize.VecNormalize`
at the environment boundary, not by the collector.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.batch import RolloutBatch
from rlox.gym_vec_env import GymVecEnv

# Environments with native Rust implementations
_NATIVE_ENV_IDS = frozenset({"CartPole-v1", "CartPole", "Pendulum-v1", "Pendulum"})


class RolloutCollector:
    """Collect on-policy rollout data from vectorized environments.

    Orchestrates the collect-then-compute pattern:
    1. Step ``n_envs`` environments for ``n_steps`` using the appropriate backend
    2. Compute GAE advantages per environment using ``rlox.compute_gae``
    3. Flatten and return a :class:`RolloutBatch`

    Normalization is no longer handled here. Wrap the environment with
    :class:`~rlox.vec_normalize.VecNormalize` to apply observation and
    reward normalization at the environment boundary.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID. Environments in ``_NATIVE_ENV_IDS``
        (CartPole-v1, Pendulum-v1) use the native Rust backend; all others use
        ``GymVecEnv`` (gymnasium SyncVectorEnv).
    n_envs : int
        Number of parallel environments.
    seed : int
        RNG seed for environment initialisation.
    device : str
        PyTorch device for output tensors.
    gamma : float
        Discount factor for GAE.
    gae_lambda : float
        Lambda parameter for GAE bias-variance tradeoff.
    reward_fn : Callable or None
        Optional reward shaping function ``(obs, actions, rewards) -> rewards``.
        Applied to raw rewards before storing.

    Example
    -------
    >>> from rlox.policies import DiscretePolicy
    >>> collector = RolloutCollector("CartPole-v1", n_envs=8, seed=0)
    >>> policy = DiscretePolicy(obs_dim=4, n_actions=2)
    >>> batch = collector.collect(policy, n_steps=128)
    >>> batch.obs.shape  # torch.Size([1024, 4])
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        seed: int = 0,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        | None = None,
        env: object | None = None,
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_fn = reward_fn

        # Allow caller to provide a pre-built env (e.g. VecNormalize wrapper).
        # When a wrapped env is supplied we MUST detect discreteness from the
        # wrapped env's action_space, not from env_id — wrappers like
        # VecNormalize forward a gym.spaces.Box even when env_id refers to a
        # natively-supported Rust env (e.g. Pendulum-v1).
        _used_rust_native = False
        if env is not None:
            self.env = env
        elif env_id in _NATIVE_ENV_IDS:
            try:
                self.env = rlox.VecEnv(n=n_envs, seed=seed, env_id=env_id)
                _used_rust_native = True
            except (ValueError, RuntimeError):
                # Rust side doesn't support this env yet; fall back to Gymnasium
                self.env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)
        else:
            self.env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)

        # Detect discrete vs continuous from the actual env's action_space.
        # Rust VecEnv exposes a dict {"type": "discrete"|"continuous", ...};
        # gymnasium / GymVecEnv / VecNormalize exposes a gym.spaces.Space.
        action_space_info = getattr(self.env, "action_space", None)
        if isinstance(action_space_info, dict):
            self._is_discrete = action_space_info.get("type") == "discrete"
        elif action_space_info is not None:
            import gymnasium as gym

            self._is_discrete = isinstance(action_space_info, gym.spaces.Discrete)
        elif _used_rust_native:
            # Older Rust builds expose no action_space property — fall back
            # to env_id heuristic, which historically only matched discrete
            # envs (CartPole).
            self._is_discrete = True
        else:
            self._is_discrete = True

        self._obs: np.ndarray | None = None  # (n_envs, obs_dim)

        # Episode statistics tracking
        self._ep_rewards = np.zeros(n_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(n_envs, dtype=np.int64)
        self._completed_rewards: list[float] = []
        self._completed_lengths: list[int] = []

    @property
    def episode_rewards(self) -> list[float]:
        """Rewards of all completed episodes since collector creation."""
        return self._completed_rewards

    @property
    def episode_lengths(self) -> list[int]:
        """Lengths of all completed episodes since collector creation."""
        return self._completed_lengths

    @torch.no_grad()
    def collect(self, policy: nn.Module, n_steps: int) -> RolloutBatch:
        """Run the policy for *n_steps* in each env and return a flat batch."""
        if self._obs is None:
            self._obs = self.env.reset_all()

        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []

        for _ in range(n_steps):
            obs_tensor = torch.as_tensor(
                self._obs, dtype=torch.float32, device=self.device
            )

            if hasattr(policy, "get_action_value"):
                actions, log_probs, values = policy.get_action_value(obs_tensor)
            else:
                actions, log_probs = policy.get_action_and_logprob(obs_tensor)
                values = policy.get_value(obs_tensor)

            # Step environment
            actions_np = actions.cpu().numpy()
            if self._is_discrete:
                step_result = self.env.step_all(actions_np.astype(np.uint32).tolist())
            else:
                step_result = self.env.step_all(actions_np.astype(np.float32))

            obs_np = self._obs  # pre-step observations for reward_fn
            raw_rewards = step_result["rewards"]
            if self.reward_fn is not None:
                raw_rewards = self.reward_fn(obs_np, actions_np, raw_rewards)

            # Obs from env is already normalized if wrapped with VecNormalize
            all_obs.append(obs_tensor)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)

            terminated = step_result["terminated"].astype(bool)
            truncated = step_result["truncated"].astype(bool)

            # Track episode statistics
            self._ep_rewards += raw_rewards
            self._ep_lengths += 1
            dones = terminated | truncated
            for i in range(self.n_envs):
                if dones[i]:
                    self._completed_rewards.append(float(self._ep_rewards[i]))
                    self._completed_lengths.append(int(self._ep_lengths[i]))
                    self._ep_rewards[i] = 0.0
                    self._ep_lengths[i] = 0

            # Truncation bootstrap: when truncated but not terminated,
            # add gamma * V(terminal_obs) to reward.
            # terminal_obs is already normalized by VecNormalize if wrapped.
            terminal_obs_list = step_result.get("terminal_obs")
            for i in range(self.n_envs):
                if truncated[i] and not terminated[i] and terminal_obs_list is not None:
                    term_obs = terminal_obs_list[i]
                    if term_obs is not None:
                        term_obs_t = torch.as_tensor(
                            term_obs, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        term_val = policy.get_value(term_obs_t).item()
                        raw_rewards[i] += self.gamma * term_val

            all_rewards.append(
                torch.as_tensor(raw_rewards.astype(np.float32), device=self.device)
            )
            # Pass only terminated (not dones) to GAE
            all_dones.append(
                torch.as_tensor(terminated.astype(np.float32), device=self.device)
            )

            # Next observations
            next_obs = step_result["obs"].copy()
            self._obs = next_obs

        # Bootstrap value for GAE
        last_obs_tensor = torch.as_tensor(
            self._obs, dtype=torch.float32, device=self.device
        )
        last_values = policy.get_value(last_obs_tensor)

        # Compute GAE in a single batched call (env-major flat layout)
        rewards_stacked = torch.stack(all_rewards)  # (n_steps, n_envs)
        values_stacked = torch.stack(all_values)  # (n_steps, n_envs)
        dones_stacked = torch.stack(all_dones)  # (n_steps, n_envs)

        # Transpose to (n_envs, n_steps) env-major, flatten contiguously
        rewards_flat = (
            rewards_stacked.T.contiguous().cpu().numpy().astype(np.float64).ravel()
        )
        values_flat = (
            values_stacked.T.contiguous().cpu().numpy().astype(np.float64).ravel()
        )
        dones_flat = (
            dones_stacked.T.contiguous().cpu().numpy().astype(np.float64).ravel()
        )
        last_vals = last_values.cpu().numpy().astype(np.float64)

        adv_flat, ret_flat = rlox.compute_gae_batched(
            rewards=rewards_flat,
            values=values_flat,
            dones=dones_flat,
            last_values=last_vals,
            n_steps=n_steps,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )

        # Reshape from env-major (n_envs, n_steps) back to (n_steps, n_envs)
        advantages_t = (
            torch.as_tensor(
                adv_flat,
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(self.n_envs, n_steps)
            .T
        )
        returns_t = (
            torch.as_tensor(
                ret_flat,
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(self.n_envs, n_steps)
            .T
        )

        # Stack: (n_steps, n_envs, ...) then flatten to (n_steps * n_envs, ...)
        obs_t = torch.stack(all_obs)  # (n_steps, n_envs, obs_dim)
        actions_t = torch.stack(
            all_actions
        )  # (n_steps, n_envs) or (n_steps, n_envs, act_dim)
        rewards_t = rewards_stacked  # already (n_steps, n_envs)
        dones_t = dones_stacked  # already (n_steps, n_envs)
        log_probs_t = torch.stack(all_log_probs)  # (n_steps, n_envs)
        values_t = values_stacked  # already (n_steps, n_envs)

        # Flatten (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        total = n_steps * self.n_envs
        return RolloutBatch(
            obs=obs_t.reshape(total, *obs_t.shape[2:]),
            actions=actions_t.reshape(total)
            if actions_t.dim() == 2
            else actions_t.reshape(total, -1),
            rewards=rewards_t.reshape(total),
            dones=dones_t.reshape(total),
            log_probs=log_probs_t.reshape(total),
            values=values_t.reshape(total),
            advantages=advantages_t.reshape(total),
            returns=returns_t.reshape(total),
        )
