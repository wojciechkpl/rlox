"""Diffusion Policy: action generation via iterative denoising.

Generates action sequences by learning to reverse a diffusion (noising)
process. Conditioned on observation history, a denoising network iteratively
refines random noise into a coherent action trajectory.

Reference:
    C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, S. Song,
    "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,"
    RSS, 2023.
    https://arxiv.org/abs/2303.04137
"""

from __future__ import annotations

import math
import sys
from typing import Any

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

from rlox.callbacks import Callback, CallbackList
from rlox.logging import LoggerCallback
from rlox.trainer import register_algorithm


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------


class NoiseSchedule:
    """Diffusion noise schedule (linear or cosine).

    Precomputes betas, alphas, and cumulative alpha products for T timesteps.

    Parameters
    ----------
    n_steps : int
        Number of diffusion timesteps T.
    schedule_type : str
        ``"linear"`` or ``"cosine"``.
    beta_start : float
        Starting beta for linear schedule.
    beta_end : float
        Ending beta for linear schedule.
    """

    def __init__(
        self,
        n_steps: int = 50,
        schedule_type: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        self.n_steps = n_steps

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_schedule(n_steps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type!r}")

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _cosine_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule from Nichol & Dhariwal (2021)."""
        steps = torch.arange(n_steps + 1, dtype=torch.float64)
        f = torch.cos((steps / n_steps + s) / (1 + s) * (math.pi / 2)) ** 2
        alpha_cumprod = f / f[0]
        betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
        return betas.clamp(0.0, 0.999).float()

    def to(self, device: torch.device) -> NoiseSchedule:
        """Move all tensors to the given device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        return self


# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------


class DenoisingMLP(nn.Module):
    """MLP-based denoising network.

    Predicts noise epsilon given (noisy_action, diffusion_timestep, obs_context).

    Parameters
    ----------
    obs_dim : int
        Per-timestep observation dimensionality.
    act_dim : int
        Per-timestep action dimensionality.
    obs_horizon : int
        Number of observation timesteps in the conditioning window.
    action_horizon : int
        Number of action timesteps to predict.
    hidden_dim : int
        Hidden layer width.
    n_diffusion_steps : int
        Maximum diffusion timestep T (for sinusoidal embedding).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        obs_horizon: int,
        action_horizon: int,
        hidden_dim: int = 256,
        n_diffusion_steps: int = 50,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # Sinusoidal timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Obs encoder: flatten obs_horizon * obs_dim -> hidden_dim
        obs_input_dim = obs_horizon * obs_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Noise predictor: takes (noisy_action_flat, time_embed, obs_embed)
        action_input_dim = action_horizon * act_dim
        input_dim = action_input_dim + hidden_dim + hidden_dim  # action + time + obs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_input_dim),
        )

        self._hidden_dim = hidden_dim

    def _sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional embedding for diffusion timesteps."""
        half_dim = self._hidden_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(-emb_scale * torch.arange(half_dim, device=timesteps.device))
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise.

        Parameters
        ----------
        noisy_action : (B, action_horizon, act_dim)
        timestep : (B,) integer diffusion timesteps
        obs : (B, obs_horizon, obs_dim)

        Returns
        -------
        eps_pred : (B, action_horizon, act_dim) predicted noise
        """
        batch_size = noisy_action.shape[0]

        # Flatten inputs
        action_flat = noisy_action.reshape(batch_size, -1)
        obs_flat = obs.reshape(batch_size, -1)

        # Embeddings
        t_emb = self._sinusoidal_embedding(timestep)
        t_emb = self.time_embed(t_emb)
        obs_emb = self.obs_encoder(obs_flat)

        # Concatenate and predict
        x = torch.cat([action_flat, t_emb, obs_emb], dim=-1)
        eps_pred = self.net(x)

        return eps_pred.reshape(batch_size, self.action_horizon, self.act_dim)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@register_algorithm("diffusion")
class DiffusionPolicy:
    """Policy as denoising diffusion -- generates action sequences via iterative denoising.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    n_diffusion_steps : int
        Number of diffusion timesteps T (default 50).
    action_horizon : int
        Number of future actions to predict (default 8).
    obs_horizon : int
        Number of past observations to condition on (default 2).
    hidden_dim : int
        Hidden layer width (default 256).
    learning_rate : float
        Adam learning rate (default 1e-4).
    batch_size : int
        Minibatch size (default 256).
    noise_schedule : str
        ``"cosine"`` or ``"linear"`` (default ``"cosine"``).
    beta_start : float
        Starting beta for linear schedule (default 0.0001).
    beta_end : float
        Ending beta for linear schedule (default 0.02).
    buffer_size : int
        Replay buffer capacity (default 1_000_000).
    n_inference_steps : int
        Denoising steps at inference (default 10).
    warmup_steps : int
        Data collection steps before training (default 500).
    seed : int
        Random seed (default 42).
    callbacks : list[Callback], optional
    logger : LoggerCallback, optional
    """

    def __init__(
        self,
        env_id: str,
        n_diffusion_steps: int = 50,
        action_horizon: int = 8,
        obs_horizon: int = 2,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        batch_size: int = 256,
        noise_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        buffer_size: int = 1_000_000,
        n_inference_steps: int = 10,
        warmup_steps: int = 500,
        seed: int = 42,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ) -> None:
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.seed = seed

        # Seed torch/np/env RNG for reproducibility.  This MUST happen
        # before the denoising network is constructed below so that its
        # weight initialisation is deterministic for a given seed
        # (precedent: pqn.py:118, crossq.py:141).  Gymnasium does not
        # propagate `env.reset(seed=...)` to the action space's RNG, and
        # `_collect_episodes` samples a random policy via
        # `action_space.sample()`, so the action space needs its own
        # `.seed()` call here too.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.action_space.seed(seed)

        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self._obs_dim = int(np.prod(obs_space.shape))
        self._act_dim = int(np.prod(act_space.shape))
        self._obs_horizon = obs_horizon
        self._action_horizon = action_horizon
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._warmup_steps = warmup_steps
        self._n_inference_steps = n_inference_steps

        # Noise schedule
        self.schedule = NoiseSchedule(
            n_steps=n_diffusion_steps,
            schedule_type=noise_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Denoising network
        self.model = DenoisingMLP(
            obs_dim=self._obs_dim,
            act_dim=self._act_dim,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
            n_diffusion_steps=n_diffusion_steps,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0
        self._n_diffusion_steps = n_diffusion_steps

        self._config = {
            "n_diffusion_steps": n_diffusion_steps,
            "action_horizon": action_horizon,
            "obs_horizon": obs_horizon,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "noise_schedule": noise_schedule,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "buffer_size": buffer_size,
            "n_inference_steps": n_inference_steps,
            "warmup_steps": warmup_steps,
            "seed": seed,
        }

    # -- Data collection -------------------------------------------------------

    def _collect_episodes(self, n_steps: int) -> list[dict[str, list]]:
        """Collect episodes using random policy."""
        episodes: list[dict[str, list]] = []
        obs, _ = self.env.reset(seed=self.seed)
        ep: dict[str, list] = {"states": [], "actions": []}

        for _ in range(n_steps):
            action = self.env.action_space.sample()
            ep["states"].append(np.asarray(obs, dtype=np.float32).flatten())
            ep["actions"].append(np.asarray(action, dtype=np.float32).flatten())

            next_obs, _reward, terminated, truncated, _ = self.env.step(action)
            obs = next_obs

            if terminated or truncated:
                episodes.append(ep)
                ep = {"states": [], "actions": []}
                obs, _ = self.env.reset()

        if ep["states"]:
            episodes.append(ep)
        return episodes

    def _sample_batch(
        self,
        episodes: list[dict[str, list]],
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of (obs_window, action_window) pairs.

        Returns
        -------
        obs_batch : (B, obs_horizon, obs_dim)
        action_batch : (B, action_horizon, act_dim)
        """
        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []

        for _ in range(self._batch_size):
            ep = episodes[rng.integers(len(episodes))]
            ep_len = len(ep["states"])
            # Need at least obs_horizon + action_horizon - 1 steps
            min_len = self._obs_horizon + self._action_horizon - 1
            if ep_len < min_len:
                # Pad short episodes
                ep = {
                    "states": ep["states"] + [ep["states"][-1]] * (min_len - ep_len),
                    "actions": ep["actions"] + [ep["actions"][-1]] * (min_len - ep_len),
                }
                ep_len = min_len

            # Random start for obs window
            max_start = ep_len - self._obs_horizon - self._action_horizon + 1
            start = rng.integers(max(1, max_start + 1))

            obs_window = np.stack(
                ep["states"][start : start + self._obs_horizon]
            )
            act_start = start + self._obs_horizon - 1
            act_window = np.stack(
                ep["actions"][act_start : act_start + self._action_horizon]
            )

            # Pad if needed
            if obs_window.shape[0] < self._obs_horizon:
                pad = self._obs_horizon - obs_window.shape[0]
                obs_window = np.pad(obs_window, ((0, pad), (0, 0)))
            if act_window.shape[0] < self._action_horizon:
                pad = self._action_horizon - act_window.shape[0]
                act_window = np.pad(act_window, ((0, pad), (0, 0)))

            obs_list.append(obs_window)
            act_list.append(act_window)

        obs_t = torch.as_tensor(np.stack(obs_list), dtype=torch.float32)
        act_t = torch.as_tensor(np.stack(act_list), dtype=torch.float32)
        return obs_t, act_t

    # -- Forward diffusion (add noise) -----------------------------------------

    def _forward_diffusion(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean actions at timestep t.

        Parameters
        ----------
        x_0 : (B, action_horizon, act_dim) clean actions
        t : (B,) integer timesteps
        noise : optional pre-sampled noise

        Returns
        -------
        x_t : noisy actions
        noise : the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_cumprod_t = self.schedule.alpha_cumprod[t]  # (B,)
        # Reshape for broadcasting: (B, 1, 1)
        alpha_cumprod_t = alpha_cumprod_t.reshape(-1, 1, 1)

        x_t = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1.0 - alpha_cumprod_t) * noise
        return x_t, noise

    # -- Reverse diffusion (denoise) -------------------------------------------

    def _reverse_diffusion(
        self,
        obs: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Run reverse diffusion from x_T to x_0.

        Parameters
        ----------
        obs : (B, obs_horizon, obs_dim)
        x_t : (B, action_horizon, act_dim) starting noisy actions

        Returns
        -------
        x_0 : (B, action_horizon, act_dim) denoised actions
        """
        self.model.eval()

        # Use evenly spaced subset of timesteps for inference
        if self._n_inference_steps < self._n_diffusion_steps:
            timesteps = torch.linspace(
                self._n_diffusion_steps - 1, 0, self._n_inference_steps
            ).long()
        else:
            timesteps = torch.arange(self._n_diffusion_steps - 1, -1, -1)

        batch_size = x_t.shape[0]

        for t_val in timesteps:
            t = torch.full((batch_size,), t_val.item(), dtype=torch.long)
            eps_pred = self.model(x_t, t, obs)

            alpha_t = self.schedule.alphas[t_val]
            alpha_cumprod_t = self.schedule.alpha_cumprod[t_val]

            # DDPM update step
            x_t = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t) * eps_pred
            )

            # Add noise for all steps except the last
            if t_val > 0:
                beta_t = self.schedule.betas[t_val]
                noise = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(beta_t) * noise

        self.model.train()
        return x_t

    # -- Training --------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Train the Diffusion Policy.

        1. Collect offline data via random policy.
        2. Train the denoising network with DDPM objective.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps for data collection.

        Returns
        -------
        dict with ``'loss'`` key.
        """
        episodes = self._collect_episodes(max(self._warmup_steps, total_timesteps))

        if not episodes:
            return {"loss": 0.0}

        rng = np.random.default_rng(self.seed)
        n_updates = max(1, total_timesteps // self._batch_size)
        total_loss = 0.0

        self.callbacks.on_training_start()

        for step in range(n_updates):
            obs_batch, action_batch = self._sample_batch(episodes, rng)

            # Sample random diffusion timesteps
            t = torch.randint(0, self._n_diffusion_steps, (obs_batch.shape[0],))

            # Forward diffusion: add noise
            noise = torch.randn_like(action_batch)
            x_t, _ = self._forward_diffusion(action_batch, t, noise)

            # Predict noise
            eps_pred = self.model(x_t, t, obs_batch)

            # DDPM loss: MSE between true noise and predicted noise
            loss = F.mse_loss(eps_pred, noise)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self._global_step += 1

            if self.logger is not None and self._global_step % 100 == 0:
                self.logger.on_train_step(self._global_step, {"loss": loss.item()})

        self.callbacks.on_training_end()

        return {"loss": total_loss / max(1, n_updates)}

    # -- Inference -------------------------------------------------------------

    def predict(
        self,
        obs: Any,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predict action(s) given observation(s).

        Parameters
        ----------
        obs : array-like, shape (obs_dim,) or (obs_horizon, obs_dim)
        deterministic : bool
            If True, uses fewer denoising steps for speed.

        Returns
        -------
        action : ndarray, shape (act_dim,)
        """
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32))

        # Ensure shape is (1, obs_horizon, obs_dim)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0).expand(self._obs_horizon, -1)
        if obs_t.ndim == 2:
            obs_t = obs_t.unsqueeze(0)

        # Start from noise
        x_t = torch.randn(1, self._action_horizon, self._act_dim)

        with torch.no_grad():
            actions = self._reverse_diffusion(obs_t, x_t)

        # Return the first action in the horizon
        return actions[0, 0].numpy()

    # -- Serialization ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Save checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self._global_step,
                "env_id": self.env_id,
                "config": self._config,
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> Self:
        """Restore from checkpoint."""
        from rlox.checkpoint import safe_torch_load

        data = safe_torch_load(path)
        eid = env_id or data.get("env_id", "Pendulum-v1")
        config = data["config"]

        dp = cls(env_id=eid, **config)
        dp.model.load_state_dict(data["model_state_dict"])
        dp.optimizer.load_state_dict(data["optimizer_state_dict"])
        dp._global_step = data.get("step", 0)
        return dp
