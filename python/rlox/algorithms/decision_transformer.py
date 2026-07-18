"""Decision Transformer: RL via sequence modeling.

Predicts actions from (return-to-go, state, action) history using a
causal transformer. Trained on offline data with supervised learning.

Reference:
    L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel,
    A. Srinivas, I. Mordatch,
    "Decision Transformer: Reinforcement Learning via Sequence Modeling,"
    NeurIPS, 2021.
    https://arxiv.org/abs/2106.01345
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

from rlox.callbacks import Callback, CallbackList
from rlox.logging import LoggerCallback
from rlox.trainer import register_algorithm


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single causal transformer block: MHA + MLP + LayerNorm (pre-norm)."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformerModel(nn.Module):
    """GPT-style transformer for (RTG, state, action) triples."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_actions: int,
        *,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        context_length: int = 20,
        dropout: float = 0.1,
        discrete: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.discrete = discrete

        # Token embeddings for the three modalities
        self.embed_rtg = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        if discrete:
            self.embed_action = nn.Embedding(n_actions, embed_dim)
        else:
            self.embed_action = nn.Linear(act_dim, embed_dim)

        # Learned positional encoding per timestep
        self.pos_embedding = nn.Embedding(context_length, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)

        # Prediction head: predict action from state token position
        if discrete:
            self.action_head = nn.Linear(embed_dim, n_actions)
        else:
            self.action_head = nn.Linear(embed_dim, act_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        states : (B, T, obs_dim)
        actions : (B, T) for discrete or (B, T, act_dim) for continuous
        returns_to_go : (B, T, 1)
        timesteps : (B, T)

        Returns
        -------
        action_preds : (B, T, n_actions) for discrete or (B, T, act_dim)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Clamp timesteps to context_length
        timesteps = timesteps.clamp(0, self.context_length - 1)
        pos = self.pos_embedding(timesteps)  # (B, T, E)

        # Embed each modality
        rtg_emb = self.embed_rtg(returns_to_go) + pos  # (B, T, E)
        state_emb = self.embed_state(states) + pos  # (B, T, E)
        if self.discrete:
            act_emb = self.embed_action(actions.long()) + pos  # (B, T, E)
        else:
            act_emb = self.embed_action(actions) + pos  # (B, T, E)

        # Interleave: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
        # Shape: (B, 3*T, E)
        tokens = torch.stack([rtg_emb, state_emb, act_emb], dim=2)
        tokens = tokens.reshape(batch_size, 3 * seq_len, self.embed_dim)
        tokens = self.drop(tokens)

        # Causal mask
        total_len = 3 * seq_len
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            tokens = block(tokens, attn_mask=causal_mask)

        tokens = self.ln_f(tokens)

        # Extract state token positions (index 1, 4, 7, ... = 3*t + 1)
        state_positions = torch.arange(1, 3 * seq_len, 3, device=tokens.device)
        state_tokens = tokens[:, state_positions, :]  # (B, T, E)

        return self.action_head(state_tokens)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@register_algorithm("dt")
class DecisionTransformer:
    """RL as sequence modeling -- predict actions from (RTG, state, action) history.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    context_length : int
        Number of timesteps in the context window (default 20).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of transformer layers (default 3).
    embed_dim : int
        Embedding dimension (default 128).
    learning_rate : float
        Adam learning rate (default 1e-4).
    batch_size : int
        Minibatch size for offline training (default 64).
    dropout : float
        Dropout rate (default 0.1).
    target_return : float
        Desired return for evaluation (default 200.0).
    warmup_steps : int
        Number of data collection steps before training (default 500).
    seed : int
        Random seed (default 42).
    callbacks : list[Callback], optional
    logger : LoggerCallback, optional
    """

    def __init__(
        self,
        env_id: str,
        context_length: int = 20,
        n_heads: int = 4,
        n_layers: int = 3,
        embed_dim: int = 128,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        dropout: float = 0.1,
        target_return: float = 200.0,
        warmup_steps: int = 500,
        seed: int = 42,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ) -> None:
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.seed = seed

        # Seed torch/np/env RNG for reproducibility.  This MUST happen
        # before the transformer model is constructed below so that its
        # weight initialisation is deterministic for a given seed
        # (precedent: pqn.py:118, crossq.py:141).  Gymnasium does not
        # propagate `env.reset(seed=...)` to the action space's RNG, and
        # `_collect_episodes` samples a random policy via
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
            self.n_actions = int(act_space.n)
            self.act_dim = 1
        else:
            self.n_actions = 0
            self.act_dim = int(np.prod(act_space.shape))

        self.context_length = context_length
        self.batch_size = batch_size
        self.target_return = target_return
        self.warmup_steps = warmup_steps
        self._learning_rate = learning_rate

        self.model = DecisionTransformerModel(
            obs_dim=obs_dim,
            act_dim=self.act_dim,
            n_actions=self.n_actions,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            context_length=context_length,
            dropout=dropout,
            discrete=self.discrete,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

        # Config for save/load
        self._config = {
            "context_length": context_length,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "embed_dim": embed_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout": dropout,
            "target_return": target_return,
            "warmup_steps": warmup_steps,
            "seed": seed,
        }

    # -- Data collection -------------------------------------------------------

    def _collect_episodes(self, n_steps: int) -> list[dict[str, list]]:
        """Collect episodes using random policy for offline data."""
        episodes: list[dict[str, list]] = []
        obs, _ = self.env.reset(seed=self.seed)
        ep: dict[str, list] = {"states": [], "actions": [], "rewards": []}

        for _ in range(n_steps):
            action = self.env.action_space.sample()
            ep["states"].append(np.asarray(obs, dtype=np.float32))
            ep["actions"].append(action)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            ep["rewards"].append(float(reward))
            obs = next_obs

            if terminated or truncated:
                episodes.append(ep)
                ep = {"states": [], "actions": [], "rewards": []}
                obs, _ = self.env.reset()

        if ep["states"]:
            episodes.append(ep)
        return episodes

    @staticmethod
    def _compute_rtg(rewards: list[float], gamma: float = 1.0) -> list[float]:
        """Compute return-to-go for an episode."""
        rtg = [0.0] * len(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            rtg[t] = running
        return rtg

    def _sample_batch(
        self,
        episodes: list[dict[str, list]],
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of subsequences from episodes."""
        states_batch = []
        actions_batch = []
        rtg_batch = []
        timesteps_batch = []

        for _ in range(self.batch_size):
            ep = episodes[rng.integers(len(episodes))]
            ep_len = len(ep["states"])
            rtg = self._compute_rtg(ep["rewards"])

            # Random start index
            start = rng.integers(max(1, ep_len - self.context_length + 1))
            end = min(start + self.context_length, ep_len)
            sl = end - start

            s = np.stack(ep["states"][start:end])
            a = np.array(ep["actions"][start:end])
            r = np.array(rtg[start:end])
            t = np.arange(start, end)

            # Pad to context_length
            pad_len = self.context_length - sl
            if pad_len > 0:
                s = np.pad(s, ((0, pad_len), (0, 0)))
                a = np.pad(a, (0, pad_len))
                r = np.pad(r, (0, pad_len))
                t = np.pad(t, (0, pad_len))

            states_batch.append(s)
            actions_batch.append(a)
            rtg_batch.append(r)
            timesteps_batch.append(t)

        states_t = torch.as_tensor(np.stack(states_batch), dtype=torch.float32)
        actions_t = torch.as_tensor(np.stack(actions_batch), dtype=torch.long if self.discrete else torch.float32)
        rtg_t = torch.as_tensor(np.stack(rtg_batch), dtype=torch.float32).unsqueeze(-1)
        timesteps_t = torch.as_tensor(np.stack(timesteps_batch), dtype=torch.long)

        return states_t, actions_t, rtg_t, timesteps_t

    # -- Training --------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Train the Decision Transformer.

        1. Collect offline data via random policy.
        2. Train the transformer to predict actions given (RTG, state, action).

        Parameters
        ----------
        total_timesteps : int
            Total environment steps for data collection.

        Returns
        -------
        dict with 'loss' key.
        """
        episodes = self._collect_episodes(max(self.warmup_steps, total_timesteps))

        if not episodes:
            return {"loss": 0.0}

        rng = np.random.default_rng(self.seed)
        n_updates = max(1, total_timesteps // self.batch_size)
        total_loss = 0.0

        self.callbacks.on_training_start()

        for step in range(n_updates):
            states, actions, rtg, timesteps = self._sample_batch(episodes, rng)
            action_preds = self.model(states, actions, rtg, timesteps)

            if self.discrete:
                loss = F.cross_entropy(
                    action_preds.reshape(-1, self.n_actions),
                    actions.reshape(-1).long(),
                )
            else:
                loss = F.mse_loss(action_preds, actions)

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
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        deterministic: bool = True,
    ) -> int | np.ndarray:
        """Predict the next action given context.

        Parameters
        ----------
        states : (1, T, obs_dim) tensor
        actions : (1, T) or (1, T, act_dim) tensor
        returns_to_go : (1, T, 1) tensor
        timesteps : (1, T) tensor

        Returns
        -------
        action : int (discrete) or ndarray (continuous)
        """
        self.model.eval()
        with torch.no_grad():
            action_preds = self.model(states, actions, returns_to_go, timesteps)
            # Take the last timestep prediction
            last_pred = action_preds[:, -1, :]

            if self.discrete:
                if deterministic:
                    action = last_pred.argmax(dim=-1).item()
                else:
                    action = torch.distributions.Categorical(logits=last_pred).sample().item()
            else:
                action = last_pred.squeeze(0).numpy()

        self.model.train()
        return action

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
        eid = env_id or data.get("env_id", "CartPole-v1")
        config = data["config"]

        dt = cls(env_id=eid, **config)
        dt.model.load_state_dict(data["model_state_dict"])
        dt.optimizer.load_state_dict(data["optimizer_state_dict"])
        dt._global_step = data.get("step", 0)
        return dt
