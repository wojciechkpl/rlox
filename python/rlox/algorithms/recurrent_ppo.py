"""Recurrent PPO: LSTM actor-critic with truncated BPTT over episode segments.

A self-contained variant of PPO (Schulman et al., 2017) for partially-
observable / memory-dependent tasks. The actor-critic shares an
``encoder -> LSTM`` trunk; the LSTM carries a per-environment recurrent
state ``(h, c)`` across timesteps and resets to zero whenever an episode
boundary (``done``) is crossed.

Design summary
---------------
- **Collection**: the per-env ``(h, c)`` state is carried across the whole
  rollout (and across rollouts, detached between updates). Each step
  records whether the *current* observation is the first of a (possibly
  new) episode via ``episode_starts`` — the shifted-by-one form of
  ``done`` that is needed to know when to reset the *incoming* hidden
  state, as distinct from ``dones`` (``terminated``-only, fed to GAE).
- **BPTT reconstruction**: after a rollout, each environment's timeline is
  split into per-episode *segments* (:func:`_split_into_segments`) so that
  no segment straddles a ``done``. Segments are padded to a common length
  per minibatch (:func:`_pad_segments`) and the LSTM is re-run once per
  segment from the correct seed state (zero for a fresh episode, or the
  rollout's carried initial state for a segment that continues an episode
  from a previous rollout). A boolean mask excludes padded steps from the
  loss (:class:`_RecurrentPPOLoss`), so gradients never flow across a
  ``done`` boundary and padding never influences the loss value.
- **Loss**: the same clipped-surrogate + (optionally clipped) value loss +
  entropy bonus as :class:`rlox.losses.PPOLoss`, adapted to the masked,
  variable-length-sequence setting.

Discrete action spaces only (see ``RecurrentPPO.__init__``); continuous
action spaces raise ``ValueError``.

PPO Reference:
    J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov,
    "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

Recurrent-PPO reference implementations consulted for the BPTT design:
    CleanRL's ``ppo_atari_lstm.py`` (per-timestep reset-masking) and
    sb3-contrib's ``RecurrentPPO`` (episode-segment padding + masking).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.gym_vec_env import GymVecEnv
from rlox.logging import LoggerCallback
from rlox.utils import detect_env_spaces

# Environments with a native Rust VecEnv implementation. Deliberately a
# private, per-file constant — see PROJECT_QUICK_REFERENCE.md's note on why
# this set is *not* shared/imported across algorithm files.
_RECURRENT_PPO_NATIVE_ENVS = frozenset({"CartPole-v1", "CartPole"})


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_min(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialisation to a Linear layer (rlox convention)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RecurrentPPOConfig:
    """Configuration for :class:`RecurrentPPO`.

    Defined locally (not in ``rlox.config``) per the self-contained-module
    constraint for this experimental algorithm.

    Attributes
    ----------
    n_envs : int
        Number of parallel environments (default 8).
    n_steps : int
        Rollout length per environment per update (default 64). Also the
        hard cap on BPTT depth, since no segment can exceed the rollout
        window.
    n_epochs : int
        Number of SGD passes over each rollout (default 4).
    n_minibatches : int
        Number of minibatches per epoch, formed by randomly partitioning
        the rollout's episode *segments* (not raw transitions — the BPTT
        unit here is a segment). Clamped to the number of segments
        available if fewer segments than minibatches are produced.
    learning_rate : float
        Adam learning rate (default 2.5e-4).
    clip_eps : float
        PPO clipping range for the probability ratio (default 0.2).
    vf_coef : float
        Value loss coefficient (default 0.5).
    ent_coef : float
        Entropy bonus coefficient (default 0.01).
    max_grad_norm : float
        Maximum gradient norm for clipping (default 0.5).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda (default 0.95).
    normalize_advantages : bool
        Whether to normalise advantages over the valid (non-padded)
        entries of each minibatch (default True).
    clip_vloss : bool
        Whether to clip the value loss (CleanRL max-of-clipped
        formulation, matching :class:`rlox.losses.PPOLoss`; default True).
    anneal_lr : bool
        Whether to linearly anneal the learning rate to zero (default True).
    hidden : int
        Width of the ``Linear -> Tanh`` observation encoder (default 64).
    lstm_hidden : int
        LSTM hidden size (default 64).
    """

    n_envs: int = 8
    n_steps: int = 64
    n_epochs: int = 4
    n_minibatches: int = 4
    learning_rate: float = 2.5e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    clip_vloss: bool = True
    anneal_lr: bool = True
    hidden: int = 64
    lstm_hidden: int = 64

    def __post_init__(self) -> None:
        _validate_positive("learning_rate", self.learning_rate)
        _validate_min("n_envs", self.n_envs, 1)
        _validate_min("n_steps", self.n_steps, 1)
        _validate_min("n_epochs", self.n_epochs, 1)
        _validate_min("n_minibatches", self.n_minibatches, 1)
        _validate_min("hidden", self.hidden, 1)
        _validate_min("lstm_hidden", self.lstm_hidden, 1)


# ---------------------------------------------------------------------------
# LSTM actor-critic
# ---------------------------------------------------------------------------


class LSTMActorCritic(nn.Module):
    """Shared-trunk LSTM actor-critic for discrete action spaces.

    ``obs -> Linear+Tanh encoder -> LSTM(h, c) -> {actor head, critic head}``

    The recurrent state ``(h, c)`` follows ``nn.LSTM``'s native shape
    convention ``(num_layers=1, batch, lstm_hidden)`` throughout, so it can
    be threaded directly in and out of ``nn.LSTM`` without reshaping.

    Parameters
    ----------
    obs_dim : int
        Observation space dimensionality.
    n_actions : int
        Number of discrete actions.
    hidden : int
        Width of the observation encoder (default 64).
    lstm_hidden : int
        LSTM hidden size (default 64).
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden: int = 64, lstm_hidden: int = 64
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.lstm_hidden = lstm_hidden

        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.lstm = nn.LSTM(hidden, lstm_hidden)
        self.actor = nn.Linear(lstm_hidden, n_actions)
        self.critic = nn.Linear(lstm_hidden, 1)

        _orthogonal_init(self.encoder[0], gain=float(np.sqrt(2)))
        _orthogonal_init(self.actor, gain=0.01)
        _orthogonal_init(self.critic, gain=1.0)

    def initial_state(
        self, n_envs: int, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a zeroed ``(h, c)`` state of shape ``(1, n_envs, lstm_hidden)``."""
        h = torch.zeros(1, n_envs, self.lstm_hidden, device=device)
        c = torch.zeros(1, n_envs, self.lstm_hidden, device=device)
        return h, c

    def get_states(
        self,
        obs: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM over a ``(T, B, obs_dim)`` sequence.

        Before processing timestep ``t``, the incoming recurrent state is
        multiplied by ``(1 - episode_starts[t])``. Where
        ``episode_starts[t, b] == 1`` this exactly zeroes the state seen by
        that step for env ``b`` — the standard truncated-BPTT reset trick
        (CleanRL / sb3-contrib): because the local Jacobian of ``0 * x``
        w.r.t. ``x`` is zero, no gradient flows back through a reset, which
        is what makes episode boundaries safe for BPTT.

        Parameters
        ----------
        obs : Tensor (T, B, obs_dim)
        lstm_state : tuple (h, c), each (1, B, lstm_hidden)
        episode_starts : Tensor (T, B) of 0/1 floats

        Returns
        -------
        features : Tensor (T, B, lstm_hidden)
        new_lstm_state : tuple (h, c), each (1, B, lstm_hidden)
        """
        h, c = lstm_state
        encoded = self.encoder(obs)  # (T, B, hidden) — pointwise per timestep
        t_len = obs.shape[0]
        outputs: list[torch.Tensor] = []
        for t in range(t_len):
            mask = (1.0 - episode_starts[t]).view(1, -1, 1)
            h = h * mask
            c = c * mask
            step_out, (h, c) = self.lstm(encoded[t : t + 1], (h, c))
            outputs.append(step_out)
        features = torch.cat(outputs, dim=0)
        return features, (h, c)

    def act(
        self,
        obs: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]
    ]:
        """Sample (or greedily pick) actions for a ``(T, B, obs_dim)`` batch.

        Returns
        -------
        action, log_prob, value : each (T, B)
        new_lstm_state : tuple (h, c), each (1, B, lstm_hidden)
        """
        features, new_state = self.get_states(obs, lstm_state, episode_starts)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, value, new_state

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-prob / entropy / value for given actions (for the PPO loss).

        Returns
        -------
        log_prob, entropy, value : each (T, B)
        """
        features, _ = self.get_states(obs, lstm_state, episode_starts)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout storage, episode-segment reconstruction, and padding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Rollout:
    """Flat ``(n_steps, n_envs, ...)`` rollout data plus BPTT bookkeeping.

    Attributes
    ----------
    obs, actions, log_probs, values, advantages, returns : Tensor (T, n_envs, ...)
    episode_starts : Tensor (T, n_envs)
        Whether ``obs[t, e]`` is the first observation of a (possibly new)
        episode, i.e. the shifted-by-one form of ``done``. Distinct from
        the ``terminated``-only flags used for GAE.
    initial_lstm_state : tuple (h, c), each (1, n_envs, lstm_hidden)
        The recurrent state carried into this rollout, i.e. the state at
        the very start of collection — needed to correctly seed the first
        segment of any environment whose episode was already in progress
        when this rollout began.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_starts: torch.Tensor
    initial_lstm_state: tuple[torch.Tensor, torch.Tensor]


class _Segment(NamedTuple):
    """A contiguous, done-free span ``[start, end]`` (inclusive) for one env."""

    env_idx: int
    start: int
    end: int


def _split_into_segments(episode_starts: np.ndarray) -> list[_Segment]:
    """Split a ``(T, n_envs)`` episode_starts array into per-episode segments.

    A new segment begins at any ``t > 0`` where ``episode_starts[t, e]`` is
    truthy (an episode boundary), so no segment straddles a ``done``. The
    very first segment of each env (``start == 0``) may or may not itself
    begin at a boundary — that is resolved later by :func:`_pad_segments`,
    which seeds it from either zero or the rollout's carried initial state.

    Parameters
    ----------
    episode_starts : np.ndarray of shape (T, n_envs)

    Returns
    -------
    list[_Segment]
        Unordered across envs; each env's own segments are contiguous and
        cover ``[0, T - 1]`` exactly once.
    """
    t_len, n_envs = episode_starts.shape
    segments: list[_Segment] = []
    for env_idx in range(n_envs):
        seg_start = 0
        for t in range(1, t_len):
            if episode_starts[t, env_idx]:
                segments.append(_Segment(env_idx, seg_start, t - 1))
                seg_start = t
        segments.append(_Segment(env_idx, seg_start, t_len - 1))
    return segments


@dataclass(frozen=True, slots=True)
class _PaddedBatch:
    """A minibatch of zero-padded, variable-length episode segments.

    All fields except ``lstm_state`` have leading shape ``(max_len, B)``
    (or ``(max_len, B, obs_dim)`` for ``obs``), where ``B`` is the number
    of segments in this minibatch. ``mask[t, b] == 1`` iff position ``t``
    is a real (non-padded) step of segment ``b``.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    mask: torch.Tensor
    episode_starts: torch.Tensor
    lstm_state: tuple[torch.Tensor, torch.Tensor]


def _pad_segments(
    segments: list[_Segment], rollout: _Rollout, lstm_hidden: int
) -> _PaddedBatch:
    """Gather *segments* out of *rollout* into a single zero-padded batch.

    Each segment's initial ``(h, c)`` is the rollout's carried
    ``initial_lstm_state`` if the segment starts at a non-boundary ``t=0``
    (episode already in progress at rollout start), otherwise zero — which
    is exactly what ``episode_starts[seg.start, seg.env_idx]`` encodes.
    Because no segment straddles a ``done``, the per-segment
    ``episode_starts`` passed to :meth:`LSTMActorCritic.get_states` is all
    zeros — the reset is already baked into the choice of initial state.
    """
    n_segments = len(segments)
    lengths = [seg.end - seg.start + 1 for seg in segments]
    max_len = max(lengths)
    obs_dim = rollout.obs.shape[-1]
    device = rollout.obs.device

    obs = torch.zeros(max_len, n_segments, obs_dim, device=device)
    actions = torch.zeros(max_len, n_segments, dtype=torch.long, device=device)
    log_probs = torch.zeros(max_len, n_segments, device=device)
    values = torch.zeros(max_len, n_segments, device=device)
    advantages = torch.zeros(max_len, n_segments, device=device)
    returns = torch.zeros(max_len, n_segments, device=device)
    mask = torch.zeros(max_len, n_segments, device=device)
    h0 = torch.zeros(1, n_segments, lstm_hidden, device=device)
    c0 = torch.zeros(1, n_segments, lstm_hidden, device=device)

    init_h, init_c = rollout.initial_lstm_state
    for i, seg in enumerate(segments):
        length = lengths[i]
        s, e, env = seg.start, seg.end, seg.env_idx
        obs[:length, i] = rollout.obs[s : e + 1, env]
        actions[:length, i] = rollout.actions[s : e + 1, env]
        log_probs[:length, i] = rollout.log_probs[s : e + 1, env]
        values[:length, i] = rollout.values[s : e + 1, env]
        advantages[:length, i] = rollout.advantages[s : e + 1, env]
        returns[:length, i] = rollout.returns[s : e + 1, env]
        mask[:length, i] = 1.0
        if not rollout.episode_starts[s, env]:
            h0[0, i] = init_h[0, env]
            c0[0, i] = init_c[0, env]

    episode_starts = torch.zeros(max_len, n_segments, device=device)
    return _PaddedBatch(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        values=values,
        advantages=advantages,
        returns=returns,
        mask=mask,
        episode_starts=episode_starts,
        lstm_state=(h0, c0),
    )


# ---------------------------------------------------------------------------
# Masked PPO loss
# ---------------------------------------------------------------------------


class _RecurrentPPOLoss:
    """Clipped PPO objective restricted to the valid (non-padded) entries
    of a :class:`_PaddedBatch`.

    Mirrors :class:`rlox.losses.PPOLoss` (same clipped surrogate, same
    ``0.5 * MSE`` inner factor for the value loss, same ``clip_vloss``
    max-of-clipped formulation) but every term is masked and averaged only
    over real (non-padded) steps.
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        clip_vloss: bool = True,
    ) -> None:
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_vloss = clip_vloss

    def __call__(
        self,
        policy: LSTMActorCritic,
        batch: _PaddedBatch,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        new_log_probs, entropy, new_values = policy.evaluate_actions(
            batch.obs, batch.lstm_state, batch.episode_starts, batch.actions
        )

        mask = batch.mask
        n_valid = mask.sum().clamp(min=1.0)

        log_ratio = new_log_probs - batch.log_probs
        ratio = log_ratio.exp()
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
        )
        policy_loss = (torch.max(pg_loss1, pg_loss2) * mask).sum() / n_valid

        if self.clip_vloss:
            v_clipped = batch.values + torch.clamp(
                new_values - batch.values, -self.clip_eps, self.clip_eps
            )
            vf_loss1 = (new_values - batch.returns) ** 2
            vf_loss2 = (v_clipped - batch.returns) ** 2
            value_loss = 0.5 * (torch.max(vf_loss1, vf_loss2) * mask).sum() / n_valid
        else:
            value_loss = (
                0.5 * (((new_values - batch.returns) ** 2) * mask).sum() / n_valid
            )

        entropy_loss = (entropy * mask).sum() / n_valid

        total_loss = (
            policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
        )

        with torch.no_grad():
            approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / n_valid
            clip_fraction = (
                ((ratio - 1.0).abs() > self.clip_eps).float() * mask
            ).sum() / n_valid

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# RecurrentPPO
# ---------------------------------------------------------------------------


class RecurrentPPO:
    """PPO with an LSTM actor-critic and truncated BPTT over episode segments.

    Discrete action spaces only — continuous action spaces raise
    ``ValueError`` at construction time.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID with a discrete action space (default
        ``"CartPole-v1"``).
    n_envs : int
        Number of parallel environments (default 8).
    seed : int
        Random seed for torch, numpy, and the environment (default 42).
    logger : LoggerCallback, optional
        Logger for metrics.
    callbacks : list[Callback], optional
        Training callbacks.
    **config_kwargs
        Override any :class:`RecurrentPPOConfig` field (e.g. ``n_steps``,
        ``hidden``, ``lstm_hidden``, ``learning_rate``).
    """

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        n_envs: int = 8,
        seed: int = 42,
        logger: LoggerCallback | None = None,
        callbacks: list[Callback] | None = None,
        **config_kwargs: object,
    ) -> None:
        # Reproducibility: seed torch/np before any weight init or env build.
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env_id = env_id
        self.seed = seed
        self.config = RecurrentPPOConfig(n_envs=n_envs, **config_kwargs)  # type: ignore[arg-type]

        obs_dim, action_space, is_discrete = detect_env_spaces(env_id)
        if not is_discrete:
            raise ValueError(
                f"RecurrentPPO only supports discrete action spaces for this "
                f"first version, got a continuous action space for "
                f"env_id={env_id!r}. Pass a discrete-action env (e.g. "
                f"'CartPole-v1')."
            )
        n_actions = int(action_space.n)
        self._obs_dim = obs_dim
        self._n_actions = n_actions

        self.policy = LSTMActorCritic(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden=self.config.hidden,
            lstm_hidden=self.config.lstm_hidden,
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5
        )
        self.loss_fn = _RecurrentPPOLoss(
            clip_eps=self.config.clip_eps,
            vf_coef=self.config.vf_coef,
            ent_coef=self.config.ent_coef,
            clip_vloss=self.config.clip_vloss,
        )

        if env_id in _RECURRENT_PPO_NATIVE_ENVS:
            try:
                self.env = rlox.VecEnv(n=self.config.n_envs, seed=seed, env_id=env_id)
            except (ValueError, RuntimeError):
                self.env = GymVecEnv(env_id, n_envs=self.config.n_envs, seed=seed)
        else:
            self.env = GymVecEnv(env_id, n_envs=self.config.n_envs, seed=seed)

        # Training-rollout recurrent state, carried across collect() calls.
        self._lstm_state = self.policy.initial_state(self.config.n_envs)
        self._episode_starts = np.ones(self.config.n_envs, dtype=np.float32)
        self._obs = self.env.reset_all()

        # Episode statistics (genuine per-episode returns, not the coarse
        # per-rollout windowed sum).
        self._ep_rewards = np.zeros(self.config.n_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(self.config.n_envs, dtype=np.int64)
        self._completed_rewards: list[float] = []
        self._completed_lengths: list[int] = []

        # Separate single-trajectory recurrent state for predict().
        self._predict_lstm_state = self.policy.initial_state(1)
        self._predict_episode_start = True

        self.logger = logger
        self.callbacks = CallbackList(callbacks)
        self._global_step = 0

    @property
    def episode_rewards(self) -> list[float]:
        """Rewards of all completed episodes since construction."""
        return self._completed_rewards

    @property
    def episode_lengths(self) -> list[int]:
        """Lengths of all completed episodes since construction."""
        return self._completed_lengths

    def _recent_mean_reward(self, window: int = 100) -> float:
        if self._completed_rewards:
            return float(np.mean(self._completed_rewards[-window:]))
        return 0.0

    @torch.no_grad()
    def _collect_rollout(self, n_steps: int) -> _Rollout:
        """Step the env(s) for *n_steps*, carrying the LSTM state across
        timesteps and resetting it at episode boundaries."""
        n_envs = self.config.n_envs
        initial_lstm_state = (
            self._lstm_state[0].clone(),
            self._lstm_state[1].clone(),
        )

        all_obs: list[torch.Tensor] = []
        all_actions: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_terminated: list[torch.Tensor] = []
        all_episode_starts: list[torch.Tensor] = []

        for _ in range(n_steps):
            obs_np = self._obs
            episode_starts_np = self._episode_starts.copy()

            obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            es_t = torch.as_tensor(episode_starts_np, dtype=torch.float32).unsqueeze(0)

            action, log_prob, value, new_state = self.policy.act(
                obs_t, self._lstm_state, es_t
            )
            self._lstm_state = new_state

            action_np = action.squeeze(0).cpu().numpy()
            step_result = self.env.step_all(action_np.astype(np.uint32).tolist())

            terminated = step_result["terminated"].astype(bool)
            truncated = step_result["truncated"].astype(bool)
            done = terminated | truncated
            rewards = step_result["rewards"]

            self._ep_rewards += rewards
            self._ep_lengths += 1
            for i in range(n_envs):
                if done[i]:
                    self._completed_rewards.append(float(self._ep_rewards[i]))
                    self._completed_lengths.append(int(self._ep_lengths[i]))
                    self._ep_rewards[i] = 0.0
                    self._ep_lengths[i] = 0

            all_obs.append(obs_t.squeeze(0))
            all_actions.append(action.squeeze(0))
            all_log_probs.append(log_prob.squeeze(0))
            all_values.append(value.squeeze(0))
            all_rewards.append(torch.as_tensor(rewards.astype(np.float32)))
            all_terminated.append(torch.as_tensor(terminated.astype(np.float32)))
            all_episode_starts.append(
                torch.as_tensor(episode_starts_np, dtype=torch.float32)
            )

            self._obs = step_result["obs"].copy()
            self._episode_starts = done.astype(np.float32)

        # Bootstrap value for GAE, using the carried (correctly reset) state.
        obs_t = torch.as_tensor(self._obs, dtype=torch.float32).unsqueeze(0)
        es_t = torch.as_tensor(self._episode_starts, dtype=torch.float32).unsqueeze(0)
        _, _, last_value, _ = self.policy.act(obs_t, self._lstm_state, es_t)
        last_values = last_value.squeeze(0)

        # Detach the persisted state so gradients never cross a rollout
        # boundary either (defensive — already under torch.no_grad()).
        self._lstm_state = (self._lstm_state[0].detach(), self._lstm_state[1].detach())

        obs = torch.stack(all_obs)
        actions = torch.stack(all_actions)
        log_probs = torch.stack(all_log_probs)
        values = torch.stack(all_values)
        rewards = torch.stack(all_rewards)
        terminated_t = torch.stack(all_terminated)
        episode_starts = torch.stack(all_episode_starts)

        # GAE via the Rust kernel — env-major flat layout, matching
        # rlox.collectors.RolloutCollector.
        rewards_flat = rewards.T.contiguous().numpy().astype(np.float64).ravel()
        values_flat = values.T.contiguous().numpy().astype(np.float64).ravel()
        dones_flat = terminated_t.T.contiguous().numpy().astype(np.float64).ravel()
        last_vals = last_values.numpy().astype(np.float64)

        adv_flat, ret_flat = rlox.compute_gae_batched(
            rewards=rewards_flat,
            values=values_flat,
            dones=dones_flat,
            last_values=last_vals,
            n_steps=n_steps,
            gamma=self.config.gamma,
            lam=self.config.gae_lambda,
        )
        advantages = (
            torch.as_tensor(adv_flat, dtype=torch.float32).reshape(n_envs, n_steps).T
        )
        returns = (
            torch.as_tensor(ret_flat, dtype=torch.float32).reshape(n_envs, n_steps).T
        )

        return _Rollout(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
            episode_starts=episode_starts,
            initial_lstm_state=initial_lstm_state,
        )

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run recurrent PPO training and return final metrics."""
        cfg = self.config
        steps_per_rollout = cfg.n_envs * cfg.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        last_metrics: dict[str, float] = {}
        self.callbacks.on_training_start()

        for update in range(n_updates):
            if cfg.anneal_lr:
                frac = 1.0 - update / n_updates
                lr = cfg.learning_rate * frac
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            rollout = self._collect_rollout(cfg.n_steps)
            segments = _split_into_segments(rollout.episode_starts.numpy())
            mean_ep_reward = self._recent_mean_reward()

            self.callbacks.on_rollout_end(mean_reward=mean_ep_reward, update=update)

            for _epoch in range(cfg.n_epochs):
                perm = np.random.permutation(len(segments))
                n_minibatches = min(cfg.n_minibatches, len(segments))
                for mb_idx in np.array_split(perm, n_minibatches):
                    if len(mb_idx) == 0:
                        continue
                    mb_segments = [segments[i] for i in mb_idx]
                    batch = _pad_segments(
                        mb_segments, rollout, lstm_hidden=cfg.lstm_hidden
                    )

                    adv = batch.advantages
                    if cfg.normalize_advantages:
                        valid = batch.mask.bool()
                        adv = (adv - adv[valid].mean()) / (adv[valid].std() + 1e-8)

                    loss, metrics = self.loss_fn(self.policy, batch, adv)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    last_metrics = metrics

                    self._global_step += 1
                    self.callbacks.on_train_batch(loss=loss.item(), **last_metrics)

            should_continue = self.callbacks.on_step(
                reward=mean_ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

            if self.logger is not None:
                self.logger.on_train_step(
                    update, {**last_metrics, "mean_reward": mean_ep_reward}
                )

        self.callbacks.on_training_end()
        last_metrics["mean_reward"] = self._recent_mean_reward()
        return last_metrics

    def reset_predict_state(self) -> None:
        """Zero the recurrent state used by :meth:`predict`.

        Call this at the start of a new evaluation episode/trajectory —
        ``predict()`` otherwise carries its hidden state across calls.
        """
        self._predict_lstm_state = self.policy.initial_state(1)
        self._predict_episode_start = True

    def predict(
        self, obs: np.ndarray | torch.Tensor, deterministic: bool = True
    ) -> int:
        """Return an action for a single observation.

        Maintains a dedicated single-trajectory recurrent state across
        calls, so callers must invoke :meth:`reset_predict_state` at the
        start of each new episode.

        Parameters
        ----------
        obs : np.ndarray or torch.Tensor
            A single observation of shape ``(obs_dim,)``.
        deterministic : bool
            If True, return the argmax action; otherwise sample.
        """
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        else:
            obs_t = obs.float()
        obs_t = obs_t.reshape(1, 1, -1)  # (T=1, B=1, obs_dim)
        es_t = torch.as_tensor(
            [[1.0 if self._predict_episode_start else 0.0]], dtype=torch.float32
        )

        with torch.no_grad():
            action, _, _, new_state = self.policy.act(
                obs_t, self._predict_lstm_state, es_t, deterministic=deterministic
            )
        self._predict_lstm_state = new_state
        self._predict_episode_start = False
        return int(action.item())
