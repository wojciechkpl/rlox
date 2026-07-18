"""Truncated Quantile Critics (TQC).

Kuznetsov et al., "Controlling Overestimation Bias with Truncated Mixture of
Continuous Distributional Quantile Critics", ICML 2020, arXiv:2005.04269.

TQC is SAC with **distributional critics**: instead of two scalar-valued
Q-networks reduced via ``min`` (SAC/TD3's overestimation-control trick), TQC
uses an ensemble of ``n_critics`` quantile critics, each predicting
``n_quantiles`` quantiles of the state-action return distribution via a
``QuantileQNetwork`` (an MLP mapping ``(obs, action) -> n_quantiles``).

Overestimation control works differently here: the next-state quantiles from
ALL critics (evaluated via a target ensemble, soft-updated exactly like
SAC's target critics) are pooled into one set of ``n_critics * n_quantiles``
values per transition, sorted, and the top
``top_quantiles_to_drop_per_net * n_critics`` (largest / most optimistic)
are DROPPED before bootstrapping. This truncation is a continuous, tunable
dial on pessimism -- unlike SAC's ``min`` of two point estimates, which is a
fixed, coarse-grained choice that discards the diversity benefit of a larger
ensemble.

Critic loss: the pairwise quantile Huber (pinball) loss -- the standard
QR-DQN loss -- between each critic's ``n_quantiles`` predicted quantiles and
the SAME shared truncated target set (every critic regresses toward the same
pooled-and-truncated targets).

Actor loss and automatic entropy tuning are unchanged from SAC: the actor
maximises the *un-truncated* ensemble mean (mean over all
``n_critics * n_quantiles`` quantiles, evaluated with the LIVE critics) minus
``alpha * log_pi``.

Reuses ``rlox.networks.SquashedGaussianPolicy``, ``rlox.ReplayBuffer``, and
SAC's automatic entropy-coefficient tuning unchanged. Continuous (Box)
action spaces only.

Self-contained by design: ``TQCConfig``, ``QuantileQNetwork``, and the
quantile-Huber loss are defined locally in this module rather than in
``rlox.config`` / ``rlox.networks`` -- registration with the shared
``Trainer`` (``ALGORITHM_REGISTRY`` / ``ALGORITHM_STATUS`` in trainer.py,
plus a ``TQCConfig`` cross-reference in config.py) is wired separately.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import rlox
from rlox.callbacks import Callback, CallbackList
from rlox.config import ConfigMixin
from rlox.logging import LoggerCallback
from rlox.networks import SquashedGaussianPolicy, polyak_update

# Standard QR-DQN / TQC Huber transition point. Exposed as a module-level
# constant (not a bare literal) so it is documented once and reused as the
# default for both TQCConfig and quantile_huber_loss.
_DEFAULT_HUBER_KAPPA = 1.0


@dataclass
class TQCConfig(ConfigMixin):
    """Configuration for TQC (Truncated Quantile Critics) training.

    Kuznetsov et al., ICML 2020 (arXiv:2005.04269). SAC with an ensemble of
    ``n_critics`` distributional (quantile) critics; the Bellman target
    pools all ``n_critics * n_quantiles`` next-state quantiles, sorts them,
    and drops the top ``top_quantiles_to_drop_per_net * n_critics`` before
    bootstrapping -- this truncation controls the overestimation bias that a
    plain ensemble mean (or even ``min``) otherwise suffers from.

    Attributes
    ----------
    learning_rate : float
        Adam learning rate for the actor, critic ensemble, and alpha
        optimisers (default 3e-4).
    buffer_size : int
        Replay buffer capacity (default 1_000_000).
    batch_size : int
        Minibatch size (default 256).
    tau : float
        Polyak averaging coefficient for the critic target ensemble
        (default 0.005).
    gamma : float
        Discount factor (default 0.99).
    target_entropy : float or None
        Target entropy for auto-tuning. None -> ``-dim(action_space)``.
    auto_entropy : bool
        Whether to automatically tune the entropy coefficient (default True).
    learning_starts : int
        Random exploration steps before training begins (default 1000).
    hidden : int
        Hidden layer width for the actor and critic networks (default 256).
    n_critics : int
        Number of quantile critics in the ensemble (default 5).
    n_quantiles : int
        Number of quantiles predicted by each critic (default 25).
    top_quantiles_to_drop_per_net : int
        Number of largest pooled quantiles dropped *per critic* before
        forming the Bellman target (default 2). The total dropped per
        transition is ``top_quantiles_to_drop_per_net * n_critics``.
    huber_kappa : float
        Huber transition point (kappa) for the quantile Huber loss
        (default 1.0, the standard QR-DQN / TQC choice).
    """

    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    target_entropy: float | None = None
    auto_entropy: bool = True
    learning_starts: int = 1000
    hidden: int = 256
    n_critics: int = 5
    n_quantiles: int = 25
    top_quantiles_to_drop_per_net: int = 2
    huber_kappa: float = _DEFAULT_HUBER_KAPPA

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {self.buffer_size}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.n_critics < 1:
            raise ValueError(f"n_critics must be >= 1, got {self.n_critics}")
        if self.n_quantiles < 1:
            raise ValueError(f"n_quantiles must be >= 1, got {self.n_quantiles}")
        if self.top_quantiles_to_drop_per_net < 0:
            raise ValueError(
                "top_quantiles_to_drop_per_net must be >= 0, got "
                f"{self.top_quantiles_to_drop_per_net}"
            )
        if self.top_quantiles_to_drop_per_net >= self.n_quantiles:
            raise ValueError(
                "top_quantiles_to_drop_per_net must be < n_quantiles (dropping "
                ">= n_quantiles would discard every quantile a critic "
                f"predicts), got top_quantiles_to_drop_per_net="
                f"{self.top_quantiles_to_drop_per_net}, n_quantiles="
                f"{self.n_quantiles}."
            )
        if self.huber_kappa <= 0:
            raise ValueError(f"huber_kappa must be positive, got {self.huber_kappa}")


class QuantileQNetwork(nn.Module):
    """MLP mapping ``(obs, action) -> n_quantiles`` predicted return quantiles.

    The per-critic building block of TQC's distributional critic ensemble.
    Structurally identical to ``rlox.networks.QNetwork`` except the final
    layer emits ``n_quantiles`` values instead of a single scalar Q-value.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    hidden : int
        Hidden layer width (default 256).
    n_quantiles : int
        Number of quantiles predicted per (obs, action) pair (default 25).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        n_quantiles: int = 25,
    ) -> None:
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_quantiles),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted quantiles, shape ``(batch, n_quantiles)``."""
        return self.net(torch.cat([obs, action], dim=-1))


def truncate_quantiles(pooled_quantiles: torch.Tensor, n_dropped: int) -> torch.Tensor:
    """Sort ``pooled_quantiles`` ascending and drop the largest ``n_dropped``.

    This is TQC's overestimation-control mechanism: pooled next-state
    quantiles are sorted, and the top (largest / most optimistic) values are
    discarded before being used as a Bellman bootstrap target, biasing the
    target towards pessimism in a continuously tunable way (as opposed to
    SAC/TD3's fixed, coarse-grained ``min`` of two point estimates).

    Parameters
    ----------
    pooled_quantiles : torch.Tensor, shape ``(batch, n_total)``
        Quantile values pooled across critics for a batch of transitions
        (``n_total`` is typically ``n_critics * n_quantiles``).
    n_dropped : int
        Number of largest (highest) quantiles to drop. Must satisfy
        ``0 <= n_dropped < pooled_quantiles.shape[-1]``.

    Returns
    -------
    torch.Tensor, shape ``(batch, n_total - n_dropped)``
        The smallest ``n_total - n_dropped`` quantiles, sorted ascending.
    """
    n_total = pooled_quantiles.shape[-1]
    if n_dropped < 0 or n_dropped >= n_total:
        raise ValueError(
            f"n_dropped must be in [0, {n_total}) for a pool of width "
            f"{n_total}, got {n_dropped}."
        )
    sorted_quantiles, _ = torch.sort(pooled_quantiles, dim=-1)
    n_keep = n_total - n_dropped
    return sorted_quantiles[..., :n_keep]


def quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = _DEFAULT_HUBER_KAPPA,
) -> torch.Tensor:
    """Pairwise quantile Huber (pinball) loss -- the QR-DQN / TQC critic loss.

    For every ``(predicted quantile i, target quantile j)`` pair, computes
    the asymmetric Huber loss weighted by
    ``|tau_i - 1{target_j < predicted_i}|`` (the standard quantile-regression
    pinball weighting), then averages over all pairs and the batch. The
    target is ALWAYS detached internally -- gradients never flow into it,
    regardless of whether the caller already detached it.

    Parameters
    ----------
    predicted : torch.Tensor, shape ``(batch, n_pred)``
        A single critic's predicted quantiles (differentiable).
    target : torch.Tensor, shape ``(batch, n_target)``
        Target quantiles (e.g. the truncated, pooled next-state quantiles).
    tau : torch.Tensor, shape ``(n_pred,)``
        Quantile fraction for each predicted quantile head, conventionally
        ``tau_i = (i + 0.5) / n_pred``.
    kappa : float
        Huber loss transition point (default 1.0).

    Returns
    -------
    torch.Tensor
        Scalar (0-dim) loss.
    """
    target = target.detach()
    # (batch, n_pred, n_target): pairwise target-minus-predicted residuals.
    td_error = target.unsqueeze(1) - predicted.unsqueeze(2)
    abs_error = td_error.abs()
    huber = torch.where(
        abs_error <= kappa,
        0.5 * td_error.pow(2),
        kappa * (abs_error - 0.5 * kappa),
    )
    # Indicator is 1 where the target under-shoots the prediction (i.e. the
    # prediction over-estimates); pinball weight is |tau - indicator|.
    indicator = (td_error.detach() < 0).to(td_error.dtype)
    weight = (tau.view(1, -1, 1) - indicator).abs()
    return (weight * huber / kappa).mean()


class TQC:
    """Truncated Quantile Critics.

    An ensemble of ``n_critics`` distributional (quantile) critics, a
    squashed Gaussian policy, and automatic entropy tuning -- SAC's actor
    and entropy machinery unchanged, with the twin scalar critics replaced
    by a truncated-quantile-ensemble critic. Uses ``rlox.ReplayBuffer`` for
    storage. Continuous (Box) action spaces only.
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
        train_freq: int = 1,
        gradient_steps: int = 1,
        n_critics: int = 5,
        n_quantiles: int = 25,
        top_quantiles_to_drop_per_net: int = 2,
        huber_kappa: float = _DEFAULT_HUBER_KAPPA,
        ent_coef: str | float = "auto",
        auto_entropy: bool = True,
        target_entropy: float | None = None,
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

        # TQC is continuous-only: the squashed Gaussian actor and the
        # quantile-regression critics both assume a continuous Box action
        # space (mirrors SAC/TD3/CrossQ).
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise ValueError(
                "TQC requires a continuous (Box) action space. "
                f"Got a discrete/unsupported action space: "
                f"{type(self.env.action_space).__name__}. "
                "Use SAC or TD3 for continuous envs; DQN/PPO for discrete envs."
            )

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        # train_freq / gradient_steps mirror SAC semantics: do
        # ``gradient_steps`` SGD updates every ``train_freq`` env steps.
        self.train_freq = max(1, int(train_freq))
        self.gradient_steps = max(1, int(gradient_steps))
        self.seed = seed

        # Seed torch/np/env RNG for reproducibility. This MUST happen
        # before the actor/critic networks are constructed below so that
        # their weight initialisation is deterministic for a given seed
        # (precedent: crossq.py). Gymnasium does not propagate
        # `env.reset(seed=...)` to the action space's RNG, and the
        # `learning_starts` exploration phase in `train()` calls
        # `action_space.sample()`, so the action space needs its own
        # `.seed()` call here too.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.action_space.seed(seed)

        # ent_coef handling mirrors SAC/CrossQ: a numeric value pins alpha
        # (auto_entropy=False); "auto" (default) learns it.
        if isinstance(ent_coef, (int, float)):
            auto_entropy = False
            self._fixed_alpha = float(ent_coef)
        else:
            self._fixed_alpha = None

        self.config = TQCConfig(
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            target_entropy=target_entropy,
            auto_entropy=auto_entropy,
            learning_starts=learning_starts,
            hidden=hidden,
            n_critics=n_critics,
            n_quantiles=n_quantiles,
            top_quantiles_to_drop_per_net=top_quantiles_to_drop_per_net,
            huber_kappa=huber_kappa,
        )

        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))
        act_high = float(self.env.action_space.high[0])

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_high = act_high

        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net
        self.huber_kappa = huber_kappa
        self.n_dropped_quantiles = top_quantiles_to_drop_per_net * n_critics
        self.n_target_quantiles = n_critics * n_quantiles - self.n_dropped_quantiles
        # Fixed per-quantile fractions tau_i = (i + 0.5) / n_quantiles,
        # precomputed once rather than rebuilt on every _update() call.
        self.tau_fractions = (
            torch.arange(n_quantiles, dtype=torch.float32) + 0.5
        ) / n_quantiles

        # Actor: reuse SAC's SquashedGaussianPolicy unchanged.
        self.actor = SquashedGaussianPolicy(obs_dim, act_dim, hidden)

        # Critic ensemble: n_critics independent quantile networks, each
        # predicting n_quantiles quantiles of the return distribution.
        # Target ensemble mirrors SAC's target-network mechanism (soft /
        # polyak update every _update() call, no delay).
        self.critics = nn.ModuleList(
            QuantileQNetwork(obs_dim, act_dim, hidden, n_quantiles)
            for _ in range(n_critics)
        )
        self.critic_targets = copy.deepcopy(self.critics)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        # One optimiser over the whole critic ensemble -- the per-critic
        # losses are summed before a single backward()/step() pair, so
        # separate per-critic optimisers would be equivalent but noisier to
        # manage (matches how `polyak_update` below treats the ensemble as
        # a single nn.Module for its target update).
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=learning_rate
        )

        # Entropy tuning -- identical mechanism to SAC.
        self.auto_entropy = auto_entropy
        if target_entropy is None:
            self.target_entropy = -float(act_dim)
        else:
            self.target_entropy = target_entropy

        if auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self._fixed_alpha if self._fixed_alpha is not None else 0.2

        self.buffer = rlox.ReplayBuffer(buffer_size, obs_dim, act_dim)

        self.callbacks = CallbackList(callbacks)
        self.logger = logger
        self._global_step = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run the training loop and return the final metrics dict."""
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
        """One gradient step: truncated-quantile critic update + actor update."""
        batch = self.buffer.sample(self.batch_size, step)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        B = obs.shape[0]

        # ------------------------------------------------------------------
        # Truncated target quantiles -- TQC's overestimation-control step.
        # Pool ALL n_critics * n_quantiles next-state quantiles (from the
        # TARGET ensemble), sort ascending, and drop the top
        # n_dropped_quantiles (largest / most optimistic) before
        # bootstrapping.
        # ------------------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            next_actions = next_actions * self.act_high

            next_z = torch.stack(
                [target(next_obs, next_actions) for target in self.critic_targets],
                dim=1,
            )  # (B, n_critics, n_quantiles)
            pooled_next_z = next_z.reshape(B, -1)  # (B, n_critics * n_quantiles)
            truncated_next_z = truncate_quantiles(
                pooled_next_z, self.n_dropped_quantiles
            )  # (B, n_target_quantiles)

            target_quantiles = rewards.unsqueeze(-1) + self.gamma * (
                1.0 - terminated
            ).unsqueeze(-1) * (
                truncated_next_z - self.alpha * next_log_prob.unsqueeze(-1)
            )

        # ------------------------------------------------------------------
        # Critic loss -- quantile Huber, each critic's predicted quantiles
        # against the SAME shared truncated target set.
        # ------------------------------------------------------------------
        critic_losses = [
            quantile_huber_loss(
                critic(obs, actions),
                target_quantiles,
                self.tau_fractions,
                self.huber_kappa,
            )
            for critic in self.critics
        ]
        critic_loss = torch.stack(critic_losses).sum()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------------
        # Actor loss -- maximise the mean of ALL (untruncated) live critics'
        # quantiles at the actor's own sampled action, minus entropy.
        # ------------------------------------------------------------------
        new_actions, log_prob = self.actor.sample(obs)
        new_actions = new_actions * self.act_high
        all_z = torch.stack(
            [critic(obs, new_actions) for critic in self.critics], dim=1
        )  # (B, n_critics, n_quantiles)
        q_new = all_z.mean(dim=(1, 2))  # (B,)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (entropy coefficient) update -- identical mechanism to SAC.
        alpha_loss_val = 0.0
        if self.auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # Soft target update -- one batched call over the whole critic
        # ensemble. nn.ModuleList IS an nn.Module, so polyak_update's
        # torch._foreach ops apply across every critic's parameters in a
        # single pass (deepcopy at construction time guarantees the two
        # ModuleLists yield parameters in the same order).
        polyak_update(self.critics, self.critic_targets, self.tau)

        return {
            "critic_loss": critic_loss.item() / self.n_critics,
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_val,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return an action for the given observation."""
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
            "critics_state_dict": self.critics.state_dict(),
            "critic_targets_state_dict": self.critic_targets.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "step": self._global_step,
            "config": self.config.to_dict(),
            "env_id": self.env_id,
            "torch_rng_state": torch.random.get_rng_state(),
        }
        if self.auto_entropy:
            data["log_alpha"] = self.log_alpha.detach().clone()
            data["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        torch.save(data, path)

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> TQC:
        """Restore TQC from a checkpoint produced by :meth:`save`."""
        from rlox.checkpoint import safe_torch_load

        data = safe_torch_load(path)
        config = data["config"]
        eid = env_id or data.get("env_id", "Pendulum-v1")

        tqc = cls(env_id=eid, **config)
        tqc.actor.load_state_dict(data["actor_state_dict"])
        tqc.critics.load_state_dict(data["critics_state_dict"])
        tqc.critic_targets.load_state_dict(data["critic_targets_state_dict"])
        tqc.actor_optimizer.load_state_dict(data["actor_optimizer_state_dict"])
        tqc.critic_optimizer.load_state_dict(data["critic_optimizer_state_dict"])
        tqc._global_step = data.get("step", 0)

        if tqc.auto_entropy and "log_alpha" in data:
            tqc.log_alpha.data.copy_(data["log_alpha"])
            tqc.alpha = tqc.log_alpha.exp().item()
            if "alpha_optimizer_state_dict" in data:
                tqc.alpha_optimizer.load_state_dict(data["alpha_optimizer_state_dict"])

        if "torch_rng_state" in data:
            torch.random.set_rng_state(data["torch_rng_state"])

        return tqc
