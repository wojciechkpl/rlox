"""Shared network architectures for off-policy algorithms (SAC, TD3, DQN).

Provides Q-networks, stochastic/deterministic policy networks, and the
Polyak (soft) target update utility.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def polyak_update(source: nn.Module, target: nn.Module, tau: float = 0.005) -> None:
    """Soft update: target = tau * source + (1 - tau) * target.

    Uses torch._foreach ops to batch the update across all parameters,
    reducing Python loop overhead.
    """
    with torch.no_grad():
        source_params = list(source.parameters())
        target_params = list(target.parameters())
        torch._foreach_mul_(target_params, 1.0 - tau)
        torch._foreach_add_(target_params, source_params, alpha=tau)


def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to all Linear and Conv2d layers.

    Recursively walks the module tree and wraps any ``nn.Linear`` or
    ``nn.Conv2d`` child with ``torch.nn.utils.spectral_norm``.

    Parameters
    ----------
    module : nn.Module
        The module to transform (modified in-place).

    Returns
    -------
    nn.Module
        The same module, for call-chaining.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            nn.utils.spectral_norm(child)
        else:
            apply_spectral_norm(child)
    return module


class QNetwork(nn.Module):
    """Twin Q-value network for SAC / TD3.

    Takes (obs, action) concatenated as input, outputs scalar Q-value.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class SquashedGaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing for SAC.

    Outputs actions in [-1, 1] with corrected log-probabilities.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """Sample action and compute log-prob with tanh correction."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterised
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds correction
        log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return y_t, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (mean through tanh)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class DeterministicPolicy(nn.Module):
    """Deterministic policy for TD3."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: int = 256, max_action: float = 1.0
    ):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(obs)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture: separate value and advantage streams."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature(obs)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class SimpleQNetwork(nn.Module):
    """Standard DQN Q-network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class BatchRenorm1d(nn.Module):
    """Batch Renormalization for 2-D (batch, features) tensors (Ioffe 2017).

    During **training** the module normalises using batch statistics and applies
    a learned affine transformation.  Renorm correction factors ``r`` and ``d``
    relate batch statistics to running statistics and are clipped to
    ``[1/r_max, r_max]`` / ``[-d_max, d_max]`` respectively.

    During **warmup** (``step_count < warmup_steps``) the corrections are
    forced to ``r=1, d=0`` so the module behaves exactly like standard
    ``BatchNorm1d``.  After warmup the clip ranges relax linearly to caps of
    ``r_max=3`` and ``d_max=5``.

    During **eval** mode running statistics are used for normalisation and
    no statistics are updated.

    Parameters
    ----------
    num_features : int
        Number of feature channels (must match the last / only non-batch dim).
    momentum : float
        EMA coefficient for running mean/var updates (default 0.01).
    eps : float
        Numerical stability term added to variance (default 1e-3).
    warmup_steps : int
        Number of training forward passes during which ``r=1, d=0`` is
        enforced (default 100_000).
    """

    # Maximum clip values after warmup.
    _R_MAX: float = 3.0
    _D_MAX: float = 5.0

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.01,
        eps: float = 1e-3,
        warmup_steps: int = 100_000,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.warmup_steps = warmup_steps

        # Learnable affine parameters.
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Running statistics (not gradients — pure EMA buffers).
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        # Step counter used to determine warmup phase.
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` with shape ``(B, num_features)``."""
        if self.training:
            return self._forward_train(x)
        return self._forward_eval(x)

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        # Batch statistics (unbiased=False to match BatchNorm convention for
        # the normalisation denominator; we use the biased estimate for running
        # stat updates as well, matching PyTorch BatchNorm behaviour).
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_std = (batch_var + self.eps).sqrt()
        running_std = (self.running_var + self.eps).sqrt()

        # Renorm correction factors (detached — no gradient through them).
        if self.step_count.item() < self.warmup_steps:
            r = torch.ones_like(batch_std)
            d = torch.zeros_like(batch_mean)
        else:
            # Progress in (0, 1] after warmup starts.
            progress = min(
                1.0,
                (self.step_count.item() - self.warmup_steps) / max(self.warmup_steps, 1),
            )
            r_max = 1.0 + progress * (self._R_MAX - 1.0)
            d_max = progress * self._D_MAX

            r = (batch_std / running_std).clamp(1.0 / r_max, r_max).detach()
            d = ((batch_mean - self.running_mean) / running_std).clamp(-d_max, d_max).detach()

        # Normalise with batch stats then apply renorm correction.
        x_hat = (x - batch_mean) / batch_std * r + d
        out = self.weight * x_hat + self.bias

        # Update running statistics with EMA.
        with torch.no_grad():
            self.running_mean.add_(self.momentum * (batch_mean.detach() - self.running_mean))
            self.running_var.add_(self.momentum * (batch_var.detach() - self.running_var))
            self.step_count.add_(1)

        return out

    def _forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        # Use frozen running statistics — no updates.
        x_hat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        return self.weight * x_hat + self.bias


class BNQNetwork(nn.Module):
    """Q-network with BatchRenorm1d placed after each hidden activation.

    Used by CrossQ as a drop-in replacement for ``QNetwork``.  Each hidden
    block is ``Linear -> ReLU -> BatchRenorm1d``, matching the SB3-contrib /
    original CrossQ paper ordering (Post-Norm placement stabilises learning
    better than Pre-Norm in off-policy critic networks).  The final linear
    maps to a scalar Q-value without normalisation.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    hidden : int
        Hidden layer width (default 256).
    bn_momentum : float
        EMA momentum for BatchRenorm1d running stats (default 0.01).
    bn_eps : float
        Numerical stability term for BatchRenorm1d (default 1e-3).
    renorm_warmup_steps : int
        Warmup steps before renorm corrections are applied (default 100_000).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        bn_momentum: float = 0.01,
        bn_eps: float = 1e-3,
        renorm_warmup_steps: int = 100_000,
    ) -> None:
        super().__init__()
        in_dim = obs_dim + act_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            BatchRenorm1d(hidden, momentum=bn_momentum, eps=bn_eps, warmup_steps=renorm_warmup_steps),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            BatchRenorm1d(hidden, momentum=bn_momentum, eps=bn_eps, warmup_steps=renorm_warmup_steps),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class LayerNormQNetwork(nn.Module):
    """LayerNorm-regularised MLP Q-network for PQN.

    The load-bearing architectural ingredient of PQN (arXiv:2407.04811):
    a LayerNorm is applied after each hidden linear layer, replacing the
    need for a target network by stabilising the TD-learning fixed point.

    Parameters
    ----------
    obs_dim : int
        Flattened observation dimension.
    act_dim : int
        Number of discrete actions.
    hidden : int
        Hidden layer width (default 128).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
