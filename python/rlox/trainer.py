"""Unified Trainer wrapping any registered algorithm via the Strategy pattern.

The ``Trainer`` class replaces the 8 individual ``XTrainer`` classes with a
single entry point.  Algorithms are looked up by name in the
:data:`ALGORITHM_REGISTRY` or passed directly as a class.

Example
-------
>>> from rlox.trainer import Trainer
>>> trainer = Trainer("ppo", env="CartPole-v1", config={"n_envs": 16})
>>> metrics = trainer.train(total_timesteps=100_000)
>>> trainer.save("checkpoint.pt")
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

from rlox.callbacks import Callback, CallbackList


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY: dict[str, type] = {}


def register_algorithm(name: str):
    """Class decorator that registers an algorithm under *name*."""
    def decorator(cls: type) -> type:
        ALGORITHM_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def _accepts_param(cls: type, param: str) -> bool:
    """Return True if *cls.__init__* has a parameter named *param*."""
    try:
        sig = inspect.signature(cls.__init__)
        return param in sig.parameters
    except (ValueError, TypeError):
        return False


def resolve_env_id(env_id: str) -> str:
    """If *env_id* is in ENV_REGISTRY and not in gymnasium, register it.

    This bridges the plugin registry with gymnasium so that algorithms
    can continue to call ``gym.make(env_id)`` transparently.

    Parameters
    ----------
    env_id : str
        Environment identifier -- either a standard gymnasium env or a
        name previously registered via ``@register_env``.

    Returns
    -------
    str
        The same *env_id*, now guaranteed to be resolvable by gymnasium
        if it was in the plugin registry.
    """
    from rlox.plugins import ENV_REGISTRY

    if env_id in ENV_REGISTRY:
        import gymnasium as gym

        if env_id not in gym.envs.registry:
            gym.register(id=env_id, entry_point=ENV_REGISTRY[env_id])
    return env_id


# ---------------------------------------------------------------------------
# Algorithm status
# ---------------------------------------------------------------------------

ALGORITHM_STATUSES: frozenset[str] = frozenset({"validated", "experimental"})

# Algorithms that have passed convergence validation (SB3 parity, multi-seed).
# Everything else defaults to "experimental".
_VALIDATED: frozenset[str] = frozenset({"ppo", "sac", "td3", "dqn", "a2c"})

# Populated after _register_builtins() runs (see bottom of module).
ALGORITHM_STATUS: dict[str, str] = {}


def algorithm_status(name: str) -> str:
    """Return the maturity status for a registered algorithm name.

    Parameters
    ----------
    name : str
        Algorithm name (case-insensitive).

    Returns
    -------
    str
        ``"validated"`` or ``"experimental"``.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    key = name.lower()
    if key not in ALGORITHM_STATUS:
        known = ", ".join(sorted(ALGORITHM_STATUS))
        raise ValueError(
            f"Unknown algorithm {name!r}. Known algorithms: {known}"
        )
    return ALGORITHM_STATUS[key]


# ---------------------------------------------------------------------------
# Unified Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified trainer wrapping any registered algorithm.

    Parameters
    ----------
    algorithm : str | type
        Algorithm name (e.g. ``"ppo"``) or the algorithm class itself.
    env : str
        Gymnasium environment ID.
    config : dict, optional
        Algorithm-specific hyperparameters forwarded to the constructor.
    callbacks : list[Callback], optional
        Training callbacks.
    logger : object, optional
        Logger instance (WandbLogger, TensorBoardLogger, etc.).
    seed : int
        Random seed (default 42).
    compile : bool
        Whether to torch.compile the policy (default False).
    """

    def __init__(
        self,
        algorithm: str | type,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        logger: Any | None = None,
        seed: int = 42,
        compile: bool = False,
    ):
        cfg = config or {}

        # Resolve custom env: register with gymnasium if from plugin registry
        env = resolve_env_id(env)

        # Resolve algorithm class and determine status/label
        if isinstance(algorithm, str):
            algo_cls = ALGORITHM_REGISTRY.get(algorithm.lower())
            if algo_cls is None:
                raise ValueError(
                    f"Unknown algorithm {algorithm!r}. "
                    f"Registered: {sorted(ALGORITHM_REGISTRY)}"
                )
            self._algo_label: str = algorithm.lower()
            self._status: str = ALGORITHM_STATUS.get(
                self._algo_label, "experimental"
            )
            if self._status == "experimental":
                validated_list = ", ".join(sorted(_VALIDATED))
                warnings.warn(
                    f"Algorithm {self._algo_label!r} is experimental: implemented "
                    f"but not convergence-validated. "
                    f"Validated algorithms: {validated_list}.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            algo_cls = algorithm
            self._algo_label = algorithm.__name__
            self._status = "experimental"

        # Build algorithm -- inspect __init__ to pass only accepted params
        algo_kwargs: dict[str, Any] = {"env_id": env, "seed": seed}
        if _accepts_param(algo_cls, "logger"):
            algo_kwargs["logger"] = logger
        if _accepts_param(algo_cls, "callbacks"):
            algo_kwargs["callbacks"] = callbacks
        if _accepts_param(algo_cls, "compile"):
            algo_kwargs["compile"] = compile
        algo_kwargs.update(cfg)

        self.algo = algo_cls(**algo_kwargs)
        self.env = env
        self._logger = logger
        self._callbacks = CallbackList(callbacks)

        # Attach logger if not accepted by algo __init__
        if logger is not None and not _accepts_param(algo_cls, "logger"):
            if hasattr(self.algo, "logger"):
                self.algo.logger = logger

    @property
    def status(self) -> str:
        """Maturity status of the algorithm: ``'validated'`` or ``'experimental'``."""
        return getattr(self, "_status", "experimental")

    def __repr__(self) -> str:
        label = getattr(self, "_algo_label", "unknown")
        st = getattr(self, "_status", "experimental")
        env = getattr(self, "env", "unknown")
        return f"Trainer(algorithm={label!r}, env={env!r}, status={st!r})"

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run training and return metrics dict."""
        return self.algo.train(total_timesteps=total_timesteps)

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        self.algo.save(path)

    def predict(self, obs: Any, deterministic: bool = True) -> Any:
        """Get action from the trained policy."""
        return self.algo.predict(obs, deterministic=deterministic)

    def evaluate(
        self,
        n_episodes: int = 10,
        seed: int = 0,
        render: bool = False,
    ) -> dict[str, float]:
        """Run deterministic evaluation and return episode statistics.

        Parameters
        ----------
        n_episodes : int
            Number of evaluation episodes (default 10).
        seed : int
            Base seed for environment resets (default 0).
        render : bool
            Whether to render the environment (default False).

        Returns
        -------
        dict with keys: mean_reward, std_reward, min_reward, max_reward,
        mean_length, n_episodes.
        """
        import gymnasium as gym
        import numpy as np

        render_mode = "human" if render else None
        env = gym.make(self.env, render_mode=render_mode)

        # Freeze VecNormalize stats during eval
        vec_normalize = getattr(self.algo, "vec_normalize", None)
        if vec_normalize is not None:
            vec_normalize.training = False

        rewards: list[float] = []
        lengths: list[int] = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_reward, ep_len, done = 0.0, 0, False
            while not done:
                if vec_normalize is not None and hasattr(vec_normalize, "normalize_obs"):
                    obs_eval = vec_normalize.normalize_obs(
                        np.asarray(obs, dtype=np.float32).reshape(1, -1)
                    ).flatten()
                else:
                    obs_eval = obs
                action = self.predict(obs_eval, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                ep_reward += float(r)
                ep_len += 1
                done = term or trunc
            rewards.append(ep_reward)
            lengths.append(ep_len)

        if vec_normalize is not None:
            vec_normalize.training = True
        env.close()

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "n_episodes": n_episodes,
        }

    def enjoy(self, n_episodes: int = 1, seed: int = 0) -> None:
        """Render the trained policy for visual inspection.

        Parameters
        ----------
        n_episodes : int
            Number of episodes to render (default 1).
        seed : int
            Base seed for environment resets (default 0).
        """
        self.evaluate(n_episodes=n_episodes, seed=seed, render=True)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        algorithm: str | type,
        env: str | None = None,
    ) -> Trainer:
        """Restore a Trainer from a saved checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        algorithm : str | type
            Algorithm name or class.
        env : str, optional
            Environment ID (uses checkpoint's env_id if None).
        """
        if isinstance(algorithm, str):
            algo_cls = ALGORITHM_REGISTRY.get(algorithm.lower())
            if algo_cls is None:
                raise ValueError(
                    f"Unknown algorithm {algorithm!r}. "
                    f"Registered: {sorted(ALGORITHM_REGISTRY)}"
                )
            algo_label: str = algorithm.lower()
            algo_status: str = ALGORITHM_STATUS.get(algo_label, "experimental")
        else:
            algo_cls = algorithm
            algo_label = algorithm.__name__
            algo_status = "experimental"

        trainer = object.__new__(cls)
        trainer.algo = algo_cls.from_checkpoint(path, env_id=env)
        trainer.env = env or trainer.algo.env_id
        trainer._callbacks = CallbackList(None)
        trainer._logger = None
        trainer._algo_label = algo_label
        trainer._status = algo_status
        return trainer


# ---------------------------------------------------------------------------
# Register all built-in algorithms (lazy to avoid circular imports)
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Import and register all built-in algorithms."""
    from rlox.algorithms.ppo import PPO
    from rlox.algorithms.sac import SAC
    from rlox.algorithms.dqn import DQN
    from rlox.algorithms.a2c import A2C
    from rlox.algorithms.td3 import TD3
    from rlox.algorithms.mappo import MAPPO
    from rlox.algorithms.dreamer import DreamerV3
    from rlox.algorithms.impala import IMPALA
    from rlox.algorithms.decision_transformer import DecisionTransformer
    from rlox.algorithms.qmix import QMIX
    from rlox.algorithms.calql import CalQL
    from rlox.algorithms.trpo import TRPO
    from rlox.algorithms.diffusion_policy import DiffusionPolicy
    from rlox.algorithms.mpo import MPO
    from rlox.algorithms.dtp import RWDTP, RCDTP
    from rlox.algorithms.pqn import PQN

    for name, cls in [
        ("ppo", PPO),
        ("sac", SAC),
        ("dqn", DQN),
        ("a2c", A2C),
        ("td3", TD3),
        ("mappo", MAPPO),
        ("dreamer", DreamerV3),
        ("impala", IMPALA),
        ("dt", DecisionTransformer),
        ("qmix", QMIX),
        ("calql", CalQL),
        ("trpo", TRPO),
        ("diffusion", DiffusionPolicy),
        ("mpo", MPO),
        ("rwdtp", RWDTP),
        ("rcdtp", RCDTP),
        ("pqn", PQN),
    ]:
        if name not in ALGORITHM_REGISTRY:
            ALGORITHM_REGISTRY[name] = cls


_register_builtins()

# Build the status mapping now that the registry is fully populated.
# Any algorithm not in _VALIDATED defaults to "experimental", so future
# registrations automatically satisfy the completeness invariant.
ALGORITHM_STATUS.update(
    {
        name: ("validated" if name in _VALIDATED else "experimental")
        for name in ALGORITHM_REGISTRY
    }
)
