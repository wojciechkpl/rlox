"""Deprecated per-algorithm trainers — use :class:`rlox.Trainer` instead.

These classes are thin backward-compatibility wrappers around the unified
:class:`~rlox.trainer.Trainer`. They emit :class:`DeprecationWarning` on
instantiation.

Migration::

    # Old (deprecated):
    from rlox.trainers import PPOTrainer
    trainer = PPOTrainer(env="CartPole-v1", seed=42)

    # New (preferred):
    from rlox import Trainer
    trainer = Trainer("ppo", env="CartPole-v1", seed=42)
"""

from __future__ import annotations

import warnings
from typing import Any

from rlox.trainer import Trainer


def _make_legacy_trainer(algo_name: str, class_name: str) -> type:
    """Generate a deprecated trainer class that delegates to Trainer."""

    def __init__(
        self: Any,
        env: str,
        config: dict[str, Any] | None = None,
        callbacks: list | None = None,
        logger: Any | None = None,
        seed: int = 42,
        compile: bool = False,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            f"{class_name} is deprecated. Use Trainer('{algo_name}', ...) instead. "
            # Canonical docs host; see site_url in mkdocs.yml. The old
            # riserally.github.io URL predates the repo transfer.
            "See https://wojciechkpl.github.io/rlox/python-guide/#unified-trainer",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg = dict(config or {})
        cfg.update(kwargs)
        self._trainer = Trainer(
            algorithm=algo_name,
            env=env,
            config=cfg,
            callbacks=callbacks,
            logger=logger,
            seed=seed,
            compile=compile,
        )

    def train(self: Any, total_timesteps: int) -> dict[str, float]:
        return self._trainer.train(total_timesteps=total_timesteps)

    def save(self: Any, path: str) -> None:
        self._trainer.save(path)

    def predict(self: Any, obs: Any, deterministic: bool = True) -> Any:
        return self._trainer.predict(obs, deterministic=deterministic)

    def from_checkpoint(
        cls: type, path: str, env: str | None = None
    ) -> Any:
        warnings.warn(
            f"{class_name}.from_checkpoint is deprecated. "
            f"Use Trainer.from_checkpoint(path, '{algo_name}', env) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Trainer.from_checkpoint(path, algorithm=algo_name, env=env)

    attrs = {
        "__init__": __init__,
        "train": train,
        "save": save,
        "predict": predict,
        "from_checkpoint": classmethod(from_checkpoint),
        "__doc__": (
            f"Deprecated: use ``Trainer('{algo_name}', ...)`` instead.\n\n"
            f"This class is a backward-compatibility wrapper around "
            f":class:`~rlox.trainer.Trainer`."
        ),
    }

    # Expose inner algo for code that accesses trainer.algo directly
    attrs["algo"] = property(lambda self: self._trainer.algo)
    attrs["env"] = property(lambda self: self._trainer.env)
    attrs["callbacks"] = property(
        lambda self: getattr(self._trainer, "_callbacks", None)
    )

    return type(class_name, (), attrs)


PPOTrainer = _make_legacy_trainer("ppo", "PPOTrainer")
SACTrainer = _make_legacy_trainer("sac", "SACTrainer")
DQNTrainer = _make_legacy_trainer("dqn", "DQNTrainer")
A2CTrainer = _make_legacy_trainer("a2c", "A2CTrainer")
TD3Trainer = _make_legacy_trainer("td3", "TD3Trainer")
MAPPOTrainer = _make_legacy_trainer("mappo", "MAPPOTrainer")
DreamerV3Trainer = _make_legacy_trainer("dreamer", "DreamerV3Trainer")
IMPALATrainer = _make_legacy_trainer("impala", "IMPALATrainer")
