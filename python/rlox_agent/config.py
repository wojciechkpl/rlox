# python/rlox_agent/config.py
#
# Component 8 (partial): BenchmarkConfig dataclass + validate_config (AC-2).
#
# Import constraints: stdlib + yaml only — no torch, no vllm, no pydantic.
# This module must be importable with python3 + pyyaml alone.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import yaml


class ConfigValidationError(Exception):
    """Raised by validate_config() when the config is incomplete or mismatched."""


@dataclass
class BenchmarkConfig:
    """All locked experimental constants for the rlox validation benchmark (AC-2).

    Every field is required — validate_config() raises ConfigValidationError
    if any is None, empty, or out of range.
    """
    # Model
    model_revision: str = ""

    # Dataset
    dataset_split: str = ""
    dataset_seed: int = 0

    # Training hyperparameters
    global_batch_size: int = 0
    rollout_count: int = 0
    max_seq_len: int = 0
    reward_fn: str = ""
    data_order_seed: int = 0
    n_steps: int = 0
    n_seeds: int = 0
    warmup_steps: int = 0

    # Adversarial sweep
    adversarial_fractions: list[float] = field(default_factory=list)

    # Sandbox / rollout
    per_sample_timeout_secs: float = 0.0

    # Version pins
    primerl_commit: str = ""
    verifiers_commit: str = ""
    vllm_version: str = ""
    torch_version: str = ""
    rlox_commit: str = ""


_REQUIRED_STRING_FIELDS: tuple[str, ...] = (
    "model_revision",
    "dataset_split",
    "reward_fn",
    "primerl_commit",
    "verifiers_commit",
    "vllm_version",
    "torch_version",
    "rlox_commit",
)

_REQUIRED_NUMERIC_FIELDS: tuple[str, ...] = (
    "dataset_seed",
    "global_batch_size",
    "rollout_count",
    "max_seq_len",
    "data_order_seed",
    "n_steps",
    "n_seeds",
    "warmup_steps",
    "per_sample_timeout_secs",
)

# Numeric fields that must be strictly positive (> 0) — a value of 0 indicates
# "unset" and is not a valid configuration for a live benchmark run.
# ``dataset_seed``, ``data_order_seed``, and ``warmup_steps`` are intentionally
# excluded: 0 is a meaningful value for each of them.
_MUST_BE_POSITIVE_FIELDS: tuple[str, ...] = (
    "global_batch_size",
    "rollout_count",
    "max_seq_len",
    "n_steps",
    "per_sample_timeout_secs",
)

# Fields that are also version pins checked against env_probe output.
_VERSION_PIN_FIELDS: tuple[str, ...] = (
    "vllm_version",
    "torch_version",
)


def _default_env_probe() -> dict[str, str]:
    """Probe the real environment for installed package versions.

    This is never called during tests (env_probe is always injected).
    It avoids importing torch/vllm at module load time.
    """
    result: dict[str, str] = {}
    try:
        import importlib.metadata as _meta
        result["vllm_version"] = _meta.version("vllm")
    except Exception:
        result["vllm_version"] = ""
    try:
        import importlib.metadata as _meta
        result["torch_version"] = _meta.version("torch")
    except Exception:
        result["torch_version"] = ""
    return result


def load_config(path: str) -> BenchmarkConfig:
    """Parse a YAML file and return a BenchmarkConfig.

    Args:
        path: path to a YAML file conforming to the benchmark_v1.yaml schema.

    Returns:
        BenchmarkConfig with fields populated from the YAML.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError: if the file is not valid YAML.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    return BenchmarkConfig(
        model_revision=data.get("model_revision", ""),
        dataset_split=data.get("dataset_split", ""),
        dataset_seed=int(data.get("dataset_seed", 0)),
        global_batch_size=int(data.get("global_batch_size", 0)),
        rollout_count=int(data.get("rollout_count", 0)),
        max_seq_len=int(data.get("max_seq_len", 0)),
        reward_fn=data.get("reward_fn", ""),
        data_order_seed=int(data.get("data_order_seed", 0)),
        n_steps=int(data.get("n_steps", 0)),
        n_seeds=int(data.get("n_seeds", 0)),
        warmup_steps=int(data.get("warmup_steps", 0)),
        adversarial_fractions=list(data.get("adversarial_fractions", [])),
        per_sample_timeout_secs=float(data.get("per_sample_timeout_secs", 0.0)),
        primerl_commit=data.get("primerl_commit", ""),
        verifiers_commit=data.get("verifiers_commit", ""),
        vllm_version=data.get("vllm_version", ""),
        torch_version=data.get("torch_version", ""),
        rlox_commit=data.get("rlox_commit", ""),
    )


def validate_config(
    config: BenchmarkConfig,
    *,
    env_probe: Callable[[], dict] | None = None,
) -> None:
    """Validate a BenchmarkConfig against required invariants and the running env.

    Args:
        config: the config to validate.
        env_probe: optional callable returning a dict of running environment
            version strings, e.g. ``{"vllm_version": "0.22.0", "torch_version":
            "2.9.0"}``. When None, the default probes the real installed packages.
            Inject a fake in tests to avoid requiring torch/vllm at test time.

    Raises:
        ConfigValidationError: if any required field is unset/None/empty;
            if n_seeds < 3; if adversarial_fractions is missing 0.0 or is
            empty; or if a version pin does not match the running environment.
    """
    # 1. Validate required string fields.
    for field_name in _REQUIRED_STRING_FIELDS:
        value = getattr(config, field_name, None)
        if not value:
            raise ConfigValidationError(
                f"Required field '{field_name}' must not be empty or None; got {value!r}"
            )

    # 2. Validate required numeric fields (must not be None).
    for field_name in _REQUIRED_NUMERIC_FIELDS:
        value = getattr(config, field_name, None)
        if value is None:
            raise ConfigValidationError(
                f"Required field '{field_name}' must not be None"
            )

    # 2b. Validate must-be-positive fields (0 means "unset" for these).
    for field_name in _MUST_BE_POSITIVE_FIELDS:
        value = getattr(config, field_name, None)
        if value is not None and value <= 0:
            raise ConfigValidationError(
                f"Required field '{field_name}' must be > 0 (got {value!r}); "
                "a value of 0 indicates an unset placeholder."
            )

    # 3. n_seeds must be >= 3.
    if config.n_seeds is None or config.n_seeds < 3:
        raise ConfigValidationError(
            f"n_seeds must be >= 3 (got {config.n_seeds!r})"
        )

    # 4. adversarial_fractions must be non-empty and contain 0.0.
    fractions = config.adversarial_fractions
    if not fractions:
        raise ConfigValidationError(
            "adversarial_fractions must not be empty or None"
        )
    if 0.0 not in fractions:
        raise ConfigValidationError(
            "adversarial_fractions must contain 0.0 as the baseline fraction "
            f"(got {fractions!r})"
        )

    # 5. Version pin check against env_probe.
    probe = env_probe if env_probe is not None else _default_env_probe
    env = probe()
    for pin_field in _VERSION_PIN_FIELDS:
        if pin_field not in env:
            continue
        config_value = getattr(config, pin_field, None)
        env_value = env[pin_field]
        if config_value != env_value:
            raise ConfigValidationError(
                f"Version pin mismatch for '{pin_field}': "
                f"config has {config_value!r} but environment reports {env_value!r}"
            )
