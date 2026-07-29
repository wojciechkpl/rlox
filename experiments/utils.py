"""Shared utilities for the rlox experiment framework.

Provides:
- bootstrap_ci: Bootstrap confidence interval computation
- measure_time: Timing with warmup, repeats, and percentiles
- save_results: JSON serialization with numpy support
- get_system_info: System metadata collection
- Reference implementations used as ground truth in correctness tests
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray | list[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    statistic: Callable = np.median,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data:
        1-D array of observations.
    n_resamples:
        Number of bootstrap resamples.
    ci:
        Confidence level (e.g. 0.95 for 95% CI).
    statistic:
        Callable that accepts an array and returns a scalar. Defaults to median.
    rng_seed:
        Seed for reproducibility.

    Returns
    -------
    (lower, upper) bounds of the CI.
    """
    arr = np.asarray(data, dtype=float)
    rng = np.random.default_rng(rng_seed)
    boot_stats = np.empty(n_resamples)
    n = len(arr)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = statistic(sample)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_stats, alpha * 100.0))
    hi = float(np.percentile(boot_stats, (1.0 - alpha) * 100.0))
    return lo, hi


def speedup_ci(
    rlox_times: list[float],
    baseline_times: list[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for speedup ratio (baseline_median / rlox_median).

    Returns
    -------
    (point_estimate, lower_ci, upper_ci)
    """
    rlox_arr = np.asarray(rlox_times, dtype=float)
    base_arr = np.asarray(baseline_times, dtype=float)
    rng = np.random.default_rng(rng_seed)
    ratios = []
    for _ in range(n_resamples):
        r = np.median(rng.choice(rlox_arr, size=len(rlox_arr), replace=True))
        b = np.median(rng.choice(base_arr, size=len(base_arr), replace=True))
        if r > 0:
            ratios.append(b / r)
    ratios_arr = np.array(ratios)
    alpha = (1.0 - ci) / 2.0
    point = float(np.median(base_arr) / np.median(rlox_arr))
    lo = float(np.percentile(ratios_arr, alpha * 100.0))
    hi = float(np.percentile(ratios_arr, (1.0 - alpha) * 100.0))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def measure_time(
    fn: Callable,
    warmup: int = 5,
    repeats: int = 100,
) -> dict[str, float]:
    """Measure wall-clock time for fn() with warmup.

    Parameters
    ----------
    fn:
        Zero-argument callable to benchmark.
    warmup:
        Number of warm-up calls (not recorded).
    repeats:
        Number of timed calls.

    Returns
    -------
    dict with keys: median_ns, mean_ns, min_ns, max_ns, p25_ns, p75_ns,
    p99_ns, iqr_ns, n_samples, times_ns.
    """
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        times.append(float(time.perf_counter_ns() - t0))
    arr = np.array(times)
    return {
        "median_ns": float(np.median(arr)),
        "mean_ns": float(np.mean(arr)),
        "min_ns": float(arr.min()),
        "max_ns": float(arr.max()),
        "p25_ns": float(np.percentile(arr, 25)),
        "p75_ns": float(np.percentile(arr, 75)),
        "p99_ns": float(np.percentile(arr, 99)),
        "iqr_ns": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "n_samples": repeats,
        "times_ns": times,
    }


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(data: dict | list, path: str | Path) -> Path:
    """Write *data* as indented JSON to *path*, creating parent dirs as needed.

    Numpy scalars and arrays are serialized automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    return path


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------


def get_system_info() -> dict[str, Any]:
    """Collect reproducibility metadata about the current system."""
    info: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "os": platform.system(),
        "python_version": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine() or "unknown",
        "cpu_count_logical": os.cpu_count(),
        "numpy_version": np.__version__,
    }

    # RAM (best-effort)
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        elif torch.backends.mps.is_available():
            info["gpu"] = "Apple MPS"
    except ImportError:
        info["torch_available"] = False

    # rlox
    try:
        import rlox
        info["rlox_available"] = True
        if hasattr(rlox, "__version__"):
            info["rlox_version"] = rlox.__version__
    except ImportError:
        info["rlox_available"] = False

    # Optional frameworks
    for pkg in ("stable_baselines3", "torchrl", "gymnasium", "envpool"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return info


# ---------------------------------------------------------------------------
# Reference implementations (ground truth)
# ---------------------------------------------------------------------------


def reference_gae_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """CleanRL/SB3-style GAE computed in a Python loop.

    This is the ground-truth reference implementation used in correctness tests.
    It is intentionally simple and readable so its correctness is obvious.

    Parameters
    ----------
    rewards:
        1-D array of shape (T,).
    values:
        1-D array of shape (T,).
    dones:
        1-D array of shape (T,); 1.0 means episode ended at step t.
    last_value:
        Estimated value after the final step.
    gamma:
        Discount factor.
    lam:
        GAE lambda.

    Returns
    -------
    (advantages, returns) each of shape (T,).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 1.0 - dones[t]
        next_value = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def reference_grpo_numpy(
    rewards: np.ndarray,
    group_size: int | None = None,
) -> np.ndarray:
    """GRPO group-normalized advantages (reference implementation).

    If *group_size* is None the entire *rewards* array is treated as one group.
    Otherwise rewards must have length divisible by *group_size* and each
    contiguous block of *group_size* elements is normalized independently.

    Parameters
    ----------
    rewards:
        1-D array of scalar rewards.
    group_size:
        Number of completions per prompt, or None for single-group.

    Returns
    -------
    1-D array of normalized advantages, same shape as *rewards*.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if group_size is None:
        mean = rewards.mean()
        std = rewards.std()
        if std < 1e-8:
            return np.zeros_like(rewards)
        return (rewards - mean) / std

    n = len(rewards)
    if n % group_size != 0:
        raise ValueError(
            f"len(rewards)={n} is not divisible by group_size={group_size}"
        )
    out = np.empty_like(rewards)
    for i in range(0, n, group_size):
        g = rewards[i : i + group_size]
        mean = g.mean()
        std = g.std()
        out[i : i + group_size] = 0.0 if std < 1e-8 else (g - mean) / std
    return out


def reference_token_kl_numpy(
    log_p: np.ndarray,
    log_q: np.ndarray,
) -> float:
    """Token-level KL divergence KL(p || q) = sum(p * log(p/q)).

    Parameters
    ----------
    log_p:
        Log-probabilities of the policy distribution.
    log_q:
        Log-probabilities of the reference distribution.

    Returns
    -------
    Scalar KL divergence value.
    """
    log_p = np.asarray(log_p, dtype=np.float64)
    log_q = np.asarray(log_q, dtype=np.float64)
    return float(np.sum(np.exp(log_p) * (log_p - log_q)))
