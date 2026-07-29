"""Shared pytest fixtures and configuration for the rlox experiment framework.

All correctness tests import from this module via pytest's automatic conftest
discovery. Performance and convergence tests reuse the same fixtures.

Marker summary
--------------
correctness  Fast (<1 s) behavioral tests — run by default.
performance  Timing benchmarks — excluded by default, opt-in with -m performance.
convergence  Full training runs — very slow, opt-in with -m convergence.
slow         Alias for convergence; used to skip in CI.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Make the public rlox repo importable without installing
_PUBLIC_REPO = Path(__file__).parent.parent.parent / "rlox"
if str(_PUBLIC_REPO) not in sys.path:
    sys.path.insert(0, str(_PUBLIC_REPO))


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "correctness: fast behavioral tests (<1 s each); run by default",
    )
    config.addinivalue_line(
        "markers",
        "performance: timing benchmarks; excluded by default (use -m performance)",
    )
    config.addinivalue_line(
        "markers",
        "convergence: full training-run tests; very slow (use -m convergence)",
    )
    config.addinivalue_line(
        "markers",
        "slow: alias for convergence; marks any long-running test",
    )


# ---------------------------------------------------------------------------
# rlox availability
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rlox_module():
    """Import and return the rlox module, skipping if unavailable."""
    try:
        import rlox
        return rlox
    except ImportError:
        pytest.skip("rlox not installed or not on PYTHONPATH")


# ---------------------------------------------------------------------------
# Optional framework availability
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def numpy_available():
    """Always available; returns the numpy module."""
    return np


@pytest.fixture(scope="session")
def torch_available():
    """Return torch if installed, otherwise skip the test."""
    try:
        import torch
        return torch
    except ImportError:
        pytest.skip("torch not installed")


@pytest.fixture(scope="session")
def sb3_available():
    """Return stable_baselines3 if installed, otherwise skip the test."""
    try:
        import stable_baselines3
        return stable_baselines3
    except ImportError:
        pytest.skip("stable-baselines3 not installed")


@pytest.fixture(scope="session")
def torchrl_available():
    """Return torchrl if installed, otherwise skip the test."""
    try:
        import torchrl
        return torchrl
    except ImportError:
        pytest.skip("torchrl not installed")


@pytest.fixture(scope="session")
def gymnasium_available():
    """Return gymnasium if installed, otherwise skip the test."""
    try:
        import gymnasium
        return gymnasium
    except ImportError:
        pytest.skip("gymnasium not installed")


@pytest.fixture(scope="session")
def envpool_available():
    """Return envpool if installed, otherwise skip the test."""
    try:
        import envpool
        return envpool
    except ImportError:
        pytest.skip("envpool not installed")


# ---------------------------------------------------------------------------
# Reproducible random data
# ---------------------------------------------------------------------------


def _make_trajectory(
    n_steps: int,
    seed: int = 42,
    done_prob: float = 0.05,
) -> dict[str, np.ndarray]:
    """Create a seeded synthetic rollout trajectory."""
    rng = np.random.default_rng(seed)
    return {
        "rewards": rng.standard_normal(n_steps).astype(np.float64),
        "values": rng.standard_normal(n_steps).astype(np.float64),
        "dones": (rng.random(n_steps) < done_prob).astype(np.float64),
        "last_value": float(rng.standard_normal()),
    }


@pytest.fixture
def trajectory_128():
    """Seeded trajectory with 128 steps."""
    return _make_trajectory(128, seed=42)


@pytest.fixture
def trajectory_2048():
    """Seeded trajectory with 2048 steps."""
    return _make_trajectory(2048, seed=42)


@pytest.fixture
def trajectory_32768():
    """Seeded trajectory with 32 768 steps."""
    return _make_trajectory(32768, seed=42)


@pytest.fixture(params=[128, 2048, 32768], ids=["128", "2048", "32768"])
def trajectory(request):
    """Parametrized fixture covering small, medium, and large trajectories."""
    return _make_trajectory(request.param, seed=42)


@pytest.fixture
def rng_seed():
    """Default RNG seed used throughout the test suite."""
    return 42


# ---------------------------------------------------------------------------
# Standard test configs
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    """Minimal configuration for fast unit tests."""
    return {
        "n_envs": 2,
        "n_steps": 32,
        "obs_dim": 4,
        "act_dim": 1,
        "gamma": 0.99,
        "lam": 0.95,
        "seed": 42,
    }


@pytest.fixture
def medium_config():
    """Moderate configuration for integration tests."""
    return {
        "n_envs": 8,
        "n_steps": 128,
        "obs_dim": 4,
        "act_dim": 1,
        "gamma": 0.99,
        "lam": 0.95,
        "seed": 42,
    }


@pytest.fixture
def large_config():
    """Large configuration for stress / performance tests."""
    return {
        "n_envs": 64,
        "n_steps": 512,
        "obs_dim": 4,
        "act_dim": 1,
        "gamma": 0.99,
        "lam": 0.95,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Results output directory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    """Session-scoped timestamped results directory.

    Uses RLOX_RESULTS_DIR env var if set (e.g. in Docker), otherwise
    falls back to a local experiments/results/ directory.
    """
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    env_dir = os.environ.get("RLOX_RESULTS_DIR")
    if env_dir:
        base = Path(env_dir) / "experiments" / ts
    else:
        base = Path(__file__).parent / "results" / ts
    base.mkdir(parents=True, exist_ok=True)
    for sub in ("correctness", "performance", "convergence", "ablation"):
        (base / sub).mkdir(exist_ok=True)
    return base
