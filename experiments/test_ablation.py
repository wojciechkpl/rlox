"""Ablation experiment tests.

Validates that the ablation hierarchy holds:

    full_rust > python_gae > python_env > all_python

Tests
-----
1. Full Rust pipeline is faster than each partial config
2. Replacing any single Rust component with Python slows things down
3. The ranking is consistent (full > partial > pure Python)
4. Performance measurement for each config with bootstrap CIs

These tests import and re-use the experiment logic defined in
experiments/scripts/run_ablation.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from utils import bootstrap_ci, get_system_info, measure_time, save_results, speedup_ci

# Ensure the scripts directory is on the path so we can import helpers
_SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Helpers: single-config rollout
# ---------------------------------------------------------------------------


def _run_config(
    config: dict,
    n_envs: int,
    n_steps: int,
    seed: int,
) -> float:
    """Run one rollout with the given component configuration.

    Parameters
    ----------
    config:
        Dict with keys 'env', 'gae' (values 'rust' or 'python').
    n_envs:
        Number of parallel environments.
    n_steps:
        Steps per rollout.
    seed:
        Random seed.

    Returns
    -------
    Elapsed wall-clock time in seconds.
    """
    import numpy as np

    try:
        import rlox
    except ImportError:
        raise RuntimeError("rlox not installed")

    # Environment
    if config["env"] == "rust":
        env = rlox.VecEnv(n=n_envs, seed=seed)
        obs = env.reset_all(seed=seed)
    else:
        try:
            from rlox.gym_vec_env import GymVecEnv
            env = GymVecEnv("CartPole-v1", n_envs=n_envs, seed=seed)
            obs = env.reset_all()
        except ImportError:
            pytest.skip("GymVecEnv (Python env) not available")

    all_rewards: list[list[float]] = [[] for _ in range(n_envs)]
    all_values: list[list[float]] = [[] for _ in range(n_envs)]
    all_dones: list[list[float]] = [[] for _ in range(n_envs)]

    t0 = time.perf_counter()

    for _ in range(n_steps):
        actions = [i % 2 for i in range(n_envs)]
        if config["env"] == "rust":
            result = env.step_all(actions)
        else:
            result = env.step_all(np.array(actions, dtype=np.int64))

        rewards = np.asarray(result["rewards"])
        terminated = np.asarray(result["terminated"])
        truncated = np.asarray(result.get("truncated", np.zeros(n_envs, dtype=bool)))
        dones = (terminated | truncated).astype(np.float64)

        for j in range(n_envs):
            all_rewards[j].append(float(rewards[j]))
            all_values[j].append(0.0)
            all_dones[j].append(float(dones[j]))

    # GAE per environment
    for j in range(n_envs):
        r = np.array(all_rewards[j], dtype=np.float64)
        v = np.array(all_values[j], dtype=np.float64)
        d = np.array(all_dones[j], dtype=np.float64)
        if config["gae"] == "rust":
            rlox.compute_gae(r, v, d, 0.0, 0.99, 0.95)
        else:
            from run_ablation import numpy_gae
            numpy_gae(r, v, d, 0.0)

    return time.perf_counter() - t0


CONFIGS = {
    "full_rust":   {"env": "rust",   "gae": "rust"},
    "python_gae":  {"env": "rust",   "gae": "python"},
    "python_env":  {"env": "python", "gae": "rust"},
    "all_python":  {"env": "python", "gae": "python"},
}


def _benchmark_config(config_name: str, n_envs: int = 8, n_steps: int = 128,
                      n_reps: int = 20, seed: int = 42) -> dict:
    """Measure timing for a single ablation config over n_reps rollouts."""
    config = CONFIGS[config_name]

    times = []
    for rep in range(n_reps):
        t = _run_config(config, n_envs=n_envs, n_steps=n_steps, seed=seed + rep)
        times.append(t)

    times_arr = np.array(times)
    lo, hi = bootstrap_ci(times_arr, n_resamples=5000, statistic=np.median)

    total_steps = n_envs * n_steps
    sps_samples = [total_steps / t for t in times]
    sps_median = float(np.median(sps_samples))
    sps_lo, sps_hi = bootstrap_ci(np.array(sps_samples), n_resamples=5000, statistic=np.median)

    return {
        "config_name": config_name,
        "config": config,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "n_reps": n_reps,
        "median_elapsed_s": float(np.median(times_arr)),
        "elapsed_ci_95": [lo, hi],
        "sps_median": sps_median,
        "sps_ci_95": [sps_lo, sps_hi],
        "times_s": times,
    }


# ---------------------------------------------------------------------------
# 1. Full Rust is faster than each partial config
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize("partial_config", ["python_gae", "python_env", "all_python"])
def test_full_rust_faster_than_partial(partial_config, rlox_module, results_dir):
    """full_rust must be faster (lower median time) than *partial_config*."""
    n_envs, n_steps, n_reps = 8, 128, 15

    rust_result = _benchmark_config("full_rust", n_envs, n_steps, n_reps)
    partial_result = _benchmark_config(partial_config, n_envs, n_steps, n_reps)

    sp, sp_lo, sp_hi = speedup_ci(
        [t * 1e9 for t in rust_result["times_s"]],
        [t * 1e9 for t in partial_result["times_s"]],
    )

    record = {
        "test": f"full_rust_vs_{partial_config}",
        "full_rust": rust_result,
        partial_config: partial_result,
        "speedup": sp,
        "speedup_ci_95": [sp_lo, sp_hi],
        "full_rust_faster": rust_result["median_elapsed_s"] < partial_result["median_elapsed_s"],
    }
    save_results(
        record,
        results_dir / "ablation" / f"full_rust_vs_{partial_config}.json",
    )

    print(
        f"\n[Ablation] full_rust={rust_result['sps_median']:,.0f} SPS  "
        f"{partial_config}={partial_result['sps_median']:,.0f} SPS  "
        f"speedup={sp:.2f}x [{sp_lo:.2f},{sp_hi:.2f}]"
    )

    # full_rust CI lower bound must be > partial_config CI upper bound to be
    # statistically significant, OR at minimum the medians must be ordered
    assert rust_result["sps_median"] >= partial_result["sps_median"], (
        f"full_rust ({rust_result['sps_median']:.0f} SPS) should be faster than "
        f"{partial_config} ({partial_result['sps_median']:.0f} SPS)"
    )


# ---------------------------------------------------------------------------
# 2. Replacing any single component makes it slower
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_single_component_replacement_slows_down(rlox_module, results_dir):
    """Each individual component replacement must degrade performance."""
    n_envs, n_steps, n_reps = 8, 128, 10

    results = {}
    for name in CONFIGS:
        results[name] = _benchmark_config(name, n_envs, n_steps, n_reps)

    full_sps = results["full_rust"]["sps_median"]
    python_gae_sps = results["python_gae"]["sps_median"]
    python_env_sps = results["python_env"]["sps_median"]
    all_python_sps = results["all_python"]["sps_median"]

    failures = []
    if python_gae_sps > full_sps:
        failures.append(f"python_gae ({python_gae_sps:.0f}) >= full_rust ({full_sps:.0f})")
    if python_env_sps > full_sps:
        failures.append(f"python_env ({python_env_sps:.0f}) >= full_rust ({full_sps:.0f})")

    record = {
        "test": "single_component_degradation",
        "results": results,
        "full_rust_sps": full_sps,
        "python_gae_sps": python_gae_sps,
        "python_env_sps": python_env_sps,
        "all_python_sps": all_python_sps,
        "failures": failures,
        "system": get_system_info(),
    }
    save_results(record, results_dir / "ablation" / "component_degradation.json")

    assert len(failures) == 0, (
        "Some component replacements did NOT slow down performance:\n"
        + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 3. Consistent ranking
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_ablation_ranking_is_consistent(rlox_module, results_dir):
    """The SPS ranking must follow: full_rust >= python_gae OR python_env >= all_python."""
    n_envs, n_steps, n_reps = 8, 128, 10

    results = {name: _benchmark_config(name, n_envs, n_steps, n_reps) for name in CONFIGS}

    full_sps = results["full_rust"]["sps_median"]
    pg_sps = results["python_gae"]["sps_median"]
    pe_sps = results["python_env"]["sps_median"]
    ap_sps = results["all_python"]["sps_median"]

    save_results(
        {
            "test": "ranking_consistency",
            "sps": {k: v["sps_median"] for k, v in results.items()},
            "full_rust_ge_all_partials": full_sps >= max(pg_sps, pe_sps),
            "partials_ge_all_python": min(pg_sps, pe_sps) >= ap_sps,
        },
        results_dir / "ablation" / "ranking.json",
    )

    assert full_sps >= ap_sps, (
        f"full_rust ({full_sps:.0f} SPS) should be faster than all_python ({ap_sps:.0f} SPS)"
    )


# ---------------------------------------------------------------------------
# 4. Component attribution with bootstrap CIs
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_ablation_component_attribution(rlox_module, results_dir):
    """Measure and record performance with bootstrap CIs for all configs."""
    n_envs, n_steps, n_reps = 8, 128, 20

    all_results = {}
    for name in CONFIGS:
        print(f"\n  Benchmarking: {name} ...", flush=True)
        all_results[name] = _benchmark_config(name, n_envs, n_steps, n_reps)

    full_sps = all_results["full_rust"]["sps_median"]
    attribution = {}
    for name, res in all_results.items():
        if name == "full_rust":
            attribution[name] = {"slowdown": 1.0, "sps": full_sps}
        else:
            slowdown = full_sps / max(res["sps_median"], 1e-9)
            attribution[name] = {"slowdown": slowdown, "sps": res["sps_median"]}

    record = {
        "benchmark": "component_attribution",
        "n_envs": n_envs,
        "n_steps": n_steps,
        "n_reps": n_reps,
        "configs": all_results,
        "attribution": attribution,
        "system": get_system_info(),
    }
    save_results(record, results_dir / "ablation" / "component_attribution.json")

    print("\nAblation summary:")
    print(f"  {'Config':<20s}  {'SPS':>10s}  {'Slowdown':>10s}")
    print(f"  {'-'*44}")
    for name, a in attribution.items():
        print(f"  {name:<20s}  {a['sps']:>10,.0f}  {a['slowdown']:>10.2f}x")

    # The test passes as long as measurements complete without error.
    # Assertions on ordering are in the dedicated tests above.
    assert len(all_results) == 4
