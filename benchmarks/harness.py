"""Shared fixtures and configuration for rlox benchmarks."""

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_N_BOOTSTRAP = 10_000


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    category: str
    framework: str
    params: dict = field(default_factory=dict)
    times_ns: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def median_ns(self) -> float:
        return float(np.median(self.times_ns)) if self.times_ns else 0.0

    @property
    def mean_ns(self) -> float:
        return float(np.mean(self.times_ns)) if self.times_ns else 0.0

    @property
    def p25_ns(self) -> float:
        return float(np.percentile(self.times_ns, 25)) if self.times_ns else 0.0

    @property
    def p75_ns(self) -> float:
        return float(np.percentile(self.times_ns, 75)) if self.times_ns else 0.0

    @property
    def p99_ns(self) -> float:
        return float(np.percentile(self.times_ns, 99)) if self.times_ns else 0.0

    @property
    def min_ns(self) -> float:
        return float(np.min(self.times_ns)) if self.times_ns else 0.0

    @property
    def max_ns(self) -> float:
        return float(np.max(self.times_ns)) if self.times_ns else 0.0

    @property
    def iqr_ns(self) -> float:
        return self.p75_ns - self.p25_ns

    @property
    def throughput(self) -> float | None:
        """Items per second, if 'n_items' in params."""
        if "n_items" in self.params and self.median_ns > 0:
            return self.params["n_items"] / (self.median_ns / 1e9)
        return None

    def summary(self) -> dict:
        d = {
            "name": self.name,
            "category": self.category,
            "framework": self.framework,
            "params": self.params,
            "median_ns": self.median_ns,
            "mean_ns": self.mean_ns,
            "p25_ns": self.p25_ns,
            "p75_ns": self.p75_ns,
            "p99_ns": self.p99_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "iqr_ns": self.iqr_ns,
            "n_samples": len(self.times_ns),
        }
        if self.throughput is not None:
            d["throughput_per_s"] = self.throughput
        d.update(self.metadata)
        return d


@dataclass
class ComparisonResult:
    """Comparison between rlox and a baseline framework."""
    benchmark_name: str
    rlox: BenchmarkResult
    baseline: BenchmarkResult
    baseline_name: str
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP

    @property
    def speedup(self) -> float:
        if self.rlox.median_ns == 0:
            return float("inf")
        return self.baseline.median_ns / self.rlox.median_ns

    @property
    def speedup_ci_95(self) -> tuple[float, float]:
        """Bootstrap 95% CI for speedup ratio."""
        if not self.rlox.times_ns or not self.baseline.times_ns:
            return (0.0, 0.0)
        rng = np.random.default_rng(self.bootstrap_seed)
        rlox_arr = np.array(self.rlox.times_ns)
        base_arr = np.array(self.baseline.times_ns)
        ratios = []
        for _ in range(self.n_bootstrap):
            r_sample = rng.choice(rlox_arr, size=len(rlox_arr), replace=True)
            b_sample = rng.choice(base_arr, size=len(base_arr), replace=True)
            r_med = np.median(r_sample)
            if r_med > 0:
                ratios.append(np.median(b_sample) / r_med)
        if not ratios:
            return (0.0, 0.0)
        return (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))

    def summary(self) -> dict:
        lo, hi = self.speedup_ci_95
        return {
            "benchmark": self.benchmark_name,
            "rlox_median_ns": self.rlox.median_ns,
            "baseline_median_ns": self.baseline.median_ns,
            "baseline_framework": self.baseline_name,
            "speedup": self.speedup,
            "speedup_ci_95_lo": lo,
            "speedup_ci_95_hi": hi,
            "significant": lo > 1.0,  # CI lower bound > 1.0 means statistically faster
        }


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def timed_run(fn, n_warmup: int = 5, n_reps: int = 30) -> list[float]:
    """Run fn() with warmup, return list of elapsed times in nanoseconds."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_reps):
        start = time.perf_counter_ns()
        fn()
        elapsed = time.perf_counter_ns() - start
        times.append(float(elapsed))
    return times


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import rlox
        info["rlox_available"] = True
    except ImportError:
        info["rlox_available"] = False
    return info


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(results: list[dict], comparisons: list[dict], output_dir: str = "benchmark_results"):
    """Write benchmark results to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "system": system_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "comparisons": comparisons,
    }
    path = Path(output_dir) / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return str(path)
