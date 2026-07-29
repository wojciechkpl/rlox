#!/usr/bin/env python3
"""Generate all paper figures from raw benchmark results.

Produces Figures 1-7 as described in PLAN.md Section 3.4:
  Fig 1: RL training loop pipeline with time breakdown (from profiling data)
  Fig 2: Architecture diagram (manual TikZ — this script generates a placeholder)
  Fig 3: Env stepping throughput vs env count (component benchmarks)
  Fig 4: Amdahl's Law time breakdown (from profiling/ablation data)
  Fig 5: Learning curves with IQM + CI bands (convergence results)
  Fig 6: SPS bar chart with error bars (convergence results)
  Fig 7: Performance profiles, Agarwal et al. style (convergence results)

Also generates:
  Fig S1: Ablation bar chart (from ablation results)
  Fig S2: Probability of improvement per task

Usage:
    python experiments/scripts/generate_figures.py results/ paper/figures/
    python experiments/scripts/generate_figures.py results/ paper/figures/ --format pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use non-interactive backend for headless generation
matplotlib.use("Agg")

# NeurIPS-quality style settings
NEURIPS_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(NEURIPS_RC)
sns.set_palette("colorblind")

# Consistent framework styling
FRAMEWORK_STYLE = {
    "rlox": {"color": "#E63946", "label": "rlox (Rust)", "marker": "o"},
    "sb3": {"color": "#457B9D", "label": "Stable-Baselines3", "marker": "s"},
    "torchrl": {"color": "#2A9D8F", "label": "TorchRL", "marker": "^"},
    "envpool": {"color": "#E9C46A", "label": "EnvPool", "marker": "D"},
    "cleanrl": {"color": "#264653", "label": "CleanRL", "marker": "v"},
}

# Reference scores for normalization (Agarwal et al.)
REFERENCE_SCORES: dict[str, tuple[float, float]] = {
    "CartPole-v1": (20.0, 500.0),
    "Acrobot-v1": (-500.0, -80.0),
    "MountainCar-v0": (-200.0, -100.0),
    "Pendulum-v1": (-1200.0, -150.0),
    "HalfCheetah-v4": (-300.0, 10000.0),
    "Hopper-v4": (0.0, 3500.0),
    "Walker2d-v4": (0.0, 5000.0),
    "Ant-v4": (-100.0, 6000.0),
    "Humanoid-v4": (0.0, 8000.0),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_convergence_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all convergence JSON results (per-seed experiment logs)."""
    results = []
    raw_dir = results_dir / "raw"
    search_dirs = [raw_dir, results_dir] if raw_dir.is_dir() else [results_dir]
    for d in search_dirs:
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("ablation") or p.name.startswith("component"):
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
                # Validate it's a convergence result (has evaluations key)
                if "evaluations" in data and "framework" in data:
                    results.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def load_ablation_results(results_dir: Path) -> dict[str, Any] | None:
    """Load ablation experiment results."""
    for candidate in [
        results_dir / "raw" / "ablation_results.json",
        results_dir / "ablation_results.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return None


def load_component_benchmarks(results_dir: Path) -> dict[str, Any] | None:
    """Load component benchmark results (env stepping, buffer ops, GAE)."""
    for candidate in [
        results_dir / "raw" / "component_benchmarks.json",
        results_dir / "component_benchmarks.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return None


def load_profiling_data(results_dir: Path) -> dict[str, Any] | None:
    """Load Amdahl's Law profiling data (time breakdown per component)."""
    for candidate in [
        results_dir / "raw" / "profiling_breakdown.json",
        results_dir / "profiling_breakdown.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _interpolate_curves(
    runs: list[dict],
    x_key: str,
    y_key: str = "mean_return",
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate learning curves to a common x-axis for averaging."""
    valid = [r for r in runs if len(r.get("evaluations", [])) >= 2]
    if not valid:
        return np.array([]), np.array([])

    x_min = max(r["evaluations"][0][x_key] for r in valid)
    x_max = min(r["evaluations"][-1][x_key] for r in valid)
    if x_max <= x_min:
        return np.array([]), np.array([])

    x_common = np.linspace(x_min, x_max, n_points)
    y_matrix = np.empty((len(valid), n_points))

    for i, run in enumerate(valid):
        xs = np.array([e[x_key] for e in run["evaluations"]])
        ys = np.array([e[y_key] for e in run["evaluations"]])
        y_matrix[i] = np.interp(x_common, xs, ys)

    return x_common, y_matrix


def _bootstrap_ci(
    y_matrix: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and bootstrapped CI at each x point."""
    n_runs, n_points = y_matrix.shape
    rng = np.random.default_rng(42)
    mean = y_matrix.mean(axis=0)
    lower = np.empty(n_points)
    upper = np.empty(n_points)
    alpha = (1.0 - ci) / 2.0

    for j in range(n_points):
        col = y_matrix[:, j]
        boots = rng.choice(col, size=(n_bootstrap, n_runs), replace=True).mean(axis=1)
        lower[j] = np.percentile(boots, 100 * alpha)
        upper[j] = np.percentile(boots, 100 * (1 - alpha))

    return mean, lower, upper


def _interquartile_mean(values: np.ndarray) -> float:
    """IQM: discard bottom 25% and top 25%, take mean."""
    arr = np.sort(values)
    n = len(arr)
    q1 = int(np.ceil(n * 0.25))
    q3 = int(np.floor(n * 0.75))
    if q1 >= q3:
        return float(np.mean(arr))
    return float(np.mean(arr[q1:q3]))


def _normalize_score(score: float, env: str) -> float:
    """Normalize to [0, 1] using reference scores."""
    if env not in REFERENCE_SCORES:
        return score
    random_s, expert_s = REFERENCE_SCORES[env]
    if expert_s == random_s:
        return 1.0 if score >= expert_s else 0.0
    return (score - random_s) / (expert_s - random_s)


def _final_return(result: dict) -> float:
    """Get mean return from last 3 evaluations."""
    evals = result.get("evaluations", [])
    if not evals:
        return float("nan")
    last_n = evals[-min(3, len(evals)) :]
    return float(np.mean([e["mean_return"] for e in last_n]))


def _fw_style(fw: str) -> dict:
    """Get plot style for a framework."""
    return FRAMEWORK_STYLE.get(fw, {"color": "#888888", "label": fw, "marker": "x"})


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def fig1_pipeline_breakdown(
    profiling: dict[str, Any] | None, output_dir: Path, fmt: str
) -> bool:
    """Fig 1: RL training loop time breakdown (horizontal stacked bar).

    Expected profiling JSON structure:
    {
        "frameworks": {
            "sb3": {"env_stepping": 45, "buffer_ops": 15, "gae": 20, "nn_update": 20},
            "rlox": {"env_stepping": 10, "buffer_ops": 3, "gae": 2, "nn_update": 85}
        },
        "unit": "percent"
    }
    """
    if profiling is None:
        print("  [SKIP] Fig 1: No profiling data found (profiling_breakdown.json)")
        return False

    frameworks = profiling.get("frameworks", {})
    if not frameworks:
        print("  [SKIP] Fig 1: Empty profiling data")
        return False

    components = ["env_stepping", "buffer_ops", "gae", "nn_update"]
    component_labels = {
        "env_stepping": "Env Stepping",
        "buffer_ops": "Buffer Ops",
        "gae": "GAE / Advantages",
        "nn_update": "NN Update",
    }
    component_colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

    fw_names = list(frameworks.keys())
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.6 * len(fw_names)))

    y_pos = np.arange(len(fw_names))
    for i, comp in enumerate(components):
        lefts = np.zeros(len(fw_names))
        for j in range(i):
            for k, fw in enumerate(fw_names):
                lefts[k] += frameworks[fw].get(components[j], 0)
        widths = [frameworks[fw].get(comp, 0) for fw in fw_names]
        ax.barh(
            y_pos,
            widths,
            left=lefts,
            color=component_colors[i],
            edgecolor="white",
            linewidth=0.5,
            label=component_labels[comp],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_fw_style(fw)["label"] for fw in fw_names])
    ax.set_xlabel("Percentage of Wall-Clock Time (%)")
    ax.set_title("RL Training Loop: Where Does the Time Go?")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    fig.tight_layout()
    out = output_dir / f"fig1_pipeline_breakdown.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Fig 1: {out}")
    return True


def fig3_env_scaling(
    component: dict[str, Any] | None, output_dir: Path, fmt: str
) -> bool:
    """Fig 3: Env stepping throughput vs env count.

    Expected component JSON structure:
    {
        "env_stepping": {
            "env_counts": [1, 2, 4, 8, 16, 32, 64, 128, 256],
            "frameworks": {
                "rlox": {"throughput_sps": [1000, 1900, ...], "ci_lower": [...], "ci_upper": [...]},
                "sb3": {...},
                ...
            }
        },
        ...
    }
    """
    if component is None:
        print("  [SKIP] Fig 3: No component benchmark data (component_benchmarks.json)")
        return False

    env_data = component.get("env_stepping", {})
    env_counts = env_data.get("env_counts", [])
    fw_data = env_data.get("frameworks", {})

    if not env_counts or not fw_data:
        print("  [SKIP] Fig 3: Missing env stepping data")
        return False

    fig, ax = plt.subplots(figsize=(7, 5))

    for fw, data in sorted(fw_data.items()):
        style = _fw_style(fw)
        throughput = data["throughput_sps"]
        ax.plot(
            env_counts[: len(throughput)],
            throughput,
            color=style["color"],
            label=style["label"],
            marker=style["marker"],
            markersize=5,
        )
        if "ci_lower" in data and "ci_upper" in data:
            ax.fill_between(
                env_counts[: len(throughput)],
                data["ci_lower"],
                data["ci_upper"],
                color=style["color"],
                alpha=0.15,
            )

    ax.set_xlabel("Number of Environments")
    ax.set_ylabel("Throughput (steps/s)")
    ax.set_title("Vectorized Environment Stepping: Throughput Scaling")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(loc="upper left")

    fig.tight_layout()
    out = output_dir / f"fig3_env_scaling.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Fig 3: {out}")
    return True


def fig4_amdahl_breakdown(
    ablation: dict[str, Any] | None, output_dir: Path, fmt: str
) -> bool:
    """Fig 4: Amdahl's Law — stacked bar showing time per component.

    Uses ablation results to derive per-component contribution.
    Ablation JSON: {"full_rust": {"sps": ..., "elapsed": ...}, "python_gae": {...}, ...}
    """
    if ablation is None:
        print("  [SKIP] Fig 4: No ablation data (ablation_results.json)")
        return False

    full = ablation.get("full_rust", {})
    python_gae = ablation.get("python_gae", {})
    python_env = ablation.get("python_env", {})
    all_python = ablation.get("all_python", {})

    if not all(d.get("sps") for d in [full, python_gae, python_env, all_python]):
        print("  [SKIP] Fig 4: Incomplete ablation data")
        return False

    # Derive time fractions from SPS differences
    # full_rust is fastest; each python_* config isolates one component's overhead
    configs = {
        "Full rlox\n(Rust)": full["sps"],
        "Python GAE\nonly": python_gae["sps"],
        "Python Env\nonly": python_env["sps"],
        "All\nPython": all_python["sps"],
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(configs.keys())
    sps_vals = list(configs.values())
    baseline = max(sps_vals)
    slowdowns = [baseline / s for s in sps_vals]

    colors = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D"]
    bars = ax.bar(names, sps_vals, color=colors, edgecolor="black", linewidth=0.5)

    for bar, slowdown in zip(bars, slowdowns):
        ax.annotate(
            f"{slowdown:.2f}x" if slowdown > 1.01 else "1.00x",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Steps Per Second (SPS)")
    ax.set_title("Ablation: Marginal Contribution of Each Rust Component")
    ax.axhline(y=full["sps"], color="#E63946", linestyle="--", alpha=0.4, linewidth=1)

    fig.tight_layout()
    out = output_dir / f"fig4_amdahl_breakdown.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Fig 4: {out}")
    return True


def fig5_learning_curves(
    results: list[dict], output_dir: Path, fmt: str
) -> bool:
    """Fig 5: Learning curves with IQM + CI bands.

    One subplot row per (algo, env) pair. Two columns: vs steps, vs wall-clock.
    """
    if not results:
        print("  [SKIP] Fig 5: No convergence results found")
        return False

    # Group by (algo, env) -> framework -> runs
    tasks: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        tasks[(r["algorithm"], r["environment"])][r["framework"]].append(r)

    if not tasks:
        print("  [SKIP] Fig 5: No valid tasks")
        return False

    task_keys = sorted(tasks.keys())
    n_tasks = len(task_keys)

    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 3.5 * n_tasks), squeeze=False)

    for row, (algo, env) in enumerate(task_keys):
        fw_runs = tasks[(algo, env)]
        for col, (x_key, xlabel) in enumerate(
            [("step", "Environment Steps"), ("wall_clock_s", "Wall-Clock Time (s)")]
        ):
            ax = axes[row, col]
            for fw, runs in sorted(fw_runs.items()):
                x, y_mat = _interpolate_curves(runs, x_key)
                if len(x) == 0:
                    continue
                mean, lower, upper = _bootstrap_ci(y_mat)
                style = _fw_style(fw)
                ax.plot(x, mean, color=style["color"], label=style["label"])
                ax.fill_between(x, lower, upper, color=style["color"], alpha=0.2)

            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel("Mean Episodic Return")
            ax.legend(loc="lower right", fontsize=8)
            if col == 0:
                ax.set_title(f"{algo} on {env}", fontweight="bold")

    fig.tight_layout()
    out = output_dir / f"fig5_learning_curves.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Fig 5: {out}")
    return True


def fig6_sps_comparison(
    results: list[dict], output_dir: Path, fmt: str
) -> bool:
    """Fig 6: Grouped bar chart of SPS with error bars."""
    if not results:
        print("  [SKIP] Fig 6: No convergence results")
        return False

    by_task: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        sps = r.get("training_metrics", {}).get("mean_sps", 0.0)
        if sps > 0:
            by_task[(r["algorithm"], r["environment"])][r["framework"]].append(sps)

    # Collect all frameworks present
    all_fws = sorted({fw for fws in by_task.values() for fw in fws})
    if len(all_fws) < 2:
        print("  [SKIP] Fig 6: Need at least 2 frameworks for comparison")
        return False

    task_labels = []
    fw_means: dict[str, list[float]] = {fw: [] for fw in all_fws}
    fw_stds: dict[str, list[float]] = {fw: [] for fw in all_fws}

    for (algo, env), fws in sorted(by_task.items()):
        has_data = sum(1 for fw in all_fws if fw in fws)
        if has_data < 2:
            continue
        task_labels.append(f"{algo}\n{env}")
        for fw in all_fws:
            vals = fws.get(fw, [])
            fw_means[fw].append(np.mean(vals) if vals else 0)
            fw_stds[fw].append(np.std(vals) if vals else 0)

    if not task_labels:
        print("  [SKIP] Fig 6: No tasks with multi-framework data")
        return False

    fig, ax = plt.subplots(figsize=(max(8, len(task_labels) * 1.5), 5.5))
    x = np.arange(len(task_labels))
    width = 0.8 / len(all_fws)

    for i, fw in enumerate(all_fws):
        style = _fw_style(fw)
        offset = (i - len(all_fws) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            fw_means[fw],
            width,
            yerr=fw_stds[fw],
            label=style["label"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            alpha=0.85,
        )

    # Speedup annotations (rlox vs first non-rlox framework)
    if "rlox" in all_fws:
        other = [fw for fw in all_fws if fw != "rlox"]
        if other:
            base_fw = other[0]
            for i in range(len(task_labels)):
                if fw_means[base_fw][i] > 0 and fw_means["rlox"][i] > 0:
                    speedup = fw_means["rlox"][i] / fw_means[base_fw][i]
                    top = max(fw_means["rlox"][i], fw_means[base_fw][i])
                    ax.annotate(
                        f"{speedup:.1f}x",
                        xy=(x[i], top),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=8)
    ax.set_ylabel("Steps Per Second (SPS)")
    ax.set_title("Training Throughput Comparison")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = output_dir / f"fig6_sps_comparison.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Fig 6: {out}")
    return True


def fig7_performance_profiles(
    results: list[dict], output_dir: Path, fmt: str
) -> bool:
    """Fig 7: Agarwal et al. performance profiles."""
    if not results:
        print("  [SKIP] Fig 7: No convergence results")
        return False

    fw_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        score = _final_return(r)
        if np.isnan(score):
            continue
        norm = _normalize_score(score, r["environment"])
        fw_scores[r["framework"]].append(norm)

    if len(fw_scores) < 2:
        print("  [SKIP] Fig 7: Need at least 2 frameworks")
        return False

    taus = np.linspace(0, 1.5, 200)

    fig, ax = plt.subplots(figsize=(7, 5))

    for fw, scores in sorted(fw_scores.items()):
        arr = np.asarray(scores)
        fractions = [float(np.mean(arr >= t)) for t in taus]
        style = _fw_style(fw)
        ax.plot(taus, fractions, color=style["color"], label=style["label"], linewidth=2)

    ax.set_xlabel(r"Normalized Score Threshold ($\tau$)")
    ax.set_ylabel(r"Fraction of Runs $\geq \tau$")
    ax.set_title("Performance Profile (Agarwal et al., 2021)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = output_dir / f"fig7_performance_profiles.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Fig 7: {out}")
    return True


def figs1_ablation_detail(
    ablation: dict[str, Any] | None, output_dir: Path, fmt: str
) -> bool:
    """Fig S1: Detailed ablation with marginal slowdown annotations."""
    if ablation is None:
        print("  [SKIP] Fig S1: No ablation data")
        return False

    # Already covered by fig4 but with a different layout for the appendix:
    # horizontal bars with slowdown factor labeled
    configs = {}
    for name, data in ablation.items():
        if isinstance(data, dict) and "sps" in data:
            configs[name] = data["sps"]

    if len(configs) < 2:
        print("  [SKIP] Fig S1: Insufficient ablation configs")
        return False

    baseline_sps = max(configs.values())
    sorted_configs = sorted(configs.items(), key=lambda x: x[1], reverse=True)
    names = [c[0].replace("_", " ").title() for c in sorted_configs]
    sps_vals = [c[1] for c in sorted_configs]

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.7 * len(names)))
    y_pos = np.arange(len(names))
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(names)))

    bars = ax.barh(y_pos, sps_vals, color=colors, edgecolor="black", linewidth=0.5)

    for bar, sps in zip(bars, sps_vals):
        slowdown = baseline_sps / sps
        label = f"  {sps:,.0f} SPS ({slowdown:.2f}x slowdown)"
        ax.text(
            bar.get_width() + baseline_sps * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Steps Per Second (SPS)")
    ax.set_title("Ablation: Component-Level Contribution")
    ax.set_xlim(0, baseline_sps * 1.5)
    ax.invert_yaxis()

    fig.tight_layout()
    out = output_dir / f"figs1_ablation_detail.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Fig S1: {out}")
    return True


def figs2_probability_of_improvement(
    results: list[dict], output_dir: Path, fmt: str
) -> bool:
    """Fig S2: P(rlox > SB3) per task."""
    if not results:
        print("  [SKIP] Fig S2: No convergence results")
        return False

    by_task: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        score = _final_return(r)
        if not np.isnan(score):
            by_task[(r["algorithm"], r["environment"])][r["framework"]].append(score)

    tasks = []
    probs = []
    rng = np.random.default_rng(42)

    for (algo, env), fw_scores in sorted(by_task.items()):
        rlox_s = np.asarray(fw_scores.get("rlox", []))
        sb3_s = np.asarray(fw_scores.get("sb3", []))
        if len(rlox_s) == 0 or len(sb3_s) == 0:
            continue

        n = min(len(rlox_s), len(sb3_s))
        n_bootstrap = 10_000
        wins = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            wins[b] = float(np.mean(rlox_s[idx]) > np.mean(sb3_s[idx]))

        tasks.append(f"{algo}\n{env}")
        probs.append(float(np.mean(wins)))

    if not tasks:
        print("  [SKIP] Fig S2: No paired rlox vs SB3 tasks")
        return False

    fig, ax = plt.subplots(figsize=(max(7, len(tasks) * 1.2), 5))
    x = np.arange(len(tasks))
    colors = ["#E63946" if p > 0.5 else "#457B9D" for p in probs]

    ax.bar(x, probs, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="P = 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("P(rlox > SB3)")
    ax.set_title("Probability of Improvement")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    out = output_dir / f"figs2_probability_of_improvement.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Fig S2: {out}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures from benchmark results"
    )
    parser.add_argument(
        "results_dir", help="Root directory containing results (with raw/ subdirectory)"
    )
    parser.add_argument("output_dir", help="Output directory for figures")
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format (default: pdf for LaTeX)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Format:            {fmt}")
    print()

    # Load all data sources
    convergence = load_convergence_results(results_dir)
    ablation = load_ablation_results(results_dir)
    component = load_component_benchmarks(results_dir)
    profiling = load_profiling_data(results_dir)

    print(f"Loaded: {len(convergence)} convergence results")
    print(f"Loaded: {'yes' if ablation else 'no'} ablation data")
    print(f"Loaded: {'yes' if component else 'no'} component benchmarks")
    print(f"Loaded: {'yes' if profiling else 'no'} profiling data")
    print()

    if not convergence and not ablation and not component and not profiling:
        print("ERROR: No data found. Expected JSON files in:")
        print(f"  {results_dir}/raw/  (convergence: <fw>_<algo>_<env>_seed<N>.json)")
        print(f"  {results_dir}/raw/ablation_results.json")
        print(f"  {results_dir}/raw/component_benchmarks.json")
        print(f"  {results_dir}/raw/profiling_breakdown.json")
        print()
        print("Run experiments first:")
        print("  python experiments/scripts/run_ablation.py")
        print("  (convergence runners from rlox/benchmarks/convergence/)")
        sys.exit(1)

    # Generate all figures
    generated = 0
    skipped = 0

    figures = [
        ("Fig 1: Pipeline Breakdown", lambda: fig1_pipeline_breakdown(profiling, output_dir, fmt)),
        ("Fig 3: Env Scaling", lambda: fig3_env_scaling(component, output_dir, fmt)),
        ("Fig 4: Amdahl Breakdown", lambda: fig4_amdahl_breakdown(ablation, output_dir, fmt)),
        ("Fig 5: Learning Curves", lambda: fig5_learning_curves(convergence, output_dir, fmt)),
        ("Fig 6: SPS Comparison", lambda: fig6_sps_comparison(convergence, output_dir, fmt)),
        ("Fig 7: Performance Profiles", lambda: fig7_performance_profiles(convergence, output_dir, fmt)),
        ("Fig S1: Ablation Detail", lambda: figs1_ablation_detail(ablation, output_dir, fmt)),
        ("Fig S2: P(Improvement)", lambda: figs2_probability_of_improvement(convergence, output_dir, fmt)),
    ]

    for name, gen_fn in figures:
        print(f"Generating {name}...")
        if gen_fn():
            generated += 1
        else:
            skipped += 1

    print(f"\nDone: {generated} generated, {skipped} skipped (missing data)")
    print(f"Figures saved to: {output_dir}")

    if skipped > 0:
        print(f"\nTo generate all figures, ensure result data exists in {results_dir}/raw/")


if __name__ == "__main__":
    main()
