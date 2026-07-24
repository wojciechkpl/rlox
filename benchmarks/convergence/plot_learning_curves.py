#!/usr/bin/env python3
"""Plot learning curves: mean +/- CI return vs steps and vs wall-clock.

Generates one figure per (algorithm, environment) pair with rlox and SB3
on the same axes. Shaded regions show bootstrapped 95% CI across seeds.

Usage:
    python plot_learning_curves.py results/
    python plot_learning_curves.py results/ --output figures/
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

DEFAULT_N_BOOTSTRAP: int = 2_000
DEFAULT_BOOTSTRAP_SEED: int = 42

FRAMEWORK_COLORS = {"rlox": "#1565c0", "sb3": "#e65100"}  # match paper (blue/orange)
FRAMEWORK_LABELS = {"rlox": "rlox (Rust)", "sb3": "Stable-Baselines3"}


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def _interpolate_curves(
    runs: list[dict],
    x_key: str,
    y_key: str = "mean_return",
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate learning curves to common x-axis for averaging.

    Returns
    -------
    x_common : (n_points,) array
    y_matrix : (n_runs, n_points) array
    """
    # Find common x range
    x_min = max(run["evaluations"][0][x_key] for run in runs if run["evaluations"])
    x_max = min(run["evaluations"][-1][x_key] for run in runs if run["evaluations"])

    x_common = np.linspace(x_min, x_max, n_points)
    y_matrix = np.empty((len(runs), n_points))

    for i, run in enumerate(runs):
        xs = np.array([e[x_key] for e in run["evaluations"]])
        ys = np.array([e[y_key] for e in run["evaluations"]])
        y_matrix[i] = np.interp(x_common, xs, ys)

    return x_common, y_matrix


def _bootstrap_ci(
    y_matrix: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and bootstrapped CI at each x point.

    Returns mean, lower, upper arrays.
    """
    n_runs, n_points = y_matrix.shape
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)

    mean = y_matrix.mean(axis=0)
    lower = np.empty(n_points)
    upper = np.empty(n_points)

    alpha = (1.0 - ci) / 2.0

    for j in range(n_points):
        col = y_matrix[:, j]
        boots = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = rng.choice(col, size=n_runs, replace=True)
            boots[b] = np.mean(sample)
        lower[j] = np.percentile(boots, 100 * alpha)
        upper[j] = np.percentile(boots, 100 * (1 - alpha))

    return mean, lower, upper


def plot_task(
    task_runs: dict[str, list[dict]],
    algo: str,
    env: str,
    output_dir: Path,
) -> None:
    """Plot learning curves for one (algo, env) pair."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for x_key, ax, xlabel in [
        ("step", axes[0], "Environment Steps"),
        ("wall_clock_s", axes[1], "Wall-Clock Time (s)"),
    ]:
        for fw, runs in sorted(task_runs.items()):
            valid_runs = [r for r in runs if len(r.get("evaluations", [])) >= 2]
            if not valid_runs:
                continue

            x, y_mat = _interpolate_curves(valid_runs, x_key)
            mean, lower, upper = _bootstrap_ci(y_mat)

            color = FRAMEWORK_COLORS.get(fw, "#666666")
            label = FRAMEWORK_LABELS.get(fw, fw)

            ax.plot(x, mean, color=color, label=label, linewidth=2)
            ax.fill_between(x, lower, upper, color=color, alpha=0.2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Episodic Return")
        ax.legend(loc="lower right")

    fig.suptitle(f"{algo} on {env}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = output_dir / f"learning_curve_{algo}_{env.replace('/', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot learning curves")
    parser.add_argument("results_dir", help="Directory with JSON result files")
    parser.add_argument("--output", default=None, help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        print("No results found.")
        return

    # Group by (algo, env) -> framework -> list of runs
    tasks: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        tasks[(r["algorithm"], r["environment"])][r["framework"]].append(r)

    for (algo, env), fw_runs in sorted(tasks.items()):
        plot_task(fw_runs, algo, env, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
