# benchmarks/agentic/run_benchmark.py
#
# Component 8 (sweep driver): run_sweep — outer driver that iterates the full
# grid of conditions × seeds × adversarial_fractions (AC-3 / AC-4).
#
# Import constraints: stdlib only — no torch, no vllm, no pydantic.
# This module must be importable with python3 alone (yaml loaded via config.py).
from __future__ import annotations

import json
import os
from typing import Callable

_CONDITIONS: list[str] = ["in_loop", "rlox"]


def run_sweep(
    config,
    run_one: Callable[[str, int, float], dict],
    metric_store_dir: str,
) -> list[dict]:
    """Execute the full benchmark sweep.

    Iterates the full grid:
        conditions = ["in_loop", "rlox"]
        seeds      = range(config.n_seeds)
        fractions  = config.adversarial_fractions

    For each (condition, seed, fraction) triple, calls::

        result = run_one(condition, seed, fraction)

    and records whether the run survived. A ``run_one`` that raises is recorded
    as ``survived=False`` (the sweep continues — no re-raise).

    Per-run result dicts are written as individual artifacts into
    ``metric_store_dir`` (one JSON file per run).

    Args:
        config: a BenchmarkConfig instance (n_seeds, adversarial_fractions).
        run_one: injectable callable — receives (condition: str, seed: int,
            fraction: float) and returns a result dict that MUST contain at
            minimum ``{"survived": bool}``.  In production this launches
            prime-rl; in tests it is a mock.
        metric_store_dir: directory path where per-run JSON artifacts are written.

    Returns:
        List of result dicts, one per grid point, in deterministic iteration
        order: outer loop over conditions, then seeds, then fractions.
        Each dict contains at minimum:
            ``{"condition": str, "seed": int, "fraction": float,
               "survived": bool}``.
    """
    os.makedirs(metric_store_dir, exist_ok=True)

    results: list[dict] = []
    run_index = 0

    for condition in _CONDITIONS:
        for seed in range(config.n_seeds):
            for fraction in config.adversarial_fractions:
                survived: bool
                try:
                    raw = run_one(condition, seed, fraction)
                    survived = bool(raw.get("survived", True))
                except Exception:
                    survived = False

                record: dict = {
                    "condition": condition,
                    "seed": seed,
                    "fraction": fraction,
                    "survived": survived,
                }

                artifact_name = (
                    f"run_{run_index:05d}"
                    f"_{condition}"
                    f"_seed{seed}"
                    f"_frac{fraction}.json"
                )
                artifact_path = os.path.join(metric_store_dir, artifact_name)
                with open(artifact_path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh)

                results.append(record)
                run_index += 1

    return results


def _build_grid(config) -> list[dict]:
    """Return the full sweep grid (deterministic order) without executing it."""
    grid: list[dict] = []
    for condition in _CONDITIONS:
        for seed in range(config.n_seeds):
            for fraction in config.adversarial_fractions:
                grid.append(
                    {"condition": condition, "seed": seed, "fraction": fraction}
                )
    return grid


def _main(argv=None) -> int:
    import argparse
    import sys
    from pathlib import Path

    # Make python/rlox/agentic importable for config.py when run as a script.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "rlox" / "agentic"))
    import config as _cfg  # noqa: E402

    p = argparse.ArgumentParser(description="rlox agentic benchmark sweep driver")
    p.add_argument("--config", required=True, help="path to benchmark_v*.yaml")
    p.add_argument("--metric-store", required=True, help="output dir for per-run artifacts")
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned run grid and exit (no execution, no GPU)")
    args = p.parse_args(argv)

    cfg = _cfg.load_config(args.config)
    grid = _build_grid(cfg)

    if args.dry_run:
        print(f"[dry-run] sweep grid: {len(grid)} runs "
              f"(2 conditions x {cfg.n_seeds} seeds x "
              f"{len(cfg.adversarial_fractions)} fractions)")
        for i, pt in enumerate(grid):
            print(f"  {i:3d}  {pt['condition']:8s} seed={pt['seed']} "
                  f"fraction={pt['fraction']}")
        return 0

    # Real execution path: enforce the pre-registered config (AC-2) first.
    _cfg.validate_config(cfg)
    raise SystemExit(
        "The real run_one (prime-rl GRPO launcher) is wired at Step 8. "
        "Use --dry-run to preview the grid."
    )


if __name__ == "__main__":
    raise SystemExit(_main())
