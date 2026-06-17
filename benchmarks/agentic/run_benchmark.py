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
