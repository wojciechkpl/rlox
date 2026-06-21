# benchmarks/agentic/run_benchmark.py
#
# Component 8 (sweep driver): run_sweep — outer driver that iterates the full
# grid of conditions × seeds × adversarial_fractions (AC-3 / AC-4).
#
# Import constraints: stdlib only — no torch, no vllm, no pydantic.
# This module must be importable with python3 alone (yaml loaded via config.py).
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_CONDITIONS: list[str] = ["in_loop", "rlox"]

# Timeout for a single in_loop (Baseline) run.  The systemd-run wrapper
# uses this so a runaway training process does not starve the host.
_IN_LOOP_TIMEOUT_SECS: int = 7200  # 2 hours


def make_trl_run_one(
    *,
    max_steps: int,
    group_size: int,
    rlox_server_url: str,
    corpus_path: str | Path,
    output_root: str | Path,
    venv_python: str | Path,
    repo_root: str | Path,
    scope_for_baseline: bool = True,
) -> Callable[[str, int, float], dict]:
    """Build a ``run_one(condition, seed, fraction) -> dict`` callable.

    The returned callable launches ``trl_grpo_run.py`` as a subprocess with
    the appropriate flags for the given condition/seed/fraction triple.

    For the ``in_loop`` (Baseline) condition the command is wrapped in
    ``systemd-run --user --scope`` to bound its memory and task count, and in
    ``timeout`` to cap wall-clock time.  The ``rlox`` (Treatment) condition
    is launched directly (the rlox server provides its own resource isolation).

    Returns a dict with:
        ``{condition, seed, fraction, survived, completed_steps,
           elapsed_secs, mean_reward_last}``

    ``survived=False`` is returned (without raising) if:
    * the subprocess exits with a non-zero return code
    * no ``summary.json`` was written by the runner
    * the subprocess times out
    """
    _venv_python = str(venv_python)
    _repo_root = Path(repo_root)
    _runner_script = str(_repo_root / "benchmarks" / "agentic" / "trl_grpo_run.py")
    _output_root = Path(output_root)
    _corpus_path = str(corpus_path)

    def run_one(condition: str, seed: int, fraction: float) -> dict:
        run_label = f"{condition}_seed{seed}_frac{fraction}"
        run_output_dir = str(_output_root / run_label)

        base_cmd: list[str] = [
            _venv_python,
            _runner_script,
            "--backend", condition,
            "--adversarial-fraction", str(fraction),
            "--seed", str(seed),
            "--max-steps", str(max_steps),
            "--group-size", str(group_size),
            "--adversarial-corpus", _corpus_path,
            "--rlox-server-url", rlox_server_url,
            "--output-dir", run_output_dir,
        ]

        if condition == "in_loop" and scope_for_baseline:
            # Wrap in systemd-run scope for host-safety resource limits.
            cmd: list[str] = [
                "systemd-run",
                "--user",
                "--scope",
                "-p", "TasksMax=2048",
                "-p", "MemoryMax=40G",
                "--quiet",
                "timeout",
                str(_IN_LOOP_TIMEOUT_SECS),
                *base_cmd,
            ]
        else:
            cmd = base_cmd

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=False,
            )
            returncode = proc.returncode
        except Exception as exc:
            logger.error("run_one(%s, %d, %.2f) subprocess error: %s", condition, seed, fraction, exc)
            elapsed = time.monotonic() - t0
            return {
                "condition": condition,
                "seed": seed,
                "fraction": fraction,
                "survived": False,
                "completed_steps": 0,
                "elapsed_secs": round(elapsed, 2),
                "mean_reward_last": 0.0,
            }

        elapsed = time.monotonic() - t0
        summary_path = Path(run_output_dir) / "summary.json"

        if returncode != 0 or not summary_path.exists():
            logger.warning(
                "run_one(%s, %d, %.2f) failed: returncode=%d summary_exists=%s",
                condition, seed, fraction, returncode, summary_path.exists(),
            )
            return {
                "condition": condition,
                "seed": seed,
                "fraction": fraction,
                "survived": False,
                "completed_steps": 0,
                "elapsed_secs": round(elapsed, 2),
                "mean_reward_last": 0.0,
            }

        with summary_path.open(encoding="utf-8") as fh:
            summary = json.load(fh)

        return {
            "condition": condition,
            "seed": seed,
            "fraction": fraction,
            "survived": bool(summary.get("survived", False)),
            "completed_steps": int(summary.get("completed_steps", 0)),
            "elapsed_secs": round(elapsed, 2),
            "mean_reward_last": float(summary.get("mean_reward_last", 0.0)),
        }

    return run_one


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

    # Make rlox_agent importable for config when run as a script.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))
    import rlox_agent.config as _cfg  # noqa: E402

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
