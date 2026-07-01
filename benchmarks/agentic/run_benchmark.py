# benchmarks/agentic/run_benchmark.py
#
# Component 8 (sweep driver): run_sweep — outer driver that iterates the full
# grid of conditions × seeds × adversarial_fractions (AC-3 / AC-4).
#
# Import constraints: stdlib only — no torch, no vllm, no pydantic.
# This module must be importable with python3 alone (yaml loaded via config.py).
# tomllib (3.11+ stdlib) and tomli_w are imported locally inside functions that
# need them, so the module stays importable without those deps at module level.
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

# Compiled patterns for _parse_rl_log — re is stdlib, safe at module level.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
# Matches: Step <N> | ... | Reward <float> | ...
# The reward group accepts only a proper float (sign, digits, optional decimal)
# to avoid spurious matches on truncated / interleaved teardown lines.
_STEP_LINE_RE = re.compile(r"Step\s+(\d+)\s*\|.*?\|\s*Reward\s+([-+]?\d+(?:\.\d+)?)")

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
            "--backend",
            condition,
            "--adversarial-fraction",
            str(fraction),
            "--seed",
            str(seed),
            "--max-steps",
            str(max_steps),
            "--group-size",
            str(group_size),
            "--adversarial-corpus",
            _corpus_path,
            "--rlox-server-url",
            rlox_server_url,
            "--output-dir",
            run_output_dir,
        ]

        if condition == "in_loop" and scope_for_baseline:
            # Wrap in systemd-run scope for host-safety resource limits.
            cmd: list[str] = [
                "systemd-run",
                "--user",
                "--scope",
                "-p",
                "TasksMax=2048",
                "-p",
                "MemoryMax=40G",
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
            logger.error(
                "run_one(%s, %d, %.2f) subprocess error: %s",
                condition,
                seed,
                fraction,
                exc,
            )
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
                condition,
                seed,
                fraction,
                returncode,
                summary_path.exists(),
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


# ---------------------------------------------------------------------------
# prime-rl GRPO launcher helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into a copy of *base*.

    - Dict values are merged recursively (not replaced wholesale).
    - List values from *overrides* replace the corresponding base list.
    - All other scalar values from *overrides* replace the base value.
    """
    result: dict[str, Any] = dict(base)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _render_run_toml(
    base_toml: str | Path,
    run_dir: Path,
    condition: str,
    seed: int,
    fraction: float,
    max_steps: int,
    rlox_server_url: str,
) -> Path:
    """Deep-merge base TOML with per-run overrides and write ``run_dir/rl.gen.toml``.

    Overrides applied:
    - ``output_dir`` = str(run_dir)
    - ``max_steps`` = max_steps
    - ``orchestrator.train.env[0].args.rollout_backend`` = condition
    - ``orchestrator.train.env[0].args.adversarial_fraction`` = fraction
    - ``orchestrator.train.env[0].args.rlox_server_url`` = rlox_server_url
      (points the Treatment ``rlox`` backend at the running verify server
      instead of the env's built-in default; harmless for the ``in_loop`` Baseline)
    - ``wandb.name`` = per-run label encoding condition / seed / fraction

    All unrelated base keys are preserved.
    """
    import tomllib
    import tomli_w

    with open(base_toml, "rb") as fh:
        base = tomllib.load(fh)

    run_label = f"{condition}_seed{seed}_frac{fraction}"
    wandb_name = f"primerl-{run_label}"

    # Build overrides dict; env list requires special handling because the
    # env is a TOML array-of-tables ([[orchestrator.train.env]]).
    base_envs: list[dict] = base.get("orchestrator", {}).get("train", {}).get("env", [])
    if base_envs:
        merged_env0 = dict(base_envs[0])
        merged_args = dict(merged_env0.get("args", {}))
        merged_args["rollout_backend"] = condition
        merged_args["adversarial_fraction"] = fraction
        merged_args["rlox_server_url"] = rlox_server_url
        merged_env0["args"] = merged_args
        new_envs = [merged_env0, *base_envs[1:]]
    else:
        logger.warning(
            "_render_run_toml: base TOML has no orchestrator.train.env entries — "
            "rendered TOML will have an empty env list; this is likely a config error."
        )
        new_envs = []

    overrides: dict[str, Any] = {
        "output_dir": str(run_dir),
        "max_steps": max_steps,
        "wandb": {"name": wandb_name},
        "orchestrator": {
            "train": {
                "env": new_envs,
            }
        },
    }

    merged = _deep_merge(base, overrides)

    dest = run_dir / "rl.gen.toml"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        tomli_w.dump(merged, fh)
    return dest


def _count_completed_steps(output_dir: Path) -> int:
    """Count ``rollouts/step_<N>/`` subdirs written by prime-rl.

    Only directories whose name suffix is purely numeric are counted to avoid
    future ``step_N_checkpoint``-style dirs inflating the count.  Note: counts
    cardinality of matching dirs — a reused output_root with leftover dirs from
    a prior run would overcount; clean sweeps use unique per-run labels so this
    is not a live concern.
    """
    rollouts_dir = output_dir / "rollouts"
    if not rollouts_dir.is_dir():
        return 0
    return sum(
        1
        for d in rollouts_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("step_")
        and d.name[len("step_") :].isdigit()
    )


def _parse_rl_log(log_path: Path) -> tuple[int, float | None]:
    """Parse ``rl.log`` and return ``(step_count, last_reward)``.

    Scans for lines matching the prime-rl progress format::

        Step <N> |   11.6s | Reward <X.XXXX> | ...

    ANSI colour escapes are stripped before matching so both plain and coloured
    output are handled.

    Returns:
        (step_count, last_reward) where step_count is the number of matched
        Step lines and last_reward is the reward on the highest-N Step line,
        or ``None`` when no Step lines are found.

    This function is total — it never raises.  Every error path (missing file,
    encoding errors, malformed reward token) is handled locally.  When a Step
    line is matched but its reward token fails to parse as a float, the step is
    still counted (it happened) but the reward for that line is skipped; the
    last VALID reward on the highest-indexed Step line is returned.
    """
    if not log_path.is_file():
        return 0, None

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0, None

    best_step: int = -1
    best_reward: float | None = None
    count = 0

    for line in text.splitlines():
        clean = _ANSI_ESCAPE_RE.sub("", line)
        m = _STEP_LINE_RE.search(clean)
        if m is None:
            continue
        count += 1
        step_idx = int(m.group(1))
        try:
            reward = float(m.group(2))
        except ValueError:
            # Malformed reward token: count the step but skip the reward update.
            continue
        if step_idx > best_step:
            best_step = step_idx
            best_reward = reward

    return count, best_reward


def _last_step_mean_reward(output_dir: Path) -> float:
    """Mean of the ``reward`` field in the highest-numbered step's train_rollouts.jsonl.

    Returns 0.0 (never raises) when:
    - the rollouts dir is absent
    - no step dirs exist
    - the JSONL rows lack a ``reward`` key
    """
    rollouts_dir = output_dir / "rollouts"
    if not rollouts_dir.is_dir():
        return 0.0

    step_dirs = sorted(
        (
            d
            for d in rollouts_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("step_")
            and d.name[len("step_") :].isdigit()
        ),
        key=lambda d: int(d.name[len("step_") :]),
    )
    if not step_dirs:
        return 0.0

    last_step_dir = step_dirs[-1]
    jsonl_path = last_step_dir / "train_rollouts.jsonl"
    if not jsonl_path.is_file():
        return 0.0

    rewards: list[float] = []
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "reward" in row:
                rewards.append(float(row["reward"]))
    except Exception:
        return 0.0

    return float(sum(rewards) / len(rewards)) if rewards else 0.0


def make_primerl_run_one(
    *,
    base_toml: str | Path,
    max_steps: int,
    group_size: int,
    rlox_server_url: str,
    prime_rl_bin: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    scope_for_baseline: bool = True,
    in_loop_timeout_secs: int = 7200,
) -> Callable[[str, int, float], dict]:
    """Build a ``run_one(condition, seed, fraction) -> dict`` callable for prime-rl.

    The returned callable launches ``<prime_rl_bin> @ <run_dir>/rl.gen.toml``
    for the given condition/seed/fraction triple.

    For the ``in_loop`` (Baseline) condition with ``scope_for_baseline=True``
    the command is wrapped in ``systemd-run --user --scope`` to bound memory
    and task count, and in ``timeout`` to cap wall-clock time.

    The ``rlox`` (Treatment) condition is launched directly.

    Per-run TOML is rendered by deep-merging *base_toml* with run-specific
    overrides (rollout_backend, adversarial_fraction, max_steps, output_dir,
    wandb.name).  The rendered file is written to ``<run_dir>/rl.gen.toml``
    and preserved as a run artifact.

    Returns a dict with EXACTLY the same seven-key shape as ``make_trl_run_one``:
        ``{condition, seed, fraction, survived, completed_steps,
           elapsed_secs, mean_reward_last}``

    ``survived=False`` is returned (without raising) if:
    * subprocess exits with non-zero return code
    * completed_steps < max_steps after the run
    * subprocess raises any exception (TimeoutExpired, OSError, …)
    """
    _base_toml = Path(base_toml)
    _prime_rl_bin = str(prime_rl_bin)
    _output_root = Path(output_root)

    def run_one(condition: str, seed: int, fraction: float) -> dict:
        run_label = f"{condition}_seed{seed}_frac{fraction}"
        run_dir = _output_root / run_label

        # Render per-run TOML before launching the subprocess.
        try:
            toml_path = _render_run_toml(
                _base_toml,
                run_dir,
                condition,
                seed,
                fraction,
                max_steps,
                rlox_server_url,
            )
        except Exception as exc:
            logger.error(
                "run_one(%s, %d, %.2f) TOML render error: %s",
                condition,
                seed,
                fraction,
                exc,
            )
            return {
                "condition": condition,
                "seed": seed,
                "fraction": fraction,
                "survived": False,
                "completed_steps": 0,
                "elapsed_secs": 0.0,
                "mean_reward_last": 0.0,
            }

        # Build the subprocess command.
        base_cmd: list[str] = [_prime_rl_bin, "@", str(toml_path)]

        if condition == "in_loop" and scope_for_baseline:
            cmd: list[str] = [
                "systemd-run",
                "--user",
                "--scope",
                "-p",
                "TasksMax=2048",
                "-p",
                "MemoryMax=40G",
                "--quiet",
                "timeout",
                str(in_loop_timeout_secs),
                *base_cmd,
            ]
        else:
            cmd = base_cmd

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        # The prime-rl ``rl`` launcher spawns its child ``orchestrator`` /
        # ``trainer`` entrypoints by bare command name (resolved via PATH). When
        # ``rl`` is invoked by absolute path (not ``uv run``), its venv bin dir is
        # not on PATH, so prepend it so the children resolve.
        _bin_dir = os.path.dirname(_prime_rl_bin)
        if _bin_dir:
            env["PATH"] = _bin_dir + os.pathsep + env.get("PATH", "")

        log_path = run_dir / "rl.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        try:
            with open(log_path, "w", encoding="utf-8") as _lf:
                subprocess.run(cmd, env=env, stdout=_lf, stderr=subprocess.STDOUT)
        except Exception as exc:
            logger.error(
                "run_one(%s, %d, %.2f) subprocess error: %s",
                condition,
                seed,
                fraction,
                exc,
            )
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

        # Prefer log-based step count; fall back to rollout-dir count; take max.
        log_step_count, log_last_reward = _parse_rl_log(log_path)
        dir_step_count = _count_completed_steps(run_dir)
        completed_steps = max(log_step_count, dir_step_count)

        # Prefer reward from the log; fall back to JSONL parse.
        if log_last_reward is not None:
            mean_reward = log_last_reward
        else:
            mean_reward = _last_step_mean_reward(run_dir)

        # Survival is purely step-count based — exit code is NOT consulted.
        # prime-rl exits 143 (SIGTERM of child processes) on a clean finish.
        survived = completed_steps >= max_steps

        if not survived:
            logger.warning(
                "run_one(%s, %d, %.2f) did not survive: completed_steps=%d/%d",
                condition,
                seed,
                fraction,
                completed_steps,
                max_steps,
            )

        return {
            "condition": condition,
            "seed": seed,
            "fraction": fraction,
            "survived": bool(survived),
            "completed_steps": int(completed_steps),
            "elapsed_secs": round(elapsed, 2),
            "mean_reward_last": float(mean_reward),
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
                raw: dict
                try:
                    raw = run_one(condition, seed, fraction)
                    survived = bool(raw.get("survived", False))
                except Exception:
                    survived = False
                    raw = {}

                # Preserve every key returned by run_one; then enforce the
                # identity keys so they are always consistent in the record.
                record: dict = dict(raw) if isinstance(raw, dict) else {}
                record.update(
                    {
                        "condition": condition,
                        "seed": seed,
                        "fraction": fraction,
                        "survived": survived,
                    }
                )

                artifact_name = (
                    f"run_{run_index:05d}_{condition}_seed{seed}_frac{fraction}.json"
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
    p.add_argument(
        "--metric-store", required=True, help="output dir for per-run artifacts"
    )
    p.add_argument(
        "--host",
        choices=["trl", "primerl"],
        default="trl",
        help="which launcher to use: 'trl' (TRL GRPO runner, default) or 'primerl'",
    )
    p.add_argument(
        "--prime-rl-bin",
        default="/home/wk/prime-rl/.venv/bin/rl",
        help="path to the prime-rl 'rl' binary (only used with --host primerl)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned run grid and exit (no execution, no GPU)",
    )
    args = p.parse_args(argv)

    cfg = _cfg.load_config(args.config)
    grid = _build_grid(cfg)

    if args.dry_run:
        print(
            f"[dry-run] sweep grid: {len(grid)} runs "
            f"(2 conditions x {cfg.n_seeds} seeds x "
            f"{len(cfg.adversarial_fractions)} fractions)"
        )
        for i, pt in enumerate(grid):
            print(
                f"  {i:3d}  {pt['condition']:8s} seed={pt['seed']} "
                f"fraction={pt['fraction']}"
            )
        return 0

    # Real execution path: enforce the pre-registered config (AC-2) first.
    _cfg.validate_config(cfg)

    if args.host == "primerl":
        _primerl_toml = Path(__file__).resolve().parent / "primerl" / "smoke.toml"
        run_one = make_primerl_run_one(
            base_toml=str(_primerl_toml),
            max_steps=cfg.max_steps,
            group_size=cfg.group_size,
            rlox_server_url=cfg.rlox_server_url,
            prime_rl_bin=args.prime_rl_bin,
            output_root=args.metric_store,
            repo_root=repo_root,
        )
    else:
        # TRL host (default).
        run_one = make_trl_run_one(
            max_steps=cfg.max_steps,
            group_size=cfg.group_size,
            rlox_server_url=cfg.rlox_server_url,
            corpus_path=str(
                repo_root / "benchmarks/agentic/corpus/adversarial_corpus_v1.json"
            ),
            output_root=args.metric_store,
            venv_python=str(repo_root / ".venv/bin/python"),
            repo_root=str(repo_root),
        )

    results = run_sweep(cfg, run_one, args.metric_store)
    print(f"[sweep] {len(results)} runs complete. Results in {args.metric_store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
