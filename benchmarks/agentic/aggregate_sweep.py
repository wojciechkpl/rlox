#!/usr/bin/env python3
"""Aggregate a completed rlox P3 sweep from the per-run summary.json files.

Reads benchmarks/agentic/sweep_out/runs/<condition>_seed<S>_frac<F>/summary.json
(written by trl_grpo_run.py) and emits p3_summary.{json,md}. Decoupled from the
sweep launcher so re-aggregation needs no GPU.

Public API
----------
aggregate(runs_dir) -> tuple[list[dict], dict]
    Parse all summary.json files under ``runs_dir`` and return:
      - rows: per-(condition, fraction) aggregate stats including reward fields
      - verdict: p3_verdict dict including guardrail block

main() -> int
    CLI entry point: reads from the default sweep output dir and writes
    p3_summary.{json,md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Guard against numpy being absent in minimal envs (CI, test host).
# numpy is only used indirectly via reporting.ci_overlap_check.
try:
    import importlib.util as _ilu

    _HAS_NUMPY = _ilu.find_spec("numpy") is not None
except Exception:
    _HAS_NUMPY = False

REPO = Path("/home/wk/rlox")
OUT = REPO / "benchmarks" / "agentic" / "sweep_out"

# Tolerance for the simple absolute-difference guardrail fallback (when per-seed
# sample lists are available we delegate to ci_overlap_check; otherwise we use
# this threshold).
_GUARDRAIL_ABS_TOL: float = 0.15


# ---------------------------------------------------------------------------
# Core aggregation function (importable for tests)
# ---------------------------------------------------------------------------


def aggregate(runs_dir: Path | str) -> tuple[list[dict], dict]:
    """Aggregate per-run summary.json files into rows + p3_verdict.

    Parameters
    ----------
    runs_dir:
        Directory containing one sub-directory per run, each with a
        ``summary.json`` produced by ``trl_grpo_run.train()``.

    Returns
    -------
    (rows, verdict) where:
        rows    — list[dict]: one dict per (condition, fraction) group with
                  fields: condition, fraction, n, survived,
                  mean_elapsed_secs, mean_completed_steps,
                  mean_reward_last (compat alias),
                  mean_final_reward, mean_reward.
        verdict — dict: p3_verdict with treatment_survives_all_fractions,
                  per-fraction survival/timing, and a ``guardrail`` block.
    """
    runs_dir = Path(runs_dir)

    # ------------------------------------------------------------------
    # Load all summary.json files
    # ------------------------------------------------------------------
    runs: list[dict] = []
    for sj in sorted(runs_dir.glob("*/summary.json")):
        try:
            d = json.loads(sj.read_text())
        except Exception:
            continue
        runs.append(d)

    if not runs:
        return [], {
            "treatment_survives_all_fractions": False,
            "guardrail": {"passed": False},
        }

    # ------------------------------------------------------------------
    # Group by (condition, fraction)
    # ------------------------------------------------------------------
    agg: dict[tuple[str, float], list[dict]] = {}
    for r in runs:
        cond = r.get("backend", "unknown")
        frac = float(r.get("adversarial_fraction", 0.0))
        agg.setdefault((cond, frac), []).append(r)

    # ------------------------------------------------------------------
    # Build per-group rows
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for (cond, frac), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(rs)
        mean_final = (
            sum(
                float(x.get("final_reward", x.get("mean_reward_last", 0.0))) for x in rs
            )
            / n
        )
        mean_rwd = (
            sum(float(x.get("mean_reward", x.get("mean_reward_last", 0.0))) for x in rs)
            / n
        )
        rows.append(
            {
                "condition": cond,
                "fraction": frac,
                "n": n,
                "survived": sum(1 for x in rs if x.get("survived")),
                "mean_elapsed_secs": round(
                    sum(float(x.get("elapsed_secs", 0)) for x in rs) / n, 1
                ),
                "mean_completed_steps": round(
                    sum(int(x.get("completed_steps", 0)) for x in rs) / n, 2
                ),
                # Reward fields — new
                "mean_final_reward": round(mean_final, 4),
                "mean_reward": round(mean_rwd, 4),
                # Backward-compat alias
                "mean_reward_last": round(mean_final, 3),
            }
        )

    def _row(cond: str, frac: float) -> dict | None:
        return next(
            (x for x in rows if x["condition"] == cond and x["fraction"] == frac), None
        )

    # ------------------------------------------------------------------
    # Build p3_verdict
    # ------------------------------------------------------------------
    treat_rows = [x for x in rows if x["condition"] == "rlox"]
    base0 = _row("in_loop", 0.0)

    verdict: dict = {
        "treatment_survives_all_fractions": bool(treat_rows)
        and all(x["survived"] == x["n"] for x in treat_rows),
    }

    # Per-fraction survival / timing details
    for frac in (0.05, 0.10):
        b, t = _row("in_loop", frac), _row("rlox", frac)
        if b and t and base0:
            verdict[f"frac_{frac}"] = {
                "baseline_survived": f"{b['survived']}/{b['n']}",
                "treatment_survived": f"{t['survived']}/{t['n']}",
                "baseline_elapsed_x_vs_clean_baseline": round(
                    b["mean_elapsed_secs"] / max(base0["mean_elapsed_secs"], 1e-9), 2
                ),
                "baseline_slower_than_treatment_x": round(
                    b["mean_elapsed_secs"] / max(t["mean_elapsed_secs"], 1e-9), 2
                ),
            }

    # ------------------------------------------------------------------
    # Guardrail block: quality-parity check at fraction=0.0
    # ------------------------------------------------------------------
    verdict["guardrail"] = _compute_guardrail(agg, base0, _row("rlox", 0.0), runs_dir)

    return rows, verdict


def _compute_guardrail(
    agg: dict[tuple[str, float], list[dict]],
    base0_row: dict | None,
    treat0_row: dict | None,
    runs_dir: Path,
) -> dict:
    """Compute the quality-parity guardrail for fraction=0.0.

    Strategy:
    1. Collect per-seed final_reward values for Baseline and Treatment at frac=0.0.
    2. If numpy is available, delegate to ci_overlap_check for a statistically
       sound check.  Otherwise fall back to an absolute-difference tolerance.
    3. Report baseline_mean_final_reward, treatment_mean_final_reward, abs_diff,
       and passed (bool).

    The guardrail PASSES when Treatment and Baseline achieve comparable reward
    at fraction=0.0 (i.e., Treatment has not hurt learning).
    """
    baseline_frac0 = agg.get(("in_loop", 0.0), [])
    treatment_frac0 = agg.get(("rlox", 0.0), [])

    if not baseline_frac0 or not treatment_frac0:
        # Insufficient data — mark unavailable but keep the block present.
        return {
            "passed": False,
            "reason": "insufficient_data_at_frac_0",
            "baseline_mean_final_reward": None,
            "treatment_mean_final_reward": None,
            "abs_diff": None,
        }

    def _final_reward(r: dict) -> float:
        return float(r.get("final_reward", r.get("mean_reward_last", 0.0)))

    baseline_rewards = [_final_reward(r) for r in baseline_frac0]
    treatment_rewards = [_final_reward(r) for r in treatment_frac0]

    baseline_mean = sum(baseline_rewards) / len(baseline_rewards)
    treatment_mean = sum(treatment_rewards) / len(treatment_rewards)
    abs_diff = abs(baseline_mean - treatment_mean)

    # Try CI-overlap approach (requires numpy and reporting module).
    ci_result: dict | None = None
    if _HAS_NUMPY:
        try:
            # reporting.py lives in python/rlox/agentic/; add its parent to path
            # if not already present.  We resolve relative to this file's location.
            _reporting_dir = runs_dir.parents[2] / "python" / "rlox" / "agentic"
            _reporting_str = str(_reporting_dir)
            if _reporting_str not in sys.path:
                sys.path.insert(0, _reporting_str)

            from reporting import ci_overlap_check  # type: ignore[import]

            # ci_overlap_check expects list[list[float]] — one sub-list per seed.
            # Each seed contributes one final_reward value; wrap in a list to form
            # a single-element "sample" (bootstrap still works on n=1 but CIs will
            # be degenerate; this is intentional — when only 1 seed the CI
            # collapses to a point and overlap is equivalent to value equality).
            baseline_by_seed = [[v] for v in baseline_rewards]
            treatment_by_seed = [[v] for v in treatment_rewards]

            # Align lists to the shorter of the two (in case seed counts differ).
            min_len = min(len(baseline_by_seed), len(treatment_by_seed))
            ci_result = ci_overlap_check(
                baseline_by_seed[:min_len],
                treatment_by_seed[:min_len],
            )
        except Exception:
            ci_result = None

    if ci_result is not None:
        # ci_overlap_check passes if >= 2/3 seed pairs overlap.  However,
        # when each seed contributes only 1 sample (final_reward only), the
        # bootstrap CI degenerates to a point interval and will never overlap
        # unless values are identical.  In that degenerate case fall back to
        # the abs-diff check so the guardrail remains meaningful.
        n_pairs = ci_result.get("n_pairs", 0)
        all_degenerate = n_pairs > 0 and all(
            e["baseline_ci"][0] == e["baseline_ci"][1]
            and e["treatment_ci"][0] == e["treatment_ci"][1]
            for e in ci_result.get("per_seed", [])
        )
        if all_degenerate:
            passed = abs_diff <= _GUARDRAIL_ABS_TOL
            method = (
                f"abs_diff_tolerance_{_GUARDRAIL_ABS_TOL}_fallback_from_degenerate_ci"
            )
            detail = {"ci_degenerate": True}
        else:
            passed = bool(ci_result["passed"])
            method = "ci_overlap_check"
            detail = {k: ci_result[k] for k in ("n_overlap", "n_pairs", "per_seed")}
    else:
        # Fallback: simple absolute-difference tolerance.
        passed = abs_diff <= _GUARDRAIL_ABS_TOL
        method = f"abs_diff_tolerance_{_GUARDRAIL_ABS_TOL}"
        detail = {}

    result: dict = {
        "passed": passed,
        "method": method,
        "baseline_mean_final_reward": round(baseline_mean, 4),
        "treatment_mean_final_reward": round(treatment_mean, 4),
        "abs_diff": round(abs_diff, 4),
    }
    result.update(detail)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    rows, verdict = aggregate(OUT / "runs")

    if not rows:
        print("no summaries found under", OUT / "runs")
        return 1

    n_runs = sum(r["n"] for r in rows)

    # Write JSON output
    summary = {
        "n_runs": n_runs,
        "aggregate": rows,
        "p3_verdict": verdict,
    }
    (OUT / "p3_summary.json").write_text(json.dumps(summary, indent=2))

    # Write Markdown table (extended with reward columns)
    md = [
        "# rlox benchmark — P3 sweep result",
        "",
        f"{n_runs} runs (2 conditions x seeds x fractions), Qwen3-4B-Instruct-2507 + LoRA, "
        "6 GRPO steps each, single RTX 5090.",
        "",
        "P3 = adversarial-code containment. **Treatment** = rlox sandbox (`/verify`, out-of-process). "
        "**Baseline** = in-process `in_loop` exec (scope-bounded for host safety).",
        "",
        "| condition | injection | survived | mean steps | mean final reward | mean reward | mean elapsed (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in rows:
        md.append(
            f"| {x['condition']} | {x['fraction']:.2f} | {x['survived']}/{x['n']} | "
            f"{x['mean_completed_steps']} | {x['mean_final_reward']} | "
            f"{x['mean_reward']} | {x['mean_elapsed_secs']} |"
        )
    md += ["", "## P3 verdict", "```json", json.dumps(verdict, indent=2), "```"]
    (OUT / "p3_summary.md").write_text("\n".join(md) + "\n")

    # Console output
    print(f"=== P3 SWEEP AGGREGATE ({n_runs} runs) ===")
    for x in rows:
        print(
            f"  {x['condition']:8s} frac={x['fraction']:.2f}  "
            f"survived={x['survived']}/{x['n']}  "
            f"steps={x['mean_completed_steps']}  "
            f"final_reward={x['mean_final_reward']}  "
            f"mean_reward={x['mean_reward']}  "
            f"elapsed={x['mean_elapsed_secs']}s"
        )
    print("verdict:", json.dumps(verdict))
    g = verdict.get("guardrail", {})
    print(
        f"guardrail: passed={g.get('passed')}  "
        f"baseline={g.get('baseline_mean_final_reward')}  "
        f"treatment={g.get('treatment_mean_final_reward')}  "
        f"abs_diff={g.get('abs_diff')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
