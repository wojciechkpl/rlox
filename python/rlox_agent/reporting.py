# python/rlox_agent/reporting.py
#
# Step 7b: statistical reporting and quality-gate layer.
#
# Implements:
#   bootstrap_ci   — percentile bootstrap 95% CI on the mean (AC-7)
#   ci_overlap_check — AC-9 guardrail: >=2/3 seed-pair CI overlaps on pass@1
#   write_summary  — writes summary.json + summary.csv (AC-7 machine-readable)
#   assess_go_no_go — pre-registered P1/P3/guardrail thresholds (PRD Success Metrics)
#
# Import constraints: stdlib + numpy ONLY.  No torch, no scipy, no vllm.
# Bootstrap is implemented with numpy.random.default_rng (deterministic by seed).
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "bootstrap_ci",
    "ci_overlap_check",
    "write_summary",
    "assess_go_no_go",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PRIMARY_METRICS = ("gpu_util", "rollouts_per_sec", "contagion_events", "pass_at_1")


def _intervals_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """Return True iff [a_lo, a_hi] and [b_lo, b_hi] overlap.

    Two intervals [a, b] and [c, d] overlap iff a <= d and c <= b.
    """
    return a_lo <= b_hi and b_lo <= a_hi


# ---------------------------------------------------------------------------
# 1. bootstrap_ci
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Any,
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean.

    Parameters
    ----------
    values:
        1-D array-like of numeric samples.
    confidence:
        Coverage level, e.g. 0.95 for a 95% CI.
    n_resamples:
        Number of bootstrap resamples.  Default 1000 (AC-7 minimum).
    seed:
        Integer seed for numpy.random.default_rng — makes the result
        DETERMINISTIC for a given (values, confidence, n_resamples, seed).

    Returns
    -------
    (low, high) : tuple[float, float]
        low  <= mean(values) <= high  (for typical data)
        For all-equal data: low == high == that value.
    """
    arr = np.asarray(values, dtype=float)

    # Degenerate cases: single element or all-equal values.
    if arr.size <= 1 or float(arr.min()) == float(arr.max()):
        v = float(arr.mean())
        return (v, v)

    rng = np.random.default_rng(seed)
    # Draw n_resamples bootstrap samples and compute mean of each.
    indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[indices].mean(axis=1)

    alpha = 1.0 - confidence
    low = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return (low, high)


# ---------------------------------------------------------------------------
# 2. ci_overlap_check
# ---------------------------------------------------------------------------

def ci_overlap_check(
    baseline_pass1_by_seed: list[list[float]],
    treatment_pass1_by_seed: list[list[float]],
    *,
    seed: int = 0,
) -> dict:
    """AC-9 quality-parity guardrail.

    For each aligned seed pair (index i), compute 95% bootstrap CIs on the
    per-seed final pass@1 sample lists and check whether the two CIs overlap.
    The guardrail PASSES if CIs overlap for >= 2 of 3 seed pairs.

    Parameters
    ----------
    baseline_pass1_by_seed:
        list of length n_seeds; each element is a list/array of float pass@1
        samples for that seed.
    treatment_pass1_by_seed:
        Same shape, aligned by seed index.
    seed:
        RNG seed forwarded to bootstrap_ci for each pair.

    Returns
    -------
    dict with keys:
        "passed"   : bool    — True iff n_overlap >= 2
        "n_overlap": int     — number of seed pairs whose CIs overlap
        "n_pairs"  : int     — total seed pairs evaluated
        "per_seed" : list[dict] — one entry per pair:
                        {"seed_idx": int,
                         "baseline_ci": (float, float),
                         "treatment_ci": (float, float),
                         "overlap": bool}
    """
    n_pairs = len(baseline_pass1_by_seed)
    per_seed: list[dict] = []

    for i, (b_samples, t_samples) in enumerate(
        zip(baseline_pass1_by_seed, treatment_pass1_by_seed)
    ):
        b_ci = bootstrap_ci(b_samples, confidence=0.95, seed=seed)
        t_ci = bootstrap_ci(t_samples, confidence=0.95, seed=seed)
        overlap = _intervals_overlap(b_ci[0], b_ci[1], t_ci[0], t_ci[1])
        per_seed.append(
            {
                "seed_idx": i,
                "baseline_ci": b_ci,
                "treatment_ci": t_ci,
                "overlap": overlap,
            }
        )

    n_overlap = sum(1 for e in per_seed if e["overlap"])
    return {
        "passed": bool(n_overlap >= 2),
        "n_overlap": n_overlap,
        "n_pairs": n_pairs,
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# 3. write_summary
# ---------------------------------------------------------------------------

def write_summary(results: dict, out_dir: Any) -> tuple[str, str]:
    """Write AC-7 machine-readable summary to out_dir.

    Creates out_dir if it does not already exist (including parents).

    Writes:
        summary.json  — top-level keys: "conditions", "fractions", "seeds",
                        "metrics" (nested by condition -> str(fraction) ->
                        metric_name -> {"mean": float, "ci_low": float,
                        "ci_high": float})
        summary.csv   — header row + one data row per (condition, fraction) cell
                        with mean and CI columns for each primary metric.

    Parameters
    ----------
    results:
        Dict matching the schema produced by _make_results().
    out_dir:
        Path-like or str.  Created (including parents) if absent.

    Returns
    -------
    (json_path, csv_path) : tuple[str, str]
        Absolute paths to the two written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions: list[str] = results["conditions"]
    fractions: list[float] = results["fractions"]
    seeds: list[int] = results["seeds"]
    raw_metrics: dict = results["metrics"]

    # Build a mapping from float fraction value to the actual string key used in
    # the metrics dict.  The stored keys may be "0.10" while str(0.1) == "0.1",
    # so we resolve by matching float values.
    first_cond_metrics: dict = raw_metrics[conditions[0]]
    frac_key_map: dict[float, str] = {}
    for frac in fractions:
        # Try exact str() first, then scan existing keys by float comparison.
        frac_str = str(frac)
        if frac_str in first_cond_metrics:
            frac_key_map[frac] = frac_str
        else:
            for existing_key in first_cond_metrics:
                try:
                    if float(existing_key) == frac:
                        frac_key_map[frac] = existing_key
                        break
                except ValueError:
                    pass
            else:
                raise KeyError(
                    f"No metrics key found for fraction {frac!r} "
                    f"in {list(first_cond_metrics.keys())}"
                )

    # Build the JSON metrics tree: condition -> str(fraction) -> metric -> {mean, ci_low, ci_high}
    json_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for cond in conditions:
        json_metrics[cond] = {}
        for frac in fractions:
            frac_key = frac_key_map[frac]
            cell = raw_metrics[cond][frac_key]
            json_metrics[cond][frac_key] = {}
            for metric_name, samples in cell.items():
                lo, hi = bootstrap_ci(samples)
                mean_val = float(np.mean(samples))
                json_metrics[cond][frac_key][metric_name] = {
                    "mean": mean_val,
                    "ci_low": lo,
                    "ci_high": hi,
                }

    json_data = {
        "conditions": conditions,
        "fractions": fractions,
        "seeds": seeds,
        "metrics": json_metrics,
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_data, fh, indent=2)

    # Build CSV: header = condition, fraction, then per-metric mean/ci_low/ci_high.
    metric_names = list(raw_metrics[conditions[0]][frac_key_map[fractions[0]]].keys())
    header = ["condition", "fraction"]
    for m in metric_names:
        header.extend([f"{m}_mean", f"{m}_ci_low", f"{m}_ci_high"])

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for cond in conditions:
            for frac in fractions:
                frac_key = frac_key_map[frac]
                row: list[Any] = [cond, frac]
                for m in metric_names:
                    cell_stats = json_metrics[cond][frac_key][m]
                    row.extend(
                        [cell_stats["mean"], cell_stats["ci_low"], cell_stats["ci_high"]]
                    )
                writer.writerow(row)

    return (str(json_path.resolve()), str(csv_path.resolve()))


# ---------------------------------------------------------------------------
# 4. assess_go_no_go
# ---------------------------------------------------------------------------

# PRE-REGISTERED THRESHOLDS — do not alter these values.
_P1_GPU_UTIL_MIN: float = 80.0          # treatment GPU util must be >= this
_P1_GPU_UTIL_GAP_MIN: float = 30.0      # (treatment - baseline) gap must be >= this
_P1_THROUGHPUT_RATIO_MIN: float = 1.5   # (treatment / baseline) rollouts ratio >= this

_P3_SURVIVAL_REQUIRED: int = 3          # all 3 seeds must survive at each fraction
_P3_TTC_MARGIN_SECS: float = 1.0        # mean_time_to_contain <= timeout + this


def assess_go_no_go(results: dict) -> dict:
    """Encode pre-registered thresholds and return the go/no-go assessment.

    PRE-REGISTERED THRESHOLDS (from PRD v1.1 Success Metrics — must not be
    changed by the implementer):

    P1 (GPU util / throughput):
        PASS iff ALL of:
          - results["treatment"]["gpu_util_mean"] >= 80.0
          - (results["treatment"]["gpu_util_mean"]
             - results["baseline"]["gpu_util_mean"]) >= 30.0
          - (results["treatment"]["rollouts_per_sec_mean"]
             / results["baseline"]["rollouts_per_sec_mean"]) >= 1.5

    P3 (adversarial containment — lead claim):
        PASS iff ALL of:
          - results["treatment"]["contagion_events_total"] == 0
          - results["treatment"]["survival_count_at_5pct"] == 3
          - results["treatment"]["survival_count_at_10pct"] == 3
          - results["baseline"]["crashed_at_5pct_or_10pct"] is True
          - results["treatment"]["mean_time_to_contain_secs"]
            <= results["per_sample_timeout_secs"] + 1.0

    guardrail (quality parity):
        PASS iff ci_overlap_check(
            results["baseline"]["pass1_by_seed"],
            results["treatment"]["pass1_by_seed"],
        )["passed"] is True

    overall:
        PASS iff P1 AND P3 AND guardrail all pass.

    Returns
    -------
    dict with keys:
        "p1"       : "pass" | "fail"
        "p3"       : "pass" | "fail"
        "guardrail": "pass" | "fail"
        "overall"  : "pass" | "fail"
        "detail"   : dict with boolean flags:
                     "p1_util_ok", "p1_gap_ok", "p1_ratio_ok",
                     "p3_contagion_ok", "p3_treatment_survival_ok",
                     "p3_baseline_crashed_ok", "p3_time_to_contain_ok",
                     "guardrail_ci_overlap_ok"
    """
    treatment = results["treatment"]
    baseline = results["baseline"]
    timeout = results["per_sample_timeout_secs"]

    # --- P1 ---
    p1_util_ok: bool = treatment["gpu_util_mean"] >= _P1_GPU_UTIL_MIN
    p1_gap_ok: bool = (
        treatment["gpu_util_mean"] - baseline["gpu_util_mean"]
    ) >= _P1_GPU_UTIL_GAP_MIN
    p1_ratio_ok: bool = (
        treatment["rollouts_per_sec_mean"] / baseline["rollouts_per_sec_mean"]
    ) >= _P1_THROUGHPUT_RATIO_MIN
    p1_pass = p1_util_ok and p1_gap_ok and p1_ratio_ok

    # --- P3 ---
    p3_contagion_ok: bool = treatment["contagion_events_total"] == 0
    p3_treatment_survival_ok: bool = (
        treatment["survival_count_at_5pct"] == _P3_SURVIVAL_REQUIRED
        and treatment["survival_count_at_10pct"] == _P3_SURVIVAL_REQUIRED
    )
    p3_baseline_crashed_ok: bool = baseline["crashed_at_5pct_or_10pct"] is True
    p3_time_to_contain_ok: bool = (
        treatment["mean_time_to_contain_secs"] <= timeout + _P3_TTC_MARGIN_SECS
    )
    p3_pass = (
        p3_contagion_ok
        and p3_treatment_survival_ok
        and p3_baseline_crashed_ok
        and p3_time_to_contain_ok
    )

    # --- guardrail ---
    overlap_result = ci_overlap_check(
        baseline["pass1_by_seed"],
        treatment["pass1_by_seed"],
    )
    guardrail_ci_overlap_ok: bool = overlap_result["passed"]
    guardrail_pass = guardrail_ci_overlap_ok

    # --- overall ---
    overall_pass = p1_pass and p3_pass and guardrail_pass

    def _verdict(passed: bool) -> str:
        return "pass" if passed else "fail"

    return {
        "p1": _verdict(p1_pass),
        "p3": _verdict(p3_pass),
        "guardrail": _verdict(guardrail_pass),
        "overall": _verdict(overall_pass),
        "detail": {
            "p1_util_ok": p1_util_ok,
            "p1_gap_ok": p1_gap_ok,
            "p1_ratio_ok": p1_ratio_ok,
            "p3_contagion_ok": p3_contagion_ok,
            "p3_treatment_survival_ok": p3_treatment_survival_ok,
            "p3_baseline_crashed_ok": p3_baseline_crashed_ok,
            "p3_time_to_contain_ok": p3_time_to_contain_ok,
            "guardrail_ci_overlap_ok": guardrail_ci_overlap_ok,
        },
    }
