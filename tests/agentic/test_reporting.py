"""RED-phase tests for Step 7b: reporting + quality-gate layer (AC-7, AC-9).

Contracts being specified
--------------------------

1. bootstrap_ci(values, *, confidence=0.95, n_resamples=1000, seed=0)
   -> tuple[float, float]

   Percentile bootstrap CI on the mean.  DETERMINISTIC for a fixed seed.
   For typical data: low <= mean(values) <= high.
   For all-equal data: low == high == that value.

2. ci_overlap_check(baseline_pass1_by_seed, treatment_pass1_by_seed, *, seed=0)
   -> dict

   AC-9 guardrail.  Input: per-seed lists of pass@1 samples, one list per seed,
   aligned by index.  For each seed pair computes 95% bootstrap CIs and checks
   overlap.  PASSES if >= 2 of 3 seed pairs overlap.

   Returned dict keys:
       "passed"    bool
       "n_overlap" int
       "n_pairs"   int
       "per_seed"  list[dict]  one per pair, each with
                   {"seed_idx", "baseline_ci", "treatment_ci", "overlap"}

3. write_summary(results: dict, out_dir) -> tuple[str, str]

   Writes summary.json + summary.csv into out_dir (creates dir if absent).
   Returns (json_path, csv_path) as absolute strings.
   CSV has a header row + one row per (condition, fraction) cell.
   JSON has top-level keys: "conditions", "fractions", "seeds", "metrics".

4. assess_go_no_go(results: dict) -> dict

   Pre-registered PRD thresholds.  Returns dict with keys:
       "p1", "p3", "guardrail", "overall"  — each "pass" | "fail"
       "detail"  — dict with per-threshold boolean flags.

Interface assumptions (implementer MUST honour)
------------------------------------------------
- Module path: python/rlox/agentic/reporting.py
- Top-level import: ``import reporting`` (conftest injects python/rlox/agentic/)
- bootstrap_ci uses numpy.random.default_rng(seed) — deterministic, not time-based.
- ci_overlap_check input shape: list of n_seeds sub-lists, one pass@1 sample list
  per seed.  Treatment and baseline lists are aligned by index.
- assess_go_no_go results dict schema: see stub docstring for canonical shape.
- write_summary results dict: same schema as assess_go_no_go (see _make_results()).

All imports are top-level: conftest injects python/rlox/agentic/ onto sys.path.
"""
from __future__ import annotations

import csv
import json
import math
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

# Top-level import — never "from rlox.agentic import ..."
import reporting
from reporting import (
    assess_go_no_go,
    bootstrap_ci,
    ci_overlap_check,
    write_summary,
)


# ---------------------------------------------------------------------------
# Shared helpers / canonical results-dict factory
# ---------------------------------------------------------------------------

def _make_results(
    *,
    treatment_gpu_util: float = 85.0,
    baseline_gpu_util: float = 45.0,
    treatment_rollouts: float = 3.0,
    baseline_rollouts: float = 2.0,
    contagion_total: int = 0,
    treatment_survival_5pct: int = 3,
    treatment_survival_10pct: int = 3,
    baseline_crashed: bool = True,
    mean_time_to_contain: float = 25.0,
    per_sample_timeout: float = 30.0,
    # pass@1 by seed — 3 seeds, 5 eval samples each (default: identical
    # baseline and treatment so CIs always overlap — the "all pass" default)
    baseline_pass1: list[list[float]] | None = None,
    treatment_pass1: list[list[float]] | None = None,
) -> dict:
    """Return a canonical results dict in the shape expected by assess_go_no_go
    and write_summary.

    Default values satisfy ALL pre-registered thresholds (overall: "pass").
    Each parameter lets a single threshold be flipped for negative tests.
    """
    if baseline_pass1 is None:
        baseline_pass1 = [[0.7, 0.72, 0.68, 0.71, 0.69] for _ in range(3)]
    if treatment_pass1 is None:
        # Overlapping but not identical — CIs will overlap
        treatment_pass1 = [[0.71, 0.73, 0.69, 0.72, 0.70] for _ in range(3)]

    return {
        "per_sample_timeout_secs": per_sample_timeout,
        "baseline": {
            "gpu_util_mean": baseline_gpu_util,
            "rollouts_per_sec_mean": baseline_rollouts,
            "crashed_at_5pct_or_10pct": baseline_crashed,
            "pass1_by_seed": baseline_pass1,
        },
        "treatment": {
            "gpu_util_mean": treatment_gpu_util,
            "rollouts_per_sec_mean": treatment_rollouts,
            "contagion_events_total": contagion_total,
            "survival_count_at_5pct": treatment_survival_5pct,
            "survival_count_at_10pct": treatment_survival_10pct,
            "mean_time_to_contain_secs": mean_time_to_contain,
            "pass1_by_seed": treatment_pass1,
        },
        # write_summary also uses these two keys for the CSV/JSON structure
        "conditions": ["in_loop", "rlox"],
        "fractions": [0.0, 0.01, 0.05, 0.10],
        "seeds": [0, 1, 2],
        # Per-(condition, fraction) metric rows for write_summary
        "metrics": {
            "in_loop": {
                "0.0":  {"gpu_util": [45.0, 44.0, 46.0],
                         "rollouts_per_sec": [2.0, 2.1, 1.9],
                         "contagion_events": [0, 0, 0],
                         "pass_at_1": [0.70, 0.72, 0.68]},
                "0.01": {"gpu_util": [44.0, 45.0, 43.0],
                         "rollouts_per_sec": [2.0, 2.0, 2.0],
                         "contagion_events": [0, 0, 0],
                         "pass_at_1": [0.70, 0.71, 0.69]},
                "0.05": {"gpu_util": [40.0, 41.0, 39.0],
                         "rollouts_per_sec": [1.8, 1.9, 1.7],
                         "contagion_events": [1, 0, 2],
                         "pass_at_1": [0.65, 0.66, 0.64]},
                "0.10": {"gpu_util": [35.0, 36.0, 34.0],
                         "rollouts_per_sec": [1.5, 1.6, 1.4],
                         "contagion_events": [2, 1, 3],
                         "pass_at_1": [0.60, 0.61, 0.59]},
            },
            "rlox": {
                "0.0":  {"gpu_util": [85.0, 86.0, 84.0],
                         "rollouts_per_sec": [3.0, 3.1, 2.9],
                         "contagion_events": [0, 0, 0],
                         "pass_at_1": [0.71, 0.73, 0.69]},
                "0.01": {"gpu_util": [85.0, 85.0, 85.0],
                         "rollouts_per_sec": [3.0, 3.0, 3.0],
                         "contagion_events": [0, 0, 0],
                         "pass_at_1": [0.71, 0.72, 0.70]},
                "0.05": {"gpu_util": [84.0, 85.0, 83.0],
                         "rollouts_per_sec": [2.9, 3.0, 2.8],
                         "contagion_events": [0, 0, 0],
                         "pass_at_1": [0.71, 0.72, 0.70]},
                "0.10": {"gpu_util": [84.0, 84.0, 84.0],
                         "rollouts_per_sec": [2.9, 2.9, 2.9],
                         "contagion_events": [0, 0, 0],
                         "pass_at_1": [0.71, 0.72, 0.70]},
            },
        },
    }


# ---------------------------------------------------------------------------
# 1. bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCi:
    """Contract: percentile bootstrap mean CI, deterministic for a fixed seed."""

    # --- A) Return shape and basic validity ---

    def test_returns_two_element_tuple(self):
        low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(low, float) and isinstance(high, float), (
            "bootstrap_ci must return a tuple of two floats"
        )

    def test_low_le_mean_le_high(self):
        """For typical data: low <= mean(values) <= high."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = float(np.mean(values))
        low, high = bootstrap_ci(values)
        assert low <= mean <= high, (
            f"Expected low ({low}) <= mean ({mean}) <= high ({high})"
        )

    def test_low_le_high(self):
        """low must not exceed high under any circumstances."""
        low, high = bootstrap_ci([10.0, 20.0, 15.0, 25.0])
        assert low <= high, f"CI lower bound {low} exceeds upper bound {high}"

    def test_degenerate_all_equal_gives_point_ci(self):
        """All-equal values must give low == high == that value."""
        low, high = bootstrap_ci([7.0, 7.0, 7.0, 7.0, 7.0])
        assert math.isclose(low, 7.0, rel_tol=1e-9), (
            f"Degenerate case: expected low=7.0, got {low}"
        )
        assert math.isclose(high, 7.0, rel_tol=1e-9), (
            f"Degenerate case: expected high=7.0, got {high}"
        )

    # --- B) Determinism ---

    def test_same_seed_same_result(self):
        """Two calls with the same seed and values must return identical CIs."""
        values = [0.5, 0.6, 0.4, 0.7, 0.55, 0.65, 0.45]
        result_a = bootstrap_ci(values, seed=42)
        result_b = bootstrap_ci(values, seed=42)
        assert result_a == result_b, (
            f"bootstrap_ci must be deterministic for seed=42. "
            f"Got {result_a} vs {result_b}"
        )

    def test_different_seeds_may_differ(self):
        """Different seeds should (very likely) produce different CIs.

        Uses enough data points that two independent resamples give distinct
        results with overwhelming probability (probability of collision <1e-6).
        """
        values = list(range(50))  # 0..49
        low_0, high_0 = bootstrap_ci(values, seed=0)
        low_1, high_1 = bootstrap_ci(values, seed=999)
        # They might theoretically match, but for these inputs it is
        # astronomically unlikely — assert at least one bound differs.
        # (If this spuriously fails, seed choices can be adjusted.)
        different = (low_0 != low_1) or (high_0 != high_1)
        assert different, (
            "Different seeds should produce different CI estimates for diverse data"
        )

    def test_deterministic_not_time_based(self):
        """Calling twice in rapid succession with seed=0 gives same result."""
        values = [0.3, 0.5, 0.4, 0.6, 0.35]
        r1 = bootstrap_ci(values, seed=0)
        r2 = bootstrap_ci(values, seed=0)
        assert r1 == r2, (
            "bootstrap_ci must not use wall-clock time or random.random() — "
            f"two calls with seed=0 returned {r1} and {r2}"
        )

    # --- C) Width ordering (tight cluster vs spread) ---

    def test_tight_cluster_gives_narrow_ci(self):
        """Values tightly clustered around a mean give a narrow CI width."""
        tight = [5.0 + 0.01 * i for i in range(20)]
        low_t, high_t = bootstrap_ci(tight)
        width_tight = high_t - low_t
        assert width_tight < 0.5, (
            f"Tight cluster should give narrow CI; width={width_tight}"
        )

    def test_spread_data_gives_wider_ci_than_tight(self):
        """Spread data should produce a wider CI than the tight cluster."""
        rng = np.random.default_rng(7)
        tight = list(rng.normal(loc=5.0, scale=0.05, size=50))
        spread = list(rng.normal(loc=5.0, scale=5.0, size=50))

        _, _ = bootstrap_ci(tight, seed=0)
        low_t, high_t = bootstrap_ci(tight, seed=0)
        low_s, high_s = bootstrap_ci(spread, seed=0)

        width_tight = high_t - low_t
        width_spread = high_s - low_s

        assert width_spread > width_tight, (
            f"Spread data (width={width_spread}) must give wider CI "
            f"than tight data (width={width_tight})"
        )

    # --- D) Confidence level has effect ---

    def test_higher_confidence_gives_wider_ci(self):
        """99% CI must be wider than (or equal to) 90% CI."""
        values = list(range(1, 31))  # 1..30
        low_90, high_90 = bootstrap_ci(values, confidence=0.90, seed=0)
        low_99, high_99 = bootstrap_ci(values, confidence=0.99, seed=0)
        width_90 = high_90 - low_90
        width_99 = high_99 - low_99
        assert width_99 >= width_90, (
            f"99% CI (width={width_99}) must be >= 90% CI (width={width_90})"
        )

    # --- E) Array-like input accepted ---

    def test_accepts_numpy_array(self):
        """bootstrap_ci must accept a numpy array, not just a Python list."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        low, high = bootstrap_ci(arr)
        assert low <= high

    @pytest.mark.parametrize("values,expected_mean", [
        ([0.0], 0.0),
        ([1.0], 1.0),
    ])
    def test_single_element_input(self, values, expected_mean):
        """Single-element input: low == high == that value."""
        low, high = bootstrap_ci(values)
        assert math.isclose(low, expected_mean, rel_tol=1e-9), (
            f"Single-element: expected low={expected_mean}, got {low}"
        )
        assert math.isclose(high, expected_mean, rel_tol=1e-9), (
            f"Single-element: expected high={expected_mean}, got {high}"
        )


# ---------------------------------------------------------------------------
# 2. ci_overlap_check
# ---------------------------------------------------------------------------

class TestCiOverlapCheck:
    """Contract: AC-9 guardrail — pass iff >= 2/3 seed-pair CIs overlap."""

    # --- A) All 3 pairs overlapping => passed=True ---

    def test_all_3_overlapping_passes(self):
        """3 seed pairs all with overlapping CIs => passed=True, n_overlap=3."""
        # Same tight values for both conditions — CIs must overlap
        baseline = [[0.70, 0.71, 0.69] for _ in range(3)]
        treatment = [[0.70, 0.71, 0.69] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        assert result["passed"] is True, (
            f"3/3 overlapping CIs must give passed=True, got: {result}"
        )
        assert result["n_overlap"] == 3, (
            f"Expected n_overlap=3, got {result['n_overlap']}"
        )
        assert result["n_pairs"] == 3, (
            f"Expected n_pairs=3, got {result['n_pairs']}"
        )

    # --- B) 2 of 3 overlapping => passed=True ---

    def test_2_of_3_overlapping_passes(self):
        """Exactly 2/3 overlapping => passed=True (boundary case)."""
        # Seeds 0 and 1: identical values => CIs overlap
        # Seed 2: treatment is far from baseline => CIs do NOT overlap
        baseline = [
            [0.50, 0.50, 0.50, 0.50, 0.50],  # seed 0 — tight around 0.50
            [0.60, 0.60, 0.60, 0.60, 0.60],  # seed 1 — tight around 0.60
            [0.30, 0.30, 0.30, 0.30, 0.30],  # seed 2 — tight around 0.30
        ]
        treatment = [
            [0.50, 0.50, 0.50, 0.50, 0.50],  # seed 0 — overlaps
            [0.60, 0.60, 0.60, 0.60, 0.60],  # seed 1 — overlaps
            [0.99, 0.99, 0.99, 0.99, 0.99],  # seed 2 — far away, no overlap
        ]
        result = ci_overlap_check(baseline, treatment, seed=0)
        assert result["passed"] is True, (
            f"2/3 overlapping CIs must give passed=True, got: {result}"
        )
        assert result["n_overlap"] == 2, (
            f"Expected n_overlap=2, got {result['n_overlap']}"
        )

    # --- C) 1 of 3 overlapping => passed=False ---

    def test_1_of_3_overlapping_fails(self):
        """Only 1/3 overlapping => passed=False."""
        baseline = [
            [0.50, 0.50, 0.50, 0.50, 0.50],  # seed 0 — overlaps treatment
            [0.30, 0.30, 0.30, 0.30, 0.30],  # seed 1 — no overlap
            [0.30, 0.30, 0.30, 0.30, 0.30],  # seed 2 — no overlap
        ]
        treatment = [
            [0.50, 0.50, 0.50, 0.50, 0.50],  # seed 0 — overlaps
            [0.99, 0.99, 0.99, 0.99, 0.99],  # seed 1 — far, no overlap
            [0.99, 0.99, 0.99, 0.99, 0.99],  # seed 2 — far, no overlap
        ]
        result = ci_overlap_check(baseline, treatment, seed=0)
        assert result["passed"] is False, (
            f"1/3 overlapping CIs must give passed=False, got: {result}"
        )
        assert result["n_overlap"] == 1, (
            f"Expected n_overlap=1, got {result['n_overlap']}"
        )

    # --- D) 0 of 3 overlapping => passed=False ---

    def test_0_of_3_overlapping_fails(self):
        """No overlapping pairs => passed=False, n_overlap=0."""
        baseline = [[0.30, 0.30, 0.30, 0.30] for _ in range(3)]
        treatment = [[0.99, 0.99, 0.99, 0.99] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment, seed=0)
        assert result["passed"] is False, (
            f"0/3 overlapping must give passed=False, got: {result}"
        )
        assert result["n_overlap"] == 0, (
            f"Expected n_overlap=0, got {result['n_overlap']}"
        )

    # --- E) Return dict structure ---

    def test_return_has_all_required_keys(self):
        """Returned dict must have: passed, n_overlap, n_pairs, per_seed."""
        baseline = [[0.5, 0.5] for _ in range(3)]
        treatment = [[0.5, 0.5] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        for key in ("passed", "n_overlap", "n_pairs", "per_seed"):
            assert key in result, (
                f"ci_overlap_check result missing required key '{key}'"
            )

    def test_per_seed_has_correct_length(self):
        """per_seed list must have one entry per seed pair."""
        baseline = [[0.5, 0.5] for _ in range(3)]
        treatment = [[0.5, 0.5] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        assert len(result["per_seed"]) == 3, (
            f"per_seed must have 3 entries for 3 seed pairs, "
            f"got {len(result['per_seed'])}"
        )

    def test_per_seed_entries_have_required_keys(self):
        """Each per_seed entry must have: seed_idx, baseline_ci, treatment_ci, overlap."""
        baseline = [[0.5, 0.5] for _ in range(3)]
        treatment = [[0.5, 0.5] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        for i, entry in enumerate(result["per_seed"]):
            for key in ("seed_idx", "baseline_ci", "treatment_ci", "overlap"):
                assert key in entry, (
                    f"per_seed[{i}] missing key '{key}': {entry}"
                )

    def test_per_seed_seed_idx_values(self):
        """per_seed[i]['seed_idx'] must equal i."""
        baseline = [[0.5, 0.5] for _ in range(3)]
        treatment = [[0.5, 0.5] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        for i, entry in enumerate(result["per_seed"]):
            assert entry["seed_idx"] == i, (
                f"per_seed[{i}]['seed_idx'] must be {i}, got {entry['seed_idx']}"
            )

    def test_ci_fields_are_two_tuples(self):
        """baseline_ci and treatment_ci must each be (float, float) tuples."""
        baseline = [[0.5, 0.5, 0.5] for _ in range(3)]
        treatment = [[0.5, 0.5, 0.5] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        for i, entry in enumerate(result["per_seed"]):
            b_ci = entry["baseline_ci"]
            t_ci = entry["treatment_ci"]
            assert (
                len(b_ci) == 2 and isinstance(b_ci[0], float)
                and isinstance(b_ci[1], float)
            ), f"per_seed[{i}]['baseline_ci'] must be (float, float), got {b_ci}"
            assert (
                len(t_ci) == 2 and isinstance(t_ci[0], float)
                and isinstance(t_ci[1], float)
            ), f"per_seed[{i}]['treatment_ci'] must be (float, float), got {t_ci}"

    def test_n_pairs_matches_input_length(self):
        """n_pairs must equal the length of the per-seed input lists."""
        baseline = [[0.5, 0.5] for _ in range(3)]
        treatment = [[0.5, 0.5] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        assert result["n_pairs"] == 3, (
            f"n_pairs must equal len(baseline_pass1_by_seed)=3, got {result['n_pairs']}"
        )

    def test_passed_is_bool(self):
        """'passed' must be a Python bool."""
        baseline = [[0.5, 0.5] for _ in range(3)]
        treatment = [[0.5, 0.5] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        assert isinstance(result["passed"], bool), (
            f"'passed' must be a bool, got {type(result['passed'])}"
        )

    def test_n_overlap_consistent_with_per_seed(self):
        """n_overlap must equal the count of per_seed entries where overlap==True."""
        baseline = [[0.5, 0.5] for _ in range(3)]
        treatment = [[0.5, 0.5] for _ in range(3)]
        result = ci_overlap_check(baseline, treatment)
        computed_n_overlap = sum(1 for e in result["per_seed"] if e["overlap"])
        assert result["n_overlap"] == computed_n_overlap, (
            f"n_overlap={result['n_overlap']} must equal sum of per_seed overlap "
            f"bools={computed_n_overlap}"
        )

    # --- F) Boundary at exactly >= 2 ---

    def test_exactly_2_overlap_is_boundary_pass(self):
        """Exactly 2/3 is the lower passing boundary — must return passed=True."""
        # Construct inputs where exactly 2 pairs trivially overlap (identical)
        # and 1 pair definitively does not.
        same = [0.80, 0.80, 0.80, 0.80, 0.80]
        far = [0.01, 0.01, 0.01, 0.01, 0.01]
        baseline = [same[:], same[:], same[:]]
        treatment = [same[:], same[:], far[:]]  # third pair: baseline ~0.80, treatment ~0.01

        result = ci_overlap_check(baseline, treatment, seed=0)
        assert result["passed"] is True, (
            f"2/3 overlap is the passing boundary; must be True. Got: {result}"
        )


# ---------------------------------------------------------------------------
# 3. write_summary
# ---------------------------------------------------------------------------

class TestWriteSummary:
    """Contract: writes summary.json + summary.csv; creates out_dir if absent."""

    # --- A) Files are created ---

    def test_both_files_created(self, tmp_path: Path):
        """write_summary must create both summary.json and summary.csv."""
        results = _make_results()
        json_path, csv_path = write_summary(results, tmp_path)
        assert Path(json_path).exists(), f"summary.json not found at {json_path}"
        assert Path(csv_path).exists(), f"summary.csv not found at {csv_path}"

    def test_returns_two_strings(self, tmp_path: Path):
        """Return value must be a tuple of two strings."""
        results = _make_results()
        ret = write_summary(results, tmp_path)
        assert isinstance(ret, tuple) and len(ret) == 2, (
            f"write_summary must return a 2-tuple, got {type(ret)!r}"
        )
        json_path, csv_path = ret
        assert isinstance(json_path, str), (
            f"json_path must be a str, got {type(json_path)}"
        )
        assert isinstance(csv_path, str), (
            f"csv_path must be a str, got {type(csv_path)}"
        )

    def test_out_dir_created_if_absent(self, tmp_path: Path):
        """write_summary must create out_dir (including parents) if it does not exist."""
        new_dir = tmp_path / "deep" / "nested" / "output"
        assert not new_dir.exists(), "Precondition: directory must not exist"
        results = _make_results()
        write_summary(results, new_dir)
        assert new_dir.exists(), (
            "write_summary must create out_dir if it does not exist"
        )

    # --- B) JSON is parseable and has correct top-level keys ---

    def test_json_is_parseable(self, tmp_path: Path):
        """summary.json must be valid JSON."""
        results = _make_results()
        json_path, _ = write_summary(results, tmp_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), (
            f"summary.json must deserialize to a dict, got {type(data)}"
        )

    def test_json_has_required_top_level_keys(self, tmp_path: Path):
        """summary.json must have: conditions, fractions, seeds, metrics."""
        results = _make_results()
        json_path, _ = write_summary(results, tmp_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for key in ("conditions", "fractions", "seeds", "metrics"):
            assert key in data, (
                f"summary.json missing required top-level key '{key}'. "
                f"Keys found: {list(data.keys())}"
            )

    def test_json_conditions_match_input(self, tmp_path: Path):
        """summary.json 'conditions' must match results['conditions']."""
        results = _make_results()
        json_path, _ = write_summary(results, tmp_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["conditions"] == results["conditions"], (
            f"conditions mismatch: {data['conditions']} vs {results['conditions']}"
        )

    def test_json_fractions_match_input(self, tmp_path: Path):
        """summary.json 'fractions' must match results['fractions']."""
        results = _make_results()
        json_path, _ = write_summary(results, tmp_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["fractions"] == results["fractions"], (
            f"fractions mismatch: {data['fractions']} vs {results['fractions']}"
        )

    def test_json_metrics_has_mean_and_ci(self, tmp_path: Path):
        """Each metric cell in summary.json must have 'mean', 'ci_low', 'ci_high'."""
        results = _make_results()
        json_path, _ = write_summary(results, tmp_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        # Check at least one condition / fraction / metric cell
        for cond, fracs in metrics.items():
            for frac_key, metric_cells in fracs.items():
                for metric_name, cell in metric_cells.items():
                    for sub in ("mean", "ci_low", "ci_high"):
                        assert sub in cell, (
                            f"metrics[{cond!r}][{frac_key!r}][{metric_name!r}] "
                            f"missing '{sub}': {cell}"
                        )

    # --- C) CSV is parseable and has correct structure ---

    def test_csv_is_parseable(self, tmp_path: Path):
        """summary.csv must be a valid CSV file (parseable by csv.reader)."""
        results = _make_results()
        _, csv_path = write_summary(results, tmp_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 1, "summary.csv must have at least a header row"

    def test_csv_has_header_row(self, tmp_path: Path):
        """The first row of summary.csv must be a non-empty header."""
        results = _make_results()
        _, csv_path = write_summary(results, tmp_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        assert len(header) >= 1, (
            f"CSV header row is empty: {header!r}"
        )
        # The header must contain at least "condition" and "fraction"
        header_lower = [h.lower() for h in header]
        assert any("condition" in h for h in header_lower), (
            f"CSV header must include a 'condition' column. Header: {header}"
        )
        assert any("fraction" in h for h in header_lower), (
            f"CSV header must include a 'fraction' column. Header: {header}"
        )

    def test_csv_data_row_count(self, tmp_path: Path):
        """CSV must have one data row per (condition, fraction) combination."""
        results = _make_results()
        n_conditions = len(results["conditions"])  # 2
        n_fractions = len(results["fractions"])   # 4
        expected_data_rows = n_conditions * n_fractions  # 8

        _, csv_path = write_summary(results, tmp_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        data_rows = rows[1:]  # skip header
        assert len(data_rows) == expected_data_rows, (
            f"Expected {expected_data_rows} data rows "
            f"({n_conditions} conditions × {n_fractions} fractions), "
            f"got {len(data_rows)}"
        )

    def test_csv_mean_and_ci_columns_present(self, tmp_path: Path):
        """CSV header must include mean and CI columns for the primary metrics."""
        results = _make_results()
        _, csv_path = write_summary(results, tmp_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

        headers_lower = [h.lower() for h in headers]
        # At least one of the primary metrics must appear with a mean column
        has_mean = any("mean" in h for h in headers_lower)
        assert has_mean, (
            f"CSV header must include at least one '_mean' column. "
            f"Headers: {headers}"
        )

    # --- D) Paths are absolute strings pointing inside out_dir ---

    def test_returned_paths_are_absolute(self, tmp_path: Path):
        """Returned (json_path, csv_path) must be absolute path strings."""
        results = _make_results()
        json_path, csv_path = write_summary(results, tmp_path)
        assert os.path.isabs(json_path), (
            f"json_path must be absolute, got {json_path!r}"
        )
        assert os.path.isabs(csv_path), (
            f"csv_path must be absolute, got {csv_path!r}"
        )

    def test_returned_paths_end_with_correct_filenames(self, tmp_path: Path):
        """Returned paths must end with 'summary.json' and 'summary.csv'."""
        results = _make_results()
        json_path, csv_path = write_summary(results, tmp_path)
        assert json_path.endswith("summary.json"), (
            f"First return value must end with 'summary.json', got {json_path!r}"
        )
        assert csv_path.endswith("summary.csv"), (
            f"Second return value must end with 'summary.csv', got {csv_path!r}"
        )


# ---------------------------------------------------------------------------
# 4. assess_go_no_go
# ---------------------------------------------------------------------------

class TestAssessGoNoGo:
    """Contract: pre-registered thresholds from PRD v1.1 Success Metrics."""

    # --- A) Happy path — all thresholds satisfied ---

    def test_all_pass_when_all_thresholds_met(self):
        """Default results dict meeting all thresholds -> all 'pass', overall 'pass'."""
        results = _make_results()
        out = assess_go_no_go(results)
        assert out["p1"] == "pass", (
            f"P1 must pass with default (all-satisfying) results. Got: {out['p1']}"
        )
        assert out["p3"] == "pass", (
            f"P3 must pass with default results. Got: {out['p3']}"
        )
        assert out["guardrail"] == "pass", (
            f"guardrail must pass with default results. Got: {out['guardrail']}"
        )
        assert out["overall"] == "pass", (
            f"overall must pass when all axes pass. Got: {out['overall']}"
        )

    # --- B) Return structure ---

    def test_return_has_all_required_keys(self):
        """Return dict must have: p1, p3, guardrail, overall, detail."""
        results = _make_results()
        out = assess_go_no_go(results)
        for key in ("p1", "p3", "guardrail", "overall", "detail"):
            assert key in out, (
                f"assess_go_no_go result missing required key '{key}'"
            )

    def test_verdict_values_are_pass_or_fail(self):
        """p1, p3, guardrail, overall must each be exactly 'pass' or 'fail'."""
        results = _make_results()
        out = assess_go_no_go(results)
        for key in ("p1", "p3", "guardrail", "overall"):
            assert out[key] in ("pass", "fail"), (
                f"'{key}' must be 'pass' or 'fail', got {out[key]!r}"
            )

    def test_detail_has_required_per_threshold_keys(self):
        """detail dict must include the documented per-threshold boolean flags."""
        results = _make_results()
        out = assess_go_no_go(results)
        detail = out["detail"]
        required_detail_keys = (
            "p1_util_ok",
            "p1_gap_ok",
            "p1_ratio_ok",
            "p3_contagion_ok",
            "p3_treatment_survival_ok",
            "p3_baseline_crashed_ok",
            "p3_time_to_contain_ok",
            "guardrail_ci_overlap_ok",
        )
        for k in required_detail_keys:
            assert k in detail, (
                f"detail missing required key '{k}'. Keys: {list(detail.keys())}"
            )

    def test_overall_pass_iff_all_three_pass(self):
        """overall='pass' iff p1='pass' AND p3='pass' AND guardrail='pass'."""
        results = _make_results()
        out = assess_go_no_go(results)
        all_pass = (out["p1"] == "pass" and out["p3"] == "pass"
                    and out["guardrail"] == "pass")
        if all_pass:
            assert out["overall"] == "pass"
        else:
            assert out["overall"] == "fail"

    # --- C) P1 individual threshold flips ---

    def test_p1_fails_when_treatment_gpu_util_below_80(self):
        """Treatment GPU util 79% < 80% threshold => P1 fail, overall fail."""
        results = _make_results(treatment_gpu_util=79.0)
        out = assess_go_no_go(results)
        assert out["p1"] == "fail", (
            f"P1 must fail when treatment GPU util=79% < 80%. Got: {out['p1']}"
        )
        assert out["overall"] == "fail", (
            f"overall must fail when P1 fails. Got: {out['overall']}"
        )

    def test_p1_fails_when_gpu_util_gap_below_30pp(self):
        """Treatment=75%, Baseline=50% => gap=25pp < 30pp threshold => P1 fail."""
        results = _make_results(treatment_gpu_util=75.0, baseline_gpu_util=50.0)
        out = assess_go_no_go(results)
        assert out["p1"] == "fail", (
            f"P1 must fail when gap=25pp < 30pp. Got: {out['p1']}"
        )
        assert out["overall"] == "fail"

    def test_p1_fails_when_throughput_ratio_below_1_5(self):
        """Treatment/Baseline throughput ratio = 1.4 < 1.5 => P1 fail."""
        # 2.8 / 2.0 = 1.4 < 1.5
        results = _make_results(treatment_rollouts=2.8, baseline_rollouts=2.0)
        out = assess_go_no_go(results)
        assert out["p1"] == "fail", (
            f"P1 must fail when throughput ratio=1.4 < 1.5. Got: {out['p1']}"
        )
        assert out["overall"] == "fail"

    def test_p1_passes_at_exact_80_util(self):
        """Treatment GPU util exactly 80% is at the threshold — must pass."""
        # Also ensure gap >= 30pp: baseline=50%, gap=30pp, ratio>=1.5
        results = _make_results(
            treatment_gpu_util=80.0,
            baseline_gpu_util=50.0,
            treatment_rollouts=3.0,
            baseline_rollouts=2.0,
        )
        out = assess_go_no_go(results)
        assert out["p1"] == "pass", (
            f"P1 must pass at exactly 80% util. Got: {out['p1']}"
        )

    def test_p1_passes_at_exact_ratio_1_5(self):
        """Throughput ratio exactly 1.5 is at the threshold — must pass."""
        results = _make_results(
            treatment_gpu_util=85.0,
            baseline_gpu_util=50.0,
            treatment_rollouts=3.0,
            baseline_rollouts=2.0,  # ratio = 3.0 / 2.0 = 1.5
        )
        out = assess_go_no_go(results)
        assert out["p1"] == "pass", (
            f"P1 must pass at exact ratio 1.5. Got: {out['p1']}"
        )

    # --- D) P3 individual threshold flips ---

    def test_p3_fails_when_contagion_events_nonzero(self):
        """Any contagion event in Treatment => P3 fail, overall fail."""
        results = _make_results(contagion_total=1)
        out = assess_go_no_go(results)
        assert out["p3"] == "fail", (
            f"P3 must fail when contagion_events_total=1. Got: {out['p3']}"
        )
        assert out["overall"] == "fail"

    def test_p3_fails_when_treatment_survival_at_5pct_not_3(self):
        """Treatment survival at 5% < 3/3 => P3 fail."""
        results = _make_results(treatment_survival_5pct=2)
        out = assess_go_no_go(results)
        assert out["p3"] == "fail", (
            f"P3 must fail when treatment survival at 5%=2/3. Got: {out['p3']}"
        )
        assert out["overall"] == "fail"

    def test_p3_fails_when_treatment_survival_at_10pct_not_3(self):
        """Treatment survival at 10% < 3/3 => P3 fail."""
        results = _make_results(treatment_survival_10pct=2)
        out = assess_go_no_go(results)
        assert out["p3"] == "fail", (
            f"P3 must fail when treatment survival at 10%=2/3. Got: {out['p3']}"
        )
        assert out["overall"] == "fail"

    def test_p3_fails_when_baseline_did_not_crash(self):
        """Baseline with zero crashes/stalls at 5% or 10% => P3 fail."""
        results = _make_results(baseline_crashed=False)
        out = assess_go_no_go(results)
        assert out["p3"] == "fail", (
            f"P3 must fail when baseline did not crash. Got: {out['p3']}"
        )
        assert out["overall"] == "fail"

    def test_p3_fails_when_mean_time_to_contain_too_slow(self):
        """Mean time-to-contain > per_sample_timeout + 1s => P3 fail."""
        # timeout=30s, time-to-contain=32s > 31s cap
        results = _make_results(
            per_sample_timeout=30.0,
            mean_time_to_contain=32.0,
        )
        out = assess_go_no_go(results)
        assert out["p3"] == "fail", (
            f"P3 must fail when mean_time_to_contain=32.0 > timeout+1=31.0. "
            f"Got: {out['p3']}"
        )
        assert out["overall"] == "fail"

    def test_p3_passes_at_exact_time_to_contain_boundary(self):
        """Mean time-to-contain exactly at per_sample_timeout + 1s => P3 passes
        (the threshold is <=, not <)."""
        results = _make_results(
            per_sample_timeout=30.0,
            mean_time_to_contain=31.0,
        )
        out = assess_go_no_go(results)
        # P3 time-to-contain sub-criterion must pass at the boundary
        assert out["detail"]["p3_time_to_contain_ok"] is True, (
            f"p3_time_to_contain_ok must be True at mean_ttc=31.0 == timeout+1. "
            f"Got: {out['detail']['p3_time_to_contain_ok']}"
        )

    # --- E) Guardrail individual flip ---

    def test_guardrail_fails_when_cis_do_not_overlap(self):
        """Non-overlapping pass@1 CIs => guardrail fail, overall fail."""
        # All 3 baseline seeds around 0.30, all 3 treatment seeds around 0.99
        # => 0/3 pairs overlap => ci_overlap_check fails => guardrail fails
        baseline_p1 = [[0.30, 0.30, 0.30, 0.30, 0.30] for _ in range(3)]
        treatment_p1 = [[0.99, 0.99, 0.99, 0.99, 0.99] for _ in range(3)]
        results = _make_results(
            baseline_pass1=baseline_p1,
            treatment_pass1=treatment_p1,
        )
        out = assess_go_no_go(results)
        assert out["guardrail"] == "fail", (
            f"guardrail must fail when CIs do not overlap. Got: {out['guardrail']}"
        )
        assert out["overall"] == "fail"

    # --- F) Overall logic: each single-axis failure kills overall ---

    @pytest.mark.parametrize("kwargs,description", [
        ({"treatment_gpu_util": 79.0}, "P1: util below threshold"),
        ({"treatment_rollouts": 2.8, "baseline_rollouts": 2.0},
         "P1: throughput ratio below 1.5"),
        ({"contagion_total": 1}, "P3: contagion event"),
        ({"treatment_survival_5pct": 2}, "P3: treatment did not survive 5%"),
        ({"baseline_crashed": False}, "P3: baseline did not crash"),
        ({"mean_time_to_contain": 32.0, "per_sample_timeout": 30.0},
         "P3: time-to-contain exceeds cap"),
    ])
    def test_single_threshold_failure_causes_overall_fail(
        self, kwargs, description
    ):
        """Flipping one threshold to 'fail' must cause overall='fail'."""
        results = _make_results(**kwargs)
        out = assess_go_no_go(results)
        assert out["overall"] == "fail", (
            f"overall must fail when {description}. Got overall={out['overall']!r}, "
            f"p1={out['p1']!r}, p3={out['p3']!r}, guardrail={out['guardrail']!r}"
        )

    # --- G) Detail flags match verdicts ---

    def test_p1_detail_flags_all_true_when_p1_passes(self):
        """When P1 passes, all p1_* detail flags must be True."""
        results = _make_results()
        out = assess_go_no_go(results)
        if out["p1"] == "pass":
            for flag in ("p1_util_ok", "p1_gap_ok", "p1_ratio_ok"):
                assert out["detail"][flag] is True, (
                    f"detail[{flag!r}] must be True when P1 passes"
                )

    def test_p3_detail_flags_all_true_when_p3_passes(self):
        """When P3 passes, all p3_* detail flags must be True."""
        results = _make_results()
        out = assess_go_no_go(results)
        if out["p3"] == "pass":
            for flag in (
                "p3_contagion_ok",
                "p3_treatment_survival_ok",
                "p3_baseline_crashed_ok",
                "p3_time_to_contain_ok",
            ):
                assert out["detail"][flag] is True, (
                    f"detail[{flag!r}] must be True when P3 passes"
                )

    def test_p1_util_detail_flag_false_when_util_low(self):
        """detail['p1_util_ok'] must be False when treatment util < 80%."""
        results = _make_results(treatment_gpu_util=79.0)
        out = assess_go_no_go(results)
        assert out["detail"]["p1_util_ok"] is False, (
            f"p1_util_ok must be False for treatment_gpu_util=79.0. "
            f"Got: {out['detail']['p1_util_ok']}"
        )

    def test_p3_contagion_flag_false_when_events_nonzero(self):
        """detail['p3_contagion_ok'] must be False when contagion_events_total > 0."""
        results = _make_results(contagion_total=2)
        out = assess_go_no_go(results)
        assert out["detail"]["p3_contagion_ok"] is False, (
            f"p3_contagion_ok must be False when contagion_total=2. "
            f"Got: {out['detail']['p3_contagion_ok']}"
        )

    def test_guardrail_ci_overlap_flag_matches_guardrail_verdict(self):
        """detail['guardrail_ci_overlap_ok'] must be True iff guardrail='pass'."""
        results = _make_results()
        out = assess_go_no_go(results)
        if out["guardrail"] == "pass":
            assert out["detail"]["guardrail_ci_overlap_ok"] is True
        else:
            assert out["detail"]["guardrail_ci_overlap_ok"] is False
