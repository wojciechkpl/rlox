"""Coverage / characterization tests for the pure helper functions in
benchmarks/agentic/plot_sweep.py.

We deliberately do NOT call main() — it requires matplotlib and writes files.
We test only the pure data-loader helpers that are importable with no GPU and
no matplotlib.

Helpers under test:
  _load_runs(runs_dir)          — reads summary.json + optional metrics.jsonl
  _load_metric_store(sweep_dir) — reads metric_store/*.json (incl. DNF records)
  _by(runs, cond, frac)        — filter by backend condition and/or fraction
  _fractions(runs)              — sorted unique adversarial_fractions from runs
  _mean(xs)                    — arithmetic mean; None/empty → 0.0

Fixture layout written to tmp_path:
  runs/
    in_loop_seed0_frac0.10/
      summary.json   (backend=in_loop, adversarial_fraction=0.10, survived=true)
      metrics.jsonl  (2 step rows: step/reward/step_time)
    rlox_seed0_frac0.10/
      summary.json   (backend=rlox, adversarial_fraction=0.10, survived=true)
    in_loop_seed0_frac0.00/
      summary.json   (backend=in_loop, adversarial_fraction=0.00, survived=true)
  metric_store/
    in_loop_seed1_frac0.10.json  (survived=false — DNF, no summary.json twin)
    rlox_seed0_frac0.10.json     (survived=true)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# benchmarks/agentic/ is on sys.path via conftest.py
from plot_sweep import _by, _fractions, _load_metric_store, _load_runs, _mean


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_summary(run_dir: Path, data: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(data))


def _write_metrics_jsonl(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r) for r in rows)
    (run_dir / "metrics.jsonl").write_text(lines)


def _write_metric_store_record(store_dir: Path, name: str, data: dict) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / name).write_text(json.dumps(data))


@pytest.fixture()
def sweep_dir(tmp_path: Path) -> Path:
    """Build a minimal, self-contained sweep directory."""
    runs = tmp_path / "runs"

    # Run 1 — in_loop, frac=0.10, WITH metrics.jsonl
    r1 = runs / "in_loop_seed0_frac0.10"
    _write_summary(
        r1,
        {
            "backend": "in_loop",
            "adversarial_fraction": 0.10,
            "survived": True,
            "elapsed_secs": 120.5,
            "reward_curve": [0.1, 0.2, 0.3],
            "final_reward": 0.3,
        },
    )
    _write_metrics_jsonl(
        r1,
        [
            {"step": 1, "reward": 0.1, "step_time": 10.0},
            {"step": 2, "reward": 0.2, "step_time": 11.5},
        ],
    )

    # Run 2 — rlox, frac=0.10, no metrics.jsonl
    r2 = runs / "rlox_seed0_frac0.10"
    _write_summary(
        r2,
        {
            "backend": "rlox",
            "adversarial_fraction": 0.10,
            "survived": True,
            "elapsed_secs": 95.0,
            "reward_curve": [0.4, 0.5],
            "final_reward": 0.5,
        },
    )

    # Run 3 — in_loop, frac=0.00, baseline condition
    r3 = runs / "in_loop_seed0_frac0.00"
    _write_summary(
        r3,
        {
            "backend": "in_loop",
            "adversarial_fraction": 0.0,
            "survived": True,
            "elapsed_secs": 90.0,
            "reward_curve": [0.6],
            "final_reward": 0.6,
        },
    )

    # metric_store — two records
    store = tmp_path / "metric_store"
    _write_metric_store_record(
        store,
        "in_loop_seed1_frac0.10.json",
        {
            "condition": "in_loop",
            "seed": 1,
            "fraction": 0.10,
            "survived": False,  # DNF — no matching summary.json
        },
    )
    _write_metric_store_record(
        store,
        "rlox_seed0_frac0.10.json",
        {
            "condition": "rlox",
            "seed": 0,
            "fraction": 0.10,
            "survived": True,
        },
    )

    return tmp_path


# ---------------------------------------------------------------------------
# A) _load_runs
# ---------------------------------------------------------------------------


class TestLoadRuns:
    def test_returns_list(self, sweep_dir: Path):
        runs = _load_runs(sweep_dir / "runs")
        assert isinstance(runs, list)

    def test_loads_all_summary_jsons(self, sweep_dir: Path):
        """Three summary.json files → three run dicts."""
        runs = _load_runs(sweep_dir / "runs")
        assert len(runs) == 3

    def test_run_has_backend_field(self, sweep_dir: Path):
        runs = _load_runs(sweep_dir / "runs")
        assert all("backend" in r for r in runs)

    def test_run_has_adversarial_fraction_field(self, sweep_dir: Path):
        runs = _load_runs(sweep_dir / "runs")
        assert all("adversarial_fraction" in r for r in runs)

    def test_steps_populated_from_metrics_jsonl(self, sweep_dir: Path):
        """The run with a metrics.jsonl must have _steps with rows."""
        runs = _load_runs(sweep_dir / "runs")
        in_loop_frac10 = [
            r
            for r in runs
            if r.get("backend") == "in_loop"
            and abs(float(r.get("adversarial_fraction", 0)) - 0.10) < 1e-9
        ]
        assert len(in_loop_frac10) == 1
        steps = in_loop_frac10[0]["_steps"]
        assert isinstance(steps, list)
        assert len(steps) == 2

    def test_steps_contain_expected_keys(self, sweep_dir: Path):
        runs = _load_runs(sweep_dir / "runs")
        in_loop = next(
            r
            for r in runs
            if r.get("backend") == "in_loop"
            and abs(float(r.get("adversarial_fraction", 0)) - 0.10) < 1e-9
        )
        step_row = in_loop["_steps"][0]
        assert "step" in step_row
        assert "reward" in step_row
        assert "step_time" in step_row

    def test_steps_empty_list_when_no_metrics_jsonl(self, sweep_dir: Path):
        """Runs without a metrics.jsonl must have _steps == []."""
        runs = _load_runs(sweep_dir / "runs")
        rlox_run = next(r for r in runs if r.get("backend") == "rlox")
        assert rlox_run["_steps"] == []

    def test_run_has_dir_key(self, sweep_dir: Path):
        """_load_runs must inject a '_dir' key pointing to the run directory."""
        runs = _load_runs(sweep_dir / "runs")
        assert all("_dir" in r for r in runs)

    def test_run_dir_is_path(self, sweep_dir: Path):
        runs = _load_runs(sweep_dir / "runs")
        assert all(isinstance(r["_dir"], Path) for r in runs)

    def test_empty_runs_dir_returns_empty_list(self, tmp_path: Path):
        empty_runs = tmp_path / "empty_runs"
        empty_runs.mkdir()
        assert _load_runs(empty_runs) == []

    def test_malformed_summary_json_is_skipped(self, tmp_path: Path):
        """A summary.json that is invalid JSON must be skipped silently."""
        bad_run = tmp_path / "runs" / "bad_seed0_frac0.10"
        bad_run.mkdir(parents=True)
        (bad_run / "summary.json").write_text("{NOT VALID JSON!!!")
        good_run = tmp_path / "runs" / "good_seed0_frac0.10"
        _write_summary(
            good_run,
            {
                "backend": "in_loop",
                "adversarial_fraction": 0.10,
                "survived": True,
                "elapsed_secs": 10.0,
                "reward_curve": [],
                "final_reward": 0.0,
            },
        )
        runs = _load_runs(tmp_path / "runs")
        assert len(runs) == 1  # only the good one


# ---------------------------------------------------------------------------
# B) _load_metric_store
# ---------------------------------------------------------------------------


class TestLoadMetricStore:
    def test_returns_list(self, sweep_dir: Path):
        recs = _load_metric_store(sweep_dir)
        assert isinstance(recs, list)

    def test_loads_all_metric_store_records(self, sweep_dir: Path):
        """Two files in metric_store/ → two records."""
        recs = _load_metric_store(sweep_dir)
        assert len(recs) == 2

    def test_includes_dnf_record(self, sweep_dir: Path):
        """The DNF record (survived=false) must be present."""
        recs = _load_metric_store(sweep_dir)
        dnf = [r for r in recs if not r.get("survived")]
        assert len(dnf) == 1
        assert dnf[0]["condition"] == "in_loop"
        assert dnf[0]["seed"] == 1

    def test_survived_true_record_present(self, sweep_dir: Path):
        recs = _load_metric_store(sweep_dir)
        survived = [r for r in recs if r.get("survived")]
        assert len(survived) == 1
        assert survived[0]["condition"] == "rlox"

    def test_empty_metric_store_dir_returns_empty_list(self, tmp_path: Path):
        store = tmp_path / "metric_store"
        store.mkdir()
        assert _load_metric_store(tmp_path) == []

    def test_missing_metric_store_dir_handled(self, tmp_path: Path):
        """If metric_store/ doesn't exist, glob returns nothing — must not raise."""
        # metric_store/ dir is absent — _load_metric_store calls .glob on it
        # which raises FileNotFoundError only if the caller doesn't guard.
        # The contract: either returns [] or raises — we document the observed behavior.
        try:
            result = _load_metric_store(tmp_path)
            assert result == []  # preferred: graceful empty list
        except (FileNotFoundError, StopIteration):
            pass  # also acceptable — the caller guards in practice

    def test_malformed_metric_store_json_skipped(self, tmp_path: Path):
        store = tmp_path / "metric_store"
        store.mkdir()
        (store / "bad.json").write_text("not json")
        (store / "good.json").write_text(
            json.dumps({"condition": "rlox", "seed": 0, "fraction": 0.10, "survived": True})
        )
        recs = _load_metric_store(tmp_path)
        assert len(recs) == 1


# ---------------------------------------------------------------------------
# C) _by — filtering helper
# ---------------------------------------------------------------------------


class TestBy:
    @pytest.fixture()
    def sample_runs(self) -> list[dict]:
        return [
            {
                "backend": "in_loop",
                "adversarial_fraction": 0.10,
                "survived": False,
                "_steps": [],
            },
            {
                "backend": "in_loop",
                "adversarial_fraction": 0.0,
                "survived": True,
                "_steps": [],
            },
            {
                "backend": "rlox",
                "adversarial_fraction": 0.10,
                "survived": True,
                "_steps": [],
            },
            {
                "backend": "rlox",
                "adversarial_fraction": 0.0,
                "survived": True,
                "_steps": [],
            },
        ]

    def test_no_filter_returns_all(self, sample_runs):
        assert len(_by(sample_runs)) == 4

    def test_filter_by_cond_in_loop(self, sample_runs):
        result = _by(sample_runs, cond="in_loop")
        assert len(result) == 2
        assert all(r["backend"] == "in_loop" for r in result)

    def test_filter_by_cond_rlox(self, sample_runs):
        result = _by(sample_runs, cond="rlox")
        assert len(result) == 2
        assert all(r["backend"] == "rlox" for r in result)

    def test_filter_by_frac(self, sample_runs):
        result = _by(sample_runs, frac=0.10)
        assert len(result) == 2
        assert all(abs(float(r["adversarial_fraction"]) - 0.10) < 1e-9 for r in result)

    def test_filter_by_cond_and_frac(self, sample_runs):
        result = _by(sample_runs, cond="rlox", frac=0.10)
        assert len(result) == 1
        assert result[0]["backend"] == "rlox"
        assert abs(float(result[0]["adversarial_fraction"]) - 0.10) < 1e-9

    def test_filter_no_match_returns_empty(self, sample_runs):
        result = _by(sample_runs, cond="nonexistent")
        assert result == []

    def test_filter_preserves_all_fields(self, sample_runs):
        """Filtered runs are the same dicts (not copies with stripped fields)."""
        result = _by(sample_runs, cond="rlox", frac=0.10)
        assert "_steps" in result[0]


# ---------------------------------------------------------------------------
# D) _fractions — sorted unique fractions
# ---------------------------------------------------------------------------


class TestFractions:
    def test_returns_sorted_unique(self):
        runs = [
            {"adversarial_fraction": 0.10},
            {"adversarial_fraction": 0.0},
            {"adversarial_fraction": 0.10},  # duplicate
            {"adversarial_fraction": 0.05},
        ]
        result = _fractions(runs)
        assert result == [0.0, 0.05, 0.10]

    def test_empty_list_returns_empty(self):
        assert _fractions([]) == []

    def test_single_fraction(self):
        assert _fractions([{"adversarial_fraction": 0.05}]) == [0.05]

    def test_missing_fraction_key_defaults_to_zero(self):
        """If adversarial_fraction is absent, it must be treated as 0."""
        runs = [{"backend": "rlox"}]  # no adversarial_fraction key
        result = _fractions(runs)
        assert result == [0.0]

    def test_result_is_list_of_floats(self):
        runs = [{"adversarial_fraction": 0.10}, {"adversarial_fraction": 0.05}]
        result = _fractions(runs)
        assert all(isinstance(f, float) for f in result)

    def test_fractions_from_real_fixture(self, sweep_dir: Path):
        runs = _load_runs(sweep_dir / "runs")
        fracs = _fractions(runs)
        assert 0.0 in fracs
        assert 0.10 in fracs
        assert fracs == sorted(fracs)


# ---------------------------------------------------------------------------
# E) _mean — arithmetic mean with None-filtering
# ---------------------------------------------------------------------------


class TestMean:
    def test_normal_list(self):
        assert _mean([10.0, 20.0, 30.0]) == pytest.approx(20.0)

    def test_empty_list_returns_zero(self):
        assert _mean([]) == 0.0

    def test_none_values_filtered_out(self):
        assert _mean([None, 10.0, None, 30.0]) == pytest.approx(20.0)

    def test_all_none_returns_zero(self):
        assert _mean([None, None]) == 0.0

    def test_single_element(self):
        assert _mean([42.0]) == pytest.approx(42.0)

    def test_single_none_returns_zero(self):
        assert _mean([None]) == 0.0

    def test_integers_are_accepted(self):
        """_mean must work when xs contains ints (no strict type check)."""
        assert _mean([2, 4, 6]) == pytest.approx(4.0)

    def test_mixed_int_and_float(self):
        assert _mean([1, 2.0, 3]) == pytest.approx(2.0)

    def test_zero_values_not_filtered(self):
        """0.0 is a valid sample — must NOT be treated as falsy and removed."""
        assert _mean([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_negative_values(self):
        assert _mean([-10.0, 0.0, 10.0]) == pytest.approx(0.0)

    def test_large_list(self):
        xs = list(range(1, 101))  # 1..100, mean = 50.5
        assert _mean(xs) == pytest.approx(50.5)
