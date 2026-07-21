"""Coverage / characterization tests for aggregate_sweep.aggregate — DNF-aware
survival and mean_gpu_util.

These tests lock the "Change 3" contract: when metric_store_dir is provided,
survival is counted from the authoritative metric_store/*.json records (which
cover DNF/timeout runs that never wrote a summary.json), NOT from summary.json
files alone.

The critical scenario tested here:
  in_loop, frac=0.10: 3 seeds in metric_store — only 1 wrote a summary.json
    (the other 2 timed out → DNF, survived=false in the store).
  rlox, frac=0.10: 3 seeds, all 3 wrote summary.json, all survived.

Expected: aggregate row for in_loop@0.10 must report survived=1 (from the
metric_store, counting all 3 seeds), NOT survived=1 with n=1 (which is what
a summary-only count would produce).

Secondary contract locked:
  - mean_gpu_util is computed per aggregate row from summary.json data.
  - rows that have no summary.json (pure-DNF condition/fraction) carry None
    for mean_elapsed_secs, mean_final_reward, and mean_gpu_util.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# benchmarks/agentic/ is on sys.path via conftest.py
from aggregate_sweep import aggregate


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_summary(
    run_dir: Path,
    backend: str,
    adversarial_fraction: float,
    seed: int,
    survived: bool = True,
    elapsed_secs: float = 100.0,
    final_reward: float = 0.5,
    mean_reward: float = 0.4,
    mean_gpu_util: float | None = None,
    completed_steps: int = 10,
) -> None:
    """Write a minimal summary.json for one run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "backend": backend,
        "adversarial_fraction": adversarial_fraction,
        "seed": seed,
        "survived": survived,
        "elapsed_secs": elapsed_secs,
        "final_reward": final_reward,
        "mean_reward": mean_reward,
        "completed_steps": completed_steps,
        "reward_curve": [mean_reward],
        "mean_reward_last": final_reward,
    }
    if mean_gpu_util is not None:
        data["mean_gpu_util"] = mean_gpu_util
    (run_dir / "summary.json").write_text(json.dumps(data))


def _make_metric_store_record(
    store_dir: Path,
    condition: str,
    seed: int,
    fraction: float,
    survived: bool,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    name = f"{condition}_seed{seed}_frac{fraction:.2f}.json"
    (store_dir / name).write_text(
        json.dumps(
            {
                "condition": condition,
                "seed": seed,
                "fraction": fraction,
                "survived": survived,
            }
        )
    )


@pytest.fixture()
def dnf_sweep(tmp_path: Path) -> tuple[Path, Path]:
    """Fixture with the DNF scenario.

    in_loop frac=0.10: 3 seeds in metric_store, only seed 0 wrote summary.json.
      seed 0 — survived=True  (summary.json + metric_store)
      seed 1 — survived=False (metric_store only, DNF — no summary.json)
      seed 2 — survived=False (metric_store only, DNF — no summary.json)

    rlox frac=0.10: 3 seeds, all 3 wrote summary.json, all survived.
      seed 0 — survived=True (summary.json + metric_store)
      seed 1 — survived=True (summary.json + metric_store)
      seed 2 — survived=True (summary.json + metric_store)
    """
    runs_dir = tmp_path / "runs"
    store_dir = tmp_path / "metric_store"

    # --- in_loop frac=0.10 ---
    # seed 0: survived, has summary
    _make_summary(
        runs_dir / "in_loop_seed0_frac0.10",
        backend="in_loop",
        adversarial_fraction=0.10,
        seed=0,
        survived=True,
        elapsed_secs=120.0,
        final_reward=0.3,
        mean_reward=0.25,
    )
    _make_metric_store_record(store_dir, "in_loop", 0, 0.10, survived=True)
    # seed 1: DNF — no summary.json, metric_store says survived=False
    _make_metric_store_record(store_dir, "in_loop", 1, 0.10, survived=False)
    # seed 2: DNF — no summary.json, metric_store says survived=False
    _make_metric_store_record(store_dir, "in_loop", 2, 0.10, survived=False)

    # --- rlox frac=0.10 ---
    for seed in range(3):
        _make_summary(
            runs_dir / f"rlox_seed{seed}_frac0.10",
            backend="rlox",
            adversarial_fraction=0.10,
            seed=seed,
            survived=True,
            elapsed_secs=90.0,
            final_reward=0.6,
            mean_reward=0.55,
        )
        _make_metric_store_record(store_dir, "rlox", seed, 0.10, survived=True)

    return runs_dir, store_dir


@pytest.fixture()
def dnf_sweep_with_gpu(tmp_path: Path) -> tuple[Path, Path]:
    """Fixture identical to dnf_sweep but summaries include mean_gpu_util."""
    runs_dir = tmp_path / "runs"
    store_dir = tmp_path / "metric_store"

    # in_loop frac=0.10 — seed 0 only (DNF for seeds 1, 2)
    _make_summary(
        runs_dir / "in_loop_seed0_frac0.10",
        backend="in_loop",
        adversarial_fraction=0.10,
        seed=0,
        survived=True,
        elapsed_secs=120.0,
        final_reward=0.3,
        mean_reward=0.25,
        mean_gpu_util=40.0,
    )
    _make_metric_store_record(store_dir, "in_loop", 0, 0.10, survived=True)
    _make_metric_store_record(store_dir, "in_loop", 1, 0.10, survived=False)
    _make_metric_store_record(store_dir, "in_loop", 2, 0.10, survived=False)

    # rlox frac=0.10 — all 3 seeds
    gpu_values = [80.0, 85.0, 90.0]
    for seed in range(3):
        _make_summary(
            runs_dir / f"rlox_seed{seed}_frac0.10",
            backend="rlox",
            adversarial_fraction=0.10,
            seed=seed,
            survived=True,
            elapsed_secs=90.0,
            final_reward=0.6,
            mean_reward=0.55,
            mean_gpu_util=gpu_values[seed],
        )
        _make_metric_store_record(store_dir, "rlox", seed, 0.10, survived=True)

    return runs_dir, store_dir


# ---------------------------------------------------------------------------
# A) aggregate return shape
# ---------------------------------------------------------------------------


class TestAggregateReturnShape:
    def test_returns_two_tuple(self, dnf_sweep):
        runs_dir, store_dir = dnf_sweep
        result = aggregate(runs_dir, metric_store_dir=store_dir)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_rows_is_list(self, dnf_sweep):
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        assert isinstance(rows, list)

    def test_verdict_is_dict(self, dnf_sweep):
        runs_dir, store_dir = dnf_sweep
        _, verdict = aggregate(runs_dir, metric_store_dir=store_dir)
        assert isinstance(verdict, dict)

    def test_rows_contain_two_entries(self, dnf_sweep):
        """One row per (condition, fraction) group → 2 rows for our fixture."""
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        assert len(rows) == 2

    def test_row_has_required_keys(self, dnf_sweep):
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        required = {"condition", "fraction", "n", "survived"}
        for row in rows:
            assert required.issubset(row.keys()), (
                f"Row is missing required keys. Has: {set(row.keys())}"
            )


# ---------------------------------------------------------------------------
# B) DNF-aware survival: the key contract
# ---------------------------------------------------------------------------


class TestDNFAwareSurvival:
    def test_in_loop_frac010_survived_is_1(self, dnf_sweep):
        """With 3 seeds in metric_store and only 1 surviving, survived must be 1."""
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        in_loop_row = next(
            r for r in rows
            if r["condition"] == "in_loop" and abs(r["fraction"] - 0.10) < 1e-9
        )
        assert in_loop_row["survived"] == 1, (
            f"Expected survived=1 (DNF-aware count), got {in_loop_row['survived']}"
        )

    def test_in_loop_frac010_n_is_3(self, dnf_sweep):
        """n must reflect ALL seeds in the metric_store (not just completed runs)."""
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        in_loop_row = next(
            r for r in rows
            if r["condition"] == "in_loop" and abs(r["fraction"] - 0.10) < 1e-9
        )
        assert in_loop_row["n"] == 3, (
            f"Expected n=3 (from metric_store covering all 3 seeds), got {in_loop_row['n']}"
        )

    def test_in_loop_survived_fraction_is_one_of_three(self, dnf_sweep):
        """The effective survival rate (survived/n) is 1/3 ≈ 0.333, not 1/1."""
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        in_loop_row = next(
            r for r in rows
            if r["condition"] == "in_loop" and abs(r["fraction"] - 0.10) < 1e-9
        )
        rate = in_loop_row["survived"] / in_loop_row["n"]
        assert abs(rate - 1 / 3) < 1e-9, (
            f"Survival rate must be 1/3, got {in_loop_row['survived']}/{in_loop_row['n']}"
        )

    def test_rlox_frac010_survived_is_3(self, dnf_sweep):
        """rlox has 3/3 seeds surviving → survived must equal 3."""
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        rlox_row = next(
            r for r in rows
            if r["condition"] == "rlox" and abs(r["fraction"] - 0.10) < 1e-9
        )
        assert rlox_row["survived"] == 3

    def test_rlox_frac010_n_is_3(self, dnf_sweep):
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        rlox_row = next(
            r for r in rows
            if r["condition"] == "rlox" and abs(r["fraction"] - 0.10) < 1e-9
        )
        assert rlox_row["n"] == 3

    def test_without_metric_store_in_loop_n_is_1(self, dnf_sweep):
        """Without metric_store, only the 1 completed summary.json is counted —
        this is the PRE-Change-3 behavior that the DNF fix corrects."""
        runs_dir, _store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=None)
        in_loop_row = next(
            r for r in rows
            if r["condition"] == "in_loop" and abs(r["fraction"] - 0.10) < 1e-9
        )
        # Without metric_store only 1 summary.json exists → n=1
        assert in_loop_row["n"] == 1, (
            f"Without metric_store, n must equal number of summary.json files (1), "
            f"got {in_loop_row['n']}"
        )

    def test_without_metric_store_survived_appears_100_percent(self, dnf_sweep):
        """Without metric_store, the single surviving seed gives survived=1/n=1 = 100%.
        This documents the pre-fix over-counting bug that metric_store corrects."""
        runs_dir, _store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=None)
        in_loop_row = next(
            r for r in rows
            if r["condition"] == "in_loop" and abs(r["fraction"] - 0.10) < 1e-9
        )
        # 1 survived / 1 n = 100% — the buggy figure before the DNF fix
        assert in_loop_row["survived"] == in_loop_row["n"]


# ---------------------------------------------------------------------------
# C) Reward / elapsed stats are computed from completed summary.json runs only
# ---------------------------------------------------------------------------


class TestRewardAndElapsedStats:
    def test_in_loop_mean_elapsed_from_one_completed_run(self, dnf_sweep):
        """mean_elapsed_secs is averaged over the 1 completed summary.json."""
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        in_loop_row = next(
            r for r in rows
            if r["condition"] == "in_loop" and abs(r["fraction"] - 0.10) < 1e-9
        )
        assert in_loop_row["mean_elapsed_secs"] == pytest.approx(120.0)

    def test_rlox_mean_elapsed_from_three_completed_runs(self, dnf_sweep):
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        rlox_row = next(
            r for r in rows
            if r["condition"] == "rlox" and abs(r["fraction"] - 0.10) < 1e-9
        )
        assert rlox_row["mean_elapsed_secs"] == pytest.approx(90.0)

    def test_in_loop_mean_final_reward(self, dnf_sweep):
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        in_loop_row = next(
            r for r in rows
            if r["condition"] == "in_loop" and abs(r["fraction"] - 0.10) < 1e-9
        )
        assert in_loop_row["mean_final_reward"] == pytest.approx(0.3, abs=1e-3)

    def test_rlox_mean_final_reward(self, dnf_sweep):
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        rlox_row = next(
            r for r in rows
            if r["condition"] == "rlox" and abs(r["fraction"] - 0.10) < 1e-9
        )
        assert rlox_row["mean_final_reward"] == pytest.approx(0.6, abs=1e-3)


# ---------------------------------------------------------------------------
# D) mean_gpu_util per aggregate row
# ---------------------------------------------------------------------------


class TestMeanGpuUtil:
    def test_in_loop_mean_gpu_util_populated(self, dnf_sweep_with_gpu):
        runs_dir, store_dir = dnf_sweep_with_gpu
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        in_loop_row = next(
            r for r in rows
            if r["condition"] == "in_loop" and abs(r["fraction"] - 0.10) < 1e-9
        )
        # Only 1 completed run with mean_gpu_util=40.0
        assert in_loop_row["mean_gpu_util"] == pytest.approx(40.0)

    def test_rlox_mean_gpu_util_averaged_over_seeds(self, dnf_sweep_with_gpu):
        runs_dir, store_dir = dnf_sweep_with_gpu
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        rlox_row = next(
            r for r in rows
            if r["condition"] == "rlox" and abs(r["fraction"] - 0.10) < 1e-9
        )
        # Three seeds: 80 + 85 + 90 = 255 / 3 = 85.0
        assert rlox_row["mean_gpu_util"] == pytest.approx(85.0, abs=0.01)

    def test_mean_gpu_util_none_when_no_data(self, dnf_sweep):
        """When no summary.json carries mean_gpu_util, the row must have None."""
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        for row in rows:
            assert row.get("mean_gpu_util") is None, (
                f"Expected mean_gpu_util=None when summaries lack the field, "
                f"got {row.get('mean_gpu_util')} for {row['condition']}"
            )

    def test_mean_gpu_util_key_present_in_row(self, dnf_sweep):
        """mean_gpu_util key must always be present in every row (even if None)."""
        runs_dir, store_dir = dnf_sweep
        rows, _ = aggregate(runs_dir, metric_store_dir=store_dir)
        for row in rows:
            assert "mean_gpu_util" in row, (
                f"Row for {row['condition']} is missing 'mean_gpu_util' key"
            )


# ---------------------------------------------------------------------------
# E) Empty runs_dir returns sentinel (no crash)
# ---------------------------------------------------------------------------


class TestEmptyRunsDir:
    def test_empty_runs_returns_empty_rows(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        rows, verdict = aggregate(runs_dir)
        assert rows == []

    def test_empty_runs_verdict_has_false_survival(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _, verdict = aggregate(runs_dir)
        assert verdict.get("treatment_survives_all_fractions") is False

    def test_empty_runs_verdict_has_guardrail(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _, verdict = aggregate(runs_dir)
        assert "guardrail" in verdict
