"""Tests for the reward-capture fixes in trl_grpo_run.py and aggregate_sweep.py.

FIX 1: _MetricsCallback.on_log must correctly capture reward values even when
the reward is 0.0 (the old `or`-chaining treated 0.0 as missing). The summary
dict must include final_reward, mean_reward, and reward_curve.

FIX 2: aggregate_sweep.aggregate() must add mean_final_reward / mean_reward per
row and a guardrail block in p3_verdict comparing Baseline vs Treatment at
fraction 0.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# conftest.py already injects benchmarks/agentic/ and python/rlox/agentic/ onto
# sys.path — no manual insertion needed here.
from trl_grpo_run import _make_metrics_callback

import aggregate_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_fake_callback(output_dir: Path) -> tuple[Any, Any]:
    """Instantiate the metrics callback with a fake TrainerCallback base.

    Avoids requiring transformers to be importable in the test environment.
    We monkey-patch the import inside _make_metrics_callback.
    """

    # Create a minimal TrainerCallback shim
    class _FakeTrainerCallback:
        pass

    with patch.dict(
        "sys.modules", {"transformers": MagicMock(TrainerCallback=_FakeTrainerCallback)}
    ):
        cb, close = _make_metrics_callback(output_dir)
    return cb, close


def _fake_state(step: int) -> SimpleNamespace:
    return SimpleNamespace(global_step=step)


def _fake_args() -> SimpleNamespace:
    return SimpleNamespace()


def _fake_control() -> SimpleNamespace:
    return SimpleNamespace()


# ---------------------------------------------------------------------------
# A) on_log reward key resolution — must not use `or` short-circuit
# ---------------------------------------------------------------------------


class TestOnLogRewardKeyResolution:
    """The on_log handler must correctly extract reward even when value is 0.0."""

    def test_reward_key_zero_not_treated_as_missing(self, tmp_path: Path):
        """logs={'reward': 0.0} must record mean_reward=0.0, not fall back to 0.0 via a different path."""
        cb, close = _build_fake_callback(tmp_path)
        cb.on_log(_fake_args(), _fake_state(1), _fake_control(), logs={"reward": 0.0})
        rows = close()
        assert len(rows) == 1
        assert rows[0]["mean_reward"] == pytest.approx(0.0)

    def test_reward_key_nonzero_captured(self, tmp_path: Path):
        cb, close = _build_fake_callback(tmp_path)
        cb.on_log(_fake_args(), _fake_state(1), _fake_control(), logs={"reward": 0.75})
        rows = close()
        assert rows[0]["mean_reward"] == pytest.approx(0.75)

    def test_rewards_reward_func_mean_key_captured(self, tmp_path: Path):
        """TRL may emit 'rewards/reward_func/mean' as the primary key."""
        cb, close = _build_fake_callback(tmp_path)
        cb.on_log(
            _fake_args(),
            _fake_state(1),
            _fake_control(),
            logs={"rewards/reward_func/mean": 0.6},
        )
        rows = close()
        assert rows[0]["mean_reward"] == pytest.approx(0.6)

    def test_reward_key_takes_precedence_over_fallback(self, tmp_path: Path):
        """When both 'reward' and 'rewards/reward_func/mean' are present, 'reward' wins."""
        cb, close = _build_fake_callback(tmp_path)
        cb.on_log(
            _fake_args(),
            _fake_state(1),
            _fake_control(),
            logs={"reward": 0.9, "rewards/reward_func/mean": 0.5},
        )
        rows = close()
        assert rows[0]["mean_reward"] == pytest.approx(0.9)

    def test_missing_reward_key_yields_zero(self, tmp_path: Path):
        """When no reward key is present, mean_reward must default to 0.0."""
        cb, close = _build_fake_callback(tmp_path)
        cb.on_log(
            _fake_args(),
            _fake_state(1),
            _fake_control(),
            logs={"loss": 1.23},
        )
        rows = close()
        assert rows[0]["mean_reward"] == pytest.approx(0.0)

    def test_multiple_steps_rewards_captured(self, tmp_path: Path):
        """Each on_log call appends a row; all rewards captured correctly."""
        cb, close = _build_fake_callback(tmp_path)
        rewards = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i, r in enumerate(rewards):
            cb.on_log(
                _fake_args(), _fake_state(i + 1), _fake_control(), logs={"reward": r}
            )
        rows = close()
        assert len(rows) == 5
        for row, expected in zip(rows, rewards):
            assert row["mean_reward"] == pytest.approx(expected)

    def test_logs_none_is_skipped(self, tmp_path: Path):
        """on_log with logs=None must not append a row."""
        cb, close = _build_fake_callback(tmp_path)
        cb.on_log(_fake_args(), _fake_state(1), _fake_control(), logs=None)
        rows = close()
        assert len(rows) == 0

    def test_jsonl_file_written(self, tmp_path: Path):
        """Each on_log invocation must write one line to the JSONL file."""
        cb, close = _build_fake_callback(tmp_path)
        cb.on_log(_fake_args(), _fake_state(1), _fake_control(), logs={"reward": 0.4})
        cb.on_log(_fake_args(), _fake_state(2), _fake_control(), logs={"reward": 0.6})
        close()
        jsonl = tmp_path / "metrics.jsonl"
        lines = [line for line in jsonl.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["mean_reward"] == pytest.approx(0.4)
        assert parsed[1]["mean_reward"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# B) summary dict must include final_reward, mean_reward, reward_curve
# ---------------------------------------------------------------------------


def _training_row(step: int, mean_reward: float) -> dict:
    """Build a row that represents a real training-step log event.

    Real rows have ``_has_reward=True`` (set by _MetricsCallback.on_log when a
    recognised reward key is present).  Terminal train_runtime rows have
    ``_has_reward=False`` and should be excluded from reward_curve.
    """
    return {"step": step, "mean_reward": mean_reward, "_has_reward": True}


def _terminal_row(step: int) -> dict:
    """Build a row that represents TRL's end-of-training summary log event."""
    return {
        "step": step,
        "mean_reward": 0.0,
        "_has_reward": False,
        "train_runtime": 20.0,
    }


class TestSummaryRewardFields:
    """train() summary.json must carry final_reward, mean_reward, reward_curve."""

    def _build_summary_from_rows(self, rows: list[dict]) -> dict:
        """Simulate the summary computation logic from trl_grpo_run.train()."""
        from trl_grpo_run import _compute_reward_summary

        return _compute_reward_summary(rows)

    def test_empty_rows_yields_zeros(self):
        summary = self._build_summary_from_rows([])
        assert summary["final_reward"] == pytest.approx(0.0)
        assert summary["mean_reward"] == pytest.approx(0.0)
        assert summary["reward_curve"] == []

    def test_single_row_values(self):
        rows = [_training_row(1, 0.5)]
        summary = self._build_summary_from_rows(rows)
        assert summary["final_reward"] == pytest.approx(0.5)
        assert summary["mean_reward"] == pytest.approx(0.5)
        assert summary["reward_curve"] == [0.5]

    def test_multiple_rows_final_reward_is_last(self):
        rows = [
            _training_row(1, 0.2),
            _training_row(2, 0.4),
            _training_row(3, 0.8),
        ]
        summary = self._build_summary_from_rows(rows)
        assert summary["final_reward"] == pytest.approx(0.8)

    def test_multiple_rows_mean_reward(self):
        rows = [
            _training_row(1, 0.2),
            _training_row(2, 0.4),
            _training_row(3, 0.6),
        ]
        summary = self._build_summary_from_rows(rows)
        assert summary["mean_reward"] == pytest.approx(0.4)

    def test_reward_curve_is_list_of_per_step_rewards(self):
        rows = [
            _training_row(1, 0.1),
            _training_row(2, 0.3),
            _training_row(3, 0.9),
        ]
        summary = self._build_summary_from_rows(rows)
        assert summary["reward_curve"] == pytest.approx([0.1, 0.3, 0.9])

    def test_mean_reward_last_alias_equals_final_reward(self):
        """mean_reward_last is kept as an alias of final_reward."""
        rows = [_training_row(1, 0.7)]
        summary = self._build_summary_from_rows(rows)
        assert summary["mean_reward_last"] == summary["final_reward"]

    def test_all_required_keys_present(self):
        rows = [_training_row(1, 0.5)]
        summary = self._build_summary_from_rows(rows)
        for key in ("final_reward", "mean_reward", "reward_curve", "mean_reward_last"):
            assert key in summary, f"Missing key: {key!r}"

    def test_zero_reward_rows_are_not_suppressed(self):
        """Rows with reward=0.0 must contribute to mean and reward_curve."""
        rows = [
            _training_row(1, 0.0),
            _training_row(2, 0.0),
            _training_row(3, 0.6),
        ]
        summary = self._build_summary_from_rows(rows)
        assert len(summary["reward_curve"]) == 3
        assert summary["mean_reward"] == pytest.approx(0.2)
        assert summary["final_reward"] == pytest.approx(0.6)

    def test_terminal_row_excluded_from_reward_curve(self):
        """TRL's end-of-training summary row must not corrupt final_reward."""
        rows = [
            _training_row(1, 1.0),
            _training_row(2, 0.25),
            _training_row(3, 0.5),
            _terminal_row(3),  # TRL emits this after training completes
        ]
        summary = self._build_summary_from_rows(rows)
        # Terminal row must be excluded — curve should have 3 entries, not 4.
        assert len(summary["reward_curve"]) == 3
        assert summary["final_reward"] == pytest.approx(0.5)
        assert summary["mean_reward"] == pytest.approx((1.0 + 0.25 + 0.5) / 3)


# ---------------------------------------------------------------------------
# C) aggregate_sweep: per-row reward fields
# ---------------------------------------------------------------------------


class TestAggregateSweepRewardFields:
    """aggregate_sweep.aggregate() rows must include mean_final_reward and mean_reward."""

    def _make_runs_dir(self, tmp_path: Path, specs: list[dict]) -> Path:
        """Write summary.json files under tmp_path/runs/<label>/summary.json."""
        runs_dir = tmp_path / "runs"
        for spec in specs:
            label = (
                f"{spec['backend']}_seed{spec['seed']}_"
                f"frac{spec['adversarial_fraction']}"
            )
            run_dir = runs_dir / label
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(json.dumps(spec))
        return runs_dir

    def _default_spec(
        self,
        backend: str = "in_loop",
        seed: int = 0,
        fraction: float = 0.0,
        survived: bool = True,
        final_reward: float = 0.5,
        mean_reward: float = 0.4,
        reward_curve: list[float] | None = None,
    ) -> dict:
        return {
            "backend": backend,
            "seed": seed,
            "adversarial_fraction": fraction,
            "survived": survived,
            "completed_steps": 3,
            "elapsed_secs": 10.0,
            "final_reward": final_reward,
            "mean_reward": mean_reward,
            "mean_reward_last": final_reward,
            "reward_curve": reward_curve or [0.2, 0.4, final_reward],
        }

    def test_row_has_mean_final_reward(self, tmp_path: Path):
        specs = [
            self._default_spec(backend="in_loop", seed=0, final_reward=0.6),
            self._default_spec(backend="in_loop", seed=1, final_reward=0.8),
        ]
        runs_dir = self._make_runs_dir(tmp_path, specs)
        rows, _ = aggregate_sweep.aggregate(runs_dir)
        in_loop_row = next(r for r in rows if r["condition"] == "in_loop")
        assert "mean_final_reward" in in_loop_row
        assert in_loop_row["mean_final_reward"] == pytest.approx(0.7)

    def test_row_has_mean_reward(self, tmp_path: Path):
        specs = [
            self._default_spec(backend="rlox", seed=0, mean_reward=0.3),
            self._default_spec(backend="rlox", seed=1, mean_reward=0.5),
        ]
        runs_dir = self._make_runs_dir(tmp_path, specs)
        rows, _ = aggregate_sweep.aggregate(runs_dir)
        rlox_row = next(r for r in rows if r["condition"] == "rlox")
        assert "mean_reward" in rlox_row
        assert rlox_row["mean_reward"] == pytest.approx(0.4)

    def test_mean_final_reward_averaged_over_seeds(self, tmp_path: Path):
        specs = [
            self._default_spec(backend="rlox", seed=0, final_reward=0.2),
            self._default_spec(backend="rlox", seed=1, final_reward=0.4),
            self._default_spec(backend="rlox", seed=2, final_reward=0.6),
        ]
        runs_dir = self._make_runs_dir(tmp_path, specs)
        rows, _ = aggregate_sweep.aggregate(runs_dir)
        row = next(r for r in rows if r["condition"] == "rlox")
        assert row["mean_final_reward"] == pytest.approx(0.4)

    def test_existing_fields_still_present(self, tmp_path: Path):
        """survived, mean_elapsed_secs, mean_completed_steps still in rows."""
        specs = [self._default_spec(backend="in_loop", seed=0)]
        runs_dir = self._make_runs_dir(tmp_path, specs)
        rows, _ = aggregate_sweep.aggregate(runs_dir)
        row = rows[0]
        for key in ("survived", "mean_elapsed_secs", "mean_completed_steps"):
            assert key in row, f"Missing legacy field: {key!r}"


# ---------------------------------------------------------------------------
# D) aggregate_sweep: guardrail block in p3_verdict
# ---------------------------------------------------------------------------


class TestAggregateSweepGuardrail:
    """p3_verdict must contain a 'guardrail' block for fraction 0.0."""

    def _make_runs_dir(self, tmp_path: Path, specs: list[dict]) -> Path:
        runs_dir = tmp_path / "runs"
        for spec in specs:
            label = (
                f"{spec['backend']}_seed{spec['seed']}_"
                f"frac{spec['adversarial_fraction']}"
            )
            run_dir = runs_dir / label
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(json.dumps(spec))
        return runs_dir

    def _spec(
        self,
        backend: str,
        seed: int,
        fraction: float,
        final_reward: float = 0.7,
        mean_reward: float = 0.6,
    ) -> dict:
        return {
            "backend": backend,
            "seed": seed,
            "adversarial_fraction": fraction,
            "survived": True,
            "completed_steps": 3,
            "elapsed_secs": 10.0,
            "final_reward": final_reward,
            "mean_reward": mean_reward,
            "mean_reward_last": final_reward,
            "reward_curve": [0.3, 0.5, final_reward],
        }

    def _make_full_sweep(self, tmp_path: Path, n_seeds: int = 3) -> Path:
        """Create a sweep with both conditions, multiple seeds, fractions 0.0 and 0.05."""
        specs = []
        for seed in range(n_seeds):
            for frac in (0.0, 0.05):
                specs.append(self._spec("in_loop", seed, frac, final_reward=0.7))
                specs.append(self._spec("rlox", seed, frac, final_reward=0.65))
        return self._make_runs_dir(tmp_path, specs)

    def test_p3_verdict_has_guardrail_key(self, tmp_path: Path):
        runs_dir = self._make_full_sweep(tmp_path)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        assert "guardrail" in verdict, (
            f"p3_verdict missing 'guardrail' key. Keys: {list(verdict.keys())}"
        )

    def test_guardrail_has_passed_key(self, tmp_path: Path):
        runs_dir = self._make_full_sweep(tmp_path)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        guardrail = verdict["guardrail"]
        assert "passed" in guardrail, f"guardrail missing 'passed'. Got: {guardrail}"
        assert isinstance(guardrail["passed"], bool)

    def test_guardrail_has_baseline_mean_final_reward(self, tmp_path: Path):
        runs_dir = self._make_full_sweep(tmp_path)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        guardrail = verdict["guardrail"]
        assert "baseline_mean_final_reward" in guardrail

    def test_guardrail_has_treatment_mean_final_reward(self, tmp_path: Path):
        runs_dir = self._make_full_sweep(tmp_path)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        guardrail = verdict["guardrail"]
        assert "treatment_mean_final_reward" in guardrail

    def test_guardrail_reward_values_are_from_frac0(self, tmp_path: Path):
        """The guardrail values reflect fraction=0.0 only (clean condition)."""
        specs = []
        for seed in range(3):
            # frac 0.0: baseline=0.8, treatment=0.75
            specs.append(self._spec("in_loop", seed, 0.0, final_reward=0.8))
            specs.append(self._spec("rlox", seed, 0.0, final_reward=0.75))
            # frac 0.05: different rewards (should not affect guardrail)
            specs.append(self._spec("in_loop", seed, 0.05, final_reward=0.1))
            specs.append(self._spec("rlox", seed, 0.05, final_reward=0.1))
        runs_dir = self._make_runs_dir(tmp_path, specs)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        guardrail = verdict["guardrail"]
        assert guardrail["baseline_mean_final_reward"] == pytest.approx(0.8)
        assert guardrail["treatment_mean_final_reward"] == pytest.approx(0.75)

    def test_guardrail_passes_when_rewards_close(self, tmp_path: Path):
        """Treatment and baseline within tolerance → guardrail passes."""
        specs = []
        for seed in range(3):
            specs.append(self._spec("in_loop", seed, 0.0, final_reward=0.7))
            specs.append(self._spec("rlox", seed, 0.0, final_reward=0.65))
        runs_dir = self._make_runs_dir(tmp_path, specs)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        assert verdict["guardrail"]["passed"] is True

    def test_guardrail_fails_when_rewards_far_apart(self, tmp_path: Path):
        """Treatment reward much lower than baseline → guardrail fails."""
        specs = []
        for seed in range(3):
            specs.append(self._spec("in_loop", seed, 0.0, final_reward=0.9))
            specs.append(self._spec("rlox", seed, 0.0, final_reward=0.1))
        runs_dir = self._make_runs_dir(tmp_path, specs)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        assert verdict["guardrail"]["passed"] is False

    def test_guardrail_missing_frac0_data_graceful(self, tmp_path: Path):
        """If no frac=0.0 data, guardrail should still be present but marked unavailable."""
        specs = [
            self._spec("in_loop", 0, 0.05, final_reward=0.7),
            self._spec("rlox", 0, 0.05, final_reward=0.65),
        ]
        runs_dir = self._make_runs_dir(tmp_path, specs)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        guardrail = verdict["guardrail"]
        # Must have a 'passed' key; value is False or None when data is absent
        assert "passed" in guardrail

    def test_guardrail_abs_diff_reported(self, tmp_path: Path):
        """The guardrail block reports the absolute difference between conditions."""
        specs = []
        for seed in range(3):
            specs.append(self._spec("in_loop", seed, 0.0, final_reward=0.8))
            specs.append(self._spec("rlox", seed, 0.0, final_reward=0.6))
        runs_dir = self._make_runs_dir(tmp_path, specs)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        guardrail = verdict["guardrail"]
        assert "abs_diff" in guardrail
        assert guardrail["abs_diff"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# E) aggregate_sweep: aggregate() return type contract
# ---------------------------------------------------------------------------


class TestAggregateReturnType:
    def _make_runs_dir(self, tmp_path: Path) -> Path:
        runs_dir = tmp_path / "runs"
        spec = {
            "backend": "in_loop",
            "seed": 0,
            "adversarial_fraction": 0.0,
            "survived": True,
            "completed_steps": 3,
            "elapsed_secs": 5.0,
            "final_reward": 0.5,
            "mean_reward": 0.4,
            "mean_reward_last": 0.5,
            "reward_curve": [0.2, 0.4, 0.5],
        }
        run_dir = runs_dir / "in_loop_seed0_frac0.0"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(spec))
        return runs_dir

    def test_aggregate_returns_tuple(self, tmp_path: Path):
        runs_dir = self._make_runs_dir(tmp_path)
        result = aggregate_sweep.aggregate(runs_dir)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_list_of_dicts(self, tmp_path: Path):
        runs_dir = self._make_runs_dir(tmp_path)
        rows, _ = aggregate_sweep.aggregate(runs_dir)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_second_element_is_dict(self, tmp_path: Path):
        runs_dir = self._make_runs_dir(tmp_path)
        _, verdict = aggregate_sweep.aggregate(runs_dir)
        assert isinstance(verdict, dict)
