"""RED-phase tests for Component 6: MetricCollector (Step 6a).

Contract being specified
------------------------
- MetricCollector(jsonl_path, gpu_sampler=<callable>) records one JSONL line
  per step with the mandatory schema.
- Warm-up steps are written to JSONL with ``warmup=true`` but excluded from
  ``summary()`` aggregation.
- The injected ``gpu_sampler`` is the one called — no real nvidia-smi is ever
  invoked in these tests.
- ``_parse_nvidia_smi_line`` correctly parses a bare-integer nvidia-smi output
  line when tested via a fake subprocess runner.
- ``summary()`` returns mean GPU-util and mean throughput over non-warmup steps.

All imports are top-level: conftest.py injects python/rlox/agentic/ onto
sys.path, so ``import metric_collector`` works without touching rlox.__init__.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

# Top-level import — never "from rlox.agentic import ..."
import rlox_agent.metric_collector as mc
from rlox_agent.metric_collector import MetricCollector, _parse_nvidia_smi_line


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_backend_stats(
    rollouts_per_sec: float = 2.0,
    tool_calls_per_sec: float = 3.0,
) -> dict:
    """Return a minimal BackendStats-shaped dict for use in tests."""
    return {
        "batch_wall_secs": 1.0,
        "rollouts_completed": 4,
        "rollouts_per_sec": rollouts_per_sec,
        "tool_calls_per_sec": tool_calls_per_sec,
        "adversarial_injected": 0,
        "adversarial_contained": 0,
        "contagion_events": 0,
        "setup_error_events": 0,
        "time_to_contain_secs": [],
        "cgroup_freeze_events": 0,
        "cgroup_kill_events": 0,
        "oom_kill_events": 0,
        "gpu_idle_attributable_to_hang_secs": 0.0,
        "step_index": 0,
    }


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    """Return a path inside a temp dir for JSONL output."""
    return tmp_path / "metrics.jsonl"


# ---------------------------------------------------------------------------
# A) Constructor / injectable sampler
# ---------------------------------------------------------------------------

class TestMetricCollectorConstructor:
    def test_can_construct_with_injected_sampler(self, tmp_jsonl: Path):
        """MetricCollector must accept a gpu_sampler kwarg."""
        fake_sampler = lambda: 42.0
        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=fake_sampler)
        # Just constructing must not raise (real behavior tested below).
        assert collector is not None

    def test_can_construct_with_default_sampler(self, tmp_jsonl: Path):
        """MetricCollector must be constructible without supplying gpu_sampler."""
        collector = MetricCollector(str(tmp_jsonl))
        assert collector is not None

    def test_injected_sampler_is_used_not_nvidia_smi(
        self, tmp_jsonl: Path, monkeypatch
    ):
        """The injected sampler must be called — not any real subprocess.

        We monkeypatch subprocess.run/Popen to raise if called, proving that
        nvidia-smi is never shelled out to when a fake sampler is provided.
        """
        import subprocess

        def _explode(*args, **kwargs):
            raise AssertionError(
                "subprocess was called — the injected gpu_sampler must be used, "
                "not a real nvidia-smi process"
            )

        monkeypatch.setattr(subprocess, "run", _explode)
        monkeypatch.setattr(subprocess, "Popen", _explode)

        call_count = 0

        def fake_sampler() -> float:
            nonlocal call_count
            call_count += 1
            return 55.0

        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=fake_sampler)
        collector.start()
        collector.on_step(0, _make_backend_stats(), warmup=False, wall_secs=1.0)
        collector.stop()

        assert call_count >= 1, (
            "The injected gpu_sampler was never called — MetricCollector must "
            "use the provided sampler, not fall back to nvidia-smi."
        )


# ---------------------------------------------------------------------------
# B) JSONL schema — 10-step mock run
# ---------------------------------------------------------------------------

class TestJsonlSchema:
    """A 10-step mock run writes 10 JSONL lines with the correct schema."""

    _GPU_UTILS = [float(i * 10) for i in range(10)]  # 0, 10, 20, ..., 90

    def _run_10_steps(
        self,
        jsonl_path: Path,
        warmup_count: int = 2,
    ) -> list[dict]:
        """Run a 10-step mock collection and return the parsed JSONL lines."""
        utils = iter(self._GPU_UTILS)

        def fake_sampler() -> float:
            return next(utils)

        collector = MetricCollector(str(jsonl_path), gpu_sampler=fake_sampler)
        collector.start()

        for step in range(10):
            is_warmup = step < warmup_count
            collector.on_step(
                step_idx=step,
                backend_stats=_make_backend_stats(
                    rollouts_per_sec=float(step + 1),
                    tool_calls_per_sec=float(step + 2),
                ),
                warmup=is_warmup,
                wall_secs=float(step) * 0.5 + 0.1,
            )

        collector.stop()

        lines = jsonl_path.read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def test_exactly_10_lines_written(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        assert len(rows) == 10, (
            f"Expected exactly 10 JSONL lines for a 10-step run, got {len(rows)}"
        )

    def test_each_line_has_step_idx(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            assert "step_idx" in row, f"Line {i} missing 'step_idx' field"

    def test_step_idx_values_are_sequential(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        step_indices = [row["step_idx"] for row in rows]
        assert step_indices == list(range(10)), (
            f"step_idx values must be 0..9, got {step_indices}"
        )

    def test_each_line_has_gpu_util(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            assert "gpu_util" in row, f"Line {i} missing 'gpu_util' field"
            assert isinstance(row["gpu_util"], (int, float)), (
                f"Line {i}: 'gpu_util' must be numeric, got {type(row['gpu_util'])}"
            )

    def test_gpu_util_values_match_sampler_sequence(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            expected = float(i * 10)
            actual = float(row["gpu_util"])
            assert abs(actual - expected) < 1e-6, (
                f"Step {i}: expected gpu_util={expected}, got {actual}"
            )

    def test_each_line_has_rollouts_per_sec(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            assert "rollouts_per_sec" in row, (
                f"Line {i} missing 'rollouts_per_sec' field"
            )

    def test_rollouts_per_sec_values_match_backend_stats(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            expected = float(i + 1)
            actual = float(row["rollouts_per_sec"])
            assert abs(actual - expected) < 1e-6, (
                f"Step {i}: expected rollouts_per_sec={expected}, got {actual}"
            )

    def test_each_line_has_tool_calls_per_sec(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            assert "tool_calls_per_sec" in row, (
                f"Line {i} missing 'tool_calls_per_sec' field"
            )

    def test_each_line_has_wall_secs(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            assert "wall_secs" in row, f"Line {i} missing 'wall_secs' field"
            assert isinstance(row["wall_secs"], (int, float)), (
                f"Line {i}: 'wall_secs' must be numeric"
            )

    def test_wall_secs_values_match_inputs(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            expected = float(i) * 0.5 + 0.1
            actual = float(row["wall_secs"])
            assert abs(actual - expected) < 1e-6, (
                f"Step {i}: expected wall_secs={expected}, got {actual}"
            )

    def test_each_line_has_warmup_flag(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl)
        for i, row in enumerate(rows):
            assert "warmup" in row, f"Line {i} missing 'warmup' field"
            assert isinstance(row["warmup"], bool), (
                f"Line {i}: 'warmup' must be a bool, got {type(row['warmup'])}"
            )

    def test_warmup_flag_values_are_correct(self, tmp_jsonl: Path):
        rows = self._run_10_steps(tmp_jsonl, warmup_count=2)
        for i, row in enumerate(rows):
            expected_warmup = i < 2
            assert row["warmup"] == expected_warmup, (
                f"Step {i}: expected warmup={expected_warmup}, got {row['warmup']}"
            )

    def test_output_is_valid_jsonl(self, tmp_jsonl: Path):
        """Every line in the output file must be a valid JSON object."""
        self._run_10_steps(tmp_jsonl)
        for line in tmp_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            assert isinstance(obj, dict), (
                f"Each JSONL line must be a JSON object (dict), got {type(obj)}"
            )


# ---------------------------------------------------------------------------
# C) Warmup exclusion from summary()
# ---------------------------------------------------------------------------

class TestSummaryWarmupExclusion:
    """Warmup steps must be excluded from summary() mean computations."""

    def test_summary_excludes_warmup_gpu_util(self, tmp_jsonl: Path):
        """mean_gpu_util must be computed over non-warmup steps only."""
        # Warmup steps: gpu_util=100.0 (high, must be excluded)
        # Non-warmup steps: gpu_util=10.0 (low, only these should count)
        sampled = iter([100.0, 100.0, 10.0, 10.0, 10.0])

        def fake_sampler() -> float:
            return next(sampled)

        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=fake_sampler)
        collector.start()
        collector.on_step(0, _make_backend_stats(), warmup=True, wall_secs=1.0)
        collector.on_step(1, _make_backend_stats(), warmup=True, wall_secs=1.0)
        collector.on_step(2, _make_backend_stats(), warmup=False, wall_secs=1.0)
        collector.on_step(3, _make_backend_stats(), warmup=False, wall_secs=1.0)
        collector.on_step(4, _make_backend_stats(), warmup=False, wall_secs=1.0)
        collector.stop()

        result = collector.summary()
        assert "mean_gpu_util" in result, "summary() must return 'mean_gpu_util'"
        assert abs(result["mean_gpu_util"] - 10.0) < 1e-6, (
            f"mean_gpu_util must be 10.0 (warmup excluded), got {result['mean_gpu_util']}"
        )

    def test_summary_excludes_warmup_rollouts_per_sec(self, tmp_jsonl: Path):
        """mean_rollouts_per_sec must be computed over non-warmup steps only."""
        fake_sampler = lambda: 50.0

        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=fake_sampler)
        collector.start()
        # Warmup: high throughput (must be excluded)
        collector.on_step(
            0,
            _make_backend_stats(rollouts_per_sec=999.0),
            warmup=True,
            wall_secs=1.0,
        )
        # Non-warmup: low throughput (the only values that should count)
        collector.on_step(
            1,
            _make_backend_stats(rollouts_per_sec=4.0),
            warmup=False,
            wall_secs=1.0,
        )
        collector.on_step(
            2,
            _make_backend_stats(rollouts_per_sec=6.0),
            warmup=False,
            wall_secs=1.0,
        )
        collector.stop()

        result = collector.summary()
        assert "mean_rollouts_per_sec" in result, (
            "summary() must return 'mean_rollouts_per_sec'"
        )
        assert abs(result["mean_rollouts_per_sec"] - 5.0) < 1e-6, (
            f"mean_rollouts_per_sec must be 5.0 (warmup excluded), "
            f"got {result['mean_rollouts_per_sec']}"
        )

    def test_summary_excludes_warmup_tool_calls_per_sec(self, tmp_jsonl: Path):
        """mean_tool_calls_per_sec must be computed over non-warmup steps only."""
        fake_sampler = lambda: 50.0

        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=fake_sampler)
        collector.start()
        collector.on_step(
            0,
            _make_backend_stats(tool_calls_per_sec=999.0),
            warmup=True,
            wall_secs=1.0,
        )
        collector.on_step(
            1,
            _make_backend_stats(tool_calls_per_sec=3.0),
            warmup=False,
            wall_secs=1.0,
        )
        collector.on_step(
            2,
            _make_backend_stats(tool_calls_per_sec=7.0),
            warmup=False,
            wall_secs=1.0,
        )
        collector.stop()

        result = collector.summary()
        assert "mean_tool_calls_per_sec" in result, (
            "summary() must return 'mean_tool_calls_per_sec'"
        )
        assert abs(result["mean_tool_calls_per_sec"] - 5.0) < 1e-6, (
            f"mean_tool_calls_per_sec must be 5.0 (warmup excluded), "
            f"got {result['mean_tool_calls_per_sec']}"
        )

    def test_summary_all_warmup_raises(self, tmp_jsonl: Path):
        """summary() must raise ValueError when all recorded steps are warmup."""
        fake_sampler = lambda: 50.0
        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=fake_sampler)
        collector.start()
        collector.on_step(0, _make_backend_stats(), warmup=True, wall_secs=1.0)
        collector.on_step(1, _make_backend_stats(), warmup=True, wall_secs=1.0)
        collector.stop()

        with pytest.raises(ValueError):
            collector.summary()

    def test_summary_returns_dict_with_required_keys(self, tmp_jsonl: Path):
        """summary() must return a dict with the three required keys."""
        fake_sampler = lambda: 40.0
        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=fake_sampler)
        collector.start()
        collector.on_step(0, _make_backend_stats(), warmup=False, wall_secs=1.0)
        collector.stop()

        result = collector.summary()
        assert isinstance(result, dict), (
            f"summary() must return a dict, got {type(result)}"
        )
        for key in ("mean_gpu_util", "mean_rollouts_per_sec", "mean_tool_calls_per_sec"):
            assert key in result, f"summary() result missing required key '{key}'"


# ---------------------------------------------------------------------------
# D) Mean accuracy over non-warmup steps (parametric)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("warmup_count,total_steps,gpu_util_value,expected_mean", [
    (0, 5, 60.0, 60.0),   # no warmup, uniform value
    (2, 7, 30.0, 30.0),   # 2 warmup steps, uniform non-warmup value
    (3, 3, 50.0, None),   # all warmup → expect ValueError
])
def test_summary_mean_gpu_util_parametric(
    tmp_path: Path,
    warmup_count: int,
    total_steps: int,
    gpu_util_value: float,
    expected_mean: float | None,
):
    """Parametric check of summary() mean_gpu_util under various warmup configs."""
    jsonl_path = tmp_path / "m.jsonl"
    fake_sampler = lambda: gpu_util_value

    collector = MetricCollector(str(jsonl_path), gpu_sampler=fake_sampler)
    collector.start()
    for step in range(total_steps):
        collector.on_step(
            step,
            _make_backend_stats(),
            warmup=(step < warmup_count),
            wall_secs=1.0,
        )
    collector.stop()

    if expected_mean is None:
        with pytest.raises(ValueError):
            collector.summary()
    else:
        result = collector.summary()
        assert abs(result["mean_gpu_util"] - expected_mean) < 1e-6, (
            f"Expected mean_gpu_util={expected_mean}, got {result['mean_gpu_util']}"
        )


# ---------------------------------------------------------------------------
# E) _parse_nvidia_smi_line — unit tests with injected fake subprocess
# ---------------------------------------------------------------------------

class TestParseNvidiaSmiLine:
    """Test the nvidia-smi line parser without shelling out."""

    def test_bare_integer(self):
        """A bare integer string like '42' must parse to 42.0."""
        assert _parse_nvidia_smi_line("42") == pytest.approx(42.0)

    def test_integer_with_percent_and_space(self):
        """'42 %' must also parse to 42.0."""
        assert _parse_nvidia_smi_line("42 %") == pytest.approx(42.0)

    def test_zero(self):
        assert _parse_nvidia_smi_line("0") == pytest.approx(0.0)

    def test_hundred(self):
        assert _parse_nvidia_smi_line("100") == pytest.approx(100.0)

    def test_leading_trailing_whitespace(self):
        """Whitespace around the integer must be stripped."""
        assert _parse_nvidia_smi_line("  73  ") == pytest.approx(73.0)

    def test_non_numeric_raises_value_error(self):
        """A line with non-numeric content must raise ValueError."""
        with pytest.raises(ValueError):
            _parse_nvidia_smi_line("N/A")

    def test_empty_line_raises_value_error(self):
        with pytest.raises(ValueError):
            _parse_nvidia_smi_line("")


# ---------------------------------------------------------------------------
# F) Lifecycle ordering
# ---------------------------------------------------------------------------

class TestLifecycleOrdering:
    def test_on_step_before_start_raises(self, tmp_jsonl: Path):
        """Calling on_step before start must raise (RuntimeError or similar)."""
        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=lambda: 50.0)
        with pytest.raises(Exception):
            collector.on_step(0, _make_backend_stats(), warmup=False, wall_secs=1.0)

    def test_summary_before_any_step_raises(self, tmp_jsonl: Path):
        """summary() before any on_step call must raise."""
        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=lambda: 50.0)
        collector.start()
        # No steps recorded yet
        with pytest.raises(Exception):
            collector.summary()

    def test_jsonl_file_created_on_start(self, tmp_jsonl: Path):
        """The JSONL file must be created (or truncated) when start() is called."""
        assert not tmp_jsonl.exists(), "Precondition: file must not exist before start()"
        collector = MetricCollector(str(tmp_jsonl), gpu_sampler=lambda: 50.0)
        collector.start()
        collector.stop()
        assert tmp_jsonl.exists(), "JSONL file must exist after start()+stop()"
