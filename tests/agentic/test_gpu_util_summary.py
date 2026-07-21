"""Coverage / characterization tests for trl_grpo_run._summarise_gpu_util.

This function converts a list of GPU-utilisation readings into a summary dict.
It is a pure function (no I/O, no side effects) so it is safe to test in a
light venv without torch.

Contract:
  _summarise_gpu_util(samples: list[float]) -> dict

  When samples is non-empty:
    - "mean_gpu_util" is the rounded arithmetic mean of the samples.
    - "min_gpu_util"  is the rounded minimum.
    - "gpu_util_samples" is len(samples).
    - All three values are non-None.

  When samples is empty:
    - "mean_gpu_util" is None.
    - "min_gpu_util"  is None.
    - "gpu_util_samples" is 0 (not None).

Import strategy: import the module at module level (no torch needed at module
load time — verified manually before writing this file).  If this import ever
fails with ImportError/ModuleNotFoundError that is itself a test finding.
"""

from __future__ import annotations

import pytest

# benchmarks/agentic/ is on sys.path via conftest.py.
# trl_grpo_run does NOT import torch at module scope — only inside train() /
# seed_everything() / build_dataset() etc., which we never call here.
from trl_grpo_run import _summarise_gpu_util


# ---------------------------------------------------------------------------
# A) Module-level import does not pull torch
# ---------------------------------------------------------------------------


class TestModuleImportNoBadSideEffects:
    def test_function_importable(self):
        """_summarise_gpu_util must be importable without torch installed."""
        assert callable(_summarise_gpu_util)

    def test_torch_not_imported_at_module_level(self):
        """Importing trl_grpo_run must not import torch into sys.modules.
        (Torch is only used lazily inside train() / seed_everything().)"""
        import sys

        # After the top-level import of trl_grpo_run (done at test-module load
        # time), torch must NOT appear in sys.modules.
        assert "torch" not in sys.modules, (
            "torch was imported at trl_grpo_run module level — this breaks the "
            "light-venv test environment. Move torch imports inside the functions "
            "that need them."
        )


# ---------------------------------------------------------------------------
# B) Empty samples list — the "no GPU" case
# ---------------------------------------------------------------------------


class TestEmptySamples:
    def test_returns_dict(self):
        result = _summarise_gpu_util([])
        assert isinstance(result, dict)

    def test_mean_gpu_util_is_none(self):
        result = _summarise_gpu_util([])
        assert result["mean_gpu_util"] is None

    def test_min_gpu_util_is_none(self):
        result = _summarise_gpu_util([])
        assert result["min_gpu_util"] is None

    def test_gpu_util_samples_is_zero(self):
        result = _summarise_gpu_util([])
        assert result["gpu_util_samples"] == 0

    def test_gpu_util_samples_is_int(self):
        result = _summarise_gpu_util([])
        assert isinstance(result["gpu_util_samples"], int)

    def test_all_expected_keys_present_when_empty(self):
        result = _summarise_gpu_util([])
        assert set(result.keys()) >= {"mean_gpu_util", "min_gpu_util", "gpu_util_samples"}


# ---------------------------------------------------------------------------
# C) Normal (non-empty) samples list
# ---------------------------------------------------------------------------


class TestNormalSamples:
    def test_returns_dict(self):
        result = _summarise_gpu_util([50.0, 60.0, 70.0])
        assert isinstance(result, dict)

    def test_mean_gpu_util_is_not_none(self):
        result = _summarise_gpu_util([50.0, 60.0, 70.0])
        assert result["mean_gpu_util"] is not None

    def test_mean_gpu_util_correct_value(self):
        """mean = (50 + 60 + 70) / 3 = 60.0"""
        result = _summarise_gpu_util([50.0, 60.0, 70.0])
        assert result["mean_gpu_util"] == pytest.approx(60.0, abs=0.01)

    def test_min_gpu_util_is_not_none(self):
        result = _summarise_gpu_util([50.0, 60.0, 70.0])
        assert result["min_gpu_util"] is not None

    def test_min_gpu_util_correct_value(self):
        result = _summarise_gpu_util([50.0, 60.0, 70.0])
        assert result["min_gpu_util"] == pytest.approx(50.0, abs=0.01)

    def test_gpu_util_samples_correct_count(self):
        result = _summarise_gpu_util([50.0, 60.0, 70.0])
        assert result["gpu_util_samples"] == 3

    def test_gpu_util_samples_is_int(self):
        result = _summarise_gpu_util([50.0, 60.0, 70.0])
        assert isinstance(result["gpu_util_samples"], int)

    def test_all_expected_keys_present_when_non_empty(self):
        result = _summarise_gpu_util([80.0])
        assert set(result.keys()) >= {"mean_gpu_util", "min_gpu_util", "gpu_util_samples"}

    def test_single_element_mean_equals_value(self):
        result = _summarise_gpu_util([75.5])
        assert result["mean_gpu_util"] == pytest.approx(75.5, abs=0.01)

    def test_single_element_min_equals_value(self):
        result = _summarise_gpu_util([75.5])
        assert result["min_gpu_util"] == pytest.approx(75.5, abs=0.01)

    def test_single_element_samples_is_1(self):
        result = _summarise_gpu_util([75.5])
        assert result["gpu_util_samples"] == 1


# ---------------------------------------------------------------------------
# D) Rounding behaviour (documented: values are round(…, 2))
# ---------------------------------------------------------------------------


class TestRounding:
    def test_mean_is_rounded_to_2_decimal_places(self):
        """(10 + 20 + 30) / 3 = 20.0 exactly; use a case that requires rounding."""
        # 1/3 ≈ 0.3333... → rounded to 2dp = 0.33
        result = _summarise_gpu_util([0.0, 0.0, 1.0])
        # mean = 1/3 ≈ 0.333... → round to 2dp → 0.33
        assert result["mean_gpu_util"] == round(1 / 3, 2)

    def test_min_is_rounded_to_2_decimal_places(self):
        result = _summarise_gpu_util([33.333, 66.666])
        assert result["min_gpu_util"] == round(33.333, 2)

    def test_mean_rounded_for_three_even_values(self):
        result = _summarise_gpu_util([50.0, 60.0, 70.0])
        # 60.0 has no fractional part — rounding should leave it unchanged
        assert result["mean_gpu_util"] == 60.0


# ---------------------------------------------------------------------------
# E) Parametric cases — (samples, expected_mean, expected_min, expected_n)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "samples, expected_mean, expected_min, expected_n",
    [
        # All zeros — valid GPU idle state
        ([0.0, 0.0, 0.0], 0.0, 0.0, 3),
        # All 100 — full utilisation
        ([100.0, 100.0], 100.0, 100.0, 2),
        # Mixed values
        ([20.0, 80.0], 50.0, 20.0, 2),
        # Single high value
        ([99.0], 99.0, 99.0, 1),
        # Ascending series
        ([10.0, 20.0, 30.0, 40.0], 25.0, 10.0, 4),
    ],
)
def test_summarise_parametric(samples, expected_mean, expected_min, expected_n):
    result = _summarise_gpu_util(samples)
    assert result["mean_gpu_util"] == pytest.approx(expected_mean, abs=0.01)
    assert result["min_gpu_util"] == pytest.approx(expected_min, abs=0.01)
    assert result["gpu_util_samples"] == expected_n


# ---------------------------------------------------------------------------
# F) Value boundaries — GPU util is clamped to [0, 100] in practice
#    (the function itself does not validate; we just document that it passes
#    through whatever the sampler produces).
# ---------------------------------------------------------------------------


class TestValueBoundaries:
    def test_zero_util_list(self):
        result = _summarise_gpu_util([0.0, 0.0])
        assert result["mean_gpu_util"] == pytest.approx(0.0)
        assert result["min_gpu_util"] == pytest.approx(0.0)
        assert result["gpu_util_samples"] == 2

    def test_hundred_util_list(self):
        result = _summarise_gpu_util([100.0, 100.0, 100.0])
        assert result["mean_gpu_util"] == pytest.approx(100.0)
        assert result["min_gpu_util"] == pytest.approx(100.0)

    def test_large_sample_count(self):
        """Stress test: 3600 samples (one per second of a 1-hour run)."""
        samples = [float(i % 101) for i in range(3600)]
        result = _summarise_gpu_util(samples)
        assert result["gpu_util_samples"] == 3600
        assert result["mean_gpu_util"] is not None
        assert result["min_gpu_util"] is not None
