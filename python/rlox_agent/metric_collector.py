# python/rlox_agent/metric_collector.py
#
# Component 6: MetricCollector (Step 6a).
#
# Import constraints: stdlib only — no torch, no vllm, no pydantic, no numpy.
from __future__ import annotations

import json
import subprocess
from statistics import mean
from typing import Callable


# ---------------------------------------------------------------------------
# Default GPU sampler (injectable for testing)
# ---------------------------------------------------------------------------

def _parse_nvidia_smi_line(line: str) -> float:
    """Parse a single output line from ``nvidia-smi --query-gpu=utilization.gpu
    --format=csv,noheader,nounits``.

    Accepts bare integers (``"42"``), percent-suffixed (``"42 %"``), and
    leading/trailing whitespace (``"  73  "``).

    Args:
        line: one line of nvidia-smi output.

    Returns:
        GPU utilisation as a float (0–100).

    Raises:
        ValueError: if the line is empty or cannot be parsed as a number.
    """
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"Cannot parse empty nvidia-smi line: {line!r}")
    # Remove trailing " %" if present
    token = stripped.split()[0]
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse nvidia-smi line as float: {line!r}"
        ) from exc


def _run_nvidia_smi() -> str:
    """Run nvidia-smi and return the raw stdout string.

    Separated from parsing so tests can unit-test ``_parse_nvidia_smi_line``
    without a GPU and integration-test the full sampler by injecting a fake
    runner.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _default_gpu_sampler() -> float:
    """Sample current GPU utilization via nvidia-smi.

    Returns utilization as a float in [0, 100].
    Raises RuntimeError if nvidia-smi is unavailable or output cannot be parsed.
    """
    try:
        raw = _run_nvidia_smi()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"Failed to run nvidia-smi: {exc}") from exc
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    try:
        return _parse_nvidia_smi_line(line)
    except ValueError as exc:
        raise RuntimeError(f"Failed to parse nvidia-smi output: {exc}") from exc


# ---------------------------------------------------------------------------
# MetricCollector
# ---------------------------------------------------------------------------

class MetricCollector:
    """Per-step metric collector for the rlox benchmark harness.

    Records GPU utilization, throughput (from BackendStats), wall-clock time,
    and a warmup flag for each training step. Persists one JSON object per
    step to a JSONL file. Provides ``summary()`` over non-warmup steps only.

    The GPU utilization sampler is injectable via the constructor so tests can
    supply a deterministic fake sampler without shelling out to nvidia-smi.

    Args:
        jsonl_path: path to the output JSONL file. Created (or truncated) on
            ``start()``.
        gpu_sampler: callable that returns the current GPU utilization as a
            float in [0, 100]. Defaults to an nvidia-smi-based implementation.
            Tests MUST inject a fake sampler; real nvidia-smi must never be
            called during the test suite.
    """

    def __init__(
        self,
        jsonl_path: str,
        gpu_sampler: Callable[[], float] | None = None,
    ) -> None:
        self._jsonl_path = jsonl_path
        self._gpu_sampler: Callable[[], float] = (
            gpu_sampler if gpu_sampler is not None else _default_gpu_sampler
        )
        self._file = None
        self._started = False
        # Non-warmup step metrics for summary()
        self._non_warmup_gpu_utils: list[float] = []
        self._non_warmup_rollouts_per_sec: list[float] = []
        self._non_warmup_tool_calls_per_sec: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the output file (create or truncate) and initialise state."""
        self._file = open(self._jsonl_path, "w", encoding="utf-8")  # noqa: SIM115
        self._started = True
        self._non_warmup_gpu_utils = []
        self._non_warmup_rollouts_per_sec = []
        self._non_warmup_tool_calls_per_sec = []

    def stop(self) -> None:
        """Flush and close the output file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def on_step(
        self,
        step_idx: int,
        backend_stats: dict,
        *,
        warmup: bool,
        wall_secs: float,
    ) -> None:
        """Record metrics for one training step and append a line to the JSONL.

        Args:
            step_idx: zero-based index of the training step.
            backend_stats: dict containing at minimum ``"rollouts_per_sec"``
                and ``"tool_calls_per_sec"`` float fields.
            warmup: if True this step is a warm-up step and is excluded from
                ``summary()`` aggregation.
            wall_secs: wall-clock duration of this step in seconds.

        Raises:
            RuntimeError: if called before ``start()``.
        """
        if not self._started:
            raise RuntimeError(
                "MetricCollector.on_step called before start(). "
                "Call start() first."
            )

        gpu_util: float = self._gpu_sampler()
        rollouts_per_sec: float = float(backend_stats["rollouts_per_sec"])
        tool_calls_per_sec: float = float(backend_stats["tool_calls_per_sec"])

        record = {
            "step_idx": step_idx,
            "gpu_util": gpu_util,
            "rollouts_per_sec": rollouts_per_sec,
            "tool_calls_per_sec": tool_calls_per_sec,
            "wall_secs": wall_secs,
            "warmup": warmup,
        }
        self._file.write(json.dumps(record) + "\n")

        if not warmup:
            self._non_warmup_gpu_utils.append(gpu_util)
            self._non_warmup_rollouts_per_sec.append(rollouts_per_sec)
            self._non_warmup_tool_calls_per_sec.append(tool_calls_per_sec)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return mean GPU utilization and mean throughput over non-warmup steps.

        Returns:
            dict with keys:
                ``"mean_gpu_util"`` (float),
                ``"mean_rollouts_per_sec"`` (float),
                ``"mean_tool_calls_per_sec"`` (float).

        Raises:
            ValueError: if no non-warmup steps have been recorded yet.
        """
        if not self._non_warmup_gpu_utils:
            raise ValueError(
                "summary() called with no non-warmup steps recorded. "
                "Ensure at least one on_step(..., warmup=False) has been called."
            )
        return {
            "mean_gpu_util": mean(self._non_warmup_gpu_utils),
            "mean_rollouts_per_sec": mean(self._non_warmup_rollouts_per_sec),
            "mean_tool_calls_per_sec": mean(self._non_warmup_tool_calls_per_sec),
        }
