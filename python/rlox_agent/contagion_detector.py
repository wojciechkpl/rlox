# python/rlox_agent/contagion_detector.py
#
# Component 7: ContagionDetector (Step 6b, AC-5).
#
# Import constraints: stdlib only — no torch, no vllm, no pydantic, no numpy.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ContagionEvent:
    """One detected contagion event.

    Attributes:
        step: the step index at which the event was detected.
        kind: a short label, e.g. ``"vmrss_spike"``, ``"fd_leak"``,
            ``"server_reported"``.
        detail: human-readable description.
    """
    step: int
    kind: str
    detail: str


@dataclass
class ContagionReport:
    """Aggregated result returned by ``ContagionDetector.stop()``.

    Attributes:
        total_events: total count of contagion events. Zero means clean run.
        events: list of individual :class:`ContagionEvent` instances.
            Length always equals ``total_events``.
    """
    total_events: int
    events: list[ContagionEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default /proc reader (injectable for testing)
# ---------------------------------------------------------------------------

def _default_proc_reader(pid: int) -> dict:
    """Read selected fields from ``/proc/<pid>/status``.

    Returns:
        dict with at minimum ``{"VmRSS_kb": int, "FDSize": int}``.

    Raises:
        FileNotFoundError: if the process no longer exists.
        ValueError: if the status file cannot be parsed.
    """
    path = f"/proc/{pid}/status"
    result: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                # Format: "VmRSS:    123456 kB"
                parts = line.split()
                result["VmRSS_kb"] = int(parts[1])
            elif line.startswith("FDSize:"):
                # Format: "FDSize:   256"
                parts = line.split()
                result["FDSize"] = int(parts[1])
    if "VmRSS_kb" not in result or "FDSize" not in result:
        raise ValueError(
            f"Could not parse VmRSS_kb or FDSize from {path}"
        )
    return result


# ---------------------------------------------------------------------------
# ContagionDetector
# ---------------------------------------------------------------------------

_VMRSS_WINDOW_SIZE = 10          # rolling-baseline window length
_VMRSS_SPIKE_THRESHOLD = 0.20   # >20 % above baseline triggers rule (a)
_FD_GROWTH_THRESHOLD = 100       # strictly >100 triggers rule (b) consideration
_FD_GRACE_STEPS = 2              # must persist for more than this many steps


class ContagionDetector:
    """Monitor the training process for contagion during adversarial injection.

    Contagion event rules (AC-5):

    (a) **VmRSS spike** — VmRSS grows > 20 % above the 10-step rolling
        baseline during a step flagged adversarial.
    (b) **FD leak** — FDSize grows > 100 above baseline and does not return
        within 2 steps.
    (c) **Server-reported** — ``backend_stats["contagion_events"] > 0``.

    Args:
        proc_reader: callable ``(pid: int) -> dict`` returning at minimum
            ``{"VmRSS_kb": int, "FDSize": int}``. Defaults to a
            ``/proc/<pid>/status``-based reader.
    """

    def __init__(
        self,
        proc_reader: Callable[[int], dict] | None = None,
    ) -> None:
        self._proc_reader: Callable[[int], dict] = (
            proc_reader if proc_reader is not None else _default_proc_reader
        )
        self._pid: int | None = None
        self._started = False
        self._events: list[ContagionEvent] = []

        # VmRSS rolling baseline: last N readings
        self._vmrss_window: deque[int] = deque(maxlen=_VMRSS_WINDOW_SIZE)

        # FD leak tracking:
        #   _fd_baseline: the FDSize value considered "normal" (first reading or
        #     the last value before a jump).
        #   _fd_onset_step: step at which FD first exceeded baseline + threshold,
        #     or None if no active jump.
        self._fd_baseline: int | None = None
        self._fd_onset_step: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, training_pid: int) -> None:
        """Begin monitoring the given training process.

        Args:
            training_pid: PID of the training process to monitor.
        """
        self._pid = training_pid
        self._started = True
        self._events = []
        self._vmrss_window = deque(maxlen=_VMRSS_WINDOW_SIZE)
        self._fd_baseline = None
        self._fd_onset_step = None

    def stop(self) -> ContagionReport:
        """Stop monitoring and return the accumulated report.

        Returns:
            :class:`ContagionReport` with all detected events.

        Raises:
            RuntimeError: if called before ``start()``.
        """
        if not self._started:
            raise RuntimeError(
                "ContagionDetector.stop called before start(). "
                "Call start(training_pid) first."
            )
        self._started = False
        return ContagionReport(
            total_events=len(self._events),
            events=list(self._events),
        )

    # ------------------------------------------------------------------
    # Per-step hook
    # ------------------------------------------------------------------

    def on_step(self, step_idx: int, backend_stats: dict) -> None:
        """Check for contagion at the end of one training step.

        Args:
            step_idx: zero-based index of the training step.
            backend_stats: dict containing at minimum ``"contagion_events"``
                (int) and optionally ``"is_adversarial"`` (bool).

        Raises:
            RuntimeError: if called before ``start()``.
        """
        if not self._started:
            raise RuntimeError(
                "ContagionDetector.on_step called before start(). "
                "Call start(training_pid) first."
            )

        assert self._pid is not None  # guaranteed by start()

        # Read current process stats via the injected (or default) reader.
        proc_stats = self._proc_reader(self._pid)
        vmrss_kb: int = int(proc_stats["VmRSS_kb"])
        fd_size: int = int(proc_stats["FDSize"])

        # ------------------------------------------------------------------
        # Rule (a): VmRSS spike > 20 % above 10-step rolling baseline
        #           during an adversarial step.
        # ------------------------------------------------------------------
        is_adversarial: bool = bool(backend_stats.get("is_adversarial", False))

        if self._vmrss_window:
            vmrss_baseline = mean(self._vmrss_window)
            if (
                is_adversarial
                and vmrss_kb > vmrss_baseline * (1.0 + _VMRSS_SPIKE_THRESHOLD)
            ):
                pct = (vmrss_kb - vmrss_baseline) / vmrss_baseline * 100.0
                self._events.append(
                    ContagionEvent(
                        step=step_idx,
                        kind="vmrss_spike",
                        detail=(
                            f"VmRSS grew {pct:.1f}% above 10-step rolling baseline "
                            f"(baseline={vmrss_baseline:.0f} kB, "
                            f"current={vmrss_kb} kB)"
                        ),
                    )
                )

        # Always update the VmRSS rolling window with the current reading.
        self._vmrss_window.append(vmrss_kb)

        # ------------------------------------------------------------------
        # Rule (b): FD leak — FDSize grows > 100 above baseline and does not
        #           return within 2 steps.
        #
        # The FD "baseline" is the FDSize reading at the time the detector
        # was started (first call), or the last stable value before a jump.
        # We use a simple approach: track the stable baseline as the first
        # reading if no jump is active; once a jump resolves (either it
        # recovers or we emit and move on), we update the baseline to the
        # current level.
        # ------------------------------------------------------------------
        if self._fd_baseline is None:
            # First reading — establish the baseline.
            self._fd_baseline = fd_size
        else:
            fd_growth = fd_size - self._fd_baseline

            if self._fd_onset_step is None:
                # No active jump. Check if this reading starts one.
                if fd_growth > _FD_GROWTH_THRESHOLD:
                    self._fd_onset_step = step_idx
                # If no jump, we do NOT update the baseline from elevated readings;
                # the baseline remains anchored to the stable value.
            else:
                # Active jump started at _fd_onset_step. Check persistence.
                steps_elevated = step_idx - self._fd_onset_step

                if fd_growth <= _FD_GROWTH_THRESHOLD:
                    # Recovered — cancel the onset (transient).
                    self._fd_onset_step = None
                    # Optionally update baseline to current (now stable) value.
                    self._fd_baseline = fd_size
                elif steps_elevated > _FD_GRACE_STEPS:
                    # Still elevated and persistence exceeds grace window → emit.
                    # Only emit once per onset (then reset onset so we don't
                    # re-emit for every subsequent step).
                    self._events.append(
                        ContagionEvent(
                            step=step_idx,
                            kind="fd_leak",
                            detail=(
                                f"FDSize grew {fd_growth} above baseline "
                                f"(baseline={self._fd_baseline}, "
                                f"current={fd_size}) and persisted for "
                                f"{steps_elevated} steps"
                            ),
                        )
                    )
                    # Reset onset so we emit only once per leak episode.
                    self._fd_onset_step = None
                    # Update baseline to the current elevated level so a further
                    # jump from here is tracked correctly.
                    self._fd_baseline = fd_size

        # ------------------------------------------------------------------
        # Rule (c): Server-reported contagion_events > 0.
        # ------------------------------------------------------------------
        server_contagion: int = int(backend_stats.get("contagion_events", 0))
        if server_contagion > 0:
            self._events.append(
                ContagionEvent(
                    step=step_idx,
                    kind="server_reported",
                    detail=(
                        f"backend_stats reported {server_contagion} contagion "
                        f"event(s) at step {step_idx}"
                    ),
                )
            )
