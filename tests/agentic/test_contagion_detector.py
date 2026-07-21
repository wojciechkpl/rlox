"""RED-phase tests for Component 7: ContagionDetector (Step 6b, AC-5).

Contract being specified
------------------------
ContagionDetector monitors a training PID for state changes attributable to
adversarial injection.  The process-stat reader is injectable so tests never
touch a real /proc entry.

Contagion rules under test:

  (a) VmRSS spike  — VmRSS grows >20 % above the 10-step rolling baseline
      during a step flagged adversarial; transient normal noise must NOT fire.
  (b) FD leak      — FDSize grows >100 above baseline and does NOT return
      within 2 steps; a transient bump that recovers within 2 steps is benign.
  (c) Server-side  — backend_stats["contagion_events"] > 0 → event.
  (d) Clean run    — flat stats, contagion_events=0 → total_events == 0.

ContagionReport: total_events (int), events (list[ContagionEvent]).
ContagionEvent: step (int), kind (str), detail (str).

All imports are top-level: conftest.py injects python/rlox/agentic/ onto
sys.path.
"""
from __future__ import annotations

from dataclasses import fields as dc_fields
import dataclasses

import pytest

# Top-level imports — never "from rlox.agentic import ..."
import rlox_agent.contagion_detector as cd
from rlox_agent.contagion_detector import (
    ContagionDetector,
    ContagionEvent,
    ContagionReport,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_PID = 99999  # never actually read — the injected proc_reader intercepts it


def _clean_stats(contagion_events: int = 0, is_adversarial: bool = False) -> dict:
    """Return a minimal BackendStats-shaped dict with no anomalies."""
    return {
        "batch_wall_secs": 1.0,
        "rollouts_completed": 4,
        "rollouts_per_sec": 2.0,
        "tool_calls_per_sec": 3.0,
        "adversarial_injected": 1 if is_adversarial else 0,
        "adversarial_contained": 1 if is_adversarial else 0,
        "contagion_events": contagion_events,
        "setup_error_events": 0,
        "time_to_contain_secs": [],
        "cgroup_freeze_events": 0,
        "cgroup_kill_events": 0,
        "oom_kill_events": 0,
        "gpu_idle_attributable_to_hang_secs": 0.0,
        "step_index": 0,
        "is_adversarial": is_adversarial,
    }


def _const_proc_reader(vmrss_kb: int, fd_size: int):
    """Return a proc_reader callable that always returns fixed values."""
    def reader(pid: int) -> dict:
        return {"VmRSS_kb": vmrss_kb, "FDSize": fd_size}
    return reader


def _sequence_proc_reader(readings: list[dict]):
    """Return a proc_reader that pops from a sequence of dicts per call."""
    it = iter(readings)

    def reader(pid: int) -> dict:
        return next(it)

    return reader


# ---------------------------------------------------------------------------
# A) Data types
# ---------------------------------------------------------------------------

class TestContagionEventDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ContagionEvent)

    def test_has_step_field(self):
        field_names = {f.name for f in dc_fields(ContagionEvent)}
        assert "step" in field_names

    def test_has_kind_field(self):
        field_names = {f.name for f in dc_fields(ContagionEvent)}
        assert "kind" in field_names

    def test_has_detail_field(self):
        field_names = {f.name for f in dc_fields(ContagionEvent)}
        assert "detail" in field_names

    def test_can_construct(self):
        evt = ContagionEvent(step=3, kind="vmrss_spike", detail="grew 25%")
        assert evt.step == 3
        assert evt.kind == "vmrss_spike"
        assert evt.detail == "grew 25%"


class TestContagionReportDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ContagionReport)

    def test_has_total_events_field(self):
        field_names = {f.name for f in dc_fields(ContagionReport)}
        assert "total_events" in field_names

    def test_has_events_field(self):
        field_names = {f.name for f in dc_fields(ContagionReport)}
        assert "events" in field_names

    def test_total_events_is_int(self):
        report = ContagionReport(total_events=0)
        assert isinstance(report.total_events, int)

    def test_events_is_list(self):
        report = ContagionReport(total_events=0)
        assert isinstance(report.events, list)

    def test_events_length_matches_total(self):
        """total_events must equal len(events) in a well-formed report."""
        evt = ContagionEvent(step=1, kind="test", detail="x")
        report = ContagionReport(total_events=1, events=[evt])
        assert report.total_events == len(report.events)


# ---------------------------------------------------------------------------
# B) Constructor / injectable proc_reader
# ---------------------------------------------------------------------------

class TestContagionDetectorConstructor:
    def test_can_construct_with_injected_reader(self):
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        assert detector is not None

    def test_can_construct_with_default_reader(self):
        """ContagionDetector must be constructible without supplying proc_reader."""
        detector = ContagionDetector()
        assert detector is not None

    def test_injected_reader_is_used_not_proc(self, monkeypatch):
        """When a fake reader is injected, real /proc must never be opened.

        We monkeypatch builtins.open so that opening a /proc path raises, proving
        the injected reader is the one used.
        """
        import builtins

        original_open = builtins.open

        def guarded_open(path, *args, **kwargs):
            if str(path).startswith("/proc"):
                raise AssertionError(
                    f"Real /proc was opened ({path}) — the injected proc_reader "
                    "must be used instead"
                )
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", guarded_open)

        call_count = 0

        def fake_reader(pid: int) -> dict:
            nonlocal call_count
            call_count += 1
            return {"VmRSS_kb": 200_000, "FDSize": 100}

        detector = ContagionDetector(proc_reader=fake_reader)
        detector.start(_FAKE_PID)

        # Run a few steps — the fake reader must be called, not /proc
        for step in range(5):
            detector.on_step(step, _clean_stats())

        detector.stop()

        assert call_count >= 1, (
            "The injected proc_reader was never called — ContagionDetector must "
            "call the provided reader, not fall back to /proc"
        )


# ---------------------------------------------------------------------------
# C) Clean run → zero events
# ---------------------------------------------------------------------------

class TestCleanRun:
    def test_clean_run_produces_zero_total_events(self):
        """Flat stats + contagion_events=0 must yield total_events == 0."""
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(15):
            detector.on_step(step, _clean_stats(contagion_events=0))

        report = detector.stop()

        assert isinstance(report, ContagionReport), (
            f"stop() must return a ContagionReport, got {type(report)}"
        )
        assert report.total_events == 0, (
            f"Clean run must produce total_events=0, got {report.total_events}. "
            f"Events: {report.events}"
        )
        assert report.events == [], (
            f"Clean run must produce empty events list, got {report.events}"
        )

    def test_noisy_but_bounded_vmrss_growth_no_event(self):
        """VmRSS noise within 20 % of baseline must NOT trigger a contagion event.

        Uses a gradually increasing VmRSS that grows, but never exceeds 20 % of
        the rolling baseline in a single step.
        """
        # Build a sequence of readings: start at 100_000 KB, increase by 1 %
        # per step (well within the 20 % threshold).
        readings = [
            {"VmRSS_kb": int(100_000 * (1.01 ** i)), "FDSize": 50}
            for i in range(20)
        ]
        reader = _sequence_proc_reader(readings)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(20):
            detector.on_step(step, _clean_stats())

        report = detector.stop()
        assert report.total_events == 0, (
            f"Gradual 1%/step VmRSS growth must NOT trigger contagion events, "
            f"got total_events={report.total_events}. Events: {report.events}"
        )


# ---------------------------------------------------------------------------
# D) Rule (c) — server-reported contagion_events > 0
# ---------------------------------------------------------------------------

class TestServerReportedContagion:
    def test_contagion_events_one_triggers_event(self):
        """backend_stats with contagion_events=1 must produce >= 1 contagion event."""
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        # First few steps are clean
        for step in range(3):
            detector.on_step(step, _clean_stats(contagion_events=0))

        # One step with server-reported contagion
        detector.on_step(3, _clean_stats(contagion_events=1))

        # A few more clean steps
        for step in range(4, 8):
            detector.on_step(step, _clean_stats(contagion_events=0))

        report = detector.stop()

        assert report.total_events >= 1, (
            f"backend_stats contagion_events=1 must produce >= 1 ContagionEvent, "
            f"got total_events={report.total_events}"
        )

    def test_contagion_events_three_all_counted(self):
        """Three separate server-reported contagion steps must each produce an event."""
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(10):
            # Steps 2, 5, 8 have server-reported contagion
            contagion = 1 if step in (2, 5, 8) else 0
            detector.on_step(step, _clean_stats(contagion_events=contagion))

        report = detector.stop()
        assert report.total_events >= 3, (
            f"Three server-reported contagion steps must produce >= 3 events, "
            f"got total_events={report.total_events}"
        )

    def test_server_event_kind_label(self):
        """Server-reported events must have kind 'server_reported' (or similar)."""
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)
        detector.on_step(0, _clean_stats(contagion_events=1))
        report = detector.stop()

        assert report.total_events >= 1
        server_events = [e for e in report.events if "server" in e.kind.lower()]
        assert len(server_events) >= 1, (
            f"At least one event must have 'server' in its kind label. "
            f"Actual kinds: {[e.kind for e in report.events]}"
        )


# ---------------------------------------------------------------------------
# E) Rule (a) — VmRSS spike >20 % above 10-step rolling baseline
# ---------------------------------------------------------------------------

class TestVmRssSpikeDetection:
    def test_vmrss_spike_above_20_percent_triggers_event(self):
        """A VmRSS spike >20 % above rolling baseline during adversarial step → event.

        Strategy: establish a 10-step baseline at 100_000 KB, then spike to
        125_000 KB (25 % above baseline) on an adversarial step.
        """
        # 10 baseline readings, then spike, then back to normal
        readings = (
            [{"VmRSS_kb": 100_000, "FDSize": 50}] * 11
            + [{"VmRSS_kb": 125_000, "FDSize": 50}]  # step 11: 25% spike
            + [{"VmRSS_kb": 100_000, "FDSize": 50}] * 3
        )
        reader = _sequence_proc_reader(readings)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        # Steps 0–10: baseline establishment, non-adversarial
        for step in range(11):
            detector.on_step(step, _clean_stats(is_adversarial=False))

        # Step 11: adversarial, VmRSS spikes 25 % above baseline
        detector.on_step(11, _clean_stats(is_adversarial=True))

        # Steps 12–14: normal
        for step in range(12, 15):
            detector.on_step(step, _clean_stats(is_adversarial=False))

        report = detector.stop()

        vmrss_events = [e for e in report.events if "vmrss" in e.kind.lower()]
        assert len(vmrss_events) >= 1, (
            f"A 25% VmRSS spike on an adversarial step must trigger a vmrss event. "
            f"total_events={report.total_events}, events={report.events}"
        )

    def test_vmrss_spike_below_threshold_no_event(self):
        """A VmRSS increase of exactly 19 % must NOT trigger a contagion event."""
        # 10-step baseline at 100_000, then 119_000 on an adversarial step (19 %)
        readings = (
            [{"VmRSS_kb": 100_000, "FDSize": 50}] * 11
            + [{"VmRSS_kb": 119_000, "FDSize": 50}]  # step 11: 19% — below threshold
            + [{"VmRSS_kb": 100_000, "FDSize": 50}] * 3
        )
        reader = _sequence_proc_reader(readings)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(11):
            detector.on_step(step, _clean_stats(is_adversarial=False))

        detector.on_step(11, _clean_stats(is_adversarial=True))

        for step in range(12, 15):
            detector.on_step(step, _clean_stats(is_adversarial=False))

        report = detector.stop()

        vmrss_events = [e for e in report.events if "vmrss" in e.kind.lower()]
        assert len(vmrss_events) == 0, (
            f"A 19% VmRSS increase (below 20% threshold) must NOT trigger a vmrss event. "
            f"Triggered events: {vmrss_events}"
        )

    def test_vmrss_spike_non_adversarial_step_behavior(self):
        """A VmRSS spike on a non-adversarial step should NOT trigger rule (a).

        Rule (a) is specifically for adversarial steps.  Spikes on clean steps
        may or may not produce events under other rules, but must not be rule-a
        vmrss events.
        """
        readings = (
            [{"VmRSS_kb": 100_000, "FDSize": 50}] * 11
            + [{"VmRSS_kb": 130_000, "FDSize": 50}]  # 30% spike on non-adversarial step
            + [{"VmRSS_kb": 100_000, "FDSize": 50}] * 3
        )
        reader = _sequence_proc_reader(readings)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(11):
            detector.on_step(step, _clean_stats(is_adversarial=False))

        # Non-adversarial step with large VmRSS jump
        detector.on_step(11, _clean_stats(is_adversarial=False))

        for step in range(12, 15):
            detector.on_step(step, _clean_stats(is_adversarial=False))

        report = detector.stop()

        # Must NOT produce a rule-(a) vmrss_spike event
        vmrss_spike_events = [
            e for e in report.events
            if "vmrss" in e.kind.lower() and "spike" in e.kind.lower()
        ]
        assert len(vmrss_spike_events) == 0, (
            f"Rule (a) vmrss_spike must only fire on adversarial steps. "
            f"Got vmrss spike events on non-adversarial step: {vmrss_spike_events}"
        )


# ---------------------------------------------------------------------------
# F) Rule (b) — FD leak: FDSize grows >100 and does not return within 2 steps
# ---------------------------------------------------------------------------

class TestFdLeakDetection:
    def test_fd_jump_above_100_persisting_triggers_event(self):
        """FDSize jumps +120 and remains high for >2 steps → event.

        The event must be of kind 'fd_leak' (or contain 'fd' in the kind string).
        """
        # Baseline: FDSize=50 for 5 steps, then jump to 170 and stay there.
        readings = (
            [{"VmRSS_kb": 100_000, "FDSize": 50}] * 5   # baseline
            + [{"VmRSS_kb": 100_000, "FDSize": 170}] * 6  # jump +120, persists
        )
        reader = _sequence_proc_reader(readings)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(11):
            detector.on_step(step, _clean_stats())

        report = detector.stop()

        fd_events = [e for e in report.events if "fd" in e.kind.lower()]
        assert len(fd_events) >= 1, (
            f"FDSize jump of +120 persisting >2 steps must produce >= 1 fd event. "
            f"total_events={report.total_events}, events={report.events}"
        )

    def test_fd_jump_above_100_transient_recovery_no_event(self):
        """FDSize jumps +120 but returns to baseline within 2 steps → no fd event.

        Per the spec: 'does not return within 2 steps' is what triggers rule (b).
        A transient bump that recovers within 2 steps is benign.
        """
        # Baseline: FDSize=50 for 5 steps
        # Then: FDSize=170 for 1 step (jump of +120)
        # Then: FDSize=55 for 2 steps (recovered within 2 steps)
        # Then: FDSize=50 for remaining steps
        readings = (
            [{"VmRSS_kb": 100_000, "FDSize": 50}] * 5    # baseline
            + [{"VmRSS_kb": 100_000, "FDSize": 170}]      # step 5: jump +120
            + [{"VmRSS_kb": 100_000, "FDSize": 55}] * 2   # steps 6-7: recovered
            + [{"VmRSS_kb": 100_000, "FDSize": 50}] * 5   # steps 8-12: back to normal
        )
        reader = _sequence_proc_reader(readings)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(13):
            detector.on_step(step, _clean_stats())

        report = detector.stop()

        fd_events = [e for e in report.events if "fd" in e.kind.lower()]
        assert len(fd_events) == 0, (
            f"A transient FD bump that returns within 2 steps must NOT trigger "
            f"an fd event. Got fd events: {fd_events}"
        )

    def test_fd_jump_exactly_100_no_event(self):
        """FDSize grows by exactly 100 (not >100) must NOT trigger a fd event."""
        readings = (
            [{"VmRSS_kb": 100_000, "FDSize": 50}] * 5
            + [{"VmRSS_kb": 100_000, "FDSize": 150}] * 6  # jump exactly +100
        )
        reader = _sequence_proc_reader(readings)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(11):
            detector.on_step(step, _clean_stats())

        report = detector.stop()

        fd_events = [e for e in report.events if "fd" in e.kind.lower()]
        assert len(fd_events) == 0, (
            f"FDSize growth of exactly 100 (threshold is >100) must NOT trigger "
            f"an fd event. Got: {fd_events}"
        )

    def test_fd_jump_101_persisting_triggers_event(self):
        """FDSize grows by 101 (>100) and persists → event (boundary above threshold)."""
        readings = (
            [{"VmRSS_kb": 100_000, "FDSize": 50}] * 5
            + [{"VmRSS_kb": 100_000, "FDSize": 151}] * 6  # jump +101
        )
        reader = _sequence_proc_reader(readings)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(11):
            detector.on_step(step, _clean_stats())

        report = detector.stop()

        fd_events = [e for e in report.events if "fd" in e.kind.lower()]
        assert len(fd_events) >= 1, (
            f"FDSize growth of +101 (>100) persisting >2 steps must trigger an "
            f"fd event. Got total_events={report.total_events}, events={report.events}"
        )


# ---------------------------------------------------------------------------
# G) Report structure consistency
# ---------------------------------------------------------------------------

class TestReportConsistency:
    def test_total_events_equals_len_events(self):
        """total_events must always equal len(events) in the returned report."""
        # Mix of clean and server-reported contagion steps
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        for step in range(10):
            contagion = 1 if step in (3, 7) else 0
            detector.on_step(step, _clean_stats(contagion_events=contagion))

        report = detector.stop()

        assert report.total_events == len(report.events), (
            f"total_events ({report.total_events}) must equal len(events) "
            f"({len(report.events)})"
        )

    def test_each_event_has_required_fields(self):
        """Every ContagionEvent in the report must have step, kind, and detail."""
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)
        detector.on_step(0, _clean_stats(contagion_events=1))
        report = detector.stop()

        for evt in report.events:
            assert isinstance(evt, ContagionEvent), (
                f"Each element of events must be a ContagionEvent, got {type(evt)}"
            )
            assert isinstance(evt.step, int), f"event.step must be int, got {type(evt.step)}"
            assert isinstance(evt.kind, str) and evt.kind, (
                f"event.kind must be a non-empty str, got {evt.kind!r}"
            )
            assert isinstance(evt.detail, str), (
                f"event.detail must be a str, got {type(evt.detail)}"
            )

    def test_event_step_index_matches_step_argument(self):
        """ContagionEvent.step must equal the step_idx argument of on_step()."""
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        # Only step 4 has a server-reported event
        for step in range(8):
            contagion = 1 if step == 4 else 0
            detector.on_step(step, _clean_stats(contagion_events=contagion))

        report = detector.stop()

        assert report.total_events >= 1
        # The event must be attributed to step 4
        event_steps = [e.step for e in report.events]
        assert 4 in event_steps, (
            f"Event must be attributed to step 4 (where contagion_events=1 was passed), "
            f"got event steps: {event_steps}"
        )

    def test_stop_returns_contagion_report_instance(self):
        """stop() must return a ContagionReport instance."""
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)
        detector.on_step(0, _clean_stats())
        report = detector.stop()
        assert isinstance(report, ContagionReport), (
            f"stop() must return a ContagionReport, got {type(report)}"
        )


# ---------------------------------------------------------------------------
# H) Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_on_step_before_start_raises(self):
        """on_step before start must raise (RuntimeError or similar)."""
        detector = ContagionDetector(proc_reader=_const_proc_reader(100_000, 50))
        with pytest.raises(Exception):
            detector.on_step(0, _clean_stats())

    def test_stop_before_start_raises(self):
        """stop before start must raise."""
        detector = ContagionDetector(proc_reader=_const_proc_reader(100_000, 50))
        with pytest.raises(Exception):
            detector.stop()

    def test_multiple_steps_accumulate_correctly(self):
        """Multiple server-reported events across steps accumulate in total_events."""
        reader = _const_proc_reader(vmrss_kb=100_000, fd_size=50)
        detector = ContagionDetector(proc_reader=reader)
        detector.start(_FAKE_PID)

        # 3 separate steps each with contagion_events=1
        for step in range(9):
            contagion = 1 if step in (1, 4, 7) else 0
            detector.on_step(step, _clean_stats(contagion_events=contagion))

        report = detector.stop()
        # Must have at least 3 events (one per contagion_events=1 step)
        assert report.total_events >= 3, (
            f"3 separate contagion steps must produce >= 3 events, "
            f"got {report.total_events}"
        )
