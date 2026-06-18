"""Tests for make_trl_run_one (FIX 3).

Verifies that make_trl_run_one builds the correct argv for in_loop (scope-
wrapped) and rlox (unwrapped) conditions using a fake subprocess runner.

No real GPU is exercised here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

# benchmarks/agentic/ is on sys.path via conftest.py
from run_benchmark import make_trl_run_one


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_PYTHON = "/fake/venv/bin/python"
_FAKE_REPO_ROOT = "/fake/repo"
_FAKE_CORPUS = "/fake/repo/benchmarks/agentic/corpus/adversarial_corpus_v1.json"
_FAKE_SERVER_URL = "http://localhost:8231"


def _make_factory(
    output_root: str | Path,
    *,
    scope_for_baseline: bool = True,
) -> Callable[[str, int, float], dict]:
    """Create a run_one callable backed by the given tmp output root."""
    return make_trl_run_one(
        max_steps=5,
        group_size=4,
        rlox_server_url=_FAKE_SERVER_URL,
        corpus_path=_FAKE_CORPUS,
        output_root=str(output_root),
        venv_python=_FAKE_PYTHON,
        repo_root=_FAKE_REPO_ROOT,
        scope_for_baseline=scope_for_baseline,
    )


def _make_summary(survived: bool = True, steps: int = 5, reward: float = 0.5) -> dict:
    return {
        "survived": survived,
        "completed_steps": steps,
        "mean_reward_last": reward,
        "elapsed_secs": 10.0,
        "backend": "rlox",
        "adversarial_fraction": 0.0,
        "seed": 0,
        "max_steps": 5,
        "group_size": 4,
    }


class _FakeProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def _fake_run_writing_summary(
    output_root: Path,
    condition: str,
    seed: int,
    fraction: float,
    *,
    returncode: int = 0,
    write_summary: bool = True,
    summary: dict | None = None,
) -> Callable:
    """Return a fake subprocess.run callable that optionally writes summary.json."""

    def _run(cmd, **kwargs):
        if write_summary:
            run_label = f"{condition}_seed{seed}_frac{fraction}"
            out_dir = output_root / run_label
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = summary if summary is not None else _make_summary()
            (out_dir / "summary.json").write_text(json.dumps(payload))
        return _FakeProcess(returncode=returncode)

    return _run


# ---------------------------------------------------------------------------
# A) argv for rlox condition (Treatment) — NOT scope-wrapped
# ---------------------------------------------------------------------------

class TestRloxArgv:
    def test_rlox_command_is_not_scope_wrapped(self, tmp_path: Path):
        """The rlox condition must launch trl_grpo_run.py directly, no systemd-run."""
        run_one = _make_factory(tmp_path)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "rlox", 0, 0.0)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("rlox", 0, 0.0)

        assert len(captured_cmd) == 1
        cmd = captured_cmd[0]
        assert cmd[0] == _FAKE_PYTHON
        assert "systemd-run" not in cmd
        assert "timeout" not in cmd

    def test_rlox_command_contains_backend_rlox(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)

        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "rlox", 0, 0.0)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("rlox", 0, 0.0)

        cmd = captured_cmd[0]
        assert "--backend" in cmd
        assert cmd[cmd.index("--backend") + 1] == "rlox"

    def test_rlox_command_includes_all_required_flags(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "rlox", 2, 0.1)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("rlox", 2, 0.1)

        cmd = captured_cmd[0]
        for flag in (
            "--adversarial-fraction",
            "--seed",
            "--max-steps",
            "--group-size",
            "--adversarial-corpus",
            "--rlox-server-url",
            "--output-dir",
        ):
            assert flag in cmd, f"Missing flag {flag!r} in argv: {cmd}"

    def test_rlox_command_correct_flag_values(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "rlox", 1, 0.05)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("rlox", 1, 0.05)

        cmd = captured_cmd[0]

        def _val(flag: str) -> str:
            return cmd[cmd.index(flag) + 1]

        assert _val("--seed") == "1"
        assert _val("--adversarial-fraction") == "0.05"
        assert _val("--max-steps") == "5"
        assert _val("--group-size") == "4"
        assert _val("--adversarial-corpus") == _FAKE_CORPUS
        assert _val("--rlox-server-url") == _FAKE_SERVER_URL


# ---------------------------------------------------------------------------
# B) argv for in_loop condition (Baseline) — scope-wrapped
# ---------------------------------------------------------------------------

class TestInLoopArgv:
    def test_in_loop_command_is_scope_wrapped(self, tmp_path: Path):
        """The in_loop condition must be wrapped with systemd-run --user --scope."""
        run_one = _make_factory(tmp_path, scope_for_baseline=True)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "in_loop", 0, 0.0)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("in_loop", 0, 0.0)

        cmd = captured_cmd[0]
        assert cmd[0] == "systemd-run", f"Expected systemd-run first, got {cmd[0]!r}"
        assert "--user" in cmd
        assert "--scope" in cmd

    def test_in_loop_command_has_timeout(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "in_loop", 0, 0.0)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("in_loop", 0, 0.0)

        assert "timeout" in captured_cmd[0]

    def test_in_loop_command_has_memory_limit(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "in_loop", 0, 0.0)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("in_loop", 0, 0.0)

        assert "MemoryMax" in " ".join(captured_cmd[0])

    def test_in_loop_command_still_passes_backend_flag(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "in_loop", 0, 0.0)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("in_loop", 0, 0.0)

        cmd = captured_cmd[0]
        assert "--backend" in cmd
        assert cmd[cmd.index("--backend") + 1] == "in_loop"

    def test_scope_for_baseline_false_skips_wrap(self, tmp_path: Path):
        """When scope_for_baseline=False, in_loop must NOT use systemd-run."""
        run_one = _make_factory(tmp_path, scope_for_baseline=False)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            return _fake_run_writing_summary(tmp_path, "in_loop", 0, 0.0)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("in_loop", 0, 0.0)

        assert captured_cmd[0][0] != "systemd-run"


# ---------------------------------------------------------------------------
# C) CUDA_VISIBLE_DEVICES is set to "0"
# ---------------------------------------------------------------------------

class TestCudaEnv:
    def test_cuda_visible_devices_set_to_0(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)
        captured_env: list[dict] = []

        def fake_run(cmd, env=None, **kwargs):
            if env is not None:
                captured_env.append(dict(env))
            return _fake_run_writing_summary(tmp_path, "rlox", 0, 0.0)(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("rlox", 0, 0.0)

        assert len(captured_env) == 1
        assert captured_env[0].get("CUDA_VISIBLE_DEVICES") == "0"


# ---------------------------------------------------------------------------
# D) Result dict structure
# ---------------------------------------------------------------------------

class TestResultDict:
    def test_survived_true_when_summary_has_survived_true(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)

        def fake_run(cmd, **kwargs):
            return _fake_run_writing_summary(
                tmp_path, "rlox", 0, 0.0,
                summary=_make_summary(survived=True, steps=5, reward=0.8),
            )(cmd, **kwargs)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is True
        assert result["completed_steps"] == 5
        assert result["mean_reward_last"] == pytest.approx(0.8)
        assert result["condition"] == "rlox"
        assert result["seed"] == 0
        assert result["fraction"] == 0.0

    def test_survived_false_on_nonzero_exit(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)

        def fake_run(cmd, **kwargs):
            return _FakeProcess(returncode=1)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is False

    def test_survived_false_when_summary_missing(self, tmp_path: Path):
        """returncode=0 but no summary.json written must yield survived=False."""
        run_one = _make_factory(tmp_path)

        def fake_run(cmd, **kwargs):
            # Deliberately do NOT write summary.json
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is False

    def test_survived_false_on_subprocess_exception(self, tmp_path: Path):
        run_one = _make_factory(tmp_path)

        def fake_run(cmd, **kwargs):
            raise OSError("process failed to start")

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is False

    def test_output_dir_encodes_condition_seed_fraction(self, tmp_path: Path):
        """The --output-dir flag value must identify the run triple."""
        run_one = _make_factory(tmp_path)
        captured_cmd: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.append(list(cmd))
            # Extract output dir from cmd and write summary there.
            idx = cmd.index("--output-dir")
            out_dir = Path(cmd[idx + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary.json").write_text(json.dumps(_make_summary()))
            return _FakeProcess()

        with patch("run_benchmark.subprocess.run", side_effect=fake_run):
            run_one("rlox", 3, 0.1)

        cmd = captured_cmd[0]
        output_dir = cmd[cmd.index("--output-dir") + 1]
        assert "rlox" in output_dir
        assert "seed3" in output_dir
        assert "0.1" in output_dir
