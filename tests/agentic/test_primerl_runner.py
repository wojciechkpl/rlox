"""RED-phase tests for Step 8: make_primerl_run_one (prime-rl GRPO launcher).

Contract being specified
------------------------
make_primerl_run_one(
    *,
    base_toml: str | Path,
    max_steps: int,
    group_size: int,
    rlox_server_url: str,
    prime_rl_bin: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    scope_for_baseline: bool = True,
    in_loop_timeout_secs: int = 7200,
) -> Callable[[str, int, float], dict]

The returned callable ``run_one(condition, seed, fraction) -> dict`` must return
EXACTLY the same seven-key shape as ``make_trl_run_one``:
    {condition, seed, fraction, survived(bool), completed_steps(int),
     elapsed_secs(float), mean_reward_last(float)}

Survival semantics differ from the TRL runner: prime-rl does not write a
``summary.json``.  Instead the launcher counts ``rollouts/step_*/`` subdirs
written by prime-rl and reads the last ``train_rollouts.jsonl``.

Interface assumptions (implementer MUST honour):
  - Symbol: ``make_primerl_run_one`` in ``benchmarks/agentic/run_benchmark.py``.
  - The per-run TOML is rendered to ``<run_dir>/rl.gen.toml`` via deep-merge.
  - ``rollout_backend`` and ``adversarial_fraction`` override
    ``orchestrator.train.env[0].args.*``.
  - Survival = returncode 0 AND completed_steps >= max_steps.
  - mean_reward_last = mean of ``reward`` field in the last step's JSONL;
    returns 0.0 (no raise) when the rollouts dir is absent.
  - For ``in_loop`` + ``scope_for_baseline=True``: command is
    ``systemd-run --user --scope -p TasksMax=... -p MemoryMax=...
      ... timeout <in_loop_timeout_secs> <prime_rl_bin> @ <run>.toml``.
  - For ``rlox``: command is ``<prime_rl_bin> @ <run>.toml`` directly.
  - ``CUDA_VISIBLE_DEVICES`` is set to ``"0"`` in the subprocess env.
  - All subprocess / timeout failures return ``survived=False`` without raising.

All tests run on laptop / CI — NO GPU required.  A fake ``rl`` binary
(written into tmp_path) simulates prime-rl's file-writing behaviour.
``subprocess.run`` is monkeypatched where a fake binary is inconvenient.

Run with:
    ./.venv/bin/python -m pytest tests/agentic/test_primerl_runner.py -v
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
import tomllib
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

# benchmarks/agentic/ is on sys.path via conftest.py.
# RED phase: make_primerl_run_one does not exist in run_benchmark yet.
# We import the module (which exists) and reach for the symbol at call-time so
# that pytest can COLLECT all tests; each test fails at the _make_factory call
# with AttributeError rather than blocking the whole module at import time.
import run_benchmark as _run_benchmark_mod  # the module exists; the symbol does not yet

def make_primerl_run_one(*args, **kwargs):  # noqa: E302
    """Shim: delegates to the real symbol when it exists; raises AttributeError otherwise."""
    fn = getattr(_run_benchmark_mod, "make_primerl_run_one", None)
    if fn is None:
        raise AttributeError(
            "make_primerl_run_one does not exist in run_benchmark yet (RED phase)"
        )
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_FAKE_SERVER_URL = "http://localhost:8231"
_FAKE_BIN_NAME = "rl"

# A minimal base TOML that mirrors the structure of benchmarks/agentic/primerl/smoke.toml.
# Written into fixtures so tests never depend on the on-disk smoke.toml path.
_BASE_TOML_CONTENT = """\
max_steps = 20
seq_len = 1024

[ckpt]

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[wandb]
project = "rlox-agentic-smoke"
name = "smoke"

[trainer.model]
impl = "auto"

[trainer.model.ac]
freq = 1

[trainer.model.lora]
rank = 32
alpha = 64

[trainer.optim]
lr = 1e-5

[orchestrator]
batch_size = 8
group_size = 4

[orchestrator.train.sampling]
max_completion_tokens = 512

[[orchestrator.train.env]]
id = "rlox-verify"
args = { rollout_backend = "in_loop", adversarial_fraction = 0.0, per_sample_timeout_secs = 5.0, group_size = 4, n_problems = 16 }

[inference]
enable_lora = true
max_lora_rank = 32
gpu_memory_utilization = 0.40
vllm_extra = { max_model_len = 1024 }

[orchestrator.renderer]
name = "auto"
"""

# Expected result dict keys — must be identical to make_trl_run_one output.
_REQUIRED_KEYS = {
    "condition",
    "seed",
    "fraction",
    "survived",
    "completed_steps",
    "elapsed_secs",
    "mean_reward_last",
}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_toml(tmp_path: Path) -> Path:
    """Write the minimal base TOML to tmp_path and return its path."""
    p = tmp_path / "smoke.toml"
    p.write_text(_BASE_TOML_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    """A fake repo root (just needs to exist)."""
    return tmp_path


@pytest.fixture()
def prime_rl_bin(tmp_path: Path) -> Path:
    """Path where tests can write a fake ``rl`` binary."""
    return tmp_path / _FAKE_BIN_NAME


def _write_fake_rl(prime_rl_bin: Path, body: str) -> None:
    """Write an executable Python-based fake ``rl`` script at the given path."""
    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        {body}
    """)
    prime_rl_bin.write_text(script, encoding="utf-8")
    prime_rl_bin.chmod(prime_rl_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def _make_factory(
    base_toml: Path,
    output_root: Path,
    prime_rl_bin: Path,
    repo_root: Path,
    *,
    max_steps: int = 3,
    group_size: int = 4,
    scope_for_baseline: bool = True,
    in_loop_timeout_secs: int = 60,
) -> Callable[[str, int, float], dict]:
    """Build a run_one callable for tests."""
    return make_primerl_run_one(
        base_toml=str(base_toml),
        max_steps=max_steps,
        group_size=group_size,
        rlox_server_url=_FAKE_SERVER_URL,
        prime_rl_bin=str(prime_rl_bin),
        output_root=str(output_root),
        repo_root=str(repo_root),
        scope_for_baseline=scope_for_baseline,
        in_loop_timeout_secs=in_loop_timeout_secs,
    )


def _write_rollouts(run_dir: Path, n_steps: int, rewards_per_step: list[list[float]] | None = None) -> None:
    """Fabricate ``rollouts/step_{k}/train_rollouts.jsonl`` inside run_dir."""
    rollouts_dir = run_dir / "rollouts"
    for k in range(n_steps):
        step_dir = rollouts_dir / f"step_{k}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if rewards_per_step is not None and k < len(rewards_per_step):
            rows = [{"reward": r, "prompt": "x", "response": "y"} for r in rewards_per_step[k]]
        else:
            rows = [{"reward": 1.0, "prompt": "x", "response": "y"}]
        jsonl = "\n".join(json.dumps(row) for row in rows)
        (step_dir / "train_rollouts.jsonl").write_text(jsonl, encoding="utf-8")


def _make_step_line(step: int, reward: float, *, ansi: bool = False) -> str:
    """Return a single prime-rl progress line for the given step and reward.

    When ``ansi=True`` the line is wrapped in ANSI colour escape sequences
    exactly as the real ``rl`` launcher emits them, to exercise ANSI stripping.
    """
    plain = f"Step {step} |   11.6s | Reward {reward:.4f} | Trainable 4/8 (50.0%) | Turns 1.0"
    if ansi:
        # Wrap with a common ANSI green prefix and reset suffix.
        return f"\x1b[32m{plain}\x1b[0m"
    return plain


def _write_rl_log(run_dir: Path, step_rewards: list[float], *, ansi: bool = False) -> Path:
    """Write ``<run_dir>/rl.log`` with one Step line per entry in step_rewards.

    step_rewards[k] is the reward for step k (0-indexed).
    Returns the path to the written log file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [_make_step_line(k, r, ansi=ansi) for k, r in enumerate(step_rewards)]
    log_path = run_dir / "rl.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


# ---------------------------------------------------------------------------
# A) Return shape parity with make_trl_run_one
# ---------------------------------------------------------------------------

class TestReturnShapeParity:
    """The returned dict must carry exactly the same keys as make_trl_run_one."""

    def test_make_primerl_run_one_returns_callable(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root)
        assert callable(run_one), "make_primerl_run_one must return a callable"

    def test_result_has_all_required_keys(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Happy path: result dict must have EXACTLY the seven required keys."""
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            # Determine the run dir from the cmd (the @ arg is the TOML path)
            # The run dir is the parent of the rl.gen.toml file
            toml_path = _extract_toml_path_from_cmd(cmd)
            if toml_path:
                run_dir = Path(toml_path).parent
                _write_rollouts(run_dir, max_steps)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert set(result.keys()) == _REQUIRED_KEYS, (
            f"Result keys {set(result.keys())} != required {_REQUIRED_KEYS}"
        )

    def test_result_types_match_contract(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """survived must be bool, completed_steps int, elapsed_secs float,
        mean_reward_last float."""
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            toml_path = _extract_toml_path_from_cmd(cmd)
            if toml_path:
                run_dir = Path(toml_path).parent
                _write_rollouts(run_dir, max_steps)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert isinstance(result["survived"], bool)
        assert isinstance(result["completed_steps"], int)
        assert isinstance(result["elapsed_secs"], float)
        assert isinstance(result["mean_reward_last"], float)

    def test_result_echoes_condition_seed_fraction(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        max_steps = 1
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            toml_path = _extract_toml_path_from_cmd(cmd)
            if toml_path:
                _write_rollouts(Path(toml_path).parent, max_steps)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 7, 0.15)

        assert result["condition"] == "rlox"
        assert result["seed"] == 7
        assert result["fraction"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# B) Per-run TOML templating
# ---------------------------------------------------------------------------

class TestTomlTemplating:
    """The launcher must render rl.gen.toml by deep-merging base + overrides."""

    def _capture_toml_path(self, cmd: list[str]) -> str | None:
        return _extract_toml_path_from_cmd(cmd)

    def test_gen_toml_is_written_to_run_dir(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """rl.gen.toml must exist inside the per-run directory before exec."""
        captured_toml_paths: list[str] = []
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                # Write rollout so survived=True path is exercised (not essential here)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        assert captured_toml_paths, "subprocess.run was never called with a @ toml arg"
        toml_path = Path(captured_toml_paths[0])
        assert toml_path.exists(), f"rl.gen.toml was not written at {toml_path}"
        assert toml_path.name == "rl.gen.toml"

    def test_gen_toml_sets_rollout_backend(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Rendered TOML must set orchestrator.train.env[0].args.rollout_backend == condition."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        backend = toml_data["orchestrator"]["train"]["env"][0]["args"]["rollout_backend"]
        assert backend == "rlox"

    def test_gen_toml_sets_rollout_backend_in_loop(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=False,
        )
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        backend = toml_data["orchestrator"]["train"]["env"][0]["args"]["rollout_backend"]
        assert backend == "in_loop"

    def test_gen_toml_sets_adversarial_fraction(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.10)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        frac = toml_data["orchestrator"]["train"]["env"][0]["args"]["adversarial_fraction"]
        assert frac == pytest.approx(0.10)

    def test_gen_toml_sets_max_steps(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Rendered TOML must override max_steps to the value given to make_primerl_run_one."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=7)
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 7)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        assert toml_data["max_steps"] == 7

    def test_gen_toml_sets_wandb_name(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Rendered TOML must set a per-run wandb.name (must differ from base 'smoke')."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 3, 0.05)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        wandb_name = toml_data["wandb"]["name"]
        # Must be a non-empty string that identifies the run
        assert isinstance(wandb_name, str) and wandb_name, "wandb.name must be set"
        # Must not be the generic base name "smoke"
        assert wandb_name != "smoke", (
            "wandb.name must be per-run (encode condition/seed/fraction), not 'smoke'"
        )

    def test_gen_toml_preserves_model_name(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """model.name from the base TOML must survive the merge (not dropped)."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        assert toml_data["model"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"

    def test_gen_toml_preserves_lora_rank(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """trainer.model.lora.rank must be preserved from the base TOML."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        assert toml_data["trainer"]["model"]["lora"]["rank"] == 32

    def test_gen_toml_preserves_gpu_memory_utilization(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """inference.gpu_memory_utilization must be preserved from base TOML."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        assert toml_data["inference"]["gpu_memory_utilization"] == pytest.approx(0.40)

    @pytest.mark.parametrize("condition,seed,fraction", [
        ("rlox", 0, 0.0),
        ("rlox", 2, 0.05),
        ("in_loop", 1, 0.10),
    ])
    def test_gen_toml_output_dir_key_is_run_dir(
        self,
        condition: str, seed: int, fraction: float,
        base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path,
    ):
        """If output_dir is set in the rendered TOML, it must equal the run dir."""
        # Assumption: the implementer MAY or MAY NOT write output_dir into the TOML
        # (the plan says to set it; this test enforces it IS set and IS correct).
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=False,
        )
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one(condition, seed, fraction)

        assert captured_toml_paths
        toml_data = _read_gen_toml(captured_toml_paths[0])
        run_dir = str(Path(captured_toml_paths[0]).parent)
        # The rendered TOML must have an output_dir entry matching the run dir.
        assert "output_dir" in toml_data, "output_dir must be set in rl.gen.toml"
        assert toml_data["output_dir"] == run_dir


# ---------------------------------------------------------------------------
# C) Command assembly
# ---------------------------------------------------------------------------

class TestCommandAssembly:
    """Verify the subprocess argv for each condition."""

    def test_rlox_condition_executes_bin_directly(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """rlox condition: command must start with prime_rl_bin, no systemd-run."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        assert captured_cmds, "subprocess.run was never called"
        cmd = captured_cmds[0]
        assert cmd[0] == str(prime_rl_bin), (
            f"rlox condition must call prime_rl_bin directly; got {cmd[0]!r}"
        )
        assert "systemd-run" not in cmd
        assert "timeout" not in cmd

    def test_rlox_condition_uses_at_sign_syntax(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """The ``@`` flag must appear immediately after the prime_rl_bin."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        cmd = captured_cmds[0]
        assert "@" in cmd, f"Command must include '@' for TOML file syntax; got {cmd}"
        at_idx = cmd.index("@")
        # @ must be right after prime_rl_bin
        assert at_idx == 1, f"'@' must be at index 1 (after bin); got index {at_idx}"

    def test_rlox_command_toml_path_ends_with_gen_toml(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """The TOML path in the command must point to rl.gen.toml."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        toml_path = _extract_toml_path_from_cmd(captured_cmds[0])
        assert toml_path is not None
        assert Path(toml_path).name == "rl.gen.toml"

    def test_in_loop_with_scope_uses_systemd_run(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """in_loop + scope_for_baseline=True must wrap with systemd-run --user --scope."""
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=True,
        )
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            # systemd-run wraps the command, so the run dir is still inside tmp_path.
            # Locate the toml from the full cmd.
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        assert captured_cmds
        cmd = captured_cmds[0]
        assert cmd[0] == "systemd-run", f"Expected systemd-run first, got {cmd[0]!r}"
        assert "--user" in cmd
        assert "--scope" in cmd

    def test_in_loop_with_scope_includes_timeout(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """The systemd-run wrap must include the ``timeout`` command."""
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=True, in_loop_timeout_secs=999,
        )
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        cmd = captured_cmds[0]
        assert "timeout" in cmd, f"Expected 'timeout' in cmd; got {cmd}"

    def test_in_loop_timeout_value_matches_parameter(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """The timeout value in the command must equal in_loop_timeout_secs."""
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=True, in_loop_timeout_secs=123,
        )
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        cmd = captured_cmds[0]
        timeout_idx = cmd.index("timeout")
        assert cmd[timeout_idx + 1] == "123", (
            f"timeout value must be '123'; got {cmd[timeout_idx + 1]!r}"
        )

    def test_in_loop_with_scope_has_tasks_max(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """systemd-run command must include -p TasksMax=..."""
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=True,
        )
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        cmd_str = " ".join(captured_cmds[0])
        assert "TasksMax" in cmd_str, f"Expected TasksMax in command; got {cmd_str}"

    def test_in_loop_with_scope_has_memory_max(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """systemd-run command must include -p MemoryMax=..."""
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=True,
        )
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        cmd_str = " ".join(captured_cmds[0])
        assert "MemoryMax" in cmd_str, f"Expected MemoryMax in command; got {cmd_str}"

    def test_in_loop_with_scope_false_no_systemd(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """scope_for_baseline=False must skip systemd-run even for in_loop."""
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=False,
        )
        captured_cmds: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        cmd = captured_cmds[0]
        assert cmd[0] != "systemd-run", "scope_for_baseline=False must not use systemd-run"
        assert "timeout" not in cmd


# ---------------------------------------------------------------------------
# D) CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------------------------

class TestCudaEnv:
    def test_cuda_visible_devices_set_to_0_for_rlox(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_envs: list[dict] = []

        def fake_subprocess_run(cmd, env=None, **kwargs):
            if env is not None:
                captured_envs.append(dict(env))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        assert captured_envs, "No env was captured — subprocess.run env kwarg not passed"
        assert captured_envs[0].get("CUDA_VISIBLE_DEVICES") == "0"

    def test_cuda_visible_devices_set_to_0_for_in_loop(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=1, scope_for_baseline=False,
        )
        captured_envs: list[dict] = []

        def fake_subprocess_run(cmd, env=None, **kwargs):
            if env is not None:
                captured_envs.append(dict(env))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        assert captured_envs
        assert captured_envs[0].get("CUDA_VISIBLE_DEVICES") == "0"


# ---------------------------------------------------------------------------
# E) Survival = success (fake rl writes all step dirs, exits 0)
# ---------------------------------------------------------------------------

class TestSurvivalSuccess:
    """survived=True when rc==0 and all step dirs are written."""

    def test_survived_true_when_all_steps_written_and_rc0(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        max_steps = 3
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, max_steps)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is True
        assert result["completed_steps"] == max_steps

    def test_completed_steps_equals_max_steps_on_success(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        max_steps = 5
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, max_steps)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["completed_steps"] == max_steps

    def test_elapsed_secs_is_non_negative(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        max_steps = 1
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, max_steps)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["elapsed_secs"] >= 0.0


# ---------------------------------------------------------------------------
# F) DNF (crash) — fake rl writes k < max_steps dirs then exits non-zero
# ---------------------------------------------------------------------------

class TestDnfCrash:
    """survived=False + completed_steps == k when only k < max_steps steps are logged."""

    @pytest.mark.parametrize("k,max_steps", [(0, 3), (1, 3), (2, 5)])
    def test_survived_false_on_partial_steps(
        self,
        k: int, max_steps: int,
        base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path,
    ):
        """k log lines (k < max_steps) => survived=False, completed_steps == k.

        Uses the log as the primary step source (new contract); rc is non-zero
        (crash) but survival is determined by step count, not rc.
        """
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                run_dir = Path(p).parent
                # Write k log lines (fewer than max_steps) — no rollout dirs
                if k > 0:
                    _write_rl_log(run_dir, [0.5] * k)
            return _FakeProcess(returncode=1)  # crashed

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is False
        assert result["completed_steps"] == k

    def test_survived_true_when_all_steps_logged_but_rc_143(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """A clean prime-rl finish: logs all max_steps Step lines, exits 143 (SIGTERM).

        The OLD contract was: rc!=0 => survived=False.
        The NEW contract is: survived = (completed_steps >= max_steps), rc-independent.
        Exit 143 is the normal completion signal from prime-rl (SIGTERM of children).
        """
        max_steps = 3
        step_rewards = [0.5, 0.6, 0.75]  # three steps, last reward = 0.75
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                run_dir = Path(p).parent
                _write_rl_log(run_dir, step_rewards)
            return _FakeProcess(returncode=143)  # SIGTERM — the normal prime-rl finish

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is True, (
            "survived must be True when all steps are logged, even with rc=143 (SIGTERM)"
        )
        assert result["completed_steps"] == max_steps
        assert result["mean_reward_last"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# F2) Log capture and log-based parsing (new contract from live validation)
#
# The launcher now redirects rl stdout+stderr to <run_dir>/rl.log.
# completed_steps = max(log_step_count, rollout_dir_count).
# mean_reward_last = Reward value on the highest Step N line (ANSI-stripped);
# fallback to jsonl parse when log has no Step lines.
# ---------------------------------------------------------------------------

class TestLogCapture:
    """Launcher must capture rl output to rl.log and parse step count / reward from it."""

    def test_rl_log_is_written_to_run_dir(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """subprocess.run must be called with stdout/stderr redirected so rl.log is produced."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=1)
        captured_kwargs: list[dict] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_kwargs.append(dict(kwargs))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                # Simulate what the launcher should do: write rl.log
                _write_rl_log(Path(p).parent, [0.5])
            return _FakeProcess(returncode=143)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        # The launcher must NOT use capture_output=False (which discards output).
        # It must redirect to a file so rl.log exists.
        assert captured_kwargs, "subprocess.run was never called"
        kw = captured_kwargs[0]
        assert kw.get("capture_output") is not False or "stdout" in kw or "stderr" in kw, (
            "launcher must capture stdout/stderr (not discard with capture_output=False)"
        )

    def test_log_based_step_count_plain(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """completed_steps must be parsed from Step N lines in rl.log (plain, no ANSI)."""
        max_steps = 4
        step_rewards = [0.5, 0.6, 0.7, 0.8]
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rl_log(Path(p).parent, step_rewards, ansi=False)
            return _FakeProcess(returncode=143)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["completed_steps"] == max_steps
        assert result["survived"] is True

    def test_log_based_step_count_ansi_wrapped(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Step lines wrapped in ANSI colour codes must still be counted correctly."""
        max_steps = 3
        step_rewards = [0.4, 0.6, 0.9]
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rl_log(Path(p).parent, step_rewards, ansi=True)
            return _FakeProcess(returncode=143)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["completed_steps"] == max_steps, (
            "ANSI-wrapped Step lines must be stripped and counted; "
            f"got completed_steps={result['completed_steps']}, expected {max_steps}"
        )
        assert result["survived"] is True

    def test_mean_reward_last_from_log_plain(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """mean_reward_last must equal the Reward value on the highest-N Step line (plain)."""
        max_steps = 3
        step_rewards = [0.5, 0.6, 0.75]  # last step reward = 0.75
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rl_log(Path(p).parent, step_rewards, ansi=False)
            return _FakeProcess(returncode=143)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["mean_reward_last"] == pytest.approx(0.75), (
            f"mean_reward_last must be 0.75 (last Step line reward); "
            f"got {result['mean_reward_last']}"
        )

    def test_mean_reward_last_from_log_ansi_wrapped(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """mean_reward_last must be parsed from ANSI-wrapped Step lines."""
        max_steps = 2
        step_rewards = [0.3, 0.9]  # last = 0.9
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rl_log(Path(p).parent, step_rewards, ansi=True)
            return _FakeProcess(returncode=143)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["mean_reward_last"] == pytest.approx(0.9)

    def test_log_step_count_takes_priority_over_rollout_dirs_when_higher(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """When log has more steps than rollout dirs, log count wins (max rule).

        Real scenario: prime-rl only persists rollouts/step_0 due to gating,
        but the log has all N Step lines.
        """
        max_steps = 4
        step_rewards = [0.5, 0.6, 0.7, 0.8]  # 4 log lines
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                run_dir = Path(p).parent
                # Only ONE rollout dir (gated saving), but FOUR log lines
                _write_rollouts(run_dir, 1)
                _write_rl_log(run_dir, step_rewards)
            return _FakeProcess(returncode=143)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["completed_steps"] == max_steps, (
            f"Log count (4) must win over rollout dir count (1); "
            f"got completed_steps={result['completed_steps']}"
        )
        assert result["survived"] is True

    def test_rollout_dir_count_used_as_fallback_when_log_has_no_step_lines(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """When rl.log exists but has no Step N lines, rollout dir count is the fallback."""
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                run_dir = Path(p).parent
                # Log with no Step lines (e.g. only startup noise)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "rl.log").write_text(
                    "Starting server...\nConfig loaded.\n", encoding="utf-8"
                )
                # Rollout dirs provide the fallback count
                _write_rollouts(run_dir, max_steps)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["completed_steps"] == max_steps

    def test_mean_reward_falls_back_to_jsonl_when_log_has_no_step_lines(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """If log has no Step lines, mean_reward_last must come from train_rollouts.jsonl."""
        max_steps = 1
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                run_dir = Path(p).parent
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "rl.log").write_text("No step lines here.\n", encoding="utf-8")
                _write_rollouts(run_dir, max_steps, [[0.42]])
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["mean_reward_last"] == pytest.approx(0.42), (
            f"mean_reward_last should fall back to jsonl (0.42); got {result['mean_reward_last']}"
        )

    def test_survival_rc_independent_all_steps_logged_rc_nonzero(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """survived depends only on completed_steps >= max_steps, not on rc.

        Any nonzero rc (1, 2, 143, 255…) with all steps logged → survived=True.
        """
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        for rc in (1, 2, 143, 255):
            def fake_subprocess_run(cmd, _rc=rc, **kwargs):
                p = _extract_toml_path_from_cmd(cmd)
                if p:
                    _write_rl_log(Path(p).parent, [0.5] * max_steps)
                return _FakeProcess(returncode=_rc)

            with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
                result = run_one("rlox", 0, 0.0)

            assert result["survived"] is True, (
                f"survived must be True with {max_steps} steps logged and rc={rc}"
            )

    def test_survival_false_when_fewer_log_lines_than_max_steps(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """If fewer Step lines than max_steps, survived=False regardless of rc."""
        max_steps = 5
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                # Only 2 steps logged (crash mid-run)
                _write_rl_log(Path(p).parent, [0.5, 0.6])
            return _FakeProcess(returncode=1)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is False
        assert result["completed_steps"] == 2

    def test_log_step_indices_are_parsed_not_counted_sequentially(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """The parser must find the highest step index N from 'Step N |' lines.

        This distinguishes a parser that counts lines from one that reads the N.
        A log with steps [0, 1, 3] (step 2 missing) has highest N=3 → 4 steps seen
        is ambiguous; the contract says count of distinct Step lines, so 3 here.
        For a clean run 0..N-1 the count equals N (the normal case we exercise here).
        """
        max_steps = 3
        step_rewards = [0.3, 0.5, 0.8]  # steps 0, 1, 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rl_log(Path(p).parent, step_rewards)
            return _FakeProcess(returncode=143)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["completed_steps"] == 3
        assert result["mean_reward_last"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# F3) Log-parsing resilience — malformed reward tokens must not propagate
#
# Regression for: _parse_rl_log calls float(m.group(2)) outside its try/except,
# so a malformed reward token (e.g. "Reward -0.3.3") raises ValueError which
# propagates through run_one (also outside the subprocess try/except) and
# mis-classifies a completed run as DNF.
#
# Fixed contract:
#   - run_one NEVER raises from log-parsing errors.
#   - A malformed reward token on the LAST Step line: step is still COUNTED
#     (completed_steps increments), but the reward value is SKIPPED; the
#     reported mean_reward_last falls back to the last VALID reward, or 0.0.
#   - A Step line with NO Reward field: same — counted but reward skipped.
# ---------------------------------------------------------------------------

class TestLogParsingResilience:
    """_parse_rl_log must never raise; malformed reward tokens skip the value only."""

    # Helper: write a log with explicitly constructed raw lines so we can inject
    # malformed reward tokens that normal _write_rl_log cannot produce.
    @staticmethod
    def _write_raw_log(run_dir: Path, lines: list[str]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "rl.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_malformed_reward_token_does_not_raise(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """run_one must not propagate ValueError when a reward token is malformed."""
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                self._write_raw_log(Path(p).parent, [
                    # step 0 — valid
                    "Step 0 |   11.6s | Reward 0.5000 | Trainable 4/8 (50.0%) | Turns 1.0",
                    # step 1 — malformed reward token ("-0.3.3" has two dots)
                    "Step 1 |   12.1s | Reward -0.3.3 | Trainable 4/8 (50.0%) | Turns 1.0",
                ])
            return _FakeProcess(returncode=143)

        # Must not raise ValueError (or anything else).
        try:
            with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
                result = run_one("rlox", 0, 0.0)
        except Exception as exc:
            pytest.fail(
                f"run_one raised {type(exc).__name__} on malformed reward token: {exc!r}"
            )

        assert isinstance(result, dict), "run_one must return a dict, not raise"

    def test_malformed_reward_step_is_still_counted(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """A Step line with a malformed reward token still increments completed_steps."""
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                self._write_raw_log(Path(p).parent, [
                    "Step 0 |   11.6s | Reward 0.5000 | Trainable 4/8 (50.0%) | Turns 1.0",
                    "Step 1 |   12.1s | Reward -0.3.3 | Trainable 4/8 (50.0%) | Turns 1.0",
                ])
            return _FakeProcess(returncode=143)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["completed_steps"] == max_steps, (
            f"Both steps must be counted even when the last reward is malformed; "
            f"got completed_steps={result['completed_steps']}"
        )
        assert result["survived"] is True, (
            "survived must be True: completed_steps == max_steps despite malformed reward"
        )

    def test_malformed_last_reward_falls_back_to_previous_valid_reward(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """When the last Step line's reward token cannot be parsed as a float,
        mean_reward_last falls back to the last VALID reward from earlier steps.

        The malformed reward token used here (``Reward NaN``) is not matched by
        the regex (``NaN`` has no leading digit/sign), so that step line is not
        counted — the test verifies that completed_steps reflects only matched
        Step lines and mean_reward_last is the last valid one.

        Assumption: the implementer counts only lines that fully match
        ``Step N | ... | Reward <parseable-float> | ...``; an unmatched line
        is not counted toward completed_steps.
        """
        # With 2 valid steps and 1 unparseable step, completed_steps == 2.
        # max_steps is also set to 2 so survived == True.
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                self._write_raw_log(Path(p).parent, [
                    # step 0 — valid, reward 0.4
                    "Step 0 |   11.6s | Reward 0.4000 | Trainable 4/8 (50.0%) | Turns 1.0",
                    # step 1 — valid, reward 0.6 — this IS the last matched line
                    "Step 1 |   12.1s | Reward 0.6000 | Trainable 4/8 (50.0%) | Turns 1.0",
                    # step 2 — "NaN" is not matched by the regex → step not counted,
                    # reward not updated; step 1 remains the last valid reward
                    "Step 2 |   13.0s | Reward NaN | Trainable 4/8 (50.0%) | Turns 1.0",
                ])
            return _FakeProcess(returncode=143)

        try:
            with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
                result = run_one("rlox", 0, 0.0)
        except Exception as exc:
            pytest.fail(f"run_one raised {type(exc).__name__} on NaN reward token: {exc!r}")

        # Only 2 lines matched (steps 0 and 1); step 2 "NaN" line is not matched.
        assert result["completed_steps"] == max_steps, (
            f"completed_steps must be {max_steps} (matched lines only); "
            f"got {result['completed_steps']}"
        )
        # Step 1 (0.6000) is the last valid matched reward.
        assert result["mean_reward_last"] == pytest.approx(0.6), (
            f"mean_reward_last must be last valid reward (0.6); "
            f"got {result['mean_reward_last']}"
        )
        assert result["survived"] is True

    def test_step_line_missing_reward_field_does_not_raise(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """A Step line with no Reward field at all must not raise and must not be counted
        for the reward value (but is still counted as a step if the Step N | pattern matches
        — implementation may or may not match; what matters is NO RAISE and valid result)."""
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                self._write_raw_log(Path(p).parent, [
                    # step 0 — valid, reward 0.7
                    "Step 0 |   11.6s | Reward 0.7000 | Trainable 4/8 (50.0%) | Turns 1.0",
                    # step 1 — no Reward field; the regex won't match it, so it does not
                    # contribute to count or reward; step 0 remains the only counted line
                    "Step 1 |   12.1s | Loss 0.003 | Trainable 4/8 (50.0%) | Turns 1.0",
                ])
            return _FakeProcess(returncode=143)

        try:
            with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
                result = run_one("rlox", 0, 0.0)
        except Exception as exc:
            pytest.fail(
                f"run_one raised {type(exc).__name__} on Step line without Reward field: {exc!r}"
            )

        assert isinstance(result, dict)
        # The valid step (step 0, reward 0.7) must be reflected
        assert result["mean_reward_last"] == pytest.approx(0.7), (
            f"mean_reward_last should be 0.7 (from the valid Step 0 line); "
            f"got {result['mean_reward_last']}"
        )

    def test_all_rewards_unparseable_returns_zero_not_raises(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """If every Step line's reward token is unrecognisable by the regex,
        mean_reward_last must be 0.0 — no raise.

        Uses ``Reward NaN`` on all lines (NaN has no leading digit, so the
        Reward group does not match and the whole Step line goes uncounted).
        completed_steps == 0, survived == False (0 < max_steps).
        """
        max_steps = 2
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                self._write_raw_log(Path(p).parent, [
                    "Step 0 |   11.6s | Reward NaN | Trainable 4/8 (50.0%) | Turns 1.0",
                    "Step 1 |   12.1s | Reward NaN | Trainable 4/8 (50.0%) | Turns 1.0",
                ])
            return _FakeProcess(returncode=143)

        try:
            with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
                result = run_one("rlox", 0, 0.0)
        except Exception as exc:
            pytest.fail(f"run_one raised {type(exc).__name__} on all-NaN log: {exc!r}")

        assert result["mean_reward_last"] == pytest.approx(0.0), (
            f"mean_reward_last must be 0.0 when no reward tokens are parseable; "
            f"got {result['mean_reward_last']}"
        )
        # No lines matched → completed_steps == 0 (rollout dirs also absent).
        assert result["completed_steps"] == 0
        assert result["survived"] is False


# ---------------------------------------------------------------------------
# G) Timeout / stall — subprocess.TimeoutExpired or equivalent
# ---------------------------------------------------------------------------

class TestTimeout:
    """survived=False when the subprocess times out."""

    def test_survived_false_on_timeout_expired(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=3, in_loop_timeout_secs=1,
        )

        def fake_subprocess_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, timeout=1)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("in_loop", 0, 0.0)

        assert result["survived"] is False

    def test_survived_false_on_timeout_returns_dict_not_raises(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """TimeoutExpired must be caught — run_one must NOT propagate the exception."""
        run_one = _make_factory(
            base_toml, tmp_path, prime_rl_bin, repo_root,
            max_steps=3, in_loop_timeout_secs=1,
        )

        def fake_subprocess_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, timeout=1)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            # Must not raise
            result = run_one("in_loop", 0, 0.0)

        assert isinstance(result, dict)
        assert result["survived"] is False


# ---------------------------------------------------------------------------
# H) mean_reward_last
# ---------------------------------------------------------------------------

class TestMeanRewardLast:
    """mean_reward_last = mean(reward) in the last step's train_rollouts.jsonl."""

    def test_mean_reward_last_exact_value(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Assert exact mean for a known fixture with 3 steps."""
        max_steps = 3
        rewards_per_step = [
            [0.5, 0.5],          # step_0 — irrelevant
            [0.3, 0.7],          # step_1 — irrelevant
            [1.0, 0.0, 0.5],    # step_2 — LAST; mean = 0.5
        ]
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, max_steps, rewards_per_step)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["mean_reward_last"] == pytest.approx(0.5)

    def test_mean_reward_last_single_row(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        max_steps = 2
        rewards_per_step = [
            [0.9],   # step_0
            [0.7],   # step_1 — LAST
        ]
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, max_steps, rewards_per_step)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["mean_reward_last"] == pytest.approx(0.7)

    def test_mean_reward_last_is_zero_when_no_rollouts_dir(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """If rollouts dir is absent (crash before any step), mean_reward_last == 0.0 — no raise."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=3)

        def fake_subprocess_run(cmd, **kwargs):
            # Do NOT write any rollout directories
            return _FakeProcess(returncode=1)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["mean_reward_last"] == pytest.approx(0.0)

    def test_mean_reward_last_uses_only_last_step(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Mean is computed from step_{N-1} only — earlier steps must not influence it."""
        max_steps = 3
        rewards_per_step = [
            [100.0, 200.0],   # step_0 — must be IGNORED
            [300.0, 400.0],   # step_1 — must be IGNORED
            [1.0, 3.0],       # step_2 — LAST; mean = 2.0
        ]
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, max_steps, rewards_per_step)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["mean_reward_last"] == pytest.approx(2.0)

    def test_mean_reward_zero_when_reward_field_absent(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """If jsonl rows lack the 'reward' key, mean_reward_last falls back to 0.0."""
        max_steps = 1
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                run_dir = Path(p).parent
                step_dir = run_dir / "rollouts" / "step_0"
                step_dir.mkdir(parents=True, exist_ok=True)
                # Rows without 'reward' key
                rows = [{"score": 0.9, "prompt": "x"}, {"score": 0.8}]
                (step_dir / "train_rollouts.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
                )
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["mean_reward_last"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# I) Exception resilience
# ---------------------------------------------------------------------------

class TestExceptionResilience:
    """Any subprocess exception must result in survived=False without propagating."""

    def test_os_error_returns_survived_false(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=3)

        def fake_subprocess_run(cmd, **kwargs):
            raise OSError("binary not found")

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is False

    def test_runtime_error_returns_survived_false(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=3)

        def fake_subprocess_run(cmd, **kwargs):
            raise RuntimeError("unexpected failure")

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("rlox", 0, 0.0)

        assert result["survived"] is False

    def test_exception_result_has_all_keys(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Even on exception the full seven-key dict must be returned."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=3)

        def fake_subprocess_run(cmd, **kwargs):
            raise OSError("fail")

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            result = run_one("in_loop", 1, 0.05)

        assert set(result.keys()) == _REQUIRED_KEYS
        assert result["condition"] == "in_loop"
        assert result["seed"] == 1
        assert result["fraction"] == pytest.approx(0.05)
        assert result["completed_steps"] == 0
        assert result["mean_reward_last"] == pytest.approx(0.0)

    def test_exception_does_not_propagate(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """run_one must never propagate — mirrors make_trl_run_one's contract."""
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=3)

        def fake_subprocess_run(cmd, **kwargs):
            raise ValueError("internal boom")

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            try:
                run_one("rlox", 0, 0.0)
            except Exception as exc:
                pytest.fail(f"run_one propagated an exception: {exc!r}")


# ---------------------------------------------------------------------------
# J) run_one drops cleanly into run_sweep (interface compatibility)
# ---------------------------------------------------------------------------

class TestRunSweepCompatibility:
    """make_primerl_run_one's returned callable must work as a drop-in for run_sweep."""

    def test_run_sweep_accepts_primerl_run_one(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """run_sweep must complete without error when given a primerl run_one."""
        from run_benchmark import run_sweep  # already on sys.path via conftest

        # Minimal config stub
        class _Cfg:
            n_seeds = 1
            adversarial_fractions = [0.0]

        max_steps = 1
        run_one = _make_factory(base_toml, tmp_path, prime_rl_bin, repo_root, max_steps=max_steps)
        metric_store = str(tmp_path / "metrics")

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, max_steps)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            results = run_sweep(_Cfg(), run_one, metric_store)

        assert isinstance(results, list)
        assert len(results) == 2  # 2 conditions × 1 seed × 1 fraction
        for r in results:
            assert "survived" in r


# ---------------------------------------------------------------------------
# K1) rlox_server_url is threaded into the rendered TOML env args
#
# Regression: _render_run_toml gained a `rlox_server_url` param that sets
# orchestrator.train.env[0].args.rlox_server_url in the generated rl.gen.toml.
# Existing rollout_backend / adversarial_fraction overrides must be unaffected.
# ---------------------------------------------------------------------------

class TestServerUrlInToml:
    """rlox_server_url must appear in the rendered rl.gen.toml env[0].args,
    unconditionally for both condition='rlox' and condition='in_loop'."""

    @pytest.mark.parametrize("condition", ["rlox", "in_loop"])
    def test_gen_toml_sets_rlox_server_url(
        self,
        condition: str,
        base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path,
    ):
        """rlox_server_url value passed to make_primerl_run_one must appear in rl.gen.toml."""
        expected_url = "http://localhost:9999"
        run_one = make_primerl_run_one(
            base_toml=str(base_toml),
            max_steps=1,
            group_size=4,
            rlox_server_url=expected_url,
            prime_rl_bin=str(prime_rl_bin),
            output_root=str(tmp_path),
            repo_root=str(repo_root),
            scope_for_baseline=False,
            in_loop_timeout_secs=60,
        )
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one(condition, 0, 0.0)

        assert captured_toml_paths, "subprocess.run was never called"
        toml_data = _read_gen_toml(captured_toml_paths[0])
        actual_url = toml_data["orchestrator"]["train"]["env"][0]["args"]["rlox_server_url"]
        assert actual_url == expected_url, (
            f"rlox_server_url in rl.gen.toml: {actual_url!r} != {expected_url!r}"
        )

    def test_gen_toml_server_url_does_not_clobber_rollout_backend(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Setting rlox_server_url must not overwrite rollout_backend in env[0].args."""
        run_one = make_primerl_run_one(
            base_toml=str(base_toml),
            max_steps=1,
            group_size=4,
            rlox_server_url="http://localhost:7777",
            prime_rl_bin=str(prime_rl_bin),
            output_root=str(tmp_path),
            repo_root=str(repo_root),
            scope_for_baseline=False,
            in_loop_timeout_secs=60,
        )
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        args = toml_data["orchestrator"]["train"]["env"][0]["args"]
        assert args["rollout_backend"] == "rlox"
        assert args["rlox_server_url"] == "http://localhost:7777"

    def test_gen_toml_server_url_does_not_clobber_adversarial_fraction(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """Setting rlox_server_url must not overwrite adversarial_fraction in env[0].args."""
        run_one = make_primerl_run_one(
            base_toml=str(base_toml),
            max_steps=1,
            group_size=4,
            rlox_server_url="http://localhost:7777",
            prime_rl_bin=str(prime_rl_bin),
            output_root=str(tmp_path),
            repo_root=str(repo_root),
            scope_for_baseline=False,
            in_loop_timeout_secs=60,
        )
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.15)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        args = toml_data["orchestrator"]["train"]["env"][0]["args"]
        assert args["adversarial_fraction"] == pytest.approx(0.15)
        assert args["rlox_server_url"] == "http://localhost:7777"

    def test_gen_toml_all_three_env_args_set_together(
        self, base_toml: Path, tmp_path: Path, prime_rl_bin: Path, repo_root: Path
    ):
        """All three overrides — rollout_backend, adversarial_fraction, rlox_server_url —
        must be present simultaneously in env[0].args."""
        server_url = "http://verify.local:8231"
        run_one = make_primerl_run_one(
            base_toml=str(base_toml),
            max_steps=1,
            group_size=4,
            rlox_server_url=server_url,
            prime_rl_bin=str(prime_rl_bin),
            output_root=str(tmp_path),
            repo_root=str(repo_root),
            scope_for_baseline=False,
            in_loop_timeout_secs=60,
        )
        captured_toml_paths: list[str] = []

        def fake_subprocess_run(cmd, **kwargs):
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                captured_toml_paths.append(p)
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 2, 0.10)

        toml_data = _read_gen_toml(captured_toml_paths[0])
        args = toml_data["orchestrator"]["train"]["env"][0]["args"]
        assert args["rollout_backend"] == "in_loop"
        assert args["adversarial_fraction"] == pytest.approx(0.10)
        assert args["rlox_server_url"] == server_url


# ---------------------------------------------------------------------------
# K2) prime-rl venv bin dir is prepended to PATH in the subprocess env
#
# Regression: run_one now prepends os.path.dirname(prime_rl_bin) to PATH so
# that prime-rl's child processes (orchestrator / trainer) can be resolved by
# bare name via PATH when rl is invoked by absolute path rather than uv run.
# CUDA_VISIBLE_DEVICES must still be "0" alongside the PATH change.
# ---------------------------------------------------------------------------

class TestPrimeRlBinDirOnPath:
    """os.path.dirname(prime_rl_bin) must be the FIRST entry in PATH passed
    to subprocess.run, and CUDA_VISIBLE_DEVICES must remain "0"."""

    _NON_TRIVIAL_BIN = "/opt/prime-rl/.venv/bin/rl"
    _EXPECTED_BIN_DIR = "/opt/prime-rl/.venv/bin"

    def _make_run_one_with_bin(
        self,
        bin_path: str,
        base_toml: Path,
        output_root: Path,
        repo_root: Path,
        *,
        scope_for_baseline: bool = False,
    ):
        return make_primerl_run_one(
            base_toml=str(base_toml),
            max_steps=1,
            group_size=4,
            rlox_server_url=_FAKE_SERVER_URL,
            prime_rl_bin=bin_path,
            output_root=str(output_root),
            repo_root=str(repo_root),
            scope_for_baseline=scope_for_baseline,
            in_loop_timeout_secs=60,
        )

    def test_bin_dir_is_first_path_entry_for_rlox(
        self, base_toml: Path, tmp_path: Path, repo_root: Path
    ):
        """For condition='rlox', dirname(prime_rl_bin) is the first PATH component."""
        run_one = self._make_run_one_with_bin(
            self._NON_TRIVIAL_BIN, base_toml, tmp_path, repo_root
        )
        captured_envs: list[dict] = []

        def fake_subprocess_run(cmd, env=None, **kwargs):
            if env is not None:
                captured_envs.append(dict(env))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        assert captured_envs, "subprocess.run env kwarg was never captured"
        path_entries = captured_envs[0]["PATH"].split(os.pathsep)
        assert path_entries[0] == self._EXPECTED_BIN_DIR, (
            f"First PATH entry must be {self._EXPECTED_BIN_DIR!r}; "
            f"got {path_entries[0]!r}\nFull PATH: {captured_envs[0]['PATH']}"
        )

    def test_bin_dir_is_first_path_entry_for_in_loop(
        self, base_toml: Path, tmp_path: Path, repo_root: Path
    ):
        """For condition='in_loop', dirname(prime_rl_bin) is still the first PATH component."""
        run_one = self._make_run_one_with_bin(
            self._NON_TRIVIAL_BIN, base_toml, tmp_path, repo_root,
            scope_for_baseline=False,
        )
        captured_envs: list[dict] = []

        def fake_subprocess_run(cmd, env=None, **kwargs):
            if env is not None:
                captured_envs.append(dict(env))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        assert captured_envs
        path_entries = captured_envs[0]["PATH"].split(os.pathsep)
        assert path_entries[0] == self._EXPECTED_BIN_DIR, (
            f"First PATH entry must be {self._EXPECTED_BIN_DIR!r}; "
            f"got {path_entries[0]!r}"
        )

    def test_bin_dir_precedes_inherited_path(
        self, base_toml: Path, tmp_path: Path, repo_root: Path
    ):
        """The injected bin dir must come BEFORE the inherited PATH (not appended)."""
        run_one = self._make_run_one_with_bin(
            self._NON_TRIVIAL_BIN, base_toml, tmp_path, repo_root
        )
        captured_envs: list[dict] = []

        def fake_subprocess_run(cmd, env=None, **kwargs):
            if env is not None:
                captured_envs.append(dict(env))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        assert captured_envs
        path_val = captured_envs[0]["PATH"]
        inherited_path = os.environ.get("PATH", "")
        if inherited_path:
            assert path_val.startswith(self._EXPECTED_BIN_DIR + os.pathsep), (
                f"PATH must start with '{self._EXPECTED_BIN_DIR}{os.pathsep}'; "
                f"got: {path_val!r}"
            )

    def test_cuda_visible_devices_still_set_alongside_path(
        self, base_toml: Path, tmp_path: Path, repo_root: Path
    ):
        """PATH injection must not displace CUDA_VISIBLE_DEVICES from the env."""
        run_one = self._make_run_one_with_bin(
            self._NON_TRIVIAL_BIN, base_toml, tmp_path, repo_root
        )
        captured_envs: list[dict] = []

        def fake_subprocess_run(cmd, env=None, **kwargs):
            if env is not None:
                captured_envs.append(dict(env))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("rlox", 0, 0.0)

        assert captured_envs
        assert captured_envs[0].get("CUDA_VISIBLE_DEVICES") == "0", (
            "CUDA_VISIBLE_DEVICES must remain '0' after PATH injection"
        )

    def test_bin_dir_path_and_cuda_both_set_for_in_loop(
        self, base_toml: Path, tmp_path: Path, repo_root: Path
    ):
        """For in_loop condition both PATH prepend and CUDA_VISIBLE_DEVICES must be set."""
        run_one = self._make_run_one_with_bin(
            self._NON_TRIVIAL_BIN, base_toml, tmp_path, repo_root,
            scope_for_baseline=False,
        )
        captured_envs: list[dict] = []

        def fake_subprocess_run(cmd, env=None, **kwargs):
            if env is not None:
                captured_envs.append(dict(env))
            p = _extract_toml_path_from_cmd(cmd)
            if p:
                _write_rollouts(Path(p).parent, 1)
            return _FakeProcess(returncode=0)

        with patch("run_benchmark.subprocess.run", side_effect=fake_subprocess_run):
            run_one("in_loop", 0, 0.0)

        assert captured_envs
        env_captured = captured_envs[0]
        assert env_captured.get("CUDA_VISIBLE_DEVICES") == "0"
        assert env_captured["PATH"].split(os.pathsep)[0] == self._EXPECTED_BIN_DIR


# ---------------------------------------------------------------------------
# K) run_sweep key-preservation regression
#
# Regression for the bug where run_sweep rebuilt a 4-key record dict instead
# of passing through all keys returned by run_one.  Downstream sweep drivers
# that read elapsed_secs / completed_steps / mean_reward_last from the results
# list would KeyError against the old implementation.
#
# Contract: run_sweep must preserve EVERY key that run_one returns, both in
# the in-memory results list AND in the per-run JSON artifact it writes to disk.
# ---------------------------------------------------------------------------

class TestRunSweepKeyPreservation:
    """run_sweep must pass all run_one keys through to results and metric-store JSON.

    Uses a pure stub run_one (no subprocess, no make_primerl_run_one) so this
    regression is independent of the Step 8 implementation and tests ONLY the
    run_sweep contract.
    """

    def _stub_run_one(self, condition: str, seed: int, fraction: float) -> dict:
        """Stub that returns the full 7-key dict exactly as make_primerl_run_one would."""
        return {
            "condition": condition,
            "seed": seed,
            "fraction": fraction,
            "survived": True,
            "completed_steps": 5,
            "elapsed_secs": 12.34,
            "mean_reward_last": 0.75,
        }

    def _make_cfg(self, *, n_seeds: int = 1, fractions: list[float] | None = None):
        class _Cfg:
            pass
        cfg = _Cfg()
        cfg.n_seeds = n_seeds
        cfg.adversarial_fractions = fractions if fractions is not None else [0.0, 0.10]
        return cfg

    def test_run_sweep_results_contain_elapsed_secs(self, tmp_path: Path):
        """Regression: elapsed_secs must survive run_sweep — was dropped by 4-key truncation."""
        from run_benchmark import run_sweep

        cfg = self._make_cfg(n_seeds=1, fractions=[0.0, 0.10])
        results = run_sweep(cfg, self._stub_run_one, str(tmp_path / "store"))

        for r in results:
            assert "elapsed_secs" in r, (
                f"elapsed_secs was dropped by run_sweep; keys present: {set(r.keys())}"
            )
            assert isinstance(r["elapsed_secs"], float)
            assert r["elapsed_secs"] == pytest.approx(12.34)

    def test_run_sweep_results_contain_completed_steps(self, tmp_path: Path):
        """Regression: completed_steps must survive run_sweep — was dropped by 4-key truncation."""
        from run_benchmark import run_sweep

        cfg = self._make_cfg(n_seeds=1, fractions=[0.0, 0.10])
        results = run_sweep(cfg, self._stub_run_one, str(tmp_path / "store"))

        for r in results:
            assert "completed_steps" in r, (
                f"completed_steps was dropped by run_sweep; keys present: {set(r.keys())}"
            )
            assert isinstance(r["completed_steps"], int)
            assert r["completed_steps"] == 5

    def test_run_sweep_results_contain_mean_reward_last(self, tmp_path: Path):
        """Regression: mean_reward_last must survive run_sweep — was dropped by 4-key truncation."""
        from run_benchmark import run_sweep

        cfg = self._make_cfg(n_seeds=1, fractions=[0.0, 0.10])
        results = run_sweep(cfg, self._stub_run_one, str(tmp_path / "store"))

        for r in results:
            assert "mean_reward_last" in r, (
                f"mean_reward_last was dropped by run_sweep; keys present: {set(r.keys())}"
            )
            assert isinstance(r["mean_reward_last"], float)
            assert r["mean_reward_last"] == pytest.approx(0.75)

    def test_run_sweep_results_have_all_seven_keys(self, tmp_path: Path):
        """Each result record must carry exactly the 7 standard keys — no truncation."""
        from run_benchmark import run_sweep

        cfg = self._make_cfg(n_seeds=1, fractions=[0.0, 0.10])
        results = run_sweep(cfg, self._stub_run_one, str(tmp_path / "store"))

        assert len(results) == 4, f"Expected 4 records (2 cond × 1 seed × 2 frac), got {len(results)}"
        for r in results:
            missing = _REQUIRED_KEYS - set(r.keys())
            assert not missing, (
                f"run_sweep truncated keys from result record: missing {missing}; "
                f"present {set(r.keys())}"
            )

    def test_run_sweep_artifact_json_contains_elapsed_secs(self, tmp_path: Path):
        """The per-run JSON artifact written to metric_store_dir must also contain elapsed_secs."""
        from run_benchmark import run_sweep

        store_dir = tmp_path / "store"
        cfg = self._make_cfg(n_seeds=1, fractions=[0.0])
        run_sweep(cfg, self._stub_run_one, str(store_dir))

        artifact_files = sorted(store_dir.glob("*.json"))
        assert artifact_files, "run_sweep wrote no JSON artifacts to metric_store_dir"

        for artifact in artifact_files:
            data = json.loads(artifact.read_text(encoding="utf-8"))
            assert "elapsed_secs" in data, (
                f"Artifact {artifact.name} is missing elapsed_secs; keys: {set(data.keys())}"
            )
            assert data["elapsed_secs"] == pytest.approx(12.34)

    def test_run_sweep_artifact_json_contains_completed_steps(self, tmp_path: Path):
        """The per-run JSON artifact must contain completed_steps."""
        from run_benchmark import run_sweep

        store_dir = tmp_path / "store"
        cfg = self._make_cfg(n_seeds=1, fractions=[0.0])
        run_sweep(cfg, self._stub_run_one, str(store_dir))

        for artifact in sorted(store_dir.glob("*.json")):
            data = json.loads(artifact.read_text(encoding="utf-8"))
            assert "completed_steps" in data, (
                f"Artifact {artifact.name} is missing completed_steps; keys: {set(data.keys())}"
            )
            assert data["completed_steps"] == 5

    def test_run_sweep_artifact_json_contains_mean_reward_last(self, tmp_path: Path):
        """The per-run JSON artifact must contain mean_reward_last."""
        from run_benchmark import run_sweep

        store_dir = tmp_path / "store"
        cfg = self._make_cfg(n_seeds=1, fractions=[0.0])
        run_sweep(cfg, self._stub_run_one, str(store_dir))

        for artifact in sorted(store_dir.glob("*.json")):
            data = json.loads(artifact.read_text(encoding="utf-8"))
            assert "mean_reward_last" in data, (
                f"Artifact {artifact.name} is missing mean_reward_last; keys: {set(data.keys())}"
            )
            assert data["mean_reward_last"] == pytest.approx(0.75)

    def test_run_sweep_identity_keys_are_correct_per_grid_point(self, tmp_path: Path):
        """Identity keys condition/seed/fraction/survived must be correct for each grid point."""
        from run_benchmark import run_sweep

        cfg = self._make_cfg(n_seeds=1, fractions=[0.0, 0.10])
        results = run_sweep(cfg, self._stub_run_one, str(tmp_path / "store"))

        # Deterministic order: outer=condition, inner=fraction (1 seed)
        expected_conditions = ["in_loop", "in_loop", "rlox", "rlox"]
        expected_fractions = [0.0, 0.10, 0.0, 0.10]

        assert len(results) == 4
        for i, (r, exp_cond, exp_frac) in enumerate(
            zip(results, expected_conditions, expected_fractions)
        ):
            assert r["condition"] == exp_cond, (
                f"Record {i}: condition={r['condition']!r}, expected {exp_cond!r}"
            )
            assert r["fraction"] == pytest.approx(exp_frac), (
                f"Record {i}: fraction={r['fraction']}, expected {exp_frac}"
            )
            assert r["seed"] == 0
            assert r["survived"] is True

    def test_run_sweep_preserves_full_dict_round_trip(self, tmp_path: Path):
        """Full round-trip: stub returns 7 keys, results list has 7 keys, artifact has 7 keys."""
        from run_benchmark import run_sweep

        store_dir = tmp_path / "store"
        cfg = self._make_cfg(n_seeds=1, fractions=[0.0])
        results = run_sweep(cfg, self._stub_run_one, str(store_dir))

        # In-memory record
        assert len(results) == 2  # 2 conditions
        for r in results:
            missing = _REQUIRED_KEYS - set(r.keys())
            assert not missing, f"In-memory record missing keys: {missing}"

        # On-disk artifacts
        artifacts = sorted(store_dir.glob("*.json"))
        assert len(artifacts) == 2
        for artifact in artifacts:
            data = json.loads(artifact.read_text(encoding="utf-8"))
            missing = _REQUIRED_KEYS - set(data.keys())
            assert not missing, f"Artifact {artifact.name} missing keys: {missing}"


# ---------------------------------------------------------------------------
# Private helpers (test-module-internal only)
# ---------------------------------------------------------------------------

class _FakeProcess:
    """Minimal subprocess.CompletedProcess substitute."""
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def _extract_toml_path_from_cmd(cmd: list[str]) -> str | None:
    """Return the TOML path argument (the token after '@') from a subprocess argv."""
    for i, token in enumerate(cmd):
        if token == "@" and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def _read_gen_toml(path: str) -> dict:
    """Read a rendered TOML file and return as a Python dict."""
    with open(path, "rb") as fh:
        return tomllib.load(fh)
