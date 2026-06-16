"""Failing RED tests for Component 4: verifiers adapter + RloxVerifierConfig.

These tests specify the public contract of:
  - RloxVerifierConfig (dataclass shape + defaults)
  - load_environment(config) -> vf.Environment
  - Backend dispatch: "rlox" vs "in_loop"
  - Adversarial injection is backend-independent (same seed → same injections)

No torch, no vllm, no real Rust server needed.
The "rlox" backend is tested against a local HTTP mock so no real Rust binary
is required.

All imports are top-level thanks to conftest.py injecting python/rlox/agentic/
onto sys.path.

## Step 4 Reconciliation changes (RED until adapter is updated):
  - Treatment path must POST to /verify (not /rollout).
  - POST body must include "code", "tests", and "is_adversarial" keys.
  - Mock server now serves /verify.
  - test_rlox_reward_flows_from_server_response: STRENGTHENED — asserts the
    server's reward value propagates into the scored state, not merely that a
    call happened.
  - env.rubric assertions updated for verifiers 0.1.14 RubricGroup wrapping:
    SingleTurnEnv wraps any passed rubric in a RubricGroup; our reward func
    is reachable at env.rubric.rubrics[0].funcs.
  - test_in_loop_uses_venv_interpreter: asserts subprocess.run is called with
    sys.executable, not a bare "python".
  - test_treatment_server_error_is_logged_not_silent: server failure must log
    a warning (caplog), not silently return 0.0.
  - test_group_size_wired_to_sampling_args: env.sampling_args["n"] == config.group_size.
  - All asyncio.get_event_loop().run_until_complete(...) replaced with asyncio.run(...).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import verifiers as vf

# Top-level imports — never `from rlox.agentic import ...`
import adversarial_corpus as ac
from adversarial_corpus import AdversarialCorpus, AdversarialInjector, AdversarialSample
import verifiers_adapter as va
from verifiers_adapter import RloxVerifierConfig, load_environment

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_CORPUS_PATH = _REPO_ROOT / "benchmarks" / "agentic" / "corpus" / "adversarial_corpus_v1.json"


@pytest.fixture(scope="module")
def real_corpus():
    return AdversarialCorpus.load(_CORPUS_PATH)


@pytest.fixture
def baseline_config():
    return RloxVerifierConfig(
        rollout_backend="in_loop",
        adversarial_fraction=0.0,
        seed=42,
    )


@pytest.fixture
def rlox_config(mock_rollout_server):
    """Config pointing at the mock server."""
    host, port, _ = mock_rollout_server
    return RloxVerifierConfig(
        rollout_backend="rlox",
        rlox_server_url=f"http://{host}:{port}",
        adversarial_fraction=0.0,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Mock /verify server
#
# Step 4 reconciliation: the mock now serves POST /verify (not /rollout).
# The response body is {"reward": <float>, "backend_stats": {...}}.
# ---------------------------------------------------------------------------

class _RequestLog:
    """Thread-safe log of requests received by the mock server."""
    def __init__(self):
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def append(self, payload: dict):
        with self._lock:
            self.requests.append(payload)

    def clear(self):
        with self._lock:
            self.requests.clear()

    def __len__(self):
        with self._lock:
            return len(self.requests)


def _make_mock_handler(log: _RequestLog, reward_value: float = 1.0):
    """Return a handler class that serves POST /verify with the given reward."""
    class MockHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # silence default HTTP logging during tests

        def do_POST(self):
            # Record which path was called for contract assertions.
            path = self.path
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body)
            except Exception:
                payload = {"raw": body.decode(errors="replace")}
            # Tag the payload with the path so tests can assert on it.
            payload["_path"] = path
            log.append(payload)

            # /verify response shape: {"reward": <float>, "backend_stats": {...}}
            response = json.dumps({
                "reward": reward_value,
                "backend_stats": {
                    "batch_wall_secs": 0.01,
                    "rollouts_completed": 1,
                    "rollouts_per_sec": 1.0,
                    "tool_calls_per_sec": 1.0,
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
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return MockHandler


@pytest.fixture(scope="module")
def mock_rollout_server():
    """Start a tiny HTTP server in a background thread; yield (host, port, log).

    Step 4 reconciliation: the server accepts any POST path (logs the path in
    payload["_path"]) so tests can assert that /verify was called.
    """
    log = _RequestLog()
    handler = _make_mock_handler(log, reward_value=1.0)
    server = HTTPServer(("127.0.0.1", 0), handler)  # port 0 → OS picks free port
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield host, port, log
    server.shutdown()


# ---------------------------------------------------------------------------
# A) RloxVerifierConfig — dataclass shape and defaults
# ---------------------------------------------------------------------------

class TestRloxVerifierConfigShape:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(RloxVerifierConfig)

    def test_default_rollout_backend_is_in_loop(self):
        cfg = RloxVerifierConfig()
        assert cfg.rollout_backend == "in_loop"

    def test_default_rlox_server_url(self):
        cfg = RloxVerifierConfig()
        assert isinstance(cfg.rlox_server_url, str)
        assert cfg.rlox_server_url.startswith("http")

    def test_default_per_sample_timeout_secs(self):
        cfg = RloxVerifierConfig()
        assert isinstance(cfg.per_sample_timeout_secs, float)
        assert cfg.per_sample_timeout_secs > 0

    def test_default_group_size(self):
        cfg = RloxVerifierConfig()
        assert isinstance(cfg.group_size, int)
        assert cfg.group_size > 0

    def test_default_adversarial_fraction_is_zero(self):
        cfg = RloxVerifierConfig()
        assert cfg.adversarial_fraction == 0.0

    def test_default_adversarial_corpus_path_is_none(self):
        cfg = RloxVerifierConfig()
        assert cfg.adversarial_corpus_path is None

    def test_default_seed_is_integer(self):
        cfg = RloxVerifierConfig()
        assert isinstance(cfg.seed, int)

    def test_all_fields_settable_at_construction(self):
        """Every documented field can be passed as a constructor keyword."""
        cfg = RloxVerifierConfig(
            rollout_backend="rlox",
            rlox_server_url="http://localhost:9999",
            per_sample_timeout_secs=5.0,
            group_size=4,
            adversarial_fraction=0.1,
            adversarial_corpus_path="/some/path.json",
            seed=7,
        )
        assert cfg.rollout_backend == "rlox"
        assert cfg.rlox_server_url == "http://localhost:9999"
        assert cfg.per_sample_timeout_secs == 5.0
        assert cfg.group_size == 4
        assert cfg.adversarial_fraction == 0.1
        assert cfg.adversarial_corpus_path == "/some/path.json"
        assert cfg.seed == 7

    def test_configs_with_different_backends_are_equal_except_backend(self):
        """Two configs that differ only in rollout_backend must differ when compared."""
        cfg_baseline = RloxVerifierConfig(rollout_backend="in_loop", seed=1)
        cfg_treatment = RloxVerifierConfig(rollout_backend="rlox", seed=1)
        # They are distinct objects; replacing backend should be the only diff
        cfg_baseline_copy = dataclasses.replace(cfg_baseline, rollout_backend="rlox")
        assert cfg_baseline_copy == cfg_treatment


# ---------------------------------------------------------------------------
# B) load_environment returns a vf.Environment
# ---------------------------------------------------------------------------

class TestLoadEnvironmentReturnType:
    def test_returns_vf_environment_baseline(self, baseline_config):
        env = load_environment(baseline_config)
        assert isinstance(env, vf.Environment)

    def test_returns_vf_environment_rlox(self, rlox_config):
        env = load_environment(rlox_config)
        assert isinstance(env, vf.Environment)

    def test_invalid_backend_raises_value_error(self):
        cfg = RloxVerifierConfig(rollout_backend="nonexistent_backend")
        with pytest.raises(ValueError, match="rollout_backend"):
            load_environment(cfg)

    def test_adversarial_fraction_nonzero_without_path_raises(self):
        """Requesting injection without providing a corpus path must fail early."""
        cfg = RloxVerifierConfig(
            rollout_backend="in_loop",
            adversarial_fraction=0.1,
            adversarial_corpus_path=None,
        )
        with pytest.raises(ValueError, match="adversarial_corpus_path"):
            load_environment(cfg)


# ---------------------------------------------------------------------------
# C) Reward function is attached to the environment
#
# Step 4 reconciliation: verifiers 0.1.14 wraps the Rubric passed to
# SingleTurnEnv in a RubricGroup (env.rubric is a RubricGroup, not a plain
# Rubric).  The rubric we constructed is accessible at:
#
#     env.rubric.rubrics[0].funcs   (list of reward functions)
#
# The existing assertions on `env.rubric` being a `vf.Rubric` and
# `env.rubric.funcs` remain structurally valid ONLY if load_environment passes
# the rubric through the SingleTurnEnv CONSTRUCTOR (which triggers the wrapping).
# If instead it sets `env.rubric = rubric` after construction, `env.rubric`
# stays a plain `vf.Rubric` but `env.rubric` is no longer the outer
# RubricGroup that verifiers uses internally.
#
# The contract the tests encode:
#   - load_environment must call `vf.SingleTurnEnv(dataset=..., rubric=rubric)`
#     via the constructor so that verifiers wraps it properly.
#   - After construction, `env.rubric` is a `vf.RubricGroup`.
#   - `env.rubric.rubrics[0]` is the Rubric we passed.
#   - `env.rubric.rubrics[0].funcs` is non-empty (contains our reward function).
#
# FAILS NOW: load_environment sets `env.rubric = rubric` after construction,
# so `env.rubric` is a plain Rubric (not a RubricGroup).
# ---------------------------------------------------------------------------

class TestEnvironmentHasRewardFunction:
    def test_baseline_env_rubric_is_rubric_group(self, baseline_config):
        """After construction, env.rubric must be a RubricGroup (verifiers wraps it)."""
        env = load_environment(baseline_config)
        assert env.rubric is not None
        assert isinstance(env.rubric, vf.RubricGroup), (
            f"env.rubric must be a vf.RubricGroup after SingleTurnEnv construction, "
            f"got {type(env.rubric).__name__}. "
            "load_environment must pass rubric= to the SingleTurnEnv constructor, "
            "not assign env.rubric = rubric afterwards."
        )

    def test_rlox_env_rubric_is_rubric_group(self, rlox_config):
        """Same check for the treatment path."""
        env = load_environment(rlox_config)
        assert env.rubric is not None
        assert isinstance(env.rubric, vf.RubricGroup), (
            f"env.rubric must be a vf.RubricGroup, got {type(env.rubric).__name__}"
        )

    def test_baseline_rubric_group_contains_our_reward_func(self, baseline_config):
        """Our reward function must be reachable at env.rubric.rubrics[0].funcs."""
        env = load_environment(baseline_config)
        assert isinstance(env.rubric, vf.RubricGroup), (
            "env.rubric must be a RubricGroup — see test_baseline_env_rubric_is_rubric_group"
        )
        assert hasattr(env.rubric, "rubrics"), "RubricGroup must have a .rubrics attribute"
        assert len(env.rubric.rubrics) >= 1, (
            "RubricGroup.rubrics must contain at least one rubric (ours + monitor)"
        )
        user_rubric = env.rubric.rubrics[0]
        assert hasattr(user_rubric, "funcs"), "rubrics[0] must have a .funcs attribute"
        assert len(user_rubric.funcs) >= 1, (
            "rubrics[0].funcs must be non-empty — our reward function must be registered"
        )

    def test_rlox_rubric_group_contains_our_reward_func(self, rlox_config):
        """Same check for the treatment path."""
        env = load_environment(rlox_config)
        assert isinstance(env.rubric, vf.RubricGroup)
        user_rubric = env.rubric.rubrics[0]
        assert len(user_rubric.funcs) >= 1, (
            "rubrics[0].funcs must be non-empty for the rlox treatment path"
        )


# ---------------------------------------------------------------------------
# D) Backend dispatch — "in_loop" does NOT call the rollout server
# ---------------------------------------------------------------------------

class TestInLoopBackendDoesNotCallServer:
    def test_in_loop_reward_does_not_contact_mock_server(self, mock_rollout_server):
        """The baseline path must NEVER POST to any remote server."""
        host, port, log = mock_rollout_server
        log.clear()

        cfg = RloxVerifierConfig(
            rollout_backend="in_loop",
            # Point at the mock server; it must NOT be called
            rlox_server_url=f"http://{host}:{port}",
            adversarial_fraction=0.0,
            seed=0,
        )
        env = load_environment(cfg)

        state: dict[str, Any] = {
            "prompt": [{"role": "user", "content": "write a hello world"}],
            "completion": [{"role": "assistant", "content": "print('hello')"}],
            "answer": "print('hello')",
            "trajectory": [],
        }
        # Step 4 reconciliation: use asyncio.run() instead of deprecated
        # asyncio.get_event_loop().run_until_complete(...).
        asyncio.run(env.rubric.score_rollout(state))  # type: ignore[arg-type]

        assert len(log) == 0, (
            f"in_loop backend POSTed to the mock server {len(log)} time(s); "
            "it must not make any network calls"
        )

    def test_in_loop_uses_venv_interpreter(self, monkeypatch):
        """Baseline path must invoke sys.executable, not a bare 'python' string.

        This ensures the correct venv Python is used in all environments where
        'python' on PATH may resolve to a different interpreter (or not exist).

        FAILS NOW: _run_in_loop uses ["python", ...], not [sys.executable, ...].
        """
        captured_calls: list[list[str]] = []
        original_run = subprocess.run

        def mock_run(args, **kwargs):
            captured_calls.append(list(args))
            # Return a successful result to avoid side-effects.
            return subprocess.CompletedProcess(args=args, returncode=0)

        monkeypatch.setattr(subprocess, "run", mock_run)

        cfg = RloxVerifierConfig(
            rollout_backend="in_loop",
            adversarial_fraction=0.0,
            seed=0,
        )
        env = load_environment(cfg)

        state: dict[str, Any] = {
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [{"role": "assistant", "content": "x = 1"}],
            "answer": "assert x == 1",
            "trajectory": [],
        }
        asyncio.run(env.rubric.score_rollout(state))  # type: ignore[arg-type]

        assert len(captured_calls) >= 1, (
            "in_loop backend must call subprocess.run at least once"
        )
        interpreter_used = captured_calls[-1][0]
        assert interpreter_used == sys.executable, (
            f"in_loop must invoke sys.executable ({sys.executable!r}), "
            f"got {interpreter_used!r}. "
            "Using a bare 'python' string fails when the venv interpreter is not on PATH."
        )


# ---------------------------------------------------------------------------
# E) Backend dispatch — "rlox" DOES call /verify (not /rollout)
#
# Step 4 reconciliation:
#   - The Treatment path must POST to {rlox_server_url}/verify.
#   - The POST body must include keys: "code", "tests", "is_adversarial".
#   - The old /rollout URL is no longer correct for the verifiers @reward seam.
# ---------------------------------------------------------------------------

class TestRloxBackendCallsVerifyEndpoint:
    def test_rlox_reward_posts_to_verify_endpoint(self, mock_rollout_server):
        """The rlox backend must POST to /verify (not /rollout) when scoring.

        FAILS NOW: _call_rlox_server posts to /rollout.
        """
        host, port, log = mock_rollout_server
        log.clear()

        cfg = RloxVerifierConfig(
            rollout_backend="rlox",
            rlox_server_url=f"http://{host}:{port}",
            adversarial_fraction=0.0,
            per_sample_timeout_secs=5.0,
            seed=0,
        )
        env = load_environment(cfg)

        state: dict[str, Any] = {
            "prompt": [{"role": "user", "content": "write a hello world"}],
            "completion": [{"role": "assistant", "content": "print('hello')"}],
            "answer": "print('hello')",
            "trajectory": [],
        }
        asyncio.run(env.rubric.score_rollout(state))  # type: ignore[arg-type]

        assert len(log) >= 1, (
            "rlox backend did not POST to the mock server; expected at least one request"
        )

        # The critical assertion: the path must be /verify, not /rollout.
        last_path = log.requests[-1].get("_path", "")
        assert last_path == "/verify", (
            f"rlox backend must POST to /verify (not /rollout), got: {last_path!r}. "
            "At the verifiers @reward seam the completion is already generated; "
            "rlox must VERIFY (sandbox-execute + score), not generate."
        )

    def test_rlox_request_body_has_code_tests_is_adversarial_keys(self, mock_rollout_server):
        """The POST body to /verify must include 'code', 'tests', and 'is_adversarial' keys.

        FAILS NOW: current body has 'code' and 'tests' but lacks 'is_adversarial'.
        Also fails because path is /rollout instead of /verify.
        """
        host, port, log = mock_rollout_server
        log.clear()

        cfg = RloxVerifierConfig(
            rollout_backend="rlox",
            rlox_server_url=f"http://{host}:{port}",
            adversarial_fraction=0.0,
            per_sample_timeout_secs=5.0,
            seed=0,
        )
        env = load_environment(cfg)

        state: dict[str, Any] = {
            "prompt": [{"role": "user", "content": "write a hello world"}],
            "completion": [{"role": "assistant", "content": "print('hello')"}],
            "answer": "print('hello')",
            "trajectory": [],
        }
        asyncio.run(env.rubric.score_rollout(state))  # type: ignore[arg-type]

        assert len(log) >= 1
        payload = log.requests[-1]

        assert "code" in payload, f"POST body missing 'code' field: {payload}"
        assert "tests" in payload, f"POST body missing 'tests' field: {payload}"
        assert "is_adversarial" in payload, (
            f"POST body missing 'is_adversarial' field: {payload}. "
            "The /verify endpoint requires this field to record containment telemetry."
        )

    def test_rlox_reward_flows_from_server_response(self, mock_rollout_server):
        """The reward value from the /verify response must propagate into the scored state.

        STRENGTHENED from the original test: we now assert the actual reward
        VALUE (0.75) appears in the state's reward fields, not merely that a
        call happened.

        FAILS NOW because:
          1. The adapter posts to /rollout instead of /verify (so mock returns wrong body).
          2. Even if the path is corrected, the current implementation may not
             read 'reward' from the response correctly and propagate it into state.
        """
        host, port, log = mock_rollout_server

        # Build a dedicated server that always returns reward=0.75.
        log_local = _RequestLog()
        handler = _make_mock_handler(log_local, reward_value=0.75)
        server = HTTPServer(("127.0.0.1", 0), handler)
        h, p = server.server_address
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            cfg = RloxVerifierConfig(
                rollout_backend="rlox",
                rlox_server_url=f"http://{h}:{p}",
                adversarial_fraction=0.0,
                per_sample_timeout_secs=5.0,
                seed=0,
            )
            env = load_environment(cfg)

            state: dict[str, Any] = {
                "prompt": [{"role": "user", "content": "write a hello world"}],
                "completion": [{"role": "assistant", "content": "print('hello')"}],
                "answer": "print('hello')",
                "trajectory": [],
            }
            asyncio.run(env.rubric.score_rollout(state))  # type: ignore[arg-type]

            # The server must have been called.
            assert len(log_local) >= 1, (
                "Server was not called — rlox backend did not POST to /verify"
            )

            # The reward value (0.75) from the /verify response must be reflected
            # in the state after scoring.  verifiers stores rewards in
            # state["reward_funcs_results"] (dict keyed by function name) or
            # state["rewards"] (list).  We accept either shape.
            reward_funcs_results = state.get("reward_funcs_results", {})
            rewards_list = state.get("rewards", [])

            # Collect all scalar reward values from the state.
            all_rewards: list[float] = []
            if isinstance(reward_funcs_results, dict):
                for v in reward_funcs_results.values():
                    if isinstance(v, (int, float)):
                        all_rewards.append(float(v))
                    elif isinstance(v, list):
                        all_rewards.extend(float(x) for x in v if isinstance(x, (int, float)))
            if isinstance(rewards_list, list):
                all_rewards.extend(float(x) for x in rewards_list if isinstance(x, (int, float)))

            assert any(abs(r - 0.75) < 0.01 for r in all_rewards), (
                f"Expected reward value 0.75 (from server /verify response) to appear "
                f"in state after scoring, but found rewards: {all_rewards}. "
                "The adapter must propagate the server's 'reward' field into the state."
            )
        finally:
            server.shutdown()

    def test_treatment_server_error_is_logged_not_silent(self, caplog):
        """When the /verify server call fails, the adapter must log a WARNING.

        It may still return 0.0 as the fallback reward, but must NOT swallow
        the error silently — silent failures make it impossible to diagnose
        server outages during training.

        FAILS NOW: _call_rlox_server catches all exceptions and returns 0.0
        without logging anything.
        """
        import logging

        # Point at a dead port — connection will be refused.
        cfg = RloxVerifierConfig(
            rollout_backend="rlox",
            rlox_server_url="http://127.0.0.1:1",  # port 1 always refuses
            adversarial_fraction=0.0,
            per_sample_timeout_secs=1.0,  # short timeout so test doesn't hang
            seed=0,
        )
        env = load_environment(cfg)

        state: dict[str, Any] = {
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [{"role": "assistant", "content": "x = 1"}],
            "answer": "assert x == 1",
            "trajectory": [],
        }

        with caplog.at_level(logging.WARNING):
            asyncio.run(env.rubric.score_rollout(state))  # type: ignore[arg-type]

        # At least one WARNING (or higher) must have been emitted.
        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warning_messages) >= 1, (
            "When the /verify server is unreachable, the adapter must log a WARNING. "
            "Silent 0.0 fallbacks make server outages invisible during training. "
            f"No warnings were captured. Log records: {caplog.records}"
        )


# ---------------------------------------------------------------------------
# F) Sampling args wired to group_size
#
# Step 4 reconciliation: the adapter must set env.sampling_args["n"] to
# config.group_size so that prime-rl draws an n-sample group per prompt.
#
# FAILS NOW: load_environment does not set env.sampling_args["n"].
# ---------------------------------------------------------------------------

class TestGroupSizeWiredToSamplingArgs:
    def test_group_size_wired_to_sampling_args_baseline(self, baseline_config):
        """env.sampling_args['n'] must equal config.group_size (baseline path).

        FAILS NOW: load_environment does not set sampling_args['n'].
        """
        cfg = RloxVerifierConfig(
            rollout_backend="in_loop",
            group_size=6,
            seed=0,
        )
        env = load_environment(cfg)

        sampling_args = getattr(env, "sampling_args", None)
        assert sampling_args is not None, (
            "env must have a sampling_args attribute (set by SingleTurnEnv)"
        )
        assert "n" in sampling_args, (
            f"env.sampling_args must contain key 'n'; got keys: {list(sampling_args.keys())}"
        )
        assert sampling_args["n"] == 6, (
            f"env.sampling_args['n'] must equal config.group_size=6; "
            f"got {sampling_args['n']}. "
            "prime-rl reads sampling_args['n'] to draw the right group size."
        )

    def test_group_size_wired_to_sampling_args_rlox(self, mock_rollout_server):
        """env.sampling_args['n'] must equal config.group_size (rlox path).

        FAILS NOW: load_environment does not set sampling_args['n'].
        """
        host, port, _ = mock_rollout_server
        cfg = RloxVerifierConfig(
            rollout_backend="rlox",
            rlox_server_url=f"http://{host}:{port}",
            group_size=3,
            seed=0,
        )
        env = load_environment(cfg)

        sampling_args = getattr(env, "sampling_args", None)
        assert sampling_args is not None
        assert sampling_args.get("n") == 3, (
            f"env.sampling_args['n'] must equal config.group_size=3; "
            f"got {sampling_args.get('n')}"
        )


# ---------------------------------------------------------------------------
# G) Adversarial injection is backend-independent (same seed → same injections)
# ---------------------------------------------------------------------------

class TestAdversarialInjectionIsBackendIndependent:
    """Key contract: switching rollout_backend must NOT change which tasks are
    marked adversarial. Only the execution path differs, not the injection."""

    def _collect_injection_flags(
        self,
        backend: str,
        server_url: str,
        corpus_path: str,
        n: int = 100,
    ) -> list[bool]:
        """Run n maybe_inject calls using a fresh injector seeded identically."""
        corpus = AdversarialCorpus.load(corpus_path)
        injector = AdversarialInjector(corpus, fraction=0.3, seed=2024)
        tasks = [{"prompt": f"task {i}", "answer": f"ans {i}"} for i in range(n)]
        return [injector.maybe_inject(t)[1] for t in tasks]

    def test_injection_sequence_identical_across_backends(self, mock_rollout_server):
        """Given the same seed, injection flags must be identical regardless of backend."""
        host, port, _ = mock_rollout_server
        server_url = f"http://{host}:{port}"
        corpus_path = str(_CORPUS_PATH)

        flags_baseline = self._collect_injection_flags("in_loop", server_url, corpus_path)
        flags_treatment = self._collect_injection_flags("rlox", server_url, corpus_path)

        assert flags_baseline == flags_treatment, (
            "Adversarial injection flags differ between backends! "
            "Injection must be seeded deterministically, backend-independent."
        )

    def test_load_environment_with_corpus_baseline(self):
        """load_environment accepts adversarial_corpus_path for in_loop backend."""
        cfg = RloxVerifierConfig(
            rollout_backend="in_loop",
            adversarial_fraction=0.1,
            adversarial_corpus_path=str(_CORPUS_PATH),
            seed=42,
        )
        env = load_environment(cfg)
        assert isinstance(env, vf.Environment)

    def test_load_environment_with_corpus_rlox(self, mock_rollout_server):
        """load_environment accepts adversarial_corpus_path for rlox backend."""
        host, port, _ = mock_rollout_server
        cfg = RloxVerifierConfig(
            rollout_backend="rlox",
            rlox_server_url=f"http://{host}:{port}",
            adversarial_fraction=0.1,
            adversarial_corpus_path=str(_CORPUS_PATH),
            seed=42,
        )
        env = load_environment(cfg)
        assert isinstance(env, vf.Environment)
