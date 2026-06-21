"""Smoke tests for the rlox_verify environment package.

Verifies:
  1. Package imports correctly.
  2. load_environment(rollout_backend="in_loop") builds an env.
  3. Dataset has 8 rows with prompt/tests/answer columns.
  4. Reward function scores a CORRECT solution → 1.0.
  5. Reward function scores a WRONG solution → 0.0.
  6. tests fallback: when tests kwarg is empty, answer is used instead.
  7. Adversarial injection fires when fraction=1.0 → reward 0.0.
  8. load_environment(rollout_backend="rlox") POSTs to /verify (mocked).
  9. Invalid backend raises ValueError.
  10. adversarial_fraction > 0 without corpus_path raises ValueError.

Run on wk-system (prime-rl venv):
    /home/wk/prime-rl/.venv/bin/python -m pytest \
        /home/wk/rlox/environments/rlox_verify/tests/test_rlox_verify.py -v

NOTE: CPU-only, no GPU, no training.
"""
from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

import verifiers as vf
import rlox_verify


# ---------------------------------------------------------------------------
# Helper: access our reward func through the RubricGroup that SingleTurnEnv
# wraps around the passed Rubric in verifiers 0.1.15.dev.
# env.rubric is a RubricGroup; our Rubric is rubrics[0]; our func is funcs[0].
# ---------------------------------------------------------------------------

def _get_reward_func(env: vf.Environment):
    return env.rubric.rubrics[0].funcs[0]


# ---------------------------------------------------------------------------
# Helper: minimal adversarial corpus temp file
# ---------------------------------------------------------------------------

def _make_adversarial_corpus_file() -> str:
    """Write a minimal adversarial corpus JSON to a temp file and return its path."""
    samples = [
        {
            "id": "test_001",
            "category": "infinite_loop",
            "language": "python",
            "code": "while True: pass",
            "expected_exit": "timeout",
        }
    ]
    data: dict[str, Any] = {"samples": samples, "sha256": ""}
    # Canonical SHA-256 matching AdversarialCorpus.load convention.
    serialised = json.dumps(data, sort_keys=True, separators=(",", ":"))
    data["sha256"] = hashlib.sha256(serialised.encode()).hexdigest()

    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp)
    tmp.flush()
    return tmp.name


# ---------------------------------------------------------------------------
# Test 1 — import
# ---------------------------------------------------------------------------

def test_import_ok():
    assert hasattr(rlox_verify, "load_environment")


# ---------------------------------------------------------------------------
# Test 2 — environment construction (in_loop)
# ---------------------------------------------------------------------------

def test_load_environment_in_loop():
    env = rlox_verify.load_environment(rollout_backend="in_loop", group_size=2)
    assert isinstance(env, vf.Environment)
    assert env.sampling_args["n"] == 2


# ---------------------------------------------------------------------------
# Test 3 — dataset shape
# ---------------------------------------------------------------------------

def test_dataset_has_expected_columns():
    env = rlox_verify.load_environment(rollout_backend="in_loop")
    ds = env.dataset
    assert ds is not None
    assert len(ds) > 0, f"Dataset must be non-empty, got {len(ds)} rows"
    cols = set(ds.column_names)
    assert "prompt" in cols, f"Missing 'prompt' column, got: {cols}"
    assert "tests" in cols, f"Missing 'tests' column, got: {cols}"
    assert "answer" in cols, f"Missing 'answer' column, got: {cols}"


def test_dataset_n_problems_truncation():
    env = rlox_verify.load_environment(rollout_backend="in_loop", n_problems=3)
    assert env.dataset is not None
    assert len(env.dataset) == 3


# ---------------------------------------------------------------------------
# Test 4 — reward: CORRECT solution → 1.0
# ---------------------------------------------------------------------------

def test_reward_correct_solution():
    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
    )
    reward_func = _get_reward_func(env)

    correct_code = "def add(a, b):\n    return a + b\n"
    tests = (
        "assert add(1, 2) == 3\n"
        "assert add(-1, 1) == 0\n"
        "assert add(0, 0) == 0\n"
    )

    # In 0.1.15.dev, task_score_fields injects dataset columns as kwargs.
    # We call the reward func directly to mirror the rubric's call pattern.
    reward = reward_func(
        prompt=[{"role": "user", "content": "Write add(a,b)"}],
        completion=[{"role": "assistant", "content": correct_code}],
        answer="",
        tests=tests,
        state={},
    )
    assert reward == 1.0, f"Expected 1.0, got {reward}"


# ---------------------------------------------------------------------------
# Test 5 — reward: WRONG solution → 0.0
# ---------------------------------------------------------------------------

def test_reward_wrong_solution():
    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
    )
    reward_func = _get_reward_func(env)

    wrong_code = "def add(a, b):\n    return a - b\n"  # intentionally wrong
    tests = (
        "assert add(1, 2) == 3\n"
        "assert add(-1, 1) == 0\n"
    )

    reward = reward_func(
        prompt=[{"role": "user", "content": "Write add(a,b)"}],
        completion=[{"role": "assistant", "content": wrong_code}],
        answer="",
        tests=tests,
        state={},
    )
    assert reward == 0.0, f"Expected 0.0, got {reward}"


# ---------------------------------------------------------------------------
# Test 6 — tests fallback via answer kwarg (backward compat)
# ---------------------------------------------------------------------------

def test_reward_tests_in_answer_fallback():
    """When tests kwarg is empty string, fall back to answer."""
    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
    )
    reward_func = _get_reward_func(env)

    correct_code = "def add(a, b):\n    return a + b\n"
    tests = "assert add(2, 3) == 5\n"

    reward = reward_func(
        prompt="Write add(a,b)",
        completion=correct_code,
        answer=tests,  # tests passed via answer (old convention)
        tests="",      # empty → falls back to answer
        state={},
    )
    assert reward == 1.0, f"Expected 1.0, got {reward}"


# ---------------------------------------------------------------------------
# Test 7 — adversarial injection fires and execution is attempted
# ---------------------------------------------------------------------------

def test_adversarial_injection_fires():
    """With fraction=1.0 the injector always fires.

    The infinite-loop adversarial sample times out (1s timeout) → reward 0.0.
    We pass a correct completion; the injector replaces it with the adversarial
    sample, so reward must be 0.0 regardless.
    """
    corpus_path = _make_adversarial_corpus_file()
    try:
        env = rlox_verify.load_environment(
            rollout_backend="in_loop",
            per_sample_timeout_secs=1.0,
            adversarial_fraction=1.0,
            adversarial_corpus_path=corpus_path,
            seed=0,
        )
        reward_func = _get_reward_func(env)
        state: dict[str, Any] = {}
        reward = reward_func(
            prompt="Write add(a,b)",
            completion="def add(a,b): return a+b",
            answer="assert add(1,2)==3",
            tests="assert add(1,2)==3",
            state=state,
        )
        # Adversarial sample times out / errors → 0.0
        assert reward == 0.0, f"Expected 0.0 (adversarial timed out), got {reward}"
    finally:
        Path(corpus_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 8 — rlox backend: construction + /verify dispatch (mocked server)
# ---------------------------------------------------------------------------

def test_load_environment_rlox_backend_construction():
    """load_environment(rollout_backend='rlox') builds env and POSTs to /verify."""
    env = rlox_verify.load_environment(
        rollout_backend="rlox",
        rlox_server_url="http://localhost:9999",
        per_sample_timeout_secs=3.0,
    )
    assert isinstance(env, vf.Environment)

    reward_func = _get_reward_func(env)

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"reward": 0.75}

    # Patch at the binding site in rlox_agent.verifiers_adapter (canonical location).
    with patch("rlox_agent.verifiers_adapter.httpx.post", return_value=mock_response) as mock_post:
        reward = reward_func(
            prompt="Write add(a,b)",
            completion="def add(a,b): return a+b",
            answer="",
            tests="assert add(1,2)==3",
            state={},
        )
        assert reward == 0.75, f"Expected 0.75 from mocked /verify, got {reward}"
        assert mock_post.called, "httpx.post was not called — rlox dispatch not triggered"
        url_called = mock_post.call_args[0][0]
        assert "/verify" in url_called, f"Expected POST to /verify, got: {url_called}"


# ---------------------------------------------------------------------------
# Test 9 — invalid backend raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Invalid rollout_backend"):
        rlox_verify.load_environment(rollout_backend="invalid_backend")


# ---------------------------------------------------------------------------
# Test 10 — adversarial_fraction > 0 without corpus_path raises ValueError
# ---------------------------------------------------------------------------

def test_adversarial_fraction_without_corpus_raises():
    with pytest.raises(ValueError, match="adversarial_corpus_path"):
        rlox_verify.load_environment(
            rollout_backend="in_loop",
            adversarial_fraction=0.5,
            adversarial_corpus_path=None,
        )
