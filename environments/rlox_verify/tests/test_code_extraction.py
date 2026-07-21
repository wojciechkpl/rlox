"""RED tests: rlox_verify reward function must apply fence extraction.

Root cause being tested
-----------------------
``rlox_verify.load_environment``'s ``_reward_func`` calls ``extract_text``
(line ~1013 of __init__.py) which simply joins the content of message dicts.
When a base model emits markdown (prose + fenced code block), the raw
markdown string is passed to ``run_in_loop``/``call_rlox_server``, which tries
to execute it as Python.  The markdown is not valid Python → SyntaxError →
reward 0.0.

The fix is to call ``extract_python_code`` (which lives in
``rlox_agent.verifiers_adapter`` after the move) after ``extract_text`` in the
non-adversarial branch of ``_reward_func``.

Fix contracts tested here
-------------------------
1. ``rlox_verify``'s reward func applies fence extraction: a markdown
   completion containing a ```python fenced correct solution scores 1.0 (not
   0.0 as today).
2. A plain (no-fence) correct completion still scores 1.0.
3. Adversarial bypass is intact: when the injector fires (fraction=1.0), the
   adversarial sample's ``.code`` is used verbatim, fence extraction is NOT
   applied, and the reward is 0.0 (the sample is an infinite loop that times
   out).

Environment requirements
------------------------
``verifiers`` and ``rlox_verify`` must be installed.  These packages are not
available on the laptop venv so every test is behind:

    pytest.importorskip("verifiers")

The tests collect cleanly on this machine (importorskip skips them) and run
for real on wk-system where the prime-rl venv has ``verifiers``.

Run on wk-system:
    RLOX_NO_MBPP=1 /home/wk/prime-rl/.venv/bin/python -m pytest \\
        /home/wk/rlox/environments/rlox_verify/tests/test_code_extraction.py -v

Why RLOX_NO_MBPP=1?
    Avoids a HuggingFace network call during tests; the fallback dataset
    (add, is_even, etc.) has well-defined tests we can drive directly.
"""
from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Guard: skip this entire module if verifiers is not installed.
# This keeps the test collection clean on the laptop while still
# being runnable on wk-system.
vf = pytest.importorskip("verifiers", reason="verifiers not installed; run on wk-system")
rlox_verify = pytest.importorskip(
    "rlox_verify", reason="rlox_verify not installed; run on wk-system"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_reward_func(env: Any) -> Any:
    """Reach our reward function through the RubricGroup that SingleTurnEnv
    wraps around the Rubric in verifiers 0.1.15.dev.

    env.rubric is a RubricGroup; our Rubric is at rubrics[0]; our func at funcs[0].
    """
    return env.rubric.rubrics[0].funcs[0]


def _make_adversarial_corpus_file() -> str:
    """Write a minimal adversarial corpus JSON (single infinite-loop sample)
    to a temp file and return its path.

    The SHA-256 is computed using the canonical convention expected by
    AdversarialCorpus.load: set sha256 to "", json.dumps(sort_keys=True,
    separators=(",", ":")), sha256hex.
    """
    samples = [
        {
            "id": "test_infinite_loop",
            "category": "infinite_loop",
            "language": "python",
            "code": "while True: pass",
            "expected_exit": "timeout",
        }
    ]
    data: dict[str, Any] = {"samples": samples, "sha256": ""}
    serialised = json.dumps(data, sort_keys=True, separators=(",", ":"))
    data["sha256"] = hashlib.sha256(serialised.encode()).hexdigest()

    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp)
    tmp.flush()
    return tmp.name


# ---------------------------------------------------------------------------
# Test 1 — reward func applies fence extraction (the core regression test)
#
# FAILS NOW: _reward_func calls extract_text(completion) but NOT
# extract_python_code, so the fenced markdown is passed raw to subprocess
# execution → SyntaxError → 0.0.
# ---------------------------------------------------------------------------

def test_fenced_markdown_completion_scores_1_0():
    """A markdown completion containing a correct solution in a ```python
    fenced block must score 1.0.

    This is the primary regression: today it scores 0.0 because the fence
    is not stripped before subprocess execution.

    FAILS NOW: reward is 0.0 instead of 1.0.
    """
    import os
    os.environ.setdefault("RLOX_NO_MBPP", "1")  # avoid network call

    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
    )
    reward_func = _get_reward_func(env)

    # A realistic base-model completion: reasoning preamble + fenced code block.
    fenced_completion = (
        "Sure! I'll solve this step by step.\n\n"
        "The function needs to add two numbers together. "
        "Here is my implementation:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "This should work for all inputs."
    )
    tests = (
        "assert add(1, 2) == 3\n"
        "assert add(-1, 1) == 0\n"
        "assert add(0, 0) == 0\n"
    )

    reward = reward_func(
        prompt=[{"role": "user", "content": "Write a Python function add(a, b)."}],
        completion=[{"role": "assistant", "content": fenced_completion}],
        answer="",
        tests=tests,
        state={},
    )
    assert reward == 1.0, (
        f"Expected 1.0 for a fenced-markdown correct completion, got {reward}. "
        "The reward function must strip the ```python fence before executing. "
        "Fix: call extract_python_code(extract_text(completion)) in _reward_func."
    )


def test_fenced_markdown_completion_last_block_used():
    """When the completion has multiple ```python blocks (reasoning + answer),
    the LAST block is extracted and scored.

    FAILS NOW: raw markdown → 0.0.
    """
    import os
    os.environ.setdefault("RLOX_NO_MBPP", "1")

    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
    )
    reward_func = _get_reward_func(env)

    # Model emits a wrong first attempt and a correct final answer.
    fenced_completion = (
        "Let me try:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a - b  # oops\n"
        "```\n"
        "Wait, I made an error. Corrected:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    tests = "assert add(2, 3) == 5\nassert add(0, 0) == 0\n"

    reward = reward_func(
        prompt=[{"role": "user", "content": "Write add(a, b)."}],
        completion=[{"role": "assistant", "content": fenced_completion}],
        answer="",
        tests=tests,
        state={},
    )
    assert reward == 1.0, (
        f"Expected 1.0 (last block is correct), got {reward}. "
        "extract_python_code must take the last ```python block."
    )


# ---------------------------------------------------------------------------
# Test 2 — plain (no-fence) correct completion still scores 1.0
#
# This must PASS after the fix as well: the fence extraction is applied but
# the fallback path returns the code unchanged, so there's no regression.
# This test is written RED here to confirm it should hold after fix.
# If it already passes today, the implementer need not worry about it — but
# it must remain green after the fix.
# ---------------------------------------------------------------------------

def test_plain_code_completion_scores_1_0():
    """A completion that is already bare Python (no fence) must still score 1.0.

    After the fix, extract_python_code is applied but falls through to the
    no-op path and returns the code unchanged.  This must not regress.

    This test may PASS today (bare code already worked before the bug).
    If it does pass, it is a green regression guard.
    """
    import os
    os.environ.setdefault("RLOX_NO_MBPP", "1")

    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
    )
    reward_func = _get_reward_func(env)

    bare_code = "def add(a, b):\n    return a + b\n"
    tests = "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n"

    reward = reward_func(
        prompt=[{"role": "user", "content": "Write add(a, b)."}],
        completion=[{"role": "assistant", "content": bare_code}],
        answer="",
        tests=tests,
        state={},
    )
    assert reward == 1.0, (
        f"Expected 1.0 for a bare-Python correct completion, got {reward}. "
        "The fix must not break completions that are already clean Python."
    )


def test_wrong_fenced_completion_scores_0_0():
    """A fenced completion whose extracted code is incorrect must score 0.0.

    After extraction the code fails the tests → 0.0.
    (Confirms extraction doesn't accidentally hide bugs.)
    """
    import os
    os.environ.setdefault("RLOX_NO_MBPP", "1")

    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
    )
    reward_func = _get_reward_func(env)

    fenced_wrong = (
        "Here is my answer:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a - b  # intentionally wrong\n"
        "```\n"
    )
    tests = "assert add(1, 2) == 3\n"

    reward = reward_func(
        prompt=[{"role": "user", "content": "Write add(a, b)."}],
        completion=[{"role": "assistant", "content": fenced_wrong}],
        answer="",
        tests=tests,
        state={},
    )
    assert reward == 0.0, (
        f"Expected 0.0 for an extracted-but-wrong completion, got {reward}."
    )


# ---------------------------------------------------------------------------
# Test 3 — adversarial bypass intact
#
# When the injector fires (fraction=1.0), the adversarial sample's .code
# attribute is used verbatim.  extract_python_code must NOT be applied to it.
# The sample is an infinite loop → times out → reward 0.0.
#
# FAILS NOW: reward may be non-zero if extraction accidentally produces
# something executable from the adversarial code string, OR because the
# injector is wired correctly but we want to pin the contract here.
# The most important assertion is that reward == 0.0 AND the injector path
# (not the fence-extraction path) was taken.
# ---------------------------------------------------------------------------

def test_adversarial_bypass_not_routed_through_fence_extraction():
    """When adversarial injection fires, the sample's .code is used verbatim.

    Contract:
    - The injector replaces the task regardless of what the completion contains.
    - extract_python_code is NOT called on adversarial code (it bypasses the
      model-completion branch entirely).
    - The adversarial sample (infinite loop) times out → reward 0.0.

    We also pass a correct fenced completion to confirm the injector truly
    replaced it (i.e. the 0.0 is not because the fenced code itself failed).

    FAILS NOW if:
      (a) The fix is not applied and the completion path is used instead of the
          adversarial path (reward could accidentally be 1.0).
      (b) The fix is applied but breaks the adversarial bypass (extract applied
          to adversarial code; outcome may differ).

    After a CORRECT fix, this test must pass: reward == 0.0 because the
    infinite-loop adversarial code is executed verbatim and times out.
    """
    import os
    os.environ.setdefault("RLOX_NO_MBPP", "1")

    corpus_path = _make_adversarial_corpus_file()
    try:
        env = rlox_verify.load_environment(
            rollout_backend="in_loop",
            per_sample_timeout_secs=1.0,  # short: the loop will time out fast
            adversarial_fraction=1.0,     # always inject
            adversarial_corpus_path=corpus_path,
            seed=0,
        )
        reward_func = _get_reward_func(env)

        # Pass a CORRECT fenced completion. After injection the adversarial
        # sample replaces it entirely — if reward != 0.0, the bypass broke.
        correct_fenced_completion = (
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n"
        )
        state: dict[str, Any] = {}
        reward = reward_func(
            prompt="Write add(a,b).",
            completion=correct_fenced_completion,
            answer="assert add(2,3)==5",
            tests="assert add(2,3)==5",
            state=state,
        )
        assert reward == 0.0, (
            f"Expected 0.0 (adversarial infinite loop timed out), got {reward}. "
            "If the injector fired correctly, the adversarial sample must NOT be "
            "routed through fence extraction — it must run its raw .code verbatim."
        )
    finally:
        Path(corpus_path).unlink(missing_ok=True)


def test_adversarial_injection_not_applied_when_fraction_zero():
    """Baseline sanity: with fraction=0.0, a correct fenced completion scores 1.0.

    This confirms the injector is dormant and does not interfere when disabled.
    Combined with the fence-extraction fix, this should score 1.0.

    FAILS NOW because fence extraction is not applied → 0.0.
    """
    import os
    os.environ.setdefault("RLOX_NO_MBPP", "1")

    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
        adversarial_fraction=0.0,  # injector off
    )
    reward_func = _get_reward_func(env)

    fenced_completion = (
        "My solution:\n"
        "```python\n"
        "def is_even(n):\n"
        "    return n % 2 == 0\n"
        "```\n"
    )
    tests = (
        "assert is_even(2) == True\n"
        "assert is_even(3) == False\n"
        "assert is_even(0) == True\n"
    )

    reward = reward_func(
        prompt=[{"role": "user", "content": "Write is_even(n)."}],
        completion=[{"role": "assistant", "content": fenced_completion}],
        answer="",
        tests=tests,
        state={},
    )
    assert reward == 1.0, (
        f"Expected 1.0 (no injection, fenced correct code), got {reward}. "
        "With fraction=0.0 the injector is off; extraction must score the code."
    )


# ---------------------------------------------------------------------------
# Test 4 — state dict records the reward (observability contract unchanged)
# ---------------------------------------------------------------------------

def test_state_dict_reward_recorded_after_fenced_completion():
    """After scoring, state['reward_funcs_results'] must contain the reward.

    This contract existed before the fix; confirm it is not broken by the
    fence-extraction change.

    FAILS NOW (indirectly): if reward is 0.0 due to missing extraction,
    the state will record 0.0 — but we assert 1.0 is recorded.
    """
    import os
    os.environ.setdefault("RLOX_NO_MBPP", "1")

    env = rlox_verify.load_environment(
        rollout_backend="in_loop",
        per_sample_timeout_secs=5.0,
    )
    reward_func = _get_reward_func(env)

    fenced_completion = (
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    tests = "assert add(1, 2) == 3\n"
    state: dict[str, Any] = {}

    reward = reward_func(
        prompt=[{"role": "user", "content": "Write add(a, b)."}],
        completion=[{"role": "assistant", "content": fenced_completion}],
        answer="",
        tests=tests,
        state=state,
    )

    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    rfr = state.get("reward_funcs_results", {})
    assert isinstance(rfr, dict), f"state['reward_funcs_results'] must be a dict, got {type(rfr)}"
    assert len(rfr) >= 1, "state['reward_funcs_results'] must contain at least one entry"
    recorded_reward = list(rfr.values())[0]
    assert recorded_reward == 1.0, (
        f"state['reward_funcs_results'] must record reward 1.0, got {recorded_reward}"
    )
