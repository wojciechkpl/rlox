"""RED tests: extract_python_code must live in rlox_agent.verifiers_adapter.

Root cause being tested
-----------------------
The function currently exists ONLY in benchmarks/agentic/trl_grpo_run.py.
The rlox_verify env reward function calls ``extract_text`` (not
``extract_python_code``) so fenced-markdown completions are executed raw
and score 0.0.

Fix contract
------------
1. ``extract_python_code`` must be importable from ``rlox_agent.verifiers_adapter``
   (the shared single-source module).
2. Its extraction logic must match the spec already implemented in trl_grpo_run.py:
   a. Last ```python ... ``` block wins.
   b. Fallback: last ``` ... ``` block.
   c. Fallback: from the first ``def``/``import``/``from`` line to end-of-string.
   d. Fallback: return text unchanged.

These tests require NO ``verifiers`` install and NO GPU — pure stdlib + pytest.

ALL tests in this file FAIL NOW because:
  ``ImportError: cannot import name 'extract_python_code' from
  'rlox_agent.verifiers_adapter'``
"""
from __future__ import annotations

import pytest

# THE IMPORT UNDER TEST — this is what must fail RED until the function is
# moved to verifiers_adapter.  The test_code_extractor.py file already passes
# against trl_grpo_run; this new file pins the SHARED-LOCATION contract.
from rlox_agent.verifiers_adapter import extract_python_code  # noqa: E402


# ---------------------------------------------------------------------------
# A) Shared-location import: basic smoke
# ---------------------------------------------------------------------------

class TestSharedLocationImport:
    """Confirm the function is callable from the shared module after import."""

    def test_callable(self):
        assert callable(extract_python_code)

    def test_returns_str(self):
        result = extract_python_code("def f(): pass")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# B) Fenced ```python ... ``` blocks
# ---------------------------------------------------------------------------

class TestPythonFencedBlock:
    """Priority 1: last ```python ... ``` block is extracted and returned."""

    def test_single_python_fence_stripped(self):
        text = (
            "Here is the answer:\n\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n\n"
            "Let me know if you need anything else."
        )
        result = extract_python_code(text)
        assert "def add(a, b):" in result
        assert "Let me know" not in result
        assert "```" not in result

    def test_last_python_fence_preferred_over_first(self):
        """When the model emits reasoning + solution, take the LAST block."""
        text = (
            "First attempt (wrong):\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a - b\n"
            "```\n"
            "Actually, correcting myself:\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n"
        )
        result = extract_python_code(text)
        assert "return a + b" in result
        assert "return a - b" not in result

    def test_python_fence_with_long_prose_preamble(self):
        preamble = (
            "I need to think about this problem carefully. "
            "The user wants a function that adds two numbers. "
            "This is a straightforward problem:\n\n"
        )
        code = "def add(a, b):\n    return a + b\n"
        text = preamble + "```python\n" + code + "```"
        result = extract_python_code(text)
        assert "def add(a, b):" in result
        assert "think about" not in result

    def test_python_fence_content_is_stripped(self):
        """Leading/trailing whitespace inside the block is stripped."""
        text = "```python\n\n   def f():\n       return 1\n\n```"
        result = extract_python_code(text)
        assert result.startswith("def f():")


# ---------------------------------------------------------------------------
# C) Generic fenced ``` ... ``` blocks (priority 2)
# ---------------------------------------------------------------------------

class TestGenericFencedBlock:
    """Priority 2: when no ```python block, take the last ``` block."""

    def test_generic_fence_extracted(self):
        text = (
            "```\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "```\n"
        )
        result = extract_python_code(text)
        assert "def multiply(a, b):" in result
        assert "```" not in result

    def test_last_generic_fence_preferred(self):
        text = (
            "```\n"
            "x = 1\n"
            "```\n"
            "Better version:\n"
            "```\n"
            "def final():\n"
            "    return 99\n"
            "```"
        )
        result = extract_python_code(text)
        assert "def final():" in result
        assert "x = 1" not in result

    def test_python_fence_takes_priority_over_generic_fence(self):
        """A ```python block must beat a plain ``` block, regardless of order."""
        text = (
            "```\n"
            "generic block\n"
            "```\n"
            "```python\n"
            "def preferred():\n"
            "    pass\n"
            "```"
        )
        result = extract_python_code(text)
        assert "def preferred():" in result
        assert "generic block" not in result


# ---------------------------------------------------------------------------
# D) Bare def / import fallback (priority 3, no fences at all)
# ---------------------------------------------------------------------------

class TestBareDefImportFallback:
    """Priority 3: when no fences, slice from the first def/import/from line."""

    def test_prose_then_def_extracts_from_def(self):
        text = (
            "The problem is simple.\n"
            "def solution(n):\n"
            "    return n * 2\n"
        )
        result = extract_python_code(text)
        assert result.startswith("def solution(n):")
        assert "The problem" not in result

    def test_prose_then_import_extracts_from_import(self):
        text = (
            "I will use the math module.\n"
            "import math\n\n"
            "def compute(x):\n"
            "    return math.sqrt(x)\n"
        )
        result = extract_python_code(text)
        assert result.startswith("import math")
        assert "I will use" not in result

    def test_prose_then_from_import_extracts_from_from(self):
        text = (
            "My solution:\n"
            "from typing import List\n\n"
            "def process(items: List[int]) -> int:\n"
            "    return sum(items)\n"
        )
        result = extract_python_code(text)
        assert result.startswith("from typing import List")


# ---------------------------------------------------------------------------
# E) No-op fallback (priority 4): text returned unchanged
# ---------------------------------------------------------------------------

class TestNoOpFallback:
    """Priority 4: when no fence or def/import marker, return text as-is."""

    def test_bare_python_code_unchanged(self):
        """A completion that IS already raw Python passes straight through."""
        code = "def add(a, b):\n    return a + b\n"
        result = extract_python_code(code)
        assert "def add(a, b):" in result

    def test_empty_string_returns_empty_string(self):
        assert extract_python_code("") == ""

    def test_pure_prose_returns_prose_unchanged(self):
        prose = "This is a description without any code."
        result = extract_python_code(prose)
        assert result == prose


# ---------------------------------------------------------------------------
# F) Parametrised edge-case table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_text,expected_substring,absent_substring", [
    # Multiple ```python blocks → last wins
    (
        "```python\ndef wrong(): pass\n```\n```python\ndef right(): pass\n```",
        "def right():",
        "def wrong():",
    ),
    # ```python preferred over ```
    (
        "```\ngeneric\n```\n```python\ndef python_wins(): pass\n```",
        "def python_wins():",
        "generic",
    ),
    # Bare def with multiline preamble
    (
        "Step 1.\nStep 2.\ndef answer():\n    return 42\n",
        "def answer():",
        "Step 1.",
    ),
    # Already bare Python — no extraction needed
    (
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):",
        None,  # nothing to check for absence
    ),
])
def test_extraction_parametrised(
    input_text: str,
    expected_substring: str,
    absent_substring: str | None,
):
    result = extract_python_code(input_text)
    assert expected_substring in result, (
        f"Expected {expected_substring!r} in result, got: {result!r}"
    )
    if absent_substring is not None:
        assert absent_substring not in result, (
            f"Unexpected {absent_substring!r} found in result: {result!r}"
        )
