"""Tests for extract_python_code (FIX 2).

The helper must strip reasoning preambles emitted by Qwen3 so the reward
function receives an executable Python snippet rather than a prose block
that contains code somewhere inside it.

Extraction priority:
1. Last ```python ... ``` fenced block.
2. Last ``` ... ``` fenced block (language-agnostic).
3. From the first ``def ``/``import ``/``from `` line to end-of-string.
4. Unchanged (no preamble detected).

Adversarial samples are NOT passed through this function — they run verbatim.
"""
from __future__ import annotations

import pytest

# benchmarks/agentic/ is on sys.path via conftest.py
from trl_grpo_run import extract_python_code


# ---------------------------------------------------------------------------
# A) Fenced ```python ... ``` block
# ---------------------------------------------------------------------------

class TestFencedPythonBlock:
    def test_single_python_block_extracted(self):
        text = (
            "Sure, here is the solution:\n\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n\n"
            "Hope that helps!"
        )
        result = extract_python_code(text)
        assert "def add(a, b):" in result
        assert "Hope that helps!" not in result

    def test_last_python_block_preferred_over_first(self):
        """When multiple fenced python blocks exist, return the last one."""
        text = (
            "First attempt:\n"
            "```python\n"
            "def wrong():\n"
            "    pass\n"
            "```\n"
            "Actually, here is the correct one:\n"
            "```python\n"
            "def correct():\n"
            "    return 42\n"
            "```\n"
        )
        result = extract_python_code(text)
        assert "def correct():" in result
        assert "def wrong():" not in result

    def test_python_block_content_stripped(self):
        text = "```python\n   def f():\n       return 1\n   \n```"
        result = extract_python_code(text)
        assert result.startswith("def f():")

    def test_python_block_with_preamble(self):
        preamble = (
            "I need to think step by step.\n"
            "First, I observe that the problem requires sorting.\n"
            "Let me write the function:\n\n"
        )
        code = "def solve(xs):\n    return sorted(xs)\n"
        text = preamble + "```python\n" + code + "```"
        result = extract_python_code(text)
        assert "def solve(xs):" in result
        assert "I need to think" not in result


# ---------------------------------------------------------------------------
# B) Generic fenced block (``` ... ```)
# ---------------------------------------------------------------------------

class TestFencedGenericBlock:
    def test_generic_block_extracted_when_no_python_tag(self):
        text = (
            "Here is the code:\n"
            "```\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "```\n"
        )
        result = extract_python_code(text)
        assert "def multiply(a, b):" in result

    def test_last_generic_block_preferred(self):
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

    def test_python_block_takes_precedence_over_generic(self):
        """A ```python block must be preferred over a plain ``` block."""
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
# C) Bare def / import fallback (no fences)
# ---------------------------------------------------------------------------

class TestBareDefFallback:
    def test_bare_def_extracted(self):
        text = (
            "Let me think about this carefully.\n"
            "The answer involves a loop.\n"
            "def solution(n):\n"
            "    return n * 2\n"
        )
        result = extract_python_code(text)
        assert result.startswith("def solution(n):")
        assert "Let me think" not in result

    def test_bare_import_extracted(self):
        text = (
            "I will use the math module.\n"
            "import math\n"
            "\n"
            "def compute(x):\n"
            "    return math.sqrt(x)\n"
        )
        result = extract_python_code(text)
        assert result.startswith("import math")
        assert "I will use" not in result

    def test_bare_from_import_extracted(self):
        text = (
            "Here is my solution:\n"
            "from typing import List\n"
            "\n"
            "def process(items: List[int]) -> int:\n"
            "    return sum(items)\n"
        )
        result = extract_python_code(text)
        assert result.startswith("from typing import List")

    def test_multiline_preamble_then_def(self):
        text = "\n".join([
            "I need to solve this step by step.",
            "1. First I consider the base case.",
            "2. Then I handle the recursive case.",
            "Here is my implementation:",
            "def fib(n):",
            "    if n <= 1:",
            "        return n",
            "    return fib(n-1) + fib(n-2)",
        ])
        result = extract_python_code(text)
        assert "def fib(n):" in result
        assert "step by step" not in result


# ---------------------------------------------------------------------------
# D) No-op path (already clean code or no markers)
# ---------------------------------------------------------------------------

class TestNoOpPath:
    def test_bare_code_unchanged(self):
        code = "def add(a, b):\n    return a + b\n"
        result = extract_python_code(code)
        assert "def add(a, b):" in result

    def test_empty_string_unchanged(self):
        result = extract_python_code("")
        assert result == ""

    def test_prose_only_unchanged(self):
        """Pure prose with no code markers must come back as-is."""
        prose = "This is a description without any code."
        result = extract_python_code(prose)
        assert result == prose


# ---------------------------------------------------------------------------
# E) Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_code_block_with_trailing_text_stripped(self):
        text = "```python\ndef f():\n    return 1\n```\nThis is trailing text."
        result = extract_python_code(text)
        assert "trailing text" not in result

    def test_only_def_line_no_body(self):
        text = "Explanation here.\ndef minimal(): pass"
        result = extract_python_code(text)
        assert "def minimal():" in result

    @pytest.mark.parametrize("lang_tag", ["python", "py", "Python"])
    def test_various_python_fence_tags(self, lang_tag: str):
        """Only 'python' tag is matched by the regex; others fall through to generic."""
        text = f"```{lang_tag}\ndef f(): pass\n```"
        result = extract_python_code(text)
        # At minimum the def line must survive.
        assert "def f():" in result
