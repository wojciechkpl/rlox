"""Guards against repo-layout mistakes that break CI collection.

Each test here encodes a failure that actually reached CI. They are cheap
(pure filesystem / AST checks, no imports of the code under test) and run in the
default `-m "not slow"` selection so they fail fast.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# tomllib is stdlib from 3.11; tomli is the 3.10 backport (see the dev extra).
try:
    from tomllib import load as _toml_load
except ModuleNotFoundError:  # Python 3.10
    _toml_load = pytest.importorskip(
        "tomli", reason="needs a TOML reader: stdlib tomllib is 3.11+"
    ).load

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that never contain project source.
_SKIP_DIRS = {
    ".git", ".venv", "venv", "target", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", "site", "book", "dist", "dist-ci",
    "_site", "public",
}


def _iter_py_files() -> list[Path]:
    out: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if _SKIP_DIRS.isdisjoint(part for part in path.relative_to(REPO_ROOT).parts):
            out.append(path)
    return out


def _defines_pytest_fixtures_or_hooks(path: Path) -> bool:
    """True if the module looks like a genuine pytest conftest."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("pytest_"):
                return True
            for dec in node.decorator_list:
                # @pytest.fixture / @pytest.fixture(...) / @fixture
                target = dec.func if isinstance(dec, ast.Call) else dec
                name = (
                    target.attr if isinstance(target, ast.Attribute)
                    else target.id if isinstance(target, ast.Name)
                    else ""
                )
                if name == "fixture":
                    return True
    return False


class TestConftestHygiene:
    """`conftest.py` is a pytest-reserved name, not a place for shared helpers.

    Regression: `benchmarks/conftest.py` held BenchmarkResult/ComparisonResult/
    timed_run and was imported as `from conftest import ...`. pytest imports every
    conftest.py under rootdir as the top-level module `conftest`, so whichever one
    it loaded first won `sys.modules["conftest"]`. `tests/agentic/conftest.py` got
    there first and five `tests/python/test_bench_*.py` modules died at collection
    with `ImportError: cannot import name 'BenchmarkResult' from 'conftest'`,
    taking the entire suite down with them. The helpers now live in
    `benchmarks/harness.py`.
    """

    def test_every_conftest_actually_defines_fixtures_or_hooks(self) -> None:
        offenders = []
        for path in _iter_py_files():
            if path.name != "conftest.py":
                continue
            if not _defines_pytest_fixtures_or_hooks(path):
                # sys.path manipulation only is a legitimate conftest use.
                src = path.read_text(encoding="utf-8")
                if "sys.path" in src:
                    continue
                offenders.append(path.relative_to(REPO_ROOT))
        assert not offenders, (
            "These conftest.py files define no fixtures/hooks, so they are really "
            "shared-helper modules wearing a pytest-reserved filename. pytest "
            "imports all of them as the single top-level module 'conftest', so they "
            "shadow each other and break `from conftest import ...`. Rename them "
            f"(e.g. harness.py) and update importers: {offenders}"
        )

    def test_nothing_imports_from_a_module_named_conftest(self) -> None:
        offenders = []
        for path in _iter_py_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "conftest":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "conftest":
                            offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        assert not offenders, (
            "`conftest` is not a stable module name to import from — pytest owns it "
            "and which file wins depends on collection order. Import the helper "
            f"module by its real name instead: {offenders}"
        )


class TestCiReferencedExtrasExist:
    """CI installs `-e ".[all]"`; pip only *warns* on an unknown extra, so a
    missing one silently installs nothing and surfaces later as a confusing
    ImportError rather than a clear failure."""

    def test_extras_referenced_by_workflows_are_declared(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            declared = set(_toml_load(fh)["project"]["optional-dependencies"])

        referenced: set[str] = set()
        for wf in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
            for match in re.finditer(r'-e\s+"?\.\[([a-zA-Z0-9,_-]+)\]"?', wf.read_text()):
                referenced.update(part.strip() for part in match.group(1).split(","))

        assert referenced, "expected at least one `pip install -e .[...]` in the workflows"

        missing = sorted(referenced - declared)
        assert not missing, (
            "A GitHub workflow installs extras that pyproject.toml does not declare. "
            "pip only warns on an unknown extra, so this fails later and less "
            f"clearly. Missing: {missing}; declared: {sorted(declared)}"
        )
