"""conftest.py for tests/agentic/

Puts python/ on sys.path so that ``import rlox_agent.X`` resolves directly
to the standalone package (python/rlox_agent/), without going through
python/rlox/__init__.py (which would pull in _rlox_core and torch, neither
of which is present in the light venv on wk-system).

Also injects benchmarks/agentic/ so that ``import run_benchmark`` works
as a top-level import (sweep driver — Step 6d).
"""
import sys
from pathlib import Path

# Resolve repo root relative to this conftest — works both locally and on wk-system
_repo_root = Path(__file__).parent.parent.parent
_python_dir = _repo_root / "python"
_python_str = str(_python_dir)
if _python_str not in sys.path:
    sys.path.insert(0, _python_str)

# Expose benchmarks/agentic/ for the sweep driver import
_benchmarks_agentic_dir = _repo_root / "benchmarks" / "agentic"
_benchmarks_agentic_str = str(_benchmarks_agentic_dir)
if _benchmarks_agentic_str not in sys.path:
    sys.path.insert(0, _benchmarks_agentic_str)
