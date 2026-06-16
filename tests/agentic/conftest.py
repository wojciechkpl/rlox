"""conftest.py for tests/agentic/

Critical: inject python/rlox/agentic/ directly onto sys.path so that
``import adversarial_corpus`` and ``import verifiers_adapter`` work as
top-level imports WITHOUT going through python/rlox/__init__.py (which would
pull in rlox._rlox_core and torch, neither of which is present in this env).
"""
import sys
import os
from pathlib import Path

# Resolve repo root relative to this conftest — works both locally and on wk-system
_repo_root = Path(__file__).parent.parent.parent
_agentic_dir = _repo_root / "python" / "rlox" / "agentic"
_agentic_str = str(_agentic_dir)
if _agentic_str not in sys.path:
    sys.path.insert(0, _agentic_str)
