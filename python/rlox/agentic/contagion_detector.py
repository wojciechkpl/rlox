"""Backward-compat shim. Canonical implementation is in rlox_agent.contagion_detector."""
from rlox_agent.contagion_detector import *  # noqa: F401, F403
from rlox_agent.contagion_detector import (  # noqa: F401
    ContagionDetector,
    ContagionEvent,
    ContagionReport,
)
