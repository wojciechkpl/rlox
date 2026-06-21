"""Backward-compat shim. Canonical implementation is in rlox_agent.config."""
from rlox_agent.config import *  # noqa: F401, F403
from rlox_agent.config import (  # noqa: F401
    BenchmarkConfig,
    ConfigValidationError,
    load_config,
    validate_config,
)
