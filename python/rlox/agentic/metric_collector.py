"""Backward-compat shim. Canonical implementation is in rlox_agent.metric_collector."""
from rlox_agent.metric_collector import *  # noqa: F401, F403
from rlox_agent.metric_collector import (  # noqa: F401
    MetricCollector,
    _parse_nvidia_smi_line,
    _run_nvidia_smi,
    _default_gpu_sampler,
)
