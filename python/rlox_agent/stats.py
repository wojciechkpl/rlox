# python/rlox_agent/stats.py
#
# Python mirror of `crates/rlox-sandbox/src/stats.rs :: BackendStats`.
#
# Contract invariants (must be kept in sync with the Rust struct):
#   - All JSON field names are snake_case and identical on both sides.
#   - `time_to_contain_secs` is a list of floats (Vec<f64> in Rust).
#   - Integer fields (`u32`, `u64`) deserialize as Python `int`.
#   - Float fields (`f32`, `f64`) deserialize as Python `float`.
#
# Import constraints: stdlib only — no torch, no pydantic, no pytest.
# This module must be importable with `python3` alone.
import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class BackendStats:
    # --- P1 throughput fields ---
    batch_wall_secs: float = 0.0
    rollouts_completed: int = 0
    rollouts_per_sec: float = 0.0
    tool_calls_per_sec: float = 0.0

    # --- P3 containment fields ---
    adversarial_injected: int = 0
    adversarial_contained: int = 0
    contagion_events: int = 0
    # Sandbox SetupError events — NOT contagion escapes; infrastructure failures only.
    setup_error_events: int = 0
    time_to_contain_secs: List[float] = field(default_factory=list)
    cgroup_freeze_events: int = 0
    cgroup_kill_events: int = 0
    oom_kill_events: int = 0

    # --- GPU idle attribution ---
    gpu_idle_attributable_to_hang_secs: float = 0.0
    step_index: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "BackendStats":
        """Construct from a plain dict (e.g. parsed JSON).

        Unknown keys are silently ignored so the Python side is forward-compatible
        with new Rust fields added in future cycles.
        """
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> "BackendStats":
        """Construct from a JSON string produced by the Rust rollout server."""
        return cls.from_dict(json.loads(s))
