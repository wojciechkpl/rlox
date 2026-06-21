"""Coverage / characterization tests for rlox_agent.stats.BackendStats.

Locks the observable contract of the BackendStats dataclass that mirrors the
Rust ``BackendStats`` struct serialised by the rlox-sandbox crate.

Contract:
  - BackendStats is a dataclass with the exact snake_case field names below.
  - BackendStats.from_dict(d) constructs from a plain dict; unknown keys are
    silently ignored (forward-compat guarantee documented in the source).
  - BackendStats.from_json(s) parses a JSON string and delegates to from_dict.
  - All numeric defaults are 0 / 0.0; time_to_contain_secs defaults to [].
  - Integer fields stay int; float fields stay float.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from rlox_agent.stats import BackendStats


# ---------------------------------------------------------------------------
# Representative fixture dict — non-trivial values so every field is tested.
# ---------------------------------------------------------------------------

_REPR_DICT: dict = {
    "batch_wall_secs": 3.14,
    "rollouts_completed": 7,
    "rollouts_per_sec": 2.5,
    "tool_calls_per_sec": 1.25,
    "adversarial_injected": 10,
    "adversarial_contained": 8,
    "contagion_events": 2,
    "setup_error_events": 3,
    "time_to_contain_secs": [0.1, 0.25, 0.5],
    "cgroup_freeze_events": 4,
    "cgroup_kill_events": 1,
    "oom_kill_events": 0,
    "gpu_idle_attributable_to_hang_secs": 12.3,
    "step_index": 5,
}


# ---------------------------------------------------------------------------
# A) Dataclass structure
# ---------------------------------------------------------------------------


class TestBackendStatsDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(BackendStats)

    def test_can_construct_with_no_args(self):
        """Dataclass must be instantiable with zero arguments (all fields have defaults)."""
        s = BackendStats()
        assert s is not None

    @pytest.mark.parametrize(
        "field_name",
        [
            "batch_wall_secs",
            "rollouts_completed",
            "rollouts_per_sec",
            "tool_calls_per_sec",
            "adversarial_injected",
            "adversarial_contained",
            "contagion_events",
            "setup_error_events",
            "time_to_contain_secs",
            "cgroup_freeze_events",
            "cgroup_kill_events",
            "oom_kill_events",
            "gpu_idle_attributable_to_hang_secs",
            "step_index",
        ],
    )
    def test_has_required_field(self, field_name: str):
        names = {f.name for f in dataclasses.fields(BackendStats)}
        assert field_name in names, f"BackendStats is missing field '{field_name}'"

    def test_time_to_contain_secs_default_is_empty_list(self):
        s = BackendStats()
        assert s.time_to_contain_secs == []
        assert isinstance(s.time_to_contain_secs, list)

    def test_mutable_defaults_are_independent(self):
        """Each instance must get its own list — not a shared mutable default."""
        a = BackendStats()
        b = BackendStats()
        a.time_to_contain_secs.append(1.0)
        assert b.time_to_contain_secs == [], (
            "time_to_contain_secs default is shared across instances — "
            "use field(default_factory=list)"
        )


# ---------------------------------------------------------------------------
# B) Numeric default values
# ---------------------------------------------------------------------------


class TestBackendStatsDefaults:
    def test_batch_wall_secs_default(self):
        assert BackendStats().batch_wall_secs == 0.0

    def test_rollouts_completed_default(self):
        assert BackendStats().rollouts_completed == 0

    def test_rollouts_per_sec_default(self):
        assert BackendStats().rollouts_per_sec == 0.0

    def test_tool_calls_per_sec_default(self):
        assert BackendStats().tool_calls_per_sec == 0.0

    def test_adversarial_injected_default(self):
        assert BackendStats().adversarial_injected == 0

    def test_adversarial_contained_default(self):
        assert BackendStats().adversarial_contained == 0

    def test_contagion_events_default(self):
        assert BackendStats().contagion_events == 0

    def test_setup_error_events_default(self):
        assert BackendStats().setup_error_events == 0

    def test_cgroup_freeze_events_default(self):
        assert BackendStats().cgroup_freeze_events == 0

    def test_cgroup_kill_events_default(self):
        assert BackendStats().cgroup_kill_events == 0

    def test_oom_kill_events_default(self):
        assert BackendStats().oom_kill_events == 0

    def test_gpu_idle_attributable_to_hang_secs_default(self):
        assert BackendStats().gpu_idle_attributable_to_hang_secs == 0.0

    def test_step_index_default(self):
        assert BackendStats().step_index == 0


# ---------------------------------------------------------------------------
# C) from_dict — happy path round-trip
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_returns_backend_stats_instance(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s, BackendStats)

    def test_batch_wall_secs_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.batch_wall_secs == pytest.approx(3.14)

    def test_rollouts_completed_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.rollouts_completed == 7

    def test_rollouts_per_sec_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.rollouts_per_sec == pytest.approx(2.5)

    def test_tool_calls_per_sec_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.tool_calls_per_sec == pytest.approx(1.25)

    def test_adversarial_injected_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.adversarial_injected == 10

    def test_adversarial_contained_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.adversarial_contained == 8

    def test_contagion_events_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.contagion_events == 2

    def test_setup_error_events_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.setup_error_events == 3

    def test_time_to_contain_secs_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.time_to_contain_secs == [0.1, 0.25, 0.5]

    def test_cgroup_freeze_events_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.cgroup_freeze_events == 4

    def test_cgroup_kill_events_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.cgroup_kill_events == 1

    def test_oom_kill_events_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.oom_kill_events == 0

    def test_gpu_idle_attributable_to_hang_secs_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.gpu_idle_attributable_to_hang_secs == pytest.approx(12.3)

    def test_step_index_populated(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert s.step_index == 5

    def test_empty_dict_gives_all_defaults(self):
        s = BackendStats.from_dict({})
        defaults = BackendStats()
        # Verify every scalar field equals the default.
        for f in dataclasses.fields(BackendStats):
            if f.name == "time_to_contain_secs":
                continue  # list — tested separately
            assert getattr(s, f.name) == getattr(defaults, f.name), (
                f"from_dict({{}}) should produce default for field '{f.name}'"
            )

    def test_empty_dict_time_to_contain_secs_is_empty_list(self):
        s = BackendStats.from_dict({})
        assert s.time_to_contain_secs == []

    def test_partial_dict_fills_remaining_with_defaults(self):
        """Only the keys present in the dict should be set; rest must stay default."""
        s = BackendStats.from_dict({"contagion_events": 99})
        assert s.contagion_events == 99
        assert s.rollouts_completed == 0
        assert s.time_to_contain_secs == []


# ---------------------------------------------------------------------------
# D) from_dict — forward-compat: unknown extra keys are silently ignored
# ---------------------------------------------------------------------------


class TestFromDictForwardCompat:
    def test_unknown_key_is_ignored(self):
        """from_dict must not raise when an unknown key is present (forward-compat)."""
        d = dict(_REPR_DICT, _future_rust_field="new_value", another_new_field=42)
        s = BackendStats.from_dict(d)  # must not raise
        assert isinstance(s, BackendStats)
        assert s.contagion_events == 2  # known fields unaffected

    def test_only_unknown_keys_gives_defaults(self):
        s = BackendStats.from_dict({"completely_unknown": True, "also_unknown": 999})
        assert s.contagion_events == 0
        assert s.rollouts_completed == 0

    def test_mix_of_known_and_unknown_sets_only_known(self):
        s = BackendStats.from_dict({"rollouts_completed": 5, "brand_new_field": "X"})
        assert s.rollouts_completed == 5
        assert not hasattr(s, "brand_new_field")


# ---------------------------------------------------------------------------
# E) from_json — JSON string round-trip
# ---------------------------------------------------------------------------


class TestFromJson:
    def test_returns_backend_stats_instance(self):
        s = BackendStats.from_json(json.dumps(_REPR_DICT))
        assert isinstance(s, BackendStats)

    def test_contagion_events_from_json(self):
        s = BackendStats.from_json(json.dumps(_REPR_DICT))
        assert s.contagion_events == 2

    def test_time_to_contain_secs_from_json(self):
        s = BackendStats.from_json(json.dumps(_REPR_DICT))
        assert s.time_to_contain_secs == pytest.approx([0.1, 0.25, 0.5])

    def test_setup_error_events_from_json(self):
        s = BackendStats.from_json(json.dumps(_REPR_DICT))
        assert s.setup_error_events == 3

    def test_cgroup_freeze_events_from_json(self):
        s = BackendStats.from_json(json.dumps(_REPR_DICT))
        assert s.cgroup_freeze_events == 4

    def test_oom_kill_events_from_json(self):
        s = BackendStats.from_json(json.dumps(_REPR_DICT))
        assert s.oom_kill_events == 0

    def test_batch_wall_secs_from_json(self):
        s = BackendStats.from_json(json.dumps(_REPR_DICT))
        assert s.batch_wall_secs == pytest.approx(3.14)

    def test_step_index_from_json(self):
        s = BackendStats.from_json(json.dumps(_REPR_DICT))
        assert s.step_index == 5

    def test_json_empty_object(self):
        s = BackendStats.from_json("{}")
        assert s.contagion_events == 0
        assert s.time_to_contain_secs == []

    def test_json_with_unknown_key_ignored(self):
        payload = dict(_REPR_DICT, future_server_field="xyz")
        s = BackendStats.from_json(json.dumps(payload))  # must not raise
        assert s.contagion_events == 2

    def test_json_time_to_contain_secs_empty_list(self):
        payload = dict(_REPR_DICT, time_to_contain_secs=[])
        s = BackendStats.from_json(json.dumps(payload))
        assert s.time_to_contain_secs == []

    def test_from_json_delegates_to_from_dict(self):
        """from_json(s) must produce the same result as from_dict(json.loads(s))."""
        json_str = json.dumps(_REPR_DICT)
        via_json = BackendStats.from_json(json_str)
        via_dict = BackendStats.from_dict(json.loads(json_str))
        assert via_json == via_dict


# ---------------------------------------------------------------------------
# F) Type preservation
# ---------------------------------------------------------------------------


class TestFieldTypes:
    """Integer fields must remain int; float fields must remain float."""

    def test_rollouts_completed_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.rollouts_completed, int)

    def test_contagion_events_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.contagion_events, int)

    def test_setup_error_events_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.setup_error_events, int)

    def test_cgroup_freeze_events_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.cgroup_freeze_events, int)

    def test_cgroup_kill_events_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.cgroup_kill_events, int)

    def test_oom_kill_events_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.oom_kill_events, int)

    def test_adversarial_injected_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.adversarial_injected, int)

    def test_adversarial_contained_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.adversarial_contained, int)

    def test_step_index_is_int(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.step_index, int)

    def test_batch_wall_secs_is_float(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.batch_wall_secs, float)

    def test_rollouts_per_sec_is_float(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.rollouts_per_sec, float)

    def test_gpu_idle_attributable_to_hang_secs_is_float(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.gpu_idle_attributable_to_hang_secs, float)

    def test_time_to_contain_secs_is_list(self):
        s = BackendStats.from_dict(_REPR_DICT)
        assert isinstance(s.time_to_contain_secs, list)
