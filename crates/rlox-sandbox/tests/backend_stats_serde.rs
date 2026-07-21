/// Step 3d — BackendStats serialization contract (Cycle 1 RED tests, updated Cycle 3)
///
/// Covers:
///   1. Rust serde round-trip: serialize → deserialize → assert equal.
///   2. JSON field name contract: assert snake_case field names are present
///      in the serialized JSON object.
///   3. Rust → JSON → Python round-trip (Linux only): invoke `python3` via
///      `std::process::Command` and load `python/rlox/agentic/stats.py`
///      directly via `importlib.util.spec_from_file_location` (bypasses the
///      heavy rlox package `__init__`), parse the JSON, and assert that
///      specific field values echoed to stdout match the Rust-side values.
///
/// Cycle-3 additions (RED until the implementer adds `setup_error_events`):
///   - `make_non_trivial_stats` includes a non-zero `setup_error_events` value.
///   - `test_backend_stats_serde_round_trip` will FAIL to compile once the
///     struct literal is updated, because `setup_error_events` does not yet
///     exist on `BackendStats`.
///   - `test_backend_stats_setup_error_events_survives_round_trip` explicitly
///     sets `setup_error_events` and asserts it survives JSON round-trip.
///   - `test_backend_stats_json_field_names_are_snake_case` now requires
///     `"setup_error_events"` in the serialized JSON.
///   - The Python round-trip test now prints and asserts `setup_error_events`.
///
/// All tests FAIL until `BackendStats` has the `setup_error_events: u32` field
/// added to both the Rust struct and the Python `BackendStats` dataclass.
use rlox_sandbox::BackendStats;

// ---------------------------------------------------------------------------
// Helper: build a BackendStats with every interesting field non-zero/non-empty.
//
// Cycle-3 change: `setup_error_events: 3` is added.
// COMPILE ERROR until the implementer adds `setup_error_events: u32` to
// the `BackendStats` struct in `src/stats.rs`.
// ---------------------------------------------------------------------------

fn make_non_trivial_stats() -> BackendStats {
    BackendStats {
        batch_wall_secs: 1.234,
        rollouts_completed: 8,
        rollouts_per_sec: 6.5,
        tool_calls_per_sec: 6.5,
        adversarial_injected: 2,
        adversarial_contained: 2,
        contagion_events: 1,
        // setup_error_events: sandbox SetupError path fires, NOT a containment escape.
        // Must be counted separately from contagion_events (a setup failure is not an escape).
        setup_error_events: 3,
        time_to_contain_secs: vec![0.042, 0.137],
        cgroup_freeze_events: 2,
        cgroup_kill_events: 2,
        oom_kill_events: 1,
        gpu_idle_attributable_to_hang_secs: 0.55,
        step_index: 42,
    }
}

// ---------------------------------------------------------------------------
// Test 3d-1: Serde round-trip (Rust serialize → deserialize → PartialEq).
//
// Fails NOW because `BackendStats` is missing the `setup_error_events` field —
// `make_non_trivial_stats()` will not compile until the field is added.
// ---------------------------------------------------------------------------

#[test]
fn test_backend_stats_serde_round_trip() {
    let original = make_non_trivial_stats();
    let json = serde_json::to_string(&original).expect("BackendStats must serialize to JSON");
    let recovered: BackendStats =
        serde_json::from_str(&json).expect("BackendStats must deserialize from JSON");
    assert_eq!(
        original, recovered,
        "BackendStats must round-trip through JSON without data loss"
    );
}

// ---------------------------------------------------------------------------
// Test 3d-2: contagion_events survives round-trip with a non-zero value.
//
// This is the AC-5 critical field — explicitly asserted to guard against
// accidental omission or default-to-zero behaviour.
// ---------------------------------------------------------------------------

#[test]
fn test_backend_stats_contagion_events_survives_round_trip() {
    let stats = BackendStats {
        contagion_events: 3,
        ..BackendStats::default()
    };
    let json = serde_json::to_string(&stats).unwrap();
    let recovered: BackendStats = serde_json::from_str(&json).unwrap();
    assert_eq!(
        recovered.contagion_events, 3,
        "contagion_events must survive serialization round-trip"
    );
}

// ---------------------------------------------------------------------------
// Test 3d-3: time_to_contain_secs (Vec<f64>) survives round-trip.
//
// AC-6 depends on this vector being preserved exactly.
// ---------------------------------------------------------------------------

#[test]
fn test_backend_stats_time_to_contain_secs_round_trip() {
    let expected = vec![0.042f64, 0.137, 1.001];
    let stats = BackendStats {
        time_to_contain_secs: expected.clone(),
        ..BackendStats::default()
    };
    let json = serde_json::to_string(&stats).unwrap();
    let recovered: BackendStats = serde_json::from_str(&json).unwrap();
    assert_eq!(
        recovered.time_to_contain_secs, expected,
        "time_to_contain_secs must survive serialization as a Vec<f64>"
    );
}

// ---------------------------------------------------------------------------
// Test 3d-4: cgroup kill/freeze/oom counters survive round-trip.
// ---------------------------------------------------------------------------

#[test]
fn test_backend_stats_cgroup_and_oom_counters_round_trip() {
    let stats = BackendStats {
        cgroup_freeze_events: 5,
        cgroup_kill_events: 5,
        oom_kill_events: 2,
        ..BackendStats::default()
    };
    let json = serde_json::to_string(&stats).unwrap();
    let recovered: BackendStats = serde_json::from_str(&json).unwrap();
    assert_eq!(recovered.cgroup_freeze_events, 5);
    assert_eq!(recovered.cgroup_kill_events, 5);
    assert_eq!(recovered.oom_kill_events, 2);
}

// ---------------------------------------------------------------------------
// Test 3d-5: JSON field names are snake_case and all P3 keys are present.
//
// Cycle-3 change: `"setup_error_events"` added to required_keys.
//
// Asserts the wire-contract field names so the Python side cannot silently
// mismatch.  This is a whitebox test on the serialized bytes, not the struct.
//
// FAILS NOW: `make_non_trivial_stats()` does not compile (missing field),
// and even if it did, `"setup_error_events"` would not be in the JSON output.
// ---------------------------------------------------------------------------

#[test]
fn test_backend_stats_json_field_names_are_snake_case() {
    let stats = make_non_trivial_stats();
    let json = serde_json::to_string(&stats).unwrap();
    let obj: serde_json::Value = serde_json::from_str(&json).unwrap();

    let required_keys = [
        // P1 throughput
        "batch_wall_secs",
        "rollouts_completed",
        "rollouts_per_sec",
        "tool_calls_per_sec",
        // P3 containment
        "adversarial_injected",
        "adversarial_contained",
        "contagion_events",
        // Cycle-3: setup errors are distinct from contagion (a SetupError is NOT an escape)
        "setup_error_events",
        "time_to_contain_secs",
        "cgroup_freeze_events",
        "cgroup_kill_events",
        "oom_kill_events",
        // GPU idle
        "gpu_idle_attributable_to_hang_secs",
        // step index
        "step_index",
    ];

    for key in &required_keys {
        assert!(
            obj.get(key).is_some(),
            "Expected JSON key '{key}' not found in BackendStats JSON output.\n\
             JSON was: {json}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3d-6: Default BackendStats has zero contagion_events, zero
//            setup_error_events, and empty time_to_contain_secs (guard against
//            non-zero defaults).
//
// Cycle-3 change: asserts `setup_error_events == 0` in the default value.
// FAILS NOW: `BackendStats::default()` does not have `setup_error_events`.
// ---------------------------------------------------------------------------

#[test]
fn test_backend_stats_default_is_zero_and_empty() {
    let stats = BackendStats::default();
    assert_eq!(stats.contagion_events, 0);
    assert_eq!(
        stats.setup_error_events, 0,
        "setup_error_events must default to 0"
    );
    assert!(
        stats.time_to_contain_secs.is_empty(),
        "time_to_contain_secs must be empty in the default BackendStats"
    );
    assert_eq!(stats.cgroup_freeze_events, 0);
    assert_eq!(stats.cgroup_kill_events, 0);
    assert_eq!(stats.oom_kill_events, 0);
}

// ---------------------------------------------------------------------------
// Test 3d-NEW: setup_error_events survives round-trip with a non-zero value.
//
// Semantics: a sandbox `SetupError` (cgroup creation failed, namespace
// clone failed, etc.) must be counted in `setup_error_events`, NOT in
// `contagion_events`.  A setup failure is NOT a containment escape.
//
// FAILS NOW: `BackendStats` has no `setup_error_events` field.
// ---------------------------------------------------------------------------

#[test]
fn test_backend_stats_setup_error_events_survives_round_trip() {
    let stats = BackendStats {
        setup_error_events: 5,
        // Ensure setup errors do NOT bleed into contagion — they are separate.
        contagion_events: 0,
        ..BackendStats::default()
    };
    let json = serde_json::to_string(&stats).unwrap();
    let recovered: BackendStats = serde_json::from_str(&json).unwrap();
    assert_eq!(
        recovered.setup_error_events, 5,
        "setup_error_events must survive serialization round-trip"
    );
    assert_eq!(
        recovered.contagion_events, 0,
        "contagion_events must remain 0 — setup errors are not containment escapes"
    );
}

// ---------------------------------------------------------------------------
// Test 3d-7 (Linux-only): Rust → JSON → Python round-trip.
//
// Cycle-3 change: the stats struct now includes `setup_error_events`, and the
// Python script now prints it on line 5, with an assertion.  The Python
// `BackendStats` dataclass must also have `setup_error_events: int = 0`.
//
// FAILS NOW for two reasons:
//   1. `BackendStats` in Rust lacks `setup_error_events` (compile error).
//   2. Even once the Rust field exists, `stats.py` is missing the field
//      (the Python script will raise AttributeError on `bs.setup_error_events`).
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
#[test]
fn test_backend_stats_rust_to_python_round_trip() {
    use std::path::PathBuf;
    use std::process::Command;

    // Cycle-3: setup_error_events = 4 (non-zero, distinct from contagion).
    let stats = BackendStats {
        batch_wall_secs: 2.5,
        rollouts_completed: 4,
        rollouts_per_sec: 1.6,
        tool_calls_per_sec: 1.6,
        adversarial_injected: 3,
        adversarial_contained: 3,
        contagion_events: 7,
        setup_error_events: 4,
        time_to_contain_secs: vec![0.11, 0.22, 0.33],
        cgroup_freeze_events: 3,
        cgroup_kill_events: 3,
        oom_kill_events: 0,
        gpu_idle_attributable_to_hang_secs: 0.0,
        step_index: 99,
    };

    let json_str = serde_json::to_string(&stats).expect("must serialize");

    // Locate stats.py relative to the workspace root.
    // On wk-system the repo is at /home/wk/rlox; detect via the CARGO_MANIFEST_DIR
    // env var (set by cargo during test compilation) and walk up to the repo root.
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo");
    // manifest_dir = <repo_root>/crates/rlox-sandbox
    let repo_root = PathBuf::from(&manifest_dir)
        .parent() // crates/
        .and_then(|p| p.parent()) // repo_root/
        .expect("could not locate repo root from CARGO_MANIFEST_DIR")
        .to_path_buf();
    let stats_py = repo_root.join("python/rlox/agentic/stats.py");
    // Post-refactor (task #10) the shim re-exports `rlox_agent.stats`, which lives
    // under `python/`. Put that on sys.path so the shim resolves it (the canonical
    // module is stdlib-only — no torch — so this stays lightweight).
    let python_dir = repo_root.join("python");

    // Python inline script: load stats.py via importlib, parse JSON, print fields.
    // Uses importlib.util.spec_from_file_location to avoid importing the full
    // rlox package (which has heavy deps like PyTorch not available in test env).
    //
    // Cycle-3: prints setup_error_events on line 5 (0-indexed).
    let python_script = format!(
        r#"
import sys, json, importlib.util

sys.path.insert(0, {python_dir_repr})
stats_py_path = {stats_py_repr}
spec = importlib.util.spec_from_file_location("rlox_agentic_stats", stats_py_path)
mod = importlib.util.module_from_spec(spec)
# Register in sys.modules BEFORE exec_module so @dataclass can resolve the module dict.
sys.modules["rlox_agentic_stats"] = mod
spec.loader.exec_module(mod)

json_str = {json_repr}
bs = mod.BackendStats.from_json(json_str)

# Print values that Rust test will assert on, one per line.
print(bs.contagion_events)
print(bs.time_to_contain_secs)
print(bs.step_index)
print(bs.adversarial_injected)
print(bs.cgroup_freeze_events)
print(bs.setup_error_events)
"#,
        stats_py_repr = format!("{:?}", stats_py.to_str().unwrap()),
        python_dir_repr = format!("{:?}", python_dir.to_str().unwrap()),
        json_repr = format!("{:?}", json_str),
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(&python_script)
        .output()
        .expect("failed to launch python3 — is it on PATH?");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Python round-trip script failed.\nstdout: {stdout}\nstderr: {stderr}"
    );

    let lines: Vec<&str> = stdout.trim().lines().collect();
    assert_eq!(
        lines.len(),
        6,
        "Expected 6 output lines from Python script (added setup_error_events), got {}.\n\
         stdout: {stdout}",
        lines.len()
    );

    // Line 0: contagion_events
    assert_eq!(
        lines[0], "7",
        "Python contagion_events mismatch: got '{}', expected '7'",
        lines[0]
    );

    // Line 1: time_to_contain_secs — Python prints a list literal
    // We parse both sides as JSON to avoid float-repr fragility.
    let py_ttcs: Vec<f64> = {
        // Python list repr uses brackets; convert to JSON by replacing nothing
        // (Python list repr for float lists is valid JSON already).
        let s = lines[1].replace("'", "\""); // safety: no string elements here
        serde_json::from_str(&s).unwrap_or_else(|e| {
            panic!(
                "Could not parse Python time_to_contain_secs output as JSON.\n\
                 Got: '{}'\nError: {e}",
                lines[1]
            )
        })
    };
    let expected_ttcs = vec![0.11f64, 0.22, 0.33];
    for (i, (got, exp)) in py_ttcs.iter().zip(expected_ttcs.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-9,
            "time_to_contain_secs[{i}]: Python={got}, Rust={exp}"
        );
    }

    // Line 2: step_index
    assert_eq!(
        lines[2], "99",
        "Python step_index mismatch: got '{}', expected '99'",
        lines[2]
    );

    // Line 3: adversarial_injected
    assert_eq!(
        lines[3], "3",
        "Python adversarial_injected mismatch: got '{}', expected '3'",
        lines[3]
    );

    // Line 4: cgroup_freeze_events
    assert_eq!(
        lines[4], "3",
        "Python cgroup_freeze_events mismatch: got '{}', expected '3'",
        lines[4]
    );

    // Line 5 (Cycle-3): setup_error_events — must be 4
    assert_eq!(
        lines[5], "4",
        "Python setup_error_events mismatch: got '{}', expected '4'.\n\
         Implementer must add `setup_error_events: int = 0` to BackendStats in stats.py",
        lines[5]
    );
}
