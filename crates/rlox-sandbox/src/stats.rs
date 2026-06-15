/// BackendStats — first-class telemetry struct returned with every `/rollout` response.
///
/// Covers P1 throughput fields and P3 containment fields.  Serialized as JSON in the
/// HTTP response body and written to the metric store by the harness.
///
/// JSON field names use `snake_case` (serde default). The Python-side `BackendStats`
/// dataclass in `python/rlox/agentic/stats.py` must mirror these names exactly.
///
/// Field contract (must not change without updating both sides):
///   - `batch_wall_secs`                — P1 throughput
///   - `rollouts_completed`             — P1 throughput
///   - `rollouts_per_sec`               — P1 throughput
///   - `tool_calls_per_sec`             — P1 throughput (= rollouts_per_sec in MVP)
///   - `adversarial_injected`           — P3 containment
///   - `adversarial_contained`          — P3 containment (must == adversarial_injected in Treatment)
///   - `contagion_events`               — P3 containment (non-zero is go/no-go failure)
///   - `setup_error_events`             — P3 containment (SetupError; NOT a contagion escape)
///   - `time_to_contain_secs`           — P3 containment per-sample (empty Vec if none)
///   - `cgroup_freeze_events`           — P3 containment
///   - `cgroup_kill_events`             — P3 containment
///   - `oom_kill_events`                — P3 containment
///   - `gpu_idle_attributable_to_hang_secs` — GPU idle attribution
///   - `step_index`                     — training step index for time-series alignment
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendStats {
    // --- P1 throughput fields ---
    pub batch_wall_secs: f64,
    pub rollouts_completed: u32,
    pub rollouts_per_sec: f32,
    /// Equal to `rollouts_per_sec` in MVP (one code-exec tool call per rollout).
    pub tool_calls_per_sec: f32,

    // --- P3 containment fields (first-class, not optional) ---
    pub adversarial_injected: u32,
    /// Must equal `adversarial_injected` in the Treatment condition (AC-5 hard line).
    pub adversarial_contained: u32,
    /// Non-zero value is a go/no-go failure for the Treatment condition.
    pub contagion_events: u32,
    /// Sandbox `SetupError` events (cgroup/namespace setup failures).
    ///
    /// A setup failure is NOT a containment escape — it must NOT increment
    /// `contagion_events`.  Counted separately so operators can distinguish
    /// infrastructure failures from adversarial-code containment failures.
    pub setup_error_events: u32,
    /// One entry per adversarial sample that required containment; empty if none injected.
    pub time_to_contain_secs: Vec<f64>,
    pub cgroup_freeze_events: u32,
    pub cgroup_kill_events: u32,
    pub oom_kill_events: u32,

    // --- GPU idle attribution ---
    /// Wall-clock seconds the GPU was idle due to a hung sandbox worker.
    pub gpu_idle_attributable_to_hang_secs: f64,
    /// Training step index — used for time-series alignment in the metric store.
    pub step_index: u64,
}

impl Default for BackendStats {
    fn default() -> Self {
        Self {
            batch_wall_secs: 0.0,
            rollouts_completed: 0,
            rollouts_per_sec: 0.0,
            tool_calls_per_sec: 0.0,
            adversarial_injected: 0,
            adversarial_contained: 0,
            contagion_events: 0,
            setup_error_events: 0,
            time_to_contain_secs: Vec::new(),
            cgroup_freeze_events: 0,
            cgroup_kill_events: 0,
            oom_kill_events: 0,
            gpu_idle_attributable_to_hang_secs: 0.0,
            step_index: 0,
        }
    }
}
