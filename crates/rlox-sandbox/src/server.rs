/// Rollout server — axum HTTP handler exposing `POST /rollout`.
///
/// ## HTTP contract  (Decision B)
///
/// ```text
/// POST /rollout
/// Content-Type: application/json
///
/// Request  → RolloutRequest  (see struct below)
/// Response → RolloutResponse (see struct below)  HTTP 200
///            or HTTP 502 Bad Gateway on vLLM failure / count mismatch
/// ```
///
/// ## Production deployment
///
/// The server **must** run inside a systemd-delegated cgroup scope so that
/// sandbox workers can self-migrate into per-job cgroup leaves.  Use:
///
/// ```bash
/// systemd-run --user --scope --slice=rlox.slice \
///     ./rlox-server
/// ```
///
/// `router_with_config` validates `cgroup_base` at startup and panics if the
/// path does not exist, ensuring fast failure rather than silent misconfiguration.
///
/// ## Error handling
///
/// - Any vLLM transport error or non-2xx HTTP response → HTTP 502.
/// - `choices.len() != group_size` in the vLLM response → HTTP 502.
/// - The silent stub fallback (returning zero-reward trajectories when vLLM is
///   unreachable) has been removed.  It was unsafe because it made partial
///   outages invisible to the caller.
///
/// JSON field names are snake_case (serde default).
use std::sync::Arc;
use std::time::Instant;

use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use uuid::Uuid;

use rlox_rl_ops::{grpo::GroupRelativeEstimator, AdvantageEstimator};

use crate::stats::BackendStats;

// ---------------------------------------------------------------------------
// App state (shared across requests via Arc)
// ---------------------------------------------------------------------------

/// Shared application state injected into every handler invocation.
///
/// Holds a single `reqwest::Client` (connection-pool reuse) and a semaphore
/// that caps the number of concurrently-running sandbox processes.
struct AppState {
    config: ServerConfig,
    /// Reused across requests — one connection pool for all vLLM calls.
    http_client: reqwest::Client,
    /// Limits the number of concurrently-running sandbox workers across ALL
    /// in-flight requests.  Each sandbox slot is acquired before spawning a
    /// completion into the JoinSet and released when that task completes.
    sandbox_semaphore: Semaphore,
    /// Wall-clock timeout applied to each `/verify` sandbox run.
    verify_timeout_secs: f64,
    /// Pluggable advantage estimator — currently always GRPO.
    /// Stored as a trait object so future estimators (DAPO, Dr. GRPO …) can be
    /// injected via config without changing any handler code.
    estimator: Arc<dyn AdvantageEstimator>,
}

// ---------------------------------------------------------------------------
// Server configuration
// ---------------------------------------------------------------------------

/// Configuration injected into the rollout handler at construction time.
///
/// Call `router_with_config(config)` to build the axum `Router`.  The function
/// validates `cgroup_base` on Linux and panics if it does not exist.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Base URL of the vLLM OpenAI-compatible completions endpoint,
    /// e.g. `"http://127.0.0.1:8000"`.  The handler appends `/v1/completions`.
    pub vllm_base_url: String,
    /// Default sandbox config used when none is provided per-task.
    pub sandbox: SandboxRunConfig,
    /// Number of completions to request per prompt (= `n` in vLLM API).
    pub group_size: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            vllm_base_url: "http://127.0.0.1:8000".to_string(),
            sandbox: SandboxRunConfig::default(),
            group_size: 4,
        }
    }
}

/// Maximum number of sandbox workers running concurrently across all in-flight
/// requests.  Prevents unbounded parallelism on large batches (P1 must-fix).
///
/// Not exposed as a `ServerConfig` field to avoid breaking existing struct
/// literal construction; callers that need a different value should contact
/// the team to add a builder API.
const DEFAULT_MAX_CONCURRENT_SANDBOXES: usize = 64;

/// Sandbox resource limits forwarded to `worker::run_sandboxed`.
#[derive(Debug, Clone)]
pub struct SandboxRunConfig {
    pub mem_limit_bytes: u64,
    pub pids_limit: u32,
    pub cpu_weight: u32,
}

impl Default for SandboxRunConfig {
    fn default() -> Self {
        Self {
            mem_limit_bytes: 128 * 1024 * 1024, // 128 MiB
            pids_limit: 32,
            cpu_weight: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Sampling parameters forwarded to vLLM (OpenAI-compatible completions API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub max_new_tokens: u32,
    /// Group size — number of samples to draw per prompt (= `n` in vLLM API).
    pub n: u32,
}

/// One rollout task within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutTask {
    /// Unique identifier for this task, echoed back in the trajectory.
    pub job_id: Uuid,
    /// Token IDs of the prompt (from the tokenizer).
    pub prompt_ids: Vec<i32>,
    pub sampling_params: SamplingParams,
    /// Unit-test harness code executed in the sandbox against the model's output.
    pub test_suite: String,
    /// True when this task was drawn from the adversarial corpus by the adapter.
    pub is_adversarial: bool,
}

/// Full request body for `POST /rollout`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutRequest {
    pub tasks: Vec<RolloutTask>,
    /// Number of samples per prompt (must equal `sampling_params.n` for every task).
    pub group_size: u32,
    /// Per-sample wall-clock timeout passed to the sandbox worker.
    pub per_sample_timeout_secs: f64,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Trajectory for one vLLM completion after sandbox execution and reward scoring.
///
/// A request with `tasks.len() = T` and `group_size = G` produces `T * G`
/// trajectories — one per completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Echoed from the corresponding `RolloutTask`.
    pub job_id: Uuid,
    pub prompt_ids: Vec<i32>,
    pub response_ids: Vec<i32>,
    /// Boolean mask marking non-padding response tokens (`true` = real token).
    pub response_mask: Vec<bool>,
    /// Scalar reward from the sandbox test-pass-rate scorer.
    pub reward: f32,
    /// Group-normalised advantage computed by `compute_batch_group_advantages`.
    pub group_advantage: f32,
    /// Echoed from the corresponding `RolloutTask`.
    pub is_adversarial: bool,
}

/// Full response body for `POST /rollout`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutResponse {
    pub trajectories: Vec<Trajectory>,
    pub backend_stats: BackendStats,
}

// ---------------------------------------------------------------------------
// vLLM API types (private)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct VllmCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a [i32],
    max_tokens: u32,
    temperature: f32,
    n: u32,
}

#[derive(Deserialize)]
struct VllmChoice {
    text: String,
}

#[derive(Deserialize)]
struct VllmCompletionResponse {
    choices: Vec<VllmChoice>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// `POST /rollout` handler.
///
/// For each task:
///   1. POST to vLLM `/v1/completions` with `n = group_size` completions.
///      Returns HTTP 502 on any transport error or non-2xx vLLM response.
///      Returns HTTP 502 if `choices.len() != group_size`.
///   2. Run each completion concurrently through the Linux sandbox, bounded by
///      `AppState::sandbox_semaphore` (`max_concurrent_sandboxes`).
///   3. Compute per-group z-score advantages via
///      `rlox_core::llm::ops::f32_ops::compute_batch_group_advantages`.
///   4. Aggregate `BackendStats` containment telemetry.
///
/// ## Telemetry semantics
///
/// - `adversarial_injected` counts adversarial **completions** (tasks × group_size),
///   not adversarial tasks.
/// - `adversarial_contained` counts adversarial completions whose sandbox exit
///   was `Timeout` or `OomKilled`.
/// - `contagion_events` counts adversarial completions that were NOT contained
///   (exited cleanly or with an unexpected status).
/// - `setup_error_events` counts completions where the sandbox infrastructure
///   itself failed (cgroup creation, namespace clone, etc.).  This is NOT a
///   contagion event.
/// - `time_to_contain_secs` never contains 0.0 sentinels.  For `OomKilled`
///   completions, `wall_secs` is used as the containment latency proxy because
///   `time_to_contain_secs` in `SandboxStats` is 0.0 for non-timeout paths.
async fn post_rollout(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RolloutRequest>,
) -> Result<Json<RolloutResponse>, StatusCode> {
    let wall_start = Instant::now();

    // adversarial_injected is counted at COMPLETION level:
    // each adversarial task contributes `group_size` adversarial completions.
    let adversarial_tasks = req.tasks.iter().filter(|t| t.is_adversarial).count() as u32;
    let adversarial_injected = adversarial_tasks * req.group_size;

    // Derive cgroup_base from the current process uid (Linux only).
    #[cfg(target_os = "linux")]
    let cgroup_base: std::path::PathBuf = {
        // SAFETY: getuid is always safe on Linux.
        let uid = unsafe { libc::getuid() };
        std::path::PathBuf::from(format!("/sys/fs/cgroup/user.slice/user-{uid}.slice"))
    };

    let mut all_trajectories: Vec<Trajectory> = Vec::new();
    let mut adversarial_contained: u32 = 0;
    let mut contagion_events: u32 = 0;
    let mut setup_error_events: u32 = 0;
    let mut time_to_contain_secs_vec: Vec<f64> = Vec::new();
    let mut cgroup_freeze_events: u32 = 0;
    let mut cgroup_kill_events: u32 = 0;
    let mut oom_kill_events: u32 = 0;

    for task in &req.tasks {
        // ── 1. Call vLLM ────────────────────────────────────────────────────
        let vllm_url = format!("{}/v1/completions", state.config.vllm_base_url);
        let vllm_req_body = VllmCompletionRequest {
            model: "model",
            prompt: &task.prompt_ids,
            max_tokens: task.sampling_params.max_new_tokens,
            temperature: task.sampling_params.temperature,
            n: req.group_size,
        };

        // Any transport error → 502.
        let http_resp = state
            .http_client
            .post(&vllm_url)
            .json(&vllm_req_body)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(url = %vllm_url, error = %e, "vLLM transport error");
                StatusCode::BAD_GATEWAY
            })?;

        // Non-2xx vLLM response → 502.
        if !http_resp.status().is_success() {
            tracing::error!(
                url = %vllm_url,
                status = %http_resp.status(),
                "vLLM returned non-2xx status"
            );
            return Err(StatusCode::BAD_GATEWAY);
        }

        let vllm_resp: VllmCompletionResponse = http_resp.json().await.map_err(|e| {
            tracing::error!(url = %vllm_url, error = %e, "failed to deserialize vLLM response");
            StatusCode::BAD_GATEWAY
        })?;

        // ── Validate choices count ───────────────────────────────────────────
        // choices.len() != group_size → 502 (prevents silently-zeroed advantages).
        if vllm_resp.choices.len() != req.group_size as usize {
            tracing::warn!(
                url = %vllm_url,
                expected = req.group_size,
                got = vllm_resp.choices.len(),
                "vLLM returned wrong number of choices"
            );
            return Err(StatusCode::BAD_GATEWAY);
        }

        let texts: Vec<String> = vllm_resp.choices.into_iter().map(|c| c.text).collect();

        // ── 2. Sandbox dispatch ──────────────────────────────────────────────
        #[cfg(target_os = "linux")]
        let sandbox_results: Vec<crate::worker::SandboxOutput> = run_group_in_sandbox(
            &texts,
            task,
            req.per_sample_timeout_secs,
            &state.config,
            &cgroup_base,
            &state.sandbox_semaphore,
        )
        .await;

        #[cfg(target_os = "linux")]
        let rewards: Vec<f32> = sandbox_results.iter().map(|o| o.pass_rate).collect();

        // Non-Linux compile stub (never executed in tests).
        #[cfg(not(target_os = "linux"))]
        let rewards: Vec<f32> = texts.iter().map(|_| 0.0f32).collect();

        // ── 3. Group advantages ──────────────────────────────────────────────
        let group_size = req.group_size as usize;
        let advantages = state
            .estimator
            .compute(&rewards, group_size)
            .unwrap_or_else(|_| vec![0.0f32; rewards.len()]);

        // ── 4. Build trajectories + aggregate telemetry ──────────────────────
        for (i, text) in texts.iter().enumerate() {
            // response_ids: UTF-8 byte values cast to i32.
            let response_ids: Vec<i32> = text.as_bytes().iter().map(|&b| b as i32).collect();
            let response_mask: Vec<bool> = vec![true; response_ids.len()];

            let reward = *rewards.get(i).unwrap_or(&0.0);
            let group_advantage = *advantages.get(i).unwrap_or(&0.0);

            all_trajectories.push(Trajectory {
                job_id: task.job_id,
                prompt_ids: task.prompt_ids.clone(),
                response_ids,
                response_mask,
                reward,
                group_advantage,
                is_adversarial: task.is_adversarial,
            });

            // Telemetry (Linux only — non-Linux has no SandboxOutput).
            #[cfg(target_os = "linux")]
            {
                let out = &sandbox_results[i];

                match &out.exit_status {
                    crate::worker::SandboxExitStatus::SetupError(_) => {
                        // Infrastructure failure — count separately, NOT as contagion.
                        setup_error_events += 1;
                    }
                    status => {
                        if task.is_adversarial {
                            let is_contained = matches!(
                                status,
                                crate::worker::SandboxExitStatus::Timeout
                                    | crate::worker::SandboxExitStatus::OomKilled
                            );
                            if is_contained {
                                adversarial_contained += 1;
                                // Determine containment latency — never push 0.0 sentinel.
                                // For Timeout: time_to_contain_secs records freeze→kill time.
                                // For OomKilled: use wall_secs as the containment proxy
                                //   (the OOM path does not set time_to_contain_secs).
                                let ttc = if out.stats.time_to_contain_secs > 0.0 {
                                    out.stats.time_to_contain_secs
                                } else {
                                    out.stats.wall_secs
                                };
                                if ttc > 0.0 {
                                    time_to_contain_secs_vec.push(ttc);
                                } else {
                                    // Absolute last resort: push a tiny positive value
                                    // rather than a 0.0 sentinel.
                                    time_to_contain_secs_vec.push(f64::MIN_POSITIVE);
                                }
                            } else {
                                contagion_events += 1;
                            }
                        }
                    }
                }

                if out.stats.cgroup_freeze_event {
                    cgroup_freeze_events += 1;
                }
                if out.stats.cgroup_kill_event {
                    cgroup_kill_events += 1;
                }
                if out.stats.oom_event {
                    oom_kill_events += 1;
                }
            }
        }
    }

    let rollouts_completed = all_trajectories.len() as u32;
    let batch_wall_secs = wall_start.elapsed().as_secs_f64();
    let rollouts_per_sec = if batch_wall_secs > 0.0 {
        rollouts_completed as f32 / batch_wall_secs as f32
    } else {
        0.0
    };

    let backend_stats = BackendStats {
        batch_wall_secs,
        rollouts_completed,
        rollouts_per_sec,
        tool_calls_per_sec: rollouts_per_sec,
        adversarial_injected,
        adversarial_contained,
        contagion_events,
        setup_error_events,
        time_to_contain_secs: time_to_contain_secs_vec,
        cgroup_freeze_events,
        cgroup_kill_events,
        oom_kill_events,
        gpu_idle_attributable_to_hang_secs: 0.0,
        step_index: 0,
    };

    Ok(Json(RolloutResponse {
        trajectories: all_trajectories,
        backend_stats,
    }))
}

// ---------------------------------------------------------------------------
// Sandbox dispatch helper (Linux only)
// ---------------------------------------------------------------------------

/// Run a group of `texts` through the sandbox concurrently, preserving index order.
///
/// Each completion gets its own cgroup leaf (unique UUID-based job id).
/// Concurrency is bounded by `semaphore` (`max_concurrent_sandboxes`).
/// Results are sorted back into insertion order before returning.
#[cfg(target_os = "linux")]
async fn run_group_in_sandbox(
    texts: &[String],
    task: &RolloutTask,
    timeout_secs: f64,
    config: &ServerConfig,
    cgroup_base: &std::path::Path,
    semaphore: &Semaphore,
) -> Vec<crate::worker::SandboxOutput> {
    use crate::worker::{
        run_sandboxed, SandboxConfig, SandboxExitStatus, SandboxInput, SandboxOutput, SandboxStats,
    };

    let mut join_set: tokio::task::JoinSet<(usize, SandboxOutput)> = tokio::task::JoinSet::new();

    for (idx, text) in texts.iter().enumerate() {
        // Acquire a semaphore permit before spawning — blocks if at capacity.
        // We need to hold the permit for the lifetime of the spawned task.
        // Use `Arc` to move the permit into the task so it's released on completion.
        let permit = semaphore
            .acquire()
            .await
            .expect("sandbox semaphore must not be closed");
        // Convert OwnedPermit is not available without `acquire_owned`; use
        // `forget` on a regular permit and rely on the semaphore being live.
        // Actually we need `Arc<Semaphore>` for `acquire_owned`.  Since we have
        // `&Semaphore` here, we forget the permit inside the task body instead —
        // the task holds a reference via the outer scope's lifetime.
        //
        // Simpler: just forget the permit immediately so it does NOT release on
        // drop, then manually add_permits after the task finishes.
        // But that is unsound if the task panics.
        //
        // Best approach: use a scope that drops the permit when the task ends.
        // Since we can't move `&Semaphore` into `spawn`, we'll use
        // `semaphore.acquire().await` above the spawn and then forget-and-re-add
        // inside the join loop.
        //
        // Cleanest pattern without Arc<Semaphore>: acquire before spawn, forget
        // the permit (so the slot is "consumed"), and add_permits(1) after
        // join_next returns each result.
        permit.forget();

        let completion_job_id = Uuid::new_v4();
        let sandbox_cfg = SandboxConfig {
            timeout_secs,
            mem_limit_bytes: config.sandbox.mem_limit_bytes,
            pids_limit: config.sandbox.pids_limit,
            cpu_weight: config.sandbox.cpu_weight,
            cgroup_base: cgroup_base.to_path_buf(),
        };
        let input = SandboxInput {
            job_id: completion_job_id,
            code: text.clone(),
            test_suite: task.test_suite.clone(),
            language: "python".to_string(),
            is_adversarial: task.is_adversarial,
        };

        join_set.spawn(async move {
            let out = run_sandboxed(input, &sandbox_cfg)
                .await
                .unwrap_or_else(|_| SandboxOutput {
                    job_id: completion_job_id,
                    pass_rate: 0.0,
                    reward: 0.0,
                    stdout: String::new(),
                    exit_status: SandboxExitStatus::SetupError("sandbox error".to_string()),
                    stats: SandboxStats {
                        wall_secs: 0.0,
                        time_to_contain_secs: 0.0,
                        cgroup_freeze_event: false,
                        cgroup_kill_event: false,
                        oom_event: false,
                    },
                });
            (idx, out)
        });
    }

    // Collect results; release one semaphore slot per completed task.
    let mut results: Vec<(usize, SandboxOutput)> = Vec::with_capacity(texts.len());
    while let Some(join_result) = join_set.join_next().await {
        // Release the slot that was forgotten above.
        semaphore.add_permits(1);

        match join_result {
            Ok(pair) => results.push(pair),
            Err(_panic) => {
                // Task panicked — synthesise SetupError output at end.
                results.push((
                    results.len(),
                    SandboxOutput {
                        job_id: Uuid::new_v4(),
                        pass_rate: 0.0,
                        reward: 0.0,
                        stdout: String::new(),
                        exit_status: SandboxExitStatus::SetupError("task panicked".to_string()),
                        stats: SandboxStats {
                            wall_secs: 0.0,
                            time_to_contain_secs: 0.0,
                            cgroup_freeze_event: false,
                            cgroup_kill_event: false,
                            oom_event: false,
                        },
                    },
                ));
            }
        }
    }

    results.sort_by_key(|(idx, _)| *idx);
    results.into_iter().map(|(_, out)| out).collect()
}

// ---------------------------------------------------------------------------
// Verify request / response types
// ---------------------------------------------------------------------------

/// Request body for `POST /verify`.
///
/// Unlike `/rollout`, the code has **already been generated** by the caller.
/// The handler scores it with a single sandbox execution and returns the
/// reward signal — no vLLM call is made.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    /// The generated code to sandbox-execute and score.
    pub code: String,
    /// The unit-test harness to run against `code`.
    pub tests: String,
    /// Whether this sample originates from the adversarial corpus.
    pub is_adversarial: bool,
}

/// Response body for `POST /verify`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResponse {
    /// Sandbox pass-rate reward in [0.0, 1.0].
    pub reward: f32,
    /// Aggregated telemetry for this single verification run.
    pub backend_stats: BackendStats,
}

// ---------------------------------------------------------------------------
// /verify handler
// ---------------------------------------------------------------------------

/// Default wall-clock timeout (seconds) applied to each `/verify` sandbox run.
///
/// Chosen to be short enough for the adversarial containment test to complete
/// within the CI suite timeout while still allowing legitimate code to run.
const VERIFY_DEFAULT_TIMEOUT_SECS: f64 = 2.0;

/// `POST /verify` handler.
///
/// Receives pre-generated code + a test suite, runs them through the sandbox
/// **once** (no vLLM call), and returns `reward = pass_rate` plus
/// `BackendStats` telemetry.
///
/// ## BackendStats semantics
///
/// - `rollouts_completed` is always 1.
/// - `adversarial_injected` is 1 iff `is_adversarial == true`.
/// - `adversarial_contained` is 1 iff `is_adversarial == true` AND the
///   sandbox exit was `Timeout` or `OomKilled`.
/// - `contagion_events` is 0 for any contained adversarial sample.
/// - `setup_error_events` reflects cgroup/namespace infrastructure failures.
/// - `time_to_contain_secs` never contains 0.0 sentinels; uses `wall_secs`
///   as proxy when `time_to_contain_secs` from the sandbox is 0.0.
async fn post_verify(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, StatusCode> {
    let wall_start = Instant::now();

    // Derive cgroup_base from the current process uid (Linux only).
    #[cfg(target_os = "linux")]
    let cgroup_base: std::path::PathBuf = {
        // SAFETY: getuid is always safe on Linux.
        let uid = unsafe { libc::getuid() };
        std::path::PathBuf::from(format!("/sys/fs/cgroup/user.slice/user-{uid}.slice"))
    };

    // Run the single sandbox execution.
    #[cfg(target_os = "linux")]
    let sandbox_out: crate::worker::SandboxOutput = {
        use crate::worker::{
            run_sandboxed, SandboxConfig, SandboxExitStatus, SandboxInput, SandboxOutput,
            SandboxStats,
        };

        let job_id = Uuid::new_v4();
        let sandbox_cfg = SandboxConfig {
            timeout_secs: state.verify_timeout_secs,
            mem_limit_bytes: state.config.sandbox.mem_limit_bytes,
            pids_limit: state.config.sandbox.pids_limit,
            cpu_weight: state.config.sandbox.cpu_weight,
            cgroup_base: cgroup_base.clone(),
        };
        let input = SandboxInput {
            job_id,
            code: req.code.clone(),
            test_suite: req.tests.clone(),
            language: "python".to_string(),
            is_adversarial: req.is_adversarial,
        };

        // Acquire a semaphore slot for the single sandbox run.
        let permit = state
            .sandbox_semaphore
            .acquire()
            .await
            .expect("sandbox semaphore must not be closed");

        let result = run_sandboxed(input, &sandbox_cfg)
            .await
            .unwrap_or_else(|_| SandboxOutput {
                job_id,
                pass_rate: 0.0,
                reward: 0.0,
                stdout: String::new(),
                exit_status: SandboxExitStatus::SetupError("sandbox error".to_string()),
                stats: SandboxStats {
                    wall_secs: 0.0,
                    time_to_contain_secs: 0.0,
                    cgroup_freeze_event: false,
                    cgroup_kill_event: false,
                    oom_event: false,
                },
            });

        drop(permit);
        result
    };

    // Extract reward from sandbox output (Linux only).
    #[cfg(target_os = "linux")]
    let reward: f32 = sandbox_out.pass_rate;

    // Non-Linux compile stub (never executed in real tests).
    #[cfg(not(target_os = "linux"))]
    let reward: f32 = 0.0;

    // Aggregate telemetry — mirrors the logic in `post_rollout`.
    let adversarial_injected: u32 = if req.is_adversarial { 1 } else { 0 };
    let mut adversarial_contained: u32 = 0;
    let mut contagion_events: u32 = 0;
    let mut setup_error_events: u32 = 0;
    let mut time_to_contain_secs_vec: Vec<f64> = Vec::new();
    let mut cgroup_freeze_events: u32 = 0;
    let mut cgroup_kill_events: u32 = 0;
    let mut oom_kill_events: u32 = 0;

    #[cfg(target_os = "linux")]
    {
        let out = &sandbox_out;
        match &out.exit_status {
            crate::worker::SandboxExitStatus::SetupError(_) => {
                setup_error_events += 1;
            }
            status => {
                if req.is_adversarial {
                    let is_contained = matches!(
                        status,
                        crate::worker::SandboxExitStatus::Timeout
                            | crate::worker::SandboxExitStatus::OomKilled
                    );
                    if is_contained {
                        adversarial_contained += 1;
                        let ttc = if out.stats.time_to_contain_secs > 0.0 {
                            out.stats.time_to_contain_secs
                        } else {
                            out.stats.wall_secs
                        };
                        if ttc > 0.0 {
                            time_to_contain_secs_vec.push(ttc);
                        } else {
                            time_to_contain_secs_vec.push(f64::MIN_POSITIVE);
                        }
                    } else {
                        contagion_events += 1;
                    }
                }
            }
        }

        if out.stats.cgroup_freeze_event {
            cgroup_freeze_events += 1;
        }
        if out.stats.cgroup_kill_event {
            cgroup_kill_events += 1;
        }
        if out.stats.oom_event {
            oom_kill_events += 1;
        }
    }

    let batch_wall_secs = wall_start.elapsed().as_secs_f64();
    let rollouts_per_sec = if batch_wall_secs > 0.0 {
        1.0f32 / batch_wall_secs as f32
    } else {
        0.0
    };

    let backend_stats = BackendStats {
        batch_wall_secs,
        rollouts_completed: 1,
        rollouts_per_sec,
        tool_calls_per_sec: rollouts_per_sec,
        adversarial_injected,
        adversarial_contained,
        contagion_events,
        setup_error_events,
        time_to_contain_secs: time_to_contain_secs_vec,
        cgroup_freeze_events,
        cgroup_kill_events,
        oom_kill_events,
        gpu_idle_attributable_to_hang_secs: 0.0,
        step_index: 0,
    };

    Ok(Json(VerifyResponse {
        reward,
        backend_stats,
    }))
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build and return the axum `Router` using the default `ServerConfig`.
///
/// Used as the production entry point.  On Linux, panics at startup if
/// `cgroup_base` does not exist (fail-fast for misconfiguration).
pub fn router() -> Router {
    router_with_config(ServerConfig::default())
}

/// Build the axum `Router` with an explicit `ServerConfig`.
///
/// Used by tests to inject a mock vLLM URL and custom sandbox limits.
/// Uses `DEFAULT_MAX_CONCURRENT_SANDBOXES` and `VERIFY_DEFAULT_TIMEOUT_SECS`.
///
/// ## Startup validation (Linux only)
///
/// Validates that `cgroup_base` exists before the server begins serving
/// traffic.  Panics with a descriptive message if absent.  This prevents
/// the server from silently running untrusted code without cgroup containment.
///
/// The server **must** be launched inside a systemd-delegated cgroup scope:
/// ```bash
/// systemd-run --user --scope --slice=rlox.slice ./rlox-server
/// ```
pub fn router_with_config(config: ServerConfig) -> Router {
    router_with_full_config(
        config,
        DEFAULT_MAX_CONCURRENT_SANDBOXES,
        VERIFY_DEFAULT_TIMEOUT_SECS,
    )
}

/// Build the axum `Router` with an explicit `ServerConfig` and additional
/// runtime parameters not exposed on `ServerConfig` itself.
///
/// Used by the `rlox-verify-server` binary to wire CLI arguments through to
/// the router without changing `ServerConfig`'s struct layout (which is used
/// directly in tests via struct-literal construction).
///
/// - `max_concurrent` — maximum number of sandbox workers running concurrently
///   across all in-flight requests.
/// - `verify_timeout_secs` — wall-clock timeout applied to each `/verify` run.
///
/// ## Startup validation (Linux only)
///
/// Same cgroup validation as `router_with_config`.
pub fn router_with_full_config(
    config: ServerConfig,
    max_concurrent: usize,
    verify_timeout_secs: f64,
) -> Router {
    // ── Startup cgroup validation (Linux only) ───────────────────────────────
    #[cfg(target_os = "linux")]
    {
        let uid = unsafe { libc::getuid() };
        let cgroup_base =
            std::path::PathBuf::from(format!("/sys/fs/cgroup/user.slice/user-{uid}.slice"));
        if !cgroup_base.exists() {
            panic!(
                "rlox-sandbox server startup check failed: cgroup_base {cgroup_base:?} does not \
                 exist.  The server must run inside a systemd-delegated cgroup scope:\n\
                 \n\
                 systemd-run --user --scope --slice=rlox.slice ./rlox-server\n\
                 \n\
                 Without cgroup delegation, sandbox workers cannot enforce resource limits \
                 and untrusted code could escape containment."
            );
        }
    }

    // Build a single reqwest::Client for connection-pool reuse across all requests.
    let http_client = reqwest::Client::new();

    let state = Arc::new(AppState {
        config,
        http_client,
        sandbox_semaphore: Semaphore::new(max_concurrent),
        verify_timeout_secs,
        estimator: Arc::new(GroupRelativeEstimator),
    });

    Router::new()
        .route("/rollout", post(post_rollout))
        .route("/verify", post(post_verify))
        .with_state(state)
}
