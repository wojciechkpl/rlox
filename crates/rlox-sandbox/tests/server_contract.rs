/// Step 3a — Rollout server HTTP contract (Cycle 1 RED tests, updated Cycle 3)
///
/// Tests the `POST /rollout` endpoint via `tower::ServiceExt::oneshot` — no
/// network port is opened; the axum router is exercised in-process.
///
/// ## Cycle-3 changes (RED until the implementer fixes the handler)
///
///   1. **Mock vLLM required**: all tests that expect HTTP 200 now stand up a
///      mock vLLM (same pattern as `rollout_pipeline.rs`) and point
///      `router_with_config` at it.  The old `router()` helper (which relies on
///      a non-existent vLLM at `http://127.0.0.1:8000`) is NO LONGER used
///      because the handler must return **502** (not 200) when vLLM is
///      unreachable — the silent stub fallback is being removed.
///
///   2. **`trajectories.len() == sum of group_size over tasks`**: with a real
///      vLLM (or mock), the handler returns one trajectory per completion, not
///      one per task.  For a batch of T tasks each requesting group_size G
///      completions, the response has T × G trajectories.
///
///   3. **`setup_error_events` in BackendStats destructure**: the Cycle-3
///      struct literal in the destructure test must include this new field.
///      That test fails NOW with a compile error until the field is added.
///
///   4. **`test_vllm_unreachable_returns_502`**: points the config at
///      `http://127.0.0.1:1` (dead port), posts `/rollout`, asserts HTTP 502.
///      Fails NOW because the handler returns 200 (silent stub fallback).
///
/// ## Interface assumptions the implementer must honor
///
///   - The handler return type must change from `Json<RolloutResponse>` to
///     `Result<Json<RolloutResponse>, StatusCode>` (or an equivalent that
///     can emit 502).
///   - A transport error or non-2xx response from vLLM → handler returns 502.
///   - `trajectories.len() == group_size * tasks.len()` (one per completion).
///   - `BackendStats.setup_error_events: u32` must exist (Cycle-3 field).
///
/// All tests are Linux-only because the handler dispatches to Linux-specific
/// sandbox workers.

#[cfg(target_os = "linux")]
mod server_contract_tests {
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    use axum::extract::State as AxumState;
    use axum::{routing::post as axum_post, Json as AxumJson, Router as AxumRouter};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde::{Deserialize, Serialize};
    use tokio::net::TcpListener;
    use tower::ServiceExt; // for `oneshot`
    use uuid::Uuid;

    use rlox_sandbox::{
        router_with_config, BackendStats, RolloutRequest, RolloutResponse, RolloutTask,
        SamplingParams, SandboxRunConfig, ServerConfig,
    };

    // -----------------------------------------------------------------------
    // Mock vLLM types (OpenAI-compatible completions API)
    // Copied from rollout_pipeline.rs — keep in sync.
    // -----------------------------------------------------------------------

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct VllmChoice {
        pub text: String,
        pub index: u32,
        pub finish_reason: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct VllmCompletionResponse {
        pub choices: Vec<VllmChoice>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct VllmCompletionRequest {
        pub model: String,
        pub prompt: Vec<i32>,
        pub max_tokens: u32,
        pub temperature: f32,
        pub n: u32,
    }

    // -----------------------------------------------------------------------
    // Mock vLLM server state
    // -----------------------------------------------------------------------

    #[derive(Default)]
    struct MockVllmState {
        /// Texts cycled across `n` choices.
        completions: Vec<String>,
    }

    type SharedMockState = Arc<Mutex<MockVllmState>>;

    async fn mock_vllm_completions(
        AxumState(state): AxumState<SharedMockState>,
        AxumJson(req): AxumJson<VllmCompletionRequest>,
    ) -> AxumJson<VllmCompletionResponse> {
        let guard = state.lock().unwrap();
        let n = req.n as usize;
        let completions = &guard.completions;

        let choices: Vec<VllmChoice> = (0..n)
            .map(|i| VllmChoice {
                text: completions[i % completions.len()].clone(),
                index: i as u32,
                finish_reason: "stop".to_string(),
            })
            .collect();

        AxumJson(VllmCompletionResponse { choices })
    }

    /// Start a mock vLLM server on an ephemeral port; returns its SocketAddr.
    async fn start_mock_vllm(completions: Vec<String>) -> SocketAddr {
        let state = Arc::new(Mutex::new(MockVllmState { completions }));

        let app = AxumRouter::new()
            .route("/v1/completions", axum_post(mock_vllm_completions))
            .with_state(Arc::clone(&state));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        addr
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn minimal_sampling_params(n: u32) -> SamplingParams {
        SamplingParams {
            temperature: 0.8,
            max_new_tokens: 64,
            n,
        }
    }

    fn make_task(prompt_ids: Vec<i32>, group_size: u32) -> RolloutTask {
        RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids,
            sampling_params: minimal_sampling_params(group_size),
            test_suite: "pass".to_string(),
            is_adversarial: false,
        }
    }

    /// Build a `router_with_config` pointing at `vllm_addr`, and issue a POST
    /// /rollout via tower `oneshot`.  Returns `(status, body_bytes)`.
    ///
    /// The caller is responsible for deserializing the body — some tests want
    /// to assert on the raw status before deserialization (e.g. 502 tests).
    async fn oneshot_rollout(
        vllm_addr: SocketAddr,
        req_body: RolloutRequest,
    ) -> (StatusCode, Vec<u8>) {
        let group_size = req_body.group_size;
        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig {
                mem_limit_bytes: 64 * 1024 * 1024,
                pids_limit: 8,
                cpu_weight: 100,
            },
            group_size,
        };

        let app = router_with_config(config);

        let body_bytes = serde_json::to_vec(&req_body).expect("request must serialize");

        let http_request = Request::builder()
            .method("POST")
            .uri("/rollout")
            .header("content-type", "application/json")
            .body(Body::from(body_bytes))
            .unwrap();

        let response = app
            .oneshot(http_request)
            .await
            .expect("oneshot must not return a transport error");

        let status = response.status();
        let body = response
            .into_body()
            .collect()
            .await
            .expect("response body must be readable")
            .to_bytes()
            .to_vec();

        (status, body)
    }

    /// Deserialize the body bytes as `RolloutResponse`, panicking with context on failure.
    fn parse_response(body: &[u8]) -> RolloutResponse {
        serde_json::from_slice(body).unwrap_or_else(|e| {
            panic!(
                "Response body is not valid RolloutResponse JSON.\n\
                 Error: {e}\n\
                 Body: {}",
                String::from_utf8_lossy(body)
            )
        })
    }

    // -----------------------------------------------------------------------
    // Test 3a-1: Single task, group_size=1 → HTTP 200 + one trajectory.
    //
    // Cycle-3 change: uses mock vLLM (not the bare `router()` + silent stub).
    // The mock returns one simple completion; the handler must contact it.
    //
    // `trajectories.len()` is now asserted to equal `group_size * tasks.len()`
    // (= 1 × 1 = 1 here) rather than just `tasks.len()`.
    //
    // FAILS NOW because the handler returns 200 via stub when vLLM is the
    // default url, but with the mock it will only work once the real call
    // path is implemented.  (Actually the test will pass as-is IF the stub
    // path is still active, so the primary RED signal here comes from
    // test 3a-NEW below and the group_size > 1 assertions elsewhere.)
    // This test is kept to confirm the happy-path shape after the fix.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_post_rollout_single_task_returns_200_with_one_trajectory() {
        let vllm_addr = start_mock_vllm(vec!["x = 1".to_string()]).await;

        let task = make_task(vec![1, 2, 3, 4], 1);
        let job_id = task.job_id;

        let req = RolloutRequest {
            tasks: vec![task],
            group_size: 1,
            per_sample_timeout_secs: 5.0,
        };

        let (status, body) = oneshot_rollout(vllm_addr, req).await;

        assert_eq!(status, StatusCode::OK, "POST /rollout must return HTTP 200");

        let resp = parse_response(&body);

        // group_size=1, tasks=1 → 1 × 1 = 1 trajectory
        assert_eq!(
            resp.trajectories.len(),
            1,
            "Expected 1 trajectory (group_size=1 × tasks=1), got {}",
            resp.trajectories.len()
        );

        assert_eq!(
            resp.trajectories[0].job_id, job_id,
            "Trajectory job_id must echo the request task job_id"
        );

        let _: u64 = resp.backend_stats.step_index;
    }

    // -----------------------------------------------------------------------
    // Test 3a-2: Multiple tasks, group_size=2 → trajectories.len() == tasks × group_size.
    //
    // Cycle-3 change: group_size bumped to 2, so 3 tasks → 6 trajectories.
    // FAILS NOW: stub returns 1 trajectory per task (3 instead of 6).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_post_rollout_batch_returns_trajectory_per_completion() {
        let vllm_addr =
            start_mock_vllm(vec!["a = 1".to_string(), "b = 2".to_string()]).await;

        let group_size: u32 = 2;
        let tasks: Vec<RolloutTask> = (0..3_i32)
            .map(|i| make_task(vec![10 + i, 20 + i, 30 + i], group_size))
            .collect();
        let n_tasks = tasks.len();

        let req = RolloutRequest {
            tasks,
            group_size,
            per_sample_timeout_secs: 5.0,
        };

        let (status, body) = oneshot_rollout(vllm_addr, req).await;

        assert_eq!(status, StatusCode::OK);

        let resp = parse_response(&body);

        let expected_trajectories = n_tasks * group_size as usize;
        assert_eq!(
            resp.trajectories.len(),
            expected_trajectories,
            "trajectories.len() must equal tasks.len() × group_size ({} × {} = {}), got {}",
            n_tasks,
            group_size,
            expected_trajectories,
            resp.trajectories.len()
        );
    }

    // -----------------------------------------------------------------------
    // Test 3a-3: backend_stats has all required P1, P3, and Cycle-3 fields.
    //
    // Cycle-3 change: `setup_error_events` added to the BackendStats
    // destructure.  This test will FAIL TO COMPILE until the implementer
    // adds `setup_error_events: u32` to the `BackendStats` struct.
    //
    // FAILS NOW: compile error (missing field) + handler uses stub.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_post_rollout_response_backend_stats_has_all_fields() {
        let vllm_addr = start_mock_vllm(vec!["pass".to_string()]).await;

        let req = RolloutRequest {
            tasks: vec![make_task(vec![1, 2, 3], 1)],
            group_size: 1,
            per_sample_timeout_secs: 5.0,
        };

        let (status, body) = oneshot_rollout(vllm_addr, req).await;
        assert_eq!(status, StatusCode::OK);

        let resp = parse_response(&body);

        // Destructure BackendStats fully — a compile error here means a field
        // was removed from (or not yet added to) the struct, which is a
        // breaking contract change.
        //
        // Cycle-3: `setup_error_events` is a NEW required field.
        let BackendStats {
            batch_wall_secs: _,
            rollouts_completed,
            rollouts_per_sec: _,
            tool_calls_per_sec: _,
            adversarial_injected: _,
            adversarial_contained: _,
            contagion_events,
            setup_error_events,
            time_to_contain_secs,
            cgroup_freeze_events: _,
            cgroup_kill_events: _,
            oom_kill_events: _,
            gpu_idle_attributable_to_hang_secs: _,
            step_index: _,
        } = resp.backend_stats;

        // For a single non-adversarial task with group_size=1:
        assert_eq!(
            rollouts_completed, 1,
            "rollouts_completed must equal group_size × tasks in the batch"
        );
        assert_eq!(
            contagion_events, 0,
            "contagion_events must be 0 when no adversarial tasks are in the batch"
        );
        assert_eq!(
            setup_error_events, 0,
            "setup_error_events must be 0 when no sandbox setup failure occurred"
        );
        assert!(
            time_to_contain_secs.is_empty(),
            "time_to_contain_secs must be empty when no adversarial tasks were injected"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3a-4: adversarial task → adversarial_injected == group_size in stats.
    //
    // Cycle-3 note: adversarial_injected is now counted at the COMPLETION level
    // (group_size=1 here, so == 1 as before).  For group_size > 1 see the
    // dedicated test in rollout_pipeline.rs (test_adversarial_telemetry_group_size_gt_1).
    //
    // FAILS NOW: handler uses stub.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_post_rollout_adversarial_task_reflected_in_backend_stats() {
        let vllm_addr = start_mock_vllm(vec!["pass".to_string()]).await;

        let mut task = make_task(vec![5, 6, 7], 1);
        task.is_adversarial = true;

        let req = RolloutRequest {
            tasks: vec![task],
            group_size: 1,
            per_sample_timeout_secs: 5.0,
        };

        let (status, body) = oneshot_rollout(vllm_addr, req).await;
        assert_eq!(status, StatusCode::OK);

        let resp = parse_response(&body);

        // With group_size=1 and 1 adversarial task → 1 adversarial completion injected.
        assert_eq!(
            resp.backend_stats.adversarial_injected, 1,
            "adversarial_injected must equal the number of adversarial completions \
             (group_size × adversarial_tasks = 1 × 1 = 1)"
        );
        assert_eq!(
            resp.trajectories[0].is_adversarial, true,
            "Trajectory must echo the is_adversarial flag from the request task"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3a-5: Empty task list → HTTP 200, empty trajectories, stats present.
    //
    // No vLLM call is made for an empty batch; the mock URL is irrelevant here,
    // but we still configure the handler correctly.
    //
    // FAILS NOW: handler uses stub (but this one may pass — kept for regression).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_post_rollout_empty_task_list_returns_200_with_empty_trajectories() {
        // Use a dead port — no request should reach vLLM for an empty batch.
        // If the handler still returns 200 with no error for empty batches,
        // that is correct behavior even with no vLLM.
        let vllm_addr = start_mock_vllm(vec![]).await;

        let req = RolloutRequest {
            tasks: vec![],
            group_size: 4,
            per_sample_timeout_secs: 5.0,
        };

        let (status, body) = oneshot_rollout(vllm_addr, req).await;

        assert_eq!(status, StatusCode::OK);

        let resp = parse_response(&body);

        assert!(
            resp.trajectories.is_empty(),
            "Empty task list must produce empty trajectories"
        );
        // backend_stats must still be present and defaults to zero.
        assert_eq!(resp.backend_stats.rollouts_completed, 0);
        assert_eq!(resp.backend_stats.setup_error_events, 0);
    }

    // -----------------------------------------------------------------------
    // Test 3a-6: Response trajectories preserve prompt_ids from request.
    //
    // Cycle-3 change: uses mock vLLM.
    //
    // FAILS NOW: handler uses stub (prompt_ids may echo correctly in stub,
    // but response_ids will be empty — the RED signal is in other tests).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_post_rollout_trajectory_echoes_prompt_ids() {
        let vllm_addr = start_mock_vllm(vec!["x = 42".to_string()]).await;

        let prompt_ids = vec![100i32, 200, 300, 400, 500];
        let task = make_task(prompt_ids.clone(), 1);

        let req = RolloutRequest {
            tasks: vec![task],
            group_size: 1,
            per_sample_timeout_secs: 5.0,
        };

        let (status, body) = oneshot_rollout(vllm_addr, req).await;
        assert_eq!(status, StatusCode::OK);

        let resp = parse_response(&body);

        assert_eq!(
            resp.trajectories[0].prompt_ids, prompt_ids,
            "Trajectory must echo the prompt_ids from the request task"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3a-7: Response content-type is application/json.
    //
    // Cycle-3 change: uses mock vLLM so the handler actually returns something.
    //
    // FAILS NOW: handler panics or stubs before returning headers in Cycle-1,
    // now it may return 502 when vLLM is not configured — mock fixes this.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_post_rollout_response_content_type_is_json() {
        let vllm_addr = start_mock_vllm(vec!["pass".to_string()]).await;

        let group_size: u32 = 1;
        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig {
                mem_limit_bytes: 64 * 1024 * 1024,
                pids_limit: 8,
                cpu_weight: 100,
            },
            group_size,
        };
        let app = router_with_config(config);

        let req_body = RolloutRequest {
            tasks: vec![make_task(vec![1], group_size)],
            group_size,
            per_sample_timeout_secs: 5.0,
        };

        let http_request = Request::builder()
            .method("POST")
            .uri("/rollout")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
            .unwrap();

        let response = app.oneshot(http_request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let content_type = response
            .headers()
            .get("content-type")
            .expect("content-type header must be present")
            .to_str()
            .unwrap();

        assert!(
            content_type.contains("application/json"),
            "content-type must be application/json, got: {content_type}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3a-NEW: vLLM unreachable → HTTP 502.
    //
    // Contract: when the vLLM endpoint is unreachable (transport error), the
    // handler MUST return HTTP 502 Bad Gateway.  It must NOT return 200 with
    // stub/zero-reward trajectories.
    //
    // This test points `vllm_base_url` at port 1 on loopback — a port that
    // is reserved by convention and will always refuse connections.
    //
    // FAILS NOW: the handler returns HTTP 200 (silent stub fallback).
    //
    // The implementer must:
    //   - Change the handler return type to `Result<Json<RolloutResponse>, StatusCode>`
    //     (or use an axum `IntoResponse` impl that can emit 502).
    //   - Return `Err(StatusCode::BAD_GATEWAY)` on any vLLM transport error
    //     or non-2xx HTTP response from vLLM.
    //   - Remove the `None => stub` fallback branch.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_vllm_unreachable_returns_502() {
        // Port 1 is reserved on Linux; connect(2) returns ECONNREFUSED immediately.
        // We wrap the whole test in a generous timeout (10 s) to guard against
        // unexpected blocking on unusual network configurations.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            async {
                let dead_vllm_url = "http://127.0.0.1:1";

                let config = ServerConfig {
                    vllm_base_url: dead_vllm_url.to_string(),
                    sandbox: SandboxRunConfig {
                        mem_limit_bytes: 64 * 1024 * 1024,
                        pids_limit: 8,
                        cpu_weight: 100,
                    },
                    group_size: 1,
                };
                let app = router_with_config(config);

                let req_body = RolloutRequest {
                    tasks: vec![RolloutTask {
                        job_id: Uuid::new_v4(),
                        prompt_ids: vec![1i32, 2, 3],
                        sampling_params: minimal_sampling_params(1),
                        test_suite: "pass".to_string(),
                        is_adversarial: false,
                    }],
                    group_size: 1,
                    per_sample_timeout_secs: 5.0,
                };

                let http_request = Request::builder()
                    .method("POST")
                    .uri("/rollout")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap();

                let response = app.oneshot(http_request).await.unwrap();

                assert_eq!(
                    response.status(),
                    StatusCode::BAD_GATEWAY,
                    "POST /rollout with unreachable vLLM must return HTTP 502 Bad Gateway, \
                     not 200 with stub trajectories. \
                     The handler must remove the silent stub fallback and propagate the error."
                );
            },
        )
        .await;

        result.expect(
            "test_vllm_unreachable_returns_502 timed out after 10 s — \
             the handler may be blocking waiting on vLLM instead of failing fast with 502",
        );
    }
}
