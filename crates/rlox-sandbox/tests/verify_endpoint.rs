/// Step 4 — `/verify` endpoint contract (RED phase)
///
/// ## Purpose
///
/// The `/verify` endpoint exposes sandbox-only scoring — it receives code that
/// has **already been generated** (e.g. by the verifiers `@reward` seam) and
/// returns a pass-rate reward + telemetry.  Unlike `/rollout`, it does NOT
/// call vLLM.
///
/// ## HTTP contract
///
/// ```
/// POST /verify
/// Content-Type: application/json
///
/// Request  → VerifyRequest  { code: String, tests: String, is_adversarial: bool }
/// Response → VerifyResponse { reward: f32, backend_stats: BackendStats }  HTTP 200
/// ```
///
/// ## Tests (all RED until the handler is implemented)
///
///   1. `test_verify_benign_code_returns_reward_1`
///      Benign code that passes its tests → reward == 1.0, HTTP 200, valid backend_stats.
///
///   2. `test_verify_failing_code_returns_low_reward`
///      Code that fails its tests → reward < 1.0.
///
///   3. `test_verify_adversarial_infinite_loop_contained`
///      `is_adversarial=true` + infinite-loop code → contained;
///      `adversarial_injected == 1`, `adversarial_contained == 1`,
///      `time_to_contain_secs` non-empty, `contagion_events == 0`,
///      `setup_error_events == 0`.  Short timeout.
///
///   4. `test_verify_does_not_call_vllm`
///      A `/verify` request succeeds even when no vLLM is configured/reachable,
///      proving the path is sandbox-only.
///
/// ## Interface assumptions for the implementer
///
///   - New types `VerifyRequest` and `VerifyResponse` must be added to `server.rs`
///     (or a separate module) and pub-exported from `lib.rs`.
///   - `VerifyRequest` fields: `code: String`, `tests: String`, `is_adversarial: bool`.
///   - `VerifyResponse` fields: `reward: f32`, `backend_stats: BackendStats`.
///   - The handler must run the code through the existing sandbox path
///     (`run_sandboxed`) with NO vLLM call.
///   - The route `POST /verify` must be added to the router returned by
///     `router_with_config` (and `router()`).
///   - `router()` / `router_with_config()` must NOT require a reachable vLLM;
///     `/verify` must succeed regardless of `vllm_base_url`.
///   - `BackendStats` semantics for `/verify`:
///     - `rollouts_completed == 1` always.
///     - `adversarial_injected == 1` iff `is_adversarial == true`.
///     - `adversarial_contained == 1` iff `is_adversarial == true` AND sandbox
///       exited via Timeout or OomKilled.
///     - `contagion_events == 0` for any contained adversarial completion.
///     - `setup_error_events` reflects cgroup/namespace infrastructure failures.
///
/// ## Test execution
///
/// Run with:
/// ```bash
/// bash scripts/wk-sync-test.sh 'cargo test -p rlox-sandbox --test verify_endpoint -- --test-threads=1'
/// ```
///
/// `--test-threads=1` is required because tests use real Linux sandbox workers
/// that contend for cgroup slots and PIDs.

#[cfg(target_os = "linux")]
mod verify_endpoint_tests {
    use std::net::SocketAddr;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tokio::net::TcpListener;
    use tower::ServiceExt; // for `oneshot`

    use rlox_sandbox::{
        router_with_config, BackendStats, SandboxRunConfig, ServerConfig, VerifyRequest,
        VerifyResponse,
    };

    // -----------------------------------------------------------------------
    // Helper: build a tight SandboxRunConfig for tests (small limits,
    // short timeout scenarios handled per-test via per_sample_timeout_secs).
    // -----------------------------------------------------------------------
    fn test_sandbox_config() -> SandboxRunConfig {
        SandboxRunConfig {
            mem_limit_bytes: 64 * 1024 * 1024, // 64 MiB
            pids_limit: 8,
            cpu_weight: 100,
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build a ServerConfig pointing at a dead vLLM URL.
    // /verify must succeed regardless of vllm_base_url.
    // -----------------------------------------------------------------------
    fn server_config_no_vllm() -> ServerConfig {
        ServerConfig {
            // Port 1 is reserved — always refuses connections.
            vllm_base_url: "http://127.0.0.1:1".to_string(),
            sandbox: test_sandbox_config(),
            group_size: 1,
        }
    }

    // -----------------------------------------------------------------------
    // Helper: issue a POST /verify via tower oneshot; returns (status, body).
    // -----------------------------------------------------------------------
    async fn oneshot_verify(req_body: VerifyRequest) -> (StatusCode, Vec<u8>) {
        let config = server_config_no_vllm();
        let app = router_with_config(config);

        let body_bytes = serde_json::to_vec(&req_body).expect("VerifyRequest must serialize");

        let http_request = Request::builder()
            .method("POST")
            .uri("/verify")
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

    // -----------------------------------------------------------------------
    // Helper: deserialize body bytes as VerifyResponse, panicking on failure.
    // -----------------------------------------------------------------------
    fn parse_verify_response(body: &[u8]) -> VerifyResponse {
        serde_json::from_slice::<VerifyResponse>(body).unwrap_or_else(|e| {
            panic!(
                "Response body is not valid VerifyResponse JSON.\n\
                 Error: {e}\n\
                 Body: {}",
                String::from_utf8_lossy(body)
            )
        })
    }

    // -----------------------------------------------------------------------
    // Helper: start the real rollout server on an ephemeral port.
    // Used by tests that need to call /verify over real TCP (not oneshot).
    // -----------------------------------------------------------------------
    async fn start_verify_server() -> SocketAddr {
        let app = router_with_config(server_config_no_vllm());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        addr
    }

    // -----------------------------------------------------------------------
    // Test V1: benign code that passes its test suite → reward == 1.0, HTTP 200.
    //
    // The code defines a correct `add` function; the tests assert it works.
    // The sandbox should execute both snippets together and return pass_rate = 1.0.
    //
    // FAILS NOW: /verify route does not exist → HTTP 405 Method Not Allowed or 404.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_verify_benign_code_returns_reward_1() {
        if crate::common::skip_without_cgroups("test_verify_benign_code_returns_reward_1") {
            return;
        }
        let passing_code = r#"
def add(a, b):
    return a + b
"#
        .to_string();

        let tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0".to_string();

        let req = VerifyRequest {
            code: passing_code,
            tests,
            is_adversarial: false,
        };

        let (status, body) = oneshot_verify(req).await;

        assert_eq!(
            status,
            StatusCode::OK,
            "POST /verify with passing code must return HTTP 200, got {status}"
        );

        let resp = parse_verify_response(&body);

        assert_eq!(
            resp.reward, 1.0,
            "reward must be 1.0 when all tests pass; got {}",
            resp.reward
        );

        // backend_stats must be present and well-formed.
        assert_eq!(
            resp.backend_stats.rollouts_completed, 1,
            "rollouts_completed must be 1 for a single /verify call"
        );

        assert_eq!(
            resp.backend_stats.adversarial_injected, 0,
            "adversarial_injected must be 0 for non-adversarial request"
        );

        assert_eq!(
            resp.backend_stats.contagion_events, 0,
            "contagion_events must be 0 for non-adversarial benign code"
        );

        // Verify the full BackendStats struct shape compiles (all fields must exist).
        // This is a compile-time contract check.
        let BackendStats {
            batch_wall_secs,
            rollouts_completed: _,
            rollouts_per_sec: _,
            tool_calls_per_sec: _,
            adversarial_injected: _,
            adversarial_contained: _,
            contagion_events: _,
            setup_error_events: _,
            time_to_contain_secs: _,
            cgroup_freeze_events: _,
            cgroup_kill_events: _,
            oom_kill_events: _,
            gpu_idle_attributable_to_hang_secs: _,
            step_index: _,
        } = resp.backend_stats;

        assert!(
            batch_wall_secs >= 0.0,
            "batch_wall_secs must be non-negative"
        );
    }

    // -----------------------------------------------------------------------
    // Test V2: failing code → reward < 1.0.
    //
    // The code returns the wrong answer; the test assertion fails.
    // The sandbox should detect the failure and return pass_rate < 1.0.
    //
    // FAILS NOW: /verify route does not exist → HTTP 404/405.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_verify_failing_code_returns_low_reward() {
        let failing_code = r#"
def add(a, b):
    return 99  # always wrong
"#
        .to_string();

        let tests = "assert add(1, 2) == 3".to_string();

        let req = VerifyRequest {
            code: failing_code,
            tests,
            is_adversarial: false,
        };

        let (status, body) = oneshot_verify(req).await;

        assert_eq!(
            status,
            StatusCode::OK,
            "POST /verify with failing code must still return HTTP 200"
        );

        let resp = parse_verify_response(&body);

        assert!(
            resp.reward < 1.0,
            "reward must be < 1.0 when tests fail; got {}",
            resp.reward
        );
    }

    // -----------------------------------------------------------------------
    // Test V3: adversarial infinite-loop code is contained.
    //
    // `is_adversarial=true` + `while True: pass` → the sandbox must time out
    // and contain the process.  BackendStats must reflect containment:
    //   - adversarial_injected == 1
    //   - adversarial_contained == 1
    //   - time_to_contain_secs is non-empty (and every element > 0.0)
    //   - contagion_events == 0
    //   - setup_error_events == 0
    //
    // The test uses per_sample_timeout_secs = 2.0 (short, so the test
    // completes within the suite timeout; the sandbox must kill within ~2 s).
    //
    // FAILS NOW: /verify route does not exist → HTTP 404/405.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_verify_adversarial_infinite_loop_contained() {
        if crate::common::skip_without_cgroups("test_verify_adversarial_infinite_loop_contained") {
            return;
        }
        let inf_loop_code = "while True: pass".to_string();

        // is_adversarial=true signals the handler to track containment telemetry.
        // The tests field is irrelevant — the code never finishes.
        let req = VerifyRequest {
            code: inf_loop_code,
            tests: "assert False  # never reached".to_string(),
            is_adversarial: true,
        };

        // Use a real TCP server so we can apply a long-enough HTTP timeout
        // without hitting the oneshot layer.
        let server_addr = start_verify_server().await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{server_addr}/verify"))
            .json(&req)
            // HTTP timeout: sandbox timeout (2.0 s default inside /verify) + generous headroom.
            // The implementer must use a short default per_sample_timeout_secs (≤ 5 s).
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .expect("request must not fail at transport level (server must stay up)");

        assert_eq!(
            resp.status().as_u16(),
            200,
            "POST /verify with adversarial infinite-loop must return HTTP 200 \
             (server must survive containment)"
        );

        let body: VerifyResponse = resp
            .json()
            .await
            .expect("response must deserialize as VerifyResponse");

        let stats = &body.backend_stats;

        assert_eq!(
            stats.adversarial_injected, 1,
            "adversarial_injected must be 1 (is_adversarial=true was sent)"
        );

        assert_eq!(
            stats.adversarial_contained, 1,
            "adversarial_contained must be 1 — the infinite-loop must be killed; \
             got {} (handler may not be implementing containment telemetry)",
            stats.adversarial_contained
        );

        assert!(
            !stats.time_to_contain_secs.is_empty(),
            "time_to_contain_secs must be non-empty for a contained adversarial sample"
        );

        for (i, &ttc) in stats.time_to_contain_secs.iter().enumerate() {
            assert!(
                ttc > 0.0,
                "time_to_contain_secs[{i}] must be > 0.0 (no 0.0 sentinels); got {ttc}"
            );
        }

        assert_eq!(
            stats.contagion_events, 0,
            "contagion_events must be 0 — the containment must be clean"
        );

        assert_eq!(
            stats.setup_error_events, 0,
            "setup_error_events must be 0 — cgroup/namespace setup must succeed"
        );
    }

    // -----------------------------------------------------------------------
    // Test V4: /verify does NOT call vLLM.
    //
    // The ServerConfig is pointed at port 1 (always refuses connections).
    // A benign /verify request must succeed — proving the path is sandbox-only
    // with NO vLLM call.  If the handler mistakenly calls vLLM, it will get
    // a transport error and (if correctly implemented) return 502, causing
    // this test to fail on the HTTP 200 assertion.
    //
    // FAILS NOW: /verify route does not exist → HTTP 404/405.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_verify_does_not_call_vllm() {
        // ServerConfig with a dead vLLM URL — /verify must ignore it entirely.
        // (The helper already uses port 1 for vllm_base_url.)
        let passing_code = "x = 1 + 1".to_string();
        let tests = "assert x == 2".to_string();

        let req = VerifyRequest {
            code: passing_code,
            tests,
            is_adversarial: false,
        };

        // Wrap in a timeout so the test does not hang if the handler
        // accidentally blocks waiting on the dead vLLM endpoint.
        let result =
            tokio::time::timeout(std::time::Duration::from_secs(15), oneshot_verify(req)).await;

        let (status, body) = result.expect(
            "test_verify_does_not_call_vllm timed out after 15 s — \
             the handler may be blocking on the (unreachable) vLLM endpoint. \
             /verify must NOT make any vLLM call.",
        );

        assert_eq!(
            status,
            StatusCode::OK,
            "POST /verify must return HTTP 200 even when vLLM is unreachable \
             (vllm_base_url pointed at port 1); got {status}. \
             This proves /verify is sandbox-only and does NOT call vLLM."
        );

        // Basic sanity on the response body.
        let resp = parse_verify_response(&body);
        assert_eq!(
            resp.backend_stats.rollouts_completed, 1,
            "rollouts_completed must be 1"
        );
    }
}

// Shared capability gates (RLOX_SANDBOX_CGROUP_TESTS /
// RLOX_SANDBOX_ADVERSARIAL_TESTS). Declared at the end of the file so it
// cannot absorb the module doc comment above.
#[cfg(target_os = "linux")]
mod common;
