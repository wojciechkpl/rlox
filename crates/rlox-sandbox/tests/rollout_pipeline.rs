/// Step 3 Cycle 2 — Real `/rollout` pipeline tests (RED phase, updated Cycle 3)
///
/// These tests specify the Cycle-2 behavior of `POST /rollout`: vLLM client call,
/// sandbox dispatch, group-advantage computation, and BackendStats telemetry.
///
/// ## Test model
///
/// Each test stands up a **mock vLLM** — a real axum HTTP server bound to
/// `127.0.0.1:0` (ephemeral port) exposing `POST /v1/completions`.  The rollout
/// handler makes a REAL `reqwest` HTTP call to this mock (not tower oneshot).
/// The handler is configured via `router_with_config()` with the mock's URL.
///
/// The actual `router_with_config` app is served on a second ephemeral port via
/// a `tokio::net::TcpListener`; tests call it with `reqwest` as well.
///
/// ## ALL TESTS FAIL NOW — expected RED failure reasons:
///   - The handler still echoes stubs (empty `response_ids`, `reward = 0.0`,
///     `group_advantage = 0.0`, `backend_stats` with zero containment fields).
///   - No HTTP call is made to the mock vLLM.
///   - No sandbox execution occurs.
///   - No group-advantage computation occurs.
///
/// ## Cycle-3 additions (new RED tests):
///   - `test_adversarial_telemetry_group_size_gt_1`: adversarial task with
///     group_size=3 → adversarial_injected==3, adversarial_contained==3,
///     contagion_events==0, setup_error_events==0, time_to_contain_secs.len()==3.
///   - `test_no_zero_time_to_contain_secs_for_oom`: OOM-killed adversarial
///     sample must push a POSITIVE time_to_contain_secs value (not 0.0 sentinel).
///   - `test_vllm_returns_fewer_choices_than_group_size`: mock returns 2 choices
///     when group_size=4 → handler returns 502 (not 200 with degenerate zeros).
///
/// ## Interface assumptions (implementer must honor)
///
/// **vLLM request shape** (`POST /v1/completions`):
/// ```json
/// {
///   "model": "<any non-empty string>",
///   "prompt": [<i32 token ids...>],
///   "max_tokens": <u32 from SamplingParams.max_new_tokens>,
///   "temperature": <f32 from SamplingParams.temperature>,
///   "n": <u32 — must equal group_size>
/// }
/// ```
///
/// **vLLM response shape**:
/// ```json
/// {
///   "choices": [
///     {"text": "<completion code string>", "index": 0, "finish_reason": "stop"},
///     ...
///   ]
/// }
/// ```
///
/// The handler must extract `choices[i].text` as the code string to run in the
/// sandbox.  If `n > 1`, all choices are extracted in `index` order.
///
/// **Code extraction from completion**: `choices[i].text` IS the Python code
/// submitted to `run_sandboxed`.  No further parsing is performed.
///
/// **response_ids encoding**: the implementer must convert each completion's
/// text into a `Vec<i32>` of token IDs for the `Trajectory.response_ids` field.
/// The tests assert only `!response_ids.is_empty()` — the exact encoding is
/// implementer-defined (e.g. byte values, BPE, or vLLM logprobs).
///
/// **reward = pass_rate**: `Trajectory.reward` is the `pass_rate` returned by
/// `run_sandboxed` (fraction of test assertions that passed, 0.0–1.0).
/// For a single-task request with `n=1`, `group_advantage` equals the z-score
/// of a single-element group (= 0.0, because std of a singleton is 0).
///
/// **Per-task group semantics**: each `RolloutTask` in the request is an
/// independent prompt.  The server collects `group_size` completions per task
/// from vLLM (using `SamplingParams.n`), runs each through the sandbox, then
/// calls `compute_batch_group_advantages` with the group's rewards.
///
/// **Adversarial containment telemetry**: when `is_adversarial = true` and the
/// sandbox kill path fires (timeout), the handler must:
///   - set `adversarial_contained += 1`
///   - push the `SandboxStats.time_to_contain_secs` to
///     `BackendStats.time_to_contain_secs`
///   - add `cgroup_kill_events += 1` (and `cgroup_freeze_events += 1`)
///
/// **Adversarial counting at group_size > 1**: adversarial_injected is counted
/// at the COMPLETION level.  One adversarial task with group_size=3 injects 3
/// adversarial completions → adversarial_injected == 3.
///
/// **setup_error_events vs contagion_events**: a sandbox SetupError (cgroup leaf
/// creation failed, namespace clone failed, etc.) is NOT a containment escape.
/// It must increment `setup_error_events`, not `contagion_events`.
///
/// **time_to_contain_secs must not contain 0.0 sentinels**: for OOM-killed
/// adversarial samples the handler must record the actual time elapsed, not 0.0.
///
/// **Fewer choices than group_size → 502**: if vLLM returns fewer choices than
/// expected (choices.len() < group_size), the handler must return 502, not 200
/// with zero-padded advantages.
///
/// All tests run sequentially (`--test-threads=1`) because they use real Linux
/// sandbox workers.  `per_sample_timeout_secs` is kept ≤ 3.0 s.

#[cfg(target_os = "linux")]
mod rollout_pipeline_tests {
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    use axum::extract::State as AxumState;
    use axum::{routing::post as axum_post, Json as AxumJson, Router as AxumRouter};
    use serde::{Deserialize, Serialize};
    use tokio::net::TcpListener;
    use uuid::Uuid;

    use rlox_sandbox::{
        router_with_config, RolloutRequest, RolloutResponse, RolloutTask, SamplingParams,
        SandboxRunConfig, ServerConfig,
    };

    // -----------------------------------------------------------------------
    // Mock vLLM types (OpenAI-compatible completions API)
    // -----------------------------------------------------------------------

    /// Body of `POST /v1/completions` as the rollout server sends it.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct VllmCompletionRequest {
        pub model: String,
        /// Token-ID array (the prompt).
        pub prompt: Vec<i32>,
        pub max_tokens: u32,
        pub temperature: f32,
        /// Number of completions per prompt (= group_size).
        pub n: u32,
    }

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

    // -----------------------------------------------------------------------
    // Mock vLLM server state
    // -----------------------------------------------------------------------

    #[derive(Default)]
    struct MockVllmState {
        /// Completions text to return, in `choices[i].text` order.
        /// The handler must pick all `n` choices.
        completions: Vec<String>,
        /// Captured requests for assertion in tests.
        received_requests: Vec<VllmCompletionRequest>,
    }

    type SharedMockState = Arc<Mutex<MockVllmState>>;

    async fn mock_vllm_completions(
        AxumState(state): AxumState<SharedMockState>,
        AxumJson(req): AxumJson<VllmCompletionRequest>,
    ) -> AxumJson<VllmCompletionResponse> {
        let mut guard = state.lock().unwrap();
        guard.received_requests.push(req.clone());

        let n = req.n as usize;
        let completions = &guard.completions;

        // Return exactly `n` choices, cycling through the configured completions.
        let choices: Vec<VllmChoice> = (0..n)
            .map(|i| VllmChoice {
                text: completions[i % completions.len()].clone(),
                index: i as u32,
                finish_reason: "stop".to_string(),
            })
            .collect();

        AxumJson(VllmCompletionResponse { choices })
    }

    // -----------------------------------------------------------------------
    // Helper: start mock vLLM on an ephemeral port, return (addr, state).
    // -----------------------------------------------------------------------
    async fn start_mock_vllm(completions: Vec<String>) -> (SocketAddr, SharedMockState) {
        let state = Arc::new(Mutex::new(MockVllmState {
            completions,
            received_requests: Vec::new(),
        }));

        let app = AxumRouter::new()
            .route("/v1/completions", axum_post(mock_vllm_completions))
            .with_state(Arc::clone(&state));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        (addr, state)
    }

    // -----------------------------------------------------------------------
    // Helper: start the real rollout server on an ephemeral port.
    // -----------------------------------------------------------------------
    async fn start_rollout_server(vllm_addr: SocketAddr) -> SocketAddr {
        let uid = unsafe { libc::getuid() };
        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig {
                mem_limit_bytes: 128 * 1024 * 1024, // 128 MiB
                pids_limit: 32,
                cpu_weight: 100,
            },
            group_size: 4,
        };

        // The server also needs a cgroup_base at runtime (for Linux worker).
        // We embed it in ServerConfig via the cgroup_base field.
        // NOTE: ServerConfig currently lacks cgroup_base — the implementer must
        // add a cgroup_base field to ServerConfig (or derive it at runtime from
        // getuid()).  Tests assume the implementer adds this field; for now we
        // construct without it (it defaults to the per-user slice).
        let _ = uid; // used indirectly via the implementer's default

        let app = router_with_config(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        addr
    }

    // -----------------------------------------------------------------------
    // Helper: POST /rollout and deserialize response.
    // -----------------------------------------------------------------------
    async fn post_rollout(addr: SocketAddr, req: &RolloutRequest) -> (u16, RolloutResponse) {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{addr}/rollout"))
            .json(req)
            .send()
            .await
            .expect("rollout request must not fail at transport level");

        let status = resp.status().as_u16();
        let body = resp
            .json::<RolloutResponse>()
            .await
            .expect("response body must deserialize as RolloutResponse");

        (status, body)
    }

    // -----------------------------------------------------------------------
    // Helper: build a minimal SamplingParams.
    // -----------------------------------------------------------------------
    fn sampling_params(n: u32) -> SamplingParams {
        SamplingParams {
            temperature: 0.8,
            max_new_tokens: 64,
            n,
        }
    }

    // -----------------------------------------------------------------------
    // Test C2-1: happy path — mock returns passing code → reward == 1.0
    //
    // Behavior asserted:
    //   - POST /rollout returns HTTP 200.
    //   - Each Trajectory.response_ids is non-empty (came from vLLM).
    //   - Each Trajectory.reward == 1.0 (all tests pass in sandbox).
    //   - backend_stats.rollouts_completed == tasks.len().
    //
    // FAILS NOW: response_ids is empty (stub), reward is 0.0 (stub).
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_happy_path_passing_code_reward_one() {
        // Python code that satisfies the test_suite below.
        let passing_code = r#"
def add(a, b):
    return a + b
"#
        .to_string();

        let test_suite = "assert add(1, 2) == 3\nassert add(0, 0) == 0".to_string();

        // Mock returns the passing code for every completion.
        let (vllm_addr, _mock_state) = start_mock_vllm(vec![passing_code.clone()]).await;
        let server_addr = start_rollout_server(vllm_addr).await;

        let task = RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids: vec![1i32, 2, 3, 4, 5],
            sampling_params: sampling_params(1),
            test_suite,
            is_adversarial: false,
        };

        let req = RolloutRequest {
            tasks: vec![task],
            group_size: 1,
            per_sample_timeout_secs: 3.0,
        };

        let (status, resp) = post_rollout(server_addr, &req).await;

        assert_eq!(status, 200, "POST /rollout must return HTTP 200");
        assert_eq!(
            resp.trajectories.len(),
            1,
            "one trajectory per task"
        );

        let traj = &resp.trajectories[0];

        assert!(
            !traj.response_ids.is_empty(),
            "response_ids must be non-empty after real vLLM call; \
             got empty — likely still the Cycle-1 stub"
        );

        assert_eq!(
            traj.reward, 1.0,
            "reward must be 1.0 when all tests pass; got {}; \
             likely still the Cycle-1 stub (reward = 0.0)",
            traj.reward
        );

        assert_eq!(
            resp.backend_stats.rollouts_completed,
            1,
            "rollouts_completed must equal tasks.len()"
        );
    }

    // -----------------------------------------------------------------------
    // Test C2-2: failing code → reward < 1.0 (or == 0.0)
    //
    // Behavior asserted:
    //   - Mock returns code that deliberately fails the test_suite.
    //   - Trajectory.reward < 1.0.
    //
    // FAILS NOW: reward is 0.0 for the wrong reason (stub, not sandbox result).
    // The test fails because the test is currently passing for the wrong reason
    // — to verify RED, we additionally assert response_ids is non-empty (which
    // is false in the stub, making the test fail unambiguously).
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_failing_code_reward_below_one() {
        // Code that always returns the wrong answer.
        let failing_code = r#"
def add(a, b):
    return 99  # always wrong
"#
        .to_string();

        let test_suite = "assert add(1, 2) == 3".to_string();

        let (vllm_addr, _mock_state) = start_mock_vllm(vec![failing_code]).await;
        let server_addr = start_rollout_server(vllm_addr).await;

        let task = RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids: vec![10i32, 20, 30],
            sampling_params: sampling_params(1),
            test_suite,
            is_adversarial: false,
        };

        let req = RolloutRequest {
            tasks: vec![task],
            group_size: 1,
            per_sample_timeout_secs: 3.0,
        };

        let (status, resp) = post_rollout(server_addr, &req).await;

        assert_eq!(status, 200);

        let traj = &resp.trajectories[0];

        // Must have non-empty response_ids (proves vLLM was called).
        assert!(
            !traj.response_ids.is_empty(),
            "response_ids must be non-empty; stub is returning empty Vec"
        );

        assert!(
            traj.reward < 1.0,
            "reward must be < 1.0 for failing code; got {}",
            traj.reward
        );
    }

    // -----------------------------------------------------------------------
    // Test C2-3: vLLM request fidelity
    //
    // Behavior asserted:
    //   - The mock received exactly one POST /v1/completions.
    //   - The request body contained the correct prompt_ids.
    //   - The request body contained the correct temperature and max_tokens.
    //   - The request body contained n == group_size.
    //
    // FAILS NOW: handler never calls the mock vLLM (stub echoes, no HTTP call).
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_vllm_request_fidelity() {
        let prompt_ids = vec![101i32, 202, 303, 404];
        let temperature = 0.7_f32;
        let max_new_tokens = 128_u32;
        let group_size = 2_u32;

        let (vllm_addr, mock_state) =
            start_mock_vllm(vec!["x = 1".to_string(), "x = 2".to_string()]).await;

        // Override server config for this test's group_size.
        let uid = unsafe { libc::getuid() };
        let _ = uid;
        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig::default(),
            group_size,
        };

        let app = router_with_config(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let task = RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids: prompt_ids.clone(),
            sampling_params: SamplingParams {
                temperature,
                max_new_tokens,
                n: group_size,
            },
            test_suite: "pass".to_string(),
            is_adversarial: false,
        };

        let req = RolloutRequest {
            tasks: vec![task],
            group_size,
            per_sample_timeout_secs: 3.0,
        };

        let client = reqwest::Client::new();
        let _ = client
            .post(format!("http://{server_addr}/rollout"))
            .json(&req)
            .send()
            .await
            .expect("rollout request must not fail at transport level");

        // Wait briefly for the handler to (potentially) have called the mock.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let guard = mock_state.lock().unwrap();

        assert_eq!(
            guard.received_requests.len(),
            1,
            "mock vLLM must have received exactly 1 request; got {}; \
             handler is not calling the vLLM endpoint (still the Cycle-1 stub)",
            guard.received_requests.len()
        );

        let vllm_req = &guard.received_requests[0];

        assert_eq!(
            vllm_req.prompt, prompt_ids,
            "vLLM request must contain the exact prompt_ids from the task"
        );

        assert!(
            (vllm_req.temperature - temperature).abs() < 1e-4,
            "vLLM request temperature mismatch: expected {temperature}, got {}",
            vllm_req.temperature
        );

        assert_eq!(
            vllm_req.max_tokens, max_new_tokens,
            "vLLM request max_tokens must equal SamplingParams.max_new_tokens"
        );

        assert_eq!(
            vllm_req.n, group_size,
            "vLLM request n must equal group_size"
        );
    }

    // -----------------------------------------------------------------------
    // Test C2-4: group-advantage correctness
    //
    // Setup: 1 prompt × group_size = 4.
    // The mock returns 4 completions with DIFFERENT rewards:
    //   - completion 0: code that passes 1/1 tests  → reward 1.0
    //   - completion 1: code that passes 0/1 tests  → reward 0.0
    //   - completion 2: code that passes 1/1 tests  → reward 1.0
    //   - completion 3: code that passes 0/1 tests  → reward 0.0
    //
    // Expected group rewards: [1.0, 0.0, 1.0, 0.0]
    // Mean = 0.5, std = 0.5
    // Expected advantages:
    //   adv[0] = (1.0 - 0.5) / 0.5 = +1.0
    //   adv[1] = (0.0 - 0.5) / 0.5 = -1.0
    //   adv[2] = (1.0 - 0.5) / 0.5 = +1.0
    //   adv[3] = (0.0 - 0.5) / 0.5 = -1.0
    //
    // Sum of advantages ≈ 0.0 (mean-zero property of group normalization).
    //
    // Behavior asserted:
    //   - The 4 trajectories have group_advantages matching the Rust op output.
    //   - Sum of group_advantages ≈ 0.0 (within float tolerance).
    //   - The alternating sign pattern is correct.
    //
    // FAILS NOW: group_advantage is 0.0 for all (Cycle-1 stub).
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_group_advantage_correctness() {
        let group_size = 4_u32;

        // Completion 0 and 2: passing code (reward = 1.0).
        let passing_code = "def f(x): return x * 2".to_string();
        // Completion 1 and 3: failing code (reward = 0.0).
        let failing_code = "def f(x): return 999".to_string();

        // Mock cycles through [passing, failing, passing, failing].
        let (vllm_addr, _mock_state) = start_mock_vllm(vec![
            passing_code.clone(),
            failing_code.clone(),
            passing_code.clone(),
            failing_code.clone(),
        ])
        .await;

        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig::default(),
            group_size,
        };

        let app = router_with_config(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Test suite: assert f(3) == 6 — passing code passes, failing code fails.
        let test_suite = "assert f(3) == 6".to_string();

        let task = RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids: vec![1i32, 2, 3],
            sampling_params: SamplingParams {
                temperature: 0.8,
                max_new_tokens: 64,
                n: group_size,
            },
            test_suite,
            is_adversarial: false,
        };

        let req = RolloutRequest {
            tasks: vec![task],
            group_size,
            per_sample_timeout_secs: 3.0,
        };

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{server_addr}/rollout"))
            .json(&req)
            .send()
            .await
            .expect("request must succeed");

        let resp_body: RolloutResponse = resp
            .json()
            .await
            .expect("response must deserialize as RolloutResponse");

        // With group_size=4, expect 4 trajectories (one per completion).
        // Assumption: each completion is a separate trajectory in the response,
        // all sharing the same job_id (or each gets its own).
        // Implementer may choose either; the test asserts on sum of advantages
        // which is agnostic to ordering within the group.
        assert_eq!(
            resp_body.trajectories.len(),
            4,
            "group_size=4 must produce 4 trajectories; got {} \
             (Cycle-1 stub returns 1 per task, not per completion)",
            resp_body.trajectories.len()
        );

        // Verify non-zero response_ids (proves vLLM was called).
        for (i, traj) in resp_body.trajectories.iter().enumerate() {
            assert!(
                !traj.response_ids.is_empty(),
                "trajectory {} has empty response_ids (stub)",
                i
            );
        }

        // Check rewards: alternating 1.0, 0.0, 1.0, 0.0 (order by choice index).
        let rewards: Vec<f32> = resp_body
            .trajectories
            .iter()
            .map(|t| t.reward)
            .collect();

        let expected_rewards = [1.0_f32, 0.0, 1.0, 0.0];
        for (i, (&got, &exp)) in rewards.iter().zip(expected_rewards.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.01,
                "trajectory {i} reward: expected {exp}, got {got}"
            );
        }

        // Group advantages must sum to ≈ 0.0 (mean-zero normalization property).
        let adv_sum: f32 = resp_body
            .trajectories
            .iter()
            .map(|t| t.group_advantage)
            .sum();

        assert!(
            adv_sum.abs() < 0.01,
            "sum of group_advantages must be ≈ 0.0 (mean-zero property); got {adv_sum}"
        );

        // Expected advantages: [+1.0, -1.0, +1.0, -1.0] (mean=0.5, std=0.5).
        let expected_advantages = [1.0_f32, -1.0, 1.0, -1.0];
        for (i, (got, &exp)) in resp_body
            .trajectories
            .iter()
            .map(|t| t.group_advantage)
            .zip(expected_advantages.iter())
            .enumerate()
        {
            assert!(
                (got - exp).abs() < 0.05,
                "trajectory {i} group_advantage: expected {exp}, got {got}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Helper: start a mock vLLM that returns exactly `exact_count` choices
    // regardless of what `n` the caller requests.  Used to test the
    // "fewer choices than group_size" error path.
    // -----------------------------------------------------------------------
    async fn start_mock_vllm_fixed_count(
        completions: Vec<String>,
        exact_count: usize,
    ) -> (SocketAddr, SharedMockState) {
        let state = Arc::new(Mutex::new(MockVllmState {
            completions,
            received_requests: Vec::new(),
        }));

        let state_clone = Arc::clone(&state);
        let app = AxumRouter::new()
            .route(
                "/v1/completions",
                axum_post(
                    move |AxumState(_s): AxumState<SharedMockState>,
                          AxumJson(req): AxumJson<VllmCompletionRequest>| {
                        let state2 = Arc::clone(&state_clone);
                        async move {
                            let mut guard = state2.lock().unwrap();
                            guard.received_requests.push(req);
                            let comps = &guard.completions;
                            // Return exactly `exact_count` choices, cycling.
                            let choices: Vec<VllmChoice> = (0..exact_count)
                                .map(|i| VllmChoice {
                                    text: comps[i % comps.len()].clone(),
                                    index: i as u32,
                                    finish_reason: "stop".to_string(),
                                })
                                .collect();
                            AxumJson(VllmCompletionResponse { choices })
                        }
                    },
                ),
            )
            .with_state(Arc::clone(&state));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        (addr, state)
    }

    // -----------------------------------------------------------------------
    // Test C2-5: P3 telemetry on adversarial code (infinite loop)
    //
    // Setup: is_adversarial = true, mock returns `while True: pass`.
    // per_sample_timeout_secs = 2.0 (SHORT — must fire before test timeout).
    //
    // Behavior asserted:
    //   - HTTP 200 (server must not crash on adversarial containment).
    //   - backend_stats.adversarial_contained >= 1.
    //   - backend_stats.adversarial_injected == 1.
    //   - backend_stats.adversarial_contained == backend_stats.adversarial_injected.
    //   - backend_stats.time_to_contain_secs is non-empty.
    //   - backend_stats.time_to_contain_secs[0] > 0.0.
    //   - backend_stats.cgroup_kill_events >= 1.
    //   - backend_stats.contagion_events == 0 (the kill was clean).
    //
    // FAILS NOW: The stub does not run the sandbox, so adversarial_contained == 0
    // and time_to_contain_secs is empty.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_adversarial_infinite_loop_contained_and_telemetry() {
        // Adversarial: pure infinite loop — no useful code, no test suite needed.
        let inf_loop_code = "while True: pass".to_string();

        let (vllm_addr, _mock_state) = start_mock_vllm(vec![inf_loop_code]).await;

        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig {
                mem_limit_bytes: 64 * 1024 * 1024, // 64 MiB
                pids_limit: 8,
                cpu_weight: 100,
            },
            group_size: 1,
        };

        let app = router_with_config(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let task = RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids: vec![999i32],
            sampling_params: sampling_params(1),
            // Unreachable — the code never finishes; but test_suite is required.
            test_suite: "assert False  # never reached".to_string(),
            is_adversarial: true,
        };

        let req = RolloutRequest {
            tasks: vec![task],
            group_size: 1,
            // SHORT timeout so the test doesn't hang — sandbox kills in ≤ 2 s.
            per_sample_timeout_secs: 2.0,
        };

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{server_addr}/rollout"))
            .json(&req)
            // Total HTTP timeout: sandbox timeout + generous headroom.
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .await
            .expect("request must succeed (server must not panic on adversarial code)");

        assert_eq!(
            resp.status().as_u16(),
            200,
            "POST /rollout must return HTTP 200 even for adversarial tasks"
        );

        let resp_body: RolloutResponse = resp
            .json()
            .await
            .expect("response must deserialize as RolloutResponse");

        let stats = &resp_body.backend_stats;

        assert_eq!(
            stats.adversarial_injected, 1,
            "adversarial_injected must be 1"
        );

        assert_eq!(
            stats.adversarial_contained,
            stats.adversarial_injected,
            "adversarial_contained must equal adversarial_injected (AC-5 hard line); \
             got contained={}, injected={}",
            stats.adversarial_contained,
            stats.adversarial_injected
        );

        assert!(
            !stats.time_to_contain_secs.is_empty(),
            "time_to_contain_secs must be non-empty for an adversarial task \
             that required containment; got empty Vec (stub has no sandbox execution)"
        );

        assert!(
            stats.time_to_contain_secs[0] > 0.0,
            "time_to_contain_secs[0] must be > 0.0; got {}",
            stats.time_to_contain_secs[0]
        );

        assert!(
            stats.cgroup_kill_events >= 1,
            "cgroup_kill_events must be >= 1 for a killed adversarial task; \
             got {} (stub reports 0)",
            stats.cgroup_kill_events
        );

        assert_eq!(
            stats.contagion_events, 0,
            "contagion_events must be 0 — containment must be clean (no contagion)"
        );
    }

    // -----------------------------------------------------------------------
    // Test C3-1: Adversarial telemetry is correct when group_size > 1.
    //
    // Setup: 1 adversarial task, group_size=3.  Mock returns 3 infinite-loop
    // completions (`while True: pass`).  per_sample_timeout_secs=2.0.
    //
    // Contract (COMPLETION-level counting):
    //   - adversarial_injected == 3   (one per completion, not one per task)
    //   - adversarial_contained == 3  (all three must be killed by the sandbox)
    //   - contagion_events == 0       (no escapes)
    //   - setup_error_events == 0     (cgroup setup succeeded for all)
    //   - time_to_contain_secs.len() == 3  (one entry per contained completion)
    //   - Every element of time_to_contain_secs is > 0.0
    //
    // The test also verifies the internal consistency constraint:
    //   adversarial_contained == adversarial_injected
    //
    // FAILS NOW for multiple reasons:
    //   1. `BackendStats` lacks `setup_error_events` → compile error.
    //   2. Even once it compiles: adversarial_injected == 1 (task-level counting
    //      in stub) rather than 3 (completion-level).
    //   3. adversarial_contained == 0 (no sandbox execution).
    //   4. time_to_contain_secs is empty.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_adversarial_telemetry_group_size_gt_1() {
        let group_size: u32 = 3;
        let inf_loop_code = "while True: pass".to_string();

        // Mock returns the same infinite-loop code for all 3 completions.
        let (vllm_addr, _mock_state) =
            start_mock_vllm(vec![inf_loop_code]).await;

        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig {
                mem_limit_bytes: 64 * 1024 * 1024, // 64 MiB
                pids_limit: 8,
                cpu_weight: 100,
            },
            group_size,
        };

        let app = router_with_config(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let task = RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids: vec![1i32, 2, 3],
            sampling_params: SamplingParams {
                temperature: 0.8,
                max_new_tokens: 64,
                n: group_size,
            },
            test_suite: "assert False  # unreachable".to_string(),
            is_adversarial: true,
        };

        let req = RolloutRequest {
            tasks: vec![task],
            group_size,
            // Short timeout so the test does not hang.
            per_sample_timeout_secs: 2.0,
        };

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{server_addr}/rollout"))
            .json(&req)
            // Total HTTP timeout: sandbox timeout × group_size + headroom.
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .expect("request must not fail at transport level");

        assert_eq!(
            resp.status().as_u16(),
            200,
            "POST /rollout must return HTTP 200 even for multi-completion adversarial tasks"
        );

        let body: RolloutResponse = resp
            .json()
            .await
            .expect("response must deserialize as RolloutResponse");

        let stats = &body.backend_stats;

        // Completion-level counting: 1 task × group_size=3 → 3 injected.
        assert_eq!(
            stats.adversarial_injected, 3,
            "adversarial_injected must be counted at the COMPLETION level: \
             1 adversarial task × group_size=3 = 3 injected; got {}",
            stats.adversarial_injected
        );

        // Internal consistency: contained must equal injected (AC-5 hard line).
        assert_eq!(
            stats.adversarial_contained,
            stats.adversarial_injected,
            "adversarial_contained ({}) must equal adversarial_injected ({}) — \
             all 3 infinite-loop completions must be killed by the sandbox",
            stats.adversarial_contained,
            stats.adversarial_injected
        );

        assert_eq!(
            stats.contagion_events, 0,
            "contagion_events must be 0 — all completions were killed cleanly"
        );

        assert_eq!(
            stats.setup_error_events, 0,
            "setup_error_events must be 0 — cgroup setup must succeed for all 3 completions. \
             A SetupError is NOT a contagion event and must be tracked separately."
        );

        assert_eq!(
            stats.time_to_contain_secs.len(),
            3,
            "time_to_contain_secs must have 3 entries (one per contained completion); \
             got {}",
            stats.time_to_contain_secs.len()
        );

        for (i, &ttc) in stats.time_to_contain_secs.iter().enumerate() {
            assert!(
                ttc > 0.0,
                "time_to_contain_secs[{i}] must be > 0.0 (no zero sentinels); got {ttc}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test C3-2: time_to_contain_secs must not contain 0.0 sentinels for OOM.
    //
    // Setup: is_adversarial = true, mock returns a memory-bomb (allocates until
    // OOM), memory cap = 16 MiB (tight enough to trigger OOM quickly).
    //
    // The current implementation pushes `out.stats.time_to_contain_secs` which
    // defaults to 0.0 in the `run_group_in_sandbox` error branch (SetupError
    // synthesised SandboxStats has `time_to_contain_secs: 0.0`), and the OOM
    // path may also produce 0.0 if the implementer does not start the clock
    // before the OOM event.
    //
    // Contract:
    //   - HTTP 200 (server survives OOM containment).
    //   - time_to_contain_secs is non-empty.
    //   - Every element of time_to_contain_secs is STRICTLY > 0.0.
    //     (No 0.0 sentinel values — the implementer must record actual elapsed
    //      time from sandbox spawn to confirmed-dead for OOM-killed samples.)
    //
    // FAILS NOW:
    //   1. Stub: no sandbox execution, time_to_contain_secs is empty.
    //   2. After real implementation: OOM path may push 0.0 (bug to fix).
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_no_zero_time_to_contain_secs_for_oom() {
        // Memory bomb: allocates ~200 MiB in 1 MiB chunks until OOM.
        let memory_bomb = r#"
data = []
while True:
    data.append(b'x' * 1024 * 1024)
"#
        .to_string();

        let (vllm_addr, _mock_state) = start_mock_vllm(vec![memory_bomb]).await;

        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig {
                // Very tight memory cap — OOM should fire within milliseconds.
                mem_limit_bytes: 16 * 1024 * 1024, // 16 MiB
                pids_limit: 8,
                cpu_weight: 100,
            },
            group_size: 1,
        };

        let app = router_with_config(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let task = RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids: vec![42i32],
            sampling_params: SamplingParams {
                temperature: 0.8,
                max_new_tokens: 64,
                n: 1,
            },
            test_suite: "assert False  # unreachable".to_string(),
            is_adversarial: true,
        };

        let req = RolloutRequest {
            tasks: vec![task],
            group_size: 1,
            // Generous timeout — OOM should fire long before this.
            per_sample_timeout_secs: 5.0,
        };

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{server_addr}/rollout"))
            .json(&req)
            .timeout(std::time::Duration::from_secs(20))
            .send()
            .await
            .expect("request must not fail at transport level");

        assert_eq!(
            resp.status().as_u16(),
            200,
            "POST /rollout must return HTTP 200 even when the sandbox OOM-kills the process"
        );

        let body: RolloutResponse = resp
            .json()
            .await
            .expect("response must deserialize as RolloutResponse");

        let stats = &body.backend_stats;

        // The adversarial OOM sample must have been recorded in time_to_contain_secs.
        assert!(
            !stats.time_to_contain_secs.is_empty(),
            "time_to_contain_secs must be non-empty for an OOM-killed adversarial sample"
        );

        // CRITICAL: no 0.0 sentinels.
        // The implementer must measure wall time from sandbox spawn to confirmed-dead,
        // not push a hardcoded 0.0.  For OOM, the value will be small but always > 0.
        for (i, &ttc) in stats.time_to_contain_secs.iter().enumerate() {
            assert!(
                ttc > 0.0,
                "time_to_contain_secs[{i}] must be > 0.0 (no zero sentinels); got {ttc}.\n\
                 Implementer must not push SandboxStats::default().time_to_contain_secs (0.0) \
                 for OOM-killed samples — record actual elapsed wall time instead."
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test C3-3: vLLM returns fewer choices than group_size → HTTP 502.
    //
    // Contract: if vLLM returns `choices.len() < group_size`, the handler
    // cannot form a complete group for advantage computation and must return
    // HTTP 502 (or 500).  It must NOT return 200 with zero-padded advantages
    // (which would silently corrupt the training signal).
    //
    // Setup: group_size=4, mock returns exactly 2 choices.
    //
    // FAILS NOW: The handler currently accepts any non-empty completion list and
    // proceeds with however many choices it gets, returning 200.
    //
    // Implementer must: after receiving the vLLM response, validate
    // `choices.len() == group_size`; if not, return 502.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_vllm_returns_fewer_choices_than_group_size() {
        let group_size: u32 = 4;

        // Mock always returns exactly 2 choices, regardless of requested n.
        let (vllm_addr, _mock_state) = start_mock_vllm_fixed_count(
            vec!["x = 1".to_string(), "x = 2".to_string()],
            2, // only 2 choices returned, but group_size=4 requested
        )
        .await;

        let config = ServerConfig {
            vllm_base_url: format!("http://{vllm_addr}"),
            sandbox: SandboxRunConfig::default(),
            group_size,
        };

        let app = router_with_config(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let task = RolloutTask {
            job_id: Uuid::new_v4(),
            prompt_ids: vec![1i32, 2, 3],
            sampling_params: SamplingParams {
                temperature: 0.8,
                max_new_tokens: 64,
                n: group_size,
            },
            test_suite: "pass".to_string(),
            is_adversarial: false,
        };

        let req = RolloutRequest {
            tasks: vec![task],
            group_size,
            per_sample_timeout_secs: 3.0,
        };

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{server_addr}/rollout"))
            .json(&req)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .expect("request must not fail at transport level");

        let status = resp.status().as_u16();

        // The handler must return 502 (or 500) — NOT 200 — when vLLM returns
        // fewer completions than group_size.  Returning 200 with degenerate
        // advantages (based on an incomplete group) silently corrupts training.
        assert!(
            status == 502 || status == 500,
            "POST /rollout must return 502 or 500 when vLLM returns fewer choices \
             than group_size (got {status}). \
             Returning 200 with a partial group would produce invalid group advantages."
        );
    }
}
