/// Step 1d — run_sandboxed integration tests
///
/// Tests exercise the full `run_sandboxed` async entry point:
///   (a) benign code → SandboxExitStatus::Clean(0), pass_rate == 1.0,
///       wall_secs is recorded (> 0.0).
///   (b) infinite sleep → SandboxExitStatus::Timeout,
///       stats.cgroup_freeze_event == true, stats.cgroup_kill_event == true,
///       stats.time_to_contain_secs <= timeout_secs + 1.0.
///   (c) wall_secs is always set (> 0.0) regardless of exit path.
///
/// All tests are Linux-only and use `#[tokio::test]`.
///
/// Assumptions about the implementation the implementer must honor:
///   - `python3` is on PATH on wk-system.
///   - The cgroup base path is the per-user delegated slice.
///   - `SandboxConfig::cpu_weight` defaults to 100; any positive value is valid.
///   - `timeout_secs` of 3.0 is enough for a trivial `print("ok")` script.
///   - After Timeout, `time_to_contain_secs` is the wall-clock from timeout
///     detection to the moment `cgroup.kill` is written (not the total).
#[cfg(target_os = "linux")]
mod integration_run_sandboxed {
    use rlox_sandbox::worker::{run_sandboxed, SandboxConfig, SandboxExitStatus, SandboxInput};
    use std::path::PathBuf;
    use uuid::Uuid;

    // -----------------------------------------------------------------------
    // Helper: build a SandboxConfig that uses the per-user cgroup slice.
    // -----------------------------------------------------------------------
    fn default_config(timeout_secs: f64) -> SandboxConfig {
        let uid = unsafe { libc::getuid() };
        SandboxConfig {
            timeout_secs,
            mem_limit_bytes: 128 * 1024 * 1024, // 128 MiB
            pids_limit: 64,
            cpu_weight: 100,
            cgroup_base: PathBuf::from(format!("/sys/fs/cgroup/user.slice/user-{uid}.slice")),
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build a SandboxInput for a Python snippet + trivial test suite.
    // -----------------------------------------------------------------------
    fn make_input(code: &str, test_suite: &str) -> SandboxInput {
        SandboxInput {
            job_id: Uuid::new_v4(),
            code: code.to_owned(),
            test_suite: test_suite.to_owned(),
            language: "python".to_owned(),
            is_adversarial: false,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1d-a: benign code returns Clean(0) with pass_rate == 1.0.
    //
    // Code: defines a function `add(a, b)` that returns `a + b`.
    // Test suite: one assertion that verifies `add(1, 2) == 3`.
    // Expected: the test passes, exit status is Clean(0), pass_rate is 1.0.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_benign_code_returns_clean_exit_and_full_pass_rate() {
        let code = r#"
def add(a, b):
    return a + b
"#;
        let test_suite = r#"
assert add(1, 2) == 3, "add(1, 2) should be 3"
"#;

        let config = default_config(10.0);
        let input = make_input(code, test_suite);

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed should not return Err for benign code");

        assert_eq!(
            output.exit_status,
            SandboxExitStatus::Clean(0),
            "expected Clean(0) for benign code, got {:?}",
            output.exit_status
        );
        assert!(
            (output.pass_rate - 1.0f32).abs() < 1e-6,
            "expected pass_rate == 1.0, got {}",
            output.pass_rate
        );
        assert!(
            output.stats.wall_secs > 0.0,
            "wall_secs must be > 0 for any execution, got {}",
            output.stats.wall_secs
        );
        assert!(
            !output.stats.cgroup_freeze_event,
            "cgroup_freeze_event must be false for a clean exit"
        );
        assert!(
            !output.stats.cgroup_kill_event,
            "cgroup_kill_event must be false for a clean exit"
        );
    }

    // -----------------------------------------------------------------------
    // Test 1d-b: infinite sleep returns Timeout with kill events set.
    //
    // Code: `import time; time.sleep(9999)` — will never exit on its own.
    // Config: timeout_secs = 3.0.
    // Expected:
    //   - exit_status == Timeout
    //   - stats.cgroup_freeze_event == true
    //   - stats.cgroup_kill_event == true
    //   - stats.time_to_contain_secs <= timeout_secs + 1.0
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_infinite_sleep_returns_timeout_with_cgroup_kill_events() {
        let code = "import time\ntime.sleep(9999)\n";
        let test_suite = ""; // no test suite needed — process never reaches it

        let timeout_secs = 3.0f64;
        let config = default_config(timeout_secs);
        let input = {
            let mut i = make_input(code, test_suite);
            i.is_adversarial = true;
            i
        };

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed should not return Err even for timeout path");

        assert_eq!(
            output.exit_status,
            SandboxExitStatus::Timeout,
            "expected Timeout for infinite sleep, got {:?}",
            output.exit_status
        );
        assert!(
            output.stats.cgroup_freeze_event,
            "cgroup_freeze_event must be true after a Timeout kill"
        );
        assert!(
            output.stats.cgroup_kill_event,
            "cgroup_kill_event must be true after a Timeout kill"
        );
        assert!(
            output.stats.time_to_contain_secs <= timeout_secs + 1.0,
            "time_to_contain_secs {} must be <= timeout_secs ({}) + 1.0",
            output.stats.time_to_contain_secs,
            timeout_secs
        );
    }

    // -----------------------------------------------------------------------
    // Test 1d-c: wall_secs is always > 0, even for the timeout path.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_wall_secs_is_positive_for_timeout_path() {
        let code = "import time\ntime.sleep(9999)\n";
        let timeout_secs = 2.0f64;
        let config = default_config(timeout_secs);
        let input = make_input(code, "");

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed should return Ok even for timeout");

        assert!(
            output.stats.wall_secs > 0.0,
            "wall_secs must be > 0 even for the timeout path, got {}",
            output.stats.wall_secs
        );
        // wall_secs should be at least as long as the timeout.
        assert!(
            output.stats.wall_secs >= timeout_secs - 0.5,
            "wall_secs {} should be close to timeout_secs {timeout_secs}",
            output.stats.wall_secs
        );
    }

    // -----------------------------------------------------------------------
    // Test 1d-d: unsupported language returns SetupError.
    //
    // Assumption: the implementer must return SetupError (not panic) when
    // the `language` field is not "python".
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_unsupported_language_returns_setup_error_or_sandbox_error() {
        let config = default_config(5.0);
        let mut input = make_input("print('hi')", "");
        input.language = "brainfuck".to_owned();

        let result = run_sandboxed(input, &config).await;

        // The implementation may return Err(SandboxError::UnsupportedLanguage)
        // OR Ok(SandboxOutput { exit_status: SetupError(_), ... }).
        // Either is a valid contract; the test just checks no panic occurs.
        match result {
            Err(rlox_sandbox::SandboxError::UnsupportedLanguage(_)) => {
                // Correct: returned as a SandboxError.
            }
            Ok(output) => {
                assert!(
                    matches!(output.exit_status, SandboxExitStatus::SetupError(_)),
                    "expected SetupError for unsupported language, got {:?}",
                    output.exit_status
                );
            }
            Err(other) => {
                // Accept any SandboxError variant — the point is it didn't panic.
                let _ = other;
            }
        }
    }
}

#[cfg(target_os = "linux")]
extern crate libc;
