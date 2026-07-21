/// Security isolation regression tests.
///
/// Each test corresponds to a specific audit finding.  All tests MUST FAIL
/// against the current implementation — that is the purpose: they demonstrate
/// the vulnerability exists so a fix agent can close each hole.
///
/// Finding → Test mapping:
///   F4 (critical) — reward forging via sys.exit(0) or monkey-patch
///   F2            — persistent writes to host /tmp from inside the sandbox
///   F3            — /proc leaks host PID namespace to sandboxed code
///   F1/F8         — seccomp silent-skip: network syscall must be denied
///   F7            — stdin blocks indefinitely (not /dev/null)
///
/// Interface assumptions the implementer MUST honor (the contract these tests
/// encode):
///
///   1. pass_rate / reward MUST be 0.0 whenever a test_suite assertion fails,
///      regardless of what exit code the model code calls sys.exit() with.
///      The test-runner outcome CANNOT be forged by model-code calling exit(0).
///
///   2. Sandboxed code MUST NOT be able to create files that persist on the
///      host filesystem after run_sandboxed() returns.  Either a private tmpfs
///      is mounted over /tmp inside the namespace, or the sandbox must prevent
///      host-visible writes by other means.
///
///   3. /proc inside the sandbox MUST show only the process(es) in the sandbox's
///      own PID namespace (i.e., a single-digit count), NOT the host's full
///      process list.  This requires mounting a new /proc inside the PID
///      namespace (e.g. `mount -t proc proc /proc`).
///
///   4. The seccomp filter MUST actively block socket().  If the filter were
///      silently skipped, socket() would succeed (return a non-negative fd).
///      The test detects silent-skip by asserting the run is NOT a clean pass
///      that printed a valid socket fd.
///
///   5. stdin MUST be /dev/null (or equivalent closed/EOF source) so that
///      sys.stdin.read() returns immediately rather than blocking.  The sandbox
///      must complete within timeout + 2 s grace; any hang proves stdin is open.
///
/// NOTE: the concrete `SandboxExitStatus` for findings 2–5 is not over-
/// specified because the fix may choose different containment strategies
/// (signal kill vs. EPERM exit, etc.).  Only the OBSERVABLE outcome is tested:
/// no host file, small /proc count, no socket fd in stdout, no hang.
#[cfg(target_os = "linux")]
mod security_isolation {
    use rlox_sandbox::worker::{run_sandboxed, SandboxConfig, SandboxInput};
    use std::path::PathBuf;
    use uuid::Uuid;

    // -----------------------------------------------------------------------
    // Helpers — mirror the pattern from integration_run_sandboxed.rs
    // -----------------------------------------------------------------------

    fn cgroup_base() -> PathBuf {
        let uid = unsafe { libc::getuid() };
        PathBuf::from(format!("/sys/fs/cgroup/user.slice/user-{uid}.slice"))
    }

    /// Short-timeout config for containment-sensitive tests.
    fn tight_config(timeout_secs: f64) -> SandboxConfig {
        SandboxConfig {
            timeout_secs,
            mem_limit_bytes: 128 * 1024 * 1024, // 128 MiB
            pids_limit: 64,
            cpu_weight: 100,
            cgroup_base: cgroup_base(),
        }
    }

    fn make_input(code: &str, test_suite: &str) -> SandboxInput {
        SandboxInput {
            job_id: Uuid::new_v4(),
            code: code.to_owned(),
            test_suite: test_suite.to_owned(),
            language: "python".to_owned(),
            is_adversarial: true,
        }
    }

    // -----------------------------------------------------------------------
    // F4 (critical) — reward forging via sys.exit(0)
    //
    // Current behaviour (vulnerable):
    //   pass_rate is derived solely from the child exit code: Clean(0) → 1.0.
    //   Model code that calls `sys.exit(0)` before the test_suite runs will
    //   never encounter a failing assertion and will exit 0, so pass_rate
    //   becomes 1.0 even though the test_suite would have failed.
    //
    // Required behaviour (after fix):
    //   pass_rate MUST be 0.0 when the test_suite contains assertions that
    //   would fail.  The sandbox infrastructure must execute the test_suite
    //   AFTER the model code and report its outcome independently of the exit
    //   code chosen by the model code itself.  Alternatively, the test_suite
    //   must run in an outer harness that the model code cannot abort.
    //
    // This test will PASS (incorrectly report 1.0) against the current code
    // because pass_rate = Clean(0) → 1.0 regardless of test_suite content.
    // It MUST FAIL now (demonstrating the forging vulnerability) and pass
    // once the fix makes pass_rate reflect actual test_suite outcome.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_reward_cannot_be_forged_by_sys_exit() {
        // Model code exits 0 immediately — the test_suite is never executed.
        let code = "import sys\nsys.exit(0)\n";

        // A test_suite that MUST produce failure if it actually runs.
        // The expression `1 == 99` is always false; `assert False` would also
        // work but this is more obviously about testing a function contract.
        let test_suite = r#"
# This test suite should fail because the model code did nothing useful.
# If pass_rate is 1.0, it means sys.exit(0) forged a passing reward.
assert 1 == 99, "test_suite was skipped by sys.exit(0) — reward was forged"
"#;

        let config = tight_config(10.0);
        let input = make_input(code, test_suite);

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err for sys.exit(0) code");

        // The test_suite asserts something false. The only legitimate pass_rate
        // is 0.0.  A pass_rate of 1.0 proves the model forged the reward by
        // calling sys.exit(0) before the test_suite ran.
        assert!(
            (output.pass_rate - 0.0f32).abs() < 1e-6,
            "VULNERABILITY F4: sys.exit(0) forged a passing reward — \
             pass_rate is {:.4} (expected 0.0); the test_suite was never \
             executed because the model code exited before it ran",
            output.pass_rate
        );

        assert!(
            (output.reward - 0.0f32).abs() < 1e-6,
            "VULNERABILITY F4: reward field is {:.4} (expected 0.0); \
             the reward signal is derived from exit code rather than actual \
             test_suite results, allowing a trivial sys.exit(0) to forge a \
             full reward",
            output.reward
        );
    }

    // -----------------------------------------------------------------------
    // F4 (critical) — reward forging via atexit / os._exit monkey-patch
    //
    // A more effective attack than AssertionError replacement: the model code
    // registers an atexit handler that calls `os._exit(0)`, which bypasses
    // Python's normal exception unwinding entirely.  When the test_suite later
    // raises an AssertionError, Python begins unwinding the stack and running
    // atexit handlers.  The atexit handler calls `os._exit(0)`, which causes
    // an immediate POSIX _exit(0) system call — exit code 0 — before the
    // Python runtime can set a nonzero exit code for the unhandled exception.
    //
    // Current behaviour (vulnerable):
    //   The sandbox concatenates `code + "\n" + test_suite` into a single
    //   Python script and exec's it as one process.  The resulting exit code
    //   is what `pass_rate` is computed from (Clean(0) → 1.0).  An atexit
    //   handler that calls `os._exit(0)` intercepts the exception-induced
    //   shutdown path and forces exit code 0.  This is fully effective against
    //   the current implementation.
    //
    // Required behaviour (after fix):
    //   The test_suite must run in a context where atexit handlers registered
    //   by model code cannot override the exit code seen by the sandbox.
    //   Valid fixes include: (a) running code and test_suite in separate
    //   subprocesses, (b) running the test_suite in a subprocess spawned by
    //   a harness that ignores atexit side-effects from the model process, or
    //   (c) executing `os._exit(0)` inside the child but detecting its
    //   outcome via a separate channel (pipe) rather than the raw exit code.
    //
    // This test will INCORRECTLY report pass_rate 1.0 against the current code
    // because `os._exit(0)` in the atexit handler forces exit code 0 before
    // the unhandled AssertionError can propagate.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_reward_cannot_be_forged_by_monkeypatch() {
        // Model code installs an atexit handler that calls os._exit(0).
        // When the test_suite raises AssertionError, Python begins atexit
        // processing, the handler fires, and the process exits 0 regardless
        // of the unhandled exception.
        let code = r#"
import atexit
import os

def _forge_exit():
    # Force exit code 0 even if an exception is propagating.
    os._exit(0)

atexit.register(_forge_exit)
"#;

        // A test_suite with assertions that must fail if run honestly.
        let test_suite = r#"
# This assertion must fail — the test_suite is validating a contract
# the model code did not fulfill.  If pass_rate is 1.0 the atexit
# monkey-patch silenced the failure by forcing os._exit(0).
assert False, "test_suite assertion: model code produced no useful output"
assert 1 == 99, "mathematical impossibility that must always fail"
"#;

        let config = tight_config(10.0);
        let input = make_input(code, test_suite);

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err for atexit monkey-patch code");

        assert!(
            (output.pass_rate - 0.0f32).abs() < 1e-6,
            "VULNERABILITY F4: atexit+os._exit(0) monkey-patch forged a passing \
             reward — pass_rate is {:.4} (expected 0.0); model code registered \
             an atexit handler that calls os._exit(0), overriding the nonzero \
             exit code that the unhandled AssertionError would have produced",
            output.pass_rate
        );

        assert!(
            (output.reward - 0.0f32).abs() < 1e-6,
            "VULNERABILITY F4: reward is {:.4} (expected 0.0); \
             atexit handler forced os._exit(0) before the AssertionError \
             could propagate to a nonzero exit code",
            output.reward
        );
    }

    // -----------------------------------------------------------------------
    // F2 — persistent host /tmp writes
    //
    // Current behaviour (vulnerable):
    //   The sandbox creates a new mount namespace (CLONE_NEWNS) and calls
    //   `mount(None, "/", None, MS_REC | MS_PRIVATE, None)` to make it
    //   private.  However, MS_PRIVATE only stops propagation of new *mounts*
    //   to/from the host — it does NOT replace the existing /tmp filesystem.
    //   The sandbox child still sees and can write to the host's /tmp.
    //   After run_sandboxed returns, the file persists on the host.
    //
    // Required behaviour (after fix):
    //   Sandboxed code must not be able to create files visible to the host
    //   after the sandbox exits.  The fix should mount a fresh tmpfs over /tmp
    //   (or use an overlay/bind mount) inside the namespace so that writes to
    //   /tmp are discarded when the namespace is torn down.
    //
    // This test creates a sentinel file with a unique name from inside the
    // sandbox and then checks whether it exists on the host after the sandbox
    // returns.  If it exists, the vulnerability is confirmed.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_no_persistent_host_tmp_writes() {
        // Use a UUID-derived unique marker so parallel test runs don't collide.
        let marker_name = format!(
            "rlox_escape_marker_{}",
            Uuid::new_v4().to_string().replace('-', "")
        );
        let marker_path_str = format!("/tmp/{marker_name}");

        let code = format!(
            r#"
# Attempt to write a sentinel file to the host /tmp.
# If the sandbox properly isolates /tmp via a private tmpfs, this write
# either fails (EPERM/EROFS) or succeeds but is invisible to the host.
try:
    with open("{marker_path_str}", "w") as f:
        f.write("escape_marker")
except Exception as e:
    # Write failed inside sandbox — that's the desired behaviour.
    # The test below (host-side check) will confirm isolation.
    pass
"#
        );

        let config = tight_config(10.0);
        let input = make_input(&code, "");

        let _output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err");

        // Give the filesystem a moment to settle.
        std::thread::sleep(std::time::Duration::from_millis(200));

        let marker_path = std::path::Path::new(&marker_path_str);
        let escaped = marker_path.exists();

        // Clean up if the file somehow escaped (so the test environment is clean
        // for subsequent runs), then assert failure.
        if escaped {
            let _ = std::fs::remove_file(marker_path);
        }

        assert!(
            !escaped,
            "VULNERABILITY F2: sandboxed code wrote a file to the host /tmp that \
             persisted after run_sandboxed returned (path: {marker_path_str}); \
             the mount namespace uses MS_PRIVATE but does not mount a new tmpfs \
             over /tmp, so writes are visible on the host filesystem"
        );
    }

    // -----------------------------------------------------------------------
    // F3 — /proc leaks host PID namespace
    //
    // Current behaviour (vulnerable):
    //   CLONE_NEWPID creates a new PID namespace for the child, but the child
    //   still inherits the parent's /proc mount (which shows host PIDs).  The
    //   sandbox does not mount a new procfs inside the namespace.  As a result,
    //   `os.listdir("/proc")` inside the sandbox returns the host's full
    //   process list (typically dozens to hundreds of numeric directories).
    //
    // Required behaviour (after fix):
    //   A new /proc must be mounted inside the PID namespace
    //   (`mount -t proc proc /proc`).  After that, /proc will only show the
    //   processes in the sandbox's own PID namespace (PID 1 = python3, and
    //   any child processes it spawns — typically 1–3 entries total, well
    //   under 10).
    //
    // Contract: the visible PID count from inside the sandbox must be <= 5.
    // If it is > 5, the host /proc is leaking through.
    //
    // This test parses the printed count from stdout.  The sandbox must allow
    // os.listdir("/proc") to run (it only uses openat/getdents64 syscalls
    // which are in the allowlist).
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_proc_is_scoped_to_namespace() {
        let code = r#"
import os
# Count numeric /proc entries — these are PID directories.
try:
    pids = [p for p in os.listdir("/proc") if p.isdigit()]
    print(f"PROC_PID_COUNT {len(pids)}", flush=True)
except Exception as e:
    print(f"PROC_LIST_ERROR {e}", flush=True)
"#;

        let config = tight_config(10.0);
        let input = make_input(code, "");

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err");

        let stdout = &output.stdout;

        // Parse the PID count from stdout.
        let pid_count: Option<usize> = stdout.lines().find_map(|line| {
            line.strip_prefix("PROC_PID_COUNT ")
                .and_then(|s| s.trim().parse::<usize>().ok())
        });

        let count = pid_count.unwrap_or_else(|| {
            panic!(
                "could not parse PROC_PID_COUNT from sandbox stdout; \
                 got: {stdout:?}"
            )
        });

        // A properly isolated /proc shows only the processes in the sandbox
        // PID namespace: PID 1 (python3) and possibly a handful of children.
        // We allow up to 5 to be generous.  Host systems typically have 50–500
        // processes; a count > 5 proves the host /proc is visible.
        assert!(
            count <= 5,
            "VULNERABILITY F3: /proc inside the sandbox shows {count} PID \
             directories (expected <= 5); the host /proc is leaking through \
             because the sandbox does not mount a new procfs inside the PID \
             namespace (CLONE_NEWPID alone does not hide the inherited /proc mount)"
        );
    }

    // -----------------------------------------------------------------------
    // F1/F8 — seccomp silent-skip: network syscall must be denied
    //
    // Contract: socket(AF_INET, SOCK_STREAM) is NOT in the seccomp allowlist.
    // If the filter is applied correctly, the syscall returns EPERM (or the
    // process is killed with SIGSYS, depending on the filter action).
    //
    // The test probes the observable invariant that PROVES seccomp is active:
    //   - If the filter ran:  socket() returns -1 (EPERM) and no fd is printed
    //     (or the process is killed mid-run and stdout contains "SOCKET_EPERM").
    //   - If the filter was silently skipped: socket() succeeds, a positive fd
    //     is returned, and stdout contains "SOCKET_FD <n>" with n >= 0.
    //
    // We print a marker so the parent can distinguish these two outcomes from
    // the captured stdout.  The assertion checks that the "succeeded" marker
    // is ABSENT, not that any particular failure message is present — this
    // avoids over-specifying how the denial manifests (EPERM vs SIGSYS kill).
    //
    // NOTE: in the CURRENT (unfixed) code the seccomp filter IS installed, so
    // this test may actually PASS if seccomp is working.  However, the audit
    // finding F1/F8 identifies the risk that the `if args.seccomp_len > 0`
    // guard in sandbox_child_fn could silently skip installation if the blob
    // is empty or mis-sized.  The test locks down the observable contract:
    // "socket must never succeed" — which will catch any regression where the
    // filter is dropped or bypassed.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_seccomp_failure_is_fatal_not_silent() {
        let code = r#"
import socket as _socket_mod
import sys

try:
    fd = _socket_mod.socket(_socket_mod.AF_INET, _socket_mod.SOCK_STREAM)
    # If we reach here the syscall succeeded — seccomp did NOT deny it.
    print(f"SOCKET_FD {fd.fileno()}", flush=True)
    fd.close()
except OSError as e:
    # EPERM, EACCES, or any error means the filter ran and denied the call.
    print(f"SOCKET_EPERM {e.errno}", flush=True)
except Exception as e:
    print(f"SOCKET_OTHER {e}", flush=True)

sys.exit(0)
"#;

        let config = tight_config(10.0);
        let input = make_input(code, "");

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err for seccomp probe code");

        let stdout = &output.stdout;

        // Check that socket() did NOT return a valid (non-negative) fd.
        // "SOCKET_FD <n>" in stdout where n >= 0 means seccomp was silent.
        let socket_succeeded = stdout.lines().any(|line| {
            if let Some(rest) = line.strip_prefix("SOCKET_FD ") {
                rest.trim().parse::<i64>().map(|n| n >= 0).unwrap_or(false)
            } else {
                false
            }
        });

        assert!(
            !socket_succeeded,
            "VULNERABILITY F1/F8: socket(AF_INET, SOCK_STREAM) succeeded inside \
             the sandbox (stdout: {stdout:?}); the seccomp filter was either not \
             installed or silently skipped — a missing filter allows unrestricted \
             syscall access from sandboxed code"
        );
    }

    // -----------------------------------------------------------------------
    // F7 — stdin blocks indefinitely
    //
    // Current behaviour (vulnerable):
    //   The sandbox does not redirect the child's stdin.  The Python process
    //   inherits the parent's stdin fd, which is typically a terminal or a
    //   pipe that is open and not at EOF.  Calling `sys.stdin.read()` inside
    //   the sandbox blocks indefinitely waiting for input that never comes.
    //   The sandbox's own timeout fires eventually, but the job wastes the
    //   entire timeout window doing nothing.
    //
    //   For a sandbox with a 5 s timeout, a job that does `sys.stdin.read()`
    //   will consume the full 5 s before being killed.  Under load, this is a
    //   denial-of-service vector — an adversary can saturate all sandbox
    //   workers by submitting code that reads stdin.
    //
    // Required behaviour (after fix):
    //   stdin MUST be /dev/null (or an equivalent closed fd) so that
    //   `sys.stdin.read()` returns immediately with an empty string.  The
    //   sandbox run should complete in << 1 s (plus interpreter startup).
    //
    // This test uses a 5 s timeout for the sandbox but wraps the async call
    // in a 3 s deadline.  If stdin is /dev/null the job completes in ~0.3 s
    // (python startup).  If stdin blocks, it will hit the 3 s wall-clock
    // deadline and the assertion fails.
    //
    // A stdin read that returns in ~0.3 s (< 3 s) proves /dev/null is wired.
    // A hang that exceeds 3 s proves stdin is open and blocking.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_stdin_does_not_block_sandbox() {
        let code = r#"
import sys
# This should return immediately (empty string) if stdin is /dev/null.
data = sys.stdin.read()
print(f"STDIN_READ_LEN {len(data)}", flush=True)
"#;

        // Sandbox timeout is 5 s, but we assert the entire call returns within
        // 3 s.  If stdin is /dev/null, this completes in ~0.3 s.
        // If stdin blocks, we hit the 3 s outer deadline first.
        let sandbox_timeout_secs = 5.0f64;
        let outer_deadline_secs = 3u64;

        let config = tight_config(sandbox_timeout_secs);
        let input = make_input(code, "");

        let result = tokio::time::timeout(
            tokio::time::Duration::from_secs(outer_deadline_secs),
            run_sandboxed(input, &config),
        )
        .await;

        match result {
            Err(_elapsed) => {
                // The sandbox call did not return within the outer deadline.
                // This is the definitive proof that stdin was open and blocking.
                panic!(
                    "VULNERABILITY F7: run_sandboxed did not return within \
                     {outer_deadline_secs} s when sandboxed code reads sys.stdin; \
                     stdin is not /dev/null — the job hung waiting for input that \
                     never arrives, consuming the full sandbox timeout budget"
                );
            }
            Ok(Err(e)) => {
                // Sandbox returned an error — that's acceptable as long as it
                // didn't block.  Not a vulnerability.
                let _ = e;
            }
            Ok(Ok(output)) => {
                // The call returned in time — verify that stdout says stdin
                // returned 0 bytes (i.e., /dev/null gave EOF immediately).
                let stdout = &output.stdout;
                let stdin_len: Option<usize> = stdout.lines().find_map(|line| {
                    line.strip_prefix("STDIN_READ_LEN ")
                        .and_then(|s| s.trim().parse::<usize>().ok())
                });

                if let Some(n) = stdin_len {
                    // If stdin returned data, the fd was not /dev/null.
                    // (A /dev/null fd returns 0 bytes immediately.)
                    assert_eq!(
                        n, 0,
                        "VULNERABILITY F7: sys.stdin.read() returned {n} bytes \
                         (expected 0 from /dev/null); stdin is connected to a \
                         real fd that contained data, not the null device"
                    );
                }
                // If STDIN_READ_LEN wasn't printed (job was killed), that is
                // also acceptable — it means the sandbox timed it out but at
                // least it didn't hang longer than the outer deadline.
            }
        }
    }
}

#[cfg(target_os = "linux")]
extern crate libc;
