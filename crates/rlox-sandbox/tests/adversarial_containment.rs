/// Adversarial containment regression tests.
///
/// These tests lock down three security bugs identified in the code review:
///
/// Bug A — cgroup self-migration silently failing
///   The child writes "0" to cgroup.procs but ignores errors.  If migration
///   fails (e.g. the parent process is not already inside a delegated cgroup
///   scope, or a permission race), the sandboxed code executes OUTSIDE the
///   cgroup.  A fork bomb's children then survive even after run_sandboxed
///   returns, because cgroup.kill only kills processes inside the leaf.
///
///   Additionally, when cgroup_base itself does not exist, run_sandboxed must
///   return Err rather than silently running untrusted code with no containment.
///
/// Bug B — unbounded stdout buffer
///   drain_fd() accumulates into an unbounded Vec<u8>.  Adversarial code that
///   writes 50 MiB to stdout causes the parent to allocate 50 MiB per job.
///   Under concurrent load this causes parent OOM-kill contagion — the
///   untrusted code's resource usage escapes the cgroup.
///
/// Bug C — CLONE_NEWUSER unfiltered in seccomp
///   The seccomp allowlist includes SYS_clone and SYS_clone3 unconditionally.
///   A comment says "CLONE_NEWUSER filtered elsewhere" but no such filter
///   exists.  A child can call clone(CLONE_NEWUSER | …) to create a nested
///   user namespace, giving it the ability to map arbitrary UIDs and potentially
///   mount filesystems inside the namespace.
///
/// All tests are Linux-only and use `#[tokio::test]`.
///
/// Assumptions the implementer MUST honor:
///   - `python3` is on PATH on wk-system.
///   - cgroup leaf is under the per-user delegated slice; the test binary is
///     run inside rlox.slice via the wk-sync-test.sh helper so self-migration
///     can succeed.
///   - After Timeout or OomKilled, NO descendant processes of the job survive.
///   - SandboxOutput.stdout is bounded to at most 2 MiB regardless of how much
///     the sandboxed process writes.
///   - Attempting clone(CLONE_NEWUSER) or unshare(CLONE_NEWUSER) from inside
///     the sandbox must fail (errno set or process seccomp-killed) so that
///     sandboxed Python never sees rc == 0 from either call.
///   - When cgroup_base does not exist, run_sandboxed must return Err, not
///     silently execute untrusted code with no resource containment.

#[cfg(target_os = "linux")]
mod adversarial_containment {
    use rlox_sandbox::worker::{run_sandboxed, SandboxConfig, SandboxExitStatus, SandboxInput};
    use std::path::PathBuf;
    use std::time::Instant;
    use uuid::Uuid;

    // -----------------------------------------------------------------------
    // Helper: per-user delegated cgroup base path.
    // -----------------------------------------------------------------------
    fn cgroup_base() -> PathBuf {
        let uid = unsafe { libc::getuid() };
        PathBuf::from(format!("/sys/fs/cgroup/user.slice/user-{uid}.slice"))
    }

    // -----------------------------------------------------------------------
    // The real-bomb tests below (fork bomb, memory bomb, pids exhaustion) are
    // gated on `RLOX_SANDBOX_ADVERSARIAL_TESTS` via
    // `crate::common::adversarial_tests_enabled()`. See tests/common/mod.rs for
    // what the gate requires and why it is an explicit opt-in rather than a
    // runtime probe.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Helper: a SandboxConfig with tight resource caps.
    // `timeout_secs` is caller-supplied; everything else is deliberately low.
    // -----------------------------------------------------------------------
    fn tight_config(timeout_secs: f64, mem_limit_bytes: u64, pids_limit: u32) -> SandboxConfig {
        SandboxConfig {
            timeout_secs,
            mem_limit_bytes,
            pids_limit,
            cpu_weight: 100,
            cgroup_base: cgroup_base(),
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build a SandboxInput.
    // -----------------------------------------------------------------------
    fn make_input(code: &str) -> SandboxInput {
        SandboxInput {
            job_id: Uuid::new_v4(),
            code: code.to_owned(),
            test_suite: String::new(),
            language: "python".to_owned(),
            is_adversarial: true,
        }
    }

    // -----------------------------------------------------------------------
    // Count live python3 processes belonging to the current user whose
    // command-line contains the rlox job marker.  Used to assert no survivors.
    //
    // We read /proc/<pid>/cmdline; any process that has a job_id string
    // anywhere in its args is counted.  A count of 0 is required post-call.
    // -----------------------------------------------------------------------
    fn count_python3_survivors(job_id: &Uuid) -> usize {
        let marker = job_id.to_string();
        let my_uid = unsafe { libc::getuid() };

        let Ok(proc_dir) = std::fs::read_dir("/proc") else {
            return 0;
        };

        let mut count = 0usize;
        for entry in proc_dir.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }

            // Check UID from /proc/<pid>/status — skip if not ours.
            let status_path = format!("/proc/{}/status", name_str);
            let Ok(status_text) = std::fs::read_to_string(&status_path) else {
                continue;
            };
            let is_ours = status_text.lines().any(|line| {
                if let Some(rest) = line.strip_prefix("Uid:") {
                    rest.split_whitespace()
                        .next()
                        .and_then(|s| s.parse::<u32>().ok())
                        .map(|uid| uid == my_uid)
                        .unwrap_or(false)
                } else {
                    false
                }
            });
            if !is_ours {
                continue;
            }

            let cmdline_path = format!("/proc/{}/cmdline", name_str);
            let Ok(cmdline_bytes) = std::fs::read(&cmdline_path) else {
                continue;
            };
            let cmdline = String::from_utf8_lossy(&cmdline_bytes);
            if cmdline.contains(&marker) {
                count += 1;
            }
        }
        count
    }

    // -----------------------------------------------------------------------
    // Test 1 — Fork bomb is fully contained within the cgroup.
    //
    // Locks down Bug A: cgroup self-migration silently failing.
    //
    // The sandbox child (sandbox_child_fn) writes "0" to cgroup.procs but the
    // error branch is absent — `if cg_fd >= 0 { … }` swallows migration
    // failures silently.  If the calling process is not in a user-delegated
    // cgroup scope, migration fails with no signal to the parent.  The child
    // then runs OUTSIDE the cgroup: pids.max is ignored, cgroup.kill has no
    // effect, and fork bomb descendants survive the call.
    //
    // This test is designed to catch the regression when it manifests (migration
    // fails → survivors > 0).  In the normal wk-system rlox.slice environment
    // migration succeeds and all assertions pass — proving the contract holds.
    //
    // Separately, it also verifies the error-propagation path: when cgroup_base
    // does not exist, run_sandboxed must return Err rather than silently
    // executing untrusted code with no cgroup containment.
    //
    // Assertions:
    //   (b) exit status is Timeout or OomKilled — not Clean.
    //   (c) stats.cgroup_kill_event is true.
    //   (d) ZERO surviving python3 processes with this job_id after return.
    //       This is the decisive check: survivors > 0 proves the cgroup missed
    //       the fork bomb's descendants (Bug A: migration silently failed).
    // -----------------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread")]
    async fn test_fork_bomb_is_fully_contained() {
        if !crate::common::adversarial_tests_enabled() {
            eprintln!(
                "SKIP test_fork_bomb_is_fully_contained: set \
                 RLOX_SANDBOX_ADVERSARIAL_TESTS=1 under a cgroup-delegated slice \
                 with a resource backstop (see scripts/wk-sync-test.sh)"
            );
            return;
        }
        let code = "import os\nwhile True:\n    try:\n        os.fork()\n    except Exception:\n        pass\n";

        let timeout_secs = 3.0f64;
        // Use a very low pids_limit so the fork bomb is starved quickly,
        // minimising the number of processes to kill and keeping the test fast
        // even on a loaded system.  pids.max is only effective when the child
        // IS inside the cgroup (Bug A: if not, the limit is invisible).
        let config = tight_config(timeout_secs, 128 * 1024 * 1024, 8);
        let input = make_input(code);
        let job_id = input.job_id;

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err for adversarial code with valid config");

        // (b) Must have been killed (not clean exit).
        assert!(
            matches!(
                output.exit_status,
                SandboxExitStatus::Timeout | SandboxExitStatus::OomKilled
            ),
            "expected Timeout or OomKilled for fork bomb, got {:?}",
            output.exit_status
        );

        // (c) cgroup_kill_event must be true.
        assert!(
            output.stats.cgroup_kill_event,
            "cgroup_kill_event must be true after containing a fork bomb; \
             if false, the child never migrated into the cgroup leaf \
             (cgroup self-migration silently failed — Bug A)"
        );

        // (d) DECISIVE: no survivors after run_sandboxed returns.
        //     If Bug A is present (child was outside cgroup), cgroup.kill is
        //     a no-op and forked descendants survive as orphan processes.
        std::thread::sleep(std::time::Duration::from_millis(500));
        let survivors = count_python3_survivors(&job_id);
        assert_eq!(
            survivors, 0,
            "fork bomb left {survivors} surviving python3 process(es) after \
             run_sandboxed returned — descendants were NOT inside the cgroup \
             when cgroup.kill fired (cgroup self-migration silently failed — Bug A)"
        );

        // ── Error path: non-existent cgroup_base must return Err ────────────
        // When cgroup_base doesn't exist, run_sandboxed must fail at setup,
        // not silently run untrusted code with no resource limits.
        let config_no_cgroup = SandboxConfig {
            timeout_secs: 3.0,
            mem_limit_bytes: 64 * 1024 * 1024,
            pids_limit: 16,
            cpu_weight: 100,
            cgroup_base: PathBuf::from("/sys/fs/cgroup/rlox-nonexistent-slice-does-not-exist"),
        };
        let input_no_cgroup = make_input("print('should not run')");
        let result_no_cgroup = run_sandboxed(input_no_cgroup, &config_no_cgroup).await;
        assert!(
            result_no_cgroup.is_err(),
            "run_sandboxed must return Err when cgroup_base does not exist; \
             got Ok({:?}) — silently running code without cgroup containment \
             is Bug A",
            result_no_cgroup.ok().map(|o| o.exit_status)
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — pids.max caps a fork bomb (defense in depth).
    //
    // With pids_limit = 16, the kernel refuses fork() calls that would exceed
    // the PID count, throttling the bomb.  The job must terminate and leave no
    // survivors.
    //
    // pids.max is only effective if the child IS inside the cgroup.  If Bug A
    // is present (migration silently failed), pids.max and cgroup.kill both
    // have no effect → the cgroup is empty → kill is a no-op.
    // -----------------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread")]
    async fn test_pids_limit_caps_fork_bomb() {
        if !crate::common::adversarial_tests_enabled() {
            eprintln!(
                "SKIP test_pids_limit_caps_fork_bomb: set \
                 RLOX_SANDBOX_ADVERSARIAL_TESTS=1 under a cgroup-delegated slice \
                 with a resource backstop (see scripts/wk-sync-test.sh)"
            );
            return;
        }
        let code = "import os\nwhile True:\n    try:\n        os.fork()\n    except Exception:\n        pass\n";

        let config = tight_config(3.0, 128 * 1024 * 1024, 16);
        let input = make_input(code);
        let job_id = input.job_id;

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err");

        assert!(
            matches!(
                output.exit_status,
                SandboxExitStatus::Timeout | SandboxExitStatus::OomKilled
            ),
            "expected Timeout or OomKilled with pids_limit=16, got {:?}",
            output.exit_status
        );

        assert!(
            output.stats.cgroup_kill_event,
            "cgroup_kill_event must be true even with pids_limit=16; \
             if false, pids.max was ineffective (child was not in cgroup — Bug A)"
        );

        std::thread::sleep(std::time::Duration::from_millis(500));
        let survivors = count_python3_survivors(&job_id);
        assert_eq!(
            survivors, 0,
            "fork bomb (pids_limit=16) left {survivors} survivors — \
             pids.max had no effect because the child was not inside the cgroup (Bug A)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3 — Memory bomb is OOM-killed promptly.
    //
    // With mem_limit_bytes = 128 MiB, growing a bytearray indefinitely must
    // trigger the OOM killer well before the 10 s timeout.  The exit status
    // must be OomKilled and the call must return in << 8 s.
    //
    // Expected failure against buggy impl: if the cgroup memory limit is not
    // applied (because the child ran outside the cgroup), the process is not
    // OOM-killed by the cgroup — instead it runs until the timeout fires.
    // Elapsed >> 8 s and exit_status == Timeout rather than OomKilled.
    // -----------------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread")]
    async fn test_memory_bomb_is_oom_killed() {
        if !crate::common::adversarial_tests_enabled() {
            eprintln!(
                "SKIP test_memory_bomb_is_oom_killed: set \
                 RLOX_SANDBOX_ADVERSARIAL_TESTS=1 under a cgroup-delegated slice \
                 with a resource backstop (see scripts/wk-sync-test.sh)"
            );
            return;
        }
        // Grows memory by 10 MiB per iteration; at 128 MiB limit this dies fast.
        let code = "x = bytearray()\nwhile True:\n    x += bytearray(10 * 1024 * 1024)\n";

        let config = tight_config(10.0, 128 * 1024 * 1024, 64);
        let input = make_input(code);

        // OOM should fire in << 8 s; give an 8 s deadline before declaring hang.
        let wall_start = Instant::now();
        let output = tokio::time::timeout(
            tokio::time::Duration::from_secs(8),
            run_sandboxed(input, &config),
        )
        .await
        .unwrap_or_else(|_| {
            panic!(
                "memory bomb did not OOM-kill within 8 s (128 MiB limit); \
                 the cgroup memory limit was not applied — Bug A: child was \
                 not inside the cgroup and waitpid may be blocking"
            )
        })
        .expect("run_sandboxed must not return Err");
        let elapsed = wall_start.elapsed().as_secs_f64();

        assert!(
            elapsed < 8.0,
            "memory bomb did not OOM promptly: elapsed {elapsed:.2} s (< 8.0 s required); \
             the memory limit may not be applied if the child was not inside the cgroup \
             (cgroup self-migration silently failed — Bug A)"
        );

        assert_eq!(
            output.exit_status,
            SandboxExitStatus::OomKilled,
            "expected OomKilled for memory bomb with 128 MiB limit, got {:?}; \
             if Timeout, the memory limit was not enforced (child outside cgroup — Bug A)",
            output.exit_status
        );
    }

    // -----------------------------------------------------------------------
    // Test 4 — Stdout flood does not exhaust the parent.
    //
    // Locks down Bug B: unbounded stdout buffer.
    //
    // Bug: drain_fd() appends into an unbounded Vec<u8>.  Code that writes
    // 50 MiB to stdout causes the parent to allocate 50 MiB per job.
    //
    // Contract: SandboxOutput.stdout must be bounded to at most 2 MiB.
    // The implementation must truncate (or stop reading) after this cap.
    //
    // Expected failure against buggy impl: stdout.len() == 50 * 1024 * 1024.
    // -----------------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread")]
    async fn test_stdout_flood_does_not_exhaust_parent() {
        // Writes exactly 50 MiB to stdout in one shot.
        let code = "import sys\nsys.stdout.write('A' * (50 * 1024 * 1024))\nsys.stdout.flush()\n";

        let config = tight_config(10.0, 256 * 1024 * 1024, 64);
        let input = make_input(code);

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err for stdout flood");

        const MAX_STDOUT_BYTES: usize = 2 * 1024 * 1024; // 2 MiB
        let stdout_len = output.stdout.len();
        assert!(
            stdout_len <= MAX_STDOUT_BYTES,
            "stdout is unbounded: got {stdout_len} bytes (> {MAX_STDOUT_BYTES} byte cap); \
             unbounded buffering is Bug B — it allows untrusted code to cause \
             parent OOM contagion outside the sandbox cgroup"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5 — Nested user namespace creation is denied.
    //
    // Locks down Bug C: CLONE_NEWUSER unfiltered in seccomp.
    //
    // Bug: SYS_clone and SYS_clone3 are in the seccomp allowlist with no
    // argument filtering.  The comment "CLONE_NEWUSER filtered elsewhere" is
    // false — no such filter exists.  A sandboxed process can call
    // clone(CLONE_NEWUSER) to enter a nested user namespace.
    //
    // This test probes both paths:
    //   (a) libc.unshare(CLONE_NEWUSER) — SYS_unshare is not in the allowlist,
    //       so this must be denied (returns -1).
    //   (b) raw clone syscall with CLONE_NEWUSER — SYS_clone IS in the allowlist
    //       without argument filtering, so this currently SUCCEEDS (Bug C).
    //       The fix: add a conditional seccomp rule that returns EPERM when
    //       (flags & CLONE_NEWUSER) != 0.
    //
    // Acceptable outcomes after the fix:
    //   - clone returns -1 (EPERM from seccomp conditional rule), OR
    //   - the process is SIGKILL'd by seccomp (SECCOMP_RET_KILL_PROCESS).
    //
    // The one forbidden outcome: "CLONE_RC 0" or "CLONE_RC <positive PID>"
    // in stdout, which indicates a nested user namespace was successfully created.
    //
    // Expected failure against current (buggy) impl: stdout contains
    // "CLONE_RC 0" followed by "CLONE_RC <positive child PID>", proving
    // the nested user namespace was created.
    // -----------------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread")]
    async fn test_nested_user_namespace_is_denied() {
        let code = r#"
import ctypes
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)

CLONE_NEWUSER = 0x10000000

# Path (a): unshare(CLONE_NEWUSER) — SYS_unshare is NOT in the seccomp
# allowlist, so this must be denied.
rc_unshare = libc.unshare(CLONE_NEWUSER)
print("UNSHARE_RC", rc_unshare, flush=True)

# Path (b): raw clone(2) with CLONE_NEWUSER flag.
# SYS_clone IS in the allowlist unconditionally (Bug C).
# A correct implementation adds a conditional rule: deny clone when
# (flags & CLONE_NEWUSER) != 0.
SIGCHLD = 17
clone_flags = CLONE_NEWUSER | SIGCHLD
SYS_clone = 56  # x86-64

try:
    rc_clone = libc.syscall(SYS_clone,
                            ctypes.c_long(clone_flags),
                            ctypes.c_long(0),  # stack (NULL = use current)
                            ctypes.c_long(0),
                            ctypes.c_long(0),
                            ctypes.c_long(0))
    if rc_clone > 0:
        # We are the parent; child was created in a nested user namespace.
        os.waitpid(rc_clone, 0)
    print("CLONE_RC", rc_clone, flush=True)
except Exception as e:
    print("CLONE_RC -1 exception:", e, flush=True)

sys.exit(0)
"#;

        let config = tight_config(5.0, 128 * 1024 * 1024, 64);
        let input = make_input(code);

        let output = run_sandboxed(input, &config)
            .await
            .expect("run_sandboxed must not return Err");

        let stdout = &output.stdout;

        // (a) unshare must NOT have succeeded.
        let unshare_succeeded = stdout
            .lines()
            .any(|line| line.starts_with("UNSHARE_RC") && line.contains(" 0"));
        assert!(
            !unshare_succeeded,
            "unshare(CLONE_NEWUSER) succeeded inside the sandbox (stdout: {:?}); \
             the seccomp filter must deny SYS_unshare",
            stdout
        );

        // (b) clone(CLONE_NEWUSER) must NOT have returned a positive child PID.
        //     rc == -1  → syscall was denied by seccomp conditional rule (correct).
        //     rc == 0   → child process (inside nested userns); parent side.
        //     rc > 0    → parent side saw a child PID — nested namespace created.
        //     Any positive result (parent OR child) means the namespace was created.
        let clone_gave_child = stdout.lines().any(|line| {
            if let Some(rest) = line.strip_prefix("CLONE_RC ") {
                let token = rest.split_whitespace().next().unwrap_or("-1");
                token.parse::<i64>().map(|n| n >= 0).unwrap_or(false)
            } else {
                false
            }
        });
        assert!(
            !clone_gave_child,
            "clone(CLONE_NEWUSER) succeeded inside the sandbox (stdout: {:?}); \
             the seccomp filter must add a conditional rule that returns EPERM \
             when (clone_flags & CLONE_NEWUSER) != 0 — Bug C",
            stdout
        );
    }
}

#[cfg(target_os = "linux")]
extern crate libc;

// Shared capability gates (RLOX_SANDBOX_CGROUP_TESTS /
// RLOX_SANDBOX_ADVERSARIAL_TESTS). Declared at the end of the file so it
// cannot absorb the module doc comment above.
#[cfg(target_os = "linux")]
mod common;
