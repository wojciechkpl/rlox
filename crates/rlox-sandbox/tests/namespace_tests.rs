/// Step 1c — namespace clone + mount setup
///
/// Tests verify the public contract of `rlox_sandbox::worker::spawn_in_namespaces`.
///
/// Design reference (design.md, Step 1c):
///   "clone(CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS)"
///   "immediately call mount(None, '/', None, MS_REC | MS_PRIVATE, None)"
///   "verify from child that /proc/self/ns/pid differs from parent"
///   "verify MS_PRIVATE | MS_REC mount does not propagate to host"
///
/// All tests call `spawn_in_namespaces()`, which is a stub returning
/// `unimplemented!()`.  Every test fails for the right reason: missing behavior.
///
/// Assumption for the implementer:
///   `unprivileged_userns_clone=1` is set (confirmed on wk-system).
///   The child must read its pid-ns symlink target via a raw libc `readlink`
///   call (no Rust heap alloc) and communicate it to the parent over a pipe,
///   because calling Rust std-lib from a forked-but-not-exec'd child that
///   shares the heap with the parent is unsafe.
#[cfg(target_os = "linux")]
mod namespace_tests {
    use rlox_sandbox::worker::spawn_in_namespaces;
    use std::fs;

    // -----------------------------------------------------------------------
    // Helper: read /proc/self/ns/pid as a String (safe in the parent process).
    // -----------------------------------------------------------------------
    fn parent_pid_ns() -> String {
        fs::read_link("/proc/self/ns/pid")
            .expect("readlink /proc/self/ns/pid should succeed")
            .to_string_lossy()
            .to_string()
    }

    // -----------------------------------------------------------------------
    // Test 1c-1: spawn_in_namespaces returns Ok and the child's pid-ns target
    // differs from the parent's.
    //
    // After CLONE_NEWPID the child is in a new pid namespace; its
    // /proc/self/ns/pid symlink target must be a different inode number than
    // the parent's.
    // -----------------------------------------------------------------------
    #[test]
    fn test_spawn_in_namespaces_child_sees_different_pid_ns() {
        let parent_ns = parent_pid_ns();

        let result = spawn_in_namespaces()
            .expect("spawn_in_namespaces should not error on wk-system");

        assert_eq!(
            result.exit_code, 0,
            "child should exit 0; got {}",
            result.exit_code
        );
        assert_ne!(
            result.pid_ns_target, parent_ns,
            "child pid-ns '{}' must differ from parent '{}' after CLONE_NEWPID",
            result.pid_ns_target, parent_ns
        );
    }

    // -----------------------------------------------------------------------
    // Test 1c-2: after spawn_in_namespaces succeeds, host /proc/mounts does
    // NOT contain any new entries — the MS_REC|MS_PRIVATE mount in the child
    // must not propagate back to the host.
    //
    // This test records the mount table before and after calling
    // spawn_in_namespaces and asserts no new lines were added.
    // -----------------------------------------------------------------------
    #[test]
    fn test_spawn_in_namespaces_mount_does_not_propagate_to_host() {
        let mounts_before =
            fs::read_to_string("/proc/mounts").expect("read /proc/mounts before");

        // spawn_in_namespaces is a stub; if it panics the test fails for the
        // right reason (missing implementation).
        let result = spawn_in_namespaces()
            .expect("spawn_in_namespaces should not error on wk-system");

        assert_eq!(result.exit_code, 0, "child should exit clean");

        let mounts_after =
            fs::read_to_string("/proc/mounts").expect("read /proc/mounts after");

        let lines_before = mounts_before.lines().count();
        let lines_after = mounts_after.lines().count();

        assert_eq!(
            lines_after, lines_before,
            "host /proc/mounts grew from {lines_before} to {lines_after} lines \
             after spawn_in_namespaces — MS_PRIVATE not preventing propagation"
        );
    }

    // -----------------------------------------------------------------------
    // Test 1c-3: spawn_in_namespaces returns Ok (not Err) when called
    // a second time — it must be repeatable / non-singleton.
    // -----------------------------------------------------------------------
    #[test]
    fn test_spawn_in_namespaces_is_repeatable() {
        for i in 0..2 {
            let result = spawn_in_namespaces()
                .unwrap_or_else(|e| panic!("call {i} failed: {e}"));
            assert_eq!(result.exit_code, 0, "call {i}: child exit code should be 0");
        }
    }
}
