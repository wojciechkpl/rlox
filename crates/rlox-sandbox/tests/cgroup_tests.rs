/// Step 1a — cgroup v2 delegation helpers
///
/// These tests verify the public contract of `rlox_sandbox::cgroup`.
/// All tests are Linux-only.  They assume cgroup v2 per-user delegation is
/// present at `/sys/fs/cgroup/user.slice/user-<uid>.slice/` with at least
/// `memory` and `pids` controllers delegated — confirmed on wk-system.
///
/// If delegation is absent the test panics with a clear message rather than
/// silently passing, so CI visibility is maintained.
#[cfg(target_os = "linux")]
mod cgroup_tests {
    use rlox_sandbox::cgroup;
    use std::path::PathBuf;

    // -----------------------------------------------------------------------
    // Helper: locate the per-user slice root and panic clearly if absent.
    // -----------------------------------------------------------------------
    fn require_user_cgroup_root() -> PathBuf {
        let uid = unsafe { libc::getuid() };
        let p = PathBuf::from(format!(
            "/sys/fs/cgroup/user.slice/user-{uid}.slice"
        ));
        if !p.exists() {
            panic!(
                "cgroup test skipped: per-user delegation not found at {p:?}. \
                 Expected on wk-system (Ubuntu 24.04, cgroup v2)."
            );
        }
        p
    }

    // -----------------------------------------------------------------------
    // Test 1a-1: user_cgroup_root returns the correct per-uid path.
    // -----------------------------------------------------------------------
    #[test]
    fn test_user_cgroup_root_returns_existing_directory() {
        let root = cgroup::user_cgroup_root();
        assert!(
            root.exists(),
            "user_cgroup_root() returned {root:?} which does not exist"
        );
        // Must be under the cgroup v2 hierarchy
        assert!(
            root.starts_with("/sys/fs/cgroup"),
            "expected path under /sys/fs/cgroup, got {root:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 1a-2: create_leaf creates the directory; destroy_leaf removes it.
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_and_destroy_cgroup_leaf_succeeds() {
        let base = require_user_cgroup_root();
        let leaf_name = format!("rlox-test-{}", uuid::Uuid::new_v4());

        let leaf_path = cgroup::create_leaf(&base, &leaf_name)
            .expect("create_leaf should succeed under per-user slice");

        assert!(
            leaf_path.exists(),
            "leaf directory should exist after create_leaf: {leaf_path:?}"
        );
        // The leaf must live somewhere in the cgroup v2 hierarchy, but NOT
        // necessarily directly under `base`: create_leaf auto-discovers the
        // first user-writable delegated ancestor (e.g. user@<uid>.service or
        // an rlox.slice scope), so asserting `base.join(name)` is over-specified.
        assert!(
            leaf_path.starts_with("/sys/fs/cgroup"),
            "leaf_path should be inside the cgroup v2 hierarchy, got {leaf_path:?}"
        );
        assert_eq!(
            leaf_path.file_name(),
            Some(std::ffi::OsStr::new(&leaf_name)),
            "leaf directory name should match the requested leaf_name, got {leaf_path:?}"
        );

        // Cleanup must also succeed
        cgroup::destroy_leaf(&leaf_path)
            .expect("destroy_leaf should succeed on an empty leaf");

        assert!(
            !leaf_path.exists(),
            "leaf directory should be gone after destroy_leaf"
        );
    }

    // -----------------------------------------------------------------------
    // Test 1a-3: write_memory_max is reflected by read_memory_max.
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_and_read_back_memory_max() {
        let base = require_user_cgroup_root();
        let leaf_name = format!("rlox-test-mem-{}", uuid::Uuid::new_v4());
        let leaf_path = cgroup::create_leaf(&base, &leaf_name).unwrap();

        let limit_bytes: u64 = 64 * 1024 * 1024; // 64 MiB
        cgroup::write_memory_max(&leaf_path, limit_bytes)
            .expect("write_memory_max should succeed");

        let read_back = cgroup::read_memory_max(&leaf_path)
            .expect("read_memory_max should succeed");

        // cgroup v2 may round up to page granularity; accept within one page.
        let page = 4096u64;
        assert!(
            read_back >= limit_bytes && read_back <= limit_bytes + page,
            "read back {read_back} bytes, expected ~{limit_bytes}"
        );

        cgroup::destroy_leaf(&leaf_path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Test 1a-4: write_pids_max is reflected by read_pids_max.
    // -----------------------------------------------------------------------
    #[test]
    fn test_write_and_read_back_pids_max() {
        let base = require_user_cgroup_root();
        let leaf_name = format!("rlox-test-pids-{}", uuid::Uuid::new_v4());
        let leaf_path = cgroup::create_leaf(&base, &leaf_name).unwrap();

        let limit: u32 = 32;
        cgroup::write_pids_max(&leaf_path, limit)
            .expect("write_pids_max should succeed");

        let read_back = cgroup::read_pids_max(&leaf_path)
            .expect("read_pids_max should succeed");

        assert_eq!(
            read_back, limit,
            "pids.max read back {read_back}, expected {limit}"
        );

        cgroup::destroy_leaf(&leaf_path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Test 1a-5: cgroup.controllers lists "memory" and "pids".
    //
    // Assumption: the per-user slice on wk-system delegates at least these two
    // controllers.  If the system is mis-configured the test fails with a
    // descriptive message.
    // -----------------------------------------------------------------------
    #[test]
    fn test_read_controllers_includes_memory_and_pids() {
        let base = require_user_cgroup_root();
        let leaf_name = format!("rlox-test-ctrl-{}", uuid::Uuid::new_v4());
        let leaf_path = cgroup::create_leaf(&base, &leaf_name).unwrap();

        let controllers = cgroup::read_controllers(&leaf_path)
            .expect("read_controllers should succeed");

        assert!(
            controllers.iter().any(|c| c == "memory"),
            "expected 'memory' in cgroup.controllers, got: {controllers:?}"
        );
        assert!(
            controllers.iter().any(|c| c == "pids"),
            "expected 'pids' in cgroup.controllers, got: {controllers:?}"
        );

        cgroup::destroy_leaf(&leaf_path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Test 1a-6: freeze + kill_subtree succeeds on a leaf with no processes.
    //
    // Contract: freeze() and kill_subtree() must not error when called on an
    // empty cgroup (the kill path must be idempotent on an already-empty group).
    // -----------------------------------------------------------------------
    #[test]
    fn test_freeze_and_kill_empty_subtree_succeeds() {
        let base = require_user_cgroup_root();
        let leaf_name = format!("rlox-test-kill-{}", uuid::Uuid::new_v4());
        let leaf_path = cgroup::create_leaf(&base, &leaf_name).unwrap();

        cgroup::freeze(&leaf_path).expect("freeze should succeed on empty cgroup");
        cgroup::kill_subtree(&leaf_path).expect("kill_subtree should succeed on empty cgroup");

        cgroup::destroy_leaf(&leaf_path).unwrap();
    }
}

// Provide libc for the uid lookup above
#[cfg(target_os = "linux")]
extern crate libc;
