//! Shared capability gates for the `rlox-sandbox` integration tests.
//!
//! Two independent host capabilities separate a CI shared runner from the
//! delegated Linux host (wk-system):
//!
//!   - **cgroup v2 self-migration** into a leaf. The sandbox child writes `"0"`
//!     to `<leaf>/cgroup.procs`; this only succeeds when the test process is
//!     already inside a user-delegated cgroup scope. Leaf *creation* alone works
//!     on CI (see `cgroup_tests.rs`), but the child cannot migrate itself in, so
//!     `run_sandboxed` returns `SetupError("child could not write to
//!     cgroup.procs …")`. Every assertion about `Clean`/`Timeout`/`OomKilled`
//!     is then vacuous — the test would be asserting against a setup failure.
//!     Gated by `RLOX_SANDBOX_CGROUP_TESTS`.
//!
//!   - **a scope-level TasksMax/MemoryMax backstop**, required before detonating
//!     genuine resource bombs (fork/memory/pids). Without it a fork bomb is
//!     neither contained nor bounded, which is both useless and unsafe.
//!     Gated by `RLOX_SANDBOX_ADVERSARIAL_TESTS`.
//!
//! `scripts/wk-sync-test.sh` wraps the run in
//! `systemd-run --user --scope --slice=rlox.slice` and exports **both**, so the
//! full suite runs there. When a variable is unset (CI, a plain SSH session, any
//! non-delegated host) the affected tests skip with an explicit message instead
//! of failing.
//!
//! These are deliberately explicit opt-ins rather than a runtime probe: on the
//! host that is supposed to have the capability the tests must *fail loudly* if
//! it regresses, never silently self-skip.

// Each test binary pulls in this module and uses only the gates it needs.
#![allow(dead_code)]

/// True when the host provides cgroup v2 self-migration (see module docs).
pub fn cgroup_tests_enabled() -> bool {
    std::env::var_os("RLOX_SANDBOX_CGROUP_TESTS").is_some()
}

/// True when the host additionally permits detonating real resource bombs.
pub fn adversarial_tests_enabled() -> bool {
    std::env::var_os("RLOX_SANDBOX_ADVERSARIAL_TESTS").is_some()
}

/// Gate for any test that calls `run_sandboxed` and asserts on the exit status.
///
/// Returns `true` when the caller should `return` early, having printed why.
pub fn skip_without_cgroups(test_name: &str) -> bool {
    if cgroup_tests_enabled() {
        return false;
    }
    eprintln!(
        "SKIP {test_name}: requires cgroup v2 self-migration. Set \
         RLOX_SANDBOX_CGROUP_TESTS=1 on a host with a cgroup-delegated slice \
         (see scripts/wk-sync-test.sh)"
    );
    true
}

/// Gate for tests that detonate genuine resource bombs.
///
/// Returns `true` when the caller should `return` early, having printed why.
pub fn skip_without_adversarial(test_name: &str) -> bool {
    if adversarial_tests_enabled() {
        return false;
    }
    eprintln!(
        "SKIP {test_name}: set RLOX_SANDBOX_ADVERSARIAL_TESTS=1 under a \
         cgroup-delegated slice with a resource backstop \
         (see scripts/wk-sync-test.sh)"
    );
    true
}
