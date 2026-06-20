/// Hard-isolation sandbox for executing untrusted, model-generated code with zero contagion.
///
/// This crate provides a Rust implementation of a Linux-based unprivileged sandbox designed to
/// safely execute arbitrary Python code (from LLM agents) with strong containment guarantees.
/// It is the **Treatment-side execution layer** for the rlox agentic-benchmark MVP (see
/// `.wf/design.md` Component 1).
///
/// ## Threat Model & Isolation Mechanisms
///
/// The sandbox defends against adversarial code in five categories:
///
/// | Threat | Mechanism | Details |
/// |--------|-----------|---------|
/// | **Infinite loops / busy-wait** | Wall-clock timeout + cgroup freeze | Timeout detection triggers `cgroup.freeze`, blocking all processes in the subtree, then `cgroup.kill` terminates them. No escape via shared state or signal handling. |
/// | **Fork bombs** | PID limit + cgroup kill | `pids.max` cap prevents process creation. Even if exhausted, all children stay in the sandbox cgroup; `cgroup.kill` kills the entire subtree atomically. |
/// | **Memory bombs** | Memory limit + OOM killer | `memory.max` triggers Linux OOM-killer; sandboxed process is selected first. Bounded stdout buffer (1 MiB cap in `MAX_STDOUT_BYTES`) prevents parent-side allocation contagion. |
/// | **Unkillable threads** | cgroup freeze + subtree kill | Threads cannot opt out of `cgroup.freeze` or `cgroup.kill` — all threads in the cgroup subtree are affected. |
/// | **Network I/O / blocking** | seccomp allowlist + network syscall denial | `socket`, `connect`, `bind`, `sendto`, `recvfrom`, etc. are denied by the BPF filter (default action: `EPERM`). Code blocks forever or crashes immediately. |
/// | **Nested user namespaces** | seccomp conditional rule on `SYS_clone` | `clone(CLONE_NEWUSER | ...)` is denied with `EPERM`. `SYS_clone3` is denied outright. Prevents privilege escalation via nested user-namespace tricks. |
/// | **File descriptor exhaustion** | seccomp scope + cgroup limits | The sandboxed process inherits minimal FDs (stdout, stderr, script file only). No listen sockets, no `/dev/*` access outside the mount namespace. |
///
/// ## Isolation Implementation
///
/// **Namespace isolation:** Each sandbox invocation calls `clone(CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS, ...)` to create a new process in all four Linux namespaces simultaneously. This atomic operation avoids AppArmor `unprivileged_userns` capability denials that occur when using `unshare()` after the fact.
///
/// **UID/GID mapping:** The child process is mapped to UID 0 (root) inside the namespace but has no real privileges outside it. The parent maps the outer UID (unprivileged user) to the child's UID 0 via `/proc/<pid>/uid_map`.
///
/// **Mount namespace:** Explicitly marked as `MS_PRIVATE | MS_REC` to prevent mounts from propagating to the host.
///
/// **seccomp-BPF allowlist:** A bytecode filter (built via `seccompiler`, Firecracker's pure-Rust BPF builder) allows only syscalls required by CPython 3.x running unit tests. Default action (for unlisted syscalls) is `EPERM`. Key exclusions:
///   - Network syscalls: `socket`, `connect`, `bind`, `sendto`, `recvfrom`, `listen`, etc.
///   - Privilege escalation: `capset`, `ptrace`, `mount`, `seccomp` (no filter modification), `clone3`, and `clone` with `CLONE_NEWUSER`.
///   - Raw syscall manipulation: `bpf`, `syslog`, `keyctl`.
///
/// **cgroup v2 resource limits:**
///   - `memory.max`: Hard cap (bytes) on physical memory. OOM-killer is triggered before the cap is exceeded.
///   - `pids.max`: Hard cap on number of processes/threads. Prevents fork bombs.
///   - `cpu.weight`: Relative CPU scheduling weight (default 100). Used for fair scheduling under contention.
///   - Freeze + Kill: On timeout, `cgroup.freeze` blocks all processes in the subtree, then `cgroup.kill` terminates them. Requires Linux 5.14+ (available on Ubuntu 24.04, kernel 6.17+).
///
/// ## Platform & Privilege Requirements
///
/// **This crate is Linux-only** (enforced by `#[cfg(target_os = "linux")]`). All modules gate their compilation on this platform.
///
/// **Unprivileged execution requirement:** The crate is designed to run as a regular, unprivileged user. It does NOT require root. Instead, it relies on **cgroup v2 user delegation**, which must be configured by the system administrator or init system (typically systemd) at startup:
///
/// ```bash
/// # Example: systemd user session delegation
/// systemctl --user start-session
/// # Confirms per-user delegated slice at:
/// # /sys/fs/cgroup/user.slice/user-<uid>.slice/
/// ```
///
/// For testing, the harness script `bash scripts/wk-sync-test.sh` handles delegation setup automatically by wrapping the test binary in a `systemd-run --user --scope` with safety backstops.
///
/// ## Module Architecture
///
/// - **`worker.rs`**: Core async entry point (`run_sandboxed`) and synchronous namespace setup (`spawn_in_namespaces`). Orchestrates cgroup creation, seccomp filter installation, clone(2), and post-timeout containment (freeze + kill).
/// - **`seccomp.rs`**: Builds a seccomp-BPF allowlist filter for CPython syscalls. Conditional rule denies `clone` with `CLONE_NEWUSER` flag.
/// - **`cgroup.rs`**: Helpers for creating leaf cgroups, writing resource limits, freezing, and killing subtrees.
/// - **`error.rs`**: Error types (`SandboxError`).
///
/// ## Async Runtime
///
/// The `run_sandboxed()` function is async and requires a Tokio runtime. It uses non-blocking sleeps
/// and `waitpid(2)` with `WNOHANG` polling to allow other tasks to run during child execution.
///
/// ## Example Usage
///
/// ```rust,no_run
/// use rlox_sandbox::{run_sandboxed, SandboxConfig, SandboxInput};
/// use std::path::PathBuf;
/// use uuid::Uuid;
///
/// #[tokio::main]
/// async fn main() {
///     let config = SandboxConfig {
///         timeout_secs: 10.0,
///         mem_limit_bytes: 128 * 1024 * 1024,  // 128 MiB
///         pids_limit: 64,
///         cpu_weight: 100,
///         cgroup_base: PathBuf::from("/sys/fs/cgroup/user.slice/user-1000.slice"),
///     };
///
///     let input = SandboxInput {
///         job_id: Uuid::new_v4(),
///         code: "def add(a, b): return a + b".to_string(),
///         test_suite: "assert add(1, 2) == 3".to_string(),
///         language: "python".to_string(),
///         is_adversarial: false,
///     };
///
///     match run_sandboxed(input, &config).await {
///         Ok(output) => println!("Job {} completed with exit: {:?}", output.job_id, output.exit_status),
///         Err(e) => eprintln!("Sandbox error: {}", e),
///     }
/// }
/// ```
///
/// This crate is **Linux-only**. All modules are gated on `target_os = "linux"`.
#[cfg(target_os = "linux")]
pub mod cgroup;

#[cfg(target_os = "linux")]
pub mod seccomp;

#[cfg(target_os = "linux")]
pub mod worker;

pub mod error;

// Step 3 (Cycle 1): rollout server HTTP skeleton and telemetry struct.
// Both modules are platform-independent (no Linux-specific syscalls).
pub mod server;
pub mod stats;

// Re-exports for callers
#[cfg(target_os = "linux")]
pub use worker::{
    run_sandboxed, spawn_in_namespaces, NamespaceChildResult, SandboxConfig, SandboxExitStatus,
    SandboxInput, SandboxOutput, SandboxStats,
};

pub use error::SandboxError;
pub use server::{
    router, router_with_config, router_with_full_config, RolloutRequest, RolloutResponse,
    RolloutTask, SamplingParams, SandboxRunConfig, ServerConfig, Trajectory, VerifyRequest,
    VerifyResponse,
};
pub use stats::BackendStats;
