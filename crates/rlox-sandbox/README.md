# rlox-sandbox

Hard-isolation sandbox for executing untrusted, model-generated code with zero contagion.

**Purpose:** Execute arbitrary Python code (from LLM agents) in a fully isolated Linux environment with strong guarantees against resource exhaustion and privilege escalation. This is the Treatment-side execution layer for the rlox agentic-benchmark MVP (see [`.wf/design.md` Component 1](./../../../.wf/design.md)).

## Isolation Guarantees

The sandbox defends against five adversarial categories using layered mechanisms:

| Category | Mechanism | Guarantee |
|----------|-----------|-----------|
| **Infinite loops** | Wall-clock timeout + cgroup freeze + `cgroup.kill` | Process is frozen instantly, then killed atomically. No signal handling or escape. |
| **Fork bombs** | `pids.max` limit + cgroup subtree kill | New process creation fails at limit. All children (even disowned) stay in cgroup; `cgroup.kill` kills the entire subtree. |
| **Memory bombs** | `memory.max` hard cap + OOM killer + bounded stdout (1 MiB) | OOM-killer targets sandboxed process first. Stdout buffer cap prevents parent-side allocation contagion even if child writes 100 MiB. |
| **Unkillable threads** | cgroup freeze + subtree kill | Threads cannot opt out of `cgroup.freeze` or `cgroup.kill`. All threads in cgroup are affected atomically. |
| **Nested user namespaces** | seccomp `clone(CLONE_NEWUSER)` denial | Conditional BPF rule returns `EPERM` if `CLONE_NEWUSER` flag is set. `clone3` denied outright. No privilege escalation possible. |

### Isolation Mechanisms

**Linux namespaces:** User, PID, network, and mount namespaces are created atomically via `clone(CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS, ...)`. This avoids AppArmor `unprivileged_userns` denials.

**seccomp-BPF allowlist:** Pure-Rust bytecode filter (via `seccompiler`) allows only syscalls required by CPython. Network, privilege escalation, and raw syscall modification syscalls are denied (default action: `EPERM`).

**cgroup v2 resource limits:**
- `memory.max`: Hard cap on physical memory (bytes)
- `pids.max`: Hard cap on process/thread count
- `cpu.weight`: Relative CPU scheduling weight
- `cgroup.freeze`: Blocks all processes in subtree
- `cgroup.kill`: Terminates all processes in subtree atomically (Linux 5.14+)

## Building & Testing

### On Linux (wk-system or similar)

```bash
# Build the crate
cargo build -p rlox-sandbox

# Run all tests (requires cgroup v2 user delegation)
cargo test -p rlox-sandbox
```

### Integration with CI/CD (macOS, Linux)

The crate **cannot build on macOS** (no Linux namespaces). Use the sync-test helper to run tests on the Linux target host:

```bash
# From the workspace root (rlox-workspace/)
bash scripts/wk-sync-test.sh 'cargo test -p rlox-sandbox -v'
```

This script:
1. Syncs the workspace to `wk-system` (Ubuntu 24.04, kernel 6.17+)
2. Wraps the test binary in `systemd-run --user --scope --slice=rlox.slice` to ensure cgroup v2 user delegation is available
3. Applies safety backstops: `TasksMax=100` and `MemoryMax=2 GiB` to prevent runaway adversarial tests from affecting the host
4. Returns the exit code and output

### Adversarial Test Execution

Tests in `tests/adversarial_containment.rs` verify that fork bombs, memory bombs, and other attacks are contained. These tests must run single-threaded to avoid race conditions:

```bash
# Run adversarial containment tests only (on Linux)
cargo test -p rlox-sandbox adversarial_containment -- --test-threads=1

# Via the sync-test helper (from macOS or Linux)
bash scripts/wk-sync-test.sh 'cargo test -p rlox-sandbox adversarial_containment -- --test-threads=1'
```

### Test Suite Overview

- **`integration_run_sandboxed.rs`** (3 tests): Benign code execution, timeout enforcement, wall-clock timing.
- **`adversarial_containment.rs`** (7 tests): Fork bombs, memory bombs, stdout flooding, nested user-namespace denial, cgroup base validation.
- **`namespace_tests.rs`** (4 tests): Namespace isolation, UID/GID mapping, mount namespace marking.
- **`cgroup_tests.rs`** (5 tests): Leaf creation, resource limit setting, freeze/kill operations.
- **`seccomp_tests.rs`** (3 tests): Filter building, CLONE_NEWUSER conditional denial, allowlist completeness.

**Total: 22 tests** covering happy-path execution, timeouts, and adversarial containment.

## Platform Requirements

**Linux only.** Requires:
- Linux 5.14+ for `cgroup.kill` (available on Ubuntu 24.04, kernel 6.17)
- cgroup v2 with per-user delegation at `/sys/fs/cgroup/user.slice/user-<uid>.slice/`
- Unprivileged user with permission to create namespaces (no root required)

The crate will not compile on macOS or Windows.

## Public API

The primary entry point is `run_sandboxed()`:

```rust
pub async fn run_sandboxed(
    input: SandboxInput,
    config: &SandboxConfig,
) -> Result<SandboxOutput, SandboxError>
```

**Input (`SandboxInput`):**
- `job_id: Uuid` — unique job identifier
- `code: String` — untrusted Python code
- `test_suite: String` — unit-test harness (executed after code)
- `language: String` — only `"python"` supported in MVP
- `is_adversarial: bool` — whether sample was injected from adversarial corpus

**Output (`SandboxOutput`):**
- `job_id: Uuid` — echoed from input
- `pass_rate: f32` — fraction of test assertions that passed (0.0–1.0)
- `reward: f32` — reward signal (identity of `pass_rate` in MVP)
- `stdout: String` — captured stdout/stderr (bounded to 1 MiB)
- `exit_status: SandboxExitStatus` — `Clean(i32)`, `Timeout`, `OomKilled`, or `SetupError(String)`
- `stats: SandboxStats` — containment telemetry (wall time, freeze/kill events, OOM events)

See `crates/rlox-sandbox/src/worker.rs` for full type definitions.

## Dependencies

- **`libc`**: Raw Linux syscall bindings (`clone`, `pipe2`, `waitpid`, etc.)
- **`seccompiler`**: Pure-Rust seccomp-BPF filter builder (Firecracker project)
- **`tokio`**: Async runtime for timeout polling
- **`serde` / `serde_json`**: JSON serialization of output
- **`uuid`**: Job ID generation

No C dependencies (libseccomp is not used; the filter is built in pure Rust).

## License

Dual-licensed under MIT or Apache 2.0 (workspace standard).

---

**See also:**
- [`.wf/design.md` - Full agentic-benchmark MVP design](./../../../.wf/design.md)
- `tests/adversarial_containment.rs` — Detailed threat model and containment verification
- `src/worker.rs` — Async entry point and namespace implementation
