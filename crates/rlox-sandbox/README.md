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

# Run all tests. --no-fail-fast matters: cargo otherwise stops at the first
# failing test *binary* and later binaries' failures stay hidden.
cargo test -p rlox-sandbox --no-fail-fast
```

Most tests here need host capabilities a plain shell (or a CI runner) does not
have, so they are gated behind two opt-in environment variables — see
`tests/common/mod.rs`:

| Variable | Unlocks | Requires |
| --- | --- | --- |
| `RLOX_SANDBOX_CGROUP_TESTS` | every test that calls `run_sandboxed` and asserts on the exit status (12 tests across 5 binaries) | cgroup v2 **self-migration** — the process must already sit inside a user-delegated cgroup scope |
| `RLOX_SANDBOX_ADVERSARIAL_TESTS` | tests that detonate real fork/memory/pids bombs (4 tests) | the above, plus a scope-level `TasksMax`/`MemoryMax` backstop |

Without them the sandbox child cannot enter its cgroup leaf, so `run_sandboxed`
returns `SetupError("child could not write to cgroup.procs …")` and every
`Clean`/`Timeout`/`OomKilled` assertion would be checking a setup failure rather
than containment. Those tests therefore skip, printing `SKIP <name>: …` (visible
with `-- --nocapture`), instead of passing vacuously or failing.

`scripts/wk-sync-test.sh` sets both. They are explicit opt-ins rather than runtime
probes on purpose: on a host that *is* supposed to have the capability, a
regression must fail loudly instead of silently self-skipping.

### Integration with CI/CD (macOS, Linux)

The crate **cannot build on macOS** (no Linux namespaces). Use the sync-test helper to run tests on the Linux target host:

```bash
# From the repo root
bash scripts/wk-sync-test.sh 'cargo test -p rlox-sandbox --no-fail-fast'
```

This script:
1. Syncs the repo to `wk-system` (Ubuntu 24.04, kernel 6.17+)
2. Wraps the test binary in `systemd-run --user --scope --slice=rlox.slice` so cgroup v2 user delegation is available (an interactive SSH session lands in a root-owned `session-N.scope`, where self-migration fails)
3. Applies safety backstops — `TasksMax=4096` and `MemoryMax=24G` by default, tunable via `WK_TASKS_MAX` / `WK_MEM_MAX` — so a containment bug cannot exhaust host PIDs or RAM
4. Exports `RLOX_SANDBOX_CGROUP_TESTS=1` and `RLOX_SANDBOX_ADVERSARIAL_TESTS=1`, since only this scope satisfies both
5. Returns the exit code and output

### Adversarial Test Execution

Tests in `tests/adversarial_containment.rs` verify that fork bombs, memory bombs, and other attacks are contained. These tests must run single-threaded to avoid race conditions:

```bash
# Run adversarial containment tests only (on Linux)
cargo test -p rlox-sandbox adversarial_containment -- --test-threads=1

# Via the sync-test helper (from macOS or Linux)
bash scripts/wk-sync-test.sh 'cargo test -p rlox-sandbox adversarial_containment -- --test-threads=1'
```

### Test Suite Overview

Counts below are from a full delegated run on `wk-system` (both gates set).
`(gated)` marks binaries containing tests that skip without the capability vars.

- **`integration_run_sandboxed.rs`** (4 tests, gated): Benign code execution, timeout enforcement, wall-clock timing, unsupported-language `SetupError`.
- **`adversarial_containment.rs`** (5 tests, gated): Fork bombs, memory bombs, pids exhaustion, stdout flooding, nested user-namespace denial.
- **`corpus_containment.rs`** (4 tests, gated): Corpus digest/category integrity (ungated) plus per-sample containment (gated).
- **`security_isolation.rs`** (6 tests, gated): Host-file access, `/proc` scoping, socket-fd leakage.
- **`server_contract.rs`** (8 tests, gated): `POST /rollout` HTTP contract, `BackendStats` fields, vLLM-unreachable → 502.
- **`rollout_pipeline.rs`** (8 tests, gated): Full vLLM → sandbox → group-advantage pipeline and containment telemetry.
- **`verify_endpoint.rs`** (4 tests, gated): `POST /verify` reward contract and adversarial containment.
- **`backend_stats_serde.rs`** (8 tests): Rust↔Python `BackendStats` wire-format round-trip.
- **`cgroup_tests.rs`** (6 tests): Leaf creation, resource-limit setting, freeze/kill operations.
- **`namespace_tests.rs`** (3 tests): Namespace isolation, UID/GID mapping, mount-namespace marking.
- **`seccomp_tests.rs`** (4 tests): Filter building, `CLONE_NEWUSER` conditional denial, allowlist completeness.

**Total: 61 tests** (60 integration + 1 doctest) covering happy-path execution,
timeouts, adversarial containment, and the HTTP contracts.

Both environments report `61 passed; 0 failed`. The gates return early rather than
using a skip attribute, so cargo counts a gated-out test as passed — on CI 16 of
the 61 are no-ops that print `SKIP <name>: …`. Run with `-- --nocapture` to see
which, and do not read a green CI run as proof that containment was exercised;
only a delegated host does that.

## Platform Requirements

**Linux only.** Requires:
- Linux 5.14+ for `cgroup.kill` (available on Ubuntu 24.04, kernel 6.17)
- cgroup v2 with per-user delegation at `/sys/fs/cgroup/user.slice/user-<uid>.slice/`
- Unprivileged user with permission to create namespaces (no root required)

The crate will not compile on macOS or Windows.

## Public API

### Rust entry point: `run_sandboxed()`

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

### HTTP API: `/verify` and `/rollout` endpoints

The crate exposes `rlox-verify-server` binary:

- **`POST /verify`** — sandbox-only reward seam for reward-level hosts (prime-rl/verifiers):
  - Input: `{code, tests, is_adversarial}` (JSON)
  - Output: `{reward, backend_stats}` (JSON)
  - Returns nonce-authenticated reward signal that code cannot forge via `sys.exit(0)` or monkeypatch

- **`POST /rollout`** — full generate+verify service (Component 2 of agentic-benchmark MVP):
  - Calls vLLM `/v1/completions` endpoint
  - Runs each completion through sandbox
  - Computes group-relative advantages via `rlox-rl-ops::GroupRelativeEstimator`
  - Returns trajectories + `BackendStats` telemetry (P3 containment: `adversarial_contained`, `contagion_events`, `setup_error_events`, `time_to_contain_secs`)

Both endpoints enforce nonce-authenticated reward integrity so model code cannot forge results.

## Dependencies

- **`rlox-rl-ops`**: Estimator-agnostic advantage ops (`GroupRelativeEstimator` for GRPO reward computation)
- **`libc`**: Raw Linux syscall bindings (`clone`, `pipe2`, `waitpid`, etc.)
- **`seccompiler`**: Pure-Rust seccomp-BPF filter builder (Firecracker project)
- **`tokio`**: Async runtime for timeout polling
- **`axum`**: HTTP server for `/verify` and `/rollout` endpoints
- **`serde` / `serde_json`**: JSON serialization of output
- **`uuid`**: Job ID generation

No C dependencies (libseccomp is not used; the filter is built in pure Rust). `rlox-rl-ops` is intentionally slim (no `rlox-core` dependency) so sandbox doesn't pull the entire training data-plane.

## License

Dual-licensed under MIT or Apache 2.0 (workspace standard).

---

**See also:**
- [`.wf/design.md` - Full agentic-benchmark MVP design](./../../../.wf/design.md)
- `tests/adversarial_containment.rs` — Detailed threat model and containment verification
- `src/worker.rs` — Async entry point and namespace implementation
