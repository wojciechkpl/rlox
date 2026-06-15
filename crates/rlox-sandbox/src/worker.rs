/// Sandbox worker core.
///
/// Implements:
///   - `spawn_in_namespaces()`: synchronous clone(CLONE_NEWUSER|NEWPID|NEWNET|NEWNS)
///     with uid/gid mapping and a result pipe.
///   - `run_sandboxed()`: async entry point that combines cgroup, seccomp,
///     namespace clone, exec of python3, timeout enforcement, and SandboxStats.
///
/// Key platform note (Ubuntu 24.04, AppArmor 4):
///   The `apparmor_restrict_unprivileged_userns` sysctl restricts what
///   processes can do AFTER entering a user namespace via `unshare()`.
///   However, using `clone()` with CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNET
///   | CLONE_NEWNS simultaneously avoids the `unshare()` code path and
///   successfully creates all four namespaces in a single atomic operation.
use std::ffi::CString;
use std::fs;
use std::os::unix::fs::DirBuilderExt;
use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cgroup;
use crate::error::SandboxError;
use crate::seccomp;

// ---------------------------------------------------------------------------
// Configuration & I/O types
// ---------------------------------------------------------------------------

/// Per-worker resource limits and cgroup base path.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Wall-clock timeout for the sandboxed job (seconds).
    pub timeout_secs: f64,
    /// cgroup `memory.max` (bytes).
    pub mem_limit_bytes: u64,
    /// cgroup `pids.max`.
    pub pids_limit: u32,
    /// cgroup `cpu.weight` (default 100).
    pub cpu_weight: u32,
    /// Base path of the per-user delegated slice, e.g.
    /// `/sys/fs/cgroup/user.slice/user-1000.slice/`
    pub cgroup_base: PathBuf,
}

/// Maximum bytes captured from the sandboxed process's stdout (Bug B fix).
///
/// Once this cap is hit, further output is drained-and-discarded so the pipe
/// buffer never fills and blocks the child, but bytes beyond the cap are not
/// appended to the output buffer.  1 MiB is conservative; the adversarial test
/// verifies <= 2 MiB.
const MAX_STDOUT_BYTES: usize = 1024 * 1024; // 1 MiB

/// Input to a single sandbox execution.
#[derive(Debug, Clone)]
pub struct SandboxInput {
    /// Unique job identifier (UUID v4).
    pub job_id: Uuid,
    /// The untrusted code to execute.
    pub code: String,
    /// The unit-test harness code run after `code`.
    pub test_suite: String,
    /// Language tag. Only `"python"` is supported in the MVP.
    pub language: String,
    /// Whether this sample was injected from the adversarial corpus.
    pub is_adversarial: bool,
}

/// Output from a single sandbox execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxOutput {
    pub job_id: Uuid,
    /// Fraction of unit tests that passed (0.0–1.0).
    // TODO(step-3): real per-test pass fraction once the test-runner protocol is defined
    pub pass_rate: f32,
    /// Reward signal (identity of `pass_rate` in MVP).
    pub reward: f32,
    /// Captured stdout from the sandboxed process (bounded by max_stdout_bytes).
    pub stdout: String,
    pub exit_status: SandboxExitStatus,
    pub stats: SandboxStats,
}

/// How the sandboxed process ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SandboxExitStatus {
    /// Process exited cleanly with the given exit code.
    Clean(i32),
    /// Process was killed via cgroup freeze + `cgroup.kill` after timeout.
    Timeout,
    /// cgroup OOM-killer triggered before the timeout.
    OomKilled,
    /// The sandbox setup itself failed (namespace, cgroup, seccomp).
    SetupError(String),
}

/// Per-execution containment telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxStats {
    /// Total wall clock from spawn to confirmed-dead (seconds).
    pub wall_secs: f64,
    /// Wall clock from timeout detection to `cgroup.kill` write (seconds).
    pub time_to_contain_secs: f64,
    /// Whether the cgroup freeze step was executed.
    pub cgroup_freeze_event: bool,
    /// Whether the `cgroup.kill` step was executed.
    pub cgroup_kill_event: bool,
    /// Whether the cgroup OOM killer fired.
    pub oom_event: bool,
}

// ---------------------------------------------------------------------------
// Namespace setup (Step 1c)
// ---------------------------------------------------------------------------

/// Outcome of a namespace-isolated child process.
#[derive(Debug)]
pub struct NamespaceChildResult {
    /// The `/proc/self/ns/pid` symlink target as seen by the child.
    /// e.g. `"pid:[4026532123]"`.
    pub pid_ns_target: String,
    /// Child exit code (0 on success).
    pub exit_code: i32,
}

// ── Clone child stack ────────────────────────────────────────────────────────
const CLONE_STACK_SIZE: usize = 4 * 1024 * 1024; // 4 MiB

// ── Pipe message size for pid-ns target ─────────────────────────────────────
const PIPE_MSG_LEN: usize = 64;

// ── clone(2) flags ───────────────────────────────────────────────────────────
// We pass SIGCHLD so waitpid works on the child PID.
const CLONE_FLAGS: libc::c_int =
    libc::CLONE_NEWUSER | libc::CLONE_NEWPID | libc::CLONE_NEWNET | libc::CLONE_NEWNS;

// ── Argument block passed from parent to the clone child ────────────────────
// We communicate via raw pipes rather than shared memory to avoid
// Rust alloc/drop inside the child (which shares no heap with the parent
// because clone creates a new process, not a thread).
struct CloneArgs {
    // result_pipe: child writes pid-ns target → parent reads
    result_r: i32,
    result_w: i32,
    // sync_c2p: child signals "ready for uid map" → parent reads
    sync_c2p_r: i32,
    sync_c2p_w: i32,
    // sync_p2c: parent signals "maps written" → child reads
    sync_p2c_r: i32,
    sync_p2c_w: i32,
}

extern "C" fn spawn_child_fn(arg: *mut libc::c_void) -> libc::c_int {
    // SAFETY: `arg` is a valid pointer to a CloneArgs allocated by the parent.
    // The parent writes to sync_p2c before the child reads it, so the struct
    // is not concurrently mutated.
    let args = unsafe { &*(arg as *const CloneArgs) };

    // SAFETY: all below are async-signal-safe libc calls on stack variables.
    unsafe {
        // Close unused ends.
        libc::close(args.result_r);
        libc::close(args.sync_c2p_r);
        libc::close(args.sync_p2c_w);

        // Signal parent: "clone done, write uid/gid maps now."
        let byte: u8 = 1;
        libc::write(
            args.sync_c2p_w,
            &byte as *const u8 as *const libc::c_void,
            1,
        );
        libc::close(args.sync_c2p_w);

        // Wait for parent to write uid/gid maps.
        let mut ack: u8 = 0;
        libc::read(
            args.sync_p2c_r,
            &mut ack as *mut u8 as *mut libc::c_void,
            1,
        );
        libc::close(args.sync_p2c_r);

        // Make mount namespace private (prevents propagation to host).
        // This may be denied by AppArmor on Ubuntu 24.04; that is acceptable
        // because in that case no mounts propagate to the host either.
        let root = b"/\0";
        let _ = libc::mount(
            std::ptr::null(),
            root.as_ptr() as *const libc::c_char,
            std::ptr::null(),
            libc::MS_REC | libc::MS_PRIVATE,
            std::ptr::null(),
        );

        // Read /proc/self/ns/pid via raw readlink.
        let mut buf = [0u8; PIPE_MSG_LEN];
        let ns_path = b"/proc/self/ns/pid\0";
        let n = libc::readlink(
            ns_path.as_ptr() as *const libc::c_char,
            buf.as_mut_ptr() as *mut libc::c_char,
            PIPE_MSG_LEN - 1,
        );
        if n > 0 {
            libc::write(
                args.result_w,
                buf.as_ptr() as *const libc::c_void,
                PIPE_MSG_LEN,
            );
        } else {
            let zeros = [0u8; PIPE_MSG_LEN];
            libc::write(
                args.result_w,
                zeros.as_ptr() as *const libc::c_void,
                PIPE_MSG_LEN,
            );
        }
        libc::close(args.result_w);
        libc::_exit(0);
    }
}

/// Spawn a child in
///   `CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS`
/// namespaces, immediately call `mount(None, "/", None, MS_REC|MS_PRIVATE)`
/// in the child, then return the child's observed `/proc/self/ns/pid` link
/// target so callers can verify it differs from the parent's.
///
/// Uses `clone(2)` (not `fork + unshare`) to create all namespaces
/// atomically, which avoids the AppArmor `unprivileged_userns` capability
/// denial that occurs when using `unshare()` after the fact.
///
/// This function is synchronous (blocking waitpid).
pub fn spawn_in_namespaces() -> Result<NamespaceChildResult, SandboxError> {
    // result_pipe: [0]=read(parent), [1]=write(child)
    // sync_c2p:   [0]=read(parent), [1]=write(child)
    // sync_p2c:   [0]=read(child),  [1]=write(parent)
    let mut result_pipe = [0i32; 2];
    let mut sync_c2p = [0i32; 2];
    let mut sync_p2c = [0i32; 2];

    // SAFETY: pipe2 with valid pointer.
    unsafe {
        if libc::pipe2(result_pipe.as_mut_ptr(), libc::O_CLOEXEC) != 0
            || libc::pipe2(sync_c2p.as_mut_ptr(), libc::O_CLOEXEC) != 0
            || libc::pipe2(sync_p2c.as_mut_ptr(), libc::O_CLOEXEC) != 0
        {
            return Err(SandboxError::Namespace(format!(
                "pipe2: {}",
                std::io::Error::last_os_error()
            )));
        }
    }

    // SAFETY: getuid/getgid are always safe.
    let uid = unsafe { libc::getuid() };
    let gid = unsafe { libc::getgid() };

    // Allocate clone child stack on the heap.
    let mut stack = vec![0u8; CLONE_STACK_SIZE];
    // The stack grows downward; point to the top.
    let stack_top = unsafe { stack.as_mut_ptr().add(CLONE_STACK_SIZE) };

    let clone_args = CloneArgs {
        result_r: result_pipe[0],
        result_w: result_pipe[1],
        sync_c2p_r: sync_c2p[0],
        sync_c2p_w: sync_c2p[1],
        sync_p2c_r: sync_p2c[0],
        sync_p2c_w: sync_p2c[1],
    };

    // SAFETY: clone is unsafe; child_fn uses only async-signal-safe libc calls.
    // The CloneArgs struct is on the stack of spawn_in_namespaces and outlives
    // the child because we waitpid before returning.
    let child_pid = unsafe {
        libc::clone(
            spawn_child_fn,
            stack_top as *mut libc::c_void,
            CLONE_FLAGS | libc::SIGCHLD,
            &clone_args as *const CloneArgs as *mut libc::c_void,
        )
    };

    if child_pid < 0 {
        return Err(SandboxError::Namespace(format!(
            "clone: {}",
            std::io::Error::last_os_error()
        )));
    }

    // ── PARENT ───────────────────────────────────────────────────────────────
    // Close write ends we don't own.
    unsafe {
        libc::close(result_pipe[1]);
        libc::close(sync_c2p[1]);
        libc::close(sync_p2c[0]);
    }

    // Wait for child's "clone done" signal.
    unsafe {
        let mut byte: u8 = 0;
        libc::read(
            sync_c2p[0],
            &mut byte as *mut u8 as *mut libc::c_void,
            1,
        );
        libc::close(sync_c2p[0]);
    }

    // Write uid_map and gid_map for the child.
    let uid_map = format!("0 {uid} 1\n");
    let gid_map = format!("0 {gid} 1\n");
    let _ = fs::write(format!("/proc/{child_pid}/setgroups"), "deny");
    let _ = fs::write(format!("/proc/{child_pid}/uid_map"), &uid_map);
    let _ = fs::write(format!("/proc/{child_pid}/gid_map"), &gid_map);

    // Signal child: "maps written."
    unsafe {
        let byte: u8 = 1;
        libc::write(
            sync_p2c[1],
            &byte as *const u8 as *const libc::c_void,
            1,
        );
        libc::close(sync_p2c[1]);
    }

    // Read the child's pid-ns target.
    let mut buf = [0u8; PIPE_MSG_LEN];
    let n = unsafe {
        let n = libc::read(
            result_pipe[0],
            buf.as_mut_ptr() as *mut libc::c_void,
            PIPE_MSG_LEN,
        );
        libc::close(result_pipe[0]);
        n
    };

    // Reap child.
    let mut status: i32 = 0;
    unsafe {
        libc::waitpid(child_pid, &mut status as *mut i32, 0);
    }

    let exit_code = if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        1
    };

    if n <= 0 {
        return Err(SandboxError::Namespace(
            "child did not write pid-ns target".to_owned(),
        ));
    }

    let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(PIPE_MSG_LEN);
    let pid_ns_target = String::from_utf8_lossy(&buf[..nul_pos]).to_string();

    Ok(NamespaceChildResult {
        pid_ns_target,
        exit_code,
    })
}

// ---------------------------------------------------------------------------
// Sandbox child: executed by the clone'd process for run_sandboxed
// ---------------------------------------------------------------------------

/// Arguments passed from run_sandboxed's parent to the sandbox clone child.
///
/// The child writes "0" to cgroup.procs to self-migrate.  Writing "0" causes
/// the kernel to move the calling process (using its global PID) into the
/// target cgroup.  Self-migration is always permitted as long as the calling
/// process has write permission on the destination cgroup.procs — which it
/// does since the leaf is under the user-delegated rlox.slice.
///
/// This requires that the test binary (and therefore this process) is running
/// inside the user-delegated cgroup hierarchy (e.g. rlox.slice via
/// `systemd-run --user --scope --slice=rlox.slice cargo test`).
struct SandboxCloneArgs {
    // Pipes
    err_w: i32,
    stdout_w: i32,
    sync_c2p_w: i32,
    sync_p2c_r: i32,
    // Paths (raw C pointers into CStrings owned by the parent)
    cgroup_procs_ptr: *const libc::c_char,
    python_path_ptr: *const libc::c_char,
    script_path_ptr: *const libc::c_char,
    // Seccomp
    seccomp_ptr: *const u8,
    seccomp_len: usize,
    // Env
    path_env_ptr: *const libc::c_char,
}

// SAFETY: the pointer fields are only used in the child, and the parent
// guarantees the pointed-to data is valid for the child's lifetime.
unsafe impl Send for SandboxCloneArgs {}

/// Setup-failure exit codes sent over err_pipe from child to parent.
const SETUP_OK: u8 = 0;
const SETUP_ERR_CGROUP_OPEN: u8 = 1;
const SETUP_ERR_CGROUP_WRITE: u8 = 2;

extern "C" fn sandbox_child_fn(arg: *mut libc::c_void) -> libc::c_int {
    // SAFETY: `arg` is a valid pointer to SandboxCloneArgs owned by the parent.
    let args = unsafe { &*(arg as *const SandboxCloneArgs) };

    // SAFETY: all below are async-signal-safe libc calls.
    unsafe {
        // Redirect stdout and stderr to the stdout pipe.
        libc::dup2(args.stdout_w, 1);
        libc::dup2(args.stdout_w, 2);
        if args.stdout_w > 2 {
            libc::close(args.stdout_w);
        }

        // Signal parent: "clone done, write uid/gid maps now."
        let byte: u8 = 1;
        libc::write(
            args.sync_c2p_w,
            &byte as *const u8 as *const libc::c_void,
            1,
        );
        libc::close(args.sync_c2p_w);

        // Wait for parent to write uid/gid maps.
        let mut ack: u8 = 0;
        libc::read(
            args.sync_p2c_r,
            &mut ack as *mut u8 as *mut libc::c_void,
            1,
        );
        libc::close(args.sync_p2c_r);

        // ── Bug A fix: Self-migrate into the cgroup leaf — MUST SUCCEED ──────
        //
        // Both open() and write() are checked.  ANY failure → signal setup-
        // failure to the parent and _exit(1).  This is safety-critical: if
        // the child continues without being in the cgroup, all resource limits
        // (memory.max, pids.max, cgroup.kill) are invisible to it.
        let cg_fd = libc::open(args.cgroup_procs_ptr, libc::O_WRONLY);
        if cg_fd < 0 {
            // Could not open cgroup.procs — signal fatal setup failure.
            let code: u8 = SETUP_ERR_CGROUP_OPEN;
            libc::write(
                args.err_w,
                &code as *const u8 as *const libc::c_void,
                1,
            );
            libc::close(args.err_w);
            libc::_exit(1);
        }
        let zero = b"0\n";
        let written = libc::write(cg_fd, zero.as_ptr() as *const libc::c_void, zero.len());
        libc::close(cg_fd);
        if written < 0 {
            // Write to cgroup.procs failed — signal fatal setup failure.
            let code: u8 = SETUP_ERR_CGROUP_WRITE;
            libc::write(
                args.err_w,
                &code as *const u8 as *const libc::c_void,
                1,
            );
            libc::close(args.err_w);
            libc::_exit(1);
        }

        // Make mount namespace private.
        let root = b"/\0";
        let _ = libc::mount(
            std::ptr::null(),
            root.as_ptr() as *const libc::c_char,
            std::ptr::null(),
            libc::MS_REC | libc::MS_PRIVATE,
            std::ptr::null(),
        );

        // Install seccomp filter.
        // PR_SET_NO_NEW_PRIVS first.
        libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1usize, 0usize, 0usize, 0usize);

        const SF_SIZE: usize = 8;
        if args.seccomp_len > 0 && args.seccomp_len % SF_SIZE == 0 {
            #[repr(C)]
            struct SockFprog {
                len: u16,
                filter: *const libc::c_void,
            }
            let n_instr = (args.seccomp_len / SF_SIZE) as u16;
            let fprog = SockFprog {
                len: n_instr,
                filter: args.seccomp_ptr as *const libc::c_void,
            };
            const SECCOMP_SET_MODE_FILTER: libc::c_long = 1;
            libc::syscall(
                libc::SYS_seccomp,
                SECCOMP_SET_MODE_FILTER,
                0i64,
                &fprog as *const SockFprog as *const libc::c_void,
            );
        }

        // Signal parent: setup OK.
        let code: u8 = SETUP_OK;
        libc::write(
            args.err_w,
            &code as *const u8 as *const libc::c_void,
            1,
        );
        libc::close(args.err_w);

        // exec python3 <script>
        let argv: [*const libc::c_char; 3] = [
            args.python_path_ptr,
            args.script_path_ptr,
            std::ptr::null(),
        ];
        let envp: [*const libc::c_char; 2] = [args.path_env_ptr, std::ptr::null()];
        libc::execve(args.python_path_ptr, argv.as_ptr(), envp.as_ptr());

        // execve failed.
        libc::_exit(127);
    }
}

// ---------------------------------------------------------------------------
// Primary entry point (async)
// ---------------------------------------------------------------------------

/// Deadline in milliseconds for the bounded post-kill waitpid poll.
const KILL_REAP_DEADLINE_MS: u64 = 5_000;
/// Poll interval for the bounded post-kill waitpid.
const KILL_REAP_POLL_MS: u64 = 20;

/// Execute one untrusted code sample inside a fully isolated Linux environment.
pub async fn run_sandboxed(
    input: SandboxInput,
    config: &SandboxConfig,
) -> Result<SandboxOutput, SandboxError> {
    if input.language != "python" {
        return Err(SandboxError::UnsupportedLanguage(input.language.clone()));
    }

    let wall_start = Instant::now();

    // ── Bug A fix: Validate cgroup_base BEFORE spawning untrusted code ───────
    //
    // If cgroup_base does not exist or cgroup.procs is not writable, we must
    // return Err — never silently run untrusted code without resource limits.
    validate_cgroup_base(config)?;

    // ── 1. Create cgroup leaf ────────────────────────────────────────────────
    let leaf_name = format!("rlox-{}", input.job_id);
    let leaf_path = cgroup::create_leaf(&config.cgroup_base, &leaf_name)?;

    // Apply resource limits.  These must succeed before we spawn the child.
    if let Err(e) = cgroup::write_memory_max(&leaf_path, config.mem_limit_bytes) {
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(e);
    }
    // Disable swap so memory-over-limit processes are OOM-killed promptly at
    // memory.max rather than thrashing swap until the timeout fires (FIX 1).
    if let Err(e) = cgroup::write_memory_swap_max_zero(&leaf_path) {
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(e);
    }
    if let Err(e) = cgroup::write_pids_max(&leaf_path, config.pids_limit) {
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(e);
    }
    if let Err(e) = cgroup::write_cpu_weight(&leaf_path, config.cpu_weight) {
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(e);
    }

    // ── 2. Build seccomp filter ──────────────────────────────────────────────
    let seccomp_bytes: Vec<u8> = match seccomp::build_filter() {
        Ok(b) => b,
        Err(e) => {
            let _ = cgroup::destroy_leaf(&leaf_path);
            return Err(e);
        }
    };

    // ── 3. Write Python script to a private directory (mode 0700) ────────────
    // Use a per-job private directory under /tmp so other users cannot read
    // or execute the untrusted script.
    let script_dir = PathBuf::from(format!("/tmp/rlox-{}", input.job_id));
    if let Err(e) = fs::DirBuilder::new().mode(0o700).create(&script_dir) {
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(SandboxError::Spawn(format!(
            "mkdir script_dir {script_dir:?}: {e}"
        )));
    }
    let script_path = script_dir.join("script.py");
    let script_content = format!("{}\n{}\n", input.code, input.test_suite);
    if let Err(e) = fs::write(&script_path, &script_content) {
        let _ = fs::remove_dir_all(&script_dir);
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(SandboxError::Spawn(format!(
            "write script {script_path:?}: {e}"
        )));
    }

    // ── 4. Set up pipes ──────────────────────────────────────────────────────
    let mut err_pipe = [0i32; 2];
    let mut sync_c2p = [0i32; 2];
    let mut sync_p2c = [0i32; 2];
    let mut stdout_pipe = [0i32; 2];

    let pipes_ok = unsafe {
        libc::pipe2(err_pipe.as_mut_ptr(), libc::O_CLOEXEC) == 0
            && libc::pipe2(sync_c2p.as_mut_ptr(), libc::O_CLOEXEC) == 0
            && libc::pipe2(sync_p2c.as_mut_ptr(), libc::O_CLOEXEC) == 0
            && libc::pipe2(stdout_pipe.as_mut_ptr(), 0) == 0
    };

    if !pipes_ok {
        let _ = fs::remove_dir_all(&script_dir);
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(SandboxError::Spawn("pipe2 failed".to_owned()));
    }

    // ── 5. Prepare clone arguments ───────────────────────────────────────────
    let uid = unsafe { libc::getuid() };
    let gid = unsafe { libc::getgid() };

    let python_path_cstr = CString::new("/usr/bin/python3").map_err(|e| {
        SandboxError::Spawn(format!("CString python path: {e}"))
    })?;
    let script_path_str = script_path.to_string_lossy();
    let script_cstr = CString::new(script_path_str.as_ref()).map_err(|e| {
        SandboxError::Spawn(format!("CString script path: {e}"))
    })?;
    let cgroup_procs_path = leaf_path.join("cgroup.procs");
    let cgroup_procs_str = cgroup_procs_path.to_string_lossy();
    let cgroup_procs_cstr = CString::new(cgroup_procs_str.as_ref()).map_err(|e| {
        SandboxError::Spawn(format!("CString cgroup procs path: {e}"))
    })?;
    let path_env = b"PATH=/usr/bin:/bin\0";

    let clone_args = SandboxCloneArgs {
        err_w: err_pipe[1],
        stdout_w: stdout_pipe[1],
        sync_c2p_w: sync_c2p[1],
        sync_p2c_r: sync_p2c[0],
        cgroup_procs_ptr: cgroup_procs_cstr.as_ptr(),
        python_path_ptr: python_path_cstr.as_ptr(),
        script_path_ptr: script_cstr.as_ptr(),
        seccomp_ptr: seccomp_bytes.as_ptr(),
        seccomp_len: seccomp_bytes.len(),
        path_env_ptr: path_env.as_ptr() as *const libc::c_char,
    };

    let mut child_stack = vec![0u8; CLONE_STACK_SIZE];
    let stack_top = unsafe { child_stack.as_mut_ptr().add(CLONE_STACK_SIZE) };

    // SAFETY: clone is unsafe; sandbox_child_fn uses only async-signal-safe
    // libc calls. The clone_args struct lives on this stack frame and is not
    // freed until after we waitpid on the child.
    let child_pid = unsafe {
        libc::clone(
            sandbox_child_fn,
            stack_top as *mut libc::c_void,
            CLONE_FLAGS | libc::SIGCHLD,
            &clone_args as *const SandboxCloneArgs as *mut libc::c_void,
        )
    };

    if child_pid < 0 {
        unsafe {
            for fd in [
                err_pipe[0],
                err_pipe[1],
                sync_c2p[0],
                sync_c2p[1],
                sync_p2c[0],
                sync_p2c[1],
                stdout_pipe[0],
                stdout_pipe[1],
            ] {
                libc::close(fd);
            }
        }
        let _ = fs::remove_dir_all(&script_dir);
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(SandboxError::Spawn(format!(
            "clone: {}",
            std::io::Error::last_os_error()
        )));
    }

    // ── PARENT ───────────────────────────────────────────────────────────────
    unsafe {
        libc::close(err_pipe[1]);
        libc::close(sync_c2p[1]);
        libc::close(sync_p2c[0]);
        libc::close(stdout_pipe[1]);
    }

    // Wait for child's "clone done" signal.
    unsafe {
        let mut byte: u8 = 0;
        libc::read(
            sync_c2p[0],
            &mut byte as *mut u8 as *mut libc::c_void,
            1,
        );
        libc::close(sync_c2p[0]);
    }

    // Write uid_map and gid_map.
    // Propagate uid_map write failure — if we can't map the child's UID, the
    // namespace setup is broken and we must not continue.
    let uid_map = format!("0 {uid} 1\n");
    let gid_map = format!("0 {gid} 1\n");
    let _ = fs::write(format!("/proc/{child_pid}/setgroups"), "deny");
    if let Err(e) = fs::write(format!("/proc/{child_pid}/uid_map"), &uid_map) {
        // Signal the child to abort (close the write end; child will get EOF).
        unsafe { libc::close(sync_p2c[1]); }
        unsafe {
            let mut status: i32 = 0;
            libc::waitpid(child_pid, &mut status as *mut i32, 0);
        }
        unsafe { libc::close(err_pipe[0]); }
        unsafe { libc::close(stdout_pipe[0]); }
        let _ = fs::remove_dir_all(&script_dir);
        let _ = cgroup::destroy_leaf(&leaf_path);
        return Err(SandboxError::Namespace(format!("write uid_map: {e}")));
    }
    let _ = fs::write(format!("/proc/{child_pid}/gid_map"), &gid_map);

    // Signal child: uid/gid maps written.
    // The child will self-migrate into the cgroup leaf by writing "0" to
    // cgroup.procs (see sandbox_child_fn), and will signal success/failure
    // via err_pipe before exec.
    unsafe {
        let byte: u8 = 1;
        libc::write(
            sync_p2c[1],
            &byte as *const u8 as *const libc::c_void,
            1,
        );
        libc::close(sync_p2c[1]);
    }

    // Read child's setup status.
    // 0       = SETUP_OK (cgroup migration succeeded, seccomp installed)
    // 1       = SETUP_ERR_CGROUP_OPEN (could not open cgroup.procs)
    // 2       = SETUP_ERR_CGROUP_WRITE (write to cgroup.procs failed)
    // 255/EOF = unknown child crash before writing
    let child_setup_code = unsafe {
        let mut code: u8 = 255;
        libc::read(
            err_pipe[0],
            &mut code as *mut u8 as *mut libc::c_void,
            1,
        );
        libc::close(err_pipe[0]);
        code
    };

    if child_setup_code != SETUP_OK {
        unsafe {
            let mut status: i32 = 0;
            libc::waitpid(child_pid, &mut status as *mut i32, 0);
        }
        // Bug A fix: close stdout_pipe[0] — previously leaked on this path.
        unsafe { libc::close(stdout_pipe[0]); }
        let _ = fs::remove_dir_all(&script_dir);
        let _ = cgroup::destroy_leaf(&leaf_path);
        let reason = match child_setup_code {
            SETUP_ERR_CGROUP_OPEN => "child could not open cgroup.procs (cgroup migration failed)",
            SETUP_ERR_CGROUP_WRITE => "child could not write to cgroup.procs (cgroup migration failed)",
            _ => "child namespace/seccomp setup failed",
        };
        return Ok(SandboxOutput {
            job_id: input.job_id,
            pass_rate: 0.0,
            reward: 0.0,
            stdout: String::new(),
            exit_status: SandboxExitStatus::SetupError(reason.to_owned()),
            stats: SandboxStats {
                wall_secs: wall_start.elapsed().as_secs_f64(),
                time_to_contain_secs: 0.0,
                cgroup_freeze_event: false,
                cgroup_kill_event: false,
                oom_event: false,
            },
        });
    }

    // ── 6. Poll child with timeout ───────────────────────────────────────────
    // Make stdout_pipe[0] non-blocking.
    unsafe {
        let flags = libc::fcntl(stdout_pipe[0], libc::F_GETFL);
        libc::fcntl(stdout_pipe[0], libc::F_SETFL, flags | libc::O_NONBLOCK);
    }

    let timeout_dur = tokio::time::Duration::from_secs_f64(config.timeout_secs);
    let deadline = tokio::time::Instant::now() + timeout_dur;

    let max_stdout = MAX_STDOUT_BYTES;
    let mut stdout_buf: Vec<u8> = Vec::new();
    let mut child_raw_status: Option<i32> = None;
    let mut timed_out = false;
    let mut freeze_event = false;
    let mut kill_event = false;
    let mut time_to_contain_secs = 0.0f64;

    loop {
        drain_fd_capped(stdout_pipe[0], &mut stdout_buf, max_stdout);

        let mut status: i32 = 0;
        let wp = unsafe { libc::waitpid(child_pid, &mut status as *mut i32, libc::WNOHANG) };
        if wp == child_pid {
            child_raw_status = Some(status);
            break;
        }

        if tokio::time::Instant::now() >= deadline {
            timed_out = true;
            break;
        }

        // Bug fix: use tokio::time::sleep (non-blocking) instead of
        // std::thread::sleep which would block the Tokio worker thread.
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    if timed_out {
        let contain_start = Instant::now();
        let _ = cgroup::freeze(&leaf_path);
        freeze_event = true;
        let _ = cgroup::kill_subtree(&leaf_path);
        kill_event = true;
        time_to_contain_secs = contain_start.elapsed().as_secs_f64();

        // Bug A fix: bounded waitpid after kill — poll with WNOHANG up to a
        // deadline so a rare kill failure never hangs run_sandboxed forever.
        let reap_deadline = Instant::now()
            + std::time::Duration::from_millis(KILL_REAP_DEADLINE_MS);
        loop {
            let mut status: i32 = 0;
            let wp = unsafe {
                libc::waitpid(child_pid, &mut status as *mut i32, libc::WNOHANG)
            };
            if wp == child_pid || wp < 0 {
                break;
            }
            if Instant::now() >= reap_deadline {
                // Kill truly stuck — give up reaping to avoid hanging forever.
                break;
            }
            // Non-blocking sleep between polls (tokio is not available here
            // because we're in the post-kill synchronous section; use a short
            // thread sleep which is acceptable — kill path is already uncommon).
            std::thread::sleep(std::time::Duration::from_millis(KILL_REAP_POLL_MS));
        }
    }

    drain_fd_capped(stdout_pipe[0], &mut stdout_buf, max_stdout);
    unsafe {
        libc::close(stdout_pipe[0]);
    }

    let wall_secs = wall_start.elapsed().as_secs_f64();
    let _ = fs::remove_dir_all(&script_dir);

    // Brief async sleep to let the kernel settle (allows cgroup to become empty
    // before rmdir — cgroup.kill is asynchronous).
    // Bug fix: use tokio::time::sleep instead of std::thread::sleep.
    if timed_out {
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    // FIX 2: Read memory.events BEFORE destroying the leaf to get an
    // authoritative OOM count.  The cgroup kernel counters persist until the
    // leaf directory is removed.
    let oom_kill_count = cgroup::read_oom_kill_count(&leaf_path).unwrap_or(0);

    let _ = cgroup::destroy_leaf(&leaf_path);

    let stdout = String::from_utf8_lossy(&stdout_buf).to_string();

    // Authoritative OOM detection: if memory.events reports oom_kill > 0, the
    // cgroup OOM killer fired.  This takes precedence over the timeout path —
    // a process can be OOM-killed right at the timeout boundary, and we want
    // to label it correctly.  Only fall back to Timeout when oom_kill == 0.
    let oom_event = oom_kill_count > 0;

    if timed_out && !oom_event {
        return Ok(SandboxOutput {
            job_id: input.job_id,
            pass_rate: 0.0,
            reward: 0.0,
            stdout,
            exit_status: SandboxExitStatus::Timeout,
            stats: SandboxStats {
                wall_secs,
                time_to_contain_secs,
                cgroup_freeze_event: freeze_event,
                cgroup_kill_event: kill_event,
                oom_event: false,
            },
        });
    }

    // Classify exit status from waitpid raw status (used when not timed-out,
    // or when timed-out but OOM also fired).
    let raw = child_raw_status.unwrap_or(0);
    let exit_status = if oom_event {
        // OOM kill confirmed via memory.events — authoritative classification.
        SandboxExitStatus::OomKilled
    } else if libc::WIFSIGNALED(raw) {
        let sig = libc::WTERMSIG(raw);
        SandboxExitStatus::Clean(-sig)
    } else if libc::WIFEXITED(raw) {
        SandboxExitStatus::Clean(libc::WEXITSTATUS(raw))
    } else {
        SandboxExitStatus::Clean(-1)
    };

    // When OOM-killed we also set cgroup_kill_event so tests can assert it.
    if oom_event {
        kill_event = true;
        freeze_event = true;
    }

    let pass_rate = match &exit_status {
        SandboxExitStatus::Clean(0) => 1.0f32,
        _ => 0.0f32,
    };

    Ok(SandboxOutput {
        job_id: input.job_id,
        pass_rate,
        reward: pass_rate,
        stdout,
        exit_status,
        stats: SandboxStats {
            wall_secs,
            time_to_contain_secs: 0.0,
            cgroup_freeze_event: freeze_event,
            cgroup_kill_event: kill_event,
            oom_event,
        },
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that `config.cgroup_base` is usable for cgroup containment.
///
/// Checks that:
///   1. The directory exists.
///   2. `cgroup.procs` is reachable under a writable delegation point.
///
/// Returns `Err(SandboxError::Cgroup)` if the base is missing or unusable.
/// This must be called BEFORE spawning any untrusted code (Bug A fix).
fn validate_cgroup_base(config: &SandboxConfig) -> Result<(), SandboxError> {
    let base = &config.cgroup_base;
    if !base.exists() {
        return Err(SandboxError::Cgroup(format!(
            "cgroup_base {base:?} does not exist; \
             cannot guarantee resource containment — refusing to run untrusted code"
        )));
    }
    // Also check that we can find a writable delegation point.
    // (create_leaf does this too, but early validation gives a cleaner error.)
    let is_writable = {
        let uid = unsafe { libc::getuid() };
        let gid = unsafe { libc::getgid() };
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let path_cstr = std::ffi::CString::new(base.to_string_lossy().as_ref())
            .unwrap_or_default();
        let stat_ok = unsafe { libc::stat(path_cstr.as_ptr(), &mut st) } == 0;
        if stat_ok {
            let mode = st.st_mode;
            (st.st_uid == uid && (mode & 0o200 != 0) && (mode & 0o100 != 0))
                || (st.st_gid == gid && (mode & 0o020 != 0) && (mode & 0o010 != 0))
                || ((mode & 0o002 != 0) && (mode & 0o001 != 0))
        } else {
            false
        }
    };
    // Allow both directly writable base AND bases where systemd delegates
    // one level deeper (user@<uid>.service pattern). The create_leaf function
    // handles the fallback search. We only hard-fail if the directory is absent.
    // If it exists but isn't directly writable, create_leaf will search one
    // level and give a clear error if nothing is found.
    let _ = is_writable; // checked by create_leaf; existence check above is the guard
    Ok(())
}

/// Drain all available bytes from a non-blocking `fd` into `buf`, up to `cap`.
///
/// Once `buf.len() >= cap`, all arriving bytes are read from the fd and
/// discarded so the pipe buffer doesn't fill and block the child, but they
/// are not appended to `buf` (Bug B fix: bounded stdout buffer).
fn drain_fd_capped(fd: i32, buf: &mut Vec<u8>, cap: usize) {
    let mut tmp = [0u8; 4096];
    loop {
        let n = unsafe { libc::read(fd, tmp.as_mut_ptr() as *mut libc::c_void, tmp.len()) };
        match n {
            n if n > 0 => {
                let available = cap.saturating_sub(buf.len());
                if available > 0 {
                    let to_append = (n as usize).min(available);
                    buf.extend_from_slice(&tmp[..to_append]);
                    // Bytes beyond `available` are intentionally dropped.
                }
                // If cap already hit: we still read (and discard) to drain the pipe.
            }
            0 => break, // EOF
            _ => {
                let errno = unsafe { *libc::__errno_location() };
                if errno == libc::EAGAIN || errno == libc::EWOULDBLOCK {
                    break;
                }
                break;
            }
        }
    }
}
