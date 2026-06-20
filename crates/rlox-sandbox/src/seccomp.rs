/// seccomp-BPF allowlist filter for the sandbox child.
///
/// Uses `seccompiler` (Firecracker's pure-Rust BPF builder) — no C libseccomp.
///
/// Allowed: syscalls required by a CPython interpreter running unit tests.
/// Denied: socket/connect/bind (network), ptrace, mount, clone with
///   CLONE_NEWUSER (nested user-ns via conditional rule), syslog, keyctl, bpf.
///
/// The filter is an *allowlist* (default action = Errno(EPERM)), which means
/// every syscall not in the list is denied.  We enumerate all syscalls needed
/// by a CPython 3.x interpreter executing pure-Python unit tests.
///
/// CLONE_NEWUSER filtering:
///   SYS_clone is permitted unconditionally for threads, but a conditional rule
///   returns EPERM when (flags & CLONE_NEWUSER) != 0 (arg0 on x86-64).
///   SYS_clone3 is denied outright because its flags live inside a struct
///   pointer that BPF cannot dereference portably.
use std::collections::BTreeMap;
use std::convert::TryInto;

use seccompiler::{BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter, SeccompRule, TargetArch};

use crate::error::SandboxError;

// ── Size of a single BPF instruction (sock_filter) in bytes ────────────────
// struct sock_filter { u16 code; u8 jt; u8 jf; u32 k; }  → 8 bytes, repr(C)
const SOCK_FILTER_SIZE: usize = 8;

// ── CLONE_NEWUSER flag value (x86-64) ───────────────────────────────────────
const CLONE_NEWUSER: u64 = 0x1000_0000;

// ── Syscall numbers on x86-64 ───────────────────────────────────────────────
// We build a comprehensive allowlist for CPython.  Any syscall that a modern
// Python 3 interpreter needs for running scripts and unit tests is included.
// Network syscalls (socket, connect, bind, sendto, recvfrom, etc.) are NOT
// included, so the default Errno action denies them.
//
// Syscalls deliberately excluded for least privilege:
//   SYS_capset  — CPython unit tests do not need to change capabilities
//   SYS_seccomp — the sandbox installs the filter before exec; the sandboxed
//                 process must not be able to modify its own filter
//   SYS_clone3  — denied outright; BPF cannot filter the clone_args struct
//                 pointer arg portably, and CPython does not require clone3
fn allowed_syscalls() -> Vec<i64> {
    vec![
        // --- process & thread lifecycle ---
        libc::SYS_read,
        libc::SYS_write,
        libc::SYS_readv,
        libc::SYS_writev,
        libc::SYS_pread64,
        libc::SYS_pwrite64,
        libc::SYS_exit,
        libc::SYS_exit_group,
        // SYS_clone: allowed with conditional rule that denies CLONE_NEWUSER
        // (see build_filter for the conditional SeccompRule)
        libc::SYS_clone,
        // SYS_clone3: denied outright — BPF cannot filter clone_args struct args
        libc::SYS_fork,
        libc::SYS_vfork,
        libc::SYS_execve,
        libc::SYS_execveat,
        libc::SYS_wait4,
        libc::SYS_waitid,
        libc::SYS_getpid,
        libc::SYS_getppid,
        libc::SYS_gettid,
        libc::SYS_getuid,
        libc::SYS_getgid,
        libc::SYS_geteuid,
        libc::SYS_getegid,
        libc::SYS_getresuid,
        libc::SYS_getresgid,
        libc::SYS_getgroups,
        libc::SYS_setuid,
        libc::SYS_setgid,
        libc::SYS_setresuid,
        libc::SYS_setresgid,
        libc::SYS_getpgid,
        libc::SYS_getpgrp,
        libc::SYS_setpgid,
        libc::SYS_getsid,
        libc::SYS_setsid,
        libc::SYS_kill,
        libc::SYS_tgkill,
        libc::SYS_tkill,
        libc::SYS_rt_sigaction,
        libc::SYS_rt_sigprocmask,
        libc::SYS_rt_sigreturn,
        libc::SYS_rt_sigsuspend,
        libc::SYS_rt_sigpending,
        libc::SYS_rt_sigtimedwait,
        libc::SYS_sigaltstack,
        // --- memory ---
        libc::SYS_mmap,
        libc::SYS_munmap,
        libc::SYS_mprotect,
        libc::SYS_mremap,
        libc::SYS_madvise,
        libc::SYS_brk,
        libc::SYS_mlock,
        libc::SYS_munlock,
        libc::SYS_mlock2,
        // --- file I/O ---
        libc::SYS_open,
        libc::SYS_openat,
        libc::SYS_openat2,
        libc::SYS_close,
        libc::SYS_close_range,
        libc::SYS_lseek,
        libc::SYS_stat,
        libc::SYS_fstat,
        libc::SYS_lstat,
        libc::SYS_newfstatat,
        libc::SYS_statx,
        libc::SYS_getdents,
        libc::SYS_getdents64,
        libc::SYS_dup,
        libc::SYS_dup2,
        libc::SYS_dup3,
        libc::SYS_ioctl,
        libc::SYS_fcntl,
        libc::SYS_flock,
        libc::SYS_fsync,
        libc::SYS_fdatasync,
        libc::SYS_truncate,
        libc::SYS_ftruncate,
        libc::SYS_chmod,
        libc::SYS_fchmod,
        libc::SYS_chown,
        libc::SYS_fchown,
        libc::SYS_lchown,
        libc::SYS_unlink,
        libc::SYS_unlinkat,
        libc::SYS_rename,
        libc::SYS_renameat,
        libc::SYS_renameat2,
        libc::SYS_mkdir,
        libc::SYS_mkdirat,
        libc::SYS_rmdir,
        libc::SYS_link,
        libc::SYS_linkat,
        libc::SYS_symlink,
        libc::SYS_symlinkat,
        libc::SYS_readlink,
        libc::SYS_readlinkat,
        libc::SYS_creat,
        libc::SYS_access,
        libc::SYS_faccessat,
        libc::SYS_faccessat2,
        libc::SYS_chdir,
        libc::SYS_fchdir,
        libc::SYS_getcwd,
        libc::SYS_pipe,
        libc::SYS_pipe2,
        libc::SYS_eventfd,
        libc::SYS_eventfd2,
        libc::SYS_epoll_create,
        libc::SYS_epoll_create1,
        libc::SYS_epoll_ctl,
        libc::SYS_epoll_wait,
        libc::SYS_epoll_pwait,
        libc::SYS_epoll_pwait2,
        libc::SYS_select,
        libc::SYS_pselect6,
        libc::SYS_poll,
        libc::SYS_ppoll,
        // inotify syscalls omitted — not needed by CPython unit tests; defense-in-depth
        // against cross-job filesystem enumeration (F9).
        // --- time ---
        libc::SYS_clock_gettime,
        libc::SYS_clock_getres,
        libc::SYS_clock_nanosleep,
        libc::SYS_gettimeofday,
        libc::SYS_nanosleep,
        libc::SYS_timer_create,
        libc::SYS_timer_settime,
        libc::SYS_timer_gettime,
        libc::SYS_timer_getoverrun,
        libc::SYS_timer_delete,
        libc::SYS_timerfd_create,
        libc::SYS_timerfd_settime,
        libc::SYS_timerfd_gettime,
        libc::SYS_alarm,
        libc::SYS_times,
        libc::SYS_getitimer,
        libc::SYS_setitimer,
        // --- futex / synchronisation ---
        libc::SYS_futex,
        libc::SYS_futex_waitv,
        libc::SYS_set_robust_list,
        libc::SYS_get_robust_list,
        // --- misc kernel interfaces Python needs ---
        libc::SYS_arch_prctl,
        libc::SYS_prctl,         // Python uses prctl; we leave it open here and
                                  // rely on namespace isolation for privilege constraints
        libc::SYS_uname,
        libc::SYS_sysinfo,
        libc::SYS_getrandom,
        libc::SYS_set_tid_address,
        libc::SYS_prlimit64,
        libc::SYS_getrlimit,
        libc::SYS_setrlimit,
        libc::SYS_umask,
        libc::SYS_sendfile,
        libc::SYS_copy_file_range,
        libc::SYS_mknod,
        libc::SYS_mknodat,
        // --- scheduling ---
        libc::SYS_sched_getaffinity,
        libc::SYS_sched_setaffinity,
        libc::SYS_sched_getparam,
        libc::SYS_sched_setparam,
        libc::SYS_sched_getscheduler,
        libc::SYS_sched_setscheduler,
        libc::SYS_sched_get_priority_max,
        libc::SYS_sched_get_priority_min,
        libc::SYS_sched_yield,
        // --- memory-mapped file helpers ---
        libc::SYS_msync,
        libc::SYS_mincore,
        // --- process resource ---
        libc::SYS_getrusage,
        libc::SYS_capget,
        // SYS_capset omitted — least privilege: sandboxed code must not change capabilities
        // SYS_seccomp omitted — sandboxed code must not modify its own filter
        // --- signals (Python ctypes / cffi) ---
        libc::SYS_signalfd,
        libc::SYS_signalfd4,
    ]
}

/// Build the seccomp-BPF filter and return it as a compiled byte blob.
///
/// The returned `Vec<u8>` is the raw bytes of the `BpfProgram`
/// (i.e. the sequence of `sock_filter` structs, each 8 bytes).
///
/// # CLONE_NEWUSER denial
///
/// `SYS_clone` is in the allowlist but with a conditional rule:
///   - If `(arg0 & CLONE_NEWUSER) != 0` → EPERM (deny nested user namespace)
///   - Otherwise → Allow
///
/// `SYS_clone3` is denied outright because its flags live inside a
/// `clone_args` struct pointer that BPF cannot dereference portably.
pub fn build_filter() -> Result<Vec<u8>, SandboxError> {
    let arch: TargetArch = std::env::consts::ARCH
        .try_into()
        .map_err(|_| SandboxError::Seccomp("unsupported architecture".to_owned()))?;

    // Build rules map. For most syscalls: empty Vec<SeccompRule> = allow unconditionally.
    // For SYS_clone: a conditional rule denies the call when CLONE_NEWUSER is set in arg0.
    // seccompiler requires BTreeMap, not HashMap.
    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = allowed_syscalls()
        .into_iter()
        .map(|nr| (nr, vec![]))
        .collect();

    // ── Conditional deny for SYS_clone with CLONE_NEWUSER ───────────────────
    // seccompiler's rule semantics: a SeccompRule is an "allow" rule with
    // optional conditions that must ALL match for the rule to fire (allow).
    // The match_action is Allow and mismatch_action is Errno(EPERM).
    //
    // We want:
    //   clone(flags & CLONE_NEWUSER == 0) → Allow
    //   clone(flags & CLONE_NEWUSER != 0) → Errno(EPERM)
    //
    // seccompiler's SeccompCondition with MaskedEq checks:
    //   (arg & mask) == val
    //
    // So we allow clone only when (arg0 & CLONE_NEWUSER) == 0.
    // Any call where CLONE_NEWUSER is set falls through to the mismatch_action (Errno EPERM).
    // MaskedEq(mask): allow clone only when (arg0 & CLONE_NEWUSER) == 0.
    // Any call with CLONE_NEWUSER set fails to match and gets mismatch_action = Errno(EPERM).
    let clone_newuser_not_set = SeccompCondition::new(
        0,                                   // arg index 0 = flags (unsigned long, 64-bit)
        SeccompCmpArgLen::Qword,             // full 64-bit comparison
        SeccompCmpOp::MaskedEq(CLONE_NEWUSER), // (flags & CLONE_NEWUSER) must equal...
        0,                                   // ...0, meaning CLONE_NEWUSER bit is NOT set
    )
    .map_err(|e| SandboxError::Seccomp(format!("SeccompCondition for clone: {e}")))?;

    let clone_allow_rule = SeccompRule::new(vec![clone_newuser_not_set])
        .map_err(|e| SandboxError::Seccomp(format!("SeccompRule for clone: {e}")))?;

    // Replace the unconditional clone entry with the conditional rule.
    rules.insert(libc::SYS_clone, vec![clone_allow_rule]);

    let filter = SeccompFilter::new(
        rules,
        // mismatch_action: default action for syscalls NOT in the list → deny
        SeccompAction::Errno(libc::EPERM as u32),
        // match_action: action for syscalls that ARE in the list → allow
        SeccompAction::Allow,
        arch,
    )
    .map_err(|e| SandboxError::Seccomp(format!("SeccompFilter::new: {e}")))?;

    let program: BpfProgram = filter
        .try_into()
        .map_err(|e| SandboxError::Seccomp(format!("filter compile: {e}")))?;

    // Serialize Vec<sock_filter> → Vec<u8> using raw byte copy.
    // Each sock_filter is #[repr(C)] and exactly SOCK_FILTER_SIZE bytes.
    let byte_len = program.len() * SOCK_FILTER_SIZE;
    let mut bytes = vec![0u8; byte_len];
    // SAFETY: sock_filter is #[repr(C)] with no padding at its size boundary;
    // the cast to *const u8 is valid for the lifetime of `program`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            program.as_ptr() as *const u8,
            bytes.as_mut_ptr(),
            byte_len,
        );
    }
    Ok(bytes)
}

/// Install the pre-compiled filter in the calling process/thread via
/// `prctl(PR_SET_NO_NEW_PRIVS, 1)` then the seccomp syscall.
///
/// Must be called *after* `clone()` inside the child.
pub fn install_filter(compiled: &[u8]) -> Result<(), SandboxError> {
    if compiled.is_empty() {
        return Err(SandboxError::Seccomp("empty BPF filter blob".to_owned()));
    }
    if !compiled.len().is_multiple_of(SOCK_FILTER_SIZE) {
        return Err(SandboxError::Seccomp(format!(
            "BPF blob length {} is not a multiple of sock_filter size {}",
            compiled.len(),
            SOCK_FILTER_SIZE
        )));
    }

    let n_instructions = compiled.len() / SOCK_FILTER_SIZE;

    // Reconstruct a BpfProgram (Vec<sock_filter>) from the byte blob.
    let mut program: BpfProgram = Vec::with_capacity(n_instructions);
    // SAFETY: We verify the byte count is an exact multiple of SOCK_FILTER_SIZE.
    // sock_filter is #[repr(C)]; all bit patterns are valid (it contains only
    // integer fields with no invalid states).
    unsafe {
        let src = compiled.as_ptr() as *const seccompiler::sock_filter;
        for i in 0..n_instructions {
            program.push(std::ptr::read_unaligned(src.add(i)));
        }
    }

    seccompiler::apply_filter(&program)
        .map_err(|e| SandboxError::Seccomp(format!("apply_filter: {e}")))
}

/// Convenience: build and immediately install the filter.
pub fn build_and_install() -> Result<(), SandboxError> {
    let blob = build_filter()?;
    install_filter(&blob)
}
