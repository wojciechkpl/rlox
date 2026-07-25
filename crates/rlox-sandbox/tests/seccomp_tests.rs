/// Step 1b — seccomp-BPF filter
///
/// Tests verify the public contract of `rlox_sandbox::seccomp`.
///
/// Test strategy for network-deny:
///   Fork a child with `std::process::Command` cannot be used directly because
///   the filter must be installed inside the child *after* it is running.
///   Instead we use `nix::unistd::fork()` so we can install the filter in the
///   child before making any network syscall.  The parent inspects the child's
///   exit status to determine whether the syscall was blocked.
///
/// Convention for child-fork tests:
///   - Child exits with code 0 if the syscall was correctly blocked (EPERM/ENOSYS/SIGSYS).
///   - Child exits with code 1 if the syscall succeeded (filter not installed).
///   - Child exits with code 2 if the filter installation itself errored.
///
/// The parent asserts exit code == 0.
#[cfg(target_os = "linux")]
mod seccomp_tests {
    use nix::sys::wait::{waitpid, WaitStatus};
    use nix::unistd::{fork, ForkResult};
    use rlox_sandbox::seccomp;

    // -----------------------------------------------------------------------
    // Test 1b-1: build_filter returns Ok and produces a non-empty blob.
    // -----------------------------------------------------------------------
    #[test]
    fn test_build_filter_returns_non_empty_bytes() {
        let compiled = seccomp::build_filter().expect("build_filter should succeed");
        assert!(
            !compiled.is_empty(),
            "compiled BPF filter must not be empty"
        );
    }

    // -----------------------------------------------------------------------
    // Test 1b-2: install_filter succeeds in a forked child without crashing.
    //
    // The filter is installed in the child; child exits 0 on success.
    // Parent asserts clean exit.  (A crash/SIGSYS here means the filter itself
    // blocks a syscall we need for the exit — that's a bug in the allowlist.)
    // -----------------------------------------------------------------------
    #[test]
    fn test_install_filter_in_child_succeeds_without_crash() {
        let compiled = seccomp::build_filter().expect("build_filter");

        let compiled_clone = compiled.clone();
        // Safety: fork is inherently unsafe; no shared data is mutated.
        match unsafe { fork() }.expect("fork should succeed") {
            ForkResult::Child => match seccomp::install_filter(&compiled_clone) {
                Ok(()) => std::process::exit(0),
                Err(_) => std::process::exit(2),
            },
            ForkResult::Parent { child } => {
                let status = waitpid(child, None).expect("waitpid");
                match status {
                    WaitStatus::Exited(_, code) => {
                        assert_eq!(
                            code, 0,
                            "child exited with {code}: \
                             0=ok, 2=filter install error"
                        );
                    }
                    other => panic!("unexpected child exit: {other:?}"),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 1b-3: after installing the filter, socket(AF_INET, SOCK_STREAM, 0)
    // is denied (child exits 0 = blocked, parent asserts this).
    //
    // The denial may manifest as EPERM, ENOSYS, or SIGSYS (killed by signal 31).
    // We accept any of these as correct containment.
    // -----------------------------------------------------------------------
    #[test]
    fn test_filter_denies_af_inet_socket_syscall() {
        let compiled = seccomp::build_filter().expect("build_filter");

        match unsafe { fork() }.expect("fork should succeed") {
            ForkResult::Child => {
                // Install filter
                if seccomp::install_filter(&compiled).is_err() {
                    std::process::exit(2);
                }
                // Attempt socket(AF_INET=2, SOCK_STREAM=1, 0)
                let ret = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
                if ret == -1 {
                    // Syscall was blocked (EPERM or ENOSYS) — correct.
                    std::process::exit(0);
                } else {
                    // Syscall succeeded — filter is NOT blocking network.
                    unsafe { libc::close(ret) };
                    std::process::exit(1);
                }
            }
            ForkResult::Parent { child } => {
                let status = waitpid(child, None).expect("waitpid");
                match status {
                    WaitStatus::Exited(_, code) => {
                        assert_eq!(
                            code, 0,
                            "child exit {code}: \
                             0=blocked(good) 1=socket succeeded(bad) 2=install error"
                        );
                    }
                    // SIGSYS (signal 31) also means seccomp killed the process — correct.
                    WaitStatus::Signaled(_, sig, _) => {
                        assert_eq!(
                            sig,
                            nix::sys::signal::Signal::SIGSYS,
                            "unexpected signal {sig:?}; expected SIGSYS from seccomp"
                        );
                    }
                    other => panic!("unexpected child exit: {other:?}"),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 1b-4: after installing the filter, write(1, …) is allowed.
    //
    // `write` to stdout must be in the allowlist; if it were blocked the
    // Python interpreter could not produce output.
    // -----------------------------------------------------------------------
    #[test]
    fn test_filter_allows_write_to_stdout() {
        let compiled = seccomp::build_filter().expect("build_filter");

        match unsafe { fork() }.expect("fork should succeed") {
            ForkResult::Child => {
                if seccomp::install_filter(&compiled).is_err() {
                    std::process::exit(2);
                }
                let msg = b"ok\n";
                let ret = unsafe { libc::write(1, msg.as_ptr() as *const libc::c_void, msg.len()) };
                if ret >= 0 {
                    std::process::exit(0); // write allowed — correct
                } else {
                    std::process::exit(1); // write blocked — bug in allowlist
                }
            }
            ForkResult::Parent { child } => {
                let status = waitpid(child, None).expect("waitpid");
                match status {
                    WaitStatus::Exited(_, code) => {
                        assert_eq!(
                            code, 0,
                            "child exit {code}: 0=write allowed(good) 1=write blocked(bad) 2=install error"
                        );
                    }
                    other => panic!("unexpected child exit: {other:?}"),
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
extern crate libc;
