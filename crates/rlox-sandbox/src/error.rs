use thiserror::Error;

/// All error variants that can be produced by the rlox-sandbox worker.
///
/// Convention matches `rlox-core/src/error.rs`: one enum, `thiserror` derives.
#[derive(Debug, Error)]
pub enum SandboxError {
    /// The cgroup leaf could not be created or written.
    #[error("cgroup error: {0}")]
    Cgroup(String),

    /// A seccomp BPF filter could not be built or installed.
    #[error("seccomp error: {0}")]
    Seccomp(String),

    /// A namespace clone/unshare/mount operation failed.
    #[error("namespace error: {0}")]
    Namespace(String),

    /// The child process could not be spawned.
    #[error("spawn error: {0}")]
    Spawn(String),

    /// An I/O error (cgroup knob read/write, /proc access, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The timeout kill sequence failed to complete cleanly.
    #[error("kill error: {0}")]
    Kill(String),

    /// The requested language is not supported by this worker.
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),
}
