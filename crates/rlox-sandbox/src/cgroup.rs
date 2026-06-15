/// cgroup v2 delegation helpers.
///
/// All operations target the per-user slice:
///   /sys/fs/cgroup/user.slice/user-<uid>.slice/<leaf>/
///
/// On systems where systemd delegates at `user@<uid>.service` (not directly at
/// `user-<uid>.slice`), `create_leaf` automatically finds the first
/// user-writable delegation point under the given parent.
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::SandboxError;

/// Returns the per-user delegated cgroup root for the current UID.
///
/// First tries `/sys/fs/cgroup/user.slice/user-<uid>.slice` (the design
/// target). If that directory is not writable for mkdir, falls back to the
/// first user-owned descendant (typically `user@<uid>.service`) where
/// controllers are delegated.
pub fn user_cgroup_root() -> PathBuf {
    // SAFETY: getuid() is always safe.
    let uid = unsafe { libc::getuid() };
    let slice = PathBuf::from(format!("/sys/fs/cgroup/user.slice/user-{uid}.slice"));

    // If the slice itself is directly writable, use it.
    if is_writable_for_mkdir(&slice) {
        return slice;
    }

    // Walk one level under the slice to find the first user-owned directory
    // with delegated controllers (typically user@<uid>.service).
    if let Ok(entries) = fs::read_dir(&slice) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && is_writable_for_mkdir(&p) {
                let controllers = read_controllers(&p).unwrap_or_default();
                if !controllers.is_empty() {
                    return p;
                }
            }
        }
    }

    // Return the slice path as a best-effort fallback (will fail at mkdir time
    // if not writable, giving a clear error).
    slice
}

/// Returns true if the current process can create a subdirectory of `path`.
///
/// We probe by checking ownership and write bit rather than attempting a
/// temporary mkdir (which would be noisy).
fn is_writable_for_mkdir(path: &Path) -> bool {
    // SAFETY: getuid/getgid are always safe.
    let uid = unsafe { libc::getuid() };
    let gid = unsafe { libc::getgid() };

    // Use libc::stat to get inode metadata without going through std::fs
    // (std::fs::metadata works fine here too).
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let path_cstr = match std::ffi::CString::new(path.to_string_lossy().as_ref()) {
        Ok(s) => s,
        Err(_) => return false,
    };
    // SAFETY: path_cstr is a valid C string; st is properly initialized.
    if unsafe { libc::stat(path_cstr.as_ptr(), &mut st) } != 0 {
        return false;
    }
    let mode = st.st_mode;
    // Owner write+execute
    if st.st_uid == uid && (mode & 0o200 != 0) && (mode & 0o100 != 0) {
        return true;
    }
    // Group write+execute
    if st.st_gid == gid && (mode & 0o020 != 0) && (mode & 0o010 != 0) {
        return true;
    }
    // Other write+execute
    if (mode & 0o002 != 0) && (mode & 0o001 != 0) {
        return true;
    }
    false
}

/// Enable `+memory +pids +cpu` controllers in `parent`'s
/// `cgroup.subtree_control` so that a newly created leaf cgroup sees them.
///
/// Silently succeeds if the controllers are already enabled (EACCES on a
/// read-only knob when controllers are already set is common and harmless).
fn enable_controllers_in_parent(parent: &Path) -> Result<(), SandboxError> {
    let ctrl_path = parent.join("cgroup.subtree_control");

    // Check what's already enabled.
    let current = fs::read_to_string(&ctrl_path).unwrap_or_default();
    let already_has_memory = current.contains("memory");
    let already_has_pids = current.contains("pids");
    let already_has_cpu = current.contains("cpu");

    if already_has_memory && already_has_pids && already_has_cpu {
        // All needed controllers already present; skip the write.
        return Ok(());
    }

    // Build the write string for missing controllers.
    let mut to_add = String::new();
    if !already_has_memory {
        to_add.push_str("+memory ");
    }
    if !already_has_pids {
        to_add.push_str("+pids ");
    }
    if !already_has_cpu {
        to_add.push_str("+cpu ");
    }
    let to_add = to_add.trim_end().to_owned();

    match fs::write(&ctrl_path, format!("{to_add}\n")) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            // If we got EACCES but the controllers are already present,
            // treat this as success (the parent was configured by the system).
            let current = fs::read_to_string(&ctrl_path).unwrap_or_default();
            if current.contains("memory") && current.contains("pids") && current.contains("cpu") {
                Ok(())
            } else {
                Err(SandboxError::Cgroup(format!(
                    "failed to enable controllers in {ctrl_path:?}: {e}"
                )))
            }
        }
        Err(e) => Err(SandboxError::Cgroup(format!(
            "failed to write controllers to {ctrl_path:?}: {e}"
        ))),
    }
}

/// Create a leaf cgroup directory under `parent` with the given `name`.
///
/// After this call the directory `parent/name/` exists.
///
/// If `parent` is not directly writable for mkdir (e.g. it is the systemd
/// slice root, which is owned by root), this function transparently tries the
/// first user-writable cgroup under `parent` (typically
/// `user@<uid>.service`). The returned path reflects where the leaf was
/// actually created.
pub fn create_leaf(parent: &Path, name: &str) -> Result<PathBuf, SandboxError> {
    // Resolve the actual writable parent under the given base.
    let actual_parent = if is_writable_for_mkdir(parent) {
        parent.to_path_buf()
    } else {
        // Find the first user-writable subdirectory under parent.
        let mut found: Option<PathBuf> = None;
        if let Ok(entries) = fs::read_dir(parent) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() && is_writable_for_mkdir(&p) {
                    found = Some(p);
                    break;
                }
            }
        }
        match found {
            Some(p) => p,
            None => {
                return Err(SandboxError::Cgroup(format!(
                    "no writable cgroup delegation found under {parent:?}; \
                     ensure systemd has delegated controllers to your user session"
                )));
            }
        }
    };

    enable_controllers_in_parent(&actual_parent)?;

    let leaf = actual_parent.join(name);
    fs::create_dir(&leaf).map_err(|e| {
        SandboxError::Cgroup(format!("failed to create cgroup leaf {leaf:?}: {e}"))
    })?;
    Ok(leaf)
}

/// Remove the cgroup leaf at `path`.
///
/// The cgroup must have no living processes; kill them first.
pub fn destroy_leaf(path: &Path) -> Result<(), SandboxError> {
    fs::remove_dir(path).map_err(|e| {
        SandboxError::Cgroup(format!("failed to remove cgroup leaf {path:?}: {e}"))
    })
}

/// Write `memory.max` (bytes) to the cgroup at `path`.
pub fn write_memory_max(path: &Path, bytes: u64) -> Result<(), SandboxError> {
    let knob = path.join("memory.max");
    fs::write(&knob, format!("{bytes}\n"))
        .map_err(|e| SandboxError::Cgroup(format!("write memory.max to {knob:?}: {e}")))
}

/// Write `pids.max` to the cgroup at `path`.
pub fn write_pids_max(path: &Path, count: u32) -> Result<(), SandboxError> {
    let knob = path.join("pids.max");
    fs::write(&knob, format!("{count}\n"))
        .map_err(|e| SandboxError::Cgroup(format!("write pids.max to {knob:?}: {e}")))
}

/// Write `cpu.weight` to the cgroup at `path`.
pub fn write_cpu_weight(path: &Path, weight: u32) -> Result<(), SandboxError> {
    let knob = path.join("cpu.weight");
    fs::write(&knob, format!("{weight}\n"))
        .map_err(|e| SandboxError::Cgroup(format!("write cpu.weight to {knob:?}: {e}")))
}

/// Read `cgroup.controllers` from `path` and return the list of enabled
/// controller names (split on whitespace).
pub fn read_controllers(path: &Path) -> Result<Vec<String>, SandboxError> {
    let knob = path.join("cgroup.controllers");
    let content = fs::read_to_string(&knob)
        .map_err(|e| SandboxError::Cgroup(format!("read cgroup.controllers {knob:?}: {e}")))?;
    Ok(content.split_whitespace().map(|s| s.to_owned()).collect())
}

/// Read `memory.max` back from the cgroup at `path`.
/// Returns `u64::MAX` if the file contains "max".
pub fn read_memory_max(path: &Path) -> Result<u64, SandboxError> {
    let knob = path.join("memory.max");
    let content = fs::read_to_string(&knob)
        .map_err(|e| SandboxError::Cgroup(format!("read memory.max {knob:?}: {e}")))?;
    let trimmed = content.trim();
    if trimmed == "max" {
        return Ok(u64::MAX);
    }
    trimmed
        .parse::<u64>()
        .map_err(|e| SandboxError::Cgroup(format!("parse memory.max '{trimmed}': {e}")))
}

/// Read `pids.max` back from the cgroup at `path`.
/// Returns `u32::MAX` if the file contains "max".
pub fn read_pids_max(path: &Path) -> Result<u32, SandboxError> {
    let knob = path.join("pids.max");
    let content = fs::read_to_string(&knob)
        .map_err(|e| SandboxError::Cgroup(format!("read pids.max {knob:?}: {e}")))?;
    let trimmed = content.trim();
    if trimmed == "max" {
        return Ok(u32::MAX);
    }
    trimmed
        .parse::<u32>()
        .map_err(|e| SandboxError::Cgroup(format!("parse pids.max '{trimmed}': {e}")))
}

/// Freeze the entire cgroup subtree rooted at `path` by writing `1` to
/// `cgroup.freeze`.
pub fn freeze(path: &Path) -> Result<(), SandboxError> {
    let knob = path.join("cgroup.freeze");
    fs::write(&knob, "1\n")
        .map_err(|e| SandboxError::Cgroup(format!("write cgroup.freeze {knob:?}: {e}")))
}

/// Kill all processes in the cgroup subtree by writing `1` to `cgroup.kill`.
/// Requires Linux 5.14+.
pub fn kill_subtree(path: &Path) -> Result<(), SandboxError> {
    let knob = path.join("cgroup.kill");
    fs::write(&knob, "1\n")
        .map_err(|e| SandboxError::Cgroup(format!("write cgroup.kill {knob:?}: {e}")))
}
