//! Sandbox isolation verification using `rlox-sandbox`.
//!
//! Runs two jobs through the full Linux sandbox and shows that:
//!
//!   1. A benign Python solution executes cleanly and earns `reward > 0`.
//!   2. An adversarial fork-bomb is fully contained — it never escapes the
//!      cgroup leaf and is killed within the configured timeout.
//!
//! **Linux-only.**  On macOS/Windows the stub below prints a short note.
//!
//! To run on wk-system (inside a delegated cgroup scope):
//!
//! ```bash
//! bash scripts/wk-sync-test.sh \
//!   'cargo run --manifest-path examples/rust/Cargo.toml --bin sandbox_verify'
//! ```
//!
//! Or manually on Linux:
//!
//! ```bash
//! systemd-run --user --scope --slice=rlox.slice -p Delegate=yes \
//!   cargo run --manifest-path examples/rust/Cargo.toml --bin sandbox_verify
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Linux body
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux {
    use std::path::PathBuf;
    use std::time::Instant;

    use rlox_sandbox::{
        run_sandboxed, SandboxConfig, SandboxExitStatus, SandboxInput, SandboxOutput,
    };
    use uuid::Uuid;

    // ── cgroup base discovery ─────────────────────────────────────────────────
    //
    // The sandbox validates `cgroup_base` before spawning untrusted code (Bug A
    // fix in worker.rs).  We discover the per-user delegated slice the same way
    // the integration tests do: `/sys/fs/cgroup/user.slice/user-<uid>.slice`.
    //
    // If the slice is not writable (i.e. the process is NOT running inside a
    // delegated cgroup scope), `run_sandboxed` will return `Err` rather than
    // silently running code without resource limits.  We detect this early and
    // print clear instructions instead of panicking.

    fn discover_cgroup_base() -> PathBuf {
        // SAFETY: getuid is always safe.
        let uid = unsafe { libc::getuid() };
        PathBuf::from(format!(
            "/sys/fs/cgroup/user.slice/user-{uid}.slice"
        ))
    }

    fn check_cgroup_or_abort(base: &PathBuf) {
        if !base.exists() {
            eprintln!("ERROR: cgroup_base {base:?} does not exist.");
            eprintln!();
            eprintln!("The sandbox requires cgroup v2 user-delegation.  Run this example");
            eprintln!("inside a delegated scope:");
            eprintln!();
            eprintln!("  systemd-run --user --scope --slice=rlox.slice -p Delegate=yes \\");
            eprintln!(
                "    cargo run --manifest-path examples/rust/Cargo.toml --bin sandbox_verify"
            );
            eprintln!();
            eprintln!("Or via the repo helper (which wraps the systemd-run automatically):");
            eprintln!();
            eprintln!("  bash scripts/wk-sync-test.sh \\");
            eprintln!(
                "    'cargo run --manifest-path examples/rust/Cargo.toml --bin sandbox_verify'"
            );
            std::process::exit(1);
        }
    }

    fn default_config(cgroup_base: PathBuf, timeout_secs: f64) -> SandboxConfig {
        SandboxConfig {
            timeout_secs,
            mem_limit_bytes: 128 * 1024 * 1024, // 128 MiB
            pids_limit: 32,
            cpu_weight: 100,
            cgroup_base,
        }
    }

    fn print_result(label: &str, input: &SandboxInput, output: &SandboxOutput, wall: f64) {
        let verdict = match &output.exit_status {
            SandboxExitStatus::Clean(0) if output.reward > 0.0 => "PASS (clean, reward > 0)",
            SandboxExitStatus::Clean(code) => {
                if output.reward > 0.0 {
                    "PASS (clean, reward > 0)"
                } else {
                    // non-zero exit or reward == 0 is still reported but not a
                    // containment failure for the benign job
                    &*Box::leak(
                        format!("WARN: exit code {code}, reward={:.3}", output.reward)
                            .into_boxed_str(),
                    )
                }
            }
            SandboxExitStatus::Timeout => "CONTAINED (Timeout)",
            SandboxExitStatus::OomKilled => "CONTAINED (OomKilled)",
            SandboxExitStatus::SetupError(msg) => {
                &*Box::leak(format!("SETUP ERROR: {msg}").into_boxed_str())
            }
        };

        println!();
        println!("── {label} ──");
        println!(
            "  job_id      : {}",
            input.job_id
        );
        println!(
            "  exit_status : {:?}",
            output.exit_status
        );
        println!("  reward      : {:.3}", output.reward);
        println!("  pass_rate   : {:.3}", output.pass_rate);
        println!("  wall_secs   : {:.3}", wall);
        println!("  verdict     : {verdict}");
    }

    #[tokio::main]
    pub async fn run() {
        let cgroup_base = discover_cgroup_base();
        check_cgroup_or_abort(&cgroup_base);

        println!("rlox-sandbox verification");
        println!("  cgroup_base : {cgroup_base:?}");

        // ── Job 1: benign solution ────────────────────────────────────────────

        let benign_code = r#"
def add(a, b):
    return a + b
"#;
        let benign_tests = r#"
assert add(1, 2) == 3, "add(1, 2) should be 3"
assert add(-1, 1) == 0, "add(-1, 1) should be 0"
"#;

        let benign_input = SandboxInput {
            job_id: Uuid::new_v4(),
            code: benign_code.to_owned(),
            test_suite: benign_tests.to_owned(),
            language: "python".to_owned(),
            is_adversarial: false,
        };

        let benign_config = default_config(cgroup_base.clone(), 15.0);
        let t0 = Instant::now();
        let benign_output = run_sandboxed(benign_input.clone(), &benign_config)
            .await
            .expect("run_sandboxed must not Err for benign code with valid cgroup");
        let benign_wall = t0.elapsed().as_secs_f64();

        print_result("job 1: benign solution", &benign_input, &benign_output, benign_wall);

        // Assert the contract.
        assert_eq!(
            benign_output.exit_status,
            SandboxExitStatus::Clean(0),
            "benign job must exit Clean(0), got {:?}",
            benign_output.exit_status
        );
        assert!(
            benign_output.reward > 0.0,
            "benign job must earn reward > 0, got {}",
            benign_output.reward
        );

        // ── Job 2: adversarial fork-bomb ──────────────────────────────────────
        //
        // The fork bomb is contained by pids.max (capped at 32 here) plus the
        // wall-clock timeout.  After `run_sandboxed` returns, all forked
        // descendants are killed atomically by `cgroup.kill` — no orphans leak.

        let fork_bomb_code = "import os\nwhile True:\n    try:\n        os.fork()\n    except Exception:\n        pass\n";

        let adversarial_input = SandboxInput {
            job_id: Uuid::new_v4(),
            code: fork_bomb_code.to_owned(),
            test_suite: String::new(),
            language: "python".to_owned(),
            is_adversarial: true,
        };

        // Short timeout so the example completes quickly.
        let adversarial_config = default_config(cgroup_base.clone(), 4.0);
        let t1 = Instant::now();
        let adversarial_output = run_sandboxed(adversarial_input.clone(), &adversarial_config)
            .await
            .expect("run_sandboxed must not Err for adversarial code with valid cgroup");
        let adversarial_wall = t1.elapsed().as_secs_f64();

        print_result(
            "job 2: fork-bomb (adversarial)",
            &adversarial_input,
            &adversarial_output,
            adversarial_wall,
        );

        // Assert containment.
        assert!(
            matches!(
                adversarial_output.exit_status,
                SandboxExitStatus::Timeout | SandboxExitStatus::OomKilled
            ),
            "fork-bomb must be contained (Timeout or OomKilled), got {:?}",
            adversarial_output.exit_status
        );
        assert_eq!(
            adversarial_output.reward, 0.0,
            "adversarial job must earn reward == 0, got {}",
            adversarial_output.reward
        );
        assert!(
            adversarial_output.stats.cgroup_kill_event,
            "cgroup_kill_event must be true — fork-bomb must have been killed via cgroup"
        );
        // Wall time must be bounded: should not exceed timeout + a small buffer.
        assert!(
            adversarial_wall < 12.0,
            "adversarial job wall time {adversarial_wall:.2} s exceeded safety bound of 12 s"
        );

        println!();
        println!("Both assertions passed.");
        println!("  benign   → Clean exit, reward > 0");
        println!("  fork-bomb → {:?}, reward == 0, cgroup_kill_event == true", adversarial_output.exit_status);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-Linux stub
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(target_os = "linux"))]
fn main() {
    println!(
        "rlox-sandbox is Linux-only (needs namespaces + cgroup v2)."
    );
    println!();
    println!("Build succeeded: the non-Linux stub compiled correctly.");
    println!("To run the full sandbox verification, use wk-system (Linux + cgroup v2):");
    println!();
    println!("  bash scripts/wk-sync-test.sh \\");
    println!("    'cargo run --manifest-path examples/rust/Cargo.toml --bin sandbox_verify'");
}

#[cfg(target_os = "linux")]
fn main() {
    linux::run();
}
