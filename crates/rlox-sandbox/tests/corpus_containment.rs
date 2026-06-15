/// Step 2a — Adversarial corpus containment tests.
///
/// These tests pin the committed adversarial corpus at
/// `benchmarks/agentic/corpus/adversarial_corpus_v1.json` (plus its companion
/// `.sha256` file) to verified containment guarantees.
///
/// ## SHA-256 convention (implementer MUST hash exactly this way — AC-2/AC-3)
///
/// The `.sha256` file contains the lowercase hex SHA-256 digest of the raw
/// UTF-8 bytes of the `adversarial_corpus_v1.json` file **with the `"sha256"`
/// field's value set to the empty string `""`**.  That is:
///
///   1. Load the JSON file as a `serde_json::Value`.
///   2. Set `root["sha256"] = ""` (empty string).
///   3. Re-serialise to compact JSON (no pretty-printing) using
///      `serde_json::to_string(&root)` — this is the canonical byte sequence.
///   4. Compute SHA-256 over those UTF-8 bytes.
///   5. The resulting lowercase hex digest must equal the string stored in
///      `adversarial_corpus_v1.sha256`.
///
/// This "blank-and-rehash" convention means the digest commits to every field
/// EXCEPT the digest placeholder itself, making the document self-consistent.
/// Compact (non-pretty) serialisation is chosen because `serde_json::to_string`
/// produces a **deterministic, alphabetical key order** for JSON objects.
/// Without the `preserve_order` feature (which this crate does NOT enable),
/// `serde_json` deserialises JSON objects into a `BTreeMap`, so keys are always
/// emitted in ascending lexicographic order regardless of the original document
/// order.
///
/// WARNING: Do NOT enable serde_json's `preserve_order` feature — it would
/// change the serialisation order, altering the canonical byte sequence and
/// invalidating every committed corpus digest.
///
/// ## Containment invariant (per sample)
///
/// For every sample in the corpus, `run_sandboxed` must:
///   - return within `timeout_secs + 3.0` wall-clock seconds (no hang),
///   - produce an exit status that is NOT `Clean(0)` (not allowed to "pass"),
///   - leave ZERO surviving python3 processes identifiable by the job UUID.
///
/// Where `expected_exit` is deterministic the test also asserts the specific
/// status:
///   - `"timeout"`  → `SandboxExitStatus::Timeout`, with
///                    `stats.cgroup_kill_event == true` and
///                    `stats.time_to_contain_secs <= timeout_secs + 1.0`
///   - `"oom"`      → `SandboxExitStatus::OomKilled`
///   - `"denied"`   → contained quickly without a cgroup kill: `Clean(nonzero)`
///                    or `SetupError(_)` — NOT `Timeout`.
///
/// ## Platform & assumptions
///
///   - All tests are Linux-only (`#[cfg(target_os = "linux")]`).
///   - `python3` is on PATH on wk-system.
///   - Tests MUST be run with `--test-threads=1` (adversarial code).
///   - cgroup base is the per-user delegated slice (derived from `getuid()`).
///   - `sha2` is NOT yet a dependency; SHA-256 is computed via a hand-rolled
///     implementation that forwards to the kernel via a /dev/stdin pipe — see
///     `sha256_bytes()` below.  The implementer may add `sha2` as a dev-dep
///     and replace this helper; the convention above is unchanged.
#[cfg(target_os = "linux")]
mod corpus_containment {
    use rlox_sandbox::worker::{
        run_sandboxed, SandboxConfig, SandboxExitStatus, SandboxInput,
    };
    use serde::Deserialize;
    use std::path::{Path, PathBuf};
    use std::time::Instant;
    use uuid::Uuid;

    // -----------------------------------------------------------------------
    // Corpus schema types
    // -----------------------------------------------------------------------

    #[derive(Debug, Deserialize)]
    struct Corpus {
        version: String,
        /// The stored digest (hex); we verify this ourselves in
        /// `test_corpus_sha256_matches`.
        sha256: String,
        samples: Vec<CorpusSample>,
    }

    #[derive(Debug, Deserialize, Clone)]
    struct CorpusSample {
        id: String,
        category: String,
        language: String,
        code: String,
        expected_exit: String,
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Absolute path to the corpus JSON file.
    ///
    /// CARGO_MANIFEST_DIR resolves to `crates/rlox-sandbox/`; we walk two
    /// levels up to the workspace root and then into the committed corpus dir.
    fn corpus_json_path() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest
            .join("../../benchmarks/agentic/corpus/adversarial_corpus_v1.json")
    }

    /// Absolute path to the companion `.sha256` file.
    fn corpus_sha256_path() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest
            .join("../../benchmarks/agentic/corpus/adversarial_corpus_v1.sha256")
    }

    /// Load and parse the corpus JSON, panicking with a descriptive message if
    /// the file does not exist (which is the expected RED failure reason).
    fn load_corpus() -> Corpus {
        let path = corpus_json_path();
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "corpus file not found — this is the expected RED failure \
                 (the file has not been committed yet): {}: {e}",
                path.display()
            )
        });
        serde_json::from_str::<Corpus>(&raw).unwrap_or_else(|e| {
            panic!(
                "corpus JSON failed to parse ({}): {e}",
                path.display()
            )
        })
    }

    /// Compute SHA-256 of `bytes` via the `sha256sum` command-line utility.
    ///
    /// This avoids adding a `sha2` crate dependency at the test stage; the
    /// implementer may replace it with a crate if they add `sha2` as a
    /// dev-dependency.  The hash itself must match exactly (same bytes in,
    /// same digest out), so the convention still holds.
    fn sha256_hex(bytes: &[u8]) -> String {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("sha256sum")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("sha256sum must be available on wk-system (Ubuntu 24.04)");

        child
            .stdin
            .take()
            .unwrap()
            .write_all(bytes)
            .expect("write to sha256sum stdin");

        let out = child.wait_with_output().expect("sha256sum must exit");
        assert!(out.status.success(), "sha256sum failed");

        // sha256sum output: "<hex>  -\n"
        let line = String::from_utf8(out.stdout).unwrap();
        line.split_whitespace()
            .next()
            .expect("sha256sum produced no output")
            .to_owned()
    }

    /// Compute the canonical digest for a corpus JSON file.
    ///
    /// Convention: parse the JSON, blank the `"sha256"` field to `""`,
    /// re-serialise with serde_json compact output (no pretty-printing),
    /// then SHA-256 the resulting UTF-8 bytes.
    fn canonical_corpus_digest(json_path: &Path) -> String {
        let raw = std::fs::read_to_string(json_path)
            .unwrap_or_else(|e| panic!("cannot read corpus file: {e}"));

        let mut root: serde_json::Value =
            serde_json::from_str(&raw).expect("corpus JSON must be valid");

        // Blank the stored digest before hashing so the document is
        // self-consistent (the digest cannot include itself).
        root["sha256"] = serde_json::Value::String(String::new());

        let canonical = serde_json::to_string(&root)
            .expect("re-serialisation must succeed");

        sha256_hex(canonical.as_bytes())
    }

    /// Build a SandboxConfig using the per-user delegated cgroup slice.
    /// Uses tight resource caps appropriate for adversarial samples.
    fn corpus_config(timeout_secs: f64) -> SandboxConfig {
        let uid = unsafe { libc::getuid() };
        SandboxConfig {
            timeout_secs,
            mem_limit_bytes: 256 * 1024 * 1024, // 256 MiB
            pids_limit: 32,
            cpu_weight: 100,
            cgroup_base: PathBuf::from(format!(
                "/sys/fs/cgroup/user.slice/user-{uid}.slice"
            )),
        }
    }

    /// Count surviving python3 processes whose cmdline contains `job_id`.
    /// Mirrors the approach used in `tests/adversarial_containment.rs`.
    fn count_python3_survivors(job_id: &Uuid) -> usize {
        let marker = job_id.to_string();
        let my_uid = unsafe { libc::getuid() };

        let Ok(proc_dir) = std::fs::read_dir("/proc") else {
            return 0;
        };

        let mut count = 0usize;
        for entry in proc_dir.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }

            // Skip processes not owned by us.
            let status_path = format!("/proc/{}/status", name_str);
            let Ok(status_text) = std::fs::read_to_string(&status_path) else {
                continue;
            };
            let is_ours = status_text.lines().any(|line| {
                if let Some(rest) = line.strip_prefix("Uid:") {
                    rest.split_whitespace()
                        .next()
                        .and_then(|s| s.parse::<u32>().ok())
                        .map(|uid| uid == my_uid)
                        .unwrap_or(false)
                } else {
                    false
                }
            });
            if !is_ours {
                continue;
            }

            let cmdline_path = format!("/proc/{}/cmdline", name_str);
            let Ok(cmdline_bytes) = std::fs::read(&cmdline_path) else {
                continue;
            };
            let cmdline = String::from_utf8_lossy(&cmdline_bytes);
            if cmdline.contains(&marker) {
                count += 1;
            }
        }
        count
    }

    // -----------------------------------------------------------------------
    // Test 1 — Corpus SHA-256 integrity (AC-2 / AC-3)
    //
    // Recomputes the SHA-256 of the corpus using the "blank-and-rehash"
    // convention described at the top of this file, then asserts it equals
    // the digest stored in `adversarial_corpus_v1.sha256`.
    //
    // EXPECTED RED FAILURE: the corpus file does not exist yet, so
    // `load_corpus()` panics with "corpus file not found".
    // -----------------------------------------------------------------------
    #[test]
    fn test_corpus_sha256_matches() {
        // This will panic with a clear message if the corpus file is absent.
        let corpus = load_corpus();

        let sha256_path = corpus_sha256_path();
        let committed_digest = std::fs::read_to_string(&sha256_path)
            .unwrap_or_else(|e| {
                panic!(
                    ".sha256 file not found ({}): {e}",
                    sha256_path.display()
                )
            })
            .trim()
            .to_owned();

        assert_eq!(
            corpus.version, "v1",
            "corpus version must be \"v1\", got {:?}",
            corpus.version
        );

        // The stored sha256 field in the JSON must equal the committed .sha256
        // file (belt-and-suspenders: both must agree).
        assert_eq!(
            corpus.sha256, committed_digest,
            "corpus JSON's sha256 field ({:?}) does not match the committed \
             .sha256 file ({:?}) — these must be kept in sync",
            corpus.sha256, committed_digest
        );

        // Now recompute from canonical bytes and assert integrity.
        let recomputed = canonical_corpus_digest(&corpus_json_path());
        assert_eq!(
            recomputed, committed_digest,
            "SHA-256 recomputed over canonical bytes ({recomputed}) does not \
             match the committed digest ({committed_digest}) — the corpus was \
             modified without updating the .sha256 file (AC-2/AC-3 integrity \
             violation)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 1b — SHA-256 convention stability guard.
    //
    // This test is INDEPENDENT of the corpus file.  It bakes in a tiny,
    // fixed JSON literal (keys intentionally given in non-alphabetical
    // document order) and a hardcoded expected digest, then runs the SAME
    // canonicalization + hash that the corpus convention uses and asserts the
    // result equals the constant.
    //
    // Purpose: catch any future change to serde_json's serialisation order
    // (e.g. someone enabling the `preserve_order` feature, or upgrading to a
    // hypothetical serde_json version that changes default ordering) at
    // `cargo test` time, rather than silently producing a wrong digest that
    // only surfaces when a committed sidecar mismatches.
    //
    // How the constant was computed (on wk-system, 2026-06-15):
    //
    //   INPUT JSON (document order): {"version":"v1","sha256":"","samples":[]}
    //
    //   After serde_json parse + blank-sha256 + compact re-serialise
    //   (BTreeMap alphabetical key order):
    //     {"samples":[],"sha256":"","version":"v1"}   ← 41 bytes
    //
    //   echo -n '{"samples":[],"sha256":"","version":"v1"}' | sha256sum
    //   → 7cf5a0341f5574eea2aae5bbd34392cf3df87d9f0e6b3db3e08d4a6cb2de037e
    //
    // This constant must never be updated except when deliberately migrating
    // the hashing convention (at which point ALL committed .sha256 sidecars
    // must also be regenerated).
    // -----------------------------------------------------------------------
    #[test]
    fn test_sha256_convention_is_stable() {
        // Tiny JSON with keys in NON-alphabetical document order.
        // serde_json (without preserve_order) must reorder them to BTreeMap
        // alphabetical: samples < sha256 < version.
        let raw = r#"{"version":"v1","sha256":"","samples":[]}"#;

        let mut root: serde_json::Value =
            serde_json::from_str(raw).expect("fixed JSON must parse");

        // Apply the blank-and-rehash convention (sha256 field is already blank
        // in this fixture, but we do it explicitly to mirror the real code path).
        root["sha256"] = serde_json::Value::String(String::new());

        let canonical = serde_json::to_string(&root)
            .expect("re-serialisation must succeed");

        // Assert the canonical form is exactly what we expect so the digest
        // constant is unambiguous.
        assert_eq!(
            canonical,
            r#"{"samples":[],"sha256":"","version":"v1"}"#,
            "serde_json key ordering changed — the canonicalization convention \
             is broken. Check whether preserve_order was enabled or serde_json \
             was upgraded with a serialisation behaviour change."
        );

        let digest = sha256_hex(canonical.as_bytes());

        // Hardcoded expected digest — computed once on wk-system (see comment
        // above). This constant MUST match the canonical bytes above.
        const EXPECTED_DIGEST: &str =
            "7cf5a0341f5574eea2aae5bbd34392cf3df87d9f0e6b3db3e08d4a6cb2de037e";

        assert_eq!(
            digest,
            EXPECTED_DIGEST,
            "SHA-256 convention stability check failed.\n\
             canonical bytes : {canonical:?}\n\
             computed digest : {digest}\n\
             expected digest : {EXPECTED_DIGEST}\n\
             This means the serialisation or hashing convention changed. \
             Do NOT silently update this constant — all committed .sha256 \
             sidecars must be regenerated if the convention changes."
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — Corpus covers all six AC-3 threat categories.
    //
    // Asserts that at least one sample exists for each of the six categories:
    //   infinite_loop | fork_bomb | memory_bomb | unkillable_thread |
    //   blocking_network | fd_exhaustion
    //
    // EXPECTED RED FAILURE: corpus file absent → panic in `load_corpus()`.
    // -----------------------------------------------------------------------
    #[test]
    fn test_corpus_covers_all_six_categories() {
        let corpus = load_corpus();

        let required_categories = [
            "infinite_loop",
            "fork_bomb",
            "memory_bomb",
            "unkillable_thread",
            "blocking_network",
            "fd_exhaustion",
        ];

        for category in &required_categories {
            let found = corpus
                .samples
                .iter()
                .any(|s| s.category == *category);
            assert!(
                found,
                "corpus is missing at least one sample with category {:?}; \
                 all six AC-3 categories must be represented",
                category
            );
        }

        // Sanity: every sample's category is one of the six.
        for sample in &corpus.samples {
            assert!(
                required_categories.contains(&sample.category.as_str()),
                "sample {:?} has unrecognised category {:?}; valid values: {:?}",
                sample.id,
                sample.category,
                required_categories
            );
        }

        // Sanity: every sample has language == "python" (MVP constraint).
        for sample in &corpus.samples {
            assert_eq!(
                sample.language, "python",
                "sample {:?} has language {:?}; only \"python\" is supported in MVP",
                sample.id, sample.language
            );
        }

        // Sanity: valid expected_exit values.
        for sample in &corpus.samples {
            assert!(
                matches!(
                    sample.expected_exit.as_str(),
                    "timeout" | "oom" | "denied"
                ),
                "sample {:?} has unknown expected_exit {:?}; valid: timeout, oom, denied",
                sample.id,
                sample.expected_exit
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — Every corpus sample is contained (the main gate).
    //
    // For each sample in the corpus:
    //   (a) `run_sandboxed` returns within `timeout_secs + 3.0` wall-clock
    //       seconds — no hang.
    //   (b) Exit status is NOT `Clean(0)` — adversarial code must not "pass".
    //   (c) ZERO surviving python3 processes remain after return.
    //   (d) Where `expected_exit` is deterministic, the specific status is
    //       asserted:
    //         "timeout" → Timeout + cgroup_kill_event=true + time_to_contain ≤ timeout+1.0
    //         "oom"     → OomKilled
    //         "denied"  → NOT Timeout (quick denial via seccomp/EPERM)
    //
    // `timeout_secs` = 3.0 s for all samples — short enough to keep the full
    // suite under a few minutes, yet long enough for the sandbox setup + kill.
    //
    // `mem_limit_bytes` = 256 MiB so memory bombs can build up quickly before
    // being OOM-killed, while staying well below wk-system capacity.
    //
    // EXPECTED RED FAILURE: corpus file absent → panic in `load_corpus()`.
    // -----------------------------------------------------------------------
    #[tokio::test(flavor = "multi_thread")]
    async fn test_every_corpus_sample_is_contained() {
        let corpus = load_corpus();

        assert!(
            !corpus.samples.is_empty(),
            "corpus must contain at least one sample"
        );

        const TIMEOUT_SECS: f64 = 3.0;
        // Wall-clock budget per sample: timeout + 3 s for cgroup freeze/kill
        // teardown plus waitpid reaping.
        const WALL_BUDGET_SECS: f64 = TIMEOUT_SECS + 3.0;

        for sample in &corpus.samples {
            let config = corpus_config(TIMEOUT_SECS);
            let job_id = Uuid::new_v4();
            let input = SandboxInput {
                job_id,
                code: sample.code.clone(),
                test_suite: String::new(),
                language: sample.language.clone(),
                is_adversarial: true,
            };

            // ── (a) No hang ──────────────────────────────────────────────────
            let wall_start = Instant::now();
            let output = tokio::time::timeout(
                tokio::time::Duration::from_secs_f64(WALL_BUDGET_SECS),
                run_sandboxed(input, &config),
            )
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "sample {:?} (category={:?}) DID NOT RETURN within {WALL_BUDGET_SECS:.1} s; \
                     run_sandboxed is hanging — the sandbox did not contain it",
                    sample.id, sample.category
                )
            })
            .unwrap_or_else(|e| {
                // Err from run_sandboxed is acceptable for setup failures, but
                // we still assert no survivors below, so convert to a synthetic
                // Clean(1) to let the survivor check run.
                // Actually: SetupError counts as contained, but we must not
                // panic here — re-check exit assertion logic below.
                //
                // For now treat Err as a contained outcome (setup blocked it).
                // The caller can inspect the error in the panic message.
                panic!(
                    "sample {:?} (category={:?}) returned Err from run_sandboxed: {e}; \
                     this is only acceptable if the corpus file is absent (RED). \
                     If the corpus file exists, the sandbox must return Ok.",
                    sample.id, sample.category
                )
            });
            let wall_elapsed = wall_start.elapsed().as_secs_f64();

            assert!(
                wall_elapsed <= WALL_BUDGET_SECS,
                "sample {:?} wall time {wall_elapsed:.2} s exceeded budget \
                 {WALL_BUDGET_SECS:.1} s",
                sample.id
            );

            // ── (b) Not a clean success ───────────────────────────────────────
            assert!(
                output.exit_status != SandboxExitStatus::Clean(0),
                "sample {:?} (category={:?}, expected_exit={:?}) exited Clean(0); \
                 adversarial code MUST NOT pass cleanly — the sandbox failed to \
                 contain it",
                sample.id, sample.category, sample.expected_exit
            );

            // ── (c) No survivors ─────────────────────────────────────────────
            // Brief settle window — same as in adversarial_containment.rs.
            std::thread::sleep(std::time::Duration::from_millis(300));
            let survivors = count_python3_survivors(&job_id);
            assert_eq!(
                survivors, 0,
                "sample {:?} (category={:?}) left {survivors} surviving python3 \
                 process(es) after run_sandboxed returned — the sandbox did not \
                 kill all descendants (cgroup kill may have failed)",
                sample.id, sample.category
            );

            // ── (d) Deterministic exit-status assertions ───────────────────
            match sample.expected_exit.as_str() {
                "timeout" => {
                    assert_eq!(
                        output.exit_status,
                        SandboxExitStatus::Timeout,
                        "sample {:?} expected Timeout, got {:?}",
                        sample.id, output.exit_status
                    );
                    assert!(
                        output.stats.cgroup_kill_event,
                        "sample {:?}: expected cgroup_kill_event=true for Timeout",
                        sample.id
                    );
                    assert!(
                        output.stats.time_to_contain_secs <= TIMEOUT_SECS + 1.0,
                        "sample {:?}: time_to_contain_secs {:.3} exceeds \
                         timeout_secs ({TIMEOUT_SECS}) + 1.0",
                        sample.id, output.stats.time_to_contain_secs
                    );
                }
                "oom" => {
                    assert_eq!(
                        output.exit_status,
                        SandboxExitStatus::OomKilled,
                        "sample {:?} expected OomKilled, got {:?}",
                        sample.id, output.exit_status
                    );
                }
                "denied" => {
                    // Must NOT be Timeout: a "denied" sample should fail
                    // quickly via seccomp EPERM or a kernel-level refusal,
                    // not by sitting at the timeout wall.
                    assert_ne!(
                        output.exit_status,
                        SandboxExitStatus::Timeout,
                        "sample {:?} expected quick denial (not Timeout), got \
                         Timeout — the syscall/resource was not denied fast enough; \
                         expected_exit=\"denied\" means the process should be stopped \
                         by a policy refusal (EPERM/SIGKILL from seccomp), not by \
                         timing out",
                        sample.id
                    );
                    // Must not be Clean(0) — already covered by assertion (b)
                    // above, but restate for clarity in the error message.
                    assert!(
                        !matches!(output.exit_status, SandboxExitStatus::Clean(0)),
                        "sample {:?}: denied sample must not exit Clean(0)",
                        sample.id
                    );
                    // Must be FAST: a seccomp/EPERM denial should exit well
                    // before the timeout wall.  We use TIMEOUT_SECS as a
                    // generous-but-meaningful upper bound (the process should
                    // be killed in milliseconds, not seconds).  This catches a
                    // miscategorised sample that actually runs to the timeout
                    // and then exits non-zero for an unrelated reason.
                    //
                    // We do NOT use a tight bound (e.g. 1 s) because sandbox
                    // setup overhead (cgroup creation, seccomp BPF load) can
                    // take hundreds of milliseconds on a loaded system.
                    assert!(
                        wall_elapsed <= TIMEOUT_SECS,
                        "sample {:?} (category={:?}, expected_exit=\"denied\") \
                         took {wall_elapsed:.2} s wall time, which exceeds \
                         TIMEOUT_SECS ({TIMEOUT_SECS:.1} s). A \"denied\" \
                         sample must exit quickly (seccomp EPERM / SIGKILL), \
                         not run to the timeout wall. Either the sample is \
                         miscategorised or the seccomp policy is not blocking \
                         the syscall.",
                        sample.id, sample.category
                    );
                }
                other => {
                    panic!(
                        "sample {:?} has unknown expected_exit {:?}; \
                         valid values: timeout, oom, denied",
                        sample.id, other
                    );
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
extern crate libc;
