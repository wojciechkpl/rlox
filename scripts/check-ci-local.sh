#!/usr/bin/env bash
# Reproduce every CI gate locally, in CI's own order, before pushing.
#
# Why this exists: CI kept going red on things a macOS working copy silently
# cannot see.
#
#   1. rlox-sandbox is Linux-only (`#[cfg(target_os = "linux")]` + seccompiler),
#      so `cargo clippy --workspace` on macOS does not even compile it — every
#      lint inside it was invisible until CI ran. This script lints that crate
#      against a Linux target, where those cfg blocks are live.
#   2. pytest aborts the whole run on a single collection error (a bad import in
#      one module takes down 2000 unrelated tests). Collection is checked first
#      and separately because it is a ~2 s check that catches a 4-minute failure.
#   3. Toolchain drift: see rust-toolchain.toml.
#
# Usage:
#   bash scripts/check-ci-local.sh            # all gates
#   bash scripts/check-ci-local.sh rust       # rust gates only
#   bash scripts/check-ci-local.sh python     # python gates only
#
# Two opt-in gates are off by default because they are slow:
#   SLOW=1  the convergence tests. CI runs these on pushes to main ONLY, so a
#           regression here cannot be caught by a PR — it turns main red after
#           merge. Use before merging anything touching a training path or a
#           convergence threshold. (~20-40 min)
#   WK=1    the full Linux sandbox suite, on wk-system (see wk-sync-test.sh);
#           those tests need cgroup v2 delegation this host lacks.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCOPE="${1:-all}"
LINUX_TARGET="x86_64-unknown-linux-gnu"
# Keep this list identical to the -A flags in .github/workflows/ci.yml.
CLIPPY_ALLOW=(
  -A clippy::too_many_arguments
  -A clippy::type_complexity
  -A clippy::empty_line_after_doc_comments
  -A clippy::manual_range_contains
  -A clippy::repeat_vec_with_capacity
)

FAILED=()
PASSED=()

# Override .cargo/config.toml's target-cpu=native, matching ci.yml.
# Must be CARGO_ENCODED_RUSTFLAGS: cargo treats an empty
# CARGO_BUILD_RUSTFLAGS as unset and falls back to the config file, so that
# spelling is a silent no-op (verified).
export CARGO_ENCODED_RUSTFLAGS=""

# pyo3 0.23 supports Python <= 3.13, but a Homebrew `python3` may already be
# newer (3.14), which makes the pyo3-ffi build script hard-fail and takes the
# whole workspace lint down with it. Prefer the repo venv, which is on a
# supported version. CI pins 3.10-3.13 so it never hits this.
if [ -z "${PYO3_PYTHON:-}" ] && [ -x .venv/bin/python ]; then
  export PYO3_PYTHON="$PWD/.venv/bin/python"
fi

run_gate() {
  local name="$1"; shift
  printf '\n\033[1;34m==> %s\033[0m\n' "$name"
  if "$@"; then
    PASSED+=("$name")
    printf '\033[0;32m    PASS: %s\033[0m\n' "$name"
  else
    FAILED+=("$name")
    printf '\033[0;31m    FAIL: %s\033[0m\n' "$name"
  fi
}

python_bin() {
  if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi
}

# --- Rust gates -------------------------------------------------------------
if [ "$SCOPE" = all ] || [ "$SCOPE" = rust ]; then
  run_gate "rustfmt" cargo fmt --all -- --check

  # Host clippy cannot build rlox-sandbox (Linux-only), so exclude it here and
  # cover it via the Linux-target gate below.
  run_gate "clippy (host, workspace minus rlox-sandbox)" \
    cargo clippy --workspace --exclude rlox-sandbox --all-targets \
    -- -D warnings "${CLIPPY_ALLOW[@]}"

  if rustup target list --installed 2>/dev/null | grep -qx "$LINUX_TARGET"; then
    # This is the gate that catches lints inside `cfg(target_os = "linux")`.
    run_gate "clippy (rlox-sandbox, $LINUX_TARGET)" \
      cargo clippy -p rlox-sandbox --all-targets --target "$LINUX_TARGET" \
      -- -D warnings "${CLIPPY_ALLOW[@]}"
  else
    printf '\n\033[0;31m==> SKIPPED clippy (rlox-sandbox, %s): target not installed.\033[0m\n' "$LINUX_TARGET"
    printf '    Install it — CI lints this code and you cannot see those lints without it:\n'
    printf '      rustup target add %s\n' "$LINUX_TARGET"
    FAILED+=("clippy (rlox-sandbox) — Linux target missing")
  fi

  # --no-fail-fast so one broken test binary does not mask the others.
  run_gate "cargo test (host, workspace minus rlox-sandbox)" \
    cargo test --workspace --exclude rlox-sandbox --no-fail-fast
fi

# --- Python gates -----------------------------------------------------------
if [ "$SCOPE" = all ] || [ "$SCOPE" = python ]; then
  PY="$(python_bin)"

  # Fast gate first: a single bad import aborts the entire pytest run.
  run_gate "pytest collection (all modules importable)" \
    "$PY" -m pytest tests/ -q --collect-only

  run_gate "ruff" "$PY" -m ruff check python/rlox/ --select E,F,W --ignore E501

  run_gate "pytest (not slow)" \
    "$PY" -m pytest tests/ -q --tb=short -m "not slow" --timeout=120 --timeout-method=thread

  # CI runs a 3.10-3.13 matrix; the venv is one version, so version-gated code
  # (e.g. a 3.11+ stdlib import without its 3.10 backport) fails only on CI.
  # tests/agentic/ is the stdlib-only subset where such imports live, so it runs
  # on 3.10 in a throwaway uv venv without needing torch or the built extension.
  # tests/test_repo_hygiene.py additionally catches this statically on any version.
  if command -v uv >/dev/null 2>&1; then
    PY310_VENV="${TMPDIR:-/tmp}/rlox-check-venv310"
    # --clear: build the venv fresh every run. Reusing it via --allow-existing
    # rewrites the venv while leaving stale site-packages metadata behind, which
    # made `uv pip install` fail on a half-installed package. uv's package cache
    # makes a clean rebuild ~1 s cold and effectively free warm, so there is
    # nothing to gain from reuse and a whole class of stale-state bugs to avoid.
    # Provisioning errors are reported, not swallowed: a gate that quietly
    # disappears is the failure mode this script exists to prevent.
    if provision=$(uv venv --python 3.10 --clear "$PY310_VENV" 2>&1 &&
         VIRTUAL_ENV="$PY310_VENV" uv pip install -q \
           pytest pytest-timeout pyyaml tomli tomli-w numpy 2>&1); then
      run_gate "pytest tests/agentic on Python 3.10 (oldest supported)" \
        "$PY310_VENV/bin/python" -m pytest tests/agentic/ -q --tb=short -m "not slow"
    else
      printf '\n\033[0;31m==> FAIL: could not provision the 3.10 venv:\033[0m\n%s\n' "$provision"
      FAILED+=("3.10 gate — venv provisioning failed")
    fi
  else
    printf '\n\033[1;33m==> SKIPPED 3.10 gate: uv not found; install uv, or rely on CI'"'"'s 3.10-3.13 matrix.\033[0m\n'
  fi
fi

# --- Slow convergence tests (opt-in) ---------------------------------------
# CI runs these only on pushes to main, so a slow-test regression cannot be caught
# by a PR — it lands on main and turns it red after merge. That is exactly how
# TQC's convergence test broke main: it had never run on a PR. ~20-40 min, hence
# opt-in rather than default. Run before merging anything that touches an
# algorithm's training path or a convergence threshold.
if [ "${SLOW:-0}" = 1 ] && { [ "$SCOPE" = all ] || [ "$SCOPE" = python ]; }; then
  PY="$(python_bin)"
  run_gate "pytest slow convergence tests (CI runs these on main only)" \
    "$PY" -m pytest tests/ -q --tb=short -m "slow" --timeout=600 --timeout-method=thread
fi

# --- Linux sandbox suite (opt-in) ------------------------------------------
if [ "${WK:-0}" = 1 ]; then
  run_gate "rlox-sandbox suite on wk-system (cgroup-delegated)" \
    bash scripts/wk-sync-test.sh 'cargo test -p rlox-sandbox --no-fail-fast'
fi

# --- Summary ----------------------------------------------------------------
printf '\n\033[1m──────── summary ────────\033[0m\n'
for g in "${PASSED[@]}"; do printf '\033[0;32m  PASS\033[0m  %s\n' "$g"; done
if [ ${#FAILED[@]} -gt 0 ]; then
  for g in "${FAILED[@]}"; do printf '\033[0;31m  FAIL\033[0m  %s\n' "$g"; done
  printf '\n\033[0;31m%d gate(s) failed — CI would fail too.\033[0m\n' "${#FAILED[@]}"
  exit 1
fi
printf '\n\033[0;32mAll local CI gates passed.\033[0m\n'
if [ "${WK:-0}" != 1 ]; then
  printf 'Note: the Linux sandbox tests were not run. Use WK=1 to include them.\n'
fi
