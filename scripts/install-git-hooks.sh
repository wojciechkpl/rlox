#!/usr/bin/env bash
# Install a pre-push hook that runs the fast CI gates before anything leaves the
# machine. Idempotent — safe to re-run.
#
#   bash scripts/install-git-hooks.sh          # install
#   bash scripts/install-git-hooks.sh --remove # uninstall
#
# The hook runs the cheap, high-yield gates (rustfmt, both clippy passes, pytest
# collection) — the ones that produced every red CI run so far. It deliberately
# does NOT run the full test suites: a pre-push hook that takes minutes gets
# bypassed with --no-verify, which defeats the purpose. Run
# `bash scripts/check-ci-local.sh` for the complete set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_DIR="$(git -C "$REPO_ROOT" rev-parse --git-path hooks)"
HOOK="$HOOK_DIR/pre-push"

if [ "${1:-}" = "--remove" ]; then
  rm -f "$HOOK"
  echo "Removed $HOOK"
  exit 0
fi

mkdir -p "$HOOK_DIR"

cat > "$HOOK" <<'HOOK_BODY'
#!/usr/bin/env bash
# Managed by scripts/install-git-hooks.sh — re-run that script to update.
# Skip with: git push --no-verify  (or PRE_PUSH_SKIP=1 git push)
set -uo pipefail

if [ "${PRE_PUSH_SKIP:-0}" = 1 ]; then
  echo "pre-push: skipped via PRE_PUSH_SKIP=1"
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Matches .github/workflows/ci.yml (and .cargo/config.toml's target-cpu=native
# would otherwise cause SIGILL under cached proc-macro dylibs).
export CARGO_BUILD_RUSTFLAGS=""

# pyo3 0.23 supports Python <= 3.13; a newer system python3 hard-fails the
# pyo3-ffi build script. The repo venv is on a supported version.
if [ -z "${PYO3_PYTHON:-}" ] && [ -x .venv/bin/python ]; then
  export PYO3_PYTHON="$PWD/.venv/bin/python"
fi

CLIPPY_ALLOW=(
  -A clippy::too_many_arguments
  -A clippy::type_complexity
  -A clippy::empty_line_after_doc_comments
  -A clippy::manual_range_contains
  -A clippy::repeat_vec_with_capacity
)
LINUX_TARGET="x86_64-unknown-linux-gnu"
FAILED=0

step() {
  local name="$1"; shift
  printf '\033[1;34mpre-push: %s\033[0m\n' "$name"
  "$@" || { printf '\033[0;31mpre-push: FAILED — %s\033[0m\n' "$name"; FAILED=1; }
}

step "rustfmt" cargo fmt --all -- --check

step "clippy (host, workspace minus rlox-sandbox)" \
  cargo clippy --workspace --exclude rlox-sandbox --all-targets \
  -- -D warnings "${CLIPPY_ALLOW[@]}"

# rlox-sandbox is Linux-only; its lints are invisible on macOS without this.
if rustup target list --installed 2>/dev/null | grep -qx "$LINUX_TARGET"; then
  step "clippy (rlox-sandbox, $LINUX_TARGET)" \
    cargo clippy -p rlox-sandbox --all-targets --target "$LINUX_TARGET" \
    -- -D warnings "${CLIPPY_ALLOW[@]}"
else
  printf '\033[0;31mpre-push: FAILED — Linux target missing; CI lints rlox-sandbox and you cannot.\033[0m\n'
  printf '  rustup target add %s\n' "$LINUX_TARGET"
  FAILED=1
fi

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
# A single bad import aborts the whole pytest run in CI.
step "pytest collection" "$PY" -m pytest tests/ -q --collect-only

if [ "$FAILED" -ne 0 ]; then
  printf '\n\033[0;31mpre-push blocked: the above would fail CI.\033[0m\n'
  printf 'Full local run: bash scripts/check-ci-local.sh\n'
  printf 'Override:       git push --no-verify\n'
  exit 1
fi

printf '\033[0;32mpre-push: fast CI gates passed.\033[0m\n'
HOOK_BODY

chmod +x "$HOOK"
echo "Installed $HOOK"
echo "Bypass once with: git push --no-verify"
