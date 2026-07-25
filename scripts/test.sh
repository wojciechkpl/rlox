#!/usr/bin/env bash
set -euo pipefail

# rlox — inner-loop test & benchmark runner
# Usage:
#   ./scripts/test.sh          Run rlox-core tests + rebuild the extension + pytest
#   ./scripts/test.sh --bench  Run tests + benchmarks, update README results
#
# This is the fast inner loop, NOT a CI-parity check: it covers rlox-core only
# (not the other crates), runs tests/python/ only (not tests/agentic/), and does
# no linting. Before pushing, run the real gate set:
#
#   bash scripts/check-ci-local.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$ROOT/.venv/bin/python"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

run_bench=false
for arg in "$@"; do
    case "$arg" in
        --bench) run_bench=true ;;
        --help|-h)
            echo "Usage: $0 [--bench]"
            echo "  --bench  Run benchmarks after tests and update README"
            exit 0
            ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

passed=0
failed=0

step() {
    echo -e "\n${BOLD}━━━ $1 ━━━${NC}"
}

ok() {
    echo -e "${GREEN}✓ $1${NC}"
    passed=$((passed + 1))
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    failed=$((failed + 1))
}

# ── 1. Rust tests ──────────────────────────────────────────────────────
step "Rust unit tests (rlox-core)"
if cargo test --package rlox-core --quiet 2>&1; then
    ok "rlox-core tests"
else
    fail "rlox-core tests"
fi

# ── 2. Build Python extension ─────────────────────────────────────────
step "Build Python extension (maturin develop --release)"
if maturin_out=$("$VENV_PYTHON" -m maturin develop --release --manifest-path "$ROOT/crates/rlox-python/Cargo.toml" 2>&1); then
    echo "$maturin_out" | tail -1
    ok "maturin build"
else
    echo "$maturin_out"
    fail "maturin build"
    echo -e "${RED}Cannot continue without a working Python extension.${NC}"
    exit 1
fi

# ── 3. Python tests ───────────────────────────────────────────────────
step "Python tests (pytest)"
if "$VENV_PYTHON" -m pytest "$ROOT/tests/python/" -v --tb=short 2>&1; then
    ok "Python tests"
else
    fail "Python tests"
fi

# ── 4. Benchmarks (optional) ──────────────────────────────────────────
if $run_bench; then
    step "Python benchmarks"
    RESULTS_DIR="$ROOT/benchmark_results"
    mkdir -p "$RESULTS_DIR"

    if "$VENV_PYTHON" "$ROOT/benchmarks/run_all.py" --output-dir "$RESULTS_DIR" 2>&1; then
        ok "Benchmarks completed"
    else
        fail "Benchmarks"
    fi

    # Update README with latest results
    step "Updating README benchmark results"
    if "$VENV_PYTHON" "$ROOT/scripts/update_readme.py" 2>&1; then
        ok "README updated"
    else
        fail "README update"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Summary ━━━${NC}"
echo -e "${GREEN}Passed: $passed${NC}"
if [ "$failed" -gt 0 ]; then
    echo -e "${RED}Failed: $failed${NC}"
    exit 1
else
    echo -e "${GREEN}All checks passed.${NC}"
fi
