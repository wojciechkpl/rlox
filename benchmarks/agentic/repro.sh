#!/usr/bin/env bash
# repro.sh — one-command reproduction of the rlox agentic validation benchmark (AC-8).
#
# Brings the host from a bare repo checkout to a ready-to-run benchmark:
#   1. Rust toolchain (rustup + cargo)            [idempotent]
#   2. uv + the agentic Python venv with pinned deps
#   3. release build of the rlox-sandbox crate
#   4. adversarial-corpus SHA-256 integrity check
#   5. launch the pre-registered sweep
#
# Requires only: a Linux host with cgroup v2 user delegation (see the rlox-sandbox
# README) and this repo checked out. On wk-system this is reached over Tailscale SSH.
#
# The sweep itself MUST run inside a systemd-delegated cgroup scope so the sandbox
# children can self-migrate (and as a host-safety backstop) — this script launches
# run_benchmark.py under `systemd-run --user --scope`.
#
# Usage:
#   bash benchmarks/agentic/repro.sh            # full setup, then launch the sweep
#   bash benchmarks/agentic/repro.sh --setup-only   # stop after step 4 (no sweep)
#   bash benchmarks/agentic/repro.sh --dry-run       # setup + print the run grid, no GPU work
set -euo pipefail

# --- locate the repo root (this script lives in benchmarks/agentic/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VENV="${RLOX_VENV:-$REPO_ROOT/.venv}"
CONFIG="${RLOX_BENCH_CONFIG:-$REPO_ROOT/benchmarks/agentic/configs/benchmark_v1.yaml}"
CORPUS="$REPO_ROOT/benchmarks/agentic/corpus/adversarial_corpus_v1.json"
METRIC_STORE="${RLOX_METRIC_STORE:-$REPO_ROOT/benchmarks/agentic/results}"

MODE="run"
case "${1:-}" in
  --setup-only) MODE="setup-only" ;;
  --dry-run)    MODE="dry-run" ;;
  "")           MODE="run" ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

log() { printf '\n=== %s ===\n' "$*"; }

# --- 1. Rust toolchain ---------------------------------------------------------
log "1/5 Rust toolchain"
if ! command -v cargo >/dev/null 2>&1; then
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "installing rustup (stable, minimal)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
  . "$HOME/.cargo/env"
fi
cargo --version

# --- 2. uv + agentic venv ------------------------------------------------------
log "2/5 uv + Python venv"
if ! command -v uv >/dev/null 2>&1; then
  [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  . "$HOME/.local/bin/env"
fi
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
fi
# Pinned light deps for the harness/adapter (NOT the GPU stack — that lives in the
# prime-rl venv and is set up separately via `cd ../prime-rl && uv sync`).
uv pip install --python "$VENV/bin/python" \
  "verifiers" "httpx" "pyyaml" "datasets" "numpy" "pytest" "ruff"
"$VENV/bin/python" --version

# --- 3. build rlox-sandbox (release) ------------------------------------------
log "3/5 build rlox-sandbox (release)"
cargo build --release -p rlox-sandbox

# --- 4. corpus integrity (AC-3) -----------------------------------------------
log "4/5 adversarial-corpus SHA-256 integrity"
"$VENV/bin/python" - "$CORPUS" <<'PY'
import sys
sys.path.insert(0, "python/rlox/agentic")
import json, adversarial_corpus as ac
path = sys.argv[1]
data = json.load(open(path))
recomputed = ac.canonical_digest(data)
committed = data["sha256"]
if recomputed != committed:
    raise SystemExit(f"CORPUS INTEGRITY FAILURE: recomputed {recomputed} != committed {committed}")
# Also load through the validating loader (raises on tamper).
ac.AdversarialCorpus.load(path)
print(f"corpus OK: {len(data['samples'])} samples, digest {committed[:16]}...")
PY

if [ "$MODE" = "setup-only" ]; then
  log "setup-only: done (skipping sweep)"
  exit 0
fi

# --- 5. launch the pre-registered sweep ---------------------------------------
log "5/5 launch sweep"
SWEEP_ARGS=(--config "$CONFIG" --metric-store "$METRIC_STORE")
[ "$MODE" = "dry-run" ] && SWEEP_ARGS+=(--dry-run)

# The sweep runs untrusted code via the sandbox → run it inside a user-delegated,
# resource-capped systemd scope (cgroup self-migration + host-safety backstop).
systemd-run --user --scope --slice=rlox.slice \
  -p TasksMax="${RLOX_TASKS_MAX:-8192}" -p MemoryMax="${RLOX_MEM_MAX:-48G}" \
  -- "$VENV/bin/python" benchmarks/agentic/run_benchmark.py "${SWEEP_ARGS[@]}"
