#!/usr/bin/env bash
# Sync the local rlox working copy to wk-system and run a command there.
#
# Why this exists: the rlox-sandbox crate is Linux-only (namespaces, seccomp,
# cgroup v2) and can only build/test on wk-system, not the macOS working copy.
# We edit locally and build/test remotely. The SSH ControlMaster (see
# ~/.ssh/config) keeps repeated calls fast.
#
# Usage:
#   bash scripts/wk-sync-test.sh                       # sync only
#   bash scripts/wk-sync-test.sh 'cargo test -p rlox-sandbox'
#   bash scripts/wk-sync-test.sh 'cargo build -p rlox-core'
#
# Env overrides: WK_HOST (default wk-system), WK_DIR (default /home/wk/rlox).
set -euo pipefail

REMOTE_HOST="${WK_HOST:-wk-system}"
REMOTE_DIR="${WK_DIR:-/home/wk/rlox}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"

rsync -az --delete \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude '.venv/' \
  --exclude '**/__pycache__/' \
  --exclude '**/*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude 'dist/' --exclude 'dist-ci/' \
  --exclude 'node_modules/' \
  --exclude 'benchmark_results/' --exclude 'results/' \
  --exclude 'sessions/' --exclude 'videos/' \
  --exclude 'site/' --exclude 'book/' --exclude 'mkdocs-docs/' \
  --exclude 'benchmarks/agentic/sweep_out/' --exclude 'benchmarks/agentic/*_out/' \
  --exclude 'benchmarks/agentic/pilot_results.json' \
  "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

if [ "$#" -gt 0 ]; then
  # Make cargo (and uv, when present) available, then run the command in the repo.
  #
  # rlox-sandbox cgroup v2 requirement: cgroup.procs self-migration only works
  # when the process is already inside a user-delegated cgroup hierarchy.
  # On wk-system, interactive SSH sessions land in session-<N>.scope (root-owned),
  # which blocks cgroup self-migration.  Wrapping cargo test in systemd-run places
  # it inside rlox.slice (user-delegated with Delegate=yes), enabling the child
  # sandbox processes to move themselves into their leaf cgroups.
  #
  # Only cargo test commands need the systemd-run wrapper; cargo build/clippy do not.
  #
  # SAFETY BACKSTOP (TasksMax/MemoryMax on the scope): the rlox-sandbox tests run
  # adversarial code (fork bombs, memory bombs). If a sandbox ever fails to contain
  # one (e.g. a test mis-points cgroup_base at a non-cgroup dir), the escaped
  # processes are still children of this scope, so capping the scope bounds the
  # blast radius and prevents host-wide PID/RAM exhaustion. Tune via WK_TASKS_MAX /
  # WK_MEM_MAX. Legitimate compile+test stays well under these.
  REMOTE_CMD="$*"
  TASKS_MAX="${WK_TASKS_MAX:-4096}"
  MEM_MAX="${WK_MEM_MAX:-24G}"
  if echo "$REMOTE_CMD" | grep -q 'cargo test'; then
    # Two capability gates (see crates/rlox-sandbox/tests/common/mod.rs), both
    # satisfied only inside this rlox.slice scope, so both are exported only here.
    # Elsewhere (CI shared runners, plain SSH) they are unset and the affected
    # tests skip instead of asserting against a vacuous SetupError.
    #
    #   RLOX_SANDBOX_CGROUP_TESTS=1      — every test that calls run_sandboxed and
    #     asserts on the exit status; needs the cgroup v2 self-migration this
    #     delegated scope provides.
    #   RLOX_SANDBOX_ADVERSARIAL_TESTS=1 — additionally detonates real resource
    #     bombs (fork/memory/pids); needs the TasksMax/MemoryMax backstop above.
    # shellcheck disable=SC2029
    ssh "$REMOTE_HOST" "systemd-run --user --scope --slice=rlox.slice -p TasksMax=$TASKS_MAX -p MemoryMax=$MEM_MAX --expand-environment=no -- bash -c 'cd \"$REMOTE_DIR\" && . \$HOME/.cargo/env && export RLOX_SANDBOX_CGROUP_TESTS=1 RLOX_SANDBOX_ADVERSARIAL_TESTS=1 && $REMOTE_CMD'"
  else
    # shellcheck disable=SC2029
    ssh "$REMOTE_HOST" "cd '$REMOTE_DIR' && { [ -f \$HOME/.cargo/env ] && . \$HOME/.cargo/env; }; { [ -f \$HOME/.local/bin/env ] && . \$HOME/.local/bin/env; }; $REMOTE_CMD"
  fi
fi
