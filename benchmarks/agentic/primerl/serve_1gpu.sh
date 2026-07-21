#!/usr/bin/env bash
# Launch the two servers the single-GPU (decoupled) prime-rl agentic smoke needs,
# on ONE RTX 5090 (GPU 0), each in the background. Idempotent-ish: run once and
# leave running, then drive runs with run_sweep_primerl.py / make_primerl_run_one.
#
#   1. rlox-verify-server (:8231) — the Treatment sandbox (/verify), inside a
#      user-delegated cgroup scope so it can freeze->kill adversarial subtrees.
#   2. prime-rl inference server (:8000) — external vLLM policy server, so the
#      trainer config can omit [inference] and run trainer+orchestrator on GPU 0.
#
# Env overrides: RLOX_DIR (default /home/wk/rlox), PRIME_RL_VENV
# (default /home/wk/prime-rl/.venv), RUN_DIR (default /home/wk/rlox_runs).
set -euo pipefail

RLOX_DIR="${RLOX_DIR:-/home/wk/rlox}"
PRIME_RL_VENV="${PRIME_RL_VENV:-/home/wk/prime-rl/.venv}"
RUN_DIR="${RUN_DIR:-/home/wk/rlox_runs}"
mkdir -p "$RUN_DIR"
cd "$RLOX_DIR"

# 1. Sandbox verify-server in a delegated scope (Treatment backend).
systemctl --user stop rlox-verify.service 2>/dev/null || true
systemctl --user reset-failed rlox-verify.service 2>/dev/null || true
systemd-run --user --unit=rlox-verify -p TasksMax=4096 -p MemoryMax=12G \
  "$RLOX_DIR/target/release/rlox-verify-server" --port 8231 --timeout-secs 5
echo "[serve] rlox-verify-server -> :8231 (unit rlox-verify.service)"

# 2. External prime-rl inference server on GPU 0.
INFER_LOG="$RUN_DIR/inference.log"
: > "$INFER_LOG"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline nohup "$PRIME_RL_VENV/bin/inference" \
  @ "$RLOX_DIR/benchmarks/agentic/primerl/infer_1gpu.toml" > "$INFER_LOG" 2>&1 &
echo "[serve] prime-rl inference -> :8000 (log $INFER_LOG, PID $!)"

# 3. Wait for both to be ready.
for i in $(seq 1 60); do
  if curl -sf -m 3 http://localhost:8000/health >/dev/null 2>&1; then
    echo "[serve] inference ready after $((i*5))s"; break
  fi
  sleep 5
done
curl -sf -m 5 -X POST http://localhost:8231/verify -H 'Content-Type: application/json' \
  -d '{"code":"print(1)","tests":"assert True","is_adversarial":false}' >/dev/null \
  && echo "[serve] verify-server ready" || { echo "[serve] verify-server NOT ready" >&2; exit 1; }

echo "[serve] both servers up. Now: python benchmarks/agentic/run_sweep_primerl.py"
