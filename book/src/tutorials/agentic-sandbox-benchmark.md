# Agentic-RL Sandbox Benchmark

This tutorial walks through the **agentic-RL validation benchmark** — an end-to-end study measuring whether hard-isolated sandboxing prevents crashes from adversarial code injection without losing learning quality.

## Headline Result

| Runs survived | 0 % | 1 % | 5 % | 10 % |
|---|---|---|---|---|
| **Baseline** (in-process) | 3/3 | 3/3 | 2/3 | 1/3 |
| **Treatment** (rlox sandbox) | 3/3 | 3/3 | 3/3 | 3/3 |

At 5–10% adversarial code injection, Baseline crashes; Treatment survives every run. Quality parity: both reach reward 0.917.

## Quick Start

```bash
# Setup (one command, Linux only)
bash benchmarks/agentic/repro.sh --setup-only

# Start sandbox server
systemd-run --user --unit=rlox-verify -p TasksMax=4096 -p MemoryMax=12G \
  target/release/rlox-verify-server --port 8231 --timeout-secs 5

# Single trial
CUDA_VISIBLE_DEVICES=0 python benchmarks/agentic/trl_grpo_run.py \
  --backend rlox \
  --adversarial-fraction 0.05 \
  --seed 0 \
  --max-steps 30 \
  --rlox-server-url http://localhost:8231 \
  --output-dir /tmp/run

# Full 24-run sweep (2 hours)
python benchmarks/agentic/run_sweep_p3.py
python benchmarks/agentic/aggregate_sweep.py
python benchmarks/agentic/plot_sweep.py --sweep-dir benchmarks/agentic/sweep_out --out-dir figures
```

## What It Proves

- **P3 (Containment):** Untrusted code executes safely in isolation.
- **Quality Parity:** Learning dynamics unchanged (sandbox is a safety layer).
- **Reward Integrity:** Model code cannot forge reward via `sys.exit(0)` or monkeypatch.

## Data Flow

Code execution backend (Baseline vs Treatment) is the only difference:
- **Baseline:** in-process subprocess (no isolation)
- **Treatment:** rlox `/verify` sandbox (namespaces + seccomp + cgroup v2)

Everything else identical: model, task, adversarial injection.

## Honest Caveats

1. **Single GPU, single task** — P1 (throughput) deferred to multi-GPU.
2. **Sandbox isolation best-effort** — Filesystem/proc info-isolation under AppArmor; containment/kill-safety are sound.
3. **Task difficulty** — Modest coding task; harder tasks may show different quality behavior.

For full scope, see `benchmarks/agentic/README.md`.

---

**Full tutorial:** [Agentic-RL Sandbox Benchmark Tutorial](../../docs/tutorials/agentic-sandbox-benchmark.md)
