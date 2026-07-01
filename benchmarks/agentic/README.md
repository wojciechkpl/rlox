# rlox agentic-RL validation benchmark

**Does decoupled, hard-isolated sandboxing of untrusted agent code prevent the crashes
and stalls that in-process execution suffers under adversarial injection — at no cost to
learning?** This benchmark answers that on a real GRPO post-training run.

It runs GRPO on a small coder model (Qwen3-4B-Instruct-2507 + LoRA) over a unit-test-verified
coding task, and flips one config key between two rollout/verification backends:

| | Baseline (`in_loop`) | Treatment (`rlox`) |
|---|---|---|
| code execution | in-process subprocess (no isolation) | decoupled `rlox-sandbox` via `POST /verify` (namespaces + seccomp + cgroup v2 freeze→kill) |

A fixed, versioned **adversarial corpus** (infinite loop, fork bomb, memory bomb, unkillable
thread, blocking network, fd exhaustion) is injected at a controlled fraction (0/1/5/10 %),
identically in both conditions. We measure **run survival**, wall-clock / step-time, GPU
utilization, and the **final reward / learning curve**.

## Headline result (30-step sweep, 3 seeds, single RTX 5090)

| runs survived | 0 % | 1 % | 5 % | 10 % |
|---|---|---|---|---|
| **Baseline** (in-loop) | 3/3 | 3/3 | **2/3** | **1/3** |
| **Treatment** (rlox)   | 3/3 | 3/3 | **3/3** | **3/3** |

- **P3 (containment):** at 5–10 % injection the Baseline crashes/stalls (1–2 of 3 runs fail;
  completed runs 3.0× slower at 5 %); rlox completes **every** run with zero contagion.
- **Guardrail (quality parity):** Baseline and Treatment reach identical final reward
  (0.917 = 0.917) — the sandbox changes *safety*, not *learning dynamics*.
- **P1 (throughput/util):** not measurable on a single GPU (single-process trainer has no
  rollout/training decoupling to exploit); deferred to a multi-GPU/async setup. P1 is prior
  art; rlox's novel contribution is **P3 + reward integrity**.

See `p3_sweep_summary.md` for the committed result; figures + the full report live in
`rlox-priv/reports/`.

## Components

| file | role |
|---|---|
| `../../crates/rlox-sandbox/` | the Rust hard-isolation sandbox + `/verify` server (`rlox-verify-server` bin) |
| `corpus/adversarial_corpus_v1.json` | the fixed, SHA-256-checked adversarial corpus |
| `../../environments/rlox_verify/` | the `verifiers` environment (coding task; Baseline/Treatment dispatch + injection) |
| `trl_grpo_run.py` | single-GPU TRL GRPOTrainer runner (`--backend {in_loop,rlox}` `--adversarial-fraction` …) |
| `run_benchmark.py` | the sweep driver (`run_sweep` / `make_trl_run_one`) |
| `run_sweep_p3.py` | launches the 2×seeds×fractions grid |
| `aggregate_sweep.py` | aggregates per-run summaries → `p3_summary.{md,json}` (survival from the metric store, incl. DNF) |
| `plot_sweep.py` | renders the figures (survival, step-time stall, guardrail, GPU-util) |
| `oq3_pilot.py` | the OQ-3 pilot (Baseline degrades on every adversarial category) |

## Running it

Linux + an NVIDIA GPU + cgroup v2 user delegation. The whole pipeline runs on the host
`wk-system` over Tailscale SSH; edit locally and sync/build/test there with
`scripts/wk-sync-test.sh`.

```bash
# 1. one-command environment + build + corpus check (see AC-8)
bash benchmarks/agentic/repro.sh --setup-only

# 2. start the Treatment /verify server inside a delegated cgroup scope
systemd-run --user --unit=rlox-verify -p TasksMax=4096 -p MemoryMax=12G \
  target/release/rlox-verify-server --port 8231 --timeout-secs 5

# 3. a single run (Baseline vs Treatment is one flag)
CUDA_VISIBLE_DEVICES=0 python benchmarks/agentic/trl_grpo_run.py \
  --backend rlox --adversarial-fraction 0.10 --seed 0 --max-steps 30 \
  --rlox-server-url http://localhost:8231 --output-dir /tmp/run

# 4. the full sweep, then aggregate + plot
python benchmarks/agentic/run_sweep_p3.py
python benchmarks/agentic/aggregate_sweep.py
python benchmarks/agentic/plot_sweep.py --sweep-dir benchmarks/agentic/sweep_out --out-dir figures
```

The GRPO trainer above uses TRL (single-GPU colocation) for the **canonical** P3 numbers.

### Running on the prime-rl host (single-GPU, decoupled)

`make_primerl_run_one` (`run_benchmark.py`) drives the benchmark's *locked* host, prime-rl +
`verifiers`, end-to-end. prime-rl's integrated `rl` launcher assigns inference and the trainer
**disjoint** GPUs (floor = 2 GPUs), so single-GPU uses a **decoupled** topology: an external vLLM
server on GPU 0 plus a trainer config that omits `[inference]` (runs trainer+orchestrator on the
same GPU, orchestrator pointed at the external server via an elastic client).

```bash
# 1. both servers on GPU 0 (verify-server :8231 + prime-rl inference :8000)
bash benchmarks/agentic/primerl/serve_1gpu.sh

# 2. one launcher run, or the smoke sweep (base_toml = primerl/smoke_1gpu.toml)
python benchmarks/agentic/run_benchmark.py --host primerl \
  --config benchmarks/agentic/configs/benchmark_v1.yaml --metric-store /tmp/ms
python benchmarks/agentic/run_sweep_primerl.py
```

Validated on wk-system (1× RTX 5090): colocation fits (~13.8 GiB inference + trainer, no OOM) and
the seam connects (prime-rl → `rlox_verify` → sandbox `/verify`, zero dispatch errors). A learning
signal / the quality-parity guardrail on prime-rl needs task calibration under prime-rl's renderer
(see the "reward 0" caveat below); the multi-GPU P1 sweep is a GCP follow-up. Full write-up in
`docs/plans/step8-primerl-launcher.md`.

## Scope & caveats (read before citing)

- **Single GPU, single task/model/trainer.** P1 (decoupling throughput uplift) is deferred to
  multi-GPU. The task is intentionally modest; a calibrated harder task is future work (an MBPP
  re-run was too hard — reward 0 — and is *not* the canonical result).
- **Sandbox isolation.** Containment / kill-safety (cgroup freeze→`cgroup.kill`, clone3 block,
  `CLONE_NEWUSER` masking, netns, mem/pid caps) and reward integrity (nonce-authenticated trusted
  runner — model code cannot forge reward via `sys.exit`/`os._exit`/monkeypatch) are sound.
  Filesystem/`/proc` info-isolation is best-effort under Ubuntu's AppArmor `unprivileged_userns`
  profile (which blocks `mount()` in the userns) and is enforced at the Python-runtime level — a
  raw-syscall adversary could bypass it; a kernel-level guarantee needs the AppArmor profile
  relaxed. Regression-tested in `crates/rlox-sandbox/tests/security_isolation.rs`.
