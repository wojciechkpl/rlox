# rlox benchmark — P3 sweep result (30-step, final)

Pre-registered grid: 2 conditions × 3 seeds × 4 injection fractions = 24 runs.
GRPO post-training of Qwen3-4B-Instruct-2507 + LoRA, 30 steps each, single RTX 5090.
Baseline = in-process (`in_loop`) code execution; Treatment = rlox sandbox (`/verify`,
out-of-process, hard-isolated). Same adversarial injection in both.

## P3 — run survival vs injection (authoritative, includes DNF/timeout failures)

| runs survived | 0% | 1% | 5% | 10% |
|---|---|---|---|---|
| **Baseline** (in-loop) | 3/3 | 3/3 | **2/3** | **1/3** |
| **Treatment** (rlox)   | 3/3 | 3/3 | **3/3** | **3/3** |

Meets the pre-registered P3 criterion: at 5–10% injection the Baseline shows ≥1
crash/stall (DNF: stalled past the 30-min per-run cap and was killed — in-loop fork
bombs starve the trainer) while rlox completes **all** runs. Completed Baseline runs at
5% take 542 s vs 181 s clean (3.0×); Treatment stays bounded (~310–432 s — it pays the
per-sample containment latency but never stalls/accumulates, because the sandbox
`cgroup.kill`s the whole subtree).

## Guardrail — quality parity (fraction 0%)

PASSED. Baseline final reward **0.917** = Treatment **0.917** (abs diff **0.0**); the
model genuinely learns (~0.9 unit-test pass rate). The sandbox changes safety, not
learning dynamics.

## P1 — throughput/utilization

Not measured on a single GPU (single-process TRL trainer has no rollout/training
decoupling to exploit); deferred to a multi-GPU/async setup. P1 is established prior
art (ProRL-Agent, SkyRL); rlox's novel contribution is P3.

## Notes
- `aggregate_sweep.py` reads per-run `summary.json`, which the 3 DNF Baseline runs never
  wrote; survival above is from the `metric_store` records (24 grid points). `plot_sweep.py`
  uses the metric store for the survival figure.
- Figures: `fig_survival`, `fig_step_time_stall`, `fig_p3_degradation`, `fig_guardrail`
  (rendered by `plot_sweep.py`; report + figures in `rlox-priv/reports/`).
- Scale-dependence: the 5% Baseline slowdown grew 1.13× (6 steps) → 3.0× (30 steps); P3
  effect scales with cumulative adversarial exposure.
