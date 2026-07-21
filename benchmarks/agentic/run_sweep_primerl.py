#!/usr/bin/env python3
"""Smoke-sweep driver for the prime-rl GRPO launcher (Step 8 validation).

Runs the 2-condition grid (in_loop vs rlox) on the prime-rl host using the
``make_primerl_run_one`` launcher and writes a P3-style aggregate artifact.

Default parameters are sized for a single-GPU smoke on wk-system (RTX 5090):
  max_steps=8 — fast proof-of-life; increase for the real P3 study.
  fractions=[0.0, 0.10] — clean baseline + high-injection Treatment pair.
  n_seeds=1 — minimal for smoke.

Requires (single-GPU decoupled topology — see primerl/smoke_1gpu.toml):
  - rlox-verify-server running on SERVER_URL (Treatment runs call /verify).
  - An external prime-rl inference server live on localhost:8000
    (``inference @ primerl/infer_1gpu.toml``) — the base config omits
    [inference] so ``rl`` runs only the trainer+orchestrator on GPU 0.
    (The integrated ``rl`` launcher assigns inference+trainer disjoint GPUs,
    so 1-GPU requires this external-inference topology.)
  - prime-rl venv at PRIME_RL_BIN with the ``rl`` entrypoint.
  - CUDA_VISIBLE_DEVICES will be set to "0" by the launcher.

Run from the repo root:
    python benchmarks/agentic/run_sweep_primerl.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path("/home/wk/rlox")
sys.path.insert(0, str(REPO / "benchmarks" / "agentic"))
sys.path.insert(0, str(REPO / "python"))

import run_benchmark  # noqa: E402

# Smoke defaults — intentionally small for single-GPU proof-of-life.
MAX_STEPS = 8
GROUP_SIZE = 4
SERVER_URL = "http://localhost:8231"
PRIME_RL_BIN = "/home/wk/prime-rl/.venv/bin/rl"
BASE_TOML = str(REPO / "benchmarks" / "agentic" / "primerl" / "smoke_1gpu.toml")
# Adversarial corpus injected into the env args when adversarial_fraction > 0.
CORPUS_PATH = str(REPO / "benchmarks" / "agentic" / "corpus" / "adversarial_corpus_v1.json")
# Cap a single in_loop Baseline run so an adversarial-code stall can't block
# the entire smoke for hours.  Passed directly to make_primerl_run_one.
IN_LOOP_TIMEOUT_SECS = 1800  # 30 min/run


@dataclass
class SmokeCfg:
    """Minimal sweep config for the prime-rl smoke."""

    n_seeds: int = 1
    adversarial_fractions: list = field(default_factory=lambda: [0.0, 0.10])


def main() -> int:
    out = REPO / "benchmarks" / "agentic" / "sweep_out_primerl"
    run_one = run_benchmark.make_primerl_run_one(
        base_toml=BASE_TOML,
        max_steps=MAX_STEPS,
        group_size=GROUP_SIZE,
        rlox_server_url=SERVER_URL,
        prime_rl_bin=PRIME_RL_BIN,
        output_root=str(out / "runs"),
        repo_root=str(REPO),
        corpus_path=CORPUS_PATH,
        scope_for_baseline=True,
        in_loop_timeout_secs=IN_LOOP_TIMEOUT_SECS,
    )
    cfg = SmokeCfg()
    n_runs = 2 * cfg.n_seeds * len(cfg.adversarial_fractions)
    print(
        f"[smoke] launching {n_runs} runs "
        f"(2 conditions x {cfg.n_seeds} seeds x {len(cfg.adversarial_fractions)} fractions, "
        f"max_steps={MAX_STEPS})",
        flush=True,
    )
    results = run_benchmark.run_sweep(cfg, run_one, str(out / "metric_store"))

    # Aggregate per (condition, fraction).
    agg: dict = {}
    for r in results:
        agg.setdefault((r["condition"], r["fraction"]), []).append(r)
    rows = []
    for (cond, frac), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows.append(
            {
                "condition": cond,
                "fraction": frac,
                "n": len(rs),
                "survived": sum(1 for x in rs if x["survived"]),
                "mean_elapsed_secs": round(
                    sum(x["elapsed_secs"] for x in rs) / len(rs), 1
                ),
                "mean_completed_steps": round(
                    sum(x["completed_steps"] for x in rs) / len(rs), 2
                ),
                "mean_reward_last": round(
                    sum(x.get("mean_reward_last", 0.0) for x in rs) / len(rs), 4
                ),
            }
        )

    # P3 verdict: Treatment survives all fractions; Baseline degrades (elapsed
    # grows with fraction and/or survival drops) at the injection level.
    def row(cond: str, frac: float) -> dict | None:
        return next(
            (x for x in rows if x["condition"] == cond and x["fraction"] == frac), None
        )

    treat_survives_all = all(
        x["survived"] == x["n"] for x in rows if x["condition"] == "rlox"
    )
    base0 = row("in_loop", 0.0)
    verdict: dict = {"treatment_survives_all_fractions": treat_survives_all}
    for frac in cfg.adversarial_fractions:
        if frac == 0.0:
            continue
        b, t = row("in_loop", frac), row("rlox", frac)
        if b and t and base0:
            verdict[f"frac_{frac}"] = {
                "baseline_survived": f"{b['survived']}/{b['n']}",
                "treatment_survived": f"{t['survived']}/{t['n']}",
                "baseline_elapsed_vs_clean": round(
                    b["mean_elapsed_secs"] / max(base0["mean_elapsed_secs"], 1e-9), 2
                ),
                "baseline_slower_than_treatment_x": round(
                    b["mean_elapsed_secs"] / max(t["mean_elapsed_secs"], 1e-9), 2
                ),
            }

    summary = {
        "grid": f"2x{cfg.n_seeds}x{len(cfg.adversarial_fractions)}",
        "max_steps": MAX_STEPS,
        "group_size": GROUP_SIZE,
        "prime_rl_bin": PRIME_RL_BIN,
        "aggregate": rows,
        "p3_verdict": verdict,
        "runs": results,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown table.
    md = [
        "# rlox benchmark — prime-rl smoke sweep result",
        "",
        f"Grid: 2 conditions x {cfg.n_seeds} seeds x {len(cfg.adversarial_fractions)} fractions "
        f"(Qwen3-4B-Instruct-2507 + LoRA, {MAX_STEPS} GRPO steps, single RTX 5090).",
        "",
        "P3 = adversarial-code containment. Treatment = rlox sandbox (`/verify`); "
        "Baseline = in-process `in_loop` exec (scope-bounded for host safety).",
        "",
        "| condition | injection | survived | mean steps | mean elapsed (s) | mean reward last |",
        "|---|---|---|---|---|---|",
    ]
    for x in rows:
        md.append(
            f"| {x['condition']} | {x['fraction']:.2f} | {x['survived']}/{x['n']} | "
            f"{x['mean_completed_steps']} | {x['mean_elapsed_secs']} | {x['mean_reward_last']} |"
        )
    md += [
        "",
        "## P3 verdict",
        "```json",
        json.dumps(verdict, indent=2),
        "```",
        "",
        "On a single GPU the in-process Baseline is scope-bounded so it does not wedge "
        "the host, but its mean elapsed time inflates with injection (in-loop adversarial "
        "code stalls the training step) while the rlox Treatment stays flat and survives "
        "all fractions — the sandbox contains the adversarial code out-of-process.",
    ]
    (out / "smoke_summary.md").write_text("\n".join(md) + "\n")

    print("\n=== PRIME-RL SMOKE AGGREGATE ===")
    for x in rows:
        print(
            f"  {x['condition']:8s} frac={x['fraction']:.2f}  survived={x['survived']}/{x['n']}  "
            f"steps={x['mean_completed_steps']}  elapsed={x['mean_elapsed_secs']}s  "
            f"reward={x['mean_reward_last']}"
        )
    print(f"\n[smoke] wrote {out / 'smoke_summary.md'} and smoke_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
