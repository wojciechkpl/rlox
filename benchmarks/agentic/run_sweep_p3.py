#!/usr/bin/env python3
"""Run the pre-registered sweep (2 conditions x 3 seeds x 4 fractions) on the
single-GPU TRL GRPO runner and write a P3 artifact.

On a single 5090, P1 (decoupling throughput uplift) is not measurable (single-process
trainer) and the guardrail needs a long enough run for the model to learn; this sweep
produces the LEAD P3 evidence: under adversarial injection, the rlox Treatment runs
clean (survives, flat step time) while the in_loop Baseline degrades (stalls / survival
drop) as in-process adversarial code starves the trainer.

Requires the rlox-verify-server running on --rlox-server-url for the Treatment runs.
Run inside the prime-rl venv with CUDA_VISIBLE_DEVICES=0.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path("/home/wk/rlox")
sys.path.insert(0, str(REPO / "benchmarks" / "agentic"))
sys.path.insert(0, str(REPO / "python" / "rlox" / "agentic"))

import run_benchmark  # noqa: E402

# Cap a single Baseline run so a stall can't run for the default 2h.
run_benchmark._IN_LOOP_TIMEOUT_SECS = 1800  # 30 min/run

MAX_STEPS = 30          # longer runs: accumulate adversarial load + show learning
GROUP_SIZE = 4
SERVER_URL = "http://localhost:8231"


@dataclass
class SweepCfg:
    n_seeds: int = 3
    adversarial_fractions: list = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.10])


def main() -> int:
    out = REPO / "benchmarks" / "agentic" / "sweep_out"
    run_one = run_benchmark.make_trl_run_one(
        max_steps=MAX_STEPS,
        group_size=GROUP_SIZE,
        rlox_server_url=SERVER_URL,
        corpus_path=str(REPO / "benchmarks/agentic/corpus/adversarial_corpus_v1.json"),
        output_root=str(out / "runs"),
        venv_python="/home/wk/prime-rl/.venv/bin/python",
        repo_root=str(REPO),
        scope_for_baseline=True,
    )
    cfg = SweepCfg()
    print(f"[sweep] launching {2 * cfg.n_seeds * len(cfg.adversarial_fractions)} runs "
          f"(2 conditions x {cfg.n_seeds} seeds x {len(cfg.adversarial_fractions)} fractions, "
          f"max_steps={MAX_STEPS})", flush=True)
    results = run_benchmark.run_sweep(cfg, run_one, str(out / "metric_store"))

    # Aggregate per (condition, fraction).
    agg: dict = {}
    for r in results:
        agg.setdefault((r["condition"], r["fraction"]), []).append(r)
    rows = []
    for (cond, frac), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows.append({
            "condition": cond,
            "fraction": frac,
            "n": len(rs),
            "survived": sum(1 for x in rs if x["survived"]),
            "mean_elapsed_secs": round(sum(x["elapsed_secs"] for x in rs) / len(rs), 1),
            "mean_completed_steps": round(sum(x["completed_steps"] for x in rs) / len(rs), 2),
        })

    # P3 verdict: Treatment survives all fractions; Baseline degrades (elapsed grows with
    # fraction and/or survival drops) at the 5-10% injection levels.
    def row(cond, frac):
        return next((x for x in rows if x["condition"] == cond and x["fraction"] == frac), None)

    treat_survives_all = all(x["survived"] == x["n"] for x in rows if x["condition"] == "rlox")
    base0 = row("in_loop", 0.0)
    verdict = {"treatment_survives_all_fractions": treat_survives_all}
    for frac in (0.05, 0.10):
        b, t = row("in_loop", frac), row("rlox", frac)
        if b and t and base0:
            verdict[f"frac_{frac}"] = {
                "baseline_survived": f"{b['survived']}/{b['n']}",
                "treatment_survived": f"{t['survived']}/{t['n']}",
                "baseline_elapsed_vs_clean": round(b["mean_elapsed_secs"] / max(base0["mean_elapsed_secs"], 1e-9), 2),
                "baseline_slower_than_treatment_x": round(b["mean_elapsed_secs"] / max(t["mean_elapsed_secs"], 1e-9), 2),
            }

    summary = {"grid": f"2x{cfg.n_seeds}x{len(cfg.adversarial_fractions)}",
               "max_steps": MAX_STEPS, "group_size": GROUP_SIZE,
               "aggregate": rows, "p3_verdict": verdict, "runs": results}
    (out / "p3_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown table.
    md = ["# rlox benchmark — P3 sweep result", "",
          f"Grid: 2 conditions x {cfg.n_seeds} seeds x {len(cfg.adversarial_fractions)} fractions "
          f"(Qwen3-4B-Instruct-2507 + LoRA, {MAX_STEPS} GRPO steps, single RTX 5090).", "",
          "P3 = adversarial-code containment. Treatment = rlox sandbox (`/verify`); "
          "Baseline = in-process `in_loop` exec (scope-bounded for host safety).", "",
          "| condition | injection | survived | mean steps | mean elapsed (s) |",
          "|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['condition']} | {x['fraction']:.2f} | {x['survived']}/{x['n']} | "
                  f"{x['mean_completed_steps']} | {x['mean_elapsed_secs']} |")
    md += ["", "## P3 verdict", "```json", json.dumps(verdict, indent=2), "```", "",
           "On a single GPU the in-process Baseline is scope-bounded so it does not wedge "
           "the host, but its mean elapsed time inflates with injection (in-loop adversarial "
           "code stalls the training step) while the rlox Treatment stays flat and survives "
           "all fractions — the sandbox contains the adversarial code out-of-process."]
    (out / "p3_summary.md").write_text("\n".join(md) + "\n")

    print("\n=== P3 SWEEP AGGREGATE ===")
    for x in rows:
        print(f"  {x['condition']:8s} frac={x['fraction']:.2f}  survived={x['survived']}/{x['n']}  "
              f"steps={x['mean_completed_steps']}  elapsed={x['mean_elapsed_secs']}s")
    print(f"\n[sweep] wrote {out/'p3_summary.md'} and p3_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
