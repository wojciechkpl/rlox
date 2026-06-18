#!/usr/bin/env python3
"""Aggregate a completed rlox P3 sweep from the per-run summary.json files.

Reads benchmarks/agentic/sweep_out/runs/<condition>_seed<S>_frac<F>/summary.json
(written by trl_grpo_run.py) and emits p3_summary.{json,md}. Decoupled from the
sweep launcher so re-aggregation needs no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/home/wk/rlox")
OUT = REPO / "benchmarks" / "agentic" / "sweep_out"


def main() -> int:
    runs = []
    for sj in sorted((OUT / "runs").glob("*/summary.json")):
        try:
            d = json.loads(sj.read_text())
        except Exception:
            continue
        runs.append(d)
    if not runs:
        print("no summaries found under", OUT / "runs")
        return 1

    agg: dict = {}
    for r in runs:
        cond = r.get("backend")
        frac = float(r.get("adversarial_fraction", 0.0))
        agg.setdefault((cond, frac), []).append(r)

    rows = []
    for (cond, frac), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(rs)
        rows.append({
            "condition": cond,
            "fraction": frac,
            "n": n,
            "survived": sum(1 for x in rs if x.get("survived")),
            "mean_elapsed_secs": round(sum(float(x.get("elapsed_secs", 0)) for x in rs) / n, 1),
            "mean_completed_steps": round(sum(int(x.get("completed_steps", 0)) for x in rs) / n, 2),
            "mean_reward_last": round(sum(float(x.get("mean_reward_last", 0)) for x in rs) / n, 3),
        })

    def row(cond, frac):
        return next((x for x in rows if x["condition"] == cond and x["fraction"] == frac), None)

    treat_rows = [x for x in rows if x["condition"] == "rlox"]
    base0 = row("in_loop", 0.0)
    verdict = {
        "treatment_survives_all_fractions": bool(treat_rows) and all(x["survived"] == x["n"] for x in treat_rows),
    }
    for frac in (0.05, 0.10):
        b, t = row("in_loop", frac), row("rlox", frac)
        if b and t and base0:
            verdict[f"frac_{frac}"] = {
                "baseline_survived": f"{b['survived']}/{b['n']}",
                "treatment_survived": f"{t['survived']}/{t['n']}",
                "baseline_elapsed_x_vs_clean_baseline": round(b["mean_elapsed_secs"] / max(base0["mean_elapsed_secs"], 1e-9), 2),
                "baseline_slower_than_treatment_x": round(b["mean_elapsed_secs"] / max(t["mean_elapsed_secs"], 1e-9), 2),
            }

    summary = {"n_runs": len(runs), "aggregate": rows, "p3_verdict": verdict, "runs": runs}
    (OUT / "p3_summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# rlox benchmark — P3 sweep result", "",
          f"{len(runs)} runs (2 conditions x seeds x fractions), Qwen3-4B-Instruct-2507 + LoRA, "
          "6 GRPO steps each, single RTX 5090.", "",
          "P3 = adversarial-code containment. **Treatment** = rlox sandbox (`/verify`, out-of-process). "
          "**Baseline** = in-process `in_loop` exec (scope-bounded for host safety).", "",
          "| condition | injection | survived | mean steps | mean reward | mean elapsed (s) |",
          "|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['condition']} | {x['fraction']:.2f} | {x['survived']}/{x['n']} | "
                  f"{x['mean_completed_steps']} | {x['mean_reward_last']} | {x['mean_elapsed_secs']} |")
    md += ["", "## P3 verdict", "```json", json.dumps(verdict, indent=2), "```"]
    (OUT / "p3_summary.md").write_text("\n".join(md) + "\n")

    print(f"=== P3 SWEEP AGGREGATE ({len(runs)} runs) ===")
    for x in rows:
        print(f"  {x['condition']:8s} frac={x['fraction']:.2f}  survived={x['survived']}/{x['n']}  "
              f"steps={x['mean_completed_steps']}  reward={x['mean_reward_last']}  "
              f"elapsed={x['mean_elapsed_secs']}s")
    print("verdict:", json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
