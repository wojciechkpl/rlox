#!/usr/bin/env python3
"""Render figures for an rlox P3 benchmark sweep.

Consumes a completed sweep output directory:
    <sweep_dir>/runs/<condition>_seed<S>_frac<F>/summary.json   (reward_curve, final_reward, survived, elapsed_secs, ...)
    <sweep_dir>/runs/<condition>_seed<S>_frac<F>/metrics.jsonl   (per-step: step, reward, step_time, ...)
    <sweep_dir>/p3_summary.json                                  (optional aggregate, written by aggregate_sweep.py)

Produces (PNG) in --out-dir:
    fig_p3_degradation.png   — mean elapsed vs injection fraction (Baseline vs Treatment)
    fig_step_time_stall.png  — per-step step_time at the highest fraction (the escalating in-loop stall)
    fig_guardrail.png        — reward learning curve at fraction 0 (Baseline vs Treatment quality parity)
    fig_survival.png         — survival rate vs injection fraction
    fig_summary.png          — all four as one 2x2 panel

Usage:
    python benchmarks/agentic/plot_sweep.py --sweep-dir <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CONDITIONS = ["in_loop", "rlox"]
LABELS = {"in_loop": "Baseline (in-loop)", "rlox": "Treatment (rlox sandbox)"}
COLORS = {"in_loop": "#d1495b", "rlox": "#2e86ab"}


def _load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for sj in sorted(runs_dir.glob("*/summary.json")):
        try:
            d = json.loads(sj.read_text())
        except Exception:
            continue
        d["_dir"] = sj.parent
        # per-step metrics, if present
        mj = sj.parent / "metrics.jsonl"
        steps = []
        if mj.exists():
            for line in mj.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        steps.append(json.loads(line))
                    except Exception:
                        pass
        d["_steps"] = steps
        runs.append(d)
    return runs


def _load_metric_store(sweep_dir: Path) -> list[dict]:
    """Authoritative per-grid-point survival (includes DNF/timeout runs that
    never wrote a summary.json). Written by run_sweep."""
    recs = []
    for mf in sorted((sweep_dir / "metric_store").glob("*.json")):
        try:
            recs.append(json.loads(mf.read_text()))
        except Exception:
            pass
    return recs


def _by(runs, cond=None, frac=None):
    out = []
    for r in runs:
        if cond is not None and r.get("backend") != cond:
            continue
        if (
            frac is not None
            and abs(float(r.get("adversarial_fraction", 0)) - frac) > 1e-9
        ):
            continue
        out.append(r)
    return out


def _fractions(runs) -> list[float]:
    return sorted({round(float(r.get("adversarial_fraction", 0)), 4) for r in runs})


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib required: pip install matplotlib")

    sweep = Path(args.sweep_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(sweep / "runs")
    if not runs:
        raise SystemExit(f"no runs under {sweep / 'runs'}")
    fracs = _fractions(runs)
    print(f"loaded {len(runs)} runs, fractions={fracs}")

    # --- Fig 1: P3 degradation — mean elapsed vs fraction --------------------
    fig1, ax = plt.subplots(figsize=(6, 4))
    for c in CONDITIONS:
        ys = [_mean([float(r["elapsed_secs"]) for r in _by(runs, c, f)]) for f in fracs]
        ax.plot(
            [f * 100 for f in fracs], ys, "o-", color=COLORS[c], label=LABELS[c], lw=2
        )
    ax.set_xlabel("adversarial injection (%)")
    ax.set_ylabel("mean wall-clock per run (s)")
    ax.set_title("P3: in-loop adversarial code stalls training; sandbox does not")
    ax.legend()
    ax.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out / "fig_p3_degradation.png", dpi=140)

    # --- Fig 2: step-time stall at the highest fraction ----------------------
    fmax = fracs[-1]
    fig2, ax = plt.subplots(figsize=(6, 4))
    for c in CONDITIONS:
        # average step_time per step index across seeds at fmax
        per_step = defaultdict(list)
        for r in _by(runs, c, fmax):
            for s in r.get("_steps", []):
                if "step_time" in s and "step" in s:
                    per_step[int(s["step"])].append(float(s["step_time"]))
        if per_step:
            xs = sorted(per_step)
            ys = [_mean(per_step[x]) for x in xs]
            ax.plot(xs, ys, "-", color=COLORS[c], label=LABELS[c], lw=2)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("step time (s)")
    ax.set_title(f"Per-step stall at {fmax * 100:.0f}% injection")
    ax.legend()
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out / "fig_step_time_stall.png", dpi=140)

    # --- Fig 3: guardrail — reward learning curve at fraction 0 ---------------
    fig3, ax = plt.subplots(figsize=(6, 4))
    for c in CONDITIONS:
        per_step = defaultdict(list)
        for r in _by(runs, c, 0.0):
            curve = r.get("reward_curve") or [
                s.get("reward") for s in r.get("_steps", [])
            ]
            for i, v in enumerate(curve, start=1):
                if v is not None:
                    per_step[i].append(float(v))
        if per_step:
            xs = sorted(per_step)
            ys = [_mean(per_step[x]) for x in xs]
            ax.plot(xs, ys, "-", color=COLORS[c], label=LABELS[c], lw=2)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("reward (unit-test pass rate)")
    ax.set_title("Guardrail: quality parity at 0% injection")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(out / "fig_guardrail.png", dpi=140)

    # --- Fig 4: survival vs fraction (authoritative incl. DNF) ---------------
    mstore = _load_metric_store(sweep)

    def _ms_by(cond, frac):
        return [
            r
            for r in mstore
            if r.get("condition") == cond
            and abs(float(r.get("fraction", 0)) - frac) < 1e-9
        ]

    surv_src = mstore if mstore else runs
    surv_fracs = sorted(
        {
            round(float(r.get("fraction", r.get("adversarial_fraction", 0))), 4)
            for r in surv_src
        }
    )
    fig4, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    xs = list(range(len(surv_fracs)))
    for i, c in enumerate(CONDITIONS):
        rates = []
        for f in surv_fracs:
            rs = _ms_by(c, f) if mstore else _by(runs, c, f)
            rates.append(
                100 * sum(1 for r in rs if r.get("survived")) / len(rs) if rs else 0
            )
        ax.bar(
            [x + (i - 0.5) * width for x in xs],
            rates,
            width,
            color=COLORS[c],
            label=LABELS[c],
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f * 100:.0f}%" for f in surv_fracs])
    ax.set_xlabel("adversarial injection")
    ax.set_ylabel("runs survived (%)")
    ax.set_title("Run survival vs injection")
    ax.set_ylim(0, 105)
    ax.legend()
    fig4.tight_layout()
    fig4.savefig(out / "fig_survival.png", dpi=140)

    # --- Fig 5: mean GPU util vs injection fraction --------------------------
    # Reads mean_gpu_util from p3_summary.json (aggregate rows) when available;
    # falls back to computing it from per-run summary.json gpu_util fields.
    p3_json = sweep / "p3_summary.json"
    gpu_rows: list[dict] = []
    if p3_json.exists():
        try:
            gpu_rows = json.loads(p3_json.read_text()).get("aggregate", [])
        except Exception:
            gpu_rows = []

    # Build per-condition series: fraction → mean_gpu_util (skip None values).
    fig5_available = False
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    for c in CONDITIONS:
        pts = [
            (r["fraction"], r["mean_gpu_util"])
            for r in gpu_rows
            if r.get("condition") == c and r.get("mean_gpu_util") is not None
        ]
        if not pts:
            # Fall back to per-run summary.json fields.
            pts = []
            for f in fracs:
                vals = [
                    float(r["mean_gpu_util"])
                    for r in _by(runs, c, f)
                    if r.get("mean_gpu_util") is not None
                ]
                if vals:
                    pts.append((f, sum(vals) / len(vals)))
        if pts:
            fig5_available = True
            xs_g = [p[0] * 100 for p in pts]
            ys_g = [p[1] for p in pts]
            ax5.plot(xs_g, ys_g, "o-", color=COLORS[c], label=LABELS[c], lw=2)
    if fig5_available:
        ax5.set_xlabel("adversarial injection (%)")
        ax5.set_ylabel("mean GPU utilisation (%)")
        ax5.set_title("GPU idle effect: in-loop adversarial code starves the GPU")
        ax5.set_ylim(-5, 105)
        ax5.legend()
        ax5.grid(alpha=0.3)
        fig5.tight_layout()
        fig5.savefig(out / "fig_gpu_util.png", dpi=140)
        print("wrote", out / "fig_gpu_util.png")
    else:
        print("fig_gpu_util: no mean_gpu_util data available — skipping")
        plt.close(fig5)

    # --- Summary print -------------------------------------------------------
    for fname, fig in [
        ("fig_p3_degradation.png", fig1),
        ("fig_step_time_stall.png", fig2),
        ("fig_guardrail.png", fig3),
        ("fig_survival.png", fig4),
    ]:
        print("wrote", out / fname)
    print(f"\nFigures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
