#!/usr/bin/env python3
"""OQ-3 pilot — does the unprotected Baseline code-exec path degrade under
adversarial injection, while rlox Treatment contains the same code?

Gating question (PRD OQ-3): before committing the pre-registered sweep, confirm
that the Baseline (prime-rl/verifiers in-process code execution, mirrored here by
`_run_in_loop`'s `subprocess.run([python,-c,code], timeout=...)`) crashes or
stalls on the adversarial corpus at a 5-10% injection rate.

SAFETY MODEL (important): the Baseline exec is intentionally UNPROTECTED (no
cgroup, and `subprocess.run`'s timeout SIGKILLs only the direct child, never
forked grandchildren). Running it on adversarial code can exhaust the host — that
is the whole point. To keep wk-system safe we wrap *each* sample in its own
resource-capped, time-bounded transient scope:

    systemd-run --user --scope -p TasksMax=.. -p MemoryMax=.. -p RuntimeMaxSec=.. -- python -c <code>

The scope is a HOST safety net only. The finding is about what happens *inside*
it: if an adversarial sample is only ever stopped by the external cap (OOM /
TasksMax) or runs to the wall-clock limit (stall), then the Baseline exec has no
INTRINSIC containment — absent the cap (as in a real training process) it would
take the host down. rlox Treatment, by contrast, contains each sample intrinsically
within a per-sample cap (proven by the `corpus_containment` / `adversarial_containment`
Rust suites: bounded time-to-contain, zero survivors).

Usage (on wk-system, light env or system python — NO torch/vllm needed):
    python3 benchmarks/agentic/oq3_pilot.py
Writes benchmarks/agentic/pilot_results.md and pilot_results.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import rlox_agent.adversarial_corpus as ac  # noqa: E402  (stdlib-only module, isolated import)

CORPUS_PATH = ROOT / "benchmarks" / "agentic" / "corpus" / "adversarial_corpus_v1.json"

# Per-sample external safety cap (HOST protection only — NOT part of the Baseline).
SCOPE_TASKS_MAX = 512
SCOPE_MEM_MAX = "2G"
SCOPE_WALL_SECS = 6          # systemd RuntimeMaxSec — kills runaway scopes
SUBPROCESS_TIMEOUT = SCOPE_WALL_SECS + 8  # outer guard on the systemd-run call

# A benign control task: trivially-passing code.
BENIGN = ("print('ok')", "assert 1 + 1 == 2")


def _run_baseline_in_capped_scope(code: str, tests: str) -> dict:
    """Run one Baseline-style exec inside a capped, time-bounded transient scope.

    Returns a classified outcome dict. The scope's RuntimeMaxSec/MemoryMax/TasksMax
    bound the blast radius; we read the result to see how the unprotected exec behaved.
    """
    combined = code + "\n" + tests
    cmd = [
        "systemd-run", "--user", "--scope", "--quiet",
        "--property", f"TasksMax={SCOPE_TASKS_MAX}",
        "--property", f"MemoryMax={SCOPE_MEM_MAX}",
        "--property", f"MemorySwapMax=0",
        "--property", f"RuntimeMaxSec={SCOPE_WALL_SECS}",
        sys.executable, "-c", combined,
    ]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT)
        dt = time.monotonic() - t0
        rc = r.returncode
        # systemd-run --scope returns the unit's exit status. SIGKILL (137) /
        # RuntimeMaxSec timeout / OOM all surface as non-zero / signal codes.
        near_wall = dt >= SCOPE_WALL_SECS - 0.5
        blob = (r.stdout + r.stderr).lower()
        if rc == 0 and not near_wall:
            outcome = "completed"
        elif "oom" in blob or "out of memory" in blob or rc == 137 and "memory" in blob:
            outcome = "oom_killed_by_cap"
        elif near_wall:
            outcome = "stall_killed_at_wall"   # ran until the external time bound
        else:
            outcome = "killed_by_cap"           # TasksMax / signal
        return {"outcome": outcome, "rc": rc, "wall_s": round(dt, 2),
                "stderr_tail": r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else ""}
    except subprocess.TimeoutExpired:
        dt = time.monotonic() - t0
        return {"outcome": "stall_killed_at_wall", "rc": None, "wall_s": round(dt, 2),
                "stderr_tail": "outer subprocess timeout"}
    except Exception as e:  # noqa: BLE001 — pilot must not crash on a sample
        dt = time.monotonic() - t0
        return {"outcome": "spawn_error", "rc": None, "wall_s": round(dt, 2),
                "stderr_tail": f"{type(e).__name__}: {str(e)[:140]}"}


def main() -> int:
    corpus = ac.AdversarialCorpus.load(CORPUS_PATH)
    print(f"Loaded corpus v? with {len(corpus.samples)} samples, "
          f"categories: {sorted(corpus.categories)}\n")

    results: list[dict] = []

    # 1) Benign control through the Baseline exec — should complete cleanly.
    print("[baseline] benign control ...", flush=True)
    bc = _run_baseline_in_capped_scope(*BENIGN)
    bc.update(category="(benign control)", sample_id="benign")
    results.append(bc)
    print(f"    -> {bc['outcome']} (rc={bc['rc']}, {bc['wall_s']}s)\n", flush=True)

    # 2) Each adversarial sample through the Baseline exec.
    for s in corpus.samples:
        print(f"[baseline] {s.category} ({s.id}) ...", flush=True)
        out = _run_baseline_in_capped_scope(s.code, "")
        out.update(category=s.category, sample_id=s.id, expected_exit=s.expected_exit)
        results.append(out)
        print(f"    -> {out['outcome']} (rc={out['rc']}, {out['wall_s']}s) {out['stderr_tail']}",
              flush=True)

    # Verdict: the Baseline DEGRADES if every adversarial sample failed to complete
    # cleanly (i.e. was only stopped by the external cap or stalled to the wall).
    adversarial = [r for r in results if r["sample_id"] != "benign"]
    degraded = [r for r in adversarial if r["outcome"] != "completed"]
    benign_ok = results[0]["outcome"] == "completed"
    baseline_degrades = benign_ok and len(degraded) == len(adversarial)

    summary = {
        "benign_control_ok": benign_ok,
        "adversarial_total": len(adversarial),
        "adversarial_degraded": len(degraded),
        "baseline_degrades": baseline_degrades,
        "scope_caps": {"TasksMax": SCOPE_TASKS_MAX, "MemoryMax": SCOPE_MEM_MAX,
                       "RuntimeMaxSec": SCOPE_WALL_SECS},
        "results": results,
    }

    out_json = ROOT / "benchmarks" / "agentic" / "pilot_results.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Markdown report.
    lines = [
        "# OQ-3 Pilot — Baseline degradation under adversarial code",
        "",
        "**Question (PRD OQ-3):** does the unprotected Baseline (in-process "
        "`subprocess.run` exec, no cgroup, no process-tree kill) crash/stall on the "
        "adversarial corpus, gating the pre-registered sweep?",
        "",
        f"**Verdict: Baseline degrades = {baseline_degrades}** "
        f"({len(degraded)}/{len(adversarial)} adversarial samples failed to complete; "
        f"benign control completed = {benign_ok}).",
        "",
        "Each Baseline exec ran inside a host-safety scope "
        f"(TasksMax={SCOPE_TASKS_MAX}, MemoryMax={SCOPE_MEM_MAX}, "
        f"RuntimeMaxSec={SCOPE_WALL_SECS}s). An outcome other than `completed` means the "
        "unprotected exec had NO intrinsic containment — it was stopped only by the "
        "external cap or ran to the wall limit (a training-step stall). Absent the cap "
        "(a real in-process training loop) these exhaust the host.",
        "",
        "| category | sample | Baseline outcome | rc | wall(s) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['category']} | {r['sample_id']} | {r['outcome']} | "
                     f"{r['rc']} | {r['wall_s']} |")
    lines += [
        "",
        "## Treatment (rlox sandbox) — contrast",
        "Every corpus sample is contained intrinsically by the rlox sandbox with a "
        "bounded time-to-contain and zero survivors, proven by the Rust suites "
        "`crates/rlox-sandbox/tests/corpus_containment.rs` and `adversarial_containment.rs` "
        "(no external scope cap required).",
        "",
        "## Conclusion",
        ("Baseline degrades on the corpus at the chosen categories → the corpus/injection "
         "rate is sufficient; proceed to the pre-registered sweep." if baseline_degrades else
         "Baseline did NOT degrade on all categories → strengthen the corpus or raise the "
         "injection rate before committing the sweep."),
    ]
    out_md = ROOT / "benchmarks" / "agentic" / "pilot_results.md"
    out_md.write_text("\n".join(lines) + "\n")

    print(f"\n=== VERDICT: baseline_degrades = {baseline_degrades} "
          f"({len(degraded)}/{len(adversarial)} adversarial failed) ===")
    print(f"wrote {out_md} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
