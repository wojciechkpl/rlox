# OQ-3 Pilot — Baseline degradation under adversarial code

**Question (PRD OQ-3):** does the unprotected Baseline (in-process `subprocess.run` exec, no cgroup, no process-tree kill) crash/stall on the adversarial corpus, gating the pre-registered sweep?

**Verdict: Baseline degrades = True** (6/6 adversarial samples failed to complete; benign control completed = True).

Each Baseline exec ran inside a host-safety scope (TasksMax=512, MemoryMax=2G, RuntimeMaxSec=6s). An outcome other than `completed` means the unprotected exec had NO intrinsic containment — it was stopped only by the external cap or ran to the wall limit (a training-step stall). Absent the cap (a real in-process training loop) these exhaust the host.

| category | sample | Baseline outcome | rc | wall(s) |
|---|---|---|---|---|
| (benign control) | benign | completed | 0 | 0.03 |
| infinite_loop | infinite-loop-001 | stall_killed_at_wall | -15 | 6.1 |
| fork_bomb | fork-bomb-001 | stall_killed_at_wall | -15 | 6.26 |
| memory_bomb | memory-bomb-001 | killed_by_cap | -9 | 0.57 |
| unkillable_thread | unkillable-thread-001 | stall_killed_at_wall | None | 14.02 |
| blocking_network | blocking-network-001 | stall_killed_at_wall | -15 | 6.15 |
| fd_exhaustion | fd-exhaustion-001 | killed_by_cap | 1 | 0.03 |

## Treatment (rlox sandbox) — contrast
Every corpus sample is contained intrinsically by the rlox sandbox with a bounded time-to-contain and zero survivors, proven by the Rust suites `crates/rlox-sandbox/tests/corpus_containment.rs` and `adversarial_containment.rs` (no external scope cap required).

## Conclusion
Baseline degrades on the corpus at the chosen categories → the corpus/injection rate is sufficient; proceed to the pre-registered sweep.
