"""Agentic sandbox verification: Baseline vs Treatment one-key swap.

Demonstrates the two-backend model at the heart of the rlox agentic benchmark:

  Baseline  — run_in_loop(code, tests, timeout)
              In-process subprocess exec; no isolation. Agrees with Treatment on
              benign code, but stalls or crashes the trainer on adversarial code
              (see benchmarks/agentic/oq3_pilot.py for the full degradation proof).

  Treatment — call_rlox_server(code, tests, is_adversarial, server_url, timeout)
              POSTs to the rlox-verify-server (Rust, namespaces + seccomp +
              cgroup v2 freeze→kill). Hard-isolated; every adversarial sample is
              contained intrinsically with bounded time-to-contain.

Section A — benign coding task
    A small Python function + unit-test string scored by both backends.
    Both should return ≈ 1.0 when the server is up; Baseline always works
    even when the server is down.

Section B — adversarial containment
    Loads the fixed adversarial corpus, demonstrates deterministic injection
    via AdversarialInjector, picks one adversarial sample (infinite loop), and
    scores it via Treatment → reward 0.0, contained.
    The Baseline path is intentionally SKIPPED for adversarial samples: an
    unprotected subprocess.run() would stall until the OS timeout, potentially
    blocking the training loop (proven in oq3_pilot.py).

Usage:
    python examples/agentic_sandbox_verify.py
    python examples/agentic_sandbox_verify.py --server-url http://localhost:8231
    python examples/agentic_sandbox_verify.py --server-url http://localhost:8231 --timeout 10.0

No torch required. Needs: httpx (stdlib + httpx only for Baseline; httpx + the
rlox-verify-server binary for Treatment).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Put python/ on sys.path so rlox_agent is importable without a pip install
# (mirrors the pattern used in benchmarks/agentic/oq3_pilot.py).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from rlox_agent.verifiers_adapter import call_rlox_server, run_in_loop  # noqa: E402
from rlox_agent.adversarial_corpus import AdversarialCorpus, AdversarialInjector  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = ROOT / "benchmarks" / "agentic" / "corpus" / "adversarial_corpus_v1.json"

# ---------------------------------------------------------------------------
# Benign task: a trivial Python function + a passing unit-test string.
# Both backends execute this and should both return 1.0.
# ---------------------------------------------------------------------------
BENIGN_CODE = """\
def add(a, b):
    return a + b
"""

BENIGN_TESTS = """\
assert add(1, 2) == 3
assert add(-1, 1) == 0
assert add(0, 0) == 0
"""

SERVER_HINT = (
    "To start the Treatment server on Linux with cgroup isolation:\n"
    "  systemd-run --user --unit=rlox-verify "
    "-p TasksMax=4096 -p MemoryMax=12G \\\n"
    "    target/release/rlox-verify-server --port 8231 --timeout-secs 5"
)


def section_a(server_url: str, timeout: float) -> None:
    """Score a benign coding task via Baseline and Treatment."""
    print("=" * 60)
    print("Section A — benign coding task")
    print("=" * 60)
    print(f"  code:  {BENIGN_CODE.strip()!r}")
    print(f"  tests: {BENIGN_TESTS.strip()!r}")
    print()

    # Baseline: in-process subprocess exec — always available, no server needed.
    baseline_reward = run_in_loop(BENIGN_CODE, BENIGN_TESTS, timeout)
    print(f"  Baseline  (run_in_loop)     reward = {baseline_reward:.3f}")

    # Treatment: POST to the Rust sandbox.  call_rlox_server returns 0.0 and
    # logs a WARNING on any network failure — we catch that outcome here and
    # surface a useful hint instead of an error.
    treatment_reward = call_rlox_server(
        BENIGN_CODE, BENIGN_TESTS, False, server_url, timeout
    )

    # Distinguish "server answered 0.0" from "server was unreachable → also 0.0".
    # We probe reachability cheaply by repeating the call at a very short
    # timeout; if it times out again we know it is down.  The simplest heuristic:
    # if Baseline == 1.0 but Treatment == 0.0, the server is likely unreachable
    # (a passing snippet should never score 0.0 in a live server).
    server_unreachable = baseline_reward == 1.0 and treatment_reward == 0.0

    if server_unreachable:
        print("  Treatment (call_rlox_server) server not reachable")
        print()
        print("  NOTE: Treatment section requires the rlox-verify-server to be running.")
        print(f"  {SERVER_HINT}")
    else:
        print(f"  Treatment (call_rlox_server) reward = {treatment_reward:.3f}")
        if abs(baseline_reward - treatment_reward) < 0.01:
            print("  -> Baseline and Treatment agree (expected for benign code).")
        else:
            print(
                f"  -> MISMATCH: baseline={baseline_reward:.3f} "
                f"treatment={treatment_reward:.3f} (investigate server logs)"
            )

    print()


def section_b(server_url: str, timeout: float) -> None:
    """Load the adversarial corpus, inject a sample, score it via Treatment."""
    print("=" * 60)
    print("Section B — adversarial containment")
    print("=" * 60)

    # --- Load and integrity-verify the fixed corpus ---
    corpus = AdversarialCorpus.load(CORPUS_PATH)
    print(
        f"  Loaded corpus with {len(corpus.samples)} samples, "
        f"categories: {sorted(corpus.categories)}"
    )

    # --- Demonstrate deterministic injection ---
    # fraction=1.0 so every call_to_maybe_inject returns an adversarial sample.
    injector = AdversarialInjector(corpus=corpus, fraction=1.0, seed=0)
    task: dict = {"prompt": "write a function that sums two numbers", "answer": BENIGN_TESTS}
    injected, is_adversarial = injector.maybe_inject(task)
    print(
        f"  Injection demo: fraction=1.0, seed=0 → "
        f"is_adversarial={is_adversarial}, sample={injected.id!r} ({injected.category})"
    )
    print()

    # --- Pick a specific adversarial sample: infinite loop ---
    # This is the canonical stall case: under the Baseline path (subprocess.run
    # with only a Python-side timeout), SIGKILL reaches only the direct child
    # process — forked grandchildren survive and block the training step.
    # See benchmarks/agentic/oq3_pilot.py for the full characterisation.
    #
    # We intentionally do NOT pass adversarial code to run_in_loop here.
    infinite_loop = next(s for s in corpus.samples if s.id == "infinite-loop-001")
    print(f"  Adversarial sample: {infinite_loop.id!r} ({infinite_loop.category})")
    print(f"  Code: {infinite_loop.code!r}")
    print()
    print("  [Baseline] SKIPPED for adversarial sample — unprotected subprocess.run()")
    print("  would stall until the OS timeout kills the child process, blocking")
    print("  the training loop. See benchmarks/agentic/oq3_pilot.py for proof.")
    print()

    # Treatment: hard-isolated sandbox — should return 0.0 and contain the loop.
    print("  [Treatment] scoring via rlox-verify-server ...")
    treatment_reward = call_rlox_server(
        infinite_loop.code, "", True, server_url, timeout
    )

    # Again, distinguish unreachable from a genuine 0.0 result.
    # For adversarial samples a returned 0.0 from the server IS the expected
    # correct answer, so we need to check reachability separately.  We use a
    # trivial benign probe to disambiguate.
    probe_reward = call_rlox_server("pass", "", False, server_url, timeout)
    server_unreachable = probe_reward == 0.0  # live server always returns 1.0 for "pass"

    if server_unreachable:
        print("  Treatment (call_rlox_server) server not reachable")
        print()
        print("  NOTE: start the server to verify adversarial containment:")
        print(f"  {SERVER_HINT}")
    else:
        print(
            f"  Treatment (call_rlox_server, is_adversarial=True) "
            f"reward = {treatment_reward:.3f}"
        )
        if treatment_reward == 0.0:
            print("  -> Adversarial sample CONTAINED (reward 0.0, training loop unaffected).")
        else:
            print(
                f"  -> UNEXPECTED reward={treatment_reward:.3f} for adversarial sample "
                "(check sandbox configuration)."
            )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic sandbox verification: Baseline vs Treatment one-key swap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The Baseline section (Section A, run_in_loop) runs without a server.\n"
            "The Treatment section (call_rlox_server) requires the rlox-verify-server.\n\n"
            + SERVER_HINT
        ),
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8231",
        metavar="URL",
        help="Base URL of the rlox-verify-server (default: http://localhost:8231)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        metavar="SECS",
        help="Per-call timeout in seconds (default: 8.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("rlox agentic sandbox verification")
    print(f"  server-url : {args.server_url}")
    print(f"  timeout    : {args.timeout}s")
    print()

    section_a(args.server_url, args.timeout)
    section_b(args.server_url, args.timeout)

    print("Done. Exit 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
