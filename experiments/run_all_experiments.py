#!/usr/bin/env python3
"""Master experiment runner for the rlox TDD experiment framework.

Runs experiments in order:
  1. Correctness  (fast, always run)
  2. Performance  (medium, skipped if correctness fails)
  3. Convergence  (slow, opt-in with --convergence)
  4. Ablation     (medium-slow, opt-in with --ablation)

Collects all results into a timestamped output directory:

    results/
      YYYY-MM-DD_HHMMSS/
        system_info.json
        correctness/
          gae.json     (pytest JUnit XML)
          buffer.json
          env.json
          llm_ops.json
          e2e.json
        performance/
          components.json
          e2e.json
        convergence/
          ...
        ablation/
          component_attribution.json
        summary.json

Generates summary.json with pass/fail counts and key metrics.

Usage
-----
    # Default: correctness + performance
    python experiments/run_all_experiments.py

    # Correctness only (fastest)
    python experiments/run_all_experiments.py --correctness-only

    # Add convergence tests (slow)
    python experiments/run_all_experiments.py --convergence

    # Add ablation tests
    python experiments/run_all_experiments.py --ablation

    # Full suite
    python experiments/run_all_experiments.py --convergence --ablation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Ensure the experiments directory is importable
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from utils import get_system_info, save_results


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def run_pytest(
    marker: str,
    test_files: list[str],
    results_dir: Path,
    extra_args: list[str] | None = None,
) -> tuple[int, str]:
    """Run pytest with the given marker on the specified test files.

    Returns (returncode, stdout+stderr output).
    """
    junit_xml = results_dir / f"junit_{marker}.xml"
    cmd = [
        sys.executable, "-m", "pytest",
        f"-m", marker,
        "-v",
        "--tb=short",
        f"--junitxml={junit_xml}",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(test_files)

    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}")

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=str(_EXPERIMENTS_DIR),
    )
    return result.returncode, ""


def parse_junit_xml(xml_path: Path) -> dict:
    """Parse a JUnit XML file into a simple summary dict."""
    if not xml_path.exists():
        return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "passed": 0}

    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "passed": 0}
        tests = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        passed = tests - failures - errors - skipped
        return {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "passed": passed,
        }
    except Exception as e:
        return {"parse_error": str(e)}


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def run_correctness_phase(results_dir: Path) -> tuple[bool, dict]:
    """Run all correctness tests. Returns (all_passed, summary_dict)."""
    test_files = [
        "test_correctness_gae.py",
        "test_correctness_buffer.py",
        "test_correctness_env.py",
        "test_correctness_llm_ops.py",
        "test_correctness_e2e.py",
    ]

    t0 = time.perf_counter()
    returncode, _ = run_pytest(
        marker="correctness",
        test_files=test_files,
        results_dir=results_dir,
    )
    elapsed = time.perf_counter() - t0

    xml_path = results_dir / "junit_correctness.xml"
    stats = parse_junit_xml(xml_path)
    all_passed = returncode == 0

    summary = {
        "phase": "correctness",
        "passed": all_passed,
        "returncode": returncode,
        "elapsed_s": elapsed,
        "stats": stats,
    }
    save_results(summary, results_dir / "correctness" / "summary.json")
    return all_passed, summary


def run_performance_phase(results_dir: Path) -> dict:
    """Run performance benchmarks (only after correctness passes)."""
    test_files = [
        "test_performance_components.py",
        "test_performance_e2e.py",
    ]

    t0 = time.perf_counter()
    returncode, _ = run_pytest(
        marker="performance",
        test_files=test_files,
        results_dir=results_dir,
    )
    elapsed = time.perf_counter() - t0

    xml_path = results_dir / "junit_performance.xml"
    stats = parse_junit_xml(xml_path)

    summary = {
        "phase": "performance",
        "passed": returncode == 0,
        "returncode": returncode,
        "elapsed_s": elapsed,
        "stats": stats,
    }
    save_results(summary, results_dir / "performance" / "summary.json")
    return summary


def run_convergence_phase(results_dir: Path) -> dict:
    """Run convergence tests (slow)."""
    t0 = time.perf_counter()
    returncode, _ = run_pytest(
        marker="convergence",
        test_files=["test_convergence.py"],
        results_dir=results_dir,
    )
    elapsed = time.perf_counter() - t0

    xml_path = results_dir / "junit_convergence.xml"
    stats = parse_junit_xml(xml_path)

    summary = {
        "phase": "convergence",
        "passed": returncode == 0,
        "returncode": returncode,
        "elapsed_s": elapsed,
        "stats": stats,
    }
    save_results(summary, results_dir / "convergence" / "summary.json")
    return summary


def run_ablation_phase(results_dir: Path) -> dict:
    """Run ablation tests."""
    t0 = time.perf_counter()
    returncode, _ = run_pytest(
        marker="performance",  # ablation tests are marked as performance
        test_files=["test_ablation.py"],
        results_dir=results_dir,
    )
    elapsed = time.perf_counter() - t0

    xml_path = results_dir / "junit_performance.xml"
    stats = parse_junit_xml(xml_path)

    summary = {
        "phase": "ablation",
        "passed": returncode == 0,
        "returncode": returncode,
        "elapsed_s": elapsed,
        "stats": stats,
    }
    save_results(summary, results_dir / "ablation" / "summary.json")
    return summary


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_global_summary(results_dir: Path, phase_summaries: list[dict]) -> None:
    """Write the top-level summary.json."""
    total_tests = sum(p.get("stats", {}).get("tests", 0) for p in phase_summaries)
    total_passed = sum(p.get("stats", {}).get("passed", 0) for p in phase_summaries)
    total_failed = sum(
        p.get("stats", {}).get("failures", 0) + p.get("stats", {}).get("errors", 0)
        for p in phase_summaries
    )
    total_skipped = sum(p.get("stats", {}).get("skipped", 0) for p in phase_summaries)

    all_phases_passed = all(p.get("passed", False) for p in phase_summaries)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results_dir": str(results_dir),
        "all_phases_passed": all_phases_passed,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "phases": phase_summaries,
        "system": get_system_info(),
    }
    save_results(summary, results_dir / "summary.json")

    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Results directory: {results_dir}")
    print(f"Overall: {'ALL PASSED' if all_phases_passed else 'SOME FAILED'}")
    print(f"Tests:   {total_passed}/{total_tests} passed, {total_failed} failed, {total_skipped} skipped")
    for p in phase_summaries:
        status = "PASS" if p.get("passed") else "FAIL"
        stats = p.get("stats", {})
        print(
            f"  [{status}] {p['phase']:<15s}  "
            f"{stats.get('passed', '?'):>4} passed  "
            f"{stats.get('failures', 0) + stats.get('errors', 0):>3} failed  "
            f"{stats.get('skipped', 0):>3} skipped  "
            f"({p.get('elapsed_s', 0):.1f}s)"
        )
    print(f"\nFull results: {results_dir / 'summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="rlox experiment runner: correctness → performance → convergence → ablation"
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Only run correctness tests (fastest)",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Also run slow convergence tests",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also run ablation performance tests",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: experiments/results/TIMESTAMP)",
    )
    args = parser.parse_args()

    # Create output directory
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = _EXPERIMENTS_DIR / "results" / ts

    for sub in ("correctness", "performance", "convergence", "ablation"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    # Save system info immediately
    save_results(get_system_info(), results_dir / "system_info.json")
    print(f"Results will be written to: {results_dir}")

    phase_summaries: list[dict] = []

    # Phase 1: Correctness (always)
    print("\n[Phase 1/4] Correctness tests")
    correctness_passed, correctness_summary = run_correctness_phase(results_dir)
    phase_summaries.append(correctness_summary)

    if not correctness_passed:
        print("\nCORRECTNESS TESTS FAILED.")
        print("Performance results would be meaningless. Stopping here.")
        print("Fix correctness failures before running performance benchmarks.")
        write_global_summary(results_dir, phase_summaries)
        return 1

    print("\n[Phase 1/4] All correctness tests passed.")

    if args.correctness_only:
        write_global_summary(results_dir, phase_summaries)
        return 0

    # Phase 2: Performance
    print("\n[Phase 2/4] Performance benchmarks")
    perf_summary = run_performance_phase(results_dir)
    phase_summaries.append(perf_summary)

    # Phase 3: Convergence (opt-in)
    if args.convergence:
        print("\n[Phase 3/4] Convergence tests (slow)")
        conv_summary = run_convergence_phase(results_dir)
        phase_summaries.append(conv_summary)
    else:
        print("\n[Phase 3/4] Convergence tests SKIPPED (use --convergence to enable)")

    # Phase 4: Ablation (opt-in)
    if args.ablation:
        print("\n[Phase 4/4] Ablation tests")
        abl_summary = run_ablation_phase(results_dir)
        phase_summaries.append(abl_summary)
    else:
        print("\n[Phase 4/4] Ablation tests SKIPPED (use --ablation to enable)")

    write_global_summary(results_dir, phase_summaries)

    all_passed = all(p.get("passed", False) for p in phase_summaries)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
