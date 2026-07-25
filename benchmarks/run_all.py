#!/usr/bin/env python3
"""
Master benchmark runner for rlox.

Runs all available benchmarks and produces a consolidated report.
Benchmarks that depend on unimplemented features are skipped gracefully.

Usage:
    python benchmarks/run_all.py [--output-dir benchmark_results] [--quick]
    python benchmarks/run_all.py --category buffer_ops
    python benchmarks/run_all.py --category e2e
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from harness import system_info


def check_phase_available(phase: str) -> bool:
    """Check if a phase's rlox components are importable."""
    checks = {
        "phase2": lambda: __import__("rlox").ExperienceTable,
        "phase3": lambda: __import__("rlox").compute_gae,
        "phase4": lambda: __import__("rlox").compute_group_advantages,
    }
    try:
        checks[phase]()
        return True
    except (ImportError, AttributeError):
        return False


def check_framework_available(name: str) -> bool:
    """Check if an optional framework is installed."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="rlox benchmark suite")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced iterations for faster feedback")
    parser.add_argument("--category", choices=[
        "all", "env_stepping", "buffer_ops", "gae", "llm", "e2e",
        "distributed", "algorithms", "mmap_buffer",
    ], default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("rlox Benchmark Suite — Three-Framework Comparison")
    print("=" * 70)

    info = system_info()
    print(f"Platform: {info['platform']}")
    print(f"Python:   {info['python_version'].split()[0]}")
    print(f"NumPy:    {info['numpy_version']}")
    print(f"CPU:      {info.get('cpu', 'unknown')} ({info['cpu_count']} cores)")
    if info.get("torch_version"):
        print(f"PyTorch:  {info['torch_version']}")
    if info.get("cuda_available"):
        print(f"GPU:      {info.get('gpu', 'unknown')}")
    print(f"rlox:     {'available' if info.get('rlox_available') else 'NOT FOUND'}")

    # Framework availability
    print("\nFramework availability:")
    for fw, pkg in [
        ("TorchRL", "torchrl"),
        ("Stable-Baselines3", "stable_baselines3"),
        ("Gymnasium", "gymnasium"),
    ]:
        available = check_framework_available(pkg)
        print(f"  {fw}: {'available' if available else 'NOT INSTALLED (benchmarks will skip)'}")

    # Phase availability
    print("\nrlox phase availability:")
    for phase, desc in [
        ("phase2", "Experience Storage (buffer, VarLenStore)"),
        ("phase3", "Training Core (GAE)"),
        ("phase4", "LLM Post-Training (GRPO, DPO, token KL)"),
    ]:
        available = check_phase_available(phase)
        status = "READY" if available else "not yet implemented"
        print(f"  {desc}: {status}")

    # Run benchmarks
    if args.category in ("all", "env_stepping"):
        try:
            from bench_env_stepping import run_all as run_env
            run_env(args.output_dir)
        except Exception as e:
            print(f"\nENV STEPPING BENCHMARK FAILED: {e}")

    if args.category in ("all", "buffer_ops"):
        if check_phase_available("phase2"):
            try:
                from bench_buffer_ops import run_all as run_buffers
                run_buffers(args.output_dir)
            except Exception as e:
                print(f"\nBUFFER OPS BENCHMARK FAILED: {e}")
        else:
            print("\n[SKIP] Buffer benchmarks — Phase 2 not implemented")

    if args.category in ("all", "gae"):
        if check_phase_available("phase3"):
            try:
                from bench_gae import run_all as run_gae
                run_gae(args.output_dir)
            except Exception as e:
                print(f"\nGAE BENCHMARK FAILED: {e}")
        else:
            print("\n[SKIP] GAE benchmarks — Phase 3 not implemented")

    if args.category in ("all", "llm"):
        if check_phase_available("phase4"):
            try:
                from bench_llm_ops import run_all as run_llm
                run_llm(args.output_dir)
            except Exception as e:
                print(f"\nLLM OPS BENCHMARK FAILED: {e}")
        else:
            print("\n[SKIP] LLM benchmarks — Phase 4 not implemented")

    if args.category in ("all", "e2e"):
        if check_phase_available("phase2") and check_phase_available("phase3"):
            try:
                from bench_e2e_rollout import run_all as run_e2e
                run_e2e(args.output_dir)
            except Exception as e:
                print(f"\nE2E ROLLOUT BENCHMARK FAILED: {e}")
        else:
            print("\n[SKIP] E2E rollout benchmarks — Phase 2+3 required")

    if args.category in ("all", "distributed"):
        try:
            from bench_distributed import run_all as run_distributed
            run_distributed(args.output_dir)
        except Exception as e:
            print(f"\nDISTRIBUTED BENCHMARK FAILED: {e}")

    if args.category in ("all", "algorithms"):
        try:
            from bench_algorithms import run_all as run_algorithms
            run_algorithms(args.output_dir)
        except Exception as e:
            print(f"\nALGORITHM BENCHMARK FAILED: {e}")

    if args.category in ("all", "mmap_buffer"):
        if check_phase_available("phase2"):
            try:
                from bench_mmap_buffer import run_all as run_mmap
                run_mmap(args.output_dir)
            except Exception as e:
                print(f"\nMMAP BUFFER BENCHMARK FAILED: {e}")
        else:
            print("\n[SKIP] Mmap buffer benchmarks — Phase 2 not implemented")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
