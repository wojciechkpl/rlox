# python/rlox/agentic/__init__.py
#
# Lightweight package init — NO heavy imports.
# This file must remain importable with plain `python3` and zero third-party
# dependencies (no torch, no vllm, no pydantic, no pytest).
#
# Components in this package:
#   stats.py              — BackendStats dataclass (Rust ↔ Python JSON contract)
#   verifiers_adapter.py  — verifiers load_environment / RloxRubric (Step 4)
#   adversarial_corpus.py — AdversarialCorpus, AdversarialInjector (Step 4b)
#   metric_collector.py   — MetricCollector, GPU sampling loop (Step 6a)
#   contagion_detector.py — ContagionDetector, ContagionReport (Step 6b)
#   config.py             — BenchmarkConfig, validate_config (Step 6c)
