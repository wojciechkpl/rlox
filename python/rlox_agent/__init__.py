# python/rlox_agent/__init__.py
#
# Standalone agentic package — NO imports at module level.
# This file must remain importable with plain python3 and zero third-party
# dependencies (no torch, no _rlox_core, no vllm, no pydantic).
#
# Canonical home for the agentic modules previously housed in rlox.agentic.
# The old rlox.agentic.* paths remain available as backward-compat shims.
#
# Components:
#   adversarial_corpus   — AdversarialCorpus, AdversarialInjector
#   verifiers_adapter    — RloxVerifierConfig, load_environment
#   stats                — BackendStats
#   metric_collector     — MetricCollector
#   contagion_detector   — ContagionDetector, ContagionReport
#   config               — BenchmarkConfig, validate_config
#   reporting            — bootstrap_ci, ci_overlap_check, write_summary, assess_go_no_go
