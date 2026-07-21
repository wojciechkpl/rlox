# python/rlox/agentic/__init__.py
#
# Canonical home: rlox_agent (python/rlox_agent/).
# This package is now a backward-compat shim — import rlox.agentic.* still
# works in the full rlox venv (torch present), but the canonical
# implementations live in rlox_agent.
#
# Components in rlox_agent:
#   adversarial_corpus   — AdversarialCorpus, AdversarialInjector
#   verifiers_adapter    — RloxVerifierConfig, load_environment
#   stats                — BackendStats
#   metric_collector     — MetricCollector
#   contagion_detector   — ContagionDetector, ContagionReport
#   config               — BenchmarkConfig, validate_config
#   reporting            — bootstrap_ci, ci_overlap_check, write_summary, assess_go_no_go
