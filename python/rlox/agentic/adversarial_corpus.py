"""Backward-compat shim. Canonical implementation is in rlox_agent.adversarial_corpus."""
from rlox_agent.adversarial_corpus import *  # noqa: F401, F403
from rlox_agent.adversarial_corpus import (  # noqa: F401
    AdversarialCorpus,
    AdversarialInjector,
    AdversarialSample,
    CorpusIntegrityError,
    canonical_digest,
)
