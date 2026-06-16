"""Failing RED tests for Component 5: AdversarialCorpus + AdversarialInjector.

These tests specify the public contract; they must fail until the implementation
is in place. They do NOT import from rlox (no torch dependency).

All imports are top-level thanks to conftest.py injecting python/rlox/agentic/
onto sys.path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

# Top-level imports — never `from rlox.agentic import ...`
import adversarial_corpus as ac
from adversarial_corpus import (
    AdversarialCorpus,
    AdversarialInjector,
    AdversarialSample,
    CorpusIntegrityError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_CORPUS_PATH = _REPO_ROOT / "benchmarks" / "agentic" / "corpus" / "adversarial_corpus_v1.json"
_EXPECTED_SHA256 = "256a3c99e1463fdca22c4134a4b59760f7bc0b6e0cc128b54ffc141fe63ca145"
_EXPECTED_CATEGORIES = {
    "infinite_loop",
    "fork_bomb",
    "memory_bomb",
    "unkillable_thread",
    "blocking_network",
    "fd_exhaustion",
}
_EXPECTED_SAMPLE_COUNT = 6

# A minimal benign task dict used as a stand-in for "whatever the trainer passes"
_BENIGN_TASK = {"prompt": "Write a hello-world function.", "answer": "print('hello')"}


def _make_tampered_corpus(tmp_path: Path) -> Path:
    """Write a copy of the real corpus with one code field mutated."""
    with open(_CORPUS_PATH) as f:
        data = json.load(f)
    # Corrupt the first sample's code — sha256 will no longer match
    data["samples"][0]["code"] = "# TAMPERED"
    dest = tmp_path / "tampered_corpus.json"
    dest.write_text(json.dumps(data))
    return dest


# ---------------------------------------------------------------------------
# A) Load + integrity verification
# ---------------------------------------------------------------------------

class TestAdversarialCorpusLoad:
    def test_load_real_corpus_succeeds(self):
        """Happy path: real corpus file loads without raising."""
        corpus = AdversarialCorpus.load(_CORPUS_PATH)
        assert corpus is not None

    def test_load_returns_adversarial_corpus_instance(self):
        corpus = AdversarialCorpus.load(_CORPUS_PATH)
        assert isinstance(corpus, AdversarialCorpus)

    def test_samples_are_adversarial_sample_instances(self):
        corpus = AdversarialCorpus.load(_CORPUS_PATH)
        assert len(corpus.samples) > 0
        for s in corpus.samples:
            assert isinstance(s, AdversarialSample)

    def test_sample_count_matches_corpus(self):
        corpus = AdversarialCorpus.load(_CORPUS_PATH)
        assert len(corpus.samples) == _EXPECTED_SAMPLE_COUNT

    def test_all_six_categories_present(self):
        """Spec: all 6 adversarial categories must be present."""
        corpus = AdversarialCorpus.load(_CORPUS_PATH)
        assert corpus.categories == _EXPECTED_CATEGORIES

    def test_sample_fields_populated(self):
        """Each AdversarialSample must expose id, category, language, code, expected_exit."""
        corpus = AdversarialCorpus.load(_CORPUS_PATH)
        for s in corpus.samples:
            assert s.id and isinstance(s.id, str)
            assert s.category and isinstance(s.category, str)
            assert s.language and isinstance(s.language, str)
            assert s.code and isinstance(s.code, str)
            assert s.expected_exit and isinstance(s.expected_exit, str)

    def test_load_accepts_string_path(self):
        """load() must accept both str and Path."""
        corpus = AdversarialCorpus.load(str(_CORPUS_PATH))
        assert isinstance(corpus, AdversarialCorpus)

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AdversarialCorpus.load(tmp_path / "nonexistent.json")

    def test_tampered_corpus_raises_integrity_error(self, tmp_path):
        """A corpus with mutated content must raise CorpusIntegrityError."""
        bad_path = _make_tampered_corpus(tmp_path)
        with pytest.raises(CorpusIntegrityError):
            AdversarialCorpus.load(bad_path)

    def test_wrong_sha256_field_raises_integrity_error(self, tmp_path):
        """If sha256 field itself is wrong, must raise CorpusIntegrityError."""
        with open(_CORPUS_PATH) as f:
            data = json.load(f)
        data["sha256"] = "0" * 64  # wrong but valid hex length
        dest = tmp_path / "wrong_sha.json"
        dest.write_text(json.dumps(data))
        with pytest.raises(CorpusIntegrityError):
            AdversarialCorpus.load(dest)

    def test_sha256_field_in_loaded_data_matches_expected(self):
        """The digest embedded in the file must equal the committed value."""
        with open(_CORPUS_PATH) as f:
            data = json.load(f)
        assert data["sha256"] == _EXPECTED_SHA256


# ---------------------------------------------------------------------------
# B) Canonical digest helper
# ---------------------------------------------------------------------------

class TestCanonicalDigest:
    def test_canonical_digest_matches_committed_value(self):
        """_canonical_digest() must reproduce the committed sha256."""
        with open(_CORPUS_PATH) as f:
            data = json.load(f)
        digest = AdversarialCorpus._canonical_digest(data)
        assert digest == _EXPECTED_SHA256

    def test_canonical_digest_is_insensitive_to_sha256_field_value(self):
        """The sha256 field's actual value must not affect the digest."""
        with open(_CORPUS_PATH) as f:
            data = json.load(f)
        data_copy = copy.deepcopy(data)
        data_copy["sha256"] = "ANYTHING"
        # Both should produce the same digest
        assert AdversarialCorpus._canonical_digest(data) == AdversarialCorpus._canonical_digest(data_copy)

    def test_canonical_digest_changes_on_sample_mutation(self):
        """Mutating a sample must produce a different digest (tamper detection)."""
        with open(_CORPUS_PATH) as f:
            data = json.load(f)
        original = AdversarialCorpus._canonical_digest(data)
        data["samples"][0]["code"] = "# hacked"
        mutated = AdversarialCorpus._canonical_digest(data)
        assert original != mutated


# ---------------------------------------------------------------------------
# C) AdversarialInjector — fraction boundaries
# ---------------------------------------------------------------------------

class TestAdversarialInjectorBoundaries:
    @pytest.fixture
    def corpus(self):
        return AdversarialCorpus.load(_CORPUS_PATH)

    def test_fraction_zero_never_injects(self, corpus):
        """fraction=0.0 must never inject, regardless of seed or task."""
        injector = AdversarialInjector(corpus, fraction=0.0, seed=0)
        for i in range(200):
            task, is_adv = injector.maybe_inject({"prompt": f"task {i}"})
            assert not is_adv
            assert task == {"prompt": f"task {i}"}

    def test_fraction_one_always_injects(self, corpus):
        """fraction=1.0 must always inject an adversarial sample."""
        injector = AdversarialInjector(corpus, fraction=1.0, seed=0)
        for i in range(200):
            _, is_adv = injector.maybe_inject({"prompt": f"task {i}"})
            assert is_adv

    def test_fraction_one_returns_adversarial_sample(self, corpus):
        """When injected, the returned task must be an AdversarialSample."""
        injector = AdversarialInjector(corpus, fraction=1.0, seed=0)
        result, is_adv = injector.maybe_inject(_BENIGN_TASK)
        assert is_adv
        assert isinstance(result, AdversarialSample)

    def test_fraction_zero_returns_original_task(self, corpus):
        """When not injected, the original task object must be returned unchanged."""
        injector = AdversarialInjector(corpus, fraction=0.0, seed=0)
        result, is_adv = injector.maybe_inject(_BENIGN_TASK)
        assert not is_adv
        assert result is _BENIGN_TASK  # identity, not just equality


# ---------------------------------------------------------------------------
# D) AdversarialInjector — stochastic fraction ≈ 0.10
# ---------------------------------------------------------------------------

class TestAdversarialInjectorStochasticFraction:
    @pytest.fixture
    def corpus(self):
        return AdversarialCorpus.load(_CORPUS_PATH)

    def test_fraction_010_within_tolerance(self, corpus):
        """At fraction=0.10 over 1000 calls the empirical rate is within ±2pp."""
        N = 1000
        injector = AdversarialInjector(corpus, fraction=0.10, seed=12345)
        hits = sum(
            injector.maybe_inject({"prompt": f"task {i}"})[1]
            for i in range(N)
        )
        empirical = hits / N
        assert abs(empirical - 0.10) <= 0.02, (
            f"Expected ~10% injection, got {empirical:.3%} over {N} calls"
        )

    def test_injection_is_deterministic_for_same_seed(self, corpus):
        """Two injectors with the same seed must produce identical sequences."""
        N = 200
        tasks = [{"prompt": f"task {i}"} for i in range(N)]

        injector_a = AdversarialInjector(corpus, fraction=0.25, seed=99)
        injector_b = AdversarialInjector(corpus, fraction=0.25, seed=99)

        results_a = [injector_a.maybe_inject(t)[1] for t in tasks]
        results_b = [injector_b.maybe_inject(t)[1] for t in tasks]
        assert results_a == results_b

    def test_different_seeds_produce_different_sequences(self, corpus):
        """Different seeds must produce (with overwhelming probability) different sequences."""
        N = 200
        tasks = [{"prompt": f"task {i}"} for i in range(N)]

        injector_a = AdversarialInjector(corpus, fraction=0.5, seed=1)
        injector_b = AdversarialInjector(corpus, fraction=0.5, seed=2)

        results_a = [injector_a.maybe_inject(t)[1] for t in tasks]
        results_b = [injector_b.maybe_inject(t)[1] for t in tasks]
        # At p=0.5 the chance two 200-length sequences are identical is ~1/2^200
        assert results_a != results_b


# ---------------------------------------------------------------------------
# E) AdversarialInjector — return-type contract
# ---------------------------------------------------------------------------

class TestAdversarialInjectorReturnContract:
    @pytest.fixture
    def corpus(self):
        return AdversarialCorpus.load(_CORPUS_PATH)

    def test_maybe_inject_returns_two_tuple(self, corpus):
        injector = AdversarialInjector(corpus, fraction=0.5, seed=7)
        result = injector.maybe_inject(_BENIGN_TASK)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_maybe_inject_second_element_is_bool(self, corpus):
        injector = AdversarialInjector(corpus, fraction=0.5, seed=7)
        _, is_adv = injector.maybe_inject(_BENIGN_TASK)
        assert isinstance(is_adv, bool)

    def test_injected_adversarial_sample_is_from_corpus(self, corpus):
        """When injected, the returned sample must be one of the corpus samples."""
        injector = AdversarialInjector(corpus, fraction=1.0, seed=0)
        seen_ids = set()
        for _ in range(50):
            task, _ = injector.maybe_inject(_BENIGN_TASK)
            seen_ids.add(task.id)
        # all returned ids must be from the corpus
        corpus_ids = {s.id for s in corpus.samples}
        assert seen_ids.issubset(corpus_ids)

    def test_injector_samples_multiple_categories_over_many_calls(self, corpus):
        """With fraction=1.0 over many calls all corpus categories should appear."""
        injector = AdversarialInjector(corpus, fraction=1.0, seed=42)
        seen_categories = set()
        for _ in range(300):
            task, _ = injector.maybe_inject(_BENIGN_TASK)
            seen_categories.add(task.category)
        assert seen_categories == _EXPECTED_CATEGORIES
