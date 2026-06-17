"""_adversarial.py — Adversarial injector for rlox_verify.

Pure-Python, stdlib-only implementation extracted from
``rlox.agentic.adversarial_corpus`` so that ``rlox_verify`` can operate as a
self-contained installable package.

The canonical implementation lives in ``python/rlox/agentic/adversarial_corpus.py``.
Changes to corpus format / injection logic MUST be kept in sync manually.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AdversarialSample:
    """One entry from the corpus JSON."""
    id: str
    category: str
    language: str
    code: str
    expected_exit: str


class CorpusIntegrityError(ValueError):
    """Raised when the corpus SHA-256 does not match."""


def _canonical_digest(data: dict) -> str:
    """Return the SHA-256 hex digest for *data* with the ``sha256`` field blanked."""
    data_copy = copy.deepcopy(data)
    data_copy["sha256"] = ""
    serialised = json.dumps(data_copy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode()).hexdigest()


class AdversarialCorpus:
    """Parsed, integrity-verified adversarial corpus."""

    def __init__(self, samples: list[AdversarialSample], categories: set[str]) -> None:
        self.samples = samples
        self.categories = categories

    @classmethod
    def load(cls, path: str | Path) -> "AdversarialCorpus":
        """Load corpus from *path*, verify SHA-256 against the embedded field."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        with path.open() as fh:
            data = json.load(fh)
        recorded = data.get("sha256", "")
        computed = _canonical_digest(data)
        if computed != recorded:
            raise CorpusIntegrityError(
                f"Corpus integrity check failed: recorded={recorded!r}, computed={computed!r}"
            )
        samples: list[AdversarialSample] = [
            AdversarialSample(
                id=s["id"],
                category=s["category"],
                language=s["language"],
                code=s["code"],
                expected_exit=s["expected_exit"],
            )
            for s in data["samples"]
        ]
        categories = {s.category for s in samples}
        return cls(samples=samples, categories=categories)


class AdversarialInjector:
    """Stochastically replaces benign tasks with adversarial samples."""

    def __init__(
        self,
        corpus: AdversarialCorpus,
        fraction: float,
        seed: int,
    ) -> None:
        self._corpus = corpus
        self._fraction = fraction
        self._rng = random.Random(seed)

    def maybe_inject(self, task: Any) -> tuple[Any, bool]:
        """Possibly replace *task* with an adversarial sample.

        Returns:
            ``(task_or_adversarial, is_adversarial)``.
        """
        if self._fraction <= 0.0:
            return task, False
        if self._fraction >= 1.0 or self._rng.random() < self._fraction:
            sample = self._rng.choice(self._corpus.samples)
            return sample, True
        return task, False
