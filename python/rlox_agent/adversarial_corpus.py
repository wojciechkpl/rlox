"""adversarial_corpus.py — Component 5: Adversarial injector.

Imports: stdlib only. No torch, no rlox, no vllm.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AdversarialSample:
    """One entry from the corpus JSON."""
    id: str
    category: str
    language: str
    code: str
    expected_exit: str


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

class CorpusIntegrityError(ValueError):
    """Raised when the corpus SHA-256 does not match."""


def canonical_digest(data: dict) -> str:
    """Return the SHA-256 hex digest for *data* with the ``sha256`` field blanked.

    Canonical convention:
      1. Deep-copy *data* and set ``sha256`` to ``""``.
      2. ``json.dumps(copy, sort_keys=True, separators=(",", ":"))``
      3. SHA-256 of the UTF-8 encoded string.
    """
    data_copy = copy.deepcopy(data)
    data_copy["sha256"] = ""
    serialised = json.dumps(data_copy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode()).hexdigest()


class AdversarialCorpus:
    """Parsed, integrity-verified adversarial corpus."""

    def __init__(self, samples: list[AdversarialSample], categories: set[str]) -> None:
        self.samples = samples
        self.categories = categories

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "AdversarialCorpus":
        """Load corpus from *path*, verify SHA-256 against the embedded field.

        Canonical digest convention: set the ``sha256`` field to ``""``, then
        ``json.dumps(data, sort_keys=True, separators=(",", ":"))`` → sha256hex.

        Raises:
            FileNotFoundError: if *path* does not exist.
            CorpusIntegrityError: if the computed digest does not match the
                recorded one (tamper detection).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        with path.open() as fh:
            data = json.load(fh)

        recorded = data.get("sha256", "")
        computed = cls._canonical_digest(data)
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

    # ------------------------------------------------------------------
    # internal helpers (may be used by tests indirectly via .load)
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_digest(data: dict) -> str:
        """Return the expected sha256 hex for *data* (with sha256 blanked)."""
        return canonical_digest(data)


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------

class AdversarialInjector:
    """Stochastically replaces benign tasks with adversarial samples.

    Args:
        corpus: An :class:`AdversarialCorpus` instance.
        fraction: Probability (0.0–1.0 inclusive) that any given task is
            replaced with an adversarial sample.
        seed: Integer seed that makes the injection sequence fully deterministic
            across runs (same seed → identical sequence of injection decisions).
    """

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
            ``(task_or_adversarial, is_adversarial)`` — the (possibly replaced)
            task and a boolean flag indicating whether injection occurred.
        """
        if self._fraction <= 0.0:
            return task, False
        if self._fraction >= 1.0 or self._rng.random() < self._fraction:
            sample = self._rng.choice(self._corpus.samples)
            return sample, True
        return task, False
