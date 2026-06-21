"""verifiers_adapter.py — Component 4: prime-rl verifiers adapter.

Imports: stdlib + verifiers + httpx + datasets only (all lazy where possible).
No torch, no rlox top-level package.

`import rlox_agent.verifiers_adapter` itself does NOT import verifiers at
module load time — the import happens inside `load_environment` so that
`import rlox_agent` works without verifiers installed.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import httpx

from rlox_agent import adversarial_corpus as _ac

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RloxVerifierConfig:
    """Configuration for the rlox verifiers adapter.

    Attributes:
        rollout_backend: ``"in_loop"`` for baseline (execute in-process) or
            ``"rlox"`` for treatment (POST to the Rust rollout server).
        rlox_server_url: Base URL of the Rust rollout server
            (e.g. ``"http://localhost:8080"``).  Unused when
            ``rollout_backend == "in_loop"``.
        per_sample_timeout_secs: HTTP / subprocess timeout per rollout.
        group_size: Number of rollouts per group (written to
            ``env.sampling_args["n"]`` so prime-rl draws the right group size).
        adversarial_fraction: Fraction of tasks to replace with adversarial
            samples (0.0 = never, 1.0 = always).
        adversarial_corpus_path: Path to the ``adversarial_corpus_v1.json``
            file (required when ``adversarial_fraction > 0``).
        seed: Integer seed for the adversarial injector PRNG.
    """
    rollout_backend: str = "in_loop"
    rlox_server_url: str = "http://localhost:8080"
    per_sample_timeout_secs: float = 30.0
    group_size: int = 8
    adversarial_fraction: float = 0.0
    adversarial_corpus_path: str | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text(messages: list[dict] | str) -> str:
    """Return the text content from a messages list or a plain string."""
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
    return "\n".join(parts)


def _run_in_loop(code: str, tests: str, timeout: float) -> float:
    """Execute *code* against *tests* in the current venv via subprocess.

    This is the Baseline path — no isolation layer, no network call.
    Uses ``sys.executable`` so the correct venv interpreter is invoked even
    when ``python`` is not on PATH (or resolves to a different version).

    Returns 1.0 if the combined snippet exits with code 0, else 0.0.
    """
    combined = code + "\n" + tests
    try:
        result = subprocess.run(
            [sys.executable, "-c", combined],
            capture_output=True,
            timeout=timeout,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return 0.0


def _call_rlox_server(
    code: str,
    tests: str,
    is_adversarial: bool,
    server_url: str,
    timeout: float,
) -> float:
    """POST ``{"code": code, "tests": tests, "is_adversarial": ...}`` to
    ``{server_url}/verify``.

    The ``/verify`` endpoint receives an already-generated completion and
    sandbox-executes it, returning a scored reward.  The ``is_adversarial``
    flag is forwarded so the server can record containment telemetry.

    Returns the ``"reward"`` field from the JSON response, or 0.0 on any
    exception (which is logged as a WARNING so server outages are visible
    in training logs).
    """
    url = f"{server_url}/verify"
    payload = {"code": code, "tests": tests, "is_adversarial": is_adversarial}
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return float(data.get("reward", 0.0))
    except Exception as exc:
        logger.warning(
            "rlox /verify call to %s failed (%s: %s); returning reward=0.0",
            url,
            type(exc).__name__,
            exc,
        )
        return 0.0


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def load_environment(config: RloxVerifierConfig) -> "vf.Environment":  # noqa: F821
    """Build a ``vf.Environment`` backed by the requested rollout backend.

    verifiers is imported lazily here so that ``import rlox_agent.verifiers_adapter``
    at the top level does not require verifiers to be installed.

    The returned environment:

    * Applies the :class:`~adversarial_corpus.AdversarialInjector` (with
      ``config.adversarial_fraction`` and ``config.seed``) before dispatching
      to EITHER backend — so injection is identical in both conditions.
    * ``rollout_backend == "rlox"``: POSTs
      ``{"code": ..., "tests": ..., "is_adversarial": <bool>}`` to
      ``config.rlox_server_url/verify`` via :mod:`httpx` and uses the JSON
      field ``reward`` from the response as the scalar reward.  Any HTTP or
      network failure is logged as a WARNING and returns reward 0.0.
    * ``rollout_backend == "in_loop"``: executes the code via
      ``sys.executable`` subprocess and computes the reward locally, WITHOUT
      contacting any remote server.

    The rubric is passed to the ``SingleTurnEnv`` **constructor** so that
    verifiers wraps it in a ``RubricGroup``.  After construction
    ``env.rubric`` is a ``vf.RubricGroup`` and our reward function is
    reachable at ``env.rubric.rubrics[0].funcs``.

    ``env.sampling_args["n"]`` is set to ``config.group_size`` so that
    prime-rl draws the right number of completions per prompt.

    The ``dataset`` argument is a required no-op placeholder
    (``datasets.Dataset.from_list([{}])``).  Rewards come entirely from the
    external backend; the dataset row is never read.

    Args:
        config: A :class:`RloxVerifierConfig` instance.

    Returns:
        A :class:`verifiers.Environment` instance.

    Raises:
        ValueError: if ``rollout_backend`` is not ``"in_loop"`` or ``"rlox"``.
        ValueError: if ``adversarial_fraction > 0`` but
            ``adversarial_corpus_path`` is ``None``.
    """
    import verifiers as vf  # lazy: not required at module import time

    _VALID_BACKENDS = {"in_loop", "rlox"}
    if config.rollout_backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid rollout_backend={config.rollout_backend!r}. "
            f"Must be one of {sorted(_VALID_BACKENDS)}."
        )

    if config.adversarial_fraction > 0 and config.adversarial_corpus_path is None:
        raise ValueError(
            "adversarial_corpus_path must be provided when adversarial_fraction > 0."
        )

    # Build the injector (may be None when fraction == 0)
    injector: _ac.AdversarialInjector | None = None
    if config.adversarial_fraction > 0 and config.adversarial_corpus_path is not None:
        corpus = _ac.AdversarialCorpus.load(config.adversarial_corpus_path)
        injector = _ac.AdversarialInjector(
            corpus=corpus,
            fraction=config.adversarial_fraction,
            seed=config.seed,
        )

    # Capture values needed inside the closure
    backend = config.rollout_backend
    server_url = config.rlox_server_url
    timeout = config.per_sample_timeout_secs

    def _reward_func(
        prompt: list[dict] | str,
        completion: list[dict] | str,
        answer: Any = "",
        state: dict | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute reward by optionally injecting an adversarial sample then
        dispatching to the configured backend.

        Also writes the reward into ``state["reward_funcs_results"]`` (a dict
        keyed by function name) so downstream consumers and tests can inspect
        per-function scores without relying on verifiers' internal ``metrics``
        key.
        """

        # Build a lightweight task dict so the injector can make its decision
        task: Any = {"prompt": prompt, "answer": answer}
        is_adversarial = False

        if injector is not None:
            task, is_adversarial = injector.maybe_inject(task)

        # Extract code (model completion) and tests (expected answer / test suite)
        if isinstance(task, _ac.AdversarialSample):
            code_text = task.code
            tests_text = ""
        else:
            code_text = _extract_text(completion)
            tests_text = answer if isinstance(answer, str) else _extract_text(answer)

        if backend == "rlox":
            reward = _call_rlox_server(
                code_text, tests_text, is_adversarial, server_url, timeout
            )
        else:
            # in_loop baseline — must NOT contact any remote server
            reward = _run_in_loop(code_text, tests_text, timeout)

        # Write per-function result into state so consumers can inspect it.
        # verifiers calls us with state= as a keyword; guard against None for
        # callers that do not pass state (e.g. direct unit tests).
        if isinstance(state, dict):
            if not isinstance(state.get("reward_funcs_results"), dict):
                state["reward_funcs_results"] = {}
            state["reward_funcs_results"][_reward_func.__name__] = reward

        return reward

    rubric = vf.Rubric()
    rubric.add_reward_func(_reward_func)

    # datasets.Dataset is required by SingleTurnEnv (a bare list raises
    # AttributeError: column_names).  We use a single placeholder row; the
    # actual rewards come from the external backend and this row is never read.
    import datasets as _datasets  # local import keeps top-level deps minimal

    placeholder_ds = _datasets.Dataset.from_list([{"prompt": "", "answer": ""}])

    # Pass rubric= to the CONSTRUCTOR so verifiers wraps it in a RubricGroup.
    # Assigning env.rubric = rubric after construction would bypass the wrapping
    # and leave env.rubric as a plain Rubric (breaking the RubricGroup contract).
    env = vf.SingleTurnEnv(dataset=placeholder_ds, rubric=rubric)

    # Wire group_size into sampling_args so prime-rl draws the right number of
    # completions per prompt.
    env.sampling_args["n"] = config.group_size

    return env
