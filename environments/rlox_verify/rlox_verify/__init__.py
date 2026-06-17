"""rlox_verify — verifiers environment package for rlox code-execution rollouts.

Wraps the rlox Baseline (``in_loop``) and Treatment (``rlox`` server) dispatch
into a standard ``verifiers`` environment that prime-rl can drive directly.

**Self-contained design**: the pure-Python backend helpers are vendored into
this package (``_backend.py``, ``_adversarial.py``) so that ``rlox_verify``
installs and runs without the Rust-extension ``rlox`` package being present in
the same venv.  The canonical implementations live in
``python/rlox/agentic/{verifiers_adapter,adversarial_corpus}.py`` — keep them
in sync when changing dispatch or corpus logic.

**Data-flow (verifiers 0.1.15.dev)**:

  dataset row → state["input"] → state["answer"]   (forwarded; unused here)
                               → state["input"]["tests"]  (extra column)

``Rubric._call_individual_reward_func`` calls ``score_objects(state)`` which
invokes ``task_score_fields``.  That method extracts every column NOT in
``TASK_INPUT_FIELDS = {"prompt", "answer", "info", "example_id"}`` and adds
them to the merged kwargs dict.  Because our dataset has a ``tests`` column,
it arrives as ``tests=`` when the reward function accepts ``**kwargs``.

**API reconciliation vs. 0.1.14**:

  * ``vf.Rubric()`` constructor: identical signature.
  * ``add_reward_func``: identical.
  * ``vf.SingleTurnEnv`` constructor: identical (``dataset=``, ``rubric=``).
  * ``score_objects`` now includes a ``task_score_fields`` pass that injects
    extra dataset columns as kwargs — this is NEW in 0.1.15.dev and is what
    makes the ``tests`` kwarg work without any special casing.
  * ``vf.Environment.__init__``: the ``dataset`` parameter now also accepts a
    callable ``DatasetBuilder`` (lazy).  We still pass eagerly-built datasets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import verifiers as vf

from rlox_verify._adversarial import AdversarialCorpus, AdversarialInjector, AdversarialSample
from rlox_verify._backend import call_rlox_server, extract_text, run_in_loop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RloxVerifyConfig:
    """Configuration for the rlox_verify environment.

    Mirrors ``rlox.agentic.verifiers_adapter.RloxVerifierConfig`` so that
    prime-rl can drive both the standalone environment package (this file) and
    the rlox-internal adapter with the same mental model.

    Attributes:
        rollout_backend: ``"in_loop"`` for Baseline (subprocess execution) or
            ``"rlox"`` for Treatment (POST to the Rust ``/verify`` server).
        rlox_server_url: Base URL of the Rust rollout server.  Unused when
            ``rollout_backend == "in_loop"``.
        per_sample_timeout_secs: Subprocess / HTTP timeout per rollout.
        group_size: Number of rollouts per example.
        adversarial_fraction: Fraction of tasks to replace with adversarial
            samples (0.0 = never, 1.0 = always).
        adversarial_corpus_path: Path to the ``adversarial_corpus_v1.json``
            file.  Required when ``adversarial_fraction > 0``.
        seed: Integer seed for the adversarial injector PRNG.
    """
    rollout_backend: str = "in_loop"
    rlox_server_url: str = "http://localhost:8080"
    per_sample_timeout_secs: float = 30.0
    group_size: int = 4
    adversarial_fraction: float = 0.0
    adversarial_corpus_path: str | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# Fixed coding dataset (8 MBPP-style problems — no external download needed)
# ---------------------------------------------------------------------------

#: Each row has ``prompt`` (user-facing string), ``answer`` (empty — kept for
#: verifiers compat), and ``tests`` (assert block executed against the model's
#: completion).
_CODING_PROBLEMS: list[dict[str, str]] = [
    {
        "prompt": (
            "Write a Python function `add(a, b)` that returns the sum of two numbers."
        ),
        "answer": "",
        "tests": (
            "assert add(1, 2) == 3\n"
            "assert add(-1, 1) == 0\n"
            "assert add(0, 0) == 0\n"
            "assert add(100, 200) == 300\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `is_even(n)` that returns True if n is even, "
            "False otherwise."
        ),
        "answer": "",
        "tests": (
            "assert is_even(2) == True\n"
            "assert is_even(3) == False\n"
            "assert is_even(0) == True\n"
            "assert is_even(-4) == True\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `reverse_string(s)` that returns the reverse "
            "of the input string."
        ),
        "answer": "",
        "tests": (
            "assert reverse_string('hello') == 'olleh'\n"
            "assert reverse_string('') == ''\n"
            "assert reverse_string('a') == 'a'\n"
            "assert reverse_string('abcd') == 'dcba'\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `count_vowels(s)` that counts the number of "
            "vowels (a, e, i, o, u, case-insensitive) in the string s."
        ),
        "answer": "",
        "tests": (
            "assert count_vowels('hello') == 2\n"
            "assert count_vowels('') == 0\n"
            "assert count_vowels('AEIOU') == 5\n"
            "assert count_vowels('rhythm') == 0\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `factorial(n)` that returns n! for non-negative "
            "integer n. You may assume n >= 0."
        ),
        "answer": "",
        "tests": (
            "assert factorial(0) == 1\n"
            "assert factorial(1) == 1\n"
            "assert factorial(5) == 120\n"
            "assert factorial(10) == 3628800\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `flatten(lst)` that takes a list of lists and "
            "returns a single flat list."
        ),
        "answer": "",
        "tests": (
            "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]\n"
            "assert flatten([[], [1], [2, 3]]) == [1, 2, 3]\n"
            "assert flatten([]) == []\n"
            "assert flatten([[5]]) == [5]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `is_palindrome(s)` that returns True if s is a "
            "palindrome (reads the same forwards and backwards), False otherwise. "
            "Comparison is case-sensitive."
        ),
        "answer": "",
        "tests": (
            "assert is_palindrome('racecar') == True\n"
            "assert is_palindrome('hello') == False\n"
            "assert is_palindrome('') == True\n"
            "assert is_palindrome('a') == True\n"
            "assert is_palindrome('Aba') == False\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `second_largest(nums)` that returns the second "
            "largest distinct value in the list. Assume len(nums) >= 2 and at least "
            "two distinct values exist."
        ),
        "answer": "",
        "tests": (
            "assert second_largest([1, 2, 3]) == 2\n"
            "assert second_largest([3, 1, 4, 1, 5, 9, 2, 6]) == 6\n"
            "assert second_largest([10, 10, 9]) == 9\n"
            "assert second_largest([5, 1]) == 1\n"
        ),
    },
]


def _build_dataset():  # returns datasets.Dataset; local import keeps top-level deps lazy
    """Return the fixed coding dataset as a ``datasets.Dataset``."""
    import datasets as _ds  # noqa: PLC0415 — local import

    ds = _ds.Dataset.from_list(_CODING_PROBLEMS)
    # Wrap prompt strings into message dicts that verifiers expects.
    return ds.map(
        lambda row: {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "answer": row["answer"],
            "tests": row["tests"],
        }
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def load_environment(
    *,
    rollout_backend: str = "in_loop",
    rlox_server_url: str = "http://localhost:8080",
    per_sample_timeout_secs: float = 30.0,
    group_size: int = 4,
    adversarial_fraction: float = 0.0,
    adversarial_corpus_path: str | None = None,
    seed: int = 42,
    dataset_name: str | None = None,
    n_problems: int | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Build a ``vf.Environment`` backed by the rlox Baseline or Treatment path.

    **How tests reach the reward function (verifiers 0.1.15.dev)**:

    The dataset has a ``tests`` column.  Verifiers stores the full dataset row
    in ``state["input"]``.  ``Rubric._call_individual_reward_func`` calls
    ``score_objects(state)`` which calls ``task_score_fields``.  That method
    adds all columns NOT in ``TASK_INPUT_FIELDS = {prompt, answer, info,
    example_id}`` to the kwargs dict.  Because the reward function here accepts
    ``tests: str = ""`` (explicit) and ``**kwargs``, ``tests`` arrives directly.

    Args:
        rollout_backend: ``"in_loop"`` (Baseline, subprocess) or ``"rlox"``
            (Treatment, POST to ``/verify``).
        rlox_server_url: Base URL of the rlox verify server.  Ignored when
            ``rollout_backend == "in_loop"``.
        per_sample_timeout_secs: Timeout for each execution / HTTP call.
        group_size: Number of rollouts per example
            (written to ``env.sampling_args["n"]``).
        adversarial_fraction: Fraction of tasks to replace with adversarial
            samples (0.0 = never, 1.0 = always).
        adversarial_corpus_path: Path to ``adversarial_corpus_v1.json``.
            Required when ``adversarial_fraction > 0``.
        seed: PRNG seed for the adversarial injector.
        dataset_name: Reserved for future HuggingFace dataset support;
            currently ignored.
        n_problems: If provided, truncate the fixed dataset to the first N rows.
        **kwargs: Silently ignored to stay compatible with prime-rl's
            ``vf.load_environment(env_args=...)`` call convention.

    Returns:
        A ``vf.SingleTurnEnv`` instance ready for prime-rl to drive.

    Raises:
        ValueError: if ``rollout_backend`` is not ``"in_loop"`` or ``"rlox"``.
        ValueError: if ``adversarial_fraction > 0`` but
            ``adversarial_corpus_path`` is ``None``.
    """
    _VALID_BACKENDS = {"in_loop", "rlox"}
    if rollout_backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid rollout_backend={rollout_backend!r}. "
            f"Must be one of {sorted(_VALID_BACKENDS)}."
        )

    if adversarial_fraction > 0 and adversarial_corpus_path is None:
        raise ValueError(
            "adversarial_corpus_path must be provided when adversarial_fraction > 0."
        )

    # Build injector (may be None when fraction == 0).
    injector: AdversarialInjector | None = None
    if adversarial_fraction > 0 and adversarial_corpus_path is not None:
        corpus = AdversarialCorpus.load(adversarial_corpus_path)
        injector = AdversarialInjector(
            corpus=corpus,
            fraction=adversarial_fraction,
            seed=seed,
        )

    # Capture for closure.
    _backend = rollout_backend
    _server_url = rlox_server_url
    _timeout = per_sample_timeout_secs

    def _reward_func(
        prompt: list[dict] | str,
        completion: list[dict] | str,
        answer: Any = "",
        state: dict | None = None,
        tests: str = "",
        **extra: Any,
    ) -> float:
        """Score one model completion against the problem's unit tests.

        **tests kwarg wiring (0.1.15.dev)**:
        ``task_score_fields`` in verifiers injects the ``tests`` dataset column
        as ``tests=`` here because: (a) the reward function explicitly declares
        ``tests: str = ""`` and (b) ``_call_individual_reward_func`` uses
        ``inspect.signature`` to detect both ``VAR_KEYWORD`` (**kwargs) and
        named params.  The ``tests`` column is NOT in TASK_INPUT_FIELDS so it
        is not filtered out.

        Dispatch:
          * Adversarial injection replaces task with an adversarial sample when
            the injector fires.  The sample's ``.code`` is executed with empty
            tests — it should time-out or error, returning 0.0.
          * ``rollout_backend == "in_loop"``: subprocess execution.
          * ``rollout_backend == "rlox"``: POST to ``/verify``.
        """
        # Build lightweight task dict for injector.
        task: Any = {"prompt": prompt, "answer": answer}
        is_adversarial = False

        if injector is not None:
            task, is_adversarial = injector.maybe_inject(task)

        # Extract code and tests from (possibly replaced) task.
        if isinstance(task, AdversarialSample):
            code_text = task.code
            tests_text = ""
        else:
            code_text = extract_text(completion)
            # Prefer the ``tests`` kwarg (dataset column) over ``answer``.
            # Fall back to ``answer`` for backward compat when ``tests`` is
            # empty (e.g. old datasets that embed tests in the answer field).
            effective_tests = tests if tests else (
                answer if isinstance(answer, str) else ""
            )
            tests_text = effective_tests

        if _backend == "rlox":
            reward = call_rlox_server(
                code_text, tests_text, is_adversarial, _server_url, _timeout
            )
        else:
            reward = run_in_loop(code_text, tests_text, _timeout)

        # Write per-function result into state for downstream inspection.
        if isinstance(state, dict):
            if not isinstance(state.get("reward_funcs_results"), dict):
                state["reward_funcs_results"] = {}
            state["reward_funcs_results"][_reward_func.__name__] = reward

        return reward

    rubric = vf.Rubric()
    rubric.add_reward_func(_reward_func)

    # Build the dataset.
    ds = _build_dataset()
    if n_problems is not None:
        ds = ds.select(range(min(n_problems, len(ds))))

    env = vf.SingleTurnEnv(dataset=ds, rubric=rubric)
    env.sampling_args["n"] = group_size

    return env
