"""rlox_verify — verifiers environment package for rlox code-execution rollouts.

Wraps the rlox Baseline (``in_loop``) and Treatment (``rlox`` server) dispatch
into a standard ``verifiers`` environment that prime-rl can drive directly.

**Single-source design**: the code-execution backend helpers and adversarial
corpus logic live in ``rlox_agent`` (``verifiers_adapter`` and
``adversarial_corpus`` modules) and are imported directly from there.  There
are no vendored copies in this package — ``rlox-agent`` is declared as a hard
dependency in ``pyproject.toml``.

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

**Dataset (MBPP)**:

  The environment loads a fixed, deterministic 50-problem slice of the MBPP
  benchmark (``datasets.load_dataset("mbpp", split="test")``, seeded selection
  with ``MBPP_SEED``).  MBPP problems are varied and genuinely harder than the
  original 8 hand-written problems, providing real learning headroom (base
  model reward well below 1.0).

  If MBPP is unavailable (no internet, dataset server down), the environment
  falls back to the embedded ``_FALLBACK_PROBLEMS`` list and logs a warning.
  Set ``RLOX_NO_MBPP=1`` to force the fallback without attempting a download.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import verifiers as vf

from rlox_agent.adversarial_corpus import (
    AdversarialCorpus,
    AdversarialInjector,
    AdversarialSample,
)
from rlox_agent.verifiers_adapter import (
    call_rlox_server,
    extract_python_code,
    extract_text,
    run_in_loop,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MBPP dataset parameters
# ---------------------------------------------------------------------------

#: Seed used to pick the deterministic MBPP slice.  Changing this seed changes
#: which problems are selected — do not change between sweep runs.
MBPP_SEED: int = 2024

#: Default number of MBPP problems to load.  Exposed so callers can override
#: via ``n_problems``.
MBPP_DEFAULT_N: int = 50

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
# Fallback coding dataset (used when MBPP is unavailable)
# ---------------------------------------------------------------------------

#: Each row has ``prompt`` (user-facing string), ``answer`` (empty — kept for
#: verifiers compat), and ``tests`` (assert block executed against the model's
#: completion).  This fallback is used when MBPP cannot be downloaded.
_FALLBACK_PROBLEMS: list[dict[str, str]] = [
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
    {
        "prompt": (
            "Write a Python function `sum_digits(n)` that returns the sum of all "
            "digits of a non-negative integer n."
        ),
        "answer": "",
        "tests": (
            "assert sum_digits(0) == 0\n"
            "assert sum_digits(9) == 9\n"
            "assert sum_digits(123) == 6\n"
            "assert sum_digits(999) == 27\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `is_prime(n)` that returns True if n is a prime "
            "number, False otherwise. n >= 2."
        ),
        "answer": "",
        "tests": (
            "assert is_prime(2) == True\n"
            "assert is_prime(3) == True\n"
            "assert is_prime(4) == False\n"
            "assert is_prime(17) == True\n"
            "assert is_prime(100) == False\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `rotate_list(lst, k)` that rotates the list "
            "to the right by k positions."
        ),
        "answer": "",
        "tests": (
            "assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]\n"
            "assert rotate_list([1, 2, 3], 0) == [1, 2, 3]\n"
            "assert rotate_list([1, 2, 3], 3) == [1, 2, 3]\n"
            "assert rotate_list([1], 5) == [1]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `word_count(sentence)` that returns a dict "
            "mapping each word (case-insensitive, split on whitespace) to its "
            "count of occurrences."
        ),
        "answer": "",
        "tests": (
            "assert word_count('hello world') == {'hello': 1, 'world': 1}\n"
            "assert word_count('the cat sat on the mat') == "
            "{'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1}\n"
            "assert word_count('') == {}\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `common_elements(a, b)` that returns a sorted "
            "list of elements that appear in both lists a and b (no duplicates)."
        ),
        "answer": "",
        "tests": (
            "assert common_elements([1, 2, 3], [2, 3, 4]) == [2, 3]\n"
            "assert common_elements([1, 2, 2, 3], [2, 2, 4]) == [2]\n"
            "assert common_elements([], [1, 2]) == []\n"
            "assert common_elements([5, 6], [5, 6]) == [5, 6]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `matrix_transpose(matrix)` that returns the "
            "transpose of a 2D list (list of lists). All rows are the same length."
        ),
        "answer": "",
        "tests": (
            "assert matrix_transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\n"
            "assert matrix_transpose([[1]]) == [[1]]\n"
            "assert matrix_transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `longest_common_prefix(strs)` that returns "
            "the longest common prefix string amongst a list of strings. Return "
            "an empty string if no common prefix exists."
        ),
        "answer": "",
        "tests": (
            "assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'\n"
            "assert longest_common_prefix(['dog', 'racecar', 'car']) == ''\n"
            "assert longest_common_prefix(['interview', 'interact', 'interface']) == 'inter'\n"
            "assert longest_common_prefix(['abc']) == 'abc'\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `running_sum(nums)` that returns a list where "
            "each element is the running (cumulative) sum of the original list."
        ),
        "answer": "",
        "tests": (
            "assert running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]\n"
            "assert running_sum([1]) == [1]\n"
            "assert running_sum([3, 1, 2, 10, 1]) == [3, 4, 6, 16, 17]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `remove_duplicates(lst)` that returns a new "
            "list with duplicates removed, preserving the original order of first "
            "appearances."
        ),
        "answer": "",
        "tests": (
            "assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]\n"
            "assert remove_duplicates([]) == []\n"
            "assert remove_duplicates([5, 5, 5]) == [5]\n"
            "assert remove_duplicates([1, 2, 3]) == [1, 2, 3]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `chunk_list(lst, n)` that splits lst into "
            "consecutive chunks of size n. The last chunk may be smaller."
        ),
        "answer": "",
        "tests": (
            "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
            "assert chunk_list([1, 2, 3], 3) == [[1, 2, 3]]\n"
            "assert chunk_list([], 3) == []\n"
            "assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `count_pairs(nums, target)` that returns the "
            "number of pairs (i, j) with i < j such that nums[i] + nums[j] == target."
        ),
        "answer": "",
        "tests": (
            "assert count_pairs([1, 2, 3, 4], 5) == 2\n"
            "assert count_pairs([1, 1, 1], 2) == 3\n"
            "assert count_pairs([1, 2, 3], 10) == 0\n"
            "assert count_pairs([], 0) == 0\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `most_frequent(lst)` that returns the element "
            "that appears most frequently in the list. If there is a tie, return "
            "the one that appears first in the list."
        ),
        "answer": "",
        "tests": (
            "assert most_frequent([1, 2, 2, 3, 3, 3]) == 3\n"
            "assert most_frequent(['a', 'b', 'a']) == 'a'\n"
            "assert most_frequent([1]) == 1\n"
            "assert most_frequent([3, 1, 3, 2, 1]) == 3\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `merge_sorted(a, b)` that merges two sorted "
            "lists a and b into a single sorted list."
        ),
        "answer": "",
        "tests": (
            "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\n"
            "assert merge_sorted([], [1, 2]) == [1, 2]\n"
            "assert merge_sorted([1, 2], []) == [1, 2]\n"
            "assert merge_sorted([1, 1, 2], [1, 3]) == [1, 1, 1, 2, 3]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `find_missing(nums)` that finds the single "
            "missing integer in a list containing n distinct integers in the range "
            "[1, n+1]."
        ),
        "answer": "",
        "tests": (
            "assert find_missing([1, 2, 4, 5]) == 3\n"
            "assert find_missing([2]) == 1\n"
            "assert find_missing([1, 2, 3, 5]) == 4\n"
            "assert find_missing([1]) == 2\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `group_anagrams(words)` that groups a list "
            "of strings into lists of anagrams. Each group is sorted internally, "
            "and the outer list is sorted by the first element of each group."
        ),
        "answer": "",
        "tests": (
            "assert group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']) == "
            "[['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]\n"
            "assert group_anagrams(['']) == [['']]\n"
            "assert group_anagrams(['a']) == [['a']]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `product_except_self(nums)` that returns a "
            "list where each element is the product of all other elements in nums. "
            "Do not use division."
        ),
        "answer": "",
        "tests": (
            "assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]\n"
            "assert product_except_self([1, 1, 1, 1]) == [1, 1, 1, 1]\n"
            "assert product_except_self([2, 3]) == [3, 2]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `binary_search(arr, target)` that returns the "
            "index of target in the sorted list arr, or -1 if not found."
        ),
        "answer": "",
        "tests": (
            "assert binary_search([1, 3, 5, 7, 9], 5) == 2\n"
            "assert binary_search([1, 3, 5, 7, 9], 6) == -1\n"
            "assert binary_search([1], 1) == 0\n"
            "assert binary_search([1, 2, 3, 4, 5], 1) == 0\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `first_unique(s)` that returns the first "
            "non-repeating character in the string s, or None if all characters "
            "repeat."
        ),
        "answer": "",
        "tests": (
            "assert first_unique('leetcode') == 'l'\n"
            "assert first_unique('aabb') is None\n"
            "assert first_unique('z') == 'z'\n"
            "assert first_unique('aabbc') == 'c'\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `max_subarray_sum(nums)` that returns the "
            "maximum sum of a contiguous subarray (Kadane's algorithm)."
        ),
        "answer": "",
        "tests": (
            "assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\n"
            "assert max_subarray_sum([1]) == 1\n"
            "assert max_subarray_sum([-1, -2, -3]) == -1\n"
            "assert max_subarray_sum([5, 4, -1, 7, 8]) == 23\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `deep_flatten(lst)` that recursively flattens "
            "a nested list of arbitrary depth."
        ),
        "answer": "",
        "tests": (
            "assert deep_flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]\n"
            "assert deep_flatten([]) == []\n"
            "assert deep_flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]\n"
            "assert deep_flatten([1]) == [1]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `caesar_cipher(text, shift)` that encodes the "
            "string text using a Caesar cipher with the given shift. Preserve case "
            "and leave non-alphabetic characters unchanged."
        ),
        "answer": "",
        "tests": (
            "assert caesar_cipher('Hello, World!', 3) == 'Khoor, Zruog!'\n"
            "assert caesar_cipher('abc', 1) == 'bcd'\n"
            "assert caesar_cipher('xyz', 3) == 'abc'\n"
            "assert caesar_cipher('ABC', 26) == 'ABC'\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `count_occurrences(text, word)` that returns "
            "the number of times word appears in text (case-insensitive, "
            "space-delimited)."
        ),
        "answer": "",
        "tests": (
            "assert count_occurrences('the cat sat on the mat', 'the') == 2\n"
            "assert count_occurrences('Hello hello HELLO', 'hello') == 3\n"
            "assert count_occurrences('foo bar baz', 'qux') == 0\n"
            "assert count_occurrences('one', 'one') == 1\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `zip_lists(*lists)` that zips multiple lists "
            "together, stopping at the shortest. Returns a list of tuples."
        ),
        "answer": "",
        "tests": (
            "assert zip_lists([1, 2, 3], ['a', 'b', 'c']) == [(1, 'a'), (2, 'b'), (3, 'c')]\n"
            "assert zip_lists([1, 2], ['a']) == [(1, 'a')]\n"
            "assert zip_lists([]) == []\n"
            "assert zip_lists([1, 2], [3, 4], [5, 6]) == [(1, 3, 5), (2, 4, 6)]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci "
            "number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...)."
        ),
        "answer": "",
        "tests": (
            "assert fibonacci(0) == 0\n"
            "assert fibonacci(1) == 1\n"
            "assert fibonacci(6) == 8\n"
            "assert fibonacci(10) == 55\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `count_set_bits(n)` that returns the number of "
            "1-bits (set bits) in the binary representation of a non-negative integer n."
        ),
        "answer": "",
        "tests": (
            "assert count_set_bits(0) == 0\n"
            "assert count_set_bits(7) == 3\n"
            "assert count_set_bits(128) == 1\n"
            "assert count_set_bits(255) == 8\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `flatten_dict(d, parent_key='', sep='.')` that "
            "flattens a nested dict by joining keys with sep. Values that are dicts "
            "are recursed into; other values are kept as-is."
        ),
        "answer": "",
        "tests": (
            "assert flatten_dict({'a': {'b': 1, 'c': 2}}) == {'a.b': 1, 'a.c': 2}\n"
            "assert flatten_dict({'x': 1}) == {'x': 1}\n"
            "assert flatten_dict({'a': {'b': {'c': 3}}}) == {'a.b.c': 3}\n"
            "assert flatten_dict({}) == {}\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `top_k(nums, k)` that returns the k largest "
            "elements from the list in descending order."
        ),
        "answer": "",
        "tests": (
            "assert top_k([3, 1, 4, 1, 5, 9, 2, 6], 3) == [9, 6, 5]\n"
            "assert top_k([1, 2, 3], 1) == [3]\n"
            "assert top_k([1], 1) == [1]\n"
            "assert top_k([5, 4, 3, 2, 1], 5) == [5, 4, 3, 2, 1]\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `is_anagram(s, t)` that returns True if "
            "string s is an anagram of string t, False otherwise. "
            "Comparison is case-sensitive."
        ),
        "answer": "",
        "tests": (
            "assert is_anagram('anagram', 'nagaram') == True\n"
            "assert is_anagram('rat', 'car') == False\n"
            "assert is_anagram('', '') == True\n"
            "assert is_anagram('abc', 'ab') == False\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `count_substring(s, sub)` that returns the "
            "number of non-overlapping occurrences of sub in s."
        ),
        "answer": "",
        "tests": (
            "assert count_substring('hello world hello', 'hello') == 2\n"
            "assert count_substring('aaaa', 'aa') == 2\n"
            "assert count_substring('abcde', 'xy') == 0\n"
            "assert count_substring('', 'a') == 0\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `gcd(a, b)` that returns the greatest common "
            "divisor of non-negative integers a and b."
        ),
        "answer": "",
        "tests": (
            "assert gcd(48, 18) == 6\n"
            "assert gcd(7, 5) == 1\n"
            "assert gcd(0, 5) == 5\n"
            "assert gcd(100, 75) == 25\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `lcm(a, b)` that returns the least common "
            "multiple of positive integers a and b."
        ),
        "answer": "",
        "tests": (
            "assert lcm(4, 6) == 12\n"
            "assert lcm(3, 7) == 21\n"
            "assert lcm(1, 5) == 5\n"
            "assert lcm(12, 18) == 36\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `flatten_and_sort(lst_of_lsts)` that flattens "
            "a list of lists and returns a sorted list of all elements."
        ),
        "answer": "",
        "tests": (
            "assert flatten_and_sort([[3, 1], [4, 1, 5], [9, 2, 6]]) == [1, 1, 2, 3, 4, 5, 6, 9]\n"
            "assert flatten_and_sort([[5], [], [3, 1]]) == [1, 3, 5]\n"
            "assert flatten_and_sort([[]]) == []\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `valid_parentheses(s)` that returns True if "
            "the string s consisting of '(', ')', '{', '}', '[', ']' is valid "
            "(every opening bracket is closed in the correct order)."
        ),
        "answer": "",
        "tests": (
            "assert valid_parentheses('()[]{}') == True\n"
            "assert valid_parentheses('([)]') == False\n"
            "assert valid_parentheses('{[]}') == True\n"
            "assert valid_parentheses('') == True\n"
            "assert valid_parentheses(']') == False\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `capitalize_words(sentence)` that capitalizes "
            "the first letter of each word in the sentence (title case)."
        ),
        "answer": "",
        "tests": (
            "assert capitalize_words('hello world') == 'Hello World'\n"
            "assert capitalize_words('the quick brown fox') == 'The Quick Brown Fox'\n"
            "assert capitalize_words('') == ''\n"
            "assert capitalize_words('a') == 'A'\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `unique_paths(m, n)` that returns the number "
            "of unique paths from the top-left to the bottom-right of an m×n grid, "
            "moving only right or down."
        ),
        "answer": "",
        "tests": (
            "assert unique_paths(3, 7) == 28\n"
            "assert unique_paths(3, 2) == 3\n"
            "assert unique_paths(1, 1) == 1\n"
            "assert unique_paths(2, 2) == 2\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `intersection(a, b)` that returns a sorted list "
            "of elements that appear in both sets a and b (as lists). No duplicates."
        ),
        "answer": "",
        "tests": (
            "assert intersection([1, 2, 3, 4], [3, 4, 5, 6]) == [3, 4]\n"
            "assert intersection([1, 2], [3, 4]) == []\n"
            "assert intersection([1, 1, 2], [1, 3]) == [1]\n"
            "assert intersection([], [1, 2]) == []\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `max_product_pair(nums)` that returns the "
            "maximum product of any two elements in nums."
        ),
        "answer": "",
        "tests": (
            "assert max_product_pair([3, 4, 5, 2]) == 20\n"
            "assert max_product_pair([-10, -3, 5, 6]) == 30\n"
            "assert max_product_pair([1, 2]) == 2\n"
            "assert max_product_pair([-1, -2, -3]) == 6\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `reverse_words(sentence)` that reverses the "
            "order of words in a sentence (split on spaces, rejoin with spaces)."
        ),
        "answer": "",
        "tests": (
            "assert reverse_words('hello world') == 'world hello'\n"
            "assert reverse_words('the quick brown fox') == 'fox brown quick the'\n"
            "assert reverse_words('single') == 'single'\n"
            "assert reverse_words('a b c') == 'c b a'\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `digit_frequency(n)` that returns a dict "
            "mapping each digit character to its count of occurrences in the "
            "string representation of positive integer n."
        ),
        "answer": "",
        "tests": (
            "assert digit_frequency(1122334) == {'1': 2, '2': 2, '3': 2, '4': 1}\n"
            "assert digit_frequency(5) == {'5': 1}\n"
            "assert digit_frequency(111) == {'1': 3}\n"
            "assert digit_frequency(1234567890) == "
            "{'1':1,'2':1,'3':1,'4':1,'5':1,'6':1,'7':1,'8':1,'9':1,'0':1}\n"
        ),
    },
    {
        "prompt": (
            "Write a Python function `power(base, exp)` that returns base raised "
            "to the power exp without using the ** operator or built-in pow(). "
            "exp is a non-negative integer."
        ),
        "answer": "",
        "tests": (
            "assert power(2, 10) == 1024\n"
            "assert power(3, 3) == 27\n"
            "assert power(5, 0) == 1\n"
            "assert power(1, 100) == 1\n"
        ),
    },
]

# Keep a backward-compatible name so trl_grpo_run.py can still import it.
# trl_grpo_run will be updated to use _load_mbpp_problems() instead, but
# this alias prevents import errors during the transition.
_CODING_PROBLEMS = _FALLBACK_PROBLEMS


# ---------------------------------------------------------------------------
# MBPP dataset loader
# ---------------------------------------------------------------------------


def _load_mbpp_problems(
    n: int = MBPP_DEFAULT_N,
    seed: int = MBPP_SEED,
) -> list[dict[str, str]]:
    """Load a fixed, deterministic slice of MBPP as rlox_verify problem dicts.

    Attempts to download/load ``google-research-datasets/mbpp`` (split="test").
    Falls back to ``_FALLBACK_PROBLEMS`` if the dataset is unavailable or if
    the env var ``RLOX_NO_MBPP=1`` is set.

    Args:
        n: Number of problems to select (default ``MBPP_DEFAULT_N = 50``).
        seed: RNG seed for the deterministic selection (default ``MBPP_SEED``).

    Returns:
        A list of dicts with keys ``"prompt"``, ``"answer"``, ``"tests"``.
    """
    if os.environ.get("RLOX_NO_MBPP", "").strip() == "1":
        logger.warning(
            "RLOX_NO_MBPP=1: using %d fallback problems instead of MBPP",
            len(_FALLBACK_PROBLEMS),
        )
        return _FALLBACK_PROBLEMS[:n]

    try:
        import datasets as _ds  # noqa: PLC0415

        # Try canonical HuggingFace dataset name first, then google-research alias.
        ds = None
        for _name in ("mbpp", "google-research-datasets/mbpp"):
            try:
                ds = _ds.load_dataset(_name, split="test", trust_remote_code=False)
                logger.info("Loaded MBPP from %r (%d rows total)", _name, len(ds))
                break
            except Exception:
                continue

        if ds is None:
            raise RuntimeError("MBPP unavailable under any known dataset name")

        # Deterministic shuffle then slice.
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected = indices[:n]

        problems: list[dict[str, str]] = []
        for idx in selected:
            row = ds[idx]
            # MBPP "test_list" is a list of assert strings; join into one block.
            test_lines: list[str] = row.get("test_list") or []
            tests_block = "\n".join(test_lines) + "\n" if test_lines else ""
            # "text" is the natural-language problem description.
            prompt_text: str = row.get("text", "")
            problems.append(
                {
                    "prompt": (
                        f"Write a Python function to solve the following problem:\n\n"
                        f"{prompt_text}"
                    ),
                    "answer": "",
                    "tests": tests_block,
                }
            )

        logger.info(
            "MBPP slice: %d problems selected (seed=%d, total_pool=%d)",
            len(problems),
            seed,
            len(ds),
        )
        return problems

    except Exception as exc:
        logger.warning(
            "MBPP load failed (%s); falling back to %d embedded problems",
            exc,
            len(_FALLBACK_PROBLEMS),
        )
        return _FALLBACK_PROBLEMS[:n]


def _build_dataset(
    n_problems: int | None = None,
    use_mbpp: bool = True,
):  # returns datasets.Dataset; local import keeps top-level deps lazy
    """Return the coding dataset as a ``datasets.Dataset``.

    Loads MBPP by default (``use_mbpp=True``).  Falls back to
    ``_FALLBACK_PROBLEMS`` when MBPP is unavailable.

    Args:
        n_problems: If provided, truncate to the first N rows after loading.
        use_mbpp: If False, skip MBPP and use ``_FALLBACK_PROBLEMS`` directly.
    """
    import datasets as _ds  # noqa: PLC0415 — local import

    if use_mbpp:
        raw_problems = _load_mbpp_problems(
            n=n_problems if n_problems is not None else MBPP_DEFAULT_N
        )
    else:
        raw_problems = _FALLBACK_PROBLEMS
        if n_problems is not None:
            raw_problems = raw_problems[:n_problems]

    ds = _ds.Dataset.from_list(raw_problems)
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
            code_text = extract_python_code(extract_text(completion))
            # Prefer the ``tests`` kwarg (dataset column) over ``answer``.
            # Fall back to ``answer`` for backward compat when ``tests`` is
            # empty (e.g. old datasets that embed tests in the answer field).
            effective_tests = (
                tests if tests else (answer if isinstance(answer, str) else "")
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

    # Build the dataset (MBPP slice by default; fallback to embedded problems
    # when MBPP is unavailable or RLOX_NO_MBPP=1).
    ds = _build_dataset(n_problems=n_problems)

    env = vf.SingleTurnEnv(dataset=ds, rubric=rubric)
    env.sampling_args["n"] = group_size

    return env
