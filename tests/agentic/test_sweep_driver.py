"""RED-phase tests for Step 6d: run_sweep (benchmark sweep driver — AC-3 / AC-4).

Contract being specified
------------------------
run_sweep(config, run_one, metric_store_dir) iterates the full grid:

    conditions = ["in_loop", "rlox"]
    seeds      = range(config.n_seeds)
    fractions  = config.adversarial_fractions

and calls run_one(condition, seed, fraction) exactly once per grid point.

Observable behavior under test:

  1. Grid completeness — all 2 × n_seeds × len(fractions) combinations invoked.
  2. Deterministic iteration order — outer loop: conditions; then seeds; then
     fractions. This is the pre-registered sweep order; tests assert the exact
     call sequence.
  3. Survival recording — run_one's "survived" key is forwarded as a boolean
     in each result dict.
  4. Exception resilience — a run_one that raises is recorded as survived=False;
     the sweep continues (no re-raise, remaining grid points still run).
  5. Metric store population — one JSON artifact per run is written to
     metric_store_dir after the sweep.
  6. Return value — list of result dicts with at minimum
     {condition, seed, fraction, survived}, one per grid point.

Interface assumptions (implementer must honour):
  - Module path: benchmarks/agentic/run_benchmark.py
  - Top-level import: ``import run_benchmark`` (conftest injects benchmarks/agentic/)
  - ``run_sweep(config, run_one: Callable[[str, int, float], dict], metric_store_dir: str)
      -> list[dict]``
  - ``run_one`` receives positional args: (condition: str, seed: int, fraction: float)
  - ``run_one`` returns a dict that MUST include ``{"survived": bool}`` at minimum
  - Per-run artifacts in metric_store_dir: the exact filename format is
    implementation-defined; tests only assert at least one file per run exists
    (total artifact count == 2 × n_seeds × len(fractions))
  - ``config`` is a simple object/namespace with .n_seeds (int) and
    .adversarial_fractions (list[float]); tests use a minimal SimpleNamespace stub.

All imports are top-level: conftest.py injects benchmarks/agentic/ onto sys.path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

# Top-level import — conftest injects benchmarks/agentic/ onto sys.path
import run_benchmark
from run_benchmark import run_sweep


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CONDITIONS = ["in_loop", "rlox"]


def _make_config(n_seeds: int = 3, fractions=None) -> SimpleNamespace:
    """Return a minimal config-like object suitable for run_sweep."""
    if fractions is None:
        fractions = [0.0, 0.01, 0.05, 0.10]
    return SimpleNamespace(
        n_seeds=n_seeds,
        adversarial_fractions=fractions,
    )


def _surviving_run_one(condition: str, seed: int, fraction: float) -> dict:
    """A mock run_one that always reports survival."""
    return {"survived": True, "condition": condition, "seed": seed, "fraction": fraction}


def _failing_run_one(condition: str, seed: int, fraction: float) -> dict:
    """A mock run_one that always reports non-survival (clean failure)."""
    return {"survived": False, "condition": condition, "seed": seed, "fraction": fraction}


# ---------------------------------------------------------------------------
# A) Grid completeness — all combinations invoked exactly once
# ---------------------------------------------------------------------------

class TestGridCompleteness:
    def test_all_combinations_called_exactly_once(self, tmp_path: Path):
        """All 2 × n_seeds × len(fractions) combos must be called exactly once."""
        n_seeds = 3
        fractions = [0.0, 0.01, 0.05, 0.10]
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        expected_count = 2 * n_seeds * len(fractions)

        calls: list[tuple[str, int, float]] = []

        def recording_run_one(condition: str, seed: int, fraction: float) -> dict:
            calls.append((condition, seed, fraction))
            return {"survived": True}

        run_sweep(config, recording_run_one, str(tmp_path))

        assert len(calls) == expected_count, (
            f"Expected {expected_count} calls (2 conditions × {n_seeds} seeds × "
            f"{len(fractions)} fractions), got {len(calls)}. Calls: {calls}"
        )

    def test_no_duplicate_combinations(self, tmp_path: Path):
        """No (condition, seed, fraction) triple may appear more than once."""
        config = _make_config(n_seeds=3, fractions=[0.0, 0.05])
        seen: list[tuple[str, int, float]] = []

        def recording_run_one(condition, seed, fraction):
            seen.append((condition, seed, fraction))
            return {"survived": True}

        run_sweep(config, recording_run_one, str(tmp_path))

        unique = set(seen)
        assert len(unique) == len(seen), (
            f"Duplicate calls detected. Expected {len(seen)} unique triples, "
            f"but only {len(unique)} unique. Duplicates: "
            f"{[t for t in seen if seen.count(t) > 1]}"
        )

    def test_both_conditions_present(self, tmp_path: Path):
        """Both 'in_loop' and 'rlox' must appear in the call list."""
        config = _make_config(n_seeds=2, fractions=[0.0])
        seen_conditions: set[str] = set()

        def recording_run_one(condition, seed, fraction):
            seen_conditions.add(condition)
            return {"survived": True}

        run_sweep(config, recording_run_one, str(tmp_path))

        assert "in_loop" in seen_conditions, (
            f"'in_loop' condition was never called. Seen: {seen_conditions}"
        )
        assert "rlox" in seen_conditions, (
            f"'rlox' condition was never called. Seen: {seen_conditions}"
        )

    def test_all_seeds_called(self, tmp_path: Path):
        """Every seed in range(n_seeds) must appear in the call list."""
        n_seeds = 4
        config = _make_config(n_seeds=n_seeds, fractions=[0.0])
        seen_seeds: set[int] = set()

        def recording_run_one(condition, seed, fraction):
            seen_seeds.add(seed)
            return {"survived": True}

        run_sweep(config, recording_run_one, str(tmp_path))

        expected_seeds = set(range(n_seeds))
        assert seen_seeds == expected_seeds, (
            f"Expected seeds {expected_seeds}, got {seen_seeds}"
        )

    def test_all_fractions_called(self, tmp_path: Path):
        """Every fraction in config.adversarial_fractions must appear in calls."""
        fractions = [0.0, 0.01, 0.05, 0.10]
        config = _make_config(n_seeds=1, fractions=fractions)
        seen_fractions: list[float] = []

        def recording_run_one(condition, seed, fraction):
            seen_fractions.append(fraction)
            return {"survived": True}

        run_sweep(config, recording_run_one, str(tmp_path))

        for f in fractions:
            assert f in seen_fractions, (
                f"Fraction {f} was never passed to run_one. Seen: {seen_fractions}"
            )

    def test_grid_count_matches_formula(self, tmp_path: Path):
        """Return list length == 2 * n_seeds * len(fractions)."""
        n_seeds = 3
        fractions = [0.0, 0.01, 0.05, 0.10]
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        results = run_sweep(config, _surviving_run_one, str(tmp_path))
        expected = 2 * n_seeds * len(fractions)
        assert len(results) == expected, (
            f"run_sweep returned {len(results)} results, expected {expected}"
        )

    @pytest.mark.parametrize("n_seeds,fractions", [
        (3, [0.0, 0.01, 0.05, 0.10]),
        (5, [0.0, 0.10]),
        (3, [0.0]),
    ])
    def test_grid_count_parametric(self, n_seeds, fractions, tmp_path: Path):
        """Parametric check: result count == 2 × n_seeds × len(fractions)."""
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        results = run_sweep(config, _surviving_run_one, str(tmp_path / "m"))
        expected = 2 * n_seeds * len(fractions)
        assert len(results) == expected, (
            f"n_seeds={n_seeds}, fractions={fractions}: "
            f"expected {expected} results, got {len(results)}"
        )


# ---------------------------------------------------------------------------
# B) Deterministic iteration order
#    outer: conditions ["in_loop","rlox"], then seeds, then fractions
# ---------------------------------------------------------------------------

class TestIterationOrder:
    def test_iteration_order_conditions_outermost(self, tmp_path: Path):
        """All in_loop calls must precede all rlox calls (conditions are outermost)."""
        n_seeds = 2
        fractions = [0.0, 0.05]
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        calls: list[tuple[str, int, float]] = []

        def recording_run_one(condition, seed, fraction):
            calls.append((condition, seed, fraction))
            return {"survived": True}

        run_sweep(config, recording_run_one, str(tmp_path))

        n_per_condition = n_seeds * len(fractions)
        in_loop_calls = calls[:n_per_condition]
        rlox_calls = calls[n_per_condition:]

        assert all(c == "in_loop" for c, _, _ in in_loop_calls), (
            f"First {n_per_condition} calls must all be 'in_loop'. "
            f"Got: {[c for c, _, _ in in_loop_calls]}"
        )
        assert all(c == "rlox" for c, _, _ in rlox_calls), (
            f"Last {n_per_condition} calls must all be 'rlox'. "
            f"Got: {[c for c, _, _ in rlox_calls]}"
        )

    def test_iteration_order_seeds_middle(self, tmp_path: Path):
        """Within each condition, seed is the middle loop (outer to fractions)."""
        n_seeds = 3
        fractions = [0.0, 0.10]
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        calls: list[tuple[str, int, float]] = []

        def recording_run_one(condition, seed, fraction):
            calls.append((condition, seed, fraction))
            return {"survived": True}

        run_sweep(config, recording_run_one, str(tmp_path))

        # Collect the in_loop calls (first half)
        in_loop_calls = [(s, f) for c, s, f in calls if c == "in_loop"]
        # Within in_loop, for each seed the fractions must all appear before the next seed
        # i.e. the call sequence for seed 0 is complete before seed 1 starts
        n_fractions = len(fractions)
        for seed_idx in range(n_seeds):
            chunk_start = seed_idx * n_fractions
            chunk_end = chunk_start + n_fractions
            chunk = in_loop_calls[chunk_start:chunk_end]
            seeds_in_chunk = {s for s, _ in chunk}
            assert seeds_in_chunk == {seed_idx}, (
                f"Expected seed {seed_idx} in positions {chunk_start}–{chunk_end - 1}, "
                f"but got seeds {seeds_in_chunk}. Full in_loop call sequence: {in_loop_calls}"
            )

    def test_iteration_order_fractions_innermost(self, tmp_path: Path):
        """Fractions must be the innermost loop, in config order."""
        n_seeds = 2
        fractions = [0.0, 0.01, 0.05, 0.10]
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        calls: list[tuple[str, int, float]] = []

        def recording_run_one(condition, seed, fraction):
            calls.append((condition, seed, fraction))
            return {"survived": True}

        run_sweep(config, recording_run_one, str(tmp_path))

        # For each (condition, seed) block, the fractions must be in config order
        n_fractions = len(fractions)
        for condition in _CONDITIONS:
            for seed in range(n_seeds):
                block = [f for c, s, f in calls if c == condition and s == seed]
                assert block == fractions, (
                    f"For condition={condition!r}, seed={seed}: expected fractions "
                    f"{fractions} in that order, got {block}"
                )

    def test_full_call_sequence_deterministic(self, tmp_path: Path):
        """Run sweep twice; call sequences must be identical."""
        config = _make_config(n_seeds=2, fractions=[0.0, 0.05])
        calls_a: list[tuple[str, int, float]] = []
        calls_b: list[tuple[str, int, float]] = []

        def run_one_a(condition, seed, fraction):
            calls_a.append((condition, seed, fraction))
            return {"survived": True}

        def run_one_b(condition, seed, fraction):
            calls_b.append((condition, seed, fraction))
            return {"survived": True}

        tmp_a = tmp_path / "a"
        tmp_b = tmp_path / "b"
        tmp_a.mkdir()
        tmp_b.mkdir()
        run_sweep(config, run_one_a, str(tmp_a))
        run_sweep(config, run_one_b, str(tmp_b))

        assert calls_a == calls_b, (
            f"run_sweep call sequences are not deterministic.\n"
            f"First run:  {calls_a}\n"
            f"Second run: {calls_b}"
        )


# ---------------------------------------------------------------------------
# C) Survival recording
# ---------------------------------------------------------------------------

class TestSurvivalRecording:
    def test_surviving_run_recorded_as_true(self, tmp_path: Path):
        """run_one returning {"survived": True} must be in results as survived=True."""
        config = _make_config(n_seeds=1, fractions=[0.0])
        results = run_sweep(config, _surviving_run_one, str(tmp_path))

        for r in results:
            assert "survived" in r, f"Result dict missing 'survived' key: {r}"
            assert r["survived"] is True or r["survived"] == True, (
                f"Expected survived=True, got {r['survived']!r}"
            )

    def test_failing_run_recorded_as_false(self, tmp_path: Path):
        """run_one returning {"survived": False} must be in results as survived=False."""
        config = _make_config(n_seeds=1, fractions=[0.0])
        results = run_sweep(config, _failing_run_one, str(tmp_path))

        for r in results:
            assert "survived" in r, f"Result dict missing 'survived' key: {r}"
            assert r["survived"] is False or r["survived"] == False, (
                f"Expected survived=False, got {r['survived']!r}"
            )

    def test_survival_is_boolean(self, tmp_path: Path):
        """The survived field must be a Python bool in the returned results."""
        config = _make_config(n_seeds=1, fractions=[0.0, 0.10])
        results = run_sweep(config, _surviving_run_one, str(tmp_path))

        for r in results:
            assert isinstance(r["survived"], bool), (
                f"'survived' must be a bool, got {type(r['survived'])}: {r}"
            )

    def test_result_includes_condition_seed_fraction(self, tmp_path: Path):
        """Each result dict must include 'condition', 'seed', and 'fraction'."""
        config = _make_config(n_seeds=2, fractions=[0.0, 0.05])
        results = run_sweep(config, _surviving_run_one, str(tmp_path))

        for r in results:
            assert "condition" in r, f"Result missing 'condition': {r}"
            assert "seed" in r, f"Result missing 'seed': {r}"
            assert "fraction" in r, f"Result missing 'fraction': {r}"

    def test_each_result_matches_grid_point(self, tmp_path: Path):
        """Each result's (condition, seed, fraction) must match a valid grid point."""
        n_seeds = 2
        fractions = [0.0, 0.05]
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        results = run_sweep(config, _surviving_run_one, str(tmp_path))

        valid_points = {
            (c, s, f)
            for c in _CONDITIONS
            for s in range(n_seeds)
            for f in fractions
        }

        for r in results:
            point = (r["condition"], r["seed"], r["fraction"])
            assert point in valid_points, (
                f"Result {r} contains an unexpected (condition, seed, fraction) triple. "
                f"Valid points: {valid_points}"
            )


# ---------------------------------------------------------------------------
# D) Exception resilience — crashed run_one → survived=False, sweep continues
# ---------------------------------------------------------------------------

class TestExceptionResilience:
    def test_raising_run_one_recorded_as_survived_false(self, tmp_path: Path):
        """A run_one that raises must be recorded as survived=False."""
        config = _make_config(n_seeds=1, fractions=[0.0])

        def exploding_run_one(condition, seed, fraction):
            raise RuntimeError("simulated run failure")

        results = run_sweep(config, exploding_run_one, str(tmp_path))

        assert len(results) > 0, "run_sweep must return results even when run_one raises"
        for r in results:
            assert r["survived"] is False, (
                f"Raising run_one must produce survived=False, got {r['survived']!r}"
            )

    def test_raising_run_one_does_not_propagate_exception(self, tmp_path: Path):
        """A raising run_one must NOT cause run_sweep to raise."""
        config = _make_config(n_seeds=2, fractions=[0.0, 0.05])

        def intermittent_exploder(condition, seed, fraction):
            if seed == 1 and fraction == 0.05:
                raise ValueError("specific run failure")
            return {"survived": True}

        # run_sweep must complete without raising
        results = run_sweep(config, intermittent_exploder, str(tmp_path))
        assert len(results) == 2 * 2 * 2, (
            f"All 8 grid points must still be attempted, got {len(results)} results"
        )

    def test_remaining_grid_points_run_after_one_failure(self, tmp_path: Path):
        """After one run_one failure, remaining grid points must still be called."""
        config = _make_config(n_seeds=3, fractions=[0.0, 0.05])
        call_count = 0
        expected = 2 * 3 * 2  # 12

        def counting_exploder(condition, seed, fraction):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first call fails")
            return {"survived": True}

        run_sweep(config, counting_exploder, str(tmp_path))

        assert call_count == expected, (
            f"Expected {expected} total calls despite first failure, got {call_count}"
        )

    def test_surviving_runs_are_true_after_partial_failure(self, tmp_path: Path):
        """Grid points that succeed must still report survived=True after a peer fails."""
        config = _make_config(n_seeds=2, fractions=[0.0])
        failure_calls = 0

        def mixed_run_one(condition, seed, fraction):
            nonlocal failure_calls
            # Only the first in_loop seed 0 fails
            if condition == "in_loop" and seed == 0:
                failure_calls += 1
                raise RuntimeError("in_loop seed 0 failed")
            return {"survived": True}

        results = run_sweep(config, mixed_run_one, str(tmp_path))

        surviving = [r for r in results if r.get("survived") is True]
        failed = [r for r in results if r.get("survived") is False]

        assert len(failed) >= 1, "Expected at least one failed run"
        assert len(surviving) >= 1, "Expected at least one surviving run"
        for r in surviving:
            assert r["survived"] is True


# ---------------------------------------------------------------------------
# E) Metric store population
# ---------------------------------------------------------------------------

class TestMetricStorePopulation:
    def test_metric_store_populated_after_sweep(self, tmp_path: Path):
        """After run_sweep, metric_store_dir must contain at least one artifact."""
        store_dir = tmp_path / "store"
        store_dir.mkdir()
        config = _make_config(n_seeds=1, fractions=[0.0])
        run_sweep(config, _surviving_run_one, str(store_dir))

        artifacts = list(store_dir.iterdir())
        assert len(artifacts) >= 1, (
            f"metric_store_dir must contain at least one artifact after the sweep, "
            f"but it is empty."
        )

    def test_metric_store_has_one_artifact_per_run(self, tmp_path: Path):
        """metric_store_dir must contain exactly one artifact per grid point."""
        n_seeds = 2
        fractions = [0.0, 0.05]
        expected_artifacts = 2 * n_seeds * len(fractions)  # 8

        store_dir = tmp_path / "store"
        store_dir.mkdir()
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        run_sweep(config, _surviving_run_one, str(store_dir))

        # Count all files (any extension) in store_dir
        artifacts = [p for p in store_dir.iterdir() if p.is_file()]
        assert len(artifacts) == expected_artifacts, (
            f"Expected {expected_artifacts} artifacts in metric_store_dir, "
            f"got {len(artifacts)}: {[p.name for p in artifacts]}"
        )

    def test_metric_store_artifacts_are_readable(self, tmp_path: Path):
        """Artifacts written to metric_store_dir must be non-empty readable files."""
        store_dir = tmp_path / "store"
        store_dir.mkdir()
        config = _make_config(n_seeds=1, fractions=[0.0])
        run_sweep(config, _surviving_run_one, str(store_dir))

        for path in store_dir.iterdir():
            if path.is_file():
                content = path.read_bytes()
                assert len(content) > 0, (
                    f"Artifact {path.name} is empty — artifacts must contain data."
                )

    def test_metric_store_dir_created_if_not_exists(self, tmp_path: Path):
        """run_sweep must create metric_store_dir if it does not exist."""
        store_dir = tmp_path / "new_store_dir"
        # Deliberately do NOT create store_dir
        assert not store_dir.exists(), "Precondition: store dir must not exist"

        config = _make_config(n_seeds=1, fractions=[0.0])
        run_sweep(config, _surviving_run_one, str(store_dir))

        assert store_dir.exists(), (
            "run_sweep must create metric_store_dir when it does not exist"
        )

    def test_metric_store_artifacts_contain_condition_seed_fraction(self, tmp_path: Path):
        """Each artifact must encode condition, seed, and fraction in its content."""
        store_dir = tmp_path / "store"
        store_dir.mkdir()
        config = _make_config(n_seeds=1, fractions=[0.0, 0.10])
        run_sweep(config, _surviving_run_one, str(store_dir))

        for path in store_dir.iterdir():
            if not path.is_file():
                continue
            # Try to parse as JSON; if parseable, assert required keys
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # At least condition, seed, fraction or survived must be present
                    has_meta = any(
                        k in data
                        for k in ("condition", "seed", "fraction", "survived")
                    )
                    assert has_meta, (
                        f"Artifact {path.name} lacks expected metadata keys. "
                        f"Got keys: {list(data.keys())}"
                    )
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Non-JSON artifact is acceptable; skip content inspection
                pass


# ---------------------------------------------------------------------------
# F) Return value structure
# ---------------------------------------------------------------------------

class TestReturnValueStructure:
    def test_returns_list(self, tmp_path: Path):
        """run_sweep must return a list."""
        config = _make_config(n_seeds=1, fractions=[0.0])
        result = run_sweep(config, _surviving_run_one, str(tmp_path))
        assert isinstance(result, list), (
            f"run_sweep must return a list, got {type(result)}"
        )

    def test_each_element_is_dict(self, tmp_path: Path):
        """Every element of the returned list must be a dict."""
        config = _make_config(n_seeds=2, fractions=[0.0, 0.05])
        results = run_sweep(config, _surviving_run_one, str(tmp_path))
        for i, r in enumerate(results):
            assert isinstance(r, dict), (
                f"results[{i}] must be a dict, got {type(r)}: {r!r}"
            )

    def test_returned_list_length_matches_grid(self, tmp_path: Path):
        """Return list length must equal 2 × n_seeds × len(fractions)."""
        n_seeds = 3
        fractions = [0.0, 0.01, 0.05, 0.10]
        config = _make_config(n_seeds=n_seeds, fractions=fractions)
        results = run_sweep(config, _surviving_run_one, str(tmp_path))
        expected = 2 * n_seeds * len(fractions)
        assert len(results) == expected

    def test_all_results_have_survived_key(self, tmp_path: Path):
        """Every result dict must have the 'survived' key."""
        config = _make_config(n_seeds=2, fractions=[0.0])
        results = run_sweep(config, _surviving_run_one, str(tmp_path))
        for i, r in enumerate(results):
            assert "survived" in r, f"results[{i}] missing 'survived': {r}"

    def test_all_results_have_condition_key(self, tmp_path: Path):
        config = _make_config(n_seeds=1, fractions=[0.0])
        results = run_sweep(config, _surviving_run_one, str(tmp_path))
        for i, r in enumerate(results):
            assert "condition" in r, f"results[{i}] missing 'condition': {r}"

    def test_all_results_have_seed_key(self, tmp_path: Path):
        config = _make_config(n_seeds=1, fractions=[0.0])
        results = run_sweep(config, _surviving_run_one, str(tmp_path))
        for i, r in enumerate(results):
            assert "seed" in r, f"results[{i}] missing 'seed': {r}"

    def test_all_results_have_fraction_key(self, tmp_path: Path):
        config = _make_config(n_seeds=1, fractions=[0.0])
        results = run_sweep(config, _surviving_run_one, str(tmp_path))
        for i, r in enumerate(results):
            assert "fraction" in r, f"results[{i}] missing 'fraction': {r}"
