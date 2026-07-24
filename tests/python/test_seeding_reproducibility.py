"""Reproducibility tests for ``seed`` across off-policy algorithms.

TDD red phase: written FIRST, before the seeding fix lands. This is a
systemic gap: ``sac.py``, ``td3.py``, ``mpo.py``, ``awr.py``,
``decision_transformer.py``, ``calql.py``, ``qmix.py``, and
``diffusion_policy.py`` accept a ``seed`` constructor kwarg but never apply
it to torch/numpy/env RNG, so ``seed`` is a no-op and runs are
non-reproducible -- undermining the project's multi-seed IQM validation
methodology. This module covers the two required algorithms, SAC and TD3;
the fix pattern is shared with ``rlox/algorithms/pqn.py`` (the original
precedent) and ``rlox/algorithms/crossq.py`` (a prior instance of the same
fix -- see ``TestCrossQSeedReproducibility`` in ``test_crossq.py``, which
this module intentionally mirrors).

Every test here is expected to FAIL until ``torch.manual_seed``,
``np.random.seed``, and ``env.action_space.seed`` are applied in
``__init__`` (before actor/critic construction) and ``env.reset(seed=...)``
is used at the first reset in ``train()``.

Interface assumption: fixing this requires applying ``seed`` to torch
(network init and stochastic policy sampling) and, since the environment's
reset state and the ``learning_starts`` random-exploration
``action_space.sample()`` calls are also sources of randomness, to
numpy/the env as well. These tests assert only the externally observable
consequence (same seed -> same outcome; different seed -> different
outcome), not which specific RNG call the implementer seeds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlox.algorithms.sac import SAC
from rlox.algorithms.td3 import TD3

# Pendulum-v1 observation: (cos(theta), sin(theta), theta_dot).
_FIXED_OBS = np.array([1.0, 0.0, 0.0], dtype=np.float32)

# Small, shared hyperparameters so short-training fixtures stay fast while
# still exercising several gradient updates (learning_starts=50 well below
# total_timesteps=800, batch_size=32 well below buffer occupancy by then).
_SHORT_TRAIN_KWARGS = dict(
    env_id="Pendulum-v1",
    hidden=32,
    batch_size=32,
    learning_starts=50,
)
_SHORT_TRAIN_TIMESTEPS = 800


def _assert_state_dicts_equal(sd_a, sd_b, label: str) -> None:
    assert sd_a.keys() == sd_b.keys(), (
        f"{label}: state_dict keys differ: {list(sd_a.keys())} vs {list(sd_b.keys())}"
    )
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), (
            f"{label}: tensor '{key}' differs between the two instances."
        )


def _assert_state_dicts_differ(sd_a, sd_b, label: str) -> None:
    assert sd_a.keys() == sd_b.keys(), (
        f"{label}: state_dict keys differ: {list(sd_a.keys())} vs {list(sd_b.keys())}"
    )
    all_equal = all(torch.equal(sd_a[key], sd_b[key]) for key in sd_a)
    assert not all_equal, (
        f"{label}: every tensor is identical between the two instances; "
        f"expected at least one to differ."
    )


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------


class TestSACSeedReproducibility:
    """``seed`` must control torch/np/env RNG for SAC."""

    # -- construction-time determinism (no training) ------------------------

    def test_same_seed_produces_identical_initial_actor_parameters(self):
        """Two SAC(seed=123) instances must have bit-identical actor init."""
        a = SAC(env_id="Pendulum-v1", seed=123)
        b = SAC(env_id="Pendulum-v1", seed=123)
        _assert_state_dicts_equal(
            a.actor.state_dict(),
            b.actor.state_dict(),
            "same seed=123, actor init",
        )

    def test_same_seed_produces_identical_initial_critic_parameters(self):
        """Two SAC(seed=123) instances must have bit-identical critic1/critic2 init."""
        a = SAC(env_id="Pendulum-v1", seed=123)
        b = SAC(env_id="Pendulum-v1", seed=123)
        _assert_state_dicts_equal(
            a.critic1.state_dict(),
            b.critic1.state_dict(),
            "same seed=123, critic1 init",
        )
        _assert_state_dicts_equal(
            a.critic2.state_dict(),
            b.critic2.state_dict(),
            "same seed=123, critic2 init",
        )

    def test_different_seeds_produce_different_initial_actor_parameters(self):
        """SAC(seed=123) and SAC(seed=456) must NOT have identical actor init.

        Guards against a degenerate 'fix' that calls torch.manual_seed with a
        hardcoded constant instead of the `seed` argument.
        """
        a = SAC(env_id="Pendulum-v1", seed=123)
        b = SAC(env_id="Pendulum-v1", seed=456)
        _assert_state_dicts_differ(
            a.actor.state_dict(),
            b.actor.state_dict(),
            "seed=123 vs seed=456, actor init",
        )

    # -- short-training determinism ------------------------------------------

    @pytest.fixture(scope="class")
    def same_seed_pair(self):
        """Two SAC(seed=123) instances trained for an identical short budget."""
        kwargs = {**_SHORT_TRAIN_KWARGS, "seed": 123}
        a = SAC(**kwargs)
        a.train(total_timesteps=_SHORT_TRAIN_TIMESTEPS)
        b = SAC(**kwargs)
        b.train(total_timesteps=_SHORT_TRAIN_TIMESTEPS)
        return a, b

    @pytest.fixture(scope="class")
    def different_seed_pair(self):
        """Two SAC instances (seed=123 vs seed=456), same short training budget."""
        a = SAC(**{**_SHORT_TRAIN_KWARGS, "seed": 123})
        a.train(total_timesteps=_SHORT_TRAIN_TIMESTEPS)
        b = SAC(**{**_SHORT_TRAIN_KWARGS, "seed": 456})
        b.train(total_timesteps=_SHORT_TRAIN_TIMESTEPS)
        return a, b

    def test_same_seed_short_training_produces_identical_predict_output(
        self, same_seed_pair
    ):
        """Same seed -> identical deterministic predict() after ~800 training steps."""
        a, b = same_seed_pair
        action_a = a.predict(_FIXED_OBS, deterministic=True)
        action_b = b.predict(_FIXED_OBS, deterministic=True)
        np.testing.assert_array_equal(
            action_a,
            action_b,
            err_msg=(
                "Two SAC(seed=123) instances trained for an identical "
                "800-step budget must produce identical predict() output on "
                "the same observation. `seed` must control torch/np/env RNG."
            ),
        )

    def test_same_seed_short_training_produces_identical_critic_weights(
        self, same_seed_pair
    ):
        """Same seed -> identical critic1 weights after ~800 training steps."""
        a, b = same_seed_pair
        _assert_state_dicts_equal(
            a.critic1.state_dict(),
            b.critic1.state_dict(),
            "same seed=123, critic1 after 800 training steps",
        )

    def test_different_seeds_short_training_produce_different_predict_output(
        self, different_seed_pair
    ):
        """Different seeds -> different predict() output after ~800 training steps."""
        a, b = different_seed_pair
        action_a = a.predict(_FIXED_OBS, deterministic=True)
        action_b = b.predict(_FIXED_OBS, deterministic=True)
        assert not np.array_equal(action_a, action_b), (
            "SAC(seed=123) and SAC(seed=456) trained for an identical "
            "800-step budget produced IDENTICAL predict() output -- "
            "different seed values must produce different (controlled) runs."
        )


# ---------------------------------------------------------------------------
# TD3
# ---------------------------------------------------------------------------


class TestTD3SeedReproducibility:
    """``seed`` must control torch/np/env RNG for TD3."""

    # -- construction-time determinism (no training) ------------------------

    def test_same_seed_produces_identical_initial_actor_parameters(self):
        """Two TD3(seed=123) instances must have bit-identical actor init."""
        a = TD3(env_id="Pendulum-v1", seed=123)
        b = TD3(env_id="Pendulum-v1", seed=123)
        _assert_state_dicts_equal(
            a.actor.state_dict(),
            b.actor.state_dict(),
            "same seed=123, actor init",
        )

    def test_same_seed_produces_identical_initial_critic_parameters(self):
        """Two TD3(seed=123) instances must have bit-identical critic1/critic2 init."""
        a = TD3(env_id="Pendulum-v1", seed=123)
        b = TD3(env_id="Pendulum-v1", seed=123)
        _assert_state_dicts_equal(
            a.critic1.state_dict(),
            b.critic1.state_dict(),
            "same seed=123, critic1 init",
        )
        _assert_state_dicts_equal(
            a.critic2.state_dict(),
            b.critic2.state_dict(),
            "same seed=123, critic2 init",
        )

    def test_different_seeds_produce_different_initial_actor_parameters(self):
        """TD3(seed=123) and TD3(seed=456) must NOT have identical actor init.

        Guards against a degenerate 'fix' that calls torch.manual_seed with a
        hardcoded constant instead of the `seed` argument.
        """
        a = TD3(env_id="Pendulum-v1", seed=123)
        b = TD3(env_id="Pendulum-v1", seed=456)
        _assert_state_dicts_differ(
            a.actor.state_dict(),
            b.actor.state_dict(),
            "seed=123 vs seed=456, actor init",
        )

    # -- short-training determinism ------------------------------------------

    @pytest.fixture(scope="class")
    def same_seed_pair(self):
        """Two TD3(seed=123) instances trained for an identical short budget."""
        kwargs = {**_SHORT_TRAIN_KWARGS, "seed": 123}
        a = TD3(**kwargs)
        a.train(total_timesteps=_SHORT_TRAIN_TIMESTEPS)
        b = TD3(**kwargs)
        b.train(total_timesteps=_SHORT_TRAIN_TIMESTEPS)
        return a, b

    @pytest.fixture(scope="class")
    def different_seed_pair(self):
        """Two TD3 instances (seed=123 vs seed=456), same short training budget."""
        a = TD3(**{**_SHORT_TRAIN_KWARGS, "seed": 123})
        a.train(total_timesteps=_SHORT_TRAIN_TIMESTEPS)
        b = TD3(**{**_SHORT_TRAIN_KWARGS, "seed": 456})
        b.train(total_timesteps=_SHORT_TRAIN_TIMESTEPS)
        return a, b

    def test_same_seed_short_training_produces_identical_predict_output(
        self, same_seed_pair
    ):
        """Same seed -> identical predict() after ~800 training steps."""
        a, b = same_seed_pair
        action_a = a.predict(_FIXED_OBS, deterministic=True)
        action_b = b.predict(_FIXED_OBS, deterministic=True)
        np.testing.assert_array_equal(
            action_a,
            action_b,
            err_msg=(
                "Two TD3(seed=123) instances trained for an identical "
                "800-step budget must produce identical predict() output on "
                "the same observation. `seed` must control torch/np/env RNG."
            ),
        )

    def test_same_seed_short_training_produces_identical_critic_weights(
        self, same_seed_pair
    ):
        """Same seed -> identical critic1 weights after ~800 training steps."""
        a, b = same_seed_pair
        _assert_state_dicts_equal(
            a.critic1.state_dict(),
            b.critic1.state_dict(),
            "same seed=123, critic1 after 800 training steps",
        )

    def test_different_seeds_short_training_produce_different_predict_output(
        self, different_seed_pair
    ):
        """Different seeds -> different predict() output after ~800 training steps."""
        a, b = different_seed_pair
        action_a = a.predict(_FIXED_OBS, deterministic=True)
        action_b = b.predict(_FIXED_OBS, deterministic=True)
        assert not np.array_equal(action_a, action_b), (
            "TD3(seed=123) and TD3(seed=456) trained for an identical "
            "800-step budget produced IDENTICAL predict() output -- "
            "different seed values must produce different (controlled) runs."
        )
