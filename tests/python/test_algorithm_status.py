"""Tests for per-algorithm maturity status surfaced through the Trainer.

Contract (does NOT exist yet — these tests are RED):

  - ``ALGORITHM_STATUS: dict[str, str]``  module-level in ``rlox.trainer``
  - ``ALGORITHM_STATUSES: frozenset``     module-level in ``rlox.trainer``
  - ``algorithm_status(name: str) -> str`` function in ``rlox.trainer``
  - ``Trainer.status``                    read-only property
  - ``Trainer.__repr__``                  includes algo name AND status string
  - Warning behaviour: experimental algos emit UserWarning at construction;
    validated algos and custom classes do NOT.

Assumptions (implementer must honor):
  - Validated names are exactly: {"ppo", "sac", "td3", "dqn", "a2c"}.
  - Every other name currently in ALGORITHM_REGISTRY is "experimental".
  - ``algorithm_status`` is case-insensitive; lookup is by lower-cased name.
  - A custom (user-supplied) algo class has ``status == "experimental"``
    and emits NO warning.
  - The completeness invariant holds: set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALIDATED = frozenset({"ppo", "sac", "td3", "dqn", "a2c"})
_EXPERIMENTAL = frozenset({
    "awr", "calql", "diffusion", "dreamer", "dt",
    "impala", "mappo", "mpo", "qmix", "rcdtp", "rwdtp", "trpo", "vpg",
})


# ---------------------------------------------------------------------------
# TestAlgorithmStatusDict
# ---------------------------------------------------------------------------


class TestAlgorithmStatusDict:
    """ALGORITHM_STATUS completeness and value invariants."""

    def test_algorithm_status_exists(self) -> None:
        """ALGORITHM_STATUS is importable from rlox.trainer."""
        from rlox.trainer import ALGORITHM_STATUS  # noqa: F401

    def test_algorithm_statuses_frozenset_exists(self) -> None:
        """ALGORITHM_STATUSES frozenset is importable from rlox.trainer."""
        from rlox.trainer import ALGORITHM_STATUSES  # noqa: F401

    def test_algorithm_statuses_contains_exactly_two_values(self) -> None:
        from rlox.trainer import ALGORITHM_STATUSES

        assert ALGORITHM_STATUSES == frozenset({"validated", "experimental"})

    def test_algorithm_status_keys_match_registry(self) -> None:
        """set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY) — completeness invariant."""
        from rlox.trainer import ALGORITHM_REGISTRY, ALGORITHM_STATUS

        assert set(ALGORITHM_STATUS) == set(ALGORITHM_REGISTRY), (
            f"ALGORITHM_STATUS keys do not match ALGORITHM_REGISTRY keys. "
            f"Missing from STATUS: {set(ALGORITHM_REGISTRY) - set(ALGORITHM_STATUS)}. "
            f"Extra in STATUS: {set(ALGORITHM_STATUS) - set(ALGORITHM_REGISTRY)}."
        )

    def test_every_value_is_valid_status(self) -> None:
        """Every value in ALGORITHM_STATUS is in ALGORITHM_STATUSES."""
        from rlox.trainer import ALGORITHM_STATUS, ALGORITHM_STATUSES

        invalid = {k: v for k, v in ALGORITHM_STATUS.items() if v not in ALGORITHM_STATUSES}
        assert not invalid, f"Invalid status values found: {invalid}"

    @pytest.mark.parametrize("name", sorted(_VALIDATED))
    def test_validated_algos_have_validated_status(self, name: str) -> None:
        from rlox.trainer import ALGORITHM_STATUS

        assert ALGORITHM_STATUS[name] == "validated", (
            f"Expected {name!r} to be 'validated', got {ALGORITHM_STATUS.get(name)!r}"
        )

    @pytest.mark.parametrize("name", sorted(_EXPERIMENTAL))
    def test_experimental_algos_have_experimental_status(self, name: str) -> None:
        from rlox.trainer import ALGORITHM_STATUS

        assert ALGORITHM_STATUS[name] == "experimental", (
            f"Expected {name!r} to be 'experimental', got {ALGORITHM_STATUS.get(name)!r}"
        )


# ---------------------------------------------------------------------------
# TestAlgorithmStatusFunction
# ---------------------------------------------------------------------------


class TestAlgorithmStatusFunction:
    """algorithm_status() function contract."""

    def test_function_exists(self) -> None:
        from rlox.trainer import algorithm_status  # noqa: F401

    def test_returns_validated_for_ppo(self) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status("ppo") == "validated"

    def test_returns_validated_for_sac(self) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status("sac") == "validated"

    def test_returns_validated_for_td3(self) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status("td3") == "validated"

    def test_returns_validated_for_dqn(self) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status("dqn") == "validated"

    def test_returns_validated_for_a2c(self) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status("a2c") == "validated"

    def test_returns_experimental_for_trpo(self) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status("trpo") == "experimental"

    def test_returns_experimental_for_vpg(self) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status("vpg") == "experimental"

    def test_case_insensitive_ppo_upper(self) -> None:
        """'PPO' (all caps) must resolve to 'validated'."""
        from rlox.trainer import algorithm_status

        assert algorithm_status("PPO") == "validated"

    def test_case_insensitive_trpo_mixed(self) -> None:
        """'Trpo' (mixed case) must resolve to 'experimental'."""
        from rlox.trainer import algorithm_status

        assert algorithm_status("Trpo") == "experimental"

    def test_unknown_name_raises_value_error(self) -> None:
        """algorithm_status raises ValueError for an unregistered name."""
        from rlox.trainer import algorithm_status

        with pytest.raises(ValueError):
            algorithm_status("nonexistent")

    def test_value_error_message_lists_known_names(self) -> None:
        """The ValueError message must contain at least one known algorithm name."""
        from rlox.trainer import algorithm_status

        with pytest.raises(ValueError, match="ppo"):
            algorithm_status("not_a_real_algo")

    @pytest.mark.parametrize("name", sorted(_VALIDATED))
    def test_all_validated_via_function(self, name: str) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status(name) == "validated"

    @pytest.mark.parametrize("name", sorted(_EXPERIMENTAL))
    def test_all_experimental_via_function(self, name: str) -> None:
        from rlox.trainer import algorithm_status

        assert algorithm_status(name) == "experimental"


# ---------------------------------------------------------------------------
# TestTrainerStatusProperty
# ---------------------------------------------------------------------------


class TestTrainerStatusProperty:
    """Trainer.status read-only property."""

    def test_trainer_has_status_property(self) -> None:
        from rlox.trainer import Trainer

        # Suppress any UserWarning from experimental algos during construction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("ppo", env="CartPole-v1")
        assert hasattr(trainer, "status")

    def test_ppo_trainer_status_is_validated(self) -> None:
        from rlox.trainer import Trainer

        trainer = Trainer("ppo", env="CartPole-v1")
        assert trainer.status == "validated"

    def test_a2c_trainer_status_is_validated(self) -> None:
        from rlox.trainer import Trainer

        trainer = Trainer("a2c", env="CartPole-v1")
        assert trainer.status == "validated"

    def test_trpo_trainer_status_is_experimental(self) -> None:
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("trpo", env="CartPole-v1")
        assert trainer.status == "experimental"

    def test_vpg_trainer_status_is_experimental(self) -> None:
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("vpg", env="CartPole-v1")
        assert trainer.status == "experimental"

    def test_status_is_read_only(self) -> None:
        """Assigning to status must raise AttributeError."""
        from rlox.trainer import Trainer

        trainer = Trainer("ppo", env="CartPole-v1")
        with pytest.raises(AttributeError):
            trainer.status = "something_else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestTrainerRepr
# ---------------------------------------------------------------------------


class TestTrainerRepr:
    """Trainer.__repr__ contains algorithm name and status string."""

    def test_repr_contains_algo_name_for_ppo(self) -> None:
        from rlox.trainer import Trainer

        trainer = Trainer("ppo", env="CartPole-v1")
        r = repr(trainer)
        assert "ppo" in r, f"Expected 'ppo' in repr, got: {r!r}"

    def test_repr_contains_status_for_ppo(self) -> None:
        from rlox.trainer import Trainer

        trainer = Trainer("ppo", env="CartPole-v1")
        r = repr(trainer)
        assert "validated" in r, f"Expected 'validated' in repr, got: {r!r}"

    def test_repr_contains_algo_name_for_trpo(self) -> None:
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("trpo", env="CartPole-v1")
        r = repr(trainer)
        assert "trpo" in r, f"Expected 'trpo' in repr, got: {r!r}"

    def test_repr_contains_status_for_trpo(self) -> None:
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("trpo", env="CartPole-v1")
        r = repr(trainer)
        assert "experimental" in r, f"Expected 'experimental' in repr, got: {r!r}"

    def test_repr_contains_algo_name_for_vpg(self) -> None:
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("vpg", env="CartPole-v1")
        r = repr(trainer)
        assert "vpg" in r, f"Expected 'vpg' in repr, got: {r!r}"

    def test_repr_contains_experimental_for_vpg(self) -> None:
        from rlox.trainer import Trainer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            trainer = Trainer("vpg", env="CartPole-v1")
        r = repr(trainer)
        assert "experimental" in r, f"Expected 'experimental' in repr, got: {r!r}"


# ---------------------------------------------------------------------------
# TestTrainerWarnings
# ---------------------------------------------------------------------------


class TestTrainerWarnings:
    """Warning behaviour at construction time."""

    def test_experimental_algo_emits_user_warning(self) -> None:
        """Constructing Trainer('trpo', ...) fires a UserWarning about 'experimental'."""
        from rlox.trainer import Trainer

        with pytest.warns(UserWarning, match="experimental"):
            Trainer("trpo", env="CartPole-v1")

    def test_experimental_warning_contains_algo_name(self) -> None:
        """The UserWarning for an experimental algo must mention the algo name."""
        from rlox.trainer import Trainer

        with pytest.warns(UserWarning, match="trpo"):
            Trainer("trpo", env="CartPole-v1")

    def test_vpg_experimental_warning_contains_vpg(self) -> None:
        """The UserWarning for vpg must mention 'vpg'."""
        from rlox.trainer import Trainer

        with pytest.warns(UserWarning, match="vpg"):
            Trainer("vpg", env="CartPole-v1")

    def test_validated_algo_emits_no_experimental_warning(self) -> None:
        """Constructing Trainer('ppo', ...) must NOT emit a UserWarning about 'experimental'."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Trainer("ppo", env="CartPole-v1")

        experimental_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "experimental" in str(w.message).lower()
        ]
        assert not experimental_warnings, (
            f"Expected no 'experimental' UserWarning for ppo, got: {experimental_warnings}"
        )

    def test_a2c_validated_emits_no_experimental_warning(self) -> None:
        """Constructing Trainer('a2c', ...) must NOT emit an 'experimental' UserWarning."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Trainer("a2c", env="CartPole-v1")

        experimental_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "experimental" in str(w.message).lower()
        ]
        assert not experimental_warnings, (
            f"Expected no 'experimental' UserWarning for a2c, got: {experimental_warnings}"
        )

    @pytest.mark.parametrize("name", sorted(_EXPERIMENTAL))
    def test_all_experimental_algos_warn(self, name: str) -> None:
        """Every experimental algo emits a UserWarning mentioning 'experimental'.

        The warning is emitted BEFORE the algorithm object is constructed
        (early in Trainer.__init__), so it fires regardless of whether the
        algo's own __init__ subsequently raises (e.g. wrong action-space type).
        Construction failure is therefore tolerated; only the warning is asserted.
        """
        from rlox.trainer import Trainer

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                Trainer(name, env="CartPole-v1")
            except Exception:
                pass  # construction may legitimately fail; we only assert the warning fired
        experimental = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "experimental" in str(w.message).lower()
        ]
        assert experimental, f"No experimental UserWarning emitted for {name!r}"

    @pytest.mark.parametrize("name,env", [
        # SAC and TD3 require a continuous action space; use Pendulum-v1.
        ("a2c", "CartPole-v1"),
        ("dqn", "CartPole-v1"),
        ("ppo", "CartPole-v1"),
        ("sac", "Pendulum-v1"),
        ("td3", "Pendulum-v1"),
    ])
    def test_all_validated_algos_do_not_warn(self, name: str, env: str) -> None:
        """No validated algo emits an 'experimental' UserWarning."""
        from rlox.trainer import Trainer

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Trainer(name, env=env)

        experimental_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "experimental" in str(w.message).lower()
        ]
        assert not experimental_warnings, (
            f"Trainer({name!r}) emitted unexpected 'experimental' UserWarning: "
            f"{experimental_warnings}"
        )


# ---------------------------------------------------------------------------
# TestCustomClassBehavior
# ---------------------------------------------------------------------------


class TestCustomClassBehavior:
    """Custom (user-supplied) algo class: status == 'experimental', no warning."""

    def _make_dummy_algo_class(self) -> type:
        """Return a minimal dummy algorithm class that Trainer can instantiate."""

        class _DummyAlgo:
            def __init__(self, env_id: str, seed: int = 42, **kwargs: Any) -> None:
                self.env_id = env_id
                self.seed = seed

            def train(self, total_timesteps: int) -> dict[str, float]:
                return {"mean_reward": 0.0}

            def save(self, path: str) -> None:
                pass

            @classmethod
            def from_checkpoint(cls, path: str, env_id: str | None = None) -> "_DummyAlgo":
                return cls(env_id=env_id or "CartPole-v1")

            def predict(self, obs: Any, deterministic: bool = True) -> Any:
                return 0

        return _DummyAlgo

    def test_custom_class_status_is_experimental(self) -> None:
        """Trainer constructed with a custom class reports status == 'experimental'."""
        from rlox.trainer import Trainer

        DummyAlgo = self._make_dummy_algo_class()
        trainer = Trainer(DummyAlgo, env="CartPole-v1")
        assert trainer.status == "experimental", (
            f"Expected 'experimental' for custom class, got {trainer.status!r}"
        )

    def test_custom_class_emits_no_warning(self) -> None:
        """Trainer constructed with a custom class emits NO UserWarning."""
        from rlox.trainer import Trainer

        DummyAlgo = self._make_dummy_algo_class()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Trainer(DummyAlgo, env="CartPole-v1")

        user_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
        ]
        assert not user_warnings, (
            f"Expected no UserWarning for custom class, got: {user_warnings}"
        )

    def test_custom_class_repr_contains_experimental(self) -> None:
        """repr of Trainer with custom class contains 'experimental'."""
        from rlox.trainer import Trainer

        DummyAlgo = self._make_dummy_algo_class()
        trainer = Trainer(DummyAlgo, env="CartPole-v1")
        r = repr(trainer)
        assert "experimental" in r, (
            f"Expected 'experimental' in repr for custom class, got: {r!r}"
        )
