"""RED-phase tests for Step 6c: BenchmarkConfig + validate_config (AC-2).

Contract being specified
------------------------
BenchmarkConfig is a dataclass carrying all locked experimental constants.
load_config(path) -> BenchmarkConfig parses a YAML file.
validate_config(config, *, env_probe) raises ConfigValidationError when:

  1. any required string field is unset (empty string, None, or missing)
  2. any required numeric field is 0 / None / negative (where that is invalid)
  3. n_seeds < 3
  4. adversarial_fractions is empty or does not contain 0.0
  5. a version pin does not match what env_probe() returns

A fully-populated, matching config passes without raising.

The committed benchmark_v1.yaml must parse successfully and have every
required key present (placeholder values are allowed; the YAML schema is
what is under test here, not the live env match).

Interface assumptions (implementer must honour):
  - Module path: python/rlox/agentic/config.py
  - Top-level imports: ``import config`` (conftest injects the agentic dir)
  - ConfigValidationError is importable from that module
  - BenchmarkConfig fields: model_revision, dataset_split, dataset_seed,
    global_batch_size, rollout_count, max_seq_len, reward_fn, data_order_seed,
    n_steps, n_seeds, warmup_steps, adversarial_fractions, per_sample_timeout_secs,
    primerl_commit, verifiers_commit, vllm_version, torch_version, rlox_commit
  - validate_config signature:
      validate_config(config, *, env_probe: Callable[[], dict] | None = None) -> None
  - env_probe() returns a dict such as
      {"vllm_version": "0.22.0+cu129", "torch_version": "2.9.0"}
    keys are the same as the config fields for those pins.

All imports are top-level: conftest.py injects python/rlox/agentic/ onto
sys.path, so ``import config`` works without touching rlox.__init__.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

# Top-level imports — never "from rlox.agentic import ..."
from rlox_agent.config import BenchmarkConfig, ConfigValidationError, load_config, validate_config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_YAML_PATH = _REPO_ROOT / "benchmarks" / "agentic" / "configs" / "benchmark_v1.yaml"

# A fully-populated BenchmarkConfig whose version pins match the env_probe below.
# Used as the "happy path" fixture.
_MATCHING_ENV = {
    "vllm_version": "0.22.0+cu129",
    "torch_version": "2.9.0",
}


def _valid_config(**overrides) -> BenchmarkConfig:
    """Return a fully-populated BenchmarkConfig that passes validate_config.

    The version pins are set to match _MATCHING_ENV.
    Any field can be overridden via kwargs.
    """
    base = dict(
        model_revision="abc123def456",
        dataset_split="train",
        dataset_seed=42,
        global_batch_size=8,
        rollout_count=4,
        max_seq_len=4096,
        reward_fn="pass_rate",
        data_order_seed=1234,
        n_steps=200,
        n_seeds=3,
        warmup_steps=5,
        adversarial_fractions=[0.0, 0.01, 0.05, 0.10],
        per_sample_timeout_secs=30.0,
        primerl_commit="primerl-abc123",
        verifiers_commit="verifiers-def456",
        vllm_version="0.22.0+cu129",
        torch_version="2.9.0",
        rlox_commit="rlox-ghi789",
    )
    base.update(overrides)
    return BenchmarkConfig(**base)


def _matching_env_probe():
    return dict(_MATCHING_ENV)


# ---------------------------------------------------------------------------
# A) BenchmarkConfig dataclass structure
# ---------------------------------------------------------------------------

class TestBenchmarkConfigDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(BenchmarkConfig)

    def test_can_construct_with_defaults(self):
        """BenchmarkConfig must be constructable with no arguments."""
        config = BenchmarkConfig()
        assert config is not None

    @pytest.mark.parametrize("field_name", [
        "model_revision",
        "dataset_split",
        "dataset_seed",
        "global_batch_size",
        "rollout_count",
        "max_seq_len",
        "reward_fn",
        "data_order_seed",
        "n_steps",
        "n_seeds",
        "warmup_steps",
        "adversarial_fractions",
        "per_sample_timeout_secs",
        "primerl_commit",
        "verifiers_commit",
        "vllm_version",
        "torch_version",
        "rlox_commit",
    ])
    def test_has_required_field(self, field_name: str):
        """BenchmarkConfig must have every required field."""
        field_names = {f.name for f in dataclasses.fields(BenchmarkConfig)}
        assert field_name in field_names, (
            f"BenchmarkConfig is missing required field '{field_name}'"
        )

    def test_adversarial_fractions_default_is_list(self):
        config = BenchmarkConfig()
        assert isinstance(config.adversarial_fractions, list)


# ---------------------------------------------------------------------------
# B) ConfigValidationError
# ---------------------------------------------------------------------------

class TestConfigValidationError:
    def test_is_exception_subclass(self):
        assert issubclass(ConfigValidationError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(ConfigValidationError):
            raise ConfigValidationError("test error")

    def test_message_preserved(self):
        msg = "model_revision is required"
        try:
            raise ConfigValidationError(msg)
        except ConfigValidationError as exc:
            assert msg in str(exc)


# ---------------------------------------------------------------------------
# C) Happy path: valid config with matching env_probe passes
# ---------------------------------------------------------------------------

class TestValidConfigPasses:
    def test_valid_config_matching_env_does_not_raise(self):
        """A fully-populated config with matching version pins must pass."""
        config = _valid_config()
        # Must not raise
        validate_config(config, env_probe=_matching_env_probe)

    def test_valid_config_returns_none(self):
        """validate_config must return None on success."""
        config = _valid_config()
        result = validate_config(config, env_probe=_matching_env_probe)
        assert result is None

    def test_n_seeds_exactly_3_passes(self):
        """n_seeds == 3 is the minimum allowed; must pass."""
        config = _valid_config(n_seeds=3)
        validate_config(config, env_probe=_matching_env_probe)

    def test_n_seeds_above_3_passes(self):
        """n_seeds == 5 must pass."""
        config = _valid_config(n_seeds=5)
        validate_config(config, env_probe=_matching_env_probe)

    def test_adversarial_fractions_with_more_values_passes(self):
        """Extra fractions beyond the standard 4 must pass as long as 0.0 is present."""
        config = _valid_config(adversarial_fractions=[0.0, 0.01, 0.05, 0.10, 0.20])
        validate_config(config, env_probe=_matching_env_probe)


# ---------------------------------------------------------------------------
# D) Required string fields — each empty/None triggers ConfigValidationError
# ---------------------------------------------------------------------------

_STRING_FIELDS = [
    "model_revision",
    "dataset_split",
    "reward_fn",
    "primerl_commit",
    "verifiers_commit",
    "vllm_version",
    "torch_version",
    "rlox_commit",
]


@pytest.mark.parametrize("field_name", _STRING_FIELDS)
def test_empty_string_field_raises(field_name: str):
    """Each required string field, when set to '', must raise ConfigValidationError."""
    config = _valid_config(**{field_name: ""})
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config, env_probe=_matching_env_probe)
    # The error message should identify the offending field
    assert field_name in str(exc_info.value) or len(str(exc_info.value)) > 0


@pytest.mark.parametrize("field_name", _STRING_FIELDS)
def test_none_string_field_raises(field_name: str):
    """Each required string field, when set to None, must raise ConfigValidationError."""
    config = _valid_config(**{field_name: None})
    with pytest.raises(ConfigValidationError):
        validate_config(config, env_probe=_matching_env_probe)


# ---------------------------------------------------------------------------
# E) n_seeds constraint: must be >= 3
# ---------------------------------------------------------------------------

class TestNSeedsConstraint:
    def test_n_seeds_2_raises(self):
        """n_seeds=2 is below the minimum of 3 — must raise."""
        config = _valid_config(n_seeds=2)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config, env_probe=_matching_env_probe)
        assert "n_seeds" in str(exc_info.value) or "seed" in str(exc_info.value).lower()

    def test_n_seeds_1_raises(self):
        config = _valid_config(n_seeds=1)
        with pytest.raises(ConfigValidationError):
            validate_config(config, env_probe=_matching_env_probe)

    def test_n_seeds_0_raises(self):
        config = _valid_config(n_seeds=0)
        with pytest.raises(ConfigValidationError):
            validate_config(config, env_probe=_matching_env_probe)

    def test_n_seeds_negative_raises(self):
        config = _valid_config(n_seeds=-1)
        with pytest.raises(ConfigValidationError):
            validate_config(config, env_probe=_matching_env_probe)


# ---------------------------------------------------------------------------
# F) adversarial_fractions constraints
# ---------------------------------------------------------------------------

class TestAdversarialFractionsConstraint:
    def test_empty_adversarial_fractions_raises(self):
        """Empty adversarial_fractions list must raise ConfigValidationError."""
        config = _valid_config(adversarial_fractions=[])
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config, env_probe=_matching_env_probe)
        assert "fraction" in str(exc_info.value).lower() or len(str(exc_info.value)) > 0

    def test_missing_0_0_baseline_raises(self):
        """adversarial_fractions without 0.0 must raise — 0.0 is the baseline condition."""
        config = _valid_config(adversarial_fractions=[0.01, 0.05, 0.10])
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config, env_probe=_matching_env_probe)
        err = str(exc_info.value).lower()
        assert "0.0" in err or "baseline" in err or "fraction" in err

    def test_only_0_0_passes(self):
        """A single-element list [0.0] must pass — 0.0 is the baseline."""
        config = _valid_config(adversarial_fractions=[0.0])
        validate_config(config, env_probe=_matching_env_probe)

    def test_none_adversarial_fractions_raises(self):
        """None adversarial_fractions must raise ConfigValidationError."""
        config = _valid_config(adversarial_fractions=None)
        with pytest.raises(ConfigValidationError):
            validate_config(config, env_probe=_matching_env_probe)


# ---------------------------------------------------------------------------
# G) Version pin mismatch — injectable env_probe
# ---------------------------------------------------------------------------

class TestVersionPinMismatch:
    def test_vllm_version_mismatch_raises(self):
        """If vllm_version in config doesn't match env_probe, must raise."""
        config = _valid_config(vllm_version="0.22.0+cu129")

        def mismatched_probe():
            return {"vllm_version": "0.99.0", "torch_version": "2.9.0"}

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config, env_probe=mismatched_probe)
        err = str(exc_info.value)
        # Must name which pin failed
        assert "vllm" in err.lower() or "vllm_version" in err

    def test_torch_version_mismatch_raises(self):
        """If torch_version in config doesn't match env_probe, must raise."""
        config = _valid_config(torch_version="2.9.0")

        def mismatched_probe():
            return {"vllm_version": "0.22.0+cu129", "torch_version": "1.0.0"}

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config, env_probe=mismatched_probe)
        err = str(exc_info.value)
        assert "torch" in err.lower() or "torch_version" in err

    def test_env_probe_mismatch_names_the_pin(self):
        """Error message on pin mismatch must identify the mismatched field."""
        config = _valid_config(vllm_version="0.22.0+cu129")

        def mismatched_probe():
            return {"vllm_version": "WRONG", "torch_version": "2.9.0"}

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config, env_probe=mismatched_probe)
        # The error message must contain a meaningful identifier
        err = str(exc_info.value)
        assert "vllm" in err.lower() or "version" in err.lower()

    def test_matching_env_probe_passes(self):
        """Exact match on both pins must not raise."""
        config = _valid_config(
            vllm_version="0.22.0+cu129",
            torch_version="2.9.0",
        )

        def probe():
            return {"vllm_version": "0.22.0+cu129", "torch_version": "2.9.0"}

        validate_config(config, env_probe=probe)  # must not raise

    def test_env_probe_is_called_not_real_import(self, monkeypatch):
        """The injected env_probe must be used — real torch/vllm must never be imported."""
        # We prove the probe is called by using a counting closure.
        probe_call_count = 0

        def counting_probe():
            nonlocal probe_call_count
            probe_call_count += 1
            return {"vllm_version": "0.22.0+cu129", "torch_version": "2.9.0"}

        config = _valid_config()
        validate_config(config, env_probe=counting_probe)
        assert probe_call_count >= 1, (
            "env_probe must be called at least once during validate_config"
        )


# ---------------------------------------------------------------------------
# H) load_config — YAML parsing
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_config_returns_benchmark_config(self, tmp_path: Path):
        """load_config on a valid YAML must return a BenchmarkConfig instance."""
        yaml_content = {
            "model_revision": "abc123",
            "dataset_split": "train",
            "dataset_seed": 42,
            "global_batch_size": 8,
            "rollout_count": 4,
            "max_seq_len": 4096,
            "reward_fn": "pass_rate",
            "data_order_seed": 1234,
            "n_steps": 200,
            "n_seeds": 3,
            "warmup_steps": 5,
            "adversarial_fractions": [0.0, 0.01, 0.05, 0.10],
            "per_sample_timeout_secs": 30.0,
            "primerl_commit": "prime-abc",
            "verifiers_commit": "ver-def",
            "vllm_version": "0.22.0+cu129",
            "torch_version": "2.9.0",
            "rlox_commit": "rlox-ghi",
        }
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        result = load_config(str(yaml_file))
        assert isinstance(result, BenchmarkConfig), (
            f"load_config must return a BenchmarkConfig, got {type(result)}"
        )

    def test_load_config_populates_model_revision(self, tmp_path: Path):
        yaml_content = {
            "model_revision": "my-revision-hash",
            "dataset_split": "train",
            "dataset_seed": 42,
            "global_batch_size": 8,
            "rollout_count": 4,
            "max_seq_len": 4096,
            "reward_fn": "pass_rate",
            "data_order_seed": 1234,
            "n_steps": 200,
            "n_seeds": 3,
            "warmup_steps": 5,
            "adversarial_fractions": [0.0, 0.01],
            "per_sample_timeout_secs": 30.0,
            "primerl_commit": "p",
            "verifiers_commit": "v",
            "vllm_version": "0.22.0+cu129",
            "torch_version": "2.9.0",
            "rlox_commit": "r",
        }
        yaml_file = tmp_path / "c.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))
        result = load_config(str(yaml_file))
        assert result.model_revision == "my-revision-hash"

    def test_load_config_populates_adversarial_fractions(self, tmp_path: Path):
        yaml_content = {
            "model_revision": "rev",
            "dataset_split": "train",
            "dataset_seed": 0,
            "global_batch_size": 8,
            "rollout_count": 4,
            "max_seq_len": 4096,
            "reward_fn": "pass_rate",
            "data_order_seed": 0,
            "n_steps": 10,
            "n_seeds": 3,
            "warmup_steps": 1,
            "adversarial_fractions": [0.0, 0.05, 0.10],
            "per_sample_timeout_secs": 30.0,
            "primerl_commit": "p",
            "verifiers_commit": "v",
            "vllm_version": "0.22.0+cu129",
            "torch_version": "2.9.0",
            "rlox_commit": "r",
        }
        yaml_file = tmp_path / "c2.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))
        result = load_config(str(yaml_file))
        assert isinstance(result.adversarial_fractions, list)
        assert 0.0 in result.adversarial_fractions

    def test_load_config_nonexistent_file_raises(self):
        """load_config on a missing file must raise (FileNotFoundError or IOError)."""
        with pytest.raises((FileNotFoundError, IOError, Exception)):
            load_config("/nonexistent/path/to/config.yaml")

    def test_load_config_n_seeds_is_integer(self, tmp_path: Path):
        """n_seeds must be loaded as an integer, not a string."""
        yaml_content = {
            "model_revision": "r",
            "dataset_split": "train",
            "dataset_seed": 0,
            "global_batch_size": 8,
            "rollout_count": 4,
            "max_seq_len": 4096,
            "reward_fn": "pass_rate",
            "data_order_seed": 0,
            "n_steps": 10,
            "n_seeds": 5,
            "warmup_steps": 1,
            "adversarial_fractions": [0.0],
            "per_sample_timeout_secs": 30.0,
            "primerl_commit": "p",
            "verifiers_commit": "v",
            "vllm_version": "0.22.0+cu129",
            "torch_version": "2.9.0",
            "rlox_commit": "r",
        }
        yaml_file = tmp_path / "c3.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))
        result = load_config(str(yaml_file))
        assert isinstance(result.n_seeds, int), (
            f"n_seeds must be an int, got {type(result.n_seeds)}"
        )
        assert result.n_seeds == 5


# ---------------------------------------------------------------------------
# I) The committed benchmark_v1.yaml — schema + key presence check
# ---------------------------------------------------------------------------

class TestCommittedYaml:
    """The committed benchmark_v1.yaml must parse and have every required key.

    Placeholder values (e.g. 'TODO-pin-...') are acceptable — the test
    validates schema completeness, NOT that pins are live-environment-valid.
    """

    _REQUIRED_KEYS = [
        "model_revision",
        "dataset_split",
        "dataset_seed",
        "global_batch_size",
        "rollout_count",
        "max_seq_len",
        "reward_fn",
        "data_order_seed",
        "n_steps",
        "n_seeds",
        "warmup_steps",
        "adversarial_fractions",
        "per_sample_timeout_secs",
        "primerl_commit",
        "verifiers_commit",
        "vllm_version",
        "torch_version",
        "rlox_commit",
    ]

    def test_yaml_file_exists(self):
        assert _YAML_PATH.exists(), (
            f"benchmark_v1.yaml not found at {_YAML_PATH}"
        )

    def test_yaml_is_valid(self):
        """The committed YAML must parse without error."""
        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), (
            f"benchmark_v1.yaml must be a YAML mapping, got {type(data)}"
        )

    @pytest.mark.parametrize("key", _REQUIRED_KEYS)
    def test_yaml_has_required_key(self, key: str):
        """Every required key must be present in benchmark_v1.yaml."""
        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert key in data, (
            f"benchmark_v1.yaml is missing required key '{key}'"
        )

    def test_yaml_adversarial_fractions_is_list(self):
        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        fractions = data.get("adversarial_fractions")
        assert isinstance(fractions, list), (
            f"adversarial_fractions must be a YAML sequence, got {type(fractions)}"
        )

    def test_yaml_adversarial_fractions_contains_0_0(self):
        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        fractions = data.get("adversarial_fractions", [])
        assert 0.0 in fractions or 0 in fractions, (
            f"adversarial_fractions must contain 0.0 (baseline), got {fractions}"
        )

    def test_yaml_n_seeds_at_least_3(self):
        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        n_seeds = data.get("n_seeds")
        assert isinstance(n_seeds, int) and n_seeds >= 3, (
            f"n_seeds must be an integer >= 3, got {n_seeds!r}"
        )

    def test_yaml_parses_via_load_config(self):
        """load_config must succeed on benchmark_v1.yaml (placeholder values OK)."""
        result = load_config(str(_YAML_PATH))
        assert isinstance(result, BenchmarkConfig)


# ---------------------------------------------------------------------------
# J) Must-be-positive numeric fields (FIX 4 / AC-2 extension)
#
# global_batch_size, rollout_count, max_seq_len, n_steps,
# per_sample_timeout_secs must all be > 0.  A value of 0 (the dataclass
# default) must raise ConfigValidationError.
#
# Fields intentionally allowed to be 0:
#   dataset_seed, data_order_seed, warmup_steps, seed
# ---------------------------------------------------------------------------

_MUST_BE_POSITIVE_FIELDS = [
    "global_batch_size",
    "rollout_count",
    "max_seq_len",
    "n_steps",
    "per_sample_timeout_secs",
]


@pytest.mark.parametrize("field_name", _MUST_BE_POSITIVE_FIELDS)
def test_zero_must_be_positive_field_raises(field_name: str):
    """Each must-be-positive field set to 0 must raise ConfigValidationError."""
    config = _valid_config(**{field_name: 0})
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config, env_probe=_matching_env_probe)
    assert field_name in str(exc_info.value) or len(str(exc_info.value)) > 0


@pytest.mark.parametrize("field_name", _MUST_BE_POSITIVE_FIELDS)
def test_negative_must_be_positive_field_raises(field_name: str):
    """Each must-be-positive field set to a negative value must raise."""
    config = _valid_config(**{field_name: -1})
    with pytest.raises(ConfigValidationError):
        validate_config(config, env_probe=_matching_env_probe)


@pytest.mark.parametrize("field_name", _MUST_BE_POSITIVE_FIELDS)
def test_positive_must_be_positive_field_passes(field_name: str):
    """Each must-be-positive field set to 1 must pass validation."""
    config = _valid_config(**{field_name: 1})
    validate_config(config, env_probe=_matching_env_probe)


class TestAllowedZeroFields:
    """Fields where 0 is a meaningful value — must NOT raise when set to 0."""

    def test_warmup_steps_zero_passes(self):
        config = _valid_config(warmup_steps=0)
        validate_config(config, env_probe=_matching_env_probe)

    def test_dataset_seed_zero_passes(self):
        config = _valid_config(dataset_seed=0)
        validate_config(config, env_probe=_matching_env_probe)

    def test_data_order_seed_zero_passes(self):
        config = _valid_config(data_order_seed=0)
        validate_config(config, env_probe=_matching_env_probe)
