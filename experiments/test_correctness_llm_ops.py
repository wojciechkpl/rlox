"""LLM ops correctness tests — TDD RED phase.

Defines the behavioral contract for rlox's GRPO advantages and token KL
divergence computations.

Test categories
---------------
1. GRPO matches numpy reference to rtol=1e-6
2. Token KL matches numpy reference to rtol=1e-6
3. GRPO with different group sizes (2, 4, 8, 16)
4. Token KL with different sequence lengths
5. Edge cases: uniform rewards, single-element group, very long sequences
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import reference_grpo_numpy, reference_token_kl_numpy


# ---------------------------------------------------------------------------
# 1. GRPO matches numpy reference
# ---------------------------------------------------------------------------


@pytest.mark.correctness
@pytest.mark.parametrize("n_prompts,k", [
    (1, 4),
    (16, 4),
    (64, 8),
    (256, 16),
])
def test_grpo_matches_numpy_reference(n_prompts, k, rlox_module):
    """compute_group_advantages must match the reference numpy implementation."""
    rng = np.random.default_rng(42)
    groups = [rng.standard_normal(k).astype(np.float64) for _ in range(n_prompts)]

    for i, rewards in enumerate(groups):
        ref = reference_grpo_numpy(rewards)
        rlox_result = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)
        np.testing.assert_allclose(
            rlox_result, ref, rtol=1e-6,
            err_msg=f"GRPO mismatch at group {i} (n_prompts={n_prompts}, k={k})",
        )


@pytest.mark.correctness
def test_grpo_output_is_normalized(rlox_module):
    """GRPO output should be approximately zero-mean, unit-variance per group."""
    rng = np.random.default_rng(7)
    rewards = rng.standard_normal(16).astype(np.float64)
    adv = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)

    assert abs(adv.mean()) < 1e-10, f"Expected zero mean, got {adv.mean()}"
    assert abs(adv.std() - 1.0) < 1e-10, f"Expected unit std, got {adv.std()}"


# ---------------------------------------------------------------------------
# 2. Token KL matches numpy reference
# ---------------------------------------------------------------------------


@pytest.mark.correctness
@pytest.mark.parametrize("seq_len", [8, 128, 512, 2048, 8192])
def test_token_kl_matches_numpy_reference(seq_len, rlox_module):
    """compute_token_kl must match the reference numpy implementation."""
    rng = np.random.default_rng(42)
    # Use negative values to simulate log-probabilities
    log_p = -np.abs(rng.standard_normal(seq_len)).astype(np.float64)
    log_q = -np.abs(rng.standard_normal(seq_len)).astype(np.float64)

    ref = reference_token_kl_numpy(log_p, log_q)
    rlox_result = float(rlox_module.compute_token_kl(log_p, log_q))

    assert abs(rlox_result - ref) / (abs(ref) + 1e-10) < 1e-6, (
        f"Token KL mismatch at seq_len={seq_len}: rlox={rlox_result}, ref={ref}"
    )


@pytest.mark.correctness
def test_token_kl_non_negative_for_valid_distributions(rlox_module):
    """KL(p||q) must be non-negative (Gibbs' inequality)."""
    rng = np.random.default_rng(13)
    # Log-softmax to get valid log-probability distributions
    logits_p = rng.standard_normal(50).astype(np.float64)
    logits_q = rng.standard_normal(50).astype(np.float64)
    log_p = logits_p - np.log(np.sum(np.exp(logits_p)))
    log_q = logits_q - np.log(np.sum(np.exp(logits_q)))

    kl = float(rlox_module.compute_token_kl(log_p, log_q))
    assert kl >= -1e-9, f"KL divergence should be non-negative, got {kl}"


@pytest.mark.correctness
def test_token_kl_zero_for_identical_distributions(rlox_module):
    """KL(p||p) == 0."""
    rng = np.random.default_rng(0)
    log_p = -np.abs(rng.standard_normal(64)).astype(np.float64)
    kl = float(rlox_module.compute_token_kl(log_p, log_p))
    assert abs(kl) < 1e-10, f"KL(p||p) should be 0, got {kl}"


# ---------------------------------------------------------------------------
# 3. GRPO with different group sizes
# ---------------------------------------------------------------------------


@pytest.mark.correctness
@pytest.mark.parametrize("group_size", [2, 4, 8, 16])
def test_grpo_group_sizes(group_size, rlox_module):
    """GRPO must work correctly for various group sizes."""
    rng = np.random.default_rng(group_size * 7)
    rewards = rng.standard_normal(group_size).astype(np.float64)

    ref = reference_grpo_numpy(rewards)
    rlox_result = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)

    np.testing.assert_allclose(rlox_result, ref, rtol=1e-6)


@pytest.mark.correctness
def test_grpo_output_shape_matches_input(rlox_module):
    """GRPO output shape must equal input shape."""
    for k in [2, 4, 8, 16, 32]:
        rng = np.random.default_rng(k)
        rewards = rng.standard_normal(k).astype(np.float64)
        result = np.asarray(rlox_module.compute_group_advantages(rewards))
        assert result.shape == (k,), f"Shape mismatch for k={k}: got {result.shape}"


# ---------------------------------------------------------------------------
# 4. Token KL with different sequence lengths
# ---------------------------------------------------------------------------


@pytest.mark.correctness
@pytest.mark.parametrize("seq_len", [1, 16, 256, 4096])
def test_token_kl_various_lengths(seq_len, rlox_module):
    """compute_token_kl must handle sequences of various lengths."""
    rng = np.random.default_rng(seq_len)
    log_p = -np.abs(rng.standard_normal(seq_len)).astype(np.float64)
    log_q = -np.abs(rng.standard_normal(seq_len)).astype(np.float64)

    ref = reference_token_kl_numpy(log_p, log_q)
    rlox_result = float(rlox_module.compute_token_kl(log_p, log_q))

    assert abs(rlox_result - ref) / (abs(ref) + 1e-10) < 1e-5


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_grpo_uniform_rewards(rlox_module):
    """Uniform rewards must produce all-zero advantages (zero std)."""
    rewards = np.ones(8, dtype=np.float64) * 3.14
    result = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)
    np.testing.assert_allclose(result, 0.0, atol=1e-12,
                                err_msg="Uniform rewards should give zero advantages")


@pytest.mark.correctness
def test_grpo_single_completion(rlox_module):
    """Single-element group: normalizing one element should produce 0."""
    rewards = np.array([5.0], dtype=np.float64)
    result = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)
    # std of a single element is 0, so result should be 0 (not NaN)
    assert not np.any(np.isnan(result)), "Single-element GRPO should not produce NaN"
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


@pytest.mark.correctness
def test_grpo_two_equal_completions(rlox_module):
    """Two identical completions: std is zero, should produce zeros not NaN."""
    rewards = np.array([1.0, 1.0], dtype=np.float64)
    result = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)
    assert not np.any(np.isnan(result))
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


@pytest.mark.correctness
def test_grpo_very_long_group(rlox_module):
    """GRPO on 1024-element group must match reference without overflow."""
    rng = np.random.default_rng(0)
    rewards = rng.standard_normal(1024).astype(np.float64)
    ref = reference_grpo_numpy(rewards)
    result = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)
    np.testing.assert_allclose(result, ref, rtol=1e-6)
    assert not np.any(np.isnan(result))


@pytest.mark.correctness
def test_token_kl_single_token(rlox_module):
    """Single-token KL must equal the exact analytic value."""
    log_p = np.array([-1.0], dtype=np.float64)
    log_q = np.array([-2.0], dtype=np.float64)
    expected = float(np.exp(-1.0) * (-1.0 - (-2.0)))  # p * log(p/q)
    result = float(rlox_module.compute_token_kl(log_p, log_q))
    assert abs(result - expected) < 1e-12


@pytest.mark.correctness
def test_token_kl_no_nan_for_extreme_log_probs(rlox_module):
    """Very negative log-probabilities (near-zero prob) must not produce NaN."""
    log_p = np.full(32, -100.0, dtype=np.float64)
    log_q = np.full(32, -200.0, dtype=np.float64)
    result = float(rlox_module.compute_token_kl(log_p, log_q))
    assert not np.isnan(result), "Extreme log-probs should not produce NaN"


@pytest.mark.correctness
def test_grpo_deterministic(rlox_module):
    """GRPO must produce identical output for identical input across 10 calls."""
    rng = np.random.default_rng(42)
    rewards = rng.standard_normal(16).astype(np.float64)

    first = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64).copy()
    for _ in range(9):
        result = np.asarray(rlox_module.compute_group_advantages(rewards), dtype=np.float64)
        np.testing.assert_array_equal(result, first)


@pytest.mark.correctness
def test_token_kl_deterministic(rlox_module):
    """Token KL must produce identical results for identical inputs."""
    rng = np.random.default_rng(7)
    log_p = rng.standard_normal(128).astype(np.float64)
    log_q = rng.standard_normal(128).astype(np.float64)

    first = float(rlox_module.compute_token_kl(log_p, log_q))
    for _ in range(9):
        result = float(rlox_module.compute_token_kl(log_p, log_q))
        assert result == first
