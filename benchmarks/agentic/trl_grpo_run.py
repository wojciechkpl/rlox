"""trl_grpo_run.py — Single-GPU GRPO training runner for the rlox benchmark.

Replaces the prime-rl multi-GPU setup with TRL GRPOTrainer, which colocates
policy, generation, and training in ONE process on ONE GPU.

Key design decisions
--------------------
* ``use_vllm=False``: TRL 1.5.1 supports vLLM ≤0.18 only; we have 0.22.
  HF generate avoids the incompatibility and is more memory-friendly on a
  single 32 GB card.
* LoRA (r=32, alpha=64) on all attention+MLP projections keeps the trainable
  parameter count low while covering the full network surface.
* ``bf16=True`` + ``gradient_checkpointing=True`` for memory efficiency.
* Reward function imports from ``rlox_agent`` (``verifiers_adapter`` and
  ``adversarial_corpus``) — the canonical single-source backend.
* Dataset: 50 MBPP problems loaded via ``rlox_verify._load_mbpp_problems``
  (seeded, deterministic slice) repeated to give the trainer enough rows;
  each row carries a ``tests`` column that the reward function reads via
  ``**kwargs``.  Falls back to the embedded 40-problem list when MBPP is
  unavailable (set RLOX_NO_MBPP=1 to force the fallback).

Usage (single-GPU, no accelerate launcher needed):

    CUDA_VISIBLE_DEVICES=0 python trl_grpo_run.py \\
        --backend in_loop \\
        --adversarial-fraction 0.0 \\
        --max-steps 3 \\
        --group-size 4 \\
        --output-dir /home/wk/rlox/benchmarks/agentic/trl_smoke_out

TRL / transformers API notes (versions in the prime-rl venv)
------------------------------------------------------------
* TRL 1.5.1: GRPOTrainer expects ``reward_funcs`` as
  ``Callable[[prompts, completions, **kwargs], list[float]]``.
  Extra dataset columns (anything NOT ``prompt`` / ``completion`` /
  ``completion_ids``) are forwarded to the reward function via ``**kwargs``,
  so ``tests`` arrives automatically.
* Transformers 5.6.2: ``AutoTokenizer.from_pretrained`` is still the right
  call; ``processing_class`` in ``GRPOTrainer.__init__`` accepts a
  ``PreTrainedTokenizerBase`` directly.
* ``remove_unused_columns`` defaults to ``False`` in ``GRPOConfig``, so the
  ``tests`` column is preserved without any special override.

Code extraction (FIX 2)
-----------------------
Qwen3 emits reasoning preambles before code.  ``extract_python_code`` strips
these so the reward function actually executes the intended function body.
Adversarial samples bypass extraction — they are already raw code snippets.

GPU-utilization capture
-----------------------
A background daemon thread polls ``nvidia-smi --query-gpu=utilization.gpu``
every ~1 second during training and computes ``mean_gpu_util``,
``min_gpu_util``, and ``gpu_util_samples`` which are written into
``summary.json``.  These fields quantify the GPU-idle effect: under in-loop
adversarial injection the GPU idles while sandboxed code runs, whereas the
Treatment (rlox server) keeps the GPU busy.  The sampler is only started if
CUDA is available; on CPU-only hosts all three fields are set to ``null``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("trl_grpo_run")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# LoRA target modules covering attention projections and MLP projections.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Per-completion execution timeout forwarded to rlox_agent.verifiers_adapter.
EXECUTION_TIMEOUT_SECS: float = 5.0

# Dataset repetitions: repeat the problem set enough times so TRL always
# has more rows than batch_size * group_size * max_steps.
# With 50 MBPP problems and DATASET_REPEAT=4 we get 200 rows — sufficient
# for up to 60 steps × group_size 4 = 240 rollouts (TRL samples with
# replacement when the dataset is exhausted, so 200 rows is fine).
DATASET_REPEAT: int = 4

# GPU utilisation polling interval in seconds.
GPU_POLL_INTERVAL_SECS: float = 1.0

# Default path to the adversarial corpus, relative to the repo root discovered
# at runtime.  Can be overridden by --adversarial-corpus on the CLI, or by the
# RLOX_ADVERSARIAL_CORPUS env var (CLI takes precedence over env var).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADVERSARIAL_CORPUS: Path = (
    _REPO_ROOT / "benchmarks" / "agentic" / "corpus" / "adversarial_corpus_v1.json"
)


# ---------------------------------------------------------------------------
# Code extraction (FIX 2)
# ---------------------------------------------------------------------------

# Regex patterns for fenced code blocks
_FENCED_PYTHON = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_FENCED_GENERIC = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str:
    """Extract Python code from a model completion that may contain preamble text.

    Extraction priority (first match wins):
    1. Last ````python ... ```` fenced block.
    2. Last ```` ``` ... ``` ```` fenced block (language-agnostic).
    3. From the first ``def `` or ``import `` line to the end of the string.
    4. Original text unchanged (no preamble detected).

    This is intentionally applied only to model completions, NOT to injected
    adversarial samples (those are raw, pre-validated code).
    """
    # 1. Try python-fenced block — take the last one in case the model emits
    #    multiple (reasoning vs actual answer pattern).
    python_matches = _FENCED_PYTHON.findall(text)
    if python_matches:
        return python_matches[-1].strip()

    # 2. Try generic fenced block.
    generic_matches = _FENCED_GENERIC.findall(text)
    if generic_matches:
        return generic_matches[-1].strip()

    # 3. Fall back to first def/import line.
    for i, line in enumerate(text.splitlines()):
        if (
            line.startswith("def ")
            or line.startswith("import ")
            or line.startswith("from ")
        ):
            return "\n".join(text.splitlines()[i:]).strip()

    # 4. Return as-is — no preamble markers found.
    return text


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    import numpy as np  # lazy: not available in test env
    import torch  # lazy: not available in test env

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_dataset(group_size: int) -> "datasets.Dataset":  # noqa: F821
    """Return a datasets.Dataset with ``prompt`` (list[dict]) and ``tests`` columns.

    Loads a fixed, seeded slice of MBPP via ``rlox_verify._load_mbpp_problems``
    and repeats it ``DATASET_REPEAT`` times so TRL always has more rows than
    ``batch_size * group_size * max_steps``.  Each ``prompt`` is a list of chat
    messages so TRL's ``is_conversational`` check passes and
    ``apply_chat_template`` is called automatically inside ``_tokenize_prompts``.

    Falls back to the embedded ``_FALLBACK_PROBLEMS`` list when MBPP is
    unavailable (network error, ``RLOX_NO_MBPP=1``, etc.).
    """
    import datasets as _ds

    from rlox_verify import _load_mbpp_problems

    # _load_mbpp_problems handles MBPP→fallback gracefully and logs which path
    # was taken.
    mbpp_problems = _load_mbpp_problems()
    n_unique = len(mbpp_problems)

    rows: list[dict[str, Any]] = []
    for problem in mbpp_problems * DATASET_REPEAT:
        rows.append(
            {
                # conversational format — list of message dicts
                "prompt": [{"role": "user", "content": problem["prompt"]}],
                "tests": problem["tests"],
            }
        )

    ds = _ds.Dataset.from_list(rows)
    logger.info(
        "Dataset built: %d unique problems × %d repeats = %d rows (group_size=%d)",
        n_unique,
        DATASET_REPEAT,
        len(ds),
        group_size,
    )
    return ds


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def make_reward_func(
    backend: str,
    adversarial_fraction: float,
    seed: int,
    rlox_server_url: str,
    corpus_path: Path | str | None = None,
) -> "Callable[..., list[float]]":  # noqa: F821
    """Factory returning the TRL-compatible reward function.

    The returned function has signature:
        reward(prompts, completions, **kwargs) -> list[float]

    TRL 1.5.1 forwards all non-standard dataset columns as lists via **kwargs,
    so ``tests`` (a list of str, one per example in the batch) arrives here
    automatically.

    Adversarial injection is seeded and stateless across calls (the injector
    carries its own RNG), so results are reproducible for a fixed seed.

    Args:
        corpus_path: path to the adversarial corpus JSON.  When
            ``adversarial_fraction > 0`` this is REQUIRED.  Resolution order:
            1. explicit ``corpus_path`` argument (CLI ``--adversarial-corpus``)
            2. ``RLOX_ADVERSARIAL_CORPUS`` env var
            3. hard-failure — FileNotFoundError is raised so the misconfiguration
               is never silently swallowed.
    """
    from rlox_agent.adversarial_corpus import (
        AdversarialCorpus,
        AdversarialInjector,
        AdversarialSample,
    )
    from rlox_agent.verifiers_adapter import call_rlox_server, extract_text, run_in_loop

    injector: AdversarialInjector | None = None
    if adversarial_fraction > 0.0:
        # Resolve corpus path: explicit arg > env var > hard failure.
        resolved_corpus: str | None = (
            str(corpus_path)
            if corpus_path is not None
            else os.environ.get("RLOX_ADVERSARIAL_CORPUS")
        )
        if resolved_corpus is None:
            raise FileNotFoundError(
                f"adversarial_fraction={adversarial_fraction:.2f} but no corpus path was "
                "provided.  Pass --adversarial-corpus <PATH> or set "
                "RLOX_ADVERSARIAL_CORPUS env var."
            )
        if not Path(resolved_corpus).exists():
            raise FileNotFoundError(
                f"Adversarial corpus not found at {resolved_corpus!r}.  "
                "Pass --adversarial-corpus <PATH> pointing to adversarial_corpus_v1.json."
            )
        corpus = AdversarialCorpus.load(resolved_corpus)
        injector = AdversarialInjector(
            corpus=corpus, fraction=adversarial_fraction, seed=seed
        )
        logger.info(
            "injection ACTIVE: fraction=%.2f corpus=%s seed=%d",
            adversarial_fraction,
            resolved_corpus,
            seed,
        )

    _backend = backend
    _url = rlox_server_url
    _timeout = EXECUTION_TIMEOUT_SECS

    def reward_func(
        prompts: list[list[dict] | str],
        completions: list[list[dict] | str],
        **kwargs: Any,
    ) -> list[float]:
        """Score each completion against its unit tests.

        ``tests`` arrives as a list[str] via kwargs because the dataset has a
        ``tests`` column and TRL 1.5.1 forwards all non-standard columns.
        """
        tests_list: list[str] = kwargs.get("tests", [""] * len(prompts))
        rewards: list[float] = []

        for prompt, completion, tests in zip(
            prompts, completions, tests_list, strict=False
        ):
            # Build a lightweight task dict for the injector.
            task: Any = {"prompt": prompt, "answer": ""}
            is_adversarial = False

            if injector is not None:
                task, is_adversarial = injector.maybe_inject(task)

            if isinstance(task, AdversarialSample):
                # Adversarial samples are pre-formed code — run verbatim, no extraction.
                code_text = task.code
                tests_text = ""
            else:
                # ``completions`` in TRL 1.5.1 are lists of message dicts
                # (conversational) when the prompt was conversational.
                raw_text = extract_text(completion)
                # Strip reasoning preambles emitted by Qwen3 before the actual code.
                code_text = extract_python_code(raw_text)
                tests_text = tests

            try:
                if _backend == "rlox":
                    r = call_rlox_server(
                        code_text, tests_text, is_adversarial, _url, _timeout
                    )
                else:
                    r = run_in_loop(code_text, tests_text, _timeout)
            except Exception as exc:
                logger.warning("Reward computation error: %s", exc)
                r = 0.0

            rewards.append(r)

        return rewards

    return reward_func


# ---------------------------------------------------------------------------
# Metrics JSONL callback
# ---------------------------------------------------------------------------


def _make_metrics_callback(
    output_dir: Path,
) -> tuple["TrainerCallback", "Callable[[], list[dict]]"]:  # noqa: F821
    """Build a TrainerCallback that writes per-step metrics to a JSONL file.

    Returns ``(callback_instance, close_fn)`` where ``close_fn()`` closes the
    file handle and returns the list of logged rows.

    We use a factory rather than a class so we can import TrainerCallback
    lazily (transformers may not be importable at module level on some envs).
    """
    from transformers import TrainerCallback

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "metrics.jsonl"
    fh = jsonl_path.open("w", encoding="utf-8")
    rows: list[dict[str, Any]] = []

    class _MetricsCallback(TrainerCallback):
        """Writes per-step reward metrics to a JSONL file."""

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, float] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None:
                return
            step = state.global_step
            # TRL 1.5.1 emits both "reward" and "rewards/reward_func/mean".
            # Older TRL versions may use "rewards/reward_func" or "train/reward".
            # Use explicit key-presence checks (not `or`) so that a value of 0.0
            # is captured correctly and not short-circuited to a fallback.
            _REWARD_KEYS = (
                "reward",
                "rewards/reward_func/mean",
                "rewards/reward_func",
                "train/reward",
            )
            mean_reward: float = 0.0
            has_reward_key = False
            for _key in _REWARD_KEYS:
                if _key in logs:
                    mean_reward = float(logs[_key])
                    has_reward_key = True
                    break
            row = {
                "step": step,
                "mean_reward": mean_reward,
                # Sentinel so _compute_reward_summary can filter out the
                # terminal train_runtime log event that TRL emits after training.
                "_has_reward": has_reward_key,
                **logs,
            }
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            logger.info("Step %d | mean_reward=%.4f | logs=%s", step, mean_reward, logs)

    def close() -> list[dict[str, Any]]:
        fh.close()
        return rows

    return _MetricsCallback(), close


# ---------------------------------------------------------------------------
# Reward summary helper
# ---------------------------------------------------------------------------


def _compute_reward_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute final_reward, mean_reward, and reward_curve from per-step metric rows.

    Filters to only rows that have ``_has_reward=True`` (set by
    _MetricsCallback.on_log when a recognised reward key was found in the TRL
    log dict).  This excludes TRL's terminal ``train_runtime`` summary log event
    which has no reward key and would otherwise corrupt ``final_reward``.

    Returns a dict with:
        final_reward  — reward at the last training step (0.0 if no rows)
        mean_reward   — arithmetic mean over training-step rows (0.0 if none)
        reward_curve  — list[float] of per-step rewards in emission order
        mean_reward_last — alias for final_reward (backward compat)
    """
    # Keep only rows that originated from a training-step log event.
    training_rows = [r for r in rows if r.get("_has_reward")]

    if not training_rows:
        return {
            "final_reward": 0.0,
            "mean_reward": 0.0,
            "reward_curve": [],
            "mean_reward_last": 0.0,
        }

    reward_curve = [float(r.get("mean_reward", 0.0)) for r in training_rows]
    final_reward = reward_curve[-1]
    mean_reward = sum(reward_curve) / len(reward_curve)
    return {
        "final_reward": final_reward,
        "mean_reward": round(mean_reward, 6),
        "reward_curve": reward_curve,
        "mean_reward_last": final_reward,  # backward-compat alias
    }


# ---------------------------------------------------------------------------
# GPU-utilisation background sampler
# ---------------------------------------------------------------------------


def _start_gpu_util_sampler(
    stop_event: threading.Event,
    interval_secs: float = GPU_POLL_INTERVAL_SECS,
) -> list[float]:
    """Start a daemon thread that polls ``nvidia-smi`` every *interval_secs*.

    The thread appends GPU-utilisation readings (0–100) to the returned list
    while ``stop_event`` is not set.  The list is shared by reference so the
    caller can inspect it after calling ``stop_event.set()``.

    Returns:
        A ``list[float]`` that the background thread appends readings to.
        The thread is a daemon so it will be killed if the main process exits.

    Note:
        If ``nvidia-smi`` is unavailable (no GPU, no CUDA toolkit) the thread
        logs a single WARNING and exits immediately — the list stays empty and
        the caller treats that as ``gpu_util_samples == 0``.
    """
    samples: list[float] = []

    def _poll() -> None:
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        line = line.strip()
                        if line:
                            try:
                                samples.append(float(line.split()[0]))
                            except ValueError:
                                pass
            except FileNotFoundError:
                logger.warning(
                    "nvidia-smi not found — GPU utilisation will not be recorded"
                )
                return
            except Exception as exc:
                logger.debug("GPU sampler poll error (non-fatal): %s", exc)
            stop_event.wait(timeout=interval_secs)

    t = threading.Thread(target=_poll, daemon=True, name="gpu_util_sampler")
    t.start()
    return samples


def _summarise_gpu_util(samples: list[float]) -> dict[str, Any]:
    """Convert a list of GPU-util readings into summary fields.

    Returns:
        dict with keys ``mean_gpu_util``, ``min_gpu_util``,
        ``gpu_util_samples``.  All three are ``None`` when ``samples`` is empty
        (no GPU or sampler error), so JSON serialisation uses ``null`` which
        aggregate_sweep.py can handle gracefully.
    """
    if not samples:
        return {
            "mean_gpu_util": None,
            "min_gpu_util": None,
            "gpu_util_samples": 0,
        }
    return {
        "mean_gpu_util": round(sum(samples) / len(samples), 2),
        "min_gpu_util": round(min(samples), 2),
        "gpu_util_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(
    backend: str,
    adversarial_fraction: float,
    seed: int,
    max_steps: int,
    group_size: int,
    rlox_server_url: str,
    output_dir: Path,
    corpus_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run GRPO training and return a summary dict."""
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    seed_everything(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer from %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%r)", tokenizer.eos_token)

    # ------------------------------------------------------------------
    # LoRA config
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    logger.info("LoRA config: r=%d, alpha=%d, targets=%s", 32, 64, LORA_TARGET_MODULES)

    # ------------------------------------------------------------------
    # GRPOConfig
    # The key memory-safety knobs for a 32 GB card:
    #   - num_generations == per_device_train_batch_size == group_size
    #     (one row per step, group_size completions generated per row)
    #   - max_completion_length=256: short to keep KV cache small
    #   - max_prompt_length=512
    #   - gradient_checkpointing=True: trades compute for activation memory
    #   - bf16=True: halves weight memory vs fp32
    #   - use_vllm=False: avoids vLLM 0.22 incompatibility; HF generate is
    #     more memory-friendly in the single-GPU colocated setup
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        # --- generation ---
        num_generations=group_size,
        max_completion_length=256,
        # max_prompt_length does not exist in TRL 1.5.1 (removed in the
        # refactor that introduced generation_kwargs); prompt truncation is
        # handled by the tokenizer's truncation_side="left" setting instead.
        use_vllm=False,
        # --- training batch ---
        # per_device_train_batch_size must divide evenly into num_generations;
        # setting equal means one optimiser step per generation batch.
        per_device_train_batch_size=group_size,
        gradient_accumulation_steps=1,
        # --- steps ---
        max_steps=max_steps,
        # --- optimiser ---
        learning_rate=1e-5,
        # --- memory ---
        bf16=True,
        gradient_checkpointing=True,
        # --- logging ---
        logging_steps=1,
        logging_strategy="steps",
        report_to="none",
        # --- checkpointing (disable for smoke test) ---
        save_strategy="no",
        # --- reproducibility ---
        seed=seed,
        data_seed=seed,
        # --- misc ---
        remove_unused_columns=False,  # keep ``tests`` column for reward func
        disable_tqdm=False,
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = build_dataset(group_size)

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------
    reward_fn = make_reward_func(
        backend=backend,
        adversarial_fraction=adversarial_fraction,
        seed=seed,
        rlox_server_url=rlox_server_url,
        corpus_path=corpus_path,
    )

    # ------------------------------------------------------------------
    # Metrics callback
    # ------------------------------------------------------------------
    metrics_cb, metrics_close = _make_metrics_callback(output_dir)

    # ------------------------------------------------------------------
    # Trainer
    # GRPOTrainer accepts ``peft_config`` and wraps the model with PEFT
    # internally.  We pass the model as a string so TRL can load it with
    # its own ``create_model_from_path`` helper (handles device_map correctly
    # in single-GPU mode).
    # ------------------------------------------------------------------
    logger.info("Instantiating GRPOTrainer (model=%s)", MODEL_NAME)
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[metrics_cb],
    )

    # ------------------------------------------------------------------
    # Train  (with background GPU-util sampler)
    # ------------------------------------------------------------------
    logger.info(
        "Starting GRPO training: max_steps=%d, group_size=%d", max_steps, group_size
    )
    survived = True
    exception_msg: str | None = None

    # Start GPU-util sampler before training begins.
    _gpu_stop = threading.Event()
    gpu_samples = _start_gpu_util_sampler(_gpu_stop)

    t0 = time.perf_counter()

    try:
        trainer.train()
    except Exception as exc:
        survived = False
        exception_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Training failed: %s", exception_msg)
    finally:
        # Always stop the sampler so it doesn't keep running after training.
        _gpu_stop.set()

    elapsed = time.perf_counter() - t0
    rows = metrics_close()

    gpu_util_summary = _summarise_gpu_util(gpu_samples)
    logger.info(
        "GPU-util summary: mean=%.1f%% min=%.1f%% samples=%d",
        gpu_util_summary.get("mean_gpu_util") or 0.0,
        gpu_util_summary.get("min_gpu_util") or 0.0,
        gpu_util_summary.get("gpu_util_samples", 0),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    completed_steps = trainer.state.global_step if hasattr(trainer, "state") else 0
    reward_summary = _compute_reward_summary(rows)

    summary: dict[str, Any] = {
        "completed_steps": completed_steps,
        "survived": survived,
        "elapsed_secs": round(elapsed, 2),
        "backend": backend,
        "adversarial_fraction": adversarial_fraction,
        "seed": seed,
        "max_steps": max_steps,
        "group_size": group_size,
        **reward_summary,
        **gpu_util_summary,
    }
    if exception_msg is not None:
        summary["exception"] = exception_msg

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", summary_path)
    logger.info("Summary: %s", json.dumps(summary, indent=2))

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-GPU TRL GRPO training runner for the rlox benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backend",
        choices=["in_loop", "rlox"],
        default="in_loop",
        help="Execution backend: in_loop=Baseline subprocess, rlox=Treatment server.",
    )
    p.add_argument(
        "--adversarial-fraction",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Fraction of tasks to replace with adversarial samples (0.0=never).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum number of GRPO training steps.",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of completions generated per prompt (GRPO group size).",
    )
    p.add_argument(
        "--rlox-server-url",
        type=str,
        default="http://localhost:8231",
        help="Base URL of the rlox verify server (used when --backend=rlox).",
    )
    p.add_argument(
        "--adversarial-corpus",
        type=Path,
        default=DEFAULT_ADVERSARIAL_CORPUS,
        metavar="PATH",
        help=(
            "Path to adversarial_corpus_v1.json.  Required when "
            "--adversarial-fraction > 0.  "
            "RLOX_ADVERSARIAL_CORPUS env var is an optional override but this "
            "CLI flag takes precedence."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/wk/rlox/benchmarks/agentic/trl_grpo_out"),
        help="Directory for metrics JSONL and summary JSON.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logger.info(
        "trl_grpo_run | backend=%s | adversarial_fraction=%.2f | seed=%d | "
        "max_steps=%d | group_size=%d | adversarial_corpus=%s | output_dir=%s",
        args.backend,
        args.adversarial_fraction,
        args.seed,
        args.max_steps,
        args.group_size,
        args.adversarial_corpus,
        args.output_dir,
    )

    summary = train(
        backend=args.backend,
        adversarial_fraction=args.adversarial_fraction,
        seed=args.seed,
        max_steps=args.max_steps,
        group_size=args.group_size,
        rlox_server_url=args.rlox_server_url,
        output_dir=args.output_dir,
        corpus_path=args.adversarial_corpus,
    )

    # Non-zero exit code when training crashed, so CI pipelines can detect it.
    if not summary.get("survived", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
