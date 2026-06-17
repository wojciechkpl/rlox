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
* Reward function reuses ``rlox_verify._backend`` and ``_adversarial``
  directly — the package is pip-installed in the prime-rl venv.
* Dataset: the 8 MBPP-style problems from ``rlox_verify._CODING_PROBLEMS``
  repeated to give the trainer enough rows; each row carries a ``tests``
  column that the reward function reads via ``**kwargs``.

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
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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

# Per-completion execution timeout forwarded to rlox_verify._backend.
EXECUTION_TIMEOUT_SECS: float = 5.0

# Dataset repetitions: repeat the 8-problem set enough times so TRL always
# has more rows than batch_size * group_size * max_steps.
DATASET_REPEAT: int = 20


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
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

    Each ``prompt`` is a list of chat messages so TRL's ``is_conversational``
    check passes and ``apply_chat_template`` is called automatically inside
    ``_tokenize_prompts``.
    """
    import datasets as _ds

    from rlox_verify import _CODING_PROBLEMS

    rows: list[dict[str, Any]] = []
    for problem in _CODING_PROBLEMS * DATASET_REPEAT:
        rows.append(
            {
                # conversational format — list of message dicts
                "prompt": [{"role": "user", "content": problem["prompt"]}],
                "tests": problem["tests"],
            }
        )

    ds = _ds.Dataset.from_list(rows)
    logger.info("Dataset built: %d rows (group_size=%d)", len(ds), group_size)
    return ds


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def make_reward_func(
    backend: str,
    adversarial_fraction: float,
    seed: int,
    rlox_server_url: str,
) -> "Callable[..., list[float]]":  # noqa: F821
    """Factory returning the TRL-compatible reward function.

    The returned function has signature:
        reward(prompts, completions, **kwargs) -> list[float]

    TRL 1.5.1 forwards all non-standard dataset columns as lists via **kwargs,
    so ``tests`` (a list of str, one per example in the batch) arrives here
    automatically.

    Adversarial injection is seeded and stateless across calls (the injector
    carries its own RNG), so results are reproducible for a fixed seed.
    """
    from rlox_verify._adversarial import AdversarialCorpus, AdversarialInjector, AdversarialSample
    from rlox_verify._backend import call_rlox_server, extract_text, run_in_loop

    injector: AdversarialInjector | None = None
    if adversarial_fraction > 0.0:
        # Corpus path is required when fraction > 0 — caller must supply it via
        # RLOX_ADVERSARIAL_CORPUS env var or we skip injection with a warning.
        corpus_path = os.environ.get("RLOX_ADVERSARIAL_CORPUS")
        if corpus_path is None:
            logger.warning(
                "adversarial_fraction=%.2f but RLOX_ADVERSARIAL_CORPUS env var is not set; "
                "falling back to fraction=0.0",
                adversarial_fraction,
            )
        else:
            corpus = AdversarialCorpus.load(corpus_path)
            injector = AdversarialInjector(corpus=corpus, fraction=adversarial_fraction, seed=seed)
            logger.info(
                "AdversarialInjector ready: fraction=%.2f, corpus=%s, seed=%d",
                adversarial_fraction,
                corpus_path,
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

        for prompt, completion, tests in zip(prompts, completions, tests_list, strict=False):
            # Build a lightweight task dict for the injector.
            task: Any = {"prompt": prompt, "answer": ""}
            is_adversarial = False

            if injector is not None:
                task, is_adversarial = injector.maybe_inject(task)

            if isinstance(task, AdversarialSample):
                code_text = task.code
                tests_text = ""
            else:
                # ``completions`` in TRL 1.5.1 are lists of message dicts
                # (conversational) when the prompt was conversational.
                code_text = extract_text(completion)
                tests_text = tests

            try:
                if _backend == "rlox":
                    r = call_rlox_server(code_text, tests_text, is_adversarial, _url, _timeout)
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


def _make_metrics_callback(output_dir: Path) -> tuple["TrainerCallback", "Callable[[], list[dict]]"]:  # noqa: F821
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
            # TRL 1.5.1 logs reward under "reward"; older or future versions
            # may use "rewards/reward_func" or "train/reward".  Try all.
            mean_reward = float(
                logs.get("reward")
                or logs.get("rewards/reward_func")
                or logs.get("train/reward")
                or 0.0
            )
            row = {"step": step, "mean_reward": mean_reward, **logs}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            logger.info("Step %d | mean_reward=%.4f | logs=%s", step, mean_reward, logs)

    def close() -> list[dict[str, Any]]:
        fh.close()
        return rows

    return _MetricsCallback(), close


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
) -> dict[str, Any]:
    """Run GRPO training and return a summary dict."""
    import datasets as _ds
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
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting GRPO training: max_steps=%d, group_size=%d", max_steps, group_size)
    survived = True
    exception_msg: str | None = None
    t0 = time.perf_counter()

    try:
        trainer.train()
    except Exception as exc:
        survived = False
        exception_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Training failed: %s", exception_msg)

    elapsed = time.perf_counter() - t0
    rows = metrics_close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    completed_steps = trainer.state.global_step if hasattr(trainer, "state") else 0
    mean_reward_last = rows[-1].get("mean_reward", 0.0) if rows else 0.0

    summary: dict[str, Any] = {
        "completed_steps": completed_steps,
        "survived": survived,
        "mean_reward_last": mean_reward_last,
        "elapsed_secs": round(elapsed, 2),
        "backend": backend,
        "adversarial_fraction": adversarial_fraction,
        "seed": seed,
        "max_steps": max_steps,
        "group_size": group_size,
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
        "max_steps=%d | group_size=%d | output_dir=%s",
        args.backend,
        args.adversarial_fraction,
        args.seed,
        args.max_steps,
        args.group_size,
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
    )

    # Non-zero exit code when training crashed, so CI pipelines can detect it.
    if not summary.get("survived", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
