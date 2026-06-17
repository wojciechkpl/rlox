"""_backend.py — Low-level execution helpers for rlox_verify.

These are the pure-Python, stdlib-only backend functions extracted from
``rlox.agentic.verifiers_adapter`` so that ``rlox_verify`` can operate as a
self-contained installable package without depending on the Rust-extension
``rlox`` package being installed.

The canonical implementations live in ``python/rlox/agentic/verifiers_adapter.py``.
Changes to dispatch / HTTP / subprocess logic MUST be kept in sync manually.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import httpx

logger = logging.getLogger(__name__)


def extract_text(messages: list[dict] | str) -> str:
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


def run_in_loop(code: str, tests: str, timeout: float) -> float:
    """Execute *code* against *tests* in the current venv via subprocess.

    Baseline path — no isolation layer, no network call.
    Uses ``sys.executable`` so the correct venv interpreter is invoked.

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


def call_rlox_server(
    code: str,
    tests: str,
    is_adversarial: bool,
    server_url: str,
    timeout: float,
) -> float:
    """POST ``{"code": code, "tests": tests, "is_adversarial": ...}`` to
    ``{server_url}/verify``.

    Returns the ``"reward"`` field from the JSON response, or 0.0 on any
    exception (logged as WARNING so server outages are visible in training logs).
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
