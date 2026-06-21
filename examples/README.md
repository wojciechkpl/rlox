# rlox examples

Self-contained tutorial scripts. Each file has a module docstring that explains
what it demonstrates. All core-RL examples require the compiled Rust extension
(`maturin develop --release` from repo root). The agentic example is torch-free
and extension-free.

---

## Core RL

These scripts use the `rlox` Python package (backed by the Rust extension) and,
where noted, PyTorch. Build the extension once before running any of them:

```bash
maturin develop --release
```

| Script | What it teaches | Run command |
|---|---|---|
| `ppo_cartpole.py` | PPO on CartPole-v1 — the three-line entry point | `.venv/bin/python examples/ppo_cartpole.py` |
| `sac_pendulum.py` | SAC on Pendulum-v1 — off-policy continuous control | `.venv/bin/python examples/sac_pendulum.py` |
| `fast_gae.py` | Rust GAE and GRPO advantage computation (drop-in, 142x speedup) | `.venv/bin/python examples/fast_gae.py` |
| `vec_env_throughput.py` | Rust-parallel VecEnv vs gymnasium SyncVectorEnv (2.7M steps/s) | `.venv/bin/python examples/vec_env_throughput.py` |
| `custom_environment.py` | Wrapping any Gymnasium env with the rlox Trainer | `.venv/bin/python examples/custom_environment.py` |
| `custom_reward_grpo.py` | GRPO with a user-supplied reward function for LLM post-training | `.venv/bin/python examples/custom_reward_grpo.py` |
| `intrinsic_motivation.py` | RND intrinsic motivation (Burda et al. 2019) + PPO on MountainCar | `.venv/bin/python examples/intrinsic_motivation.py` |
| `reward_shaping.py` | Potential-based reward shaping (PBRS) + PPO on CartPole | `.venv/bin/python examples/reward_shaping.py` |
| `meta_learning.py` | Reptile meta-learning across CartPole variants (Rust reptile_update) | `.venv/bin/python examples/meta_learning.py` |

---

## Agentic sandbox

Demonstrates the Baseline↔Treatment one-key swap that is the core of the rlox
agentic benchmark. **No torch, no compiled extension required** — only `httpx`
and the `rlox_agent` package (loaded from `python/` via `sys.path`).

| Script | What it teaches | Run command |
|---|---|---|
| `agentic_sandbox_verify.py` | Baseline in-process exec vs Treatment hard-isolated `/verify` sandbox; adversarial corpus load + injection | `python examples/agentic_sandbox_verify.py` |

The script has two sections:

- **Section A (benign task):** scores a trivial `add()` function via both
  backends. `run_in_loop` (Baseline) always runs and should print reward 1.0.
  `call_rlox_server` (Treatment) also prints 1.0 when the server is up; it
  degrades gracefully with a startup hint when the server is down.

- **Section B (adversarial containment):** loads
  `benchmarks/agentic/corpus/adversarial_corpus_v1.json`, demonstrates
  deterministic injection with `AdversarialInjector`, then scores an
  infinite-loop sample via Treatment → reward 0.0, contained. The Baseline
  path is intentionally skipped for adversarial samples (it would stall the
  training loop; see `benchmarks/agentic/oq3_pilot.py`).

### Dependencies

```
pip install httpx        # only extra needed; stdlib provides the rest
```

The Treatment backend (`call_rlox_server`) requires `rlox-verify-server` to be
running. Start it on Linux with cgroup isolation:

```bash
systemd-run --user --unit=rlox-verify -p TasksMax=4096 -p MemoryMax=12G \
  target/release/rlox-verify-server --port 8231 --timeout-secs 5
```

### Flags

```
--server-url URL    Base URL of the server (default: http://localhost:8231)
--timeout SECS      Per-call timeout (default: 8.0)
```

The script exits 0 and prints something useful even when the server is down
(Section A Baseline runs fully; Section B prints the startup hint).

---

## Further reading

- [`benchmarks/agentic/README.md`](../benchmarks/agentic/README.md) — full
  benchmark design, sweep commands, headline results (P3 containment + quality
  parity).
- [`docs/tutorials/agentic-sandbox-benchmark.md`](../docs/tutorials/agentic-sandbox-benchmark.md) — narrative tutorial walking through the benchmark from first
  principles.
