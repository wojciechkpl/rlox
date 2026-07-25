# Contributing to rlox

Thank you for your interest in contributing to rlox!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/wojciechkpl/rlox.git
cd rlox

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install maturin numpy gymnasium torch pytest ruff

# Build the Rust extension (always use --release)
maturin develop --release

# Verify
python -c "import rlox; print('rlox ready')"
```

## Running Tests

```bash
# Rust tests. --no-fail-fast matters: without it cargo stops at the first failing
# test *binary* and later binaries' failures stay hidden.
cargo test --workspace --no-fail-fast

# Python tests (900+ tests, after maturin develop)
pip install -e ".[all]"
pytest tests/python/ -q

# Quick smoke test (skip slow integration tests)
pytest tests/python/ -m "not slow" -q

# Specific test file
pytest tests/python/test_offline_rl.py -v

# Quick algorithm smoke test
pytest tests/python/test_algorithm_smoke.py -v
```

### The Linux-only sandbox tests

`rlox-sandbox` builds and runs on Linux only, and many of its tests need host
capabilities a CI shared runner cannot provide. They are gated by two env vars
(see `crates/rlox-sandbox/tests/common/mod.rs`):

| Variable | Unlocks | Requires |
| --- | --- | --- |
| `RLOX_SANDBOX_CGROUP_TESTS` | every test that calls `run_sandboxed` and asserts on the exit status | cgroup v2 **self-migration** — the process must already sit inside a user-delegated cgroup scope |
| `RLOX_SANDBOX_ADVERSARIAL_TESTS` | tests that detonate real fork/memory/pids bombs | the above, plus a scope-level `TasksMax`/`MemoryMax` backstop |

Without them the sandbox child cannot enter its cgroup leaf, `run_sandboxed`
returns `SetupError("child could not write to cgroup.procs …")`, and every
containment assertion becomes vacuous — so those tests skip with an explicit
`SKIP <name>: …` line (visible with `-- --nocapture`) instead of failing.

`scripts/wk-sync-test.sh` exports both and wraps the run in
`systemd-run --user --scope --slice=rlox.slice`, so the full suite runs there:

```bash
bash scripts/wk-sync-test.sh 'cargo test -p rlox-sandbox --no-fail-fast'
```

The gates are explicit opt-ins rather than runtime probes on purpose: on the host
that is supposed to have the capability, a regression must fail loudly instead of
silently self-skipping.

## Before you push

Run every CI gate locally in one command:

```bash
bash scripts/check-ci-local.sh          # all gates
bash scripts/check-ci-local.sh rust     # or just one half
WK=1 bash scripts/check-ci-local.sh     # + the Linux sandbox suite on wk-system
```

Install the pre-push hook once and the fast gates run automatically:

```bash
bash scripts/install-git-hooks.sh
```

## Code Style

```bash
# Rust
cargo fmt --all

# NOTE: plain `cargo clippy --workspace` FAILS on macOS — rlox-sandbox is
# Linux-only (namespaces, seccomp, cgroup v2) and its `seccompiler` dependency
# does not compile against a macOS libc. Lint it against a Linux target instead,
# otherwise every lint inside `#[cfg(target_os = "linux")]` stays invisible until
# CI runs. Both commands are what check-ci-local.sh does for you:
cargo clippy --workspace --exclude rlox-sandbox --all-targets
rustup target add x86_64-unknown-linux-gnu   # once
cargo clippy -p rlox-sandbox --all-targets --target x86_64-unknown-linux-gnu

# Python
ruff check python/
ruff format python/
```

The toolchain is pinned in `rust-toolchain.toml` so local clippy enforces exactly
the lint set CI does. Clippy adds lints every release; an unpinned `stable` meant
CI could fail on lints an older local toolchain never reported.

## Project Structure

```
crates/
  rlox-core/     # Rust data plane: buffers, envs, GAE, KL, pipeline
  rlox-nn/       # Backend-agnostic NN traits
  rlox-candle/   # Candle backend (inference + hybrid collection)
  rlox-burn/     # Burn backend (alternative)
  rlox-python/   # PyO3 bindings
python/rlox/
  algorithms/    # PPO, SAC, DQN, TD3, A2C, MAPPO, DreamerV3, IMPALA, offline RL, LLM
  offline/       # Offline RL base class + protocols
  exploration/   # Noise strategies + intrinsic rewards
  wrappers/      # VecNormalize and other env wrappers
  callbacks.py   # Training callbacks
  policies.py    # Neural network policies
  trainers.py    # High-level trainers for all algorithms
  runner.py      # Config-driven training (train_from_config)
  dashboard.py   # MetricsCollector, TerminalDashboard, HTMLReport
tests/python/    # Python test suite
docs/            # MkDocs documentation
```

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests first (TDD when possible)
3. Ensure all Rust and Python tests pass
4. Run `cargo fmt` and `ruff format`
5. Update documentation if adding new features
6. Keep PRs focused: one feature or fix per PR

## Adding a New Algorithm

1. Create `python/rlox/algorithms/your_algo.py`
2. Implement using existing primitives (buffers, GAE, etc.)
3. Add tests in `tests/python/test_your_algo.py`
4. Add to `docs/examples.md` and `docs/python-guide.md`
5. Update `docs/index.md` algorithm list

## Adding a New Rust Primitive

1. Implement in the appropriate `crates/rlox-core/src/` module
2. Add unit tests in the same file
3. Create PyO3 bindings in `crates/rlox-python/src/`
4. Register in `crates/rlox-python/src/lib.rs`
5. Export in `python/rlox/__init__.py`
6. Add Python integration tests
