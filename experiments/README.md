# Paper experiments

The correctness, performance, convergence and ablation experiments behind the
paper's numbers. This is the artifact: every quantitative claim in the paper is
produced by something in this directory.

## Running them

From the repository root:

```bash
docker compose build
docker compose run --rm experiments        # correctness + performance
docker compose run --rm experiments-full   # + convergence + ablation
```

Results are written to `./results/` on the host. The image pins dependency
versions and runs the Rust and Python test suites during the build, so a
successful build is itself a correctness check.

### Run the performance phase on native x86-64

The correctness phase is architecture-independent. The **performance** and
**ablation** phases are not: run them on a native x86-64 Linux machine.

Under emulation — e.g. `linux/amd64` on an Apple-silicon Mac — the timings are
not merely noisy, they *invert*. Emulating compiled Rust is far more expensive
than emulating Python that dispatches into natively-built NumPy, so the "all
Python" ablation appears to beat the full-Rust build:

```
FAILED test_ablation.py::test_full_rust_faster_than_partial[python_env]
  full_rust (40869 SPS) should be faster than python_env (83455 SPS)
```

Those five failures are an artifact of the platform, not a refutation of the
ablation result. There is no reliable way to detect emulation from inside the
container, so this is documented rather than auto-skipped: if you see the
ablation assertions fail, check your architecture before concluding anything.

To run without Docker (needs the extension built — `maturin develop --release`):

```bash
cd experiments
python -m pytest -m correctness -v      # exactness vs Stable-Baselines3
python -m pytest -m performance -v      # timing benchmarks
python run_all_experiments.py --convergence --ablation
```

## How this differs from `tests/`

`tests/` is the library's own suite: unit and integration tests that gate CI on
every push, and which must be fast. `experiments/` is the *paper's* suite: it
verifies numerical agreement with Stable-Baselines3 and measures the timings
reported in the tables, so individual cases can take minutes to hours. CI does
not run this directory.

## Layout

| path | what it covers |
| --- | --- |
| `test_correctness_gae.py` | GAE agrees with SB3 to floating-point tolerance |
| `test_correctness_buffer.py` | buffer sampling and priority-replay semantics |
| `test_correctness_env.py` | vectorized env stepping matches Gymnasium |
| `test_correctness_e2e.py` | end-to-end training produces valid, consistent data |
| `test_correctness_llm_ops.py` | GRPO advantages and token-level KL |
| `test_performance_components.py` | per-component timings (GAE, buffer, env) |
| `test_performance_e2e.py` | rollout collection and full training-loop timings |
| `test_convergence.py` | multi-seed convergence vs SB3 |
| `test_ablation.py` | marginal contribution of each Rust component |
| `run_all_experiments.py` | driver that runs the above and writes `results/` |
| `scripts/generate_figures.py` | renders the paper figures from `results/` |
| `scripts/run_ablation.py` | standalone ablation sweep |

## Reporting

Timings are reported as IQM with stratified bootstrap 95% confidence intervals
(Agarwal et al., 2021). `utils.py` holds the shared statistics and the
result-serialisation format.

Markers are declared in `pytest.ini`: `correctness` (fast, run by default),
`performance` (timing benchmarks, opt-in), and `convergence` (full training
runs; `slow` is an alias). The ablation cases are marked `performance`.
