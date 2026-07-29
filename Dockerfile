# syntax=docker/dockerfile:1.7
#
# rlox — Rust+Python RL framework
# Multi-stage build for reproducible benchmarking and paper experiments.
#
# Stages
#   1. rust-builder   — compile + test the full Rust workspace
#   2. python-env     — Python venv, maturin build, PyO3 bindings, pytest
#   3. experiment-runner (final) — lean runtime image with everything baked in

# Must be >= the channel in rust-toolchain.toml; the repo pin is copied into
# the build stages below so rustup honours it. This was 1.86 while the code
# had moved on to `is_multiple_of` (stable since 1.87), so `cargo build`
# failed with E0658 and the documented Docker reproduction could not run.
# tests/test_repo_hygiene.py keeps the two in step.
ARG RUST_VERSION=1.97.1
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1: rust-builder
# Compile and test the entire Cargo workspace.
# ---------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-bookworm AS rust-builder

# System dependencies required by crates:
#   protobuf-compiler  — rlox-grpc (tonic-build runs prost during build.rs)
#   python3 / python3-dev / python3-pip / python3-venv — maturin + PyO3 link
#   libssl-dev / pkg-config — common transitive deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ---- Dependency caching ----
# Copy only manifest files first so cargo fetch is cached independently of
# source changes.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
# One line per workspace member. This list drifted once already: rlox-rl-ops and
# rlox-sandbox joined the workspace but were not added here, so `cargo fetch`
# died with "failed to read crates/rlox-rl-ops/Cargo.toml" — meaning
# `docker compose build`, the first command in the paper's reproduction
# instructions, could not succeed at all.
#
# A wildcard (`COPY crates/*/Cargo.toml ...`) does NOT work: Docker flattens
# wildcard sources, so the manifests would overwrite each other. `COPY --parents`
# preserves the tree but needs the `-labs` syntax channel, which the artifact
# path should not depend on. So the list stays, and
# tests/test_repo_hygiene.py fails if it drifts from [workspace].members again.
COPY crates/rlox-core/Cargo.toml    crates/rlox-core/Cargo.toml
COPY crates/rlox-nn/Cargo.toml      crates/rlox-nn/Cargo.toml
COPY crates/rlox-rl-ops/Cargo.toml  crates/rlox-rl-ops/Cargo.toml
COPY crates/rlox-burn/Cargo.toml    crates/rlox-burn/Cargo.toml
COPY crates/rlox-candle/Cargo.toml  crates/rlox-candle/Cargo.toml
COPY crates/rlox-python/Cargo.toml  crates/rlox-python/Cargo.toml
COPY crates/rlox-bench/Cargo.toml   crates/rlox-bench/Cargo.toml
COPY crates/rlox-grpc/Cargo.toml    crates/rlox-grpc/Cargo.toml
COPY crates/rlox-sandbox/Cargo.toml crates/rlox-sandbox/Cargo.toml

# Stub out every crate's lib.rs / main.rs so cargo fetch + a dummy build
# resolves the dependency graph without needing real source.
RUN set -eux; \
    for crate in rlox-core rlox-nn rlox-rl-ops rlox-burn rlox-candle rlox-python \
                 rlox-bench rlox-grpc rlox-sandbox; do \
        mkdir -p crates/$crate/src; \
        echo "fn main() {}" > crates/$crate/src/main.rs; \
        echo "" > crates/$crate/src/lib.rs; \
    done; \
    # rlox-sandbox declares an explicit [[bin]] path — stub it too. \
    mkdir -p crates/rlox-sandbox/src/bin; \
    echo "fn main() {}" > crates/rlox-sandbox/src/bin/rlox_verify_server.rs; \
    # rlox-bench declares [[bench]] entries — stub them so cargo fetch works \
    mkdir -p crates/rlox-bench/benches; \
    for b in env_stepping micro_ops nn_backends; do \
        echo "fn main() {}" > crates/rlox-bench/benches/$b.rs; \
    done; \
    # rlox-grpc has a build.rs that needs a proto file \
    mkdir -p crates/rlox-grpc/proto; \
    touch crates/rlox-grpc/proto/env.proto

# Pre-fetch and compile dependencies (cached layer).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    cargo fetch

# ---- Full source ----
# Now replace stubs with real crate sources.
COPY crates/ crates/

# Copy any build scripts required by crates (e.g., rlox-grpc/build.rs for
# tonic-build proto compilation).
# build.rs files are already inside crates/ above.

# Compile release workspace.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    cargo build --release --workspace

# Run Rust tests — this really does fail the build now.
#
# It previously did not: `cargo test ... | tee` exits with *tee's* status, so a
# failing suite was silently swallowed and the image shipped anyway. The build
# log showed `error: test failed` and still finished with exit 0. Both this
# Dockerfile and the paper claimed the image "runs correctness tests during
# build"; only with pipefail is that true.
#
# rlox-sandbox is excluded: it is Linux-only isolation machinery (namespaces,
# seccomp, cgroup v2) that a default-profile container cannot exercise —
# `test_nested_user_namespace_is_denied` and
# `test_stdout_flood_does_not_exhaust_parent` fail purely because Docker blocks
# the capabilities they assert on. It is covered by CI and by
# scripts/wk-sync-test.sh instead, and no claim in the paper depends on it.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    cargo test --workspace --exclude rlox-sandbox --no-fail-fast 2>&1 \
      | tee /build/rust-test-output.txt
SHELL ["/bin/sh", "-c"]

# Export the compiled Rust test log and the release artifacts we need to
# carry forward (mainly for reference; maturin will rebuild from source).
RUN --mount=type=cache,target=/build/target \
    mkdir -p /artifacts/release && \
    cp -r target/release /artifacts/release/bin 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 2: python-env
# Build Python virtual environment, PyO3 extension via maturin, run pytest.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-bookworm AS python-env

# Non-root user for safer builds.
RUN groupadd -r mluser && useradd -r -g mluser -m -s /bin/bash mluser

# System packages:
#   curl        — rustup installer
#   protobuf-compiler — maturin build needs prost for rlox-grpc
#   libgl1      — MuJoCo rendering (EGL/osmesa fallback)
#   libglib2.0-0 — MuJoCo shared library dependency
#   git         — pip editable installs that fetch from git
#   build-essential — C compilation for some Python extensions
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        protobuf-compiler \
        build-essential \
        libgl1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libegl1 \
        libxrender1 \
        libxext6 \
        libgles2 \
        libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required by maturin to compile the PyO3 extension).
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.97.1
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --no-modify-path --default-toolchain ${RUST_VERSION} \
    && rustup component add rustfmt clippy

# Create virtual environment at a stable path.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# ---- Python dependency installation ----
# Split into layers ordered by change frequency (least→most likely to change).

# Layer A: build system + maturin
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        # >=1.8 required: pyproject.toml declares PEP 639 `license-files`,
        # which maturin 1.7.4 cannot parse ("wanted string or table"), so the
        # wheel build — and with it the documented Docker reproduction —
        # failed. 1.12.6 is the version this pyproject is verified against.
        "maturin==1.12.6" \
        "pip==24.3.1" \
        "setuptools==75.6.0" \
        "wheel==0.45.1"

# Layer B: numerical / core ML stack (rarely changes between experiments)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        "numpy==1.26.4" \
        "torch==2.5.1" \
        "torchvision==0.20.1"

# Layer C: RL frameworks under benchmark
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        "stable-baselines3>=2.3.0" \
        "torchrl>=0.8.0" \
        "gymnasium[mujoco]>=1.0.0" \
        "envpool>=0.8.4" || pip install --no-cache-dir \
        "stable-baselines3>=2.3.0" \
        "torchrl>=0.8.0" \
        "gymnasium[mujoco]>=1.0.0"

# Layer D: analysis + visualization stack for paper plots
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        "rliable>=1.0.8" \
        "matplotlib>=3.9" \
        "seaborn>=0.13" \
        "pandas>=2.2" \
        "pyyaml>=6.0"

# Layer E: test tooling
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        "pytest==8.3.4" \
        "pytest-timeout==2.3.1"

# Capture the full locked dependency set as a build artifact.
RUN pip freeze > /opt/venv/requirements.lock

# ---- Build rlox Python extension ----
WORKDIR /build/rlox

# Copy only the files maturin needs for its build phase before copying all
# source, so the expensive Rust recompile is cached when only Python files
# change.
COPY Cargo.toml Cargo.lock rust-toolchain.toml pyproject.toml README.md ./
COPY crates/ crates/
COPY python/ python/
# benchmarks/ is needed here so it can be forwarded to the final stage via
# COPY --from and so pytest can import conftest during the test run.
COPY benchmarks/ benchmarks/

# Build a wheel and install it (not develop mode, so the package is
# self-contained in site-packages without .pth back-references).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/rlox/target \
    maturin build --release --out /build/wheels \
    && pip install /build/wheels/rlox-*.whl

# ---- Verify: Python tests must pass ----
COPY tests/ tests/
RUN python -m pytest tests/python/ -v --timeout=120 \
        -m "not slow" \
        2>&1 | tee /build/rlox/python-test-output.txt

# ---- Copy Rust test log from previous stage ----
COPY --from=rust-builder /build/rust-test-output.txt /build/rlox/rust-test-output.txt

# ---------------------------------------------------------------------------
# Stage 3: experiment-runner (final image)
# Minimal runtime layer — experiments are mounted at runtime, not baked in.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-bookworm AS experiment-runner

LABEL org.opencontainers.image.title="rlox experiment runner" \
      org.opencontainers.image.description="Reproducible benchmarking and paper experiments for the rlox Rust+Python RL framework" \
      org.opencontainers.image.version="0.2.3" \
      org.opencontainers.image.source="https://github.com/riserally/rlox" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0" \
      rlox.rust-version="1.86" \
      rlox.python-version="3.12" \
      rlox.build-date="2026-03-18"

# Same system runtime libs required at execution time.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libegl1 \
        libxrender1 \
        libxext6 \
        libgles2 \
        libglfw3 \
        protobuf-compiler \
        git \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN groupadd -r mluser && useradd -r -g mluser -m -s /bin/bash mluser

# Copy the fully built virtual environment from python-env stage.
COPY --from=python-env /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Copy benchmark scripts and build logs from the python-env stage.
# The rlox package itself is already installed into /opt/venv (site-packages)
# which is carried by the COPY --from=python-env /opt/venv above.
COPY --from=python-env /build/rlox/benchmarks /opt/rlox/benchmarks
COPY --from=python-env /opt/venv/requirements.lock /opt/rlox/requirements.lock
COPY --from=python-env /build/rlox/rust-test-output.txt /opt/rlox/rust-test-output.txt
COPY --from=python-env /build/rlox/python-test-output.txt /opt/rlox/python-test-output.txt

# Results directory — always mount this at runtime.
RUN mkdir -p /workspace/results && chown -R mluser:mluser /workspace

WORKDIR /workspace

# ---- Environment variables ----
ENV PYTHONUNBUFFERED=1 \
    # MuJoCo render mode: headless EGL (no X server needed in container)
    MUJOCO_GL=egl \
    # rlox uses Rayon for CPU parallelism; default to all available cores.
    RAYON_NUM_THREADS=0 \
    # Experiment output directory; override at runtime if needed.
    RLOX_RESULTS_DIR=/workspace/results \
    # Suppress TF warnings leaking from gymnasium deps.
    TF_CPP_MIN_LOG_LEVEL=3 \
    # Deterministic CUDA ops where possible.
    CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- Health check ----
# Verifies that the rlox extension is importable and exports the expected API.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import rlox; assert hasattr(rlox, 'compute_gae'), 'compute_gae missing'; print('rlox OK')"

USER mluser

# Default: interactive bash so researchers can explore.
# docker-compose services override this ENTRYPOINT.
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'rlox experiment runner ready. Mount experiments at /workspace and run your scripts.'; exec bash"]
