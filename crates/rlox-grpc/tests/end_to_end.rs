//! rlox-grpc end-to-end integration tests (RED phase)
//!
//! ## What this file pins
//!
//! 1. **Smoke coverage** — there are ZERO existing tests for this crate.
//!    The first three tests establish a baseline: start an in-process tonic
//!    server, connect `RemoteEnvClient`, call `reset_batch` + `step_batch`, and
//!    assert that the returned shapes are correct.  These should PASS once the
//!    server/client wiring is working correctly; they are RED today only because
//!    the crate has no tests at all.
//!
//! 2. **terminal_obs bug** — `client.rs:59` hardcodes
//!    `terminal_obs: vec![None; num_envs]` regardless of whether any environment
//!    reached a terminal state.  `test_step_batch_terminal_obs_transmitted` drives
//!    CartPole to termination and asserts that the corresponding `terminal_obs[i]`
//!    is `Some(_)`.  With the current implementation this assertion FAILS because
//!    `terminal_obs` is always `None`.
//!
//! ## Construction assumptions the implementer must honor
//!
//!   - `EnvWorker::new(VecEnv)` — exact signature as in `server.rs`.
//!   - `proto::env_service_server::EnvServiceServer::new(worker)` wraps the
//!     worker into a tonic service.
//!   - `tonic::transport::Server::builder().add_service(svc).serve(addr)` binds
//!     and serves.
//!   - `RemoteEnvClient::connect("http://127.0.0.1:<port>")` connects.
//!   - `VecEnv::new(Vec<Box<dyn RLEnv>>)` — creates the vectorized environment.
//!   - `CartPole::new(Some(seed))` — creates a seeded CartPole.
//!   - CartPole obs dim = 4; action space = Discrete(2).
//!   - CartPole terminates (pole falls) when always stepping with action=1,
//!     typically within ~20-80 steps from the initial reset.
//!
//! ## Test runner
//!
//!   cargo test -p rlox-grpc
//!
//!   Each async test uses `#[tokio::test]`.  The tokio runtime is available via
//!   the `tokio = { features = ["full"] }` dependency already declared in
//!   `Cargo.toml` (no `[dev-dependencies]` change needed).

use std::net::TcpListener;
use std::time::Duration;

use rlox_core::env::builtins::CartPole;
use rlox_core::env::parallel::VecEnv;
use rlox_core::env::spaces::Action;
use rlox_core::env::RLEnv;
use rlox_grpc::proto::env_service_server::EnvServiceServer;
use rlox_grpc::{EnvWorker, RemoteEnvClient};

// ---------------------------------------------------------------------------
// Helper — build a small CartPole VecEnv
// ---------------------------------------------------------------------------

const NUM_ENVS: usize = 4;
const CARTPOLE_OBS_DIM: usize = 4;

fn make_cartpole_vec_env(n: usize, seed: u64) -> VecEnv {
    let envs: Vec<Box<dyn RLEnv>> = (0..n)
        .map(|i| Box::new(CartPole::new(Some(seed + i as u64))) as Box<dyn RLEnv>)
        .collect();
    VecEnv::new(envs).expect("VecEnv construction should not fail with non-empty env list")
}

// ---------------------------------------------------------------------------
// Helper — bind an ephemeral port, spawn the server, return the port
// ---------------------------------------------------------------------------

async fn spawn_server(num_envs: usize, seed: u64) -> u16 {
    // Bind port=0 so the OS assigns an available ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind to ephemeral port");
    let port = listener.local_addr().unwrap().port();
    // Release the std listener — tonic will bind to the port itself.
    drop(listener);

    let vec_env = make_cartpole_vec_env(num_envs, seed);
    let worker = EnvWorker::new(vec_env);
    let svc = EnvServiceServer::new(worker);

    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(svc)
            .serve(addr)
            .await
            .expect("server should not error");
    });

    // Give the server a moment to start listening.
    tokio::time::sleep(Duration::from_millis(50)).await;

    port
}

// ---------------------------------------------------------------------------
// Helper — connect a client, with a few retries to handle slow server startup
// ---------------------------------------------------------------------------

async fn connect_client(port: u16) -> RemoteEnvClient {
    let addr = format!("http://127.0.0.1:{port}");
    for _ in 0..10 {
        match RemoteEnvClient::connect(&addr).await {
            Ok(c) => return c,
            Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
        }
    }
    RemoteEnvClient::connect(&addr)
        .await
        .expect("client should connect within retries")
}

// ---------------------------------------------------------------------------
// Test 1: reset_batch returns the correct shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_reset_batch_returns_correct_shape() {
    let port = spawn_server(NUM_ENVS, 42).await;
    let mut client = connect_client(port).await;

    let observations = client
        .reset_batch(Some(42))
        .await
        .expect("reset_batch should succeed");

    assert_eq!(
        observations.len(),
        NUM_ENVS,
        "reset_batch should return one observation per env"
    );

    for (i, obs) in observations.iter().enumerate() {
        assert_eq!(
            obs.as_slice().len(),
            CARTPOLE_OBS_DIM,
            "env {i}: observation should have CartPole obs_dim={CARTPOLE_OBS_DIM}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: step_batch returns the correct shapes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_step_batch_returns_correct_shapes() {
    let port = spawn_server(NUM_ENVS, 99).await;
    let mut client = connect_client(port).await;

    // Reset first so the server-side envs are in a valid state.
    client
        .reset_batch(Some(99))
        .await
        .expect("reset_batch should succeed before stepping");

    let actions: Vec<Action> = (0..NUM_ENVS)
        .map(|i| Action::Discrete((i % 2) as u32))
        .collect();

    let transition = client
        .step_batch(&actions)
        .await
        .expect("step_batch should succeed");

    assert_eq!(
        transition.obs.len(),
        NUM_ENVS,
        "step_batch should return one obs per env"
    );
    assert_eq!(
        transition.rewards.len(),
        NUM_ENVS,
        "step_batch should return one reward per env"
    );
    assert_eq!(
        transition.terminated.len(),
        NUM_ENVS,
        "step_batch should return one terminated flag per env"
    );
    assert_eq!(
        transition.truncated.len(),
        NUM_ENVS,
        "step_batch should return one truncated flag per env"
    );
    assert_eq!(
        transition.terminal_obs.len(),
        NUM_ENVS,
        "step_batch should return terminal_obs with one entry per env"
    );

    for (i, obs) in transition.obs.iter().enumerate() {
        assert_eq!(
            obs.len(),
            CARTPOLE_OBS_DIM,
            "env {i}: post-step observation should have CartPole obs_dim={CARTPOLE_OBS_DIM}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: rewards on the first step are all 1.0 (CartPole always gives +1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_step_batch_cartpole_reward_is_one_on_first_step() {
    let port = spawn_server(NUM_ENVS, 77).await;
    let mut client = connect_client(port).await;

    client
        .reset_batch(Some(77))
        .await
        .expect("reset_batch should succeed");

    let actions: Vec<Action> = vec![Action::Discrete(0); NUM_ENVS];
    let transition = client
        .step_batch(&actions)
        .await
        .expect("step_batch should succeed");

    for (i, &r) in transition.rewards.iter().enumerate() {
        assert!(
            (r - 1.0).abs() < f64::EPSILON,
            "env {i}: first-step CartPole reward should be 1.0, got {r}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: terminal_obs bug pin
//
// Drive CartPole with action=1 (always push right) until at least one
// environment terminates.  After termination the VecEnv auto-resets and
// returns the fresh post-reset observation in `obs`, while `terminal_obs`
// MUST contain the observation just before the reset (needed for value
// bootstrapping).
//
// The current `client.rs:59` implementation returns `vec![None; num_envs]`
// unconditionally, so this assertion:
//
//   assert!(transition.terminal_obs[i].is_some(), ...)
//
// MUST FAIL with the current code.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_step_batch_terminal_obs_transmitted() {
    // Use 2 envs to reduce noise; action=1 causes CartPole to fall quickly.
    let num_envs = 2usize;
    let port = spawn_server(num_envs, 7).await;
    let mut client = connect_client(port).await;

    client
        .reset_batch(Some(7))
        .await
        .expect("reset_batch should succeed");

    let actions: Vec<Action> = vec![Action::Discrete(1); num_envs];

    // Step up to 200 times; CartPole with action=1 always terminates well
    // before 200 steps when starting from a near-zero initial state.
    let mut found_terminal = false;
    for _step in 0..200 {
        let transition = client
            .step_batch(&actions)
            .await
            .expect("step_batch should not error");

        for i in 0..num_envs {
            let is_done = transition.terminated[i] || transition.truncated[i];
            if is_done {
                // --- This is the assertion that pins the bug. ---
                // The server-side VecEnv populates `terminal_obs[i]` with
                // the final observation before auto-reset.  The client is
                // responsible for transmitting it.  With the current
                // `vec![None; num_envs]` it will always be None here.
                assert!(
                    transition.terminal_obs[i].is_some(),
                    "env {i} reached a terminal state (terminated={}, truncated={}) \
                     but terminal_obs[{i}] is None — the client does not transmit \
                     terminal observations (bug at client.rs:59)",
                    transition.terminated[i],
                    transition.truncated[i],
                );

                // Also check that the terminal observation has the right dimension.
                let tobs = transition.terminal_obs[i].as_ref().unwrap();
                assert_eq!(
                    tobs.len(),
                    CARTPOLE_OBS_DIM,
                    "terminal_obs[{i}] should have CartPole obs_dim={CARTPOLE_OBS_DIM}, got {}",
                    tobs.len()
                );

                found_terminal = true;
            }
        }

        if found_terminal {
            break;
        }
    }

    assert!(
        found_terminal,
        "at least one CartPole env should have terminated within 200 steps when always pushing right"
    );
}
