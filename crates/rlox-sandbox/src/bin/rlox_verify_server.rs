//! `rlox-verify-server` — standalone HTTP server exposing `POST /verify` and
//! `POST /rollout` from the `rlox-sandbox` crate.
//!
//! ## Usage
//!
//! ```bash
//! systemd-run --user --scope -p TasksMax=2048 -p MemoryMax=8G --quiet \
//!     ./rlox-verify-server \
//!     --port 8231 \
//!     --timeout-secs 5.0 \
//!     --mem-mb 512 \
//!     --pids-max 256 \
//!     --max-concurrent 64
//! ```
//!
//! The server **must** run inside a systemd-delegated cgroup scope so that
//! sandbox workers can self-migrate into per-job cgroup leaves.  Without this
//! the startup check in `router_with_full_config` will abort with a clear
//! error message.
//!
//! ## CLI arguments (all optional)
//!
//! | Flag              | Default | Description                                        |
//! |-------------------|---------|----------------------------------------------------|
//! | `--port`          | 8231    | TCP port to listen on                              |
//! | `--timeout-secs`  | 5.0     | Wall-clock timeout per `/verify` sandbox run (s)   |
//! | `--mem-mb`        | 512     | Memory limit per sandbox worker (MiB)              |
//! | `--pids-max`      | 256     | PID limit per sandbox worker                       |
//! | `--max-concurrent`| 64      | Max concurrent sandbox workers across all requests |

use std::net::{Ipv4Addr, SocketAddr};

use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// CLI parsing — no external dependencies, just std::env::args
// ---------------------------------------------------------------------------

struct Args {
    port: u16,
    timeout_secs: f64,
    mem_mb: u64,
    pids_max: u32,
    max_concurrent: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            port: 8231,
            timeout_secs: 5.0,
            mem_mb: 512,
            pids_max: 256,
            max_concurrent: 64,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;

    while i < raw.len() {
        match raw[i].as_str() {
            "--port" => {
                i += 1;
                let v = raw
                    .get(i)
                    .ok_or_else(|| "--port requires a value".to_string())?;
                args.port = v
                    .parse::<u16>()
                    .map_err(|e| format!("--port: {e}"))?;
            }
            "--timeout-secs" => {
                i += 1;
                let v = raw
                    .get(i)
                    .ok_or_else(|| "--timeout-secs requires a value".to_string())?;
                args.timeout_secs = v
                    .parse::<f64>()
                    .map_err(|e| format!("--timeout-secs: {e}"))?;
                if args.timeout_secs <= 0.0 {
                    return Err("--timeout-secs must be positive".to_string());
                }
            }
            "--mem-mb" => {
                i += 1;
                let v = raw
                    .get(i)
                    .ok_or_else(|| "--mem-mb requires a value".to_string())?;
                args.mem_mb = v
                    .parse::<u64>()
                    .map_err(|e| format!("--mem-mb: {e}"))?;
            }
            "--pids-max" => {
                i += 1;
                let v = raw
                    .get(i)
                    .ok_or_else(|| "--pids-max requires a value".to_string())?;
                args.pids_max = v
                    .parse::<u32>()
                    .map_err(|e| format!("--pids-max: {e}"))?;
            }
            "--max-concurrent" => {
                i += 1;
                let v = raw
                    .get(i)
                    .ok_or_else(|| "--max-concurrent requires a value".to_string())?;
                args.max_concurrent = v
                    .parse::<usize>()
                    .map_err(|e| format!("--max-concurrent: {e}"))?;
                if args.max_concurrent == 0 {
                    return Err("--max-concurrent must be >= 1".to_string());
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlox-verify-server\n\
                     \n\
                     USAGE:\n\
                     \n\
                       rlox-verify-server [OPTIONS]\n\
                     \n\
                     OPTIONS:\n\
                       --port           <u16>    TCP port (default: 8231)\n\
                       --timeout-secs   <f64>    /verify sandbox timeout, seconds (default: 5.0)\n\
                       --mem-mb         <u64>    Memory limit per worker, MiB (default: 512)\n\
                       --pids-max       <u32>    PID limit per worker (default: 256)\n\
                       --max-concurrent <usize>  Max concurrent workers (default: 64)\n\
                     \n\
                     The server must run inside a systemd-delegated cgroup scope:\n\
                     \n\
                       systemd-run --user --scope -p TasksMax=2048 -p MemoryMax=8G --quiet \\\n\
                           ./rlox-verify-server --port 8231"
                );
                std::process::exit(0);
            }
            flag => {
                return Err(format!("unknown flag: {flag}"));
            }
        }
        i += 1;
    }

    Ok(args)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("rlox-verify-server: argument error: {e}");
        std::process::exit(1);
    });

    // Derive cgroup_base from the current process uid so the log message is
    // informative before router_with_full_config runs the same derivation.
    #[cfg(target_os = "linux")]
    let cgroup_base = {
        // SAFETY: getuid is always safe on Linux.
        let uid = unsafe { libc::getuid() };
        std::path::PathBuf::from(format!("/sys/fs/cgroup/user.slice/user-{uid}.slice"))
    };

    #[cfg(target_os = "linux")]
    if !cgroup_base.exists() {
        eprintln!(
            "rlox-verify-server: startup check failed — cgroup base {cgroup_base:?} does not exist.\n\
             Run inside a systemd-delegated cgroup scope:\n\
             \n\
               systemd-run --user --scope -p TasksMax=2048 -p MemoryMax=8G --quiet \\\n\
                   ./rlox-verify-server --port {port}",
            port = args.port,
        );
        std::process::exit(1);
    }

    // Build ServerConfig — vllm_base_url is set to a placeholder because
    // /verify never calls vLLM.  /rollout will return 502 if actually called
    // without a real vLLM; that is correct behaviour.
    let config = rlox_sandbox::ServerConfig {
        vllm_base_url: "http://127.0.0.1:1".to_string(),
        sandbox: rlox_sandbox::SandboxRunConfig {
            mem_limit_bytes: args.mem_mb * 1024 * 1024,
            pids_limit: args.pids_max,
            cpu_weight: 100,
        },
        group_size: 1,
    };

    // router_with_full_config re-validates the cgroup base on Linux and panics
    // with a clear message if absent — the check above gives an early, clean
    // exit before we even build the router.
    let app = rlox_sandbox::router_with_full_config(config, args.max_concurrent, args.timeout_secs);

    let addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    let listener = TcpListener::bind(addr).await.unwrap_or_else(|e| {
        eprintln!("rlox-verify-server: failed to bind {addr}: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "rlox-verify-server listening on http://{addr}  \
         (timeout={timeout}s mem={mem_mb}MiB pids={pids} max-concurrent={max_concurrent})",
        timeout = args.timeout_secs,
        mem_mb = args.mem_mb,
        pids = args.pids_max,
        max_concurrent = args.max_concurrent,
    );

    axum::serve(listener, app).await.unwrap_or_else(|e| {
        eprintln!("rlox-verify-server: serve error: {e}");
        std::process::exit(1);
    });
}
