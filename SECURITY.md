# Security Policy

## Reporting a vulnerability

Please report security issues privately, **not** as a public GitHub issue.

Use [GitHub's private vulnerability reporting](https://github.com/wojciechkpl/rlox/security/advisories/new)
(Security → Advisories → *Report a vulnerability*). That creates a private thread
visible only to the maintainers.

Include what you have: affected version or commit, platform and kernel version,
a reproduction, and what an attacker gains. A partial report is better than none —
if you are unsure whether something counts, report it.

Expect an acknowledgement within a week. Please give us a chance to ship a fix
before disclosing publicly; we will credit you in the advisory unless you'd
rather stay anonymous.

## Supported versions

Fixes land on `main` and go out in the next release. Only the latest release is
supported — there are no long-term-support branches.

## Scope

Most of rlox is a training library: it runs code *you* wrote, so the usual
trust boundary is your own. The exception is the part deliberately built to run
code you do **not** trust.

### `rlox-sandbox` — in scope, and the part we most want reports about

`rlox-sandbox` executes untrusted code (typically LLM-generated) under Linux
user + PID + network + mount namespaces, a seccomp-BPF allowlist, and cgroup v2
limits. Anything that defeats that containment is a vulnerability. Concretely:

- **Escape** — reading or writing host files outside the sandbox, obtaining a
  network socket, escaping the PID or mount namespace, or nesting a user
  namespace (`CLONE_NEWUSER`) to gain capabilities.
- **Resource escape** — a fork bomb, memory bomb, or FD exhaustion whose effects
  outlive `run_sandboxed` or reach the parent process. Contagion to the host is
  the specific failure this design exists to prevent.
- **Reward forgery** — making the verifier report success for code that did not
  pass, e.g. by defeating the nonce-authenticated trusted-runner protocol with
  `sys.exit(0)`, `os._exit()`, or monkeypatching.
- **Silent loss of containment** — any path where `run_sandboxed` executes code
  *without* the cgroup, seccomp filter, or namespaces actually applied. Failing
  closed matters as much as the guarantees themselves: a sandbox that silently
  degrades is worse than one that errors.

Known limitation, already documented and **not** a new finding: filesystem and
`/proc` information isolation is best-effort under AppArmor's
`unprivileged_userns` restriction. It is enforced at the Python-runtime level, so
a raw-syscall adversary can bypass it. A kernel-level guarantee requires relaxing
the AppArmor profile. Reports that deepen or work around this are still welcome —
we would rather hear it twice than not at all.

### Out of scope

- Running untrusted code through the *training* APIs. Only `rlox-sandbox` is a
  security boundary; `Trainer`, the buffers, and the env wrappers are not.
- Loading a malicious checkpoint or config. `torch.load` and friends execute
  arbitrary code by design — treat checkpoints as you would any executable.
- Denial of service against your own training run through hyperparameters.
- Vulnerabilities in dependencies, unless rlox's usage is what makes them
  exploitable. Report those upstream; tell us if we should pin or patch.

## Verifying containment yourself

The containment guarantees are covered by 61 tests in `crates/rlox-sandbox/`.
Note that most of them **skip** without cgroup v2 delegation, and a skipped test
is counted by cargo as passed — so a green run on a machine without a delegated
cgroup scope proves nothing about containment. See
[the crate README](https://github.com/wojciechkpl/rlox/blob/main/crates/rlox-sandbox/README.md)
for the two capability gates
(`RLOX_SANDBOX_CGROUP_TESTS`, `RLOX_SANDBOX_ADVERSARIAL_TESTS`) and how to run
the suite for real.
