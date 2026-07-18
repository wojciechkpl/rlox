# CrossQ weak-convergence — diagnosis & fix strategy

> Date: 2026-07-18
> Symptom: the `@pytest.mark.slow` Pendulum test reaches **−772** at 20k steps
> (threshold −400; SAC ~−150). Above random (~−1200) so it learns *something*,
> but unreliably.
> **Diagnosis method: two independent controlled experiments** (a code-review
> ablation on Adam betas, and an actor-BN-mode ablation). Both are recorded
> below — one *refuted* the initial hypothesis, which is why we ran them.

## What is already correct (verified statically AND empirically)

- **The joint forward pass is correct.** `cat([obs,next_obs])`/`cat([act,next_act])`
  through the live critic in `train()`; split `[:B]` current / `[B:]` next matches
  concat order; `min(Q1,Q2)` over next-halves; target detached; `next_logp` is
  `(B,)`. Proof: with the betas fix (below), the *same* joint-pass code solves
  Pendulum (−166).
- No-target / `tau` invariant, `policy_delay` cadence, min-Q, α update, action
  scaling, continuous-only guard, `BatchRenorm1d` math (r/d detached, EMA update,
  eval uses frozen stats, warmup==BatchNorm) — all correct.
- **The actor update's `eval()`-mode critic query is correct** (see refuted
  hypothesis). Do NOT change it.

## Root cause (ranked, evidence-based)

### 🔴 Primary (CONFIRMED) — Adam `betas` default `(0.9, 0.999)`; CrossQ needs `β1=0.5`

Controlled experiment (change **only** the betas, same code, 20k Pendulum):

| Adam betas | greedy eval | slow test |
|---|---|---|
| `(0.9, 0.999)` (torch default, as shipped) | −716 | FAIL |
| `(0.5, 0.999)` (paper / SB3-contrib) | **−166** | PASS |

CrossQ's paper specifies **β1 = 0.5** as load-bearing — BatchNorm interacts
badly with high Adam first-moment momentum. All four optimizers in `crossq.py`
use the torch default. **This is the fix.** It also proves the joint pass is
correct (only the betas changed).

### 🔴 Critical — `seed` is a no-op → high variance + broken validation methodology

`crossq.py:123` stores `self.seed = seed` but never applies it to torch/np/env
RNG. Consequences:
- **Every run is a fresh random init.** With the unstable β1=0.9, this produces
  the flaky spread: an *explicitly-seeded* run solved (−173.7 @ seed 0), while
  the unseeded slow test / review runs failed (−772 / −716 / −517 / −786).
- Breaks the project's **multi-seed IQM validation methodology** — different
  `seed` values don't produce controlled runs.
Fix: seed torch/np/env in `__init__` (precedent: `pqn.py:118`).

### 🟠 Test coverage — 58 fast tests pass even when the policy is broken

They guard structure (no-target, BatchRenorm, counters, shapes) but **not**
convergence or joint-pass *value alignment*. A split/detach bug would pass. The
only convergence guard is the slow test, which was red **and** flaky (seed bug).

### 🟡 Secondary (faithfulness / MuJoCo parity — NOT blocking Pendulum)

- `BNQNetwork` deviates from the reference: **no input normalisation**, and
  `Linear→ReLU→BatchRenorm` (post-activation) vs the reference's input-norm +
  pre-activation. The betas fix alone solves Pendulum, but this likely costs
  parity on the planned MuJoCo sample-efficiency comparison.
- Fixed `ent_coef` not persisted on checkpoint (inherited from SAC).
- BatchRenorm r/d branch never exercised (`renorm_warmup_steps=100_000` ≫ budget)
  → add a small-`warmup_steps` unit test; consider a smaller default.
- `_R_MAX`/`_D_MAX` hardcoded — expose as params.

## Refuted hypothesis (kept as a record)

**Hypothesis:** the actor update queries the critic in `eval()` (running stats)
while it is trained in `train()` (batch stats) — a BN train/eval mismatch that
miscalibrates the actor gradient.
**Ablation (change only the actor critic-query mode, β1=0.9, seed 0):**

| actor critic query | greedy eval |
|---|---|
| `eval()` / running stats (as shipped) | **−173.7** |
| `train()` / batch stats (hypothesised "fix") | **−1298.5** (broke it) |

**Verdict: REFUTED.** Naive train-mode makes it far worse (the policy-action-only
batch has a different distribution and pollutes running stats). The shipped
eval-mode query is correct. Lesson: verify a plausible mechanism before "fixing"
it.

## Fix strategy (ordered)
1. **Adam betas** — add configurable `adam_betas=(0.5, 0.999)` on
   `CrossQ`/`CrossQConfig` (no-magic-numbers), apply to actor + both critics +
   alpha. *The fix.*
2. **Seed** — seed torch/np/env in `__init__` (PQN precedent). De-flakes the slow
   test; restores multi-seed methodology.
3. **Harden tests (TDD)** — fast value-alignment test for the joint pass
   (monkeypatch critic→identity; assert current/next halves, target detached,
   grad routing), mutation-checked; seed + fix the slow test (assert on greedy
   eval, not the warmup-dragged training mean).
4. **Re-verify** — slow Pendulum ≳ −250; then a short **multi-seed** run for
   stability. Leave the actor eval-mode and (for now) the architecture alone.
5. **Faithfulness pass (optional; needed for MuJoCo parity)** — input-norm +
   pre-activation `BNQNetwork`; persist `ent_coef`; renorm-branch test.

## Verification
- Slow Pendulum convergence passes and is deterministic (seeded).
- New fast test pins the joint-pass value alignment (mutation-checked).
- Multi-seed run shows low variance (confirms betas+seed fixed the flakiness).
- No regression in the other CrossQ fast tests or the registry/status suite.
