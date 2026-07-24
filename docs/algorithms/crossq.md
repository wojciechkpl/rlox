# CrossQ

## Intuition

CrossQ is a drop-in **SAC upgrade for continuous control that removes the target
networks entirely**. Stability instead comes from **Batch Renormalization** in
the critics plus a **joint forward pass**: $Q(s,a)$ and $Q(s',a')$ are computed
*together* in one batch so BatchNorm statistics stay consistent between the value
being trained and its bootstrap. This matches SAC's sample efficiency at an
update-to-data ratio of 1, without target networks or critic ensembles.

## Key Equations

Bootstrap target (the next-state half of the joint pass, detached):

$$
y = r + \gamma(1-d)\Big(\min_{i}Q_{\phi_i}(s', a') - \alpha \log\pi(a'|s')\Big),
\qquad a' \sim \pi(\cdot\,|\,s')
$$

Crucially, both halves are computed with the critic in **train mode (batch
statistics)** -- the same normalisation regime the critic is trained under, which
is why no target network is needed. Critic loss:

$$
L(\phi) = \sum_i \big(Q_{\phi_i}(s,a) - y\big)^2
$$

The actor and automatic-entropy losses are identical to SAC.

## Pseudocode

```
initialize BatchRenorm critics Q_phi1, Q_phi2   (NO target nets), actor pi
for step:
    sample batch (s,a,r,d,s') from replay
    a' ~ pi(s')
    # joint forward pass, critics in train() -> consistent batch stats:
    Q_both = Q([s ; s'], [a ; a']);  split -> Q(s,a), Q(s',a')
    y = ( r + gamma(1-d)(min Q(s',a') - alpha*logpi(a'|s')) ).detach()
    minimize sum_i (Q_i(s,a) - y)^2
    every policy_delay steps: SAC actor update + alpha update
```

## Usage

```python
from rlox import Trainer

trainer = Trainer("crossq", env="Pendulum-v1")
trainer.train(total_timesteps=20_000)
```

## Status

**Experimental.** Continuous action spaces only. Uses Adam `betas=(0.5, 0.999)`
-- **β1 = 0.5 is load-bearing** (the torch default 0.9 is unstable with
BatchNorm). Solves Pendulum-v1 reliably (greedy eval **−134 ± 0.5 across 4
seeds**); full MuJoCo sample-efficiency parity is a tracked follow-up.

Paper: Bhatt, Palenicek et al., *CrossQ: Batch Normalization in Deep RL for
Greater Sample Efficiency and Simplicity* (ICLR 2024, arXiv:1902.05605).
