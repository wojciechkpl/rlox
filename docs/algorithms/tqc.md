# TQC -- Truncated Quantile Critics

## Intuition

TQC replaces SAC's scalar critics with **distributional (quantile) critics** and
controls overestimation by **truncation**. It learns the return *distribution* as
a set of quantiles across an ensemble of critics, then **drops the top (most
optimistic) quantiles** when forming the target. This gives fine-grained, tunable
control over the bias–variance trade-off, generalising SAC's coarse `min(Q1,Q2)`
clipping.

## Key Equations

Each of the $N$ critics outputs $M$ quantiles $\theta^i_j(s,a)$. Pool all
$N\times M$ next-state quantiles at $a' \sim \pi(\cdot|s')$, sort ascending, and
drop the top $d\cdot N$, giving the truncated set $\{z_k\}$. Targets:

$$
y_k = r + \gamma(1-d)\big(z_k - \alpha \log\pi(a'|s')\big)
$$

Critic loss is the **quantile Huber loss** between predicted quantiles and
$\{y_k\}$, with fractions $\tau_j = (j + 0.5)/M$. The actor maximises the mean of
**all** critics' quantiles minus $\alpha\log\pi$.

## Pseudocode

```
initialize N quantile critics (+ targets), actor pi
for step:
    a' ~ pi(s');  pool all N*M target quantiles at (s', a')
    sort ascending, drop top d*N  ->  truncated targets z
    y = r + gamma(1-d)(z - alpha*logpi(a'|s'))
    minimize quantile_huber(predicted_quantiles, y)  summed over critics
    SAC actor + alpha update;  polyak-update target critics
```

## Usage

```python
from rlox import Trainer

trainer = Trainer(
    "tqc",
    env="Pendulum-v1",
    config={"n_critics": 5, "n_quantiles": 25, "top_quantiles_to_drop_per_net": 2},
)
trainer.train(total_timesteps=20_000)
```

## Status

**Experimental.** Continuous action spaces only. Solves Pendulum-v1 (greedy eval
**−155.67**, textbook default hyperparameters). Full continuous-control benchmark
parity is a tracked follow-up.

Paper: Kuznetsov et al., *Controlling Overestimation Bias with Truncated Mixture
of Continuous Distributional Quantile Critics* (ICML 2020, arXiv:2005.04269).
