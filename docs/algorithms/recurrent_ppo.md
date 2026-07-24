# Recurrent PPO (LSTM)

## Intuition

Recurrent PPO adds an **LSTM** to the PPO actor-critic so the policy can integrate
information over time -- essential for **partially observable** tasks where a
single observation is not a sufficient state. The recurrent hidden state is
carried across timesteps during collection and **reset at episode boundaries**;
PPO updates backpropagate through time (BPTT) over per-episode sequence segments.

## Key Equations

The standard PPO clipped surrogate, evaluated on recurrent outputs:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat{A}_t,\;
\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\Big]
$$

with importance ratio $\rho_t = \dfrac{\pi_\theta(a_t\,|\,s_t, h_t)}{\pi_{\theta_{\text{old}}}(a_t\,|\,s_t, h_t)}$
and LSTM hidden state $h_t$; advantages $\hat{A}_t$ come from GAE. Every loss term
is **masked** so padded / cross-episode steps contribute exactly zero, and the
hidden state is zeroed at each `done` so gradients never flow across an episode
boundary.

## Pseudocode

```
initialize LSTM actor-critic
for iteration:
    collect rollout, carrying (h, c) across steps, resetting (h, c) at done
    advantages, returns = compute_gae_batched(...)          # reuse Rust GAE
    for epoch:
        split each env's timeline into per-episode segments  (no segment crosses a done)
        re-run the LSTM over zero-padded segments (mask = real vs pad)
        minimize masked [ PPO clip + 0.5*value MSE - entropy bonus ]   (BPTT)
```

## Usage

```python
from rlox import Trainer

trainer = Trainer("recurrent_ppo", env="CartPole-v1", config={"lstm_hidden": 64})
trainer.train(total_timesteps=300_000)
```

## Status

**Experimental.** Discrete action spaces (first version). Fully solves CartPole-v1
(reward **500.0**). Designed for POMDPs; `Trainer.evaluate()` resets the recurrent
state between evaluation episodes so they do not leak state into one another.
