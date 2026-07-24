# PQN -- Parallelised Q-Network

## Intuition

PQN is a simplified, parallelised DQN that removes **both** the replay buffer and
the target network. Instead of those stabilisers it relies on **LayerNorm** (plus
ℓ² regularisation) inside the Q-network to keep temporal-difference learning
stable, and it learns from **many parallel environments** using λ-returns. The
result is a value-based method that is competitive with Rainbow while being far
simpler and much faster.

## Key Equations

Q(λ) target -- a TD(λ) return with a greedy (max-Q) bootstrap:

$$
G_t = r_t + \gamma (1 - d_t)\Big[(1-\lambda)\max_{a'} Q(s_{t+1}, a') + \lambda\, G_{t+1}\Big]
$$

Loss -- regress the taken action's value toward the λ-return, with ℓ² decay:

$$
L(\theta) = \mathbb{E}\big[(Q_\theta(s_t, a_t) - G_t)^2\big] + \eta\lVert\theta\rVert_2^2
$$

LayerNorm after each hidden linear layer replaces the target network as the
stabiliser.

## Pseudocode

```
initialize LayerNorm Q-network Q_theta   (no target net, no replay buffer)
for iteration:
    collect n_steps across n_envs with epsilon-greedy actions     # Rust VecEnv
    V_t = max_a Q(s_t, a)   (no grad);  bootstrap V_last
    G = compute_gae_batched(rewards, values=V, dones, last_value, gamma, lam)  # reuse Rust GAE
    for epoch, minibatch:
        minimize (Q(s,a) - G)^2  with gradient clipping
    decay epsilon
```

## Usage

```python
from rlox import Trainer

trainer = Trainer("pqn", env="CartPole-v1", config={"n_envs": 16, "q_lambda": 0.65})
trainer.train(total_timesteps=200_000)
```

## Status

**Experimental.** Discrete action spaces only. In rlox, PQN reuses the Rust
`compute_gae_batched` op for the Q(λ) targets (no new Rust) and the Rayon
`VecEnv` for the parallel env loop -- the architectural pairing PQN is designed
for. Learns CartPole; full benchmark parity is a tracked follow-up.

Paper: Gallici et al., *Simplifying Deep Temporal Difference Learning*
(arXiv:2407.04811).
