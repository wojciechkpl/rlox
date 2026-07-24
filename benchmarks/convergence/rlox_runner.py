"""rlox training + evaluation harness for convergence benchmarks.

Run in a separate process to avoid import contamination with SB3.
"""

from __future__ import annotations

import inspect
import resource
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import gymnasium.vector

import rlox
from rlox.batch import RolloutBatch
from rlox.callbacks import Callback
from rlox.losses import PPOLoss
from rlox.policies import DiscretePolicy
from rlox.vec_normalize import VecNormalize
from rlox.gym_vec_env import GymVecEnv

from common import (
    EvalRecord,
    ExperimentLog,
    evaluate_policy_gym,
    get_hardware_info,
    load_config,
    result_path,
)


# ---------------------------------------------------------------------------
# Observation / reward normalization (matching SB3 VecNormalize)
# ---------------------------------------------------------------------------


class _RunningMeanStd:
    """Welford's online mean/variance tracker for observation normalization."""

    def __init__(self, shape: int):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch: np.ndarray):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[np.newaxis]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        self.update(obs)
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)


# ---------------------------------------------------------------------------
# On-policy trainers (PPO, A2C) with periodic evaluation
# ---------------------------------------------------------------------------


def _collect_rollout_gym(
    env: Any,
    policy: nn.Module,
    obs: np.ndarray,
    n_steps: int,
    n_envs: int,
    gamma: float,
    gae_lambda: float,
    is_discrete: bool,
) -> tuple[RolloutBatch, np.ndarray]:
    """Collect on-policy rollout using env.step_all + rlox GAE.

    Works with GymVecEnv, VecNormalize, or gymnasium VectorEnv (via duck typing).
    Normalization is handled at the environment boundary by VecNormalize.
    Returns (batch, next_obs).
    """
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    # Support both step_all (GymVecEnv/VecNormalize) and step (gymnasium VectorEnv)
    use_step_all = hasattr(env, "step_all")

    for _ in range(n_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            actions, log_probs = policy.get_action_and_logprob(obs_tensor)
            values = policy.get_value(obs_tensor)

        if is_discrete:
            actions_env = actions.cpu().numpy().astype(np.int64)
        else:
            actions_env = actions.cpu().numpy()

        if use_step_all:
            step_result = env.step_all(actions_env)
            next_obs = step_result["obs"]
            rewards = step_result["rewards"]
            terminated = step_result["terminated"].astype(bool)
            truncated = step_result["truncated"].astype(bool)
        else:
            next_obs, rewards, terminated, truncated, _infos = env.step(actions_env)

        dones = terminated | truncated

        # Truncation bootstrap: when truncated but not terminated,
        # add gamma * V(terminal_obs) to the reward so GAE doesn't
        # treat the time-limit boundary as a death (value=0).
        raw_rewards = rewards.copy()
        if use_step_all:
            terminal_obs_list = step_result.get("terminal_obs")
            if terminal_obs_list is not None:
                for i in range(n_envs):
                    if truncated[i] and not terminated[i] and terminal_obs_list[i] is not None:
                        with torch.no_grad():
                            term_val = policy.get_value(
                                torch.as_tensor(terminal_obs_list[i], dtype=torch.float32).unsqueeze(0)
                            ).item()
                        raw_rewards[i] += gamma * term_val

        all_obs.append(obs_tensor)
        all_actions.append(actions)
        all_log_probs.append(log_probs)
        all_values.append(values)
        all_rewards.append(torch.as_tensor(raw_rewards.astype(np.float32)))
        # Pass only terminated (not dones) to GAE — truncation is handled above
        all_dones.append(torch.as_tensor(terminated.astype(np.float32)))

        obs = next_obs

    # Bootstrap value for GAE
    with torch.no_grad():
        last_values = policy.get_value(torch.as_tensor(obs, dtype=torch.float32))

    # Compute GAE per environment using rlox Rust core, then concatenate
    all_advantages = []
    all_returns = []
    for env_idx in range(n_envs):
        rewards_env = torch.stack([r[env_idx] for r in all_rewards])
        values_env = torch.stack([v[env_idx] for v in all_values])
        dones_env = torch.stack([d[env_idx] for d in all_dones])

        adv, ret = rlox.compute_gae(
            rewards=rewards_env.numpy().astype(np.float64),
            values=values_env.numpy().astype(np.float64),
            dones=dones_env.numpy().astype(np.float64),
            last_value=float(last_values[env_idx]),
            gamma=gamma,
            lam=gae_lambda,
        )
        all_advantages.append(torch.as_tensor(adv, dtype=torch.float32))
        all_returns.append(torch.as_tensor(ret, dtype=torch.float32))

    obs_t = torch.stack(all_obs)
    actions_t = torch.stack(all_actions)
    rewards_t = torch.stack(all_rewards)
    dones_t = torch.stack(all_dones)
    log_probs_t = torch.stack(all_log_probs)
    values_t = torch.stack(all_values)
    advantages_t = torch.stack(all_advantages).T
    returns_t = torch.stack(all_returns).T

    total = n_steps * n_envs
    batch = RolloutBatch(
        obs=obs_t.reshape(total, -1),
        actions=actions_t.reshape(total) if actions_t.dim() == 2 else actions_t.reshape(total, -1),
        rewards=rewards_t.reshape(total),
        dones=dones_t.reshape(total),
        log_probs=log_probs_t.reshape(total),
        values=values_t.reshape(total),
        advantages=advantages_t.reshape(total),
        returns=returns_t.reshape(total),
    )
    return batch, obs


def _run_ppo(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """PPO training loop with periodic evaluation.

    Uses GymVecEnv + VecNormalize for stepping, rlox.compute_gae for advantages.
    """
    n_envs = hp.get("n_envs", 8)
    n_steps = hp.get("n_steps", 2048)
    n_epochs = hp.get("n_epochs", 10)
    batch_size = hp.get("batch_size", 64)
    lr = hp.get("learning_rate", 3e-4)
    gamma = hp.get("gamma", 0.99)
    gae_lambda = hp.get("gae_lambda", 0.95)
    clip_eps = hp.get("clip_range", 0.2)
    ent_coef = hp.get("ent_coef", 0.0)
    vf_coef = hp.get("vf_coef", 0.5)
    max_grad_norm = hp.get("max_grad_norm", 0.5)
    normalize_obs = hp.get("normalize_obs", False)
    normalize_reward = hp.get("normalize_reward", False)

    hidden = policy_cfg.get("hidden_sizes", [64, 64])

    # Detect obs/action dimensions from a probe env
    probe_env = gym.make(env_id)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    is_discrete = hasattr(probe_env.action_space, "n")
    if is_discrete:
        n_actions = int(probe_env.action_space.n)
    else:
        n_actions = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    if is_discrete:
        policy = DiscretePolicy(obs_dim, n_actions, hidden=hidden[0])
    else:
        policy = _ContinuousActorCritic(obs_dim, n_actions, hidden)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    loss_fn = PPOLoss(
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
    )

    # Build env with VecNormalize wrapper
    raw_env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)
    vec_normalize: VecNormalize | None = None
    if normalize_obs or normalize_reward:
        vec_normalize = VecNormalize(
            raw_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            gamma=gamma,
        )
        env = vec_normalize
    else:
        env = raw_env

    obs = env.reset_all()

    steps_per_rollout = n_envs * n_steps
    n_updates = max(1, max_steps // steps_per_rollout)
    total_steps = 0
    start_time = time.monotonic()
    last_eval_step = 0

    for update in range(n_updates):
        # LR annealing
        frac = 1.0 - update / n_updates
        for pg in optimizer.param_groups:
            pg["lr"] = lr * frac

        batch, obs = _collect_rollout_gym(
            env, policy, obs, n_steps, n_envs, gamma, gae_lambda, is_discrete,
        )
        total_steps += steps_per_rollout

        for _epoch in range(n_epochs):
            for mb in batch.sample_minibatches(batch_size, shuffle=True):
                adv = mb.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                loss, _ = loss_fn(
                    policy, mb.obs, mb.actions, mb.log_probs,
                    adv, mb.returns, mb.values,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # Periodic evaluation
        if total_steps - last_eval_step >= eval_freq:
            last_eval_step = total_steps
            _do_eval(env_id, policy, is_discrete, total_steps, start_time,
                     eval_episodes, seed, log, vec_normalize=vec_normalize)

    raw_env.close()


def _run_a2c(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """A2C training loop with periodic evaluation.

    Uses gymnasium VectorEnv for stepping + rlox.compute_gae for advantages.
    """
    n_envs = hp.get("n_envs", 8)
    n_steps = hp.get("n_steps", 5)
    lr = hp.get("learning_rate", 7e-4)
    gamma = hp.get("gamma", 0.99)
    gae_lambda = hp.get("gae_lambda", 1.0)
    vf_coef = hp.get("vf_coef", 0.5)
    ent_coef = hp.get("ent_coef", 0.01)
    max_grad_norm = hp.get("max_grad_norm", 0.5)

    hidden = policy_cfg.get("hidden_sizes", [64, 64])

    probe_env = gym.make(env_id)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    n_actions = int(probe_env.action_space.n)
    probe_env.close()

    policy = DiscretePolicy(obs_dim, n_actions, hidden=hidden[0])
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=lr, eps=1e-5, alpha=0.99)

    vec_env = gymnasium.vector.SyncVectorEnv(
        [lambda i=i: gym.make(env_id) for i in range(n_envs)]
    )
    obs, _ = vec_env.reset(seed=seed)

    steps_per_rollout = n_envs * n_steps
    n_updates = max(1, max_steps // steps_per_rollout)
    total_steps = 0
    start_time = time.monotonic()
    last_eval_step = 0

    for update in range(n_updates):
        batch, obs = _collect_rollout_gym(
            vec_env, policy, obs, n_steps, n_envs, gamma, gae_lambda,
            is_discrete=True,
        )
        total_steps += steps_per_rollout

        batch_obs = batch.obs
        actions = batch.actions
        advantages = batch.advantages
        returns = batch.returns

        log_probs, entropy = policy.get_logprob_and_entropy(batch_obs, actions)
        values = policy.get_value(batch_obs)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        if total_steps - last_eval_step >= eval_freq:
            last_eval_step = total_steps
            _do_eval(env_id, policy, True, total_steps, start_time,
                     eval_episodes, seed, log)

    vec_env.close()


# ---------------------------------------------------------------------------
# Off-policy trainers — delegate to standalone algorithm classes
# ---------------------------------------------------------------------------


class _BenchmarkEvalCallback(Callback):
    """Callback that performs periodic evaluation and logs to ExperimentLog."""

    def __init__(
        self,
        env_id: str,
        log: ExperimentLog,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        start_time: float,
        get_action_fn,
    ):
        super().__init__()
        self.env_id = env_id
        self.log = log
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.start_time = start_time
        self.get_action_fn = get_action_fn
        self._last_eval_step = 0

    def on_step(self, step=None, **kwargs) -> bool:
        if step is None:
            return True
        if step - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = step
            wall_clock = time.monotonic() - self.start_time
            sps = step / max(wall_clock, 1e-9)
            mean_ret, std_ret, mean_len = evaluate_policy_gym(
                self.env_id, self.get_action_fn, self.eval_episodes, self.eval_seed,
            )
            self.log.evaluations.append(EvalRecord(
                step=step, wall_clock_s=wall_clock,
                mean_return=mean_ret, std_return=std_ret,
                ep_length=mean_len, sps=sps,
            ))
            print(
                f"  [rlox] step={step:>8d}  "
                f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
                f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
            )
        return True


def _filter_hp(hp: dict[str, Any], algo_class: type) -> dict[str, Any]:
    """Filter hyperparameters to only those accepted by the algorithm constructor."""
    valid_params = set(inspect.signature(algo_class.__init__).parameters)
    skip_keys = {"train_freq", "gradient_steps", "ent_coef"}
    return {k: v for k, v in hp.items() if k in valid_params and k not in skip_keys}


# ---------------------------------------------------------------------------
# Off-policy trainers — delegate to standalone algorithm classes
# ---------------------------------------------------------------------------


class _BenchmarkEvalCallback(Callback):
    """Callback that performs periodic evaluation and logs to ExperimentLog."""

    def __init__(
        self,
        env_id: str,
        log: ExperimentLog,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        start_time: float,
        get_action_fn,
    ):
        super().__init__()
        self.env_id = env_id
        self.log = log
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.start_time = start_time
        self.get_action_fn = get_action_fn
        self._last_eval_step = 0
        self._total_eval_time = 0.0

    def on_step(self, step=None, **kwargs) -> bool:
        if step is None:
            return True
        if step - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = step
            eval_start = time.monotonic()
            wall_clock = eval_start - self.start_time
            sps = step / max(wall_clock, 1e-9)
            mean_ret, std_ret, mean_len = evaluate_policy_gym(
                self.env_id, self.get_action_fn, self.eval_episodes, self.eval_seed,
            )
            self._total_eval_time += time.monotonic() - eval_start
            training_wall = wall_clock - self._total_eval_time
            training_sps = step / max(training_wall, 1e-9)
            self.log.evaluations.append(EvalRecord(
                step=step, wall_clock_s=wall_clock,
                mean_return=mean_ret, std_return=std_ret,
                ep_length=mean_len, sps=sps,
                training_sps=training_sps,
            ))
            print(
                f"  [rlox] step={step:>8d}  "
                f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
                f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
            )
        return True


def _filter_hp(hp: dict[str, Any], algo_class: type) -> dict[str, Any]:
    """Filter hyperparameters to only those accepted by the algorithm constructor."""
    import inspect
    valid_params = set(inspect.signature(algo_class.__init__).parameters)
    skip_keys = {"train_freq", "gradient_steps", "ent_coef"}
    return {k: v for k, v in hp.items() if k in valid_params and k not in skip_keys}


def _run_sac(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """SAC training loop using standalone SAC class."""
    from rlox.algorithms.sac import SAC

    hidden = policy_cfg.get("hidden_sizes", [256, 256])[0]
    filtered = _filter_hp(hp, SAC)
    auto_entropy = hp.get("ent_coef", "auto") == "auto"

    sac = SAC(
        env_id=env_id, hidden=hidden, seed=seed,
        auto_entropy=auto_entropy, **filtered,
    )
    start_time = time.monotonic()

    def get_action(o: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            return sac.actor.deterministic(ot).squeeze(0).numpy() * sac.act_high

    eval_cb = _BenchmarkEvalCallback(
        env_id, log, eval_freq, eval_episodes, seed + 1000, start_time, get_action,
    )
    sac.callbacks.callbacks.append(eval_cb)
    sac.train(total_timesteps=max_steps)


def _run_td3(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """TD3 training loop using standalone TD3 class."""
    from rlox.algorithms.td3 import TD3

    hidden = policy_cfg.get("hidden_sizes", [256, 256])[0]
    filtered = _filter_hp(hp, TD3)
    # Map config key names to TD3 constructor names
    if "target_policy_noise" in hp:
        filtered["target_noise"] = hp["target_policy_noise"]
    if "target_noise_clip" in hp:
        filtered["noise_clip"] = hp["target_noise_clip"]

    td3 = TD3(env_id=env_id, hidden=hidden, seed=seed, **filtered)
    start_time = time.monotonic()

    def get_action(o: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            return td3.actor(ot).squeeze(0).numpy()

    eval_cb = _BenchmarkEvalCallback(
        env_id, log, eval_freq, eval_episodes, seed + 1000, start_time, get_action,
    )
    td3.callbacks.callbacks.append(eval_cb)
    td3.train(total_timesteps=max_steps)


def _run_dqn(
    env_id: str,
    hp: dict[str, Any],
    policy_cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    eval_freq: int,
    eval_episodes: int,
    log: ExperimentLog,
) -> None:
    """DQN training loop using standalone DQN class."""
    from rlox.algorithms.dqn import DQN

    hidden = policy_cfg.get("hidden_sizes", [64, 64])[0]
    filtered = _filter_hp(hp, DQN)
    # Map config key names to DQN constructor names
    if "target_update_interval" in hp:
        filtered["target_update_freq"] = hp["target_update_interval"]

    filtered.pop("hidden", None)  # explicit hidden= below; avoid duplicate kwarg
    dqn = DQN(env_id=env_id, hidden=hidden, seed=seed, **filtered)
    start_time = time.monotonic()

    def get_action(o: np.ndarray) -> int:
        with torch.no_grad():
            ot = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            return int(dqn.q_network(ot).argmax(dim=-1).item())

    eval_cb = _BenchmarkEvalCallback(
        env_id, log, eval_freq, eval_episodes, seed + 1000, start_time, get_action,
    )
    dqn.callbacks.callbacks.append(eval_cb)
    dqn.train(total_timesteps=max_steps)


# ---------------------------------------------------------------------------
# Continuous-action PPO actor-critic
# ---------------------------------------------------------------------------


def _ortho_init(module: nn.Module, gain: float = np.sqrt(2)):
    """Orthogonal weight initialization (matching SB3)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class _ContinuousActorCritic(nn.Module):
    """Gaussian actor-critic for continuous PPO."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]):
        super().__init__()
        h = hidden_sizes[0] if hidden_sizes else 64

        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, act_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )

        # Orthogonal initialization (matching SB3 defaults)
        self.actor_mean.apply(_ortho_init)
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor_mean[-1].bias)
        self.critic.apply(_ortho_init)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def get_action_and_logprob(self, obs: torch.Tensor):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_logprob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _do_eval(
    env_id: str,
    policy: nn.Module,
    is_discrete: bool,
    total_steps: int,
    start_time: float,
    eval_episodes: int,
    seed: int,
    log: ExperimentLog,
    vec_normalize: VecNormalize | None = None,
) -> None:
    """Run evaluation and append to log."""
    wall_clock = time.monotonic() - start_time
    sps = total_steps / max(wall_clock, 1e-9)

    # Freeze stats during eval
    if vec_normalize is not None:
        vec_normalize.training = False

    if is_discrete:
        def get_action(obs: np.ndarray) -> int:
            if vec_normalize is not None:
                obs = vec_normalize.normalize_obs(
                    np.asarray(obs, dtype=np.float32)[np.newaxis]
                )[0]
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy.actor(obs_t)
                return int(logits.argmax(dim=-1).item())
    else:
        def get_action(obs: np.ndarray) -> np.ndarray:
            if vec_normalize is not None:
                obs = vec_normalize.normalize_obs(
                    np.asarray(obs, dtype=np.float32)[np.newaxis]
                )[0]
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean = policy.actor_mean(obs_t)
                return mean.squeeze(0).numpy()

    mean_ret, std_ret, mean_len = evaluate_policy_gym(
        env_id, get_action, eval_episodes, seed + 1000,
    )

    # Restore training mode
    if vec_normalize is not None:
        vec_normalize.training = True

    log.evaluations.append(EvalRecord(
        step=total_steps, wall_clock_s=wall_clock,
        mean_return=mean_ret, std_return=std_ret,
        ep_length=mean_len, sps=sps,
    ))
    print(
        f"  [rlox] step={total_steps:>8d}  "
        f"return={mean_ret:>8.1f} +/- {std_ret:>6.1f}  "
        f"SPS={sps:>7.0f}  wall={wall_clock:>6.1f}s"
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

_ALGO_DISPATCH = {
    "PPO": _run_ppo,
    "A2C": _run_a2c,
    "SAC": _run_sac,
    "TD3": _run_td3,
    "DQN": _run_dqn,
}


def run_rlox(config_path: str, seed: int, results_dir: str) -> Path:
    """Run a single rlox experiment and return the result path."""
    cfg = load_config(config_path)
    algo_name = cfg["algorithm"]
    env_id = cfg["environment"]
    hp = cfg["hyperparameters"]
    policy_cfg = cfg.get("policy", {})
    max_steps = cfg["max_steps"]
    eval_freq = cfg["eval_freq"]
    eval_episodes = cfg["eval_episodes"]

    print(f"[rlox] {algo_name} on {env_id}, seed={seed}, max_steps={max_steps}")

    log = ExperimentLog(
        framework="rlox",
        algorithm=algo_name,
        environment=env_id,
        seed=seed,
        hyperparameters=hp,
        hardware=get_hardware_info(),
    )

    runner = _ALGO_DISPATCH[algo_name]
    start_time = time.monotonic()

    runner(env_id, hp, policy_cfg, seed, max_steps, eval_freq, eval_episodes, log)

    elapsed = time.monotonic() - start_time
    log.total_wall_clock_s = elapsed
    log.total_steps = max_steps
    log.mean_sps = max_steps / max(elapsed, 1e-9)

    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        log.peak_memory_mb = rusage.ru_maxrss / (1024 * 1024)
    except Exception:
        log.peak_memory_mb = 0.0

    out_path = result_path(Path(results_dir), "rlox", algo_name, env_id, seed)
    log.save(out_path)
    print(f"[rlox] Done. Results saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rlox convergence benchmark")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_rlox(args.config, args.seed, args.results_dir)
