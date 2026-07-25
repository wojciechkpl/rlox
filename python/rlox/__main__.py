"""CLI entry point for rlox.

Usage:
    python -m rlox train --algo ppo --env CartPole-v1 --timesteps 100000
    python -m rlox train --algo sac --env Pendulum-v1 --timesteps 50000 --config config.yaml
    python -m rlox eval --checkpoint model.pt --env CartPole-v1 --episodes 10
"""

from __future__ import annotations

import argparse
import sys

def _get_algo_names() -> list[str]:
    """Get all registered algorithm names from the Trainer registry."""
    from rlox.trainer import ALGORITHM_REGISTRY
    return sorted(ALGORITHM_REGISTRY.keys())


def _is_training_config(path: str) -> bool:
    """Check if a config file is a TrainingConfig (has 'algorithm' key)."""
    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return "algorithm" in data
    elif path.endswith(".toml"):
        # Via config._load_toml, which falls back to the tomli backport: tomllib
        # is stdlib only from 3.11 and rlox supports 3.10, so a bare
        # `import tomllib` here made `rlox train --config x.toml` crash there.
        from rlox.config import _load_toml

        return "algorithm" in _load_toml(path)
    return False


def cmd_train(args):
    # If --config points to a TrainingConfig file, use the config-driven runner
    if args.config and _is_training_config(args.config):
        from rlox.config import TrainingConfig
        from rlox.runner import train_from_config

        config = (
            TrainingConfig.from_toml(args.config)
            if args.config.endswith(".toml")
            else TrainingConfig.from_yaml(args.config)
        )
        # CLI overrides (only if explicitly provided)
        if args.env:
            config.env_id = args.env
        if args.seed is not None:
            config.seed = args.seed
        if args.timesteps is not None:
            config.total_timesteps = args.timesteps

        print(
            f"Training {config.algorithm.upper()} on {config.env_id} "
            f"for {config.total_timesteps} steps (seed={config.seed})"
        )
        metrics = train_from_config(config)
        print(f"\nTraining complete. Final metrics: {metrics}")

        if args.save:
            print("Note: --save is not supported with config-driven training yet")
        return

    # Legacy arg-based path — require --algo and --env
    if not args.algo:
        parser_err = "the following arguments are required: --algo (or use --config with a TrainingConfig file)"
        print(f"error: {parser_err}", file=sys.stderr)
        sys.exit(2)
    if not args.env:
        parser_err = "the following arguments are required: --env (or use --config with a TrainingConfig file)"
        print(f"error: {parser_err}", file=sys.stderr)
        sys.exit(2)
    from rlox import Trainer
    from rlox.logging import ConsoleLogger

    config: dict = {}
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        config = cfg.get("hyperparameters", cfg)

    seed = args.seed if args.seed is not None else 42
    timesteps = args.timesteps if args.timesteps is not None else 100_000

    trainer = Trainer(
        args.algo, env=args.env, seed=seed,
        config=config, logger=ConsoleLogger(log_interval=args.log_interval),
    )

    print(
        f"Training {args.algo.upper()} on {args.env} for {timesteps} steps (seed={seed})"
    )
    metrics = trainer.train(total_timesteps=timesteps)
    print(f"\nTraining complete. Final metrics: {metrics}")

    if args.save:
        trainer.save(args.save)
        print(f"Model saved to {args.save}")


def cmd_eval(args):
    import numpy as np
    import gymnasium as gym

    from rlox import Trainer
    trainer = Trainer.from_checkpoint(args.checkpoint, algorithm=args.algo, env=args.env)
    algo = trainer.algo

    env = gym.make(args.env)
    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action = algo.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            done = term or trunc
        rewards.append(ep_reward)
        print(f"  Episode {ep + 1}: reward={ep_reward:.1f}")

    print(f"\nMean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(
        prog="rlox", description="rlox reinforcement learning CLI"
    )
    sub = parser.add_subparsers(dest="command")

    # Train
    train_p = sub.add_parser("train", help="Train an RL agent")
    train_p.add_argument("--algo", choices=_get_algo_names(),
                         help="Algorithm (not required when --config provides it)")
    train_p.add_argument("--env", help="Gymnasium environment ID")
    train_p.add_argument("--timesteps", type=int, default=None)
    train_p.add_argument("--seed", type=int, default=None)
    train_p.add_argument("--config", default=None, help="YAML config file")
    train_p.add_argument("--save", default=None, help="Path to save model checkpoint")
    train_p.add_argument("--log-interval", type=int, default=1000)

    # Eval
    eval_p = sub.add_parser("eval", help="Evaluate a trained agent")
    eval_p.add_argument("--algo", required=True, choices=_get_algo_names())
    eval_p.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    eval_p.add_argument("--env", required=True, help="Gymnasium environment ID")
    eval_p.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
