"""rlox algorithm implementations.

On-policy (use VecEnv + compute_gae):
    - :class:`PPO` — Proximal Policy Optimization (clipped surrogate)
    - :class:`VPG` — Vanilla Policy Gradient (REINFORCE with baseline + GAE)
    - :class:`A2C` — Advantage Actor-Critic (single gradient step)

Off-policy (use ReplayBuffer + gymnasium):
    - :class:`SAC` — Soft Actor-Critic (continuous actions)
    - :class:`TD3` — Twin Delayed DDPG (continuous actions)
    - :class:`DQN` — Deep Q-Network with Rainbow extensions (discrete actions)

LLM post-training:
    - :class:`GRPO` — Group Relative Policy Optimization
    - :class:`DPO` — Direct Preference Optimization
    - :class:`OnlineDPO` — Online DPO with active generation
    - :class:`BestOfN` — Best-of-N sampling baseline

Multi-agent / advanced:
    - :class:`MAPPO` — Multi-Agent PPO
    - :class:`IMPALA` — Importance Weighted Actor-Learner Architecture
    - :class:`DreamerV3` — World-model-based RL
"""

from rlox.algorithms.ppo import PPO
from rlox.config import PPOConfig, VPGConfig
from rlox.algorithms.a2c import A2C
from rlox.algorithms.vpg import VPG
from rlox.algorithms.grpo import GRPO
from rlox.algorithms.dpo import DPO
from rlox.algorithms.sac import SAC
from rlox.algorithms.td3 import TD3
from rlox.algorithms.dqn import DQN
from rlox.algorithms.online_dpo import OnlineDPO
from rlox.algorithms.best_of_n import BestOfN
from rlox.algorithms.mappo import MAPPO
from rlox.algorithms.dreamer import DreamerV3
from rlox.algorithms.impala import IMPALA, DistributedIMPALA
from rlox.algorithms.awr import AWR
from rlox.algorithms.trpo import TRPO
from rlox.algorithms.diffusion_policy import DiffusionPolicy
from rlox.algorithms.mpo import MPO
from rlox.algorithms.dtp import DecisionTreePolicy, RWDTP, RCDTP
from rlox.algorithms.pqn import PQN
from rlox.algorithms.crossq import CrossQ
from rlox.algorithms.tqc import TQC
from rlox.algorithms.recurrent_ppo import RecurrentPPO

__all__ = [
    "PPO",
    "PPOConfig",
    "VPG",
    "VPGConfig",
    "A2C",
    "GRPO",
    "DPO",
    "SAC",
    "TD3",
    "DQN",
    "OnlineDPO",
    "BestOfN",
    "MAPPO",
    "DreamerV3",
    "IMPALA",
    "DistributedIMPALA",
    "AWR",
    "TRPO",
    "DiffusionPolicy",
    "MPO",
    "DecisionTreePolicy",
    "RWDTP",
    "RCDTP",
    "PQN",
    "CrossQ",
    "TQC",
    "RecurrentPPO",
]
