"""
Decision Tree RL Agent

Specialized PPO agent for Decision Tree strategy selection training.
"""

from typing import Optional, Type
import logging
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

from .base_agent import BaseRLAgent, AgentConfig
from src.rl_agents.environments import DTForwardEnvironment, EnvironmentConfig

logger = logging.getLogger(__name__)


class DTAgent(BaseRLAgent):
    """
    RL Agent for Decision Tree strategy selection.
    
    Trains on DTForwardEnvironment to learn optimal strategy selection
    (read vs retrieve) and parameter adaptation (ddm_a).
    """
    
    def __init__(self, config: AgentConfig, env_config: Optional[EnvironmentConfig] = None):
        """
        Initialize DTAgent.
        
        Args:
            config: AgentConfig with PPO hyperparameters
            env_config: Environment configuration (if None, uses defaults)
        """
        super().__init__(config, DTForwardEnvironment)
        self.env_config = env_config or EnvironmentConfig(
            dataset_name="wine_quality",
            instances_per_episode=40,
            xai_trial_ratio=0.5,
        )
    
    def create_env(self, n_envs: int = 1, is_eval: bool = False) -> VecEnv:
        """
        Create vectorized DTForwardEnvironment.
        
        Args:
            n_envs: Number of parallel environments
            is_eval: Whether for evaluation
        
        Returns:
            VecEnv (DummyVecEnv for n_envs=1, SubprocVecEnv otherwise)
        """
        def _make_env(rank: int):
            def _init():
                env_cfg = EnvironmentConfig(
                    dataset_name=self.env_config.dataset_name,
                    model_type=self.env_config.model_type,
                    instances_per_episode=self.env_config.instances_per_episode,
                    xai_trial_ratio=self.env_config.xai_trial_ratio,
                    seed=self.config.seed + rank if self.config.seed else None,
                    verbose=self.config.verbose > 0,
                )
                return DTForwardEnvironment(env_cfg)
            return _init
        
        if n_envs == 1:
            return DummyVecEnv([_make_env(0)])
        else:
            return SubprocVecEnv([_make_env(i) for i in range(n_envs)])
