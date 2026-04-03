"""
Unified Counterfactual RL Agent

PPO agent for counterfactual explanation generation.
Supports selection from 5 strategies (DT and LR based) with depth parameter.
"""

from typing import Optional, Dict, Any
import logging
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

from .base_agent import BaseRLAgent, AgentConfig
from src.rl_agents.environments.counterfactual_env import CounterfactualEnv

logger = logging.getLogger(__name__)


class CounterfactualAgent(BaseRLAgent):
    """
    Unified RL agent for counterfactual explanation generation.
    
    Learns to select strategies and parameters that generate valid counterfactual
    explanations (feature changes that flip model predictions).
    
    Action space: MultiDiscrete([5, 3])
    - Strategy selection: 5 options (change_path_dt, zero_out_lr_heuristic, etc.)
    - Depth parameter: 3 options (for DT traversal depth)
    """
    
    def __init__(self, config: AgentConfig, env_config: Optional[Dict[str, Any]] = None):
        """
        Initialize CounterfactualAgent.
        
        Args:
            config: AgentConfig with PPO hyperparameters
            env_config: Dict with ai_dataset_loaders, ais, transforms, lr_exps, dt_exps, cog_params
        """
        self.env_config = env_config or {}
        super().__init__(config, CounterfactualEnv)
    
    def create_env(self, n_envs: int = 1, is_eval: bool = False) -> VecEnv:
        """
        Create vectorized CounterfactualEnv.
        
        Args:
            n_envs: Number of parallel environments
            is_eval: Whether for evaluation
        
        Returns:
            VecEnv (DummyVecEnv for n_envs=1, SubprocVecEnv otherwise)
        """
        def _make_env(rank: int):
            def _init():
                env = CounterfactualEnv(
                    ai_dataset_loaders=self.env_config.get('ai_dataset_loaders', {}),
                    ais=self.env_config.get('ais', {}),
                    transforms=self.env_config.get('transforms', {}),
                    lr_exps=self.env_config.get('lr_exps', {}),
                    dt_exps=self.env_config.get('dt_exps', {}),
                    cog_params=self.env_config.get('cog_params', {}),
                    instances_per_episode=40,
                    max_features=6,
                    eval_overrides=self.env_config.get('eval_overrides', {}),
                )
                return env
            return _init
        
        if n_envs == 1:
            return DummyVecEnv([_make_env(0)])
        else:
            return SubprocVecEnv([_make_env(i) for i in range(n_envs)])
