"""
Logistic Regression RL Agents

Specialized PPO agents for LR strategy selection training.
"""

from typing import Optional, Union
import logging
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

from .base_agent import BaseRLAgent, AgentConfig
from src.rl_agents.environments import (
    LRCalculationEnvironment, 
    LRHeuristicEnvironment,
    EnvironmentConfig
)

logger = logging.getLogger(__name__)


class LRCalculationAgent(BaseRLAgent):
    """
    RL Agent for LR Calculation strategy with feature selection.
    
    Trains on LRCalculationEnvironment to learn which features to focus on
    for improved reasoning.
    """
    
    def __init__(self, config: AgentConfig, env_config: Optional[EnvironmentConfig] = None):
        """
        Initialize LRCalculationAgent.
        
        Args:
            config: AgentConfig with PPO hyperparameters
            env_config: Environment configuration
        """
        super().__init__(config, LRCalculationEnvironment)
        self.env_config = env_config or EnvironmentConfig(
            dataset_name="wine_quality",
            instances_per_episode=40,
            max_features=6,
        )
    
    def create_env(self, n_envs: int = 1, is_eval: bool = False) -> VecEnv:
        """Create vectorized LRCalculationEnvironment."""
        def _make_env(rank: int):
            def _init():
                env_cfg = EnvironmentConfig(
                    dataset_name=self.env_config.dataset_name,
                    instances_per_episode=self.env_config.instances_per_episode,
                    max_features=self.env_config.max_features,
                    seed=self.config.seed + rank if self.config.seed else None,
                    verbose=self.config.verbose > 0,
                )
                return LRCalculationEnvironment(env_cfg)
            return _init
        
        if n_envs == 1:
            return DummyVecEnv([_make_env(0)])
        else:
            return SubprocVecEnv([_make_env(i) for i in range(n_envs)])


class LRHeuristicAgent(BaseRLAgent):
    """
    RL Agent for LR Heuristic strategy with parameter adaptation.
    
    Trains on LRHeuristicEnvironment to learn optimal chi parameter
    adjustment.
    """
    
    def __init__(self, config: AgentConfig, env_config: Optional[EnvironmentConfig] = None):
        """
        Initialize LRHeuristicAgent.
        
        Args:
            config: AgentConfig with PPO hyperparameters
            env_config: Environment configuration
        """
        super().__init__(config, LRHeuristicEnvironment)
        self.env_config = env_config or EnvironmentConfig(
            dataset_name="wine_quality",
            instances_per_episode=40,
        )
    
    def create_env(self, n_envs: int = 1, is_eval: bool = False) -> VecEnv:
        """Create vectorized LRHeuristicEnvironment."""
        def _make_env(rank: int):
            def _init():
                env_cfg = EnvironmentConfig(
                    dataset_name=self.env_config.dataset_name,
                    instances_per_episode=self.env_config.instances_per_episode,
                    seed=self.config.seed + rank if self.config.seed else None,
                    verbose=self.config.verbose > 0,
                )
                return LRHeuristicEnvironment(env_cfg)
            return _init
        
        if n_envs == 1:
            return DummyVecEnv([_make_env(0)])
        else:
            return SubprocVecEnv([_make_env(i) for i in range(n_envs)])


class LRAgent(BaseRLAgent):
    """
    Unified RL Agent supporting both LR strategies.
    
    Can be configured in external code to use either Calculation or Heuristic.
    """
    
    def __init__(self, config: AgentConfig, 
                 env_config: Optional[EnvironmentConfig] = None,
                 strategy_type: str = "calculation"):
        """
        Initialize unified LRAgent.
        
        Args:
            config: AgentConfig
            env_config: Environment configuration
            strategy_type: "calculation" or "heuristic"
        """
        self.strategy_type = strategy_type
        env_class = (
            LRCalculationEnvironment 
            if strategy_type == "calculation" 
            else LRHeuristicEnvironment
        )
        super().__init__(config, env_class)
        self.env_config = env_config or EnvironmentConfig(
            dataset_name="wine_quality",
            instances_per_episode=40,
        )
    
    def create_env(self, n_envs: int = 1, is_eval: bool = False) -> VecEnv:
        """Create vectorized LR environment."""
        def _make_env(rank: int):
            def _init():
                env_cfg = EnvironmentConfig(
                    dataset_name=self.env_config.dataset_name,
                    instances_per_episode=self.env_config.instances_per_episode,
                    max_features=self.env_config.max_features,
                    seed=self.config.seed + rank if self.config.seed else None,
                    verbose=self.config.verbose > 0,
                )
                
                if self.strategy_type == "calculation":
                    return LRCalculationEnvironment(env_cfg)
                else:
                    return LRHeuristicEnvironment(env_cfg)
            
            return _init
        
        if n_envs == 1:
            return DummyVecEnv([_make_env(0)])
        else:
            return SubprocVecEnv([_make_env(i) for i in range(n_envs)])
