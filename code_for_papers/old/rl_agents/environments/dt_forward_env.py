"""
Decision Tree Forward Environment

Environment for training agents to select Decision Tree retrieval strategy.

Action Space:
- MultiDiscrete([3, n_bins]) where:
  - a[0] ∈ {1,2}: 1="read", 2="retrieve"
  - a[1] ∈ {0..n_bins-1}: discretized ddm_a parameter

Observation Space:
- Compact features: [chi_norm, trial_norm, with_xai_flag, count_read, count_retrieve, succ_read, succ_retrieve]

Reward:
- accuracy - chi * predicted_time
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from gymnasium import spaces
import logging

from .base_env import BaseRLEnvironment, EnvironmentConfig
from src.reasoning_strategies import StrategyRegistry, ReasoningMode

logger = logging.getLogger(__name__)


class DTForwardEnvironment(BaseRLEnvironment):
    """
    RL environment for Decision Tree forward reasoning strategy selection.
    
    The agent learns to choose between "read" (direct model prediction) and
    "retrieve" (memory-based probabilistic reasoning) modes, plus a discretized
    ddm_a parameter for boundary adaptation.
    """
    
    # Action map
    STRATEGY_MAP = {
        1: "read",      # Direct model prediction
        2: "retrieve",  # Memory-based reasoning
    }
    
    def __init__(self, config: EnvironmentConfig):
        """Initialize Decision Tree environment."""
        super().__init__(config)
        
        # DDM parameter discretization
        self.ddm_a_bins = 5
        self.ddm_a_values = np.linspace(
            config.ddm_a_range[0], 
            config.ddm_a_range[1], 
            self.ddm_a_bins
        )
        
        # Action space: strategy selection (read/retrieve) + ddm_a bin
        self.action_space = spaces.MultiDiscrete(
            [len(self.STRATEGY_MAP) + 1, self.ddm_a_bins]
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # Episode state
        self.current_instance = None
        self.current_target = None
        self.trial_counter = 0
        self.with_xai = False
        
        # Strategy tracking
        self.strategy_counts = {"read": 0, "retrieve": 0}
        self.strategy_successes = {"read": 0, "retrieve": 0}
        
        # Load DT strategy from registry
        try:
            self.dt_strategy = StrategyRegistry.get("dt_traversal")
            if self.dt_strategy is None:
                logger.warning("dt_traversal strategy not found in registry")
        except Exception as e:
            logger.warning(f"Failed to load dt_traversal strategy: {e}")
            self.dt_strategy = None
    
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Optional dict with dataset_id, complexity, etc.
        
        Returns:
            (observation, info)
        """
        super_result = self._reset_base(seed)
        if super_result is not None:
            return super_result
        
        # Reset episode tracking
        self.trial_counter = 0
        self.strategy_counts = {"read": 0, "retrieve": 0}
        self.strategy_successes = {"read": 0, "retrieve": 0}
        self.episode_data = []
        
        # Generate first instance
        obs = self._get_observation()
        info = {"episode_start": True, "trial_num": self.trial_counter}
        
        return obs, info
    
    def _reset_base(self, seed: Optional[int]) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Seed the environment if provided."""
        if seed is not None:
            self.seed(seed)
        return None
    
    def step(self, action: Union[int, np.ndarray, List[int]]) -> Tuple[
        np.ndarray, float, bool, bool, Dict[str, Any]
    ]:
        """
        Execute one step.
        
        Args:
            action: [strategy_id, ddm_a_bin] where strategy_id ∈ {1,2}
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action)
        if action.ndim == 0:
            action = np.array([int(action)], dtype=int)
        action = action.astype(int)
        
        # Parse action
        strategy_id = int(action[0]) if len(action) > 0 else 1
        ddm_a_bin = int(action[1]) if len(action) > 1 else 0
        
        # Validate and map strategy
        if strategy_id not in self.STRATEGY_MAP:
            strategy_id = 1
        strategy_name = self.STRATEGY_MAP[strategy_id]
        ddm_a = self.ddm_a_values[ddm_a_bin % self.ddm_a_bins]
        
        # Sample cognitive parameters
        cognitive_params = self._sample_cognitive_parameters()
        cognitive_params["ddm_a"] = ddm_a
        
        # Determine whether this is an XAI trial
        self.with_xai = self.np_random.uniform() < self.config.xai_trial_ratio
        
        # Execute reasoning step
        try:
            if self.dt_strategy is None:
                # Fallback: random prediction
                pred_prob = self.np_random.uniform()
                prediction = 1 if pred_prob > 0.5 else 0
                time_cost = 0.5
            else:
                # Use loaded strategy
                strategy_mode = ReasoningMode.READ if strategy_name == "read" else ReasoningMode.RETRIEVE
                self.dt_strategy.config.mode = strategy_mode
                
                probs, time_cost, info_dict = self.dt_strategy.infer(
                    features=self.current_instance,
                    explanation=None if not self.with_xai else {},
                    **cognitive_params
                )
                prediction = 1 if probs.get(1, 0) > 0.5 else 0
        except Exception as e:
            logger.warning(f"Strategy inference failed: {e}")
            prediction = self.np_random.randint(0, 2)
            time_cost = 0.5
        
        # Compute reward
        correct = (prediction == self.current_target)
        chi = cognitive_params["chi"]
        reward = float(correct) - chi * time_cost
        
        # Update tracking
        self.strategy_counts[strategy_name] += 1
        if correct:
            self.strategy_successes[strategy_name] += 1
        
        self.trial_counter += 1
        terminated = self.trial_counter >= self.config.instances_per_episode
        truncated = False
        
        # Record episode data
        self.episode_data.append([reward, chi, time_cost, correct])
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            "trial_num": self.trial_counter,
            "strategy": strategy_name,
            "correct": correct,
            "time_cost": time_cost,
            "chi": chi,
            "reward": reward,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            [chi_norm, trial_norm, with_xai_flag, count_read, count_retrieve, succ_read, succ_retrieve]
        """
        # Sample new random instance if needed
        if self.current_instance is None or self.trial_counter >= self.config.instances_per_episode:
            self.current_instance = np.random.randn(self.config.max_features)
            self.current_target = self.np_random.randint(0, 2)
        
        total_trials = max(1, sum(self.strategy_counts.values()))
        
        obs = np.array([
            0.5,  # chi normalized (placeholder)
            min(1.0, self.trial_counter / max(1, self.config.instances_per_episode)),  # trial progress
            float(self.with_xai),  # with_xai flag
            self.strategy_counts["read"] / total_trials,  # read frequency
            self.strategy_counts["retrieve"] / total_trials,  # retrieve frequency
            self.strategy_successes["read"] / max(1, self.strategy_counts["read"]),  # read success rate
            self.strategy_successes["retrieve"] / max(1, self.strategy_counts["retrieve"]),  # retrieve success rate
        ], dtype=np.float32)
        
        return obs
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the RNG."""
        self.np_random, seed_val = self.np_random, seed
        return [seed_val] if seed_val is not None else []
