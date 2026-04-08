"""
LR Forward Environments

Environments for training agents with LR (Logistic Regression) reasoning strategies.

Two variants:
1. LRCalculationEnvironment: For LR Calculation strategy selection
2. LRHeuristicEnvironment: For LR Heuristic strategy selection

Both support feature selection as the main action space.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
from collections import deque
import numpy as np
from gymnasium import spaces
import logging

from .base_env import BaseRLEnvironment, EnvironmentConfig
from src.cognitive_models import StrategyRegistry, ReasoningMode

logger = logging.getLogger(__name__)


class LRCalculationEnvironment(BaseRLEnvironment):
    """
    RL environment for LR Calculation strategy training.
    
    Action Space:
    - MultiBinary(max_features): Feature selection mask
    
    Observation Space:
    - Features representing instance state, past selections, success rates
    """
    
    def __init__(self, config: EnvironmentConfig):
        """Initialize LR Calculation environment."""
        super().__init__(config)
        
        # Action space: binary feature selection
        self.action_space = spaces.MultiBinary(config.max_features)
        
        # Observation space: instance features + history + stats
        # [instance_features] + [past_selection] + [success_stats]
        obs_dim = config.max_features + config.max_features + 3
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Episode state
        self.current_instance = None
        self.current_target = None
        self.trial_counter = 0
        self.with_xai = False
        
        # History tracking
        self.feature_selection_history = deque(maxlen=10)
        self.success_history = deque(maxlen=10)
        
        # Load LR Calculation strategy
        try:
            self.lr_strategy = StrategyRegistry.get("lr_calculation")
            if self.lr_strategy is None:
                logger.warning("lr_calculation strategy not found in registry")
        except Exception as e:
            logger.warning(f"Failed to load lr_calculation strategy: {e}")
            self.lr_strategy = None
    
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        if seed is not None:
            self.seed(seed)
        
        self.trial_counter = 0
        self.feature_selection_history.clear()
        self.success_history.clear()
        self.episode_data = []
        
        # Generate first instance
        self._sample_instance()
        obs = self._get_observation()
        
        info = {"episode_start": True, "trial_num": self.trial_counter}
        return obs, info
    
    def step(self, action: Union[np.ndarray, List[int]]) -> Tuple[
        np.ndarray, float, bool, bool, Dict[str, Any]
    ]:
        """
        Execute one step with feature selection.
        
        Args:
            action: Binary array of length max_features
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=int)
        
        # Ensure valid feature mask
        if len(action) < self.config.max_features:
            action = np.pad(action, (0, self.config.max_features - len(action)))
        action = action[:self.config.max_features]
        
        # At least one feature must be selected
        if np.sum(action) == 0:
            action[0] = 1
        
        # Sample cognitive parameters
        cognitive_params = self._sample_cognitive_parameters()
        self.with_xai = self.np_random.uniform() < self.config.xai_trial_ratio
        
        # Execute reasoning with selected features
        try:
            if self.lr_strategy is None:
                pred_prob = self.np_random.uniform()
                prediction = 1 if pred_prob > 0.5 else 0
                time_cost = 0.5
            else:
                # Apply feature mask to instance
                masked_instance = self.current_instance * action
                
                probs, time_cost, info_dict = self.lr_strategy.infer(
                    features=masked_instance,
                    explanation=None if not self.with_xai else {},
                    **cognitive_params
                )
                prediction = 1 if probs.get(1, 0) > 0.5 else 0
        except Exception as e:
            logger.warning(f"Strategy inference failed: {e}")
            prediction = self.np_random.randint(0, 2)
            time_cost = 0.5
        
        # Compute reward: accuracy - feature selection cost
        correct = (prediction == self.current_target)
        num_features_selected = np.sum(action)
        feature_cost = num_features_selected * 0.1  # Cost per feature
        chi = cognitive_params["chi"]
        reward = float(correct) - chi * time_cost - feature_cost
        
        # Update history
        self.feature_selection_history.append(action.copy())
        self.success_history.append(correct)
        
        self.trial_counter += 1
        terminated = self.trial_counter >= self.config.instances_per_episode
        truncated = False
        
        self.episode_data.append([reward, chi, time_cost, correct])
        self._sample_instance()  # Prepare next instance
        obs = self._get_observation()
        
        info = {
            "trial_num": self.trial_counter,
            "correct": correct,
            "time_cost": time_cost,
            "num_features": int(num_features_selected),
            "reward": reward,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _sample_instance(self) -> None:
        """Generate new random instance."""
        self.current_instance = self.np_random.randn(self.config.max_features).astype(np.float32)
        self.current_target = self.np_random.randint(0, 2)
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            [instance_features, past_selection, success_rate, feature_count, trial_progress]
        """
        # Instance features (normalized)
        instance_obs = self.current_instance / (np.std(self.current_instance) + 1e-6)
        instance_obs = np.clip(instance_obs, -5, 5).astype(np.float32)
        
        # Past feature selection (mean over history)
        if self.feature_selection_history:
            past_selection = np.mean(list(self.feature_selection_history), axis=0).astype(np.float32)
        else:
            past_selection = np.zeros(self.config.max_features, dtype=np.float32)
        
        # Success statistics
        if self.success_history:
            success_rate = np.mean(self.success_history)
            num_selected_avg = np.mean([np.sum(sel) for sel in self.feature_selection_history])
        else:
            success_rate = 0.5
            num_selected_avg = 3.0
        
        trial_progress = min(1.0, self.trial_counter / max(1, self.config.instances_per_episode))
        
        # Concatenate observation
        obs = np.concatenate([
            instance_obs,
            past_selection,
            [success_rate, num_selected_avg, trial_progress]
        ]).astype(np.float32)
        
        return obs
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the RNG."""
        self.np_random, seed_val = self.np_random, seed
        return [seed_val] if seed_val is not None else []


class LRHeuristicEnvironment(BaseRLEnvironment):
    """
    RL environment for LR Heuristic strategy training.
    
    Simpler than LRCalculation - focuses on cognitive parameter selection
    rather than feature selection.
    
    Action Space:
    - Discrete(3): Parameter adjustment (decrease, keep, increase)
    """
    
    def __init__(self, config: EnvironmentConfig):
        """Initialize LR Heuristic environment."""
        super().__init__(config)
        
        # Action space: parameter tuning
        self.action_space = spaces.Discrete(3)  # [decrease, keep, increase]
        
        # Observation space
        obs_dim = config.max_features + 4  # instance + stats
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Episode state
        self.current_instance = None
        self.current_target = None
        self.trial_counter = 0
        self.with_xai = False
        
        # Parameter adaptation
        self.current_chi = 1.0
        
        # Load LR Heuristic strategy
        try:
            self.lr_heuristic = StrategyRegistry.get("lr_heuristic")
            if self.lr_heuristic is None:
                logger.warning("lr_heuristic strategy not found in registry")
        except Exception as e:
            logger.warning(f"Failed to load lr_heuristic strategy: {e}")
            self.lr_heuristic = None
    
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            self.seed(seed)
        
        self.trial_counter = 0
        self.current_chi = 1.0
        self.episode_data = []
        
        self._sample_instance()
        obs = self._get_observation()
        
        return obs, {"episode_start": True}
    
    def step(self, action: int) -> Tuple[
        np.ndarray, float, bool, bool, Dict[str, Any]
    ]:
        """
        Execute one step.
        
        Args:
            action: 0=decrease_chi, 1=keep, 2=increase_chi
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = int(action) % 3
        
        # Adjust chi parameter
        if action == 0:  # decrease
            self.current_chi *= 0.95
        elif action == 2:  # increase
            self.current_chi *= 1.05
        self.current_chi = np.clip(self.current_chi, 0.5, 3.0)
        
        # Sample other parameters
        cognitive_params = self._sample_cognitive_parameters()
        cognitive_params["chi"] = self.current_chi
        self.with_xai = self.np_random.uniform() < self.config.xai_trial_ratio
        
        # Execute reasoning
        try:
            if self.lr_heuristic is None:
                pred_prob = self.np_random.uniform()
                prediction = 1 if pred_prob > 0.5 else 0
                time_cost = 0.3
            else:
                probs, time_cost, info_dict = self.lr_heuristic.infer(
                    features=self.current_instance,
                    **cognitive_params
                )
                prediction = 1 if probs.get(1, 0) > 0.5 else 0
        except Exception as e:
            logger.warning(f"Strategy inference failed: {e}")
            prediction = self.np_random.randint(0, 2)
            time_cost = 0.3
        
        # Compute reward
        correct = (prediction == self.current_target)
        reward = float(correct) - self.current_chi * time_cost
        
        self.trial_counter += 1
        terminated = self.trial_counter >= self.config.instances_per_episode
        truncated = False
        
        self.episode_data.append([reward, self.current_chi, time_cost, correct])
        self._sample_instance()
        obs = self._get_observation()
        
        info = {
            "trial_num": self.trial_counter,
            "correct": correct,
            "chi": self.current_chi,
            "time_cost": time_cost,
            "reward": reward,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _sample_instance(self) -> None:
        """Generate new instance."""
        self.current_instance = self.np_random.randn(self.config.max_features).astype(np.float32)
        self.current_target = self.np_random.randint(0, 2)
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation."""
        instance_obs = self.current_instance / (np.std(self.current_instance) + 1e-6)
        instance_obs = np.clip(instance_obs, -5, 5).astype(np.float32)
        
        trial_progress = min(1.0, self.trial_counter / max(1, self.config.instances_per_episode))
        
        obs = np.concatenate([
            instance_obs,
            [self.current_chi / 3.0, trial_progress, float(self.with_xai), 0.5]
        ]).astype(np.float32)
        
        return obs
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed RNG."""
        self.np_random, seed_val = self.np_random, seed
        return [seed_val] if seed_val is not None else []
