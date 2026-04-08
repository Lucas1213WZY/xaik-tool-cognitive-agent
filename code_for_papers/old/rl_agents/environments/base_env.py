"""
Base RL Environment with Memory and Strategy System Integration

Abstract base class for all RL training environments that integrates:
- Unified Memory system (ACT-R / Exemplar backends)
- Reasoning Strategies registry (LR Calculation, LR Heuristic, Decision Tree)
- Gymnasium API for compatibility with Stable-Baselines3

Design:
- Manages memory lifecycle and strategy instantiation
- Provides action/observation space contracts
- Defines reward structure for cognitive strategy selection
- Supports variable cognitive parameters (chi, latency factors, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.cognitive_models.memory import UnifiedMemory, MemoryConfig, MemoryBackend
from src.cognitive_models import StrategyRegistry, StrategyConfig, StrategyType, ReasoningMode


@dataclass
class EnvironmentConfig:
    """Configuration for RL environment initialization."""
    
    # Dataset and task configuration
    dataset_name: str = "wine_quality"
    model_type: str = "mlp"
    model_path: Optional[str] = None
    
    # Memory system configuration
    memory_backend: MemoryBackend = MemoryBackend.ACTR
    decay_param: float = 0.5
    retrieval_threshold: float = -2.5
    latency_factor: float = 0.5
    activation_noise: float = 0.1
    max_assoc_strength: float = 2.0
    mismatch_penalty: float = -2.0
    
    # Cognitive parameters (can be fixed or sampled)
    chi_range: Tuple[float, float] = (0.5, 2.0)  # Time cost sensitivity
    ddm_a_range: Tuple[float, float] = (0.3, 1.0)  # DDM boundary
    ddm_s_range: Tuple[float, float] = (0.8, 1.0)  # DDM scaling
    
    # Training configuration
    max_features: int = 6
    instances_per_episode: int = 40
    xai_trial_ratio: float = 0.5  # Fraction of trials with XAI
    
    # Data loaders
    data_loader: Optional[Any] = None  # External data loader
    
    # Callbacks and utilities
    seed: Optional[int] = None
    verbose: bool = False


class BaseRLEnvironment(gym.Env, ABC):
    """
    Abstract base for all RL environments using unified memory and strategies.
    
    Responsibilities:
    1. Initialize and manage UnifiedMemory with current strategies
    2. Load reasoning strategies from StrategyRegistry
    3. Define action/observation spaces
    4. Implement reset() and step() loops with strategy integration
    5. Compute rewards based on correctness and cognitive cost
    6. Track episode statistics
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize environment with memory and strategy system.
        
        Args:
            config: EnvironmentConfig with all parameters
        
        Raises:
            ValueError: If memory backend or strategies not available
        """
        super().__init__()
        self.config = config
        self.np_random = None
        
        # Initialize memory with unified system
        self._init_memory()
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.episode_data = []
        
        if config.seed is not None:
            self.seed(config.seed)
    
    def _init_memory(self) -> None:
        """Initialize UnifiedMemory with configured backend."""
        mem_config = MemoryConfig(
            backend=self.config.memory_backend,
            decay_param=self.config.decay_param,
            retrieval_threshold=self.config.retrieval_threshold,
            latency_factor=self.config.latency_factor,
            activation_noise=self.config.activation_noise,
            max_assoc_strength=self.config.max_assoc_strength,
            mismatch_penalty=self.config.mismatch_penalty,
        )
        self.memory = UnifiedMemory(mem_config)
    
    def _load_reasoning_strategy(self, strategy_name: str, 
                                 strategy_config: Dict[str, Any]) -> Any:
        """
        Load reasoning strategy from registry with given config.
        
        Args:
            strategy_name: Name in StrategyRegistry (e.g., "lr_calculation")
            strategy_config: Additional config parameters
        
        Returns:
            Instantiated strategy object
        
        Raises:
            KeyError: If strategy not in registry
        """
        try:
            strategy_obj = StrategyRegistry.get(strategy_name)
            if strategy_obj is None:
                raise KeyError(f"Strategy '{strategy_name}' not found in registry")
            
            config = StrategyConfig(
                strategy_name=strategy_name,
                strategy_type=self._infer_strategy_type(strategy_name),
                mode=ReasoningMode.RETRIEVE,
                decay_param=self.config.decay_param,
                retrieval_threshold=self.config.retrieval_threshold,
                sensitivity=10.0,
                extra_params=strategy_config
            )
            return strategy_obj(config)
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Failed to load strategy '{strategy_name}': {e}")
            raise
    
    @staticmethod
    def _infer_strategy_type(strategy_name: str) -> StrategyType:
        """Infer strategy type from name."""
        if "lr_calculation" in strategy_name.lower():
            return StrategyType.COXAM_FORWARD
        elif "lr_heuristic" in strategy_name.lower():
            return StrategyType.COXAM_FORWARD
        elif "dt" in strategy_name.lower() or "decision" in strategy_name.lower():
            return StrategyType.COXAM_FORWARD
        else:
            return StrategyType.COXAM_FORWARD  # Default
    
    def _sample_cognitive_parameters(self) -> Dict[str, float]:
        """
        Sample cognitive parameters (chi, ddm_a, ddm_s) for trial.
        
        Returns:
            Dict with sampled parameter values
        """
        params = {
            "chi": self.np_random.uniform(*self.config.chi_range),
            "ddm_a": self.np_random.uniform(*self.config.ddm_a_range),
            "ddm_s": self.np_random.uniform(*self.config.ddm_s_range),
        }
        return params
    
    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options (dataset_id, complexity, etc.)
        
        Returns:
            (observation, info) tuple
        """
        pass
    
    @abstractmethod
    def step(self, action: Union[int, np.ndarray, List[int]]) -> Tuple[
        np.ndarray, float, bool, bool, Dict[str, Any]
    ]:
        """
        Execute one step with RL action.
        
        Args:
            action: Strategy selection and/or parameter action
        
        Returns:
            (observation, reward, terminated, truncated, info) Gymnasium standard
        """
        pass
    
    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment RNG."""
        pass
    
    def render(self, mode: str = "human") -> None:
        """Render environment state (if applicable)."""
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        if self.memory:
            self.memory.clear()
    
    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get statistics for current episode.
        
        Returns:
            Dict with episode metrics (success rate, avg reward, etc.)
        """
        if not self.episode_data:
            return {}
        
        data_array = np.array(self.episode_data)
        return {
            "episode_length": len(self.episode_data),
            "total_reward": float(np.sum(data_array[:, 0])),
            "mean_reward": float(np.mean(data_array[:, 0])),
            "accuracy": float(np.mean(data_array[:, -1])) if data_array.shape[1] > 1 else 0.0,
        }
