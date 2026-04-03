"""
Base RL Environment for CoXAM cognitive agents.

Provides common functionality for gym-compatible decision environments
used in training cognitive agent models.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvironmentConfig:
    """Configuration for RL environments."""
    
    instances_per_episode: int = 40
    max_features: int = 6
    chi_low: float = 0.0
    chi_high: float = 0.03
    xai_trial_ratio: float = 0.5
    
    # Cognitive parameters (ranges or fixed values)
    cog_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training setup
    training: bool = True
    time_penalty_scale: float = 1.0
    instance_id_pool: Optional[List[int]] = None
    
    # Dataset loaders and explainers
    ai_dataset_loaders: Optional[Dict[str, Any]] = None
    explainers: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and finalize configuration."""
        if self.instance_id_pool is None:
            self.instance_id_pool = list(range(1, 400))
        if self.ai_dataset_loaders is None:
            self.ai_dataset_loaders = {}
        if self.explainers is None:
            self.explainers = {}


class BaseRLEnvironment(gym.Env, ABC):
    """
    Base class for RL environments for CoXAM cognitive agents.
    
    Provides:
    - Memory initialization and management
    - Cognitive parameter sampling
    - Instance/batch loading
    - Observation/reward computation
    - Episode scheduling (with-XAI trials)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: EnvironmentConfig):
        """Initialize base environment with configuration."""
        super().__init__()
        self.config = config
        
        # RNG for reproducibility
        self._rng = np.random.default_rng()
        
        # Episode state
        self.step_idx: int = 0
        self.curr_chi: float = 0.0
        self.with_xai_schedule: np.ndarray = np.zeros(
            config.instances_per_episode, dtype=np.bool_
        )
        
        # Memory (will be initialized by subclass)
        self.memory = None
        
        # Data buffers
        self.X_raw: Optional[np.ndarray] = None
        self.X_norm: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        
        # Currently selected dataset
        self.current_ai_dataset_loader = None
        self.current_explainer = None
        
        # Cognitive params for this episode
        self.current_cog_params: Dict[str, float] = {}
    
    # ========== Helper Methods ==========
    
    def _seed(self, seed: Optional[int] = None) -> np.random.Generator:
        """Seed and return RNG."""
        self._rng = np.random.default_rng(seed)
        return self._rng
    
    def _build_with_xai_schedule(
        self, n: int, ratio: float
    ) -> np.ndarray:
        """Build random schedule of with-XAI trial flags."""
        n_xai = int(round(n * ratio))
        flags = np.array(
            [True] * n_xai + [False] * (n - n_xai),
            dtype=np.bool_
        )
        self._rng.shuffle(flags)
        return flags
    
    def _sample_cog_params(self) -> Dict[str, float]:
        """Sample cognitive parameters from ranges or use fixed values."""
        params = {}
        for k, v in (self.config.cog_params or {}).items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                # Range: sample uniformly
                params[k] = float(self._rng.uniform(v[0], v[1]))
            elif isinstance(v, (int, float)):
                # Fixed value
                params[k] = float(v)
            else:
                params[k] = v
        return params
    
    def _load_instances(
        self,
        indices: List[int],
        normalize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load instances by indices from current dataset loader.
        
        Returns:
            (instances, labels) as numpy arrays
        """
        if self.current_ai_dataset_loader is None:
            raise RuntimeError("No dataset loader selected for episode")
        
        instances, labels = self.current_ai_dataset_loader.load_instances(
            indices, normalize=normalize
        )
        return np.asarray(instances, dtype=np.float32), np.asarray(
            labels, dtype=np.int64
        )
    
    def _select_dataset(self) -> Tuple[Any, Any]:
        """
        Randomly select a dataset and its explainer.
        
        Returns:
            (ai_dataset_loader, explainer)
        """
        if not self.config.ai_dataset_loaders:
            raise RuntimeError("No dataset loaders configured")
        
        dataset_key = self._rng.choice(
            list(self.config.ai_dataset_loaders.keys())
        )
        loader = self.config.ai_dataset_loaders[dataset_key]
        explainer = (self.config.explainers or {}).get(dataset_key, None)
        
        return loader, explainer
    
    # ========== Abstract Methods (to be implemented by subclasses) ==========
    
    @abstractmethod
    def _initialize_memory(self):
        """Initialize memory for the episode."""
        pass
    
    @abstractmethod
    def _build_obs(self) -> np.ndarray:
        """Build observation vector for current state."""
        pass
    
    @abstractmethod
    def _run_decision_strategy(
        self, action: np.ndarray
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Run the decision strategy for the current step.
        
        Args:
            action: Action from agent
            
        Returns:
            (reward, pred_time, info_dict)
        """
        pass
    
    # ========== Gymnasium API ==========
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment for new episode."""
        if seed is not None:
            self._seed(seed)
        
        # Select dataset and explainer
        self.current_ai_dataset_loader, self.current_explainer = (
            self._select_dataset()
        )
        
        # Build with-XAI schedule
        self.with_xai_schedule = self._build_with_xai_schedule(
            self.config.instances_per_episode,
            self.config.xai_trial_ratio,
        )
        
        # Sample cognitive parameters
        self.current_cog_params = self._sample_cog_params()
        
        # Initialize memory
        self._initialize_memory()
        
        # Sample chi and instances
        self.curr_chi = float(
            self._rng.uniform(self.config.chi_low, self.config.chi_high)
        )
        
        # Load instances for episode
        indices = self._rng.choice(
            self.config.instance_id_pool,
            size=self.config.instances_per_episode,
            replace=False,
        ).tolist()
        
        self.X_raw, self.y = self._load_instances(indices, normalize=False)
        self.X_norm, _ = self._load_instances(indices, normalize=True)
        
        # Reset episode state
        self.step_idx = 0
        
        obs = self._build_obs()
        info = {
            "cog_params": self.current_cog_params.copy(),
            "chi": self.curr_chi,
        }
        
        return obs, info
    
    def step(self, action):
        """Execute one step of the environment."""
        # Check if episode is over
        if self.step_idx >= self.config.instances_per_episode:
            truncated = True
            return (
                self._build_obs(),
                0.0,
                False,
                truncated,
                {"error": "Episode already finished"},
            )
        
        # Run the decision strategy
        try:
            reward, pred_time, step_info = self._run_decision_strategy(action)
        except Exception as e:
            self.step_idx += 1
            truncated = self.step_idx >= self.config.instances_per_episode
            return (
                self._build_obs(),
                -5.0,
                False,
                truncated,
                {"error": str(e)},
            )
        
        # Advance episode
        self.step_idx += 1
        truncated = self.step_idx >= self.config.instances_per_episode
        
        obs = self._build_obs()
        info = step_info.copy()
        info.update({
            "chi": self.curr_chi,
            "cog_params": self.current_cog_params.copy(),
        })
        
        return obs, float(reward), False, truncated, info
    
    def render(self):
        """Render environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up environment resources."""
        pass
