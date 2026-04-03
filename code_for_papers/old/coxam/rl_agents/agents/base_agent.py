"""
Base RL Agent for CoXAM cognitive models.

Provides abstract base class and common functionality for agent implementations.
"""

import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    
    agent_id: str
    agent_type: str  # "dt", "lr_heuristic", "lr_calculation"
    model_weights_dir: Optional[str] = None
    model_checkpoint: Optional[str] = None
    verbose: bool = False
    
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary."""
        return cls(**data)


class RLAgent(ABC):
    """
    Abstract base class for RL agents in CoXAM.
    
    Provides:
    - Standard inference interface
    - Weight loading/saving
    - Training/evaluation mode switching
    - Metadata management
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize agent.
        
        Args:
            config: AgentConfig instance
        """
        self.config = config
        self.policy = None  # Will be loaded from weights
        self.metadata: Dict[str, Any] = {}
        
        # Create weight directory if needed
        if config.model_weights_dir:
            os.makedirs(config.model_weights_dir, exist_ok=True)
    
    # ========== Abstract Methods ==========
    
    @abstractmethod
    def predict(
        self, observation, deterministic: bool = True
    ) -> tuple:
        """
        Make prediction from observation.
        
        Args:
            observation: Observation from environment
            deterministic: Whether to use deterministic policy
            
        Returns:
            (action, info_dict)
        """
        pass
    
    @abstractmethod
    def load_weights(self, path: Optional[str] = None):
        """
        Load pre-trained weights.
        
        Args:
            path: Path to weights. If None, uses config default.
        """
        pass
    
    @abstractmethod
    def save_weights(self, path: Optional[str] = None):
        """
        Save trained weights.
        
        Args:
            path: Path to save weights. If None, uses config default.
        """
        pass
    
    # ========== Common Methods ==========
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata for this agent."""
        if not self.config.model_weights_dir:
            return {}
        
        metadata_path = os.path.join(
            self.config.model_weights_dir, "metadata.json"
        )
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        
        return self.metadata
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata for this agent."""
        if not self.config.model_weights_dir:
            return
        
        self.metadata = metadata
        metadata_path = os.path.join(
            self.config.model_weights_dir, "metadata.json"
        )
        
        os.makedirs(
            os.path.dirname(metadata_path),
            exist_ok=True
        )
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.to_dict()
    
    def set_training_mode(self, training: bool = True):
        """Set training/evaluation mode."""
        if self.policy is not None:
            if training:
                self.policy.train()
            else:
                self.policy.eval()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.config.agent_id}, "
            f"type={self.config.agent_type}, "
            f"weights_dir={self.config.model_weights_dir})"
        )
