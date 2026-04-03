"""
Base RL Agent with PPO Training/Inference

Abstract base class for all RL agents used in strategy selection.
Wraps Stable-Baselines3 PPO for training and inference.

Design:
- Manages policy network lifecycle
- Handles model save/load with metadata
- Provides training harness with monitoring
- Supports both training and evaluation modes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Type
import os
import json
import numpy as np
from datetime import datetime
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for RL agent."""
    
    # Model identity
    agent_name: str = "default_agent"
    agent_type: str = "ppo"
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Network architecture
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    activation_fn: Type = nn.Tanh
    
    # Training configuration
    total_timesteps: int = 100_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 10
    
    # Checkpointing
    save_freq: int = 10_000
    save_path: str = "./model_checkpoints"
    
    # Other parameters
    seed: Optional[int] = None
    verbose: int = 1


class BaseRLAgent(ABC):
    """
    Abstract base class for RL agents.
    
    Responsibilities:
    1. Manage PPO policy (training, inference, save/load)
    2. Track training metadata and performance
    3. Provide evaluation harness
    4. Support batched inference
    """
    
    def __init__(self, config: AgentConfig, env_type: Type):
        """
        Initialize agent.
        
        Args:
            config: AgentConfig with all parameters
            env_type: Environment class (e.g., DTForwardEnvironment)
        """
        self.config = config
        self.env_type = env_type
        
        self.model = None
        self.train_env = None
        self.eval_env = None
        
        self.training_metadata = {
            "start_time": None,
            "end_time": None,
            "total_timesteps": 0,
            "eval_results": [],
        }
    
    @abstractmethod
    def create_env(self, n_envs: int = 1, is_eval: bool = False) -> VecEnv:
        """
        Create vectorized environment.
        
        Args:
            n_envs: Number of parallel environments
            is_eval: Whether this is for evaluation
        
        Returns:
            VecEnv instance
        """
        pass
    
    def _create_ppo_model(self, env: VecEnv) -> PPO:
        """
        Create PPO model.
        
        Args:
            env: Vectorized environment
        
        Returns:
            Initialized PPO model
        """
        policy_kwargs = {
            "net_arch": self.config.net_arch,
            "activation_fn": self.config.activation_fn,
        }
        
        model = PPO(
            policy="MultiInputPolicy",  # or "MlpPolicy" depending on observation structure
            env=env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            policy_kwargs=policy_kwargs,
            seed=self.config.seed,
            verbose=self.config.verbose,
        )
        
        return model
    
    def train(self, n_envs: int = 4, total_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            n_envs: Number of parallel training environments
            total_timesteps: Optional override for training timesteps
        
        Returns:
            Dict with training results
        """
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
        
        self.training_metadata["start_time"] = datetime.now().isoformat()
        
        try:
            # Create training environment
            self.train_env = self.create_env(n_envs=n_envs, is_eval=False)
            
            # Create PPO model
            if self.model is None:
                self.model = self._create_ppo_model(self.train_env)
            else:
                self.model.set_env(self.train_env)
            
            # Setup callbacks
            os.makedirs(self.config.save_path, exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config.save_freq,
                save_path=self.config.save_path,
                name_prefix=self.config.agent_name,
            )
            
            # Train
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=checkpoint_callback,
                progress_bar=True,
            )
            
            self.training_metadata["total_timesteps"] += total_timesteps
            
            # Evaluate
            eval_results = self.evaluate(n_episodes=self.config.n_eval_episodes)
            self.training_metadata["eval_results"].append(eval_results)
            
            self.training_metadata["end_time"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "total_timesteps": total_timesteps,
                "eval_results": eval_results,
            }
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
        finally:
            if self.train_env is not None:
                self.train_env.close()
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agent on test environment.
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            Dict with evaluation metrics
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return {}
        
        try:
            eval_env = self.create_env(n_envs=1, is_eval=True)
            
            episode_rewards = []
            episode_lengths = []
            
            for _ in range(n_episodes):
                obs, _ = eval_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            eval_env.close()
            
            return {
                "mean_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "mean_length": float(np.mean(episode_lengths)),
                "n_episodes": n_episodes,
            }
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict action from observation.
        
        Args:
            observation: Input observation
            deterministic: Whether to use deterministic policy
        
        Returns:
            (action, _value) tuple from PPO
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Train or load first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def predict_batch(self, observations: List[np.ndarray], 
                     deterministic: bool = True) -> np.ndarray:
        """
        Predict actions for batch of observations.
        
        Args:
            observations: List of observations
            deterministic: Deterministic inference
        
        Returns:
            Array of actions
        """
        actions = []
        for obs in observations:
            action, _ = self.predict(obs, deterministic=deterministic)
            actions.append(action)
        return np.array(actions)
    
    def save(self, path: str, include_metadata: bool = True) -> bool:
        """
        Save model and metadata.
        
        Args:
            path: Path to save to
            include_metadata: Whether to save metadata JSON
        
        Returns:
            Success flag
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if self.model:
                self.model.save(path)
            
            if include_metadata:
                metadata_path = path + ".metadata.json"
                metadata = {
                    "config": {
                        "agent_name": self.config.agent_name,
                        "learning_rate": self.config.learning_rate,
                        "net_arch": self.config.net_arch,
                    },
                    "training": self.training_metadata,
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load model from path.
        
        Args:
            path: Path to model
        
        Returns:
            Success flag
        """
        try:
            self.model = PPO.load(path)
            
            # Load metadata if available
            metadata_path = path + ".metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.training_metadata.update(metadata.get("training", {}))
            
            logger.info(f"Model loaded from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get training metadata and config."""
        return {
            "config": {
                "agent_name": self.config.agent_name,
                "learning_rate": self.config.learning_rate,
                "net_arch": self.config.net_arch,
                "total_timesteps": self.config.total_timesteps,
            },
            "training": self.training_metadata,
        }
