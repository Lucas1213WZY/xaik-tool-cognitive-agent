"""
Logistic Regression Agent implementation.

RL agent specialized for LR forward strategies (heuristic and calculation).
"""

import os
from typing import Optional, Tuple

from stable_baselines3 import PPO

from .base_agent import RLAgent, AgentConfig


class LRAgent(RLAgent):
    """
    RL Agent for Logistic Regression strategies.
    
    Trained to decide between LR heuristic, LR calculation, DT strategies
    with or without XAI, and to select feature masks.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize LR agent.
        
        Args:
            config: AgentConfig with agent_type in ("lr_heuristic", "lr_calculation")
        """
        super().__init__(config)
        
        valid_types = ("lr_heuristic", "lr_calculation", "lr", "hybrid")
        assert (
            config.agent_type in valid_types
        ), f"Expected agent_type in {valid_types}, got {config.agent_type}"
        
        self.policy: Optional[PPO] = None
        
        # Try to load weights if provided
        if config.model_checkpoint:
            self.load_weights(config.model_checkpoint)
    
    def predict(
        self, observation, deterministic: bool = True
    ) -> Tuple:
        """
        Predict action for LR strategy.
        
        Args:
            observation: Observation from LRForwardEnvironment
            deterministic: Use deterministic policy if True
            
        Returns:
            (action, _states) - action is MultiDiscrete [strategy_id, ...feature_mask]
        """
        if self.policy is None:
            raise RuntimeError(
                "Policy not loaded. Call load_weights() first."
            )
        
        action, _states = self.policy.predict(
            observation, deterministic=deterministic
        )
        
        return action, _states
    
    def load_weights(self, path: Optional[str] = None):
        """
        Load pre-trained policy weights.
        
        Args:
            path: Path to model file. If None, uses config.model_checkpoint
        """
        if path is None:
            path = self.config.model_checkpoint
        
        if path is None:
            raise ValueError(
                "No path provided and config.model_checkpoint is None"
            )
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model weights not found at {path}")
        
        try:
            self.policy = PPO.load(path)
            if self.config.verbose:
                print(f"✓ Loaded LR agent ({self.config.agent_type}) from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {path}: {e}")
        
        # Load metadata if available
        self.load_metadata()
    
    def save_weights(self, path: Optional[str] = None):
        """
        Save trained policy weights.
        
        Args:
            path: Path to save model. If None, uses config.model_weights_dir
        """
        if self.policy is None:
            raise RuntimeError("No policy to save. Train first or load weights.")
        
        if path is None:
            if not self.config.model_weights_dir:
                raise ValueError(
                    "No save path provided and config.model_weights_dir is None"
                )
            path = os.path.join(
                self.config.model_weights_dir,
                f"{self.config.agent_id}_lr_{self.config.agent_type}_agent"
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            self.policy.save(path)
            if self.config.verbose:
                print(f"✓ Saved LR agent to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model weights to {path}: {e}")
        
        # Save metadata
        self.save_metadata({
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "save_path": path,
            "policy_class": "PPO",
        })
    
    def train(self, env, total_timesteps: int, **kwargs) -> "LRAgent":
        """
        Train LR agent on environment.
        
        Args:
            env: Training environment (LRForwardEnvironment)
            total_timesteps: Total timesteps for training
            **kwargs: Additional PPO training arguments
            
        Returns:
            self
        """
        if self.policy is None:
            # Create new policy
            self.policy = PPO(
                "MlpPolicy",
                env,
                verbose=1 if self.config.verbose else 0,
                **kwargs
            )
        
        self.policy.learn(total_timesteps=total_timesteps)
        return self
    
    def evaluate(self, env, n_episodes: int = 10) -> dict:
        """
        Evaluate agent on environment.
        
        Args:
            env: Evaluation environment
            n_episodes: Number of episodes to evaluate
            
        Returns:
            dict with evaluation metrics
        """
        if self.policy is None:
            raise RuntimeError(
                "Policy not loaded. Call load_weights() first."
            )
        
        episode_returns = []
        episode_strategy_usage = {i: 0 for i in range(1, 6)}  # 5 strategies
        
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0.0
            
            while not done:
                action, _ = self.policy.predict(obs, deterministic=True)
                strategy_id = int(action[0]) if hasattr(action, '__getitem__') else int(action)
                if strategy_id in episode_strategy_usage:
                    episode_strategy_usage[strategy_id] += 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                done = terminated or truncated
            
            episode_returns.append(episode_return)
        
        import numpy as np
        return {
            "mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "min_return": float(np.min(episode_returns)),
            "max_return": float(np.max(episode_returns)),
            "episode_returns": episode_returns,
            "strategy_usage": episode_strategy_usage,
        }
