"""
Decision Tree Agent implementation.

RL agent specialized for DT forward strategy decisions.
"""

import os
from typing import Optional, Tuple

from stable_baselines3 import PPO

from .base_agent import RLAgent, AgentConfig


class DTAgent(RLAgent):
    """
    RL Agent for Decision Tree strategy.
    
    Trained to select between read/retrieve modes and DDM-a parameter bins.
    Model weights are stored in configurable directory.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize DT agent.
        
        Args:
            config: AgentConfig with agent_type="dt"
        """
        super().__init__(config)
        assert config.agent_type == "dt", f"Expected agent_type='dt', got {config.agent_type}"
        
        self.policy: Optional[PPO] = None
        
        # Try to load weights if provided
        if config.model_checkpoint:
            self.load_weights(config.model_checkpoint)
    
    def predict(
        self, observation, deterministic: bool = True
    ) -> Tuple:
        """
        Predict action for DT strategy.
        
        Args:
            observation: Observation from DTForwardEnvironment
            deterministic: Use deterministic policy if True
            
        Returns:
            (action, _states) - action is MultiDiscrete [strategy_id, ddm_a_bin]
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
                print(f"✓ Loaded DT agent from {path}")
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
                f"{self.config.agent_id}_dt_agent"
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            self.policy.save(path)
            if self.config.verbose:
                print(f"✓ Saved DT agent to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model weights to {path}: {e}")
        
        # Save metadata
        self.save_metadata({
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "save_path": path,
            "policy_class": "PPO",
        })
    
    def train(self, env, total_timesteps: int, **kwargs) -> "DTAgent":
        """
        Train DT agent on environment.
        
        Args:
            env: Training environment (DTForwardEnvironment)
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
        
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0.0
            
            while not done:
                action, _ = self.policy.predict(obs, deterministic=True)
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
        }
