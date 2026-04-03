"""
Inference utilities for RL agents.

Provides inference managers for batch prediction and model serving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class InferenceManager:
    """
    Manages inference with trained RL agents.
    
    Supports:
    - Single instance inference
    - Batch inference
    - Ensemble inference
    - Result caching
    """
    
    def __init__(self, agent, batch_size: int = 32):
        """
        Initialize inference manager.
        
        Args:
            agent: RLAgent instance with loaded weights
            batch_size: Batch size for inference
        """
        self.agent = agent
        self.batch_size = batch_size
        self.inference_cache: Dict[str, Any] = {}
    
    def predict_single(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Predict action for single observation.
        
        Args:
            observation: Single observation vector
            deterministic: Use deterministic policy
            cache_key: Optional key to cache result
            
        Returns:
            (action, state_info)
        """
        if cache_key and cache_key in self.inference_cache:
            return self.inference_cache[cache_key]
        
        action, state_info = self.agent.predict(
            observation, deterministic=deterministic
        )
        
        if cache_key:
            self.inference_cache[cache_key] = (action, state_info)
        
        return action, state_info
    
    def predict_batch(
        self,
        observations: List[np.ndarray],
        deterministic: bool = True,
    ) -> List[Tuple[np.ndarray, Optional[Any]]]:
        """
        Predict actions for batch of observations.
        
        Args:
            observations: List of observation vectors
            deterministic: Use deterministic policy
            
        Returns:
            List of (action, state_info) tuples
        """
        results = []
        
        for i in range(0, len(observations), self.batch_size):
            batch = observations[i:i + self.batch_size]
            
            for obs in batch:
                action, state_info = self.agent.predict(
                    obs, deterministic=deterministic
                )
                results.append((action, state_info))
        
        return results
    
    def predict_with_uncertainty(
        self,
        observation: np.ndarray,
        n_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Estimate prediction uncertainty via stochastic predictions.
        
        Args:
            observation: Single observation vector
            n_samples: Number of stochastic samples
            
        Returns:
            dict with action statistics
        """
        actions = []
        
        for _ in range(n_samples):
            action, _ = self.agent.predict(
                observation, deterministic=False
            )
            actions.append(action if isinstance(action, np.ndarray) else np.array(action))
        
        actions = np.array(actions)
        
        # Compute statistics
        if actions.ndim == 1:
            action_mean = float(np.mean(actions))
            action_std = float(np.std(actions))
            action_mode = int(np.argmax(np.bincount(actions.astype(int))))
        else:
            # Multi-dimensional action
            action_mean = np.mean(actions, axis=0)
            action_std = np.std(actions, axis=0)
            action_mode = np.median(actions, axis=0).astype(int)
        
        # Deterministic action for reference
        det_action, _ = self.agent.predict(observation, deterministic=True)
        
        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "action_mode": action_mode,
            "deterministic_action": det_action,
            "n_samples": n_samples,
        }
    
    def clear_cache(self):
        """Clear inference cache."""
        self.inference_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage."""
        return {
            "cache_size": len(self.inference_cache),
            "memory_usage_bytes": sum(
                self._estimate_size(v) for v in self.inference_cache.values()
            ),
        }
    
    @staticmethod
    def _estimate_size(obj: Any) -> int:
        """Estimate memory size of object in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (int, float)):
            return 8
        elif isinstance(obj, tuple):
            return sum(InferenceManager._estimate_size(item) for item in obj)
        else:
            return 0
