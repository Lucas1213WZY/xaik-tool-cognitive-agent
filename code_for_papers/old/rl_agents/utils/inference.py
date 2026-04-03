"""
Inference Manager for RL Agents

Provides optimized batch inference, caching, and uncertainty estimation.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    action: np.ndarray
    value: float
    confidence: float = 1.0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PredictionCache:
    """
    Simple LRU cache for predictions.
    
    Useful for repeated observations in evaluation or batch inference.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached predictions
        """
        self.max_size = max_size
        self.cache: Dict[str, PredictionResult] = {}
        self.hits = 0
        self.misses = 0
    
    def _obs_key(self, obs: np.ndarray) -> str:
        """Generate cache key from observation."""
        return hash(obs.tobytes()).__str__()
    
    def get(self, obs: np.ndarray) -> Optional[PredictionResult]:
        """Get cached prediction."""
        key = self._obs_key(obs)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, obs: np.ndarray, result: PredictionResult) -> None:
        """Store prediction in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        key = self._obs_key(obs)
        self.cache[key] = result
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }


class InferenceManager:
    """
    High-level inference interface for RL agents.
    
    Features:
    - Single and batch predictions
    - Caching for repeated observations
    - Uncertainty estimation via ensemble/sampling
    - Deterministic and stochastic modes
    """
    
    def __init__(self, agent: Any, use_cache: bool = True, 
                 cache_size: int = 1000):
        """
        Initialize InferenceManager.
        
        Args:
            agent: Trained RL agent (e.g., DTAgent, LRAgent)
            use_cache: Whether to cache predictions
            cache_size: Max cached predictions
        """
        self.agent = agent
        self.use_cache = use_cache
        self.cache = PredictionCache(cache_size) if use_cache else None
        self.inference_count = 0
    
    def predict(self, observation: np.ndarray, 
                deterministic: bool = True,
                use_cache: bool = True) -> PredictionResult:
        """
        Predict action from single observation.
        
        Args:
            observation: Input observation
            deterministic: Deterministic or stochastic
            use_cache: Use cache if available
        
        Returns:
            PredictionResult
        """
        self.inference_count += 1
        
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(observation)
            if cached is not None:
                return cached
        
        # Predict
        action, _value = self.agent.predict(observation, deterministic=deterministic)
        
        result = PredictionResult(
            action=action,
            value=float(_value) if _value is not None else 0.0,
            confidence=1.0 if deterministic else 0.8
        )
        
        # Cache
        if use_cache and self.cache:
            self.cache.put(observation, result)
        
        return result
    
    def predict_batch(self, observations: List[np.ndarray],
                     deterministic: bool = True,
                     batch_size: int = 32) -> List[PredictionResult]:
        """
        Predict actions for batch of observations.
        
        Args:
            observations: List of observations
            deterministic: Deterministic inference
            batch_size: Batch size for processing
        
        Returns:
            List of PredictionResults
        """
        results = []
        
        for i in range(0, len(observations), batch_size):
            batch_obs = observations[i:i+batch_size]
            
            for obs in batch_obs:
                result = self.predict(obs, deterministic=deterministic, use_cache=True)
                results.append(result)
        
        return results
    
    def estimate_uncertainty(self, observation: np.ndarray,
                            n_samples: int = 10) -> Dict[str, float]:
        """
        Estimate uncertainty via Monte Carlo sampling.
        
        Args:
            observation: Observation
            n_samples: Number of stochastic samples
        
        Returns:
            Dict with uncertainty metrics
        """
        samples = []
        
        for _ in range(n_samples):
            result = self.predict(observation, deterministic=False, use_cache=False)
            samples.append(result.action)
        
        samples = np.array(samples)
        
        return {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
        }
    
    def predict_with_uncertainty(self, observation: np.ndarray,
                                estimate_uncertainty: bool = True,
                                n_samples: int = 10) -> Dict[str, Any]:
        """
        Predict with optional uncertainty estimation.
        
        Args:
            observation: Observation
            estimate_uncertainty: Whether to estimate uncertainty
            n_samples: Number of samples for uncertainty
        
        Returns:
            Dict with action, value, and uncertainty
        """
        result = self.predict(observation, deterministic=True)
        
        output = {
            "action": result.action,
            "value": result.value,
            "confidence": result.confidence,
        }
        
        if estimate_uncertainty:
            uncertainty = self.estimate_uncertainty(observation, n_samples)
            output["uncertainty"] = uncertainty
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = {
            "inference_count": self.inference_count,
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
