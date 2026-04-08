"""
Parameter Sampler - Generate synthetic participant parameters.

Samples cognitive parameters from estimated distributions for creating new
synthetic participants with realistic parameter values.
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats as scipy_stats
import logging

logger = logging.getLogger(__name__)


class ParameterSampler:
    """
    Sample cognitive parameters for synthetic participants from distributions.
    
    Usage:
        sampler = ParameterSampler()
        sampler.load_distributions("distributions.json")
        params = sampler.sample(dataset="adult", strategy="sensitive_features", 
                               xai_type="Importance", tested_with_xai="w/ XAI")
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.distributions: Dict[str, Dict[str, Any]] = {}
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def load_distributions(self, json_path: str) -> None:
        """
        Load parameter distributions from JSON file.
        
        Args:
            json_path: Path to distributions JSON (output from ParameterEstimator)
        """
        with open(json_path, 'r') as f:
            self.distributions = json.load(f)
        logger.info(f"Loaded {len(self.distributions)} distributions from {json_path}")
    
    def sample(self, 
               dataset: str, 
               strategy: str,
               xai_type: str = "Importance",
               tested_with_xai: str = "w/ XAI",
               method: str = "normal") -> Dict[str, float]:
        """
        Sample parameter values for a synthetic participant.
        
        Args:
            dataset: Dataset name (adult, wine_quality, forest_cover)
            strategy: Strategy name (sensitive_features, salient_features, etc.)
            xai_type: XAI type (Importance, Attribution, None)
            tested_with_xai: "w/ XAI" or "w/o XAI"
            method: Sampling method ("normal" uses Gaussian, "uniform" uses bounds)
            
        Returns:
            Dict mapping parameter names to sampled values
            
        Raises:
            KeyError: If distribution not found
        """
        # Build key in canonical format used by distribution files.
        xai_key = (xai_type or "none").lower()
        tested_with_xai_key = self._normalize_tested_with_xai(tested_with_xai)
        key = f"{dataset}/{strategy}/{xai_key}/{tested_with_xai_key}"
        
        if key not in self.distributions:
            key = self._resolve_distribution_key(
                dataset=dataset,
                strategy=strategy,
                xai_key=xai_key,
                tested_with_xai_key=tested_with_xai_key,
            )
            if key is None:
                raise KeyError(f"Distribution not found: {dataset}/{strategy}/{xai_key}/{tested_with_xai_key}")
        
        dist = self.distributions[key]["parameters"]
        sampled = {}
        
        for param_name, param_stats in dist.items():
            if method == "normal":
                # Sample from normal distribution
                value = np.random.normal(
                    loc=param_stats["mean"],
                    scale=param_stats["std"]
                )
            elif method == "uniform":
                # Sample uniformly from observed range
                value = np.random.uniform(
                    low=param_stats["min"],
                    high=param_stats["max"]
                )
            elif method == "truncated_normal":
                # Sample from truncated normal (within observed range)
                a = (param_stats["min"] - param_stats["mean"]) / param_stats["std"]
                b = (param_stats["max"] - param_stats["mean"]) / param_stats["std"]
                value = scipy_stats.truncnorm.rvs(
                    a=a, b=b,
                    loc=param_stats["mean"],
                    scale=param_stats["std"]
                )
            else:
                raise ValueError(f"Unknown sampling method: {method}")
            
            sampled[param_name] = float(value)
        
        return sampled

    def _resolve_distribution_key(self,
                                  dataset: str,
                                  strategy: str,
                                  xai_key: str,
                                  tested_with_xai_key: str) -> Optional[str]:
        """Find a compatible distribution key across naming conventions."""
        target_strategy = self._normalize_strategy_name(strategy)

        for key, payload in self.distributions.items():
            parts = key.split('/')
            if len(parts) < 4:
                continue

            ds_k, strategy_k, xai_k, with_k = parts[0], parts[1], parts[2], parts[3]
            if ds_k != dataset:
                continue
            if (xai_k or '').lower() != xai_key:
                continue
            if (with_k or '').lower() != tested_with_xai_key:
                continue

            strategy_payload = payload.get('strategy', strategy_k)
            if self._normalize_strategy_name(strategy_payload) == target_strategy:
                return key

        return None

    @staticmethod
    def _normalize_strategy_name(strategy: str) -> str:
        """Normalize strategy labels to canonical internal names."""
        strategy_lower = str(strategy).lower()

        mapping = {
            "sensitive": "sensitive_features",
            "sensitive-features": "sensitive_features",
            "sensitive_features": "sensitive_features",
            "salient": "salient_features",
            "salient-features": "salient_features",
            "salient_features": "salient_features",
            "importance": "importance_categorization",
            "importance categorization": "importance_categorization",
            "importance_categorization": "importance_categorization",
            "attribution": "attribution_sum",
            "attribution sum": "attribution_sum",
            "attribution_sum": "attribution_sum",
        }

        for key, normalized in mapping.items():
            if key in strategy_lower:
                return normalized

        return strategy_lower

    @staticmethod
    def _normalize_tested_with_xai(tested_with_xai: Any) -> str:
        """Normalize labels like 'w/ XAI' to canonical key suffixes."""
        if isinstance(tested_with_xai, bool):
            return "with_xai" if tested_with_xai else "without_xai"

        label = str(tested_with_xai).strip().lower()
        if label in {"w/ xai", "with xai", "with_xai", "with-xai", "true", "1"}:
            return "with_xai"
        if label in {"w/o xai", "without xai", "without_xai", "without-xai", "false", "0"}:
            return "without_xai"

        # Fallback keeps behavior predictable for unknown labels.
        return label.replace(" ", "_")
    
    def sample_batch(self,
                    dataset: str,
                    strategy: str,
                    xai_type: str = "Importance",
                    tested_with_xai: str = "w/ XAI",
                    n_samples: int = 10,
                    method: str = "normal") -> List[Dict[str, float]]:
        """
        Sample multiple parameter sets.
        
        Args:
            dataset: Dataset name
            strategy: Strategy name
            xai_type: XAI type
            tested_with_xai: "w/ XAI" or "w/o XAI"
            n_samples: Number of parameter sets to generate
            method: Sampling method
            
        Returns:
            List of parameter dicts
        """
        samples = []
        for _ in range(n_samples):
            sample = self.sample(
                dataset=dataset,
                strategy=strategy,
                xai_type=xai_type,
                tested_with_xai=tested_with_xai,
                method=method
            )
            samples.append(sample)
        return samples
    
    def get_available_distributions(self) -> Dict[str, List[str]]:
        """
        Get available distribution keys grouped by dataset.
        
        Returns:
            Dict mapping dataset -> list of (strategy, xai_type, with_xai) tuples
        """
        available = {}
        for key_str in self.distributions.keys():
            parts = key_str.split('/')
            if len(parts) >= 1:
                dataset = parts[0]
                if dataset not in available:
                    available[dataset] = []
                available[dataset].append(key_str)
        return available
    
    def list_strategies_for_dataset(self, dataset: str) -> List[Tuple[str, str, str]]:
        """
        List all (strategy, xai_type, tested_with_xai) combinations available for a dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of (strategy, xai_type, tested_with_xai) tuples
        """
        combinations = []
        for key_str, dist in self.distributions.items():
            if dist["dataset"] == dataset:
                combinations.append((
                    dist["strategy"],
                    dist["xai_type"],
                    dist["tested_with_xai"]
                ))
        return combinations
