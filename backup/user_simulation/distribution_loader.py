"""
Distribution Loader - Load pre-computed parameter distributions.

Loads parameter distributions from JSON files (including the pre-computed
distribution file format) for use in generating synthetic participants.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DistributionLoader:
    """
    Load parameter distributions from JSON files.
    
    Supports both formats:
    1. ParameterEstimator output (with percentiles)
    2. Pre-computed distribution snapshots
    """
    
    def __init__(self):
        self.distributions: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
    
    def load_from_json(self, json_path: str) -> None:
        """
        Load distributions from JSON file.
        
        Args:
            json_path: Path to JSON file with distributions
        """
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Distribution file not found: {json_path}")
        
        with open(json_file, 'r') as f:
            self.distributions = json.load(f)
        
        logger.info(f"Loaded {len(self.distributions)} distributions from {json_path}")
    
    def get_distribution(self, 
                        dataset: str,
                        strategy: str,
                        xai_type: str = "importance",
                        tested_with_xai: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get a specific parameter distribution.
        
        Args:
            dataset: Dataset name (adult, wine_quality, forest_cover)
            strategy: Strategy name
            xai_type: XAI type (importance, attribution, none)
            tested_with_xai: Whether with or without XAI
            
        Returns:
            Distribution dict or None if not found
        """
        # Normalize strategy name
        strategy_name = self._normalize_strategy_name(strategy)
        xai_type_norm = xai_type.lower() if xai_type else "none"
        with_xai_key = "with_xai" if tested_with_xai else "without_xai"
        
        # Try to find in formats
        key_variants = [
            f"{dataset}/{strategy_name}/{xai_type_norm}/{with_xai_key}",
            f"{dataset}/{strategy}/{xai_type}/{with_xai_key}",
        ]
        
        for key in key_variants:
            if key in self.distributions:
                return self.distributions[key]
        
        return None
    
    def list_datasets(self) -> List[str]:
        """List all datasets in distributions."""
        datasets = set()
        for key in self.distributions.keys():
            dataset = key.split('/')[0]
            datasets.add(dataset)
        return sorted(list(datasets))
    
    def list_strategies_for_dataset(self, dataset: str) -> List[Tuple[str, str, bool]]:
        """
        List all (strategy, xai_type, tested_with_xai) combinations for a dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of (strategy, xai_type, tested_with_xai) tuples
        """
        strategies = set()
        for key, dist in self.distributions.items():
            parts = key.split('/')
            if len(parts) >= 2 and parts[0] == dataset:
                strategy = dist.get("strategy", parts[1])
                xai_type = dist.get("xai_type", parts[2] if len(parts) > 2 else "unknown")
                tested_with_xai = dist.get("tested_with_xai", "").startswith("w")
                strategies.add((strategy, xai_type, tested_with_xai))
        
        return sorted(list(strategies))
    
    def get_default_params(self,
                          dataset: str,
                          strategy: str,
                          xai_type: str = "importance",
                          tested_with_xai: bool = True) -> Dict[str, float]:
        """
        Get default parameter values (means from distribution).
        
        Args:
            dataset: Dataset name
            strategy: Strategy name
            xai_type: XAI type
            tested_with_xai: Whether with XAI
            
        Returns:
            Dict mapping parameter names to mean values
        """
        dist = self.get_distribution(dataset, strategy, xai_type, tested_with_xai)
        if dist is None or "parameters" not in dist:
            logger.warning(f"No distribution found for {dataset}/{strategy}")
            return {}
        
        defaults = {}
        for param_name, param_stats in dist["parameters"].items():
            if "mean" in param_stats:
                defaults[param_name] = float(param_stats["mean"])
        
        return defaults
    
    def get_param_range(self,
                       dataset: str,
                       strategy: str,
                       param_name: str,
                       xai_type: str = "importance",
                       tested_with_xai: bool = True) -> Optional[Tuple[float, float]]:
        """
        Get (min, max) range for a parameter.
        
        Args:
            dataset: Dataset name
            strategy: Strategy name
            param_name: Parameter name
            xai_type: XAI type
            tested_with_xai: Whether with XAI
            
        Returns:
            (min, max) tuple or None if not found
        """
        dist = self.get_distribution(dataset, strategy, xai_type, tested_with_xai)
        if dist is None or "parameters" not in dist:
            return None
        
        params = dist["parameters"]
        if param_name in params:
            stats = params[param_name]
            return (float(stats["min"]), float(stats["max"]))
        
        return None
    
    def get_param_stats(self,
                       dataset: str,
                       strategy: str,
                       param_name: str,
                       xai_type: str = "importance",
                       tested_with_xai: bool = True) -> Optional[Dict[str, float]]:
        """
        Get all statistics for a parameter.
        
        Args:
            dataset: Dataset name
            strategy: Strategy name
            param_name: Parameter name
            xai_type: XAI type
            tested_with_xai: Whether with XAI
            
        Returns:
            Stats dict (mean, std, min, max, etc.) or None
        """
        dist = self.get_distribution(dataset, strategy, xai_type, tested_with_xai)
        if dist is None or "parameters" not in dist:
            return None
        
        params = dist["parameters"]
        if param_name in params:
            return params[param_name]
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all distributions.
        
        Returns:
            Dict with datasets, strategies, and counts
        """
        summary = {
            "total_distributions": len(self.distributions),
            "datasets": {},
        }
        
        for key, dist in self.distributions.items():
            dataset = dist.get("dataset", key.split('/')[0])
            
            if dataset not in summary["datasets"]:
                summary["datasets"][dataset] = {
                    "strategies": {},
                    "total": 0
                }
            
            strategy = dist.get("strategy", "unknown")
            xai_type = dist.get("xai_type", "unknown")
            tested_with_xai = dist.get("tested_with_xai", "unknown")
            
            if strategy not in summary["datasets"][dataset]["strategies"]:
                summary["datasets"][dataset]["strategies"][strategy] = {
                    "xai_types": {}
                }
            
            if xai_type not in summary["datasets"][dataset]["strategies"][strategy]["xai_types"]:
                summary["datasets"][dataset]["strategies"][strategy]["xai_types"][xai_type] = []
            
            summary["datasets"][dataset]["strategies"][strategy]["xai_types"][xai_type].append({
                "tested_with_xai": tested_with_xai,
                "n_samples": dist.get("n_samples", 0),
            })
            
            summary["datasets"][dataset]["total"] += 1
        
        return summary
    
    @staticmethod
    def _normalize_strategy_name(strategy: str) -> str:
        """Normalize strategy name to internal format."""
        strategy_lower = strategy.lower()
        
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
        
        return strategy.lower()
