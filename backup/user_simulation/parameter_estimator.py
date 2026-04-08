"""
Parameter Distribution Estimator

Extracts cognitive parameter distributions from fitted data (CSV) and exports summaries
as JSON for tracking and reproducibility.

Handles:
- Multiple datasets (adult, wine_quality, forest_cover)
- Multiple strategies (SensitiveFeatures, SalientFeatures, ImportanceCategorization, AttributionSum)
- Multiple XAI types (Importance, Attribution, None)
- With/Without XAI conditions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


# Map CoAX strategy names from CSV to internal identifiers
STRATEGY_NAME_MAP = {
    "Sensitive-features categorization": "sensitive_features",
    "Salient-features categorization": "salient_features",
    "Importance categorization": "importance_categorization",
    "Attribution sum": "attribution_sum",
}

# Strategies by XAI type
STRATEGIES_BY_XAI = {
    "Importance": ["sensitive_features", "salient_features", "importance_categorization"],
    "Attribution": ["attribution_sum"],
}

# Importance-based parameter set
IMPORTANCE_PARAMS = {"sensitivity", "k", "retrieval_threshold"}

# Attribution-based parameter set
ATTRIBUTION_PARAMS = {"scaling_factor"}


@dataclass
class ParameterStats:
    """Statistics for a single parameter."""
    param_name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistributionKey:
    """Unique identifier for a parameter distribution context."""
    appId: str
    strategy: str
    xai_type: str  # "Importance", "Attribution", or "None"
    tested_with_xai: str  # "w/ XAI" or "w/o XAI"
    
    def __hash__(self):
        return hash((self.appId, self.strategy, self.xai_type, self.tested_with_xai))
    
    def __eq__(self, other):
        if not isinstance(other, DistributionKey):
            return False
        return (self.appId == other.appId and 
                self.strategy == other.strategy and 
                self.xai_type == other.xai_type and 
                self.tested_with_xai == other.tested_with_xai)
    
    def to_key_string(self) -> str:
        """Generate a unique string key for this distribution."""
        return f"{self.appId}/{self.strategy}/{self.xai_type}/{self.tested_with_xai.replace(' ', '_').lower()}"


class ParameterEstimator:
    """
    Extracts and summarizes cognitive parameter distributions from fitted CoAX data.
    
    Usage:
        estimator = ParameterEstimator()
        estimator.load_fitted_data("fitted_data.csv")
        summary = estimator.estimate_distributions()
        estimator.save_distributions("distributions.json")
    """
    
    def __init__(self):
        self.fitted_data: Optional[pd.DataFrame] = None
        self.distributions: Dict[DistributionKey, Dict[str, ParameterStats]] = {}
        self.field_mapping: Dict[str, str] = self._build_field_mapping()
    
    def _build_field_mapping(self) -> Dict[str, str]:
        """Build mapping from CSV column names to parameter names."""
        return {
            "sensitivity": "sensitivity",
            "k": "k",
            "retrieval_threshold": "retrieval_threshold",
            "retrieval_th": "retrieval_threshold",
            "scaling_factor": "scaling_factor",
            "scaling_fa": "scaling_factor",
        }
    
    def load_fitted_data(self, csv_path: str) -> None:
        """
        Load fitted parameter data from CSV.
        
        Expected columns:
        - Strategy: Strategy name (e.g., "Sensitive-features categorization")
        - Participant Id: Unique participant ID
        - Tested w/ XAI: "w/ XAI" or "w/o XAI"
        - XAIType: "Importance", "Attribution", or "None"
        - appId: Dataset name (adult, wine_quality, forest_cover)
        - sensitivity, k, retrieval_threshold: For importance-based strategies
        - scaling_factor: For attribution-based strategies
        """
        self.fitted_data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.fitted_data)} parameter sets from {csv_path}")
    
    def estimate_distributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Estimate parameter distributions grouped by (dataset, strategy, XAI type, with/without XAI).
        
        Returns:
            Dict mapping distribution keys to parameter statistics.
            Format:
            {
                "adult/sensitive_features/importance/with_xai": {
                    "n_samples": 30,
                    "parameters": {
                        "sensitivity": {
                            "mean": 76.5,
                            "std": 12.3,
                            ...
                        },
                        ...
                    }
                },
                ...
            }
        """
        if self.fitted_data is None:
            raise ValueError("No fitted data loaded. Call load_fitted_data() first.")
        
        df = self.fitted_data
        distribution_map: Dict[str, Dict[str, Any]] = {}
        
        # Group by (appId, Strategy, XAIType, Tested w/ XAI)
        for (app_id, strategy_csv, xai_type, with_xai), group in df.groupby(
            ['appId', 'Strategy', 'XAIType', 'Tested w/ XAI'],
            dropna=False
        ):
            if pd.isna(strategy_csv):
                continue
            
            # Map strategy name
            strategy_key = STRATEGY_NAME_MAP.get(strategy_csv, strategy_csv.lower())
            xai_label = str(xai_type).strip() if not pd.isna(xai_type) else "None"
            with_xai_label = str(with_xai).strip() if not pd.isna(with_xai) else "None"
            
            dist_key = DistributionKey(
                appId=str(app_id),
                strategy=strategy_key,
                xai_type=xai_label,
                tested_with_xai=with_xai_label
            )
            
            # Determine which parameters apply
            param_names = self._get_param_names(xai_label)
            
            # Extract and compute statistics for each parameter
            param_stats = {}
            for param in param_names:
                if param not in group.columns:
                    # Try alternative column names
                    alt_cols = [col for col in group.columns if param.lower() in col.lower()]
                    if not alt_cols:
                        continue
                    param_col = alt_cols[0]
                else:
                    param_col = param
                
                # Extract valid values (drop NaN and inf)
                values = pd.to_numeric(group[param_col], errors='coerce')
                values = values[~np.isnan(values) & ~np.isinf(values)]
                
                if len(values) > 0:
                    stats = ParameterStats(
                        param_name=param,
                        count=len(values),
                        mean=float(values.mean()),
                        std=float(values.std()),
                        min=float(values.min()),
                        max=float(values.max()),
                        median=float(values.median()),
                        q25=float(values.quantile(0.25)),
                        q75=float(values.quantile(0.75)),
                    )
                    param_stats[param] = stats.to_dict()
            
            # Store distribution
            key_str = dist_key.to_key_string()
            distribution_map[key_str] = {
                "dataset": str(app_id),
                "strategy": strategy_key,
                "xai_type": xai_label,
                "tested_with_xai": with_xai_label,
                "n_samples": len(group),
                "parameters": param_stats,
            }
        
        self.distributions = distribution_map
        logger.info(f"Estimated {len(distribution_map)} parameter distributions")
        return distribution_map
    
    def _get_param_names(self, xai_type: str) -> List[str]:
        """Get applicable parameter names based on XAI type."""
        if xai_type.lower() == "importance":
            return list(IMPORTANCE_PARAMS)
        elif xai_type.lower() == "attribution":
            return list(ATTRIBUTION_PARAMS)
        else:
            return []  # No XAI or unknown type
    
    def save_distributions(self, output_path: str) -> None:
        """
        Save estimated distributions to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if not self.distributions:
            raise ValueError("No distributions estimated. Call estimate_distributions() first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.distributions, f, indent=2)
        
        logger.info(f"Saved {len(self.distributions)} distributions to {output_path}")
    
    def get_distribution_summary(self) -> Dict[str, Any]:
        """
        Get high-level summary of distributions by dataset and strategy.
        
        Returns:
            Nested dict: dataset -> strategy -> xai_type -> counts
        """
        summary = {}
        for key, dist in self.distributions.items():
            dataset = dist["dataset"]
            strategy = dist["strategy"]
            xai_type = dist["xai_type"]
            
            if dataset not in summary:
                summary[dataset] = {}
            if strategy not in summary[dataset]:
                summary[dataset][strategy] = {}
            if xai_type not in summary[dataset][strategy]:
                summary[dataset][strategy][xai_type] = []
            
            summary[dataset][strategy][xai_type].append({
                "tested_with_xai": dist["tested_with_xai"],
                "n_samples": dist["n_samples"],
            })
        
        return summary
    
    def export_summary_stats(self, output_path: str) -> None:
        """
        Export high-level summary statistics as JSON.
        
        Args:
            output_path: Path to output summary JSON
        """
        summary = self.get_distribution_summary()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported distribution summary to {output_path}")
