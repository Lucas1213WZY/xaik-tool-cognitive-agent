"""
Unified data loader system for CoAX, CoXAM, and custom data sources.

A comprehensive, extensible API layer that unifies data loading, normalization, 
filtering, and explainer management across different frameworks.

Key Components:
- UnifiedDataLoader: Main API for loading and accessing data
- Data Sources: Adapters for CoAX, CoXAM, and custom sources
- Explainers: Plugin system for model explainers (DT, LR, SHAP, LIME, etc.)
- Normalizers: Min-Max, Z-Score, and custom feature normalization
- Filters: Composable filter builder for data queries

Example Usage:
    from src.data_loaders import UnifiedDataLoader
    
    # Load CoAX data
    loader = UnifiedDataLoader.from_coax(
        feature_file="values.csv",
        metadata_file="metadata.csv"
    )
    
    # Apply filters
    filter_builder = loader.filter().by_app("wine_quality")
    loader.apply_filter(filter_builder)
    
    # Get data
    features, predictions = loader.get_instances([1, 2, 3])
    
    # Use explainers
    registry = loader.get_explainer_registry()
    dt_exp = registry.create('decision_tree', 
        explanation_df=dt_df, metadata_df=metadata_df,
        app_id="wine_quality", model_name="mlp")
"""

__version__ = "0.1.0"

# Core API
from .unified_loader import UnifiedDataLoader

# Data sources
from .sources import CoAXDataSource, CoXAMDataSource

# Explainers
from .explainers import (
    ExplainerRegistry,
    DecisionTreeExplainer,
    LogisticRegressionExplainer,
    get_registry
)

# Normalizers
from .normalizers import MinMaxNormalizer, ZScoreNormalizer

# Filters
from .filters import FilterBuilder

# Base classes
from .base import BaseDataSource, BaseExplainer, BaseNormalizer

__all__ = [
    # Core
    "UnifiedDataLoader",
    
    # Data sources
    "CoAXDataSource",
    "CoXAMDataSource",
    
    # Explainers
    "ExplainerRegistry",
    "DecisionTreeExplainer", 
    "LogisticRegressionExplainer",
    "get_registry",
    
    # Normalizers
    "MinMaxNormalizer",
    "ZScoreNormalizer",
    
    # Filters
    "FilterBuilder",
    
    # Base classes
    "BaseDataSource",
    "BaseExplainer",
    "BaseNormalizer",
]
