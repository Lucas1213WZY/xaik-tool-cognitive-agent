"""
XAI method API layer.

Import explainer interfaces, registries, and implementations from here:

    from src.xai_method import ExplainerRegistry, DecisionTreeExplainer
"""

from .base import BaseExplainer
from .api import (
    create_explainer,
    create_coxam_explainer,
    get_coxam_xai_predictions,
)
from .explainers import (
    ExplainerRegistry,
    AttributionExplainer,
    DecisionTreeExplainer,
    LogisticRegressionExplainer,
    SHAPExplainer,
    LIMEExplainer,
    LOFOExplainer,
    GradientInputExplainer,
    DeepLIFTExplainer,
    IntegratedGradientsExplainer,
    get_registry,
)

__all__ = [
    "BaseExplainer",
    "create_explainer",
    "create_coxam_explainer",
    "get_coxam_xai_predictions",
    "ExplainerRegistry",
    "AttributionExplainer",
    "DecisionTreeExplainer",
    "LogisticRegressionExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "LOFOExplainer",
    "GradientInputExplainer",
    "DeepLIFTExplainer",
    "IntegratedGradientsExplainer",
    "get_registry",
]
