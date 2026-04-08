"""Explainer implementations and registry for the unified data loader system."""

from .registry import ExplainerRegistry
from .attribution_explainer import AttributionExplainer
from .decision_tree import DecisionTreeExplainer
from .logistic_regression import LogisticRegressionExplainer

# Model-agnostic explainers
try:
    from .shap_explainer import SHAPExplainer
except ImportError:
    SHAPExplainer = None

try:
    from .lime_explainer import LIMEExplainer
except ImportError:
    LIMEExplainer = None

from .lofo_explainer import LOFOExplainer

# Gradient-based explainers
try:
    from .gradient_based_explainers import (
        GradientInputExplainer,
        DeepLIFTExplainer,
        IntegratedGradientsExplainer
    )
except ImportError:
    GradientInputExplainer = None
    DeepLIFTExplainer = None
    IntegratedGradientsExplainer = None

__all__ = [
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
]

# Global registry instance for convenience
_global_registry = None

def get_registry() -> ExplainerRegistry:
    """Get the global explainer registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ExplainerRegistry()
        
        # Register CoXAM-based explainers (always available)
        _global_registry.register('decision_tree', DecisionTreeExplainer)
        _global_registry.register('dt', DecisionTreeExplainer)
        _global_registry.register('logistic_regression', LogisticRegressionExplainer)
        _global_registry.register('lr', LogisticRegressionExplainer)
        
        # Register model-agnostic explainers
        _global_registry.register('lofo', LOFOExplainer)
        _global_registry.register('leave_one_feature_out', LOFOExplainer)
        
        if SHAPExplainer is not None:
            _global_registry.register('shap', SHAPExplainer)
            _global_registry.register('shap_kernel', SHAPExplainer)
        
        if LIMEExplainer is not None:
            _global_registry.register('lime', LIMEExplainer)
        
        # Register gradient-based explainers
        if GradientInputExplainer is not None:
            _global_registry.register('gradient_input', GradientInputExplainer)
            _global_registry.register('gradient_x_input', GradientInputExplainer)
        
        if DeepLIFTExplainer is not None:
            _global_registry.register('deeplift', DeepLIFTExplainer)
        
        if IntegratedGradientsExplainer is not None:
            _global_registry.register('integrated_gradients', IntegratedGradientsExplainer)
            _global_registry.register('ig', IntegratedGradientsExplainer)
    
    return _global_registry
