"""
XAI adapter API layer.

This layer wraps external attribution libraries such as Captum, LIME, SHAP,
and sklearn-style feature importance behind a consistent adapter interface.
"""

from .base import (
    XAIAdapter,
    XAIAdapterResult,
    baseline_from_data,
    ensure_2d,
    identity_postprocess,
    identity_preprocess,
    select_target,
)
from .api import (
    create_coxam_xai_method,
    create_xai_method_from_engine,
    generate_surrogate_xai_methods,
    get_coxam_xai_predictions,
)
from .dataset import CSVDatasetAdapter, PrecomputedCSVXAIMethod
from .feature_attribution_method import (
    DeepLiftMethod,
    GradientInputMethod,
    IntegratedGradientsMethod,
    LimeTabularMethod,
    LOFOMethod,
    SHAPKernelMethod,
    SklearnGlobalFeatureImportanceMethod,
)
from .registry import XAIAdapterRegistry, create_xai_method, get_adapter_registry
from .surrogate import (
    DecisionTreeSurrogateMethod,
    GeneratedSurrogateMethods,
    LogisticRegressionSurrogateMethod,
    generate_decision_tree_table,
    generate_logistic_regression_table,
    generate_surrogate_tables,
)

__all__ = [
    "XAIAdapter",
    "XAIAdapterResult",
    "baseline_from_data",
    "ensure_2d",
    "identity_postprocess",
    "identity_preprocess",
    "select_target",
    "create_xai_method",
    "create_xai_method_from_engine",
    "create_coxam_xai_method",
    "generate_surrogate_xai_methods",
    "get_coxam_xai_predictions",
    "CSVDatasetAdapter",
    "PrecomputedCSVXAIMethod",
    "DecisionTreeSurrogateMethod",
    "LogisticRegressionSurrogateMethod",
    "GeneratedSurrogateMethods",
    "generate_decision_tree_table",
    "generate_logistic_regression_table",
    "generate_surrogate_tables",
    "LOFOMethod",
    "SHAPKernelMethod",
    "LimeTabularMethod",
    "GradientInputMethod",
    "DeepLiftMethod",
    "IntegratedGradientsMethod",
    "SklearnGlobalFeatureImportanceMethod",
    "XAIAdapterRegistry",
    "get_adapter_registry",
]
