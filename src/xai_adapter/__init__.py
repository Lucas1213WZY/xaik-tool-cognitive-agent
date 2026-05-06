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
    create_custom_xai_method,
    create_xai_method_from_engine,
    generate_surrogate_xai_methods,
    get_coxam_xai_predictions,
)
from .attribution import (
    Attribution,
    CustomAttribution,
    DeepLift,
    DeepLiftMethod,
    GradientInput,
    GradientInputMethod,
    IntegratedGradients,
    IntegratedGradientsMethod,
    KernelShap,
    LeaveOneFeatureOut,
    LimeTabular,
    LimeTabularMethod,
    LOFOMethod,
    SHAPKernelMethod,
    SklearnFeatureImportance,
    SklearnGlobalFeatureImportanceMethod,
    make_attribution,
)
from .dataset import CSVDatasetAdapter, PrecomputedCSVXAIMethod
from .registry import XAIAdapterRegistry, create_xai_method, get_adapter_registry, register_xai_method
from .surrogate import (
    DecisionTreeSurrogateMethod,
    GeneratedSurrogateMethods,
    LogisticRegressionSurrogateMethod,
    SurrogateMethod,
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
    "register_xai_method",
    "create_custom_xai_method",
    "create_xai_method_from_engine",
    "create_coxam_xai_method",
    "generate_surrogate_xai_methods",
    "get_coxam_xai_predictions",
    "CSVDatasetAdapter",
    "PrecomputedCSVXAIMethod",
    "Attribution",
    "CustomAttribution",
    "make_attribution",
    "DeepLift",
    "GradientInput",
    "IntegratedGradients",
    "KernelShap",
    "LeaveOneFeatureOut",
    "LimeTabular",
    "SklearnFeatureImportance",
    "DecisionTreeSurrogateMethod",
    "LogisticRegressionSurrogateMethod",
    "SurrogateMethod",
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
