"""Registry for XAI adapters."""

from __future__ import annotations

from typing import Dict, Type

from .base import XAIAdapter


class XAIAdapterRegistry:
    """Small registry for adapter classes."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}

    def register(self, name: str, adapter_class: Type, *aliases: str) -> None:
        """Register an adapter class."""
        self._registry[name.lower()] = adapter_class
        for alias in aliases:
            self._registry[alias.lower()] = adapter_class

    def get_class(self, name: str) -> Type:
        """Return the adapter class for a registered name."""
        key = name.lower()
        if key not in self._registry:
            raise ValueError(f"Unknown XAI adapter '{name}'. Available: {self.list_available()}")
        return self._registry[key]

    def create(self, name: str, **kwargs):
        """Instantiate a registered adapter."""
        return self.get_class(name)(**kwargs)

    def list_available(self) -> list[str]:
        """List registered adapter names."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check whether a method name or alias is registered."""
        return name.lower() in self._registry


_GLOBAL_REGISTRY = None


def get_adapter_registry() -> XAIAdapterRegistry:
    """Return the global adapter registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        from .dataset import PrecomputedCSVXAIMethod
        from .feature_attribution_method import (
            DeepLiftMethod,
            GradientInputMethod,
            IntegratedGradientsMethod,
            LimeTabularMethod,
            LOFOMethod,
            SHAPKernelMethod,
            SklearnGlobalFeatureImportanceMethod,
        )
        from .surrogate import DecisionTreeSurrogateMethod, LogisticRegressionSurrogateMethod

        registry = XAIAdapterRegistry()
        registry.register("lofo", LOFOMethod, "leave_one_feature_out")
        registry.register("shap_kernel", SHAPKernelMethod, "shap")
        registry.register("lime_tabular", LimeTabularMethod, "lime")
        registry.register("gradient_input", GradientInputMethod, "gradient_x_input")
        registry.register("deeplift", DeepLiftMethod, "deep_lift")
        registry.register("integrated_gradients", IntegratedGradientsMethod, "ig")
        registry.register("sklearn_global", SklearnGlobalFeatureImportanceMethod, "global_feature_importance")
        registry.register("precomputed_csv", PrecomputedCSVXAIMethod, "csv", "csv_dataset", "dataset_csv")
        registry.register("decision_tree", DecisionTreeSurrogateMethod, "dt", "rules")
        registry.register("logistic_regression", LogisticRegressionSurrogateMethod, "lr", "weights")
        _GLOBAL_REGISTRY = registry
    return _GLOBAL_REGISTRY


def create_xai_method(name: str, **kwargs):
    """Create an XAI method adapter from the global registry."""
    return get_adapter_registry().create(name, **kwargs)
