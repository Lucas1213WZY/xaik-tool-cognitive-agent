"""Feature attribution methods and plugin adapters."""

from .base import Attribution, CustomAttribution, GlobalImportance, LocalAttribution, make_attribution
from .captum import CaptumAttribution, DeepLift, DeepLiftMethod, IntegratedGradients, IntegratedGradientsMethod
from .global_importance import SklearnFeatureImportance, SklearnGlobalFeatureImportanceMethod
from .gradient import GradientInput, GradientInputMethod
from .perturbation import KernelShap, LeaveOneFeatureOut, LOFOMethod, SHAPKernel, SHAPKernelMethod
from .tabular import LimeTabular, LimeTabularMethod

__all__ = [
    "Attribution",
    "LocalAttribution",
    "GlobalImportance",
    "CustomAttribution",
    "make_attribution",
    "CaptumAttribution",
    "DeepLift",
    "DeepLiftMethod",
    "IntegratedGradients",
    "IntegratedGradientsMethod",
    "GradientInput",
    "GradientInputMethod",
    "KernelShap",
    "SHAPKernel",
    "SHAPKernelMethod",
    "LeaveOneFeatureOut",
    "LOFOMethod",
    "LimeTabular",
    "LimeTabularMethod",
    "SklearnFeatureImportance",
    "SklearnGlobalFeatureImportanceMethod",
]
