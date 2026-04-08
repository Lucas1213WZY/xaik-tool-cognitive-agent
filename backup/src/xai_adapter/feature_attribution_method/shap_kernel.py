"""SHAP KernelExplainer attribution method."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..base import ArrayLike, PreprocessFn, PostprocessFn, XAIAdapter, XAIAdapterResult, ensure_2d


class SHAPKernelMethod(XAIAdapter):
    """SHAP KernelExplainer method with a sklearn-like fit/explain API."""

    method_name = "shap_kernel"

    def __init__(
        self,
        *,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: Optional[ArrayLike] = None,
        n_background_samples: int = 45,
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        self.predict_fn = predict_fn
        self.n_background_samples = int(n_background_samples)
        self.background_data = None
        self.explainer = None
        self.shap = None
        if background_data is not None:
            self.fit(background_data)

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs):
        """Fit the SHAP background summarizer and explainer."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError("SHAP is required for SHAPKernelMethod. Install with: pip install shap") from exc

        self.shap = shap
        self.background_data = ensure_2d(X)
        background = shap.kmeans(self.background_data, min(self.n_background_samples, len(self.background_data)))
        self.explainer = shap.KernelExplainer(lambda x: self.predict_fn(self.preprocessing_fn(x)), background)
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        shap_values = self.explainer(raw_instances)

        if shap_values.values.ndim == 3:
            values = shap_values.values[:, :, self.target]
        else:
            values = shap_values.values

        if getattr(shap_values, "base_values", None) is not None and np.asarray(shap_values.base_values).ndim == 2:
            base_values = np.asarray(shap_values.base_values)[:, self.target]
        else:
            base_values = np.asarray(shap_values.base_values)

        values = self._postprocess_values(raw_instances, values)
        base_values = np.asarray(base_values, dtype=float).reshape(-1)
        if base_values.size == 1 and values.shape[0] > 1:
            base_values = np.full(values.shape[0], float(base_values[0]), dtype=float)
        return XAIAdapterResult(
            values=values,
            base_values=base_values,
            method=self.method_name,
            metadata={"n_background_samples": self.n_background_samples},
        )
