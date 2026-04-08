"""Shared Captum attribution method helpers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..base import (
    ArrayLike,
    PreprocessFn,
    PostprocessFn,
    XAIAdapter,
    XAIAdapterResult,
    baseline_from_data,
    ensure_2d,
    select_target,
)


class _CaptumMethod(XAIAdapter):
    """Base class for Captum attribution methods."""

    captum_attr_cls = None
    method_name = "captum"

    def __init__(
        self,
        *,
        model,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: Optional[ArrayLike] = None,
        baseline: str = "mean",
        device: str = "cpu",
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
        **attribute_kwargs,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch and Captum are required. Install with: pip install torch captum") from exc
        if self.captum_attr_cls is None:
            raise TypeError("captum_attr_cls must be defined by subclasses")

        self.torch = torch
        self.model = model
        self.predict_fn = predict_fn
        self.baseline = baseline
        self.device = device
        self.attribute_kwargs = dict(attribute_kwargs)
        self.model.eval()
        self.model.to(device)
        self.attr = self.captum_attr_cls(self.model)
        self.baseline_tensor = None
        self.baseline_value = 0.0
        self.is_fitted = True
        if background_data is not None:
            self.fit(background_data)

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        """Fit the Captum baseline from background data."""
        if X is not None:
            baseline_vec = baseline_from_data(self.preprocessing_fn(X), self.baseline)
            self.baseline_tensor = self.torch.tensor(
                baseline_vec,
                dtype=self.torch.float32,
                device=self.device,
            ).reshape(1, -1)
            self.baseline_value = float(select_target(self.predict_fn(baseline_vec.reshape(1, -1)), self.target)[0])
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x_np = ensure_2d(self.preprocessing_fn(raw_instances))
        x = self.torch.tensor(x_np, dtype=self.torch.float32, device=self.device)

        baselines = (
            self.baseline_tensor.repeat(x.shape[0], 1)
            if self.baseline_tensor is not None
            else self.torch.zeros_like(x)
        )
        attributions = self.attr.attribute(
            x,
            baselines=baselines,
            target=self.target,
            **self.attribute_kwargs,
        )
        if isinstance(attributions, tuple):
            attributions = attributions[0]

        values = self._postprocess_values(raw_instances, attributions.detach().cpu().numpy())
        return XAIAdapterResult(
            values=values,
            base_values=np.full(values.shape[0], self.baseline_value, dtype=float),
            method=self.method_name,
            metadata={"baseline": baselines.detach().cpu().numpy()},
        )
