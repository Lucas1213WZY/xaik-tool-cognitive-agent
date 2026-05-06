"""Gradient-based attribution methods."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..base import (
    ArrayLike,
    PostprocessFn,
    PreprocessFn,
    XAIAdapterResult,
    baseline_from_data,
    ensure_2d,
    select_target,
)
from .base import LocalAttribution


class GradientInput(LocalAttribution):
    """Gradient times input method for differentiable torch models."""

    method_name = "gradient_input"

    def __init__(
        self,
        *,
        model,
        forward_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        background_data: Optional[ArrayLike] = None,
        device: str = "cpu",
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        super().__init__(
            target=target,
            preprocessing_fn=preprocessing_fn,
            postprocessing_fn=postprocessing_fn,
        )
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for GradientInput. Install with: pip install torch") from exc

        self.torch = torch
        self.model = model
        self.forward_fn = forward_fn or model
        self.predict_fn = predict_fn
        self.device = device
        self.baseline_value = 0.0
        self.model.eval()
        self.model.to(device)
        self.is_fitted = True
        if background_data is not None:
            self.fit(background_data)

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        """Optionally fit the baseline value from background data."""
        if X is not None and self.predict_fn is not None:
            baseline_vec = baseline_from_data(self.preprocessing_fn(X), "mean")
            self.baseline_value = float(select_target(self.predict_fn(baseline_vec.reshape(1, -1)), self.target)[0])
        self.is_fitted = True
        return self

    def explain(self, instances: ArrayLike) -> XAIAdapterResult:
        self._require_fitted()
        raw_instances = ensure_2d(instances)
        x_np = ensure_2d(self.preprocessing_fn(raw_instances))
        x = self.torch.tensor(x_np, dtype=self.torch.float32, requires_grad=True, device=self.device)

        outputs = self.forward_fn(x)
        selected = outputs[:, self.target] if outputs.ndim == 2 else outputs

        self.model.zero_grad(set_to_none=True)
        grads = self.torch.autograd.grad(
            outputs=selected,
            inputs=x,
            grad_outputs=self.torch.ones_like(selected),
            retain_graph=False,
            create_graph=False,
        )[0]
        attributions = (grads * x).detach().cpu().numpy()
        values = self._postprocess_values(raw_instances, attributions)
        return XAIAdapterResult(
            values=values,
            base_values=np.full(values.shape[0], self.baseline_value, dtype=float),
            method=self.method_name,
        )


GradientInputMethod = GradientInput


__all__ = [
    "GradientInput",
    "GradientInputMethod",
]
