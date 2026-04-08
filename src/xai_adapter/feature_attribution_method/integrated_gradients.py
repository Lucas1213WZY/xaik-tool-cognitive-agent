"""Captum Integrated Gradients attribution method."""

from __future__ import annotations

from ._captum import _CaptumMethod


class IntegratedGradientsMethod(_CaptumMethod):
    """Captum Integrated Gradients method."""

    method_name = "integrated_gradients"

    def __init__(self, *, n_steps: int = 50, **kwargs):
        try:
            from captum.attr import IntegratedGradients
        except ImportError as exc:
            raise ImportError(
                "Captum is required for IntegratedGradientsMethod. Install with: pip install captum"
            ) from exc
        self.captum_attr_cls = IntegratedGradients
        kwargs.setdefault("n_steps", n_steps)
        super().__init__(**kwargs)
