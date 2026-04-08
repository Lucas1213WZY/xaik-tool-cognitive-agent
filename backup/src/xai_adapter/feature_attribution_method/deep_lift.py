"""Captum DeepLift attribution method."""

from __future__ import annotations

from ._captum import _CaptumMethod


class DeepLiftMethod(_CaptumMethod):
    """Captum DeepLift method."""

    method_name = "deeplift"

    def __init__(self, **kwargs):
        try:
            from captum.attr import DeepLift
        except ImportError as exc:
            raise ImportError("Captum is required for DeepLiftMethod. Install with: pip install captum") from exc
        self.captum_attr_cls = DeepLift
        super().__init__(**kwargs)
