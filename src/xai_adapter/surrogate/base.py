"""Base interfaces for surrogate XAI methods."""

from __future__ import annotations

from typing import Any

from ..base import ArrayLike, XAIAdapter


class SurrogateMethod(XAIAdapter):
    """Base class for fitted surrogate explainers."""

    def predict(self, instances: ArrayLike):
        raise NotImplementedError

    def apply(self, raw_input: Any):
        raise NotImplementedError

    def apply_batch(self, instances):
        return [self.apply(instance) for instance in instances]
