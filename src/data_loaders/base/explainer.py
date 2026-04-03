"""Abstract base class for explainers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import numpy as np


class BaseExplainer(ABC):
    """Abstract base class for model explainers (DT, LR, SHAP, LIME, etc)."""

    def __init__(self, explainer_type: str, metadata: Dict[str, Any] = None):
        """
        Initialize the explainer.
        
        Args:
            explainer_type: Type of explainer (e.g., 'decision_tree', 'logistic_regression')
            metadata: Optional metadata (fidelity, model_name, app_id, etc.)
        """
        self.explainer_type = explainer_type
        self.metadata = metadata or {}
        self.fidelity = self.metadata.get('fidelity', None)
        self.model_name = self.metadata.get('model_name', None)
        self.app_id = self.metadata.get('app_id', None)

    @abstractmethod
    def apply(self, instance: Union[List, np.ndarray]) -> Any:
        """
        Apply the explainer to a single instance.
        
        Args:
            instance: Feature vector (raw or normalized depending on explainer)
            
        Returns:
            Prediction/explanation result (format depends on explainer type)
        """
        pass

    @abstractmethod
    def apply_batch(self, instances: List[Union[List, np.ndarray]]) -> List[Any]:
        """
        Apply the explainer to multiple instances.
        
        Args:
            instances: List of feature vectors
            
        Returns:
            List of prediction/explanation results
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the explainer."""
        return {
            'type': self.explainer_type,
            'model_name': self.model_name,
            'app_id': self.app_id,
            'fidelity': self.fidelity,
            **self.metadata
        }

    def apply_to_instance(self, instance: Union[List, np.ndarray]) -> Any:
        """Backward-compatible alias used by older simulation modules."""
        return self.apply(instance)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type='{self.explainer_type}', app_id='{self.app_id}')"
