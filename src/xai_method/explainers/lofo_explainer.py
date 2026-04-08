"""LOFO (Leave-One-Feature-Out) - Model-agnostic local feature importance."""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from .attribution_explainer import AttributionExplainer


class LOFOExplainer(AttributionExplainer):
    """
    Local Leave-One-Feature-Out (LOFO): model-agnostic feature importance.
    
    For each feature j:
      - Replace x_j with baseline_j (e.g., train mean)
      - Importance_j = f(x)_1 - f(x_{-j})_1  (delta class-1 prob)
    
    Positive importance => feature j increases prediction confidence.
    Simple, interpretable, requires only model predictions.
    
    From: src/coax/feature_importance/lofo_explainer.py
    """
    
    def __init__(self, predict_fn: Callable,
                 baseline_data: np.ndarray = None,
                 baseline_type: str = 'mean',
                 **kwargs):
        """
        Initialize LOFO explainer.
        
        Args:
            predict_fn: Callable model prediction function (returns probabilities)
            baseline_data: Data to compute baseline from
            baseline_type: 'mean', 'median', 'zeros', or custom baseline array
        """
        super().__init__(
            predict_fn=predict_fn,
            baseline_type=baseline_type,
            baseline_data=baseline_data,
            **kwargs
        )
    
    def compute_attributions(self, instance: np.ndarray) -> np.ndarray:
        """
        Compute LOFO attributions by leaving out each feature.
        
        Args:
            instance: Single instance (1D array)
            
        Returns:
            Attribution scores per feature
        """
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        n_features = instance.shape[1]
        base_prob = self.predict_fn(instance)[0, 1]
        
        attributions = np.zeros(n_features)
        for j in range(n_features):
            instance_masked = instance.copy()
            instance_masked[0, j] = self.baseline[j]
            prob_masked = self.predict_fn(instance_masked)[0, 1]
            attributions[j] = base_prob - prob_masked
        
        return attributions
    
    def apply(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain single instance."""
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        attributions = self.compute_attributions(instance)
        
        return {
            'lofo_importances': attributions,
            'feature_importance': attributions,
            'baseline_prediction': float(self.predict_fn(self.baseline.reshape(1, -1))[0, 1])
        }
    
    def apply_batch(self, instances: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Explain multiple instances."""
        if isinstance(instances, list):
            instances = np.array(instances)
        
        if len(instances.shape) == 1:
            instances = instances.reshape(1, -1)
        
        results = []
        for i in range(len(instances)):
            results.append(self.apply(instances[i]))
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get explainer metadata."""
        return {
            'name': 'lofo',
            'version': '0.1',
            'type': 'model_agnostic',
            'framework': 'numpy',
            'description': 'Local Leave-One-Feature-Out: delta prediction when feature is replaced',
            'requires_background': True,
            'parameters': {
                'baseline_type': self.baseline_type
            }
        }
