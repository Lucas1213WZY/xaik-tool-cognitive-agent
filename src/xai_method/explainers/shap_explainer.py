"""SHAP explainer - Kernel-based SHAP values for feature importance."""

import numpy as np
from typing import List, Dict, Any, Optional
from .attribution_explainer import AttributionExplainer


class SHAPExplainer(AttributionExplainer):
    """
    SHAP (SHapley Additive exPlanations) - Kernel-based approximation.
    
    Computes SHAP values using KernelExplainer for model-agnostic explanations.
    Represents each feature's contribution to pushing the prediction from base value.
    
    From: src/coax/feature_importance/shap_explainer.py
    
    Requirements:
        pip install shap
    """
    
    def __init__(self, predict_fn, background_data=None, n_background_samples: int = 45, **kwargs):
        """
        Initialize SHAP explainer.
        
        Args:
            predict_fn: Callable model prediction function (returns probabilities)
            background_data: Background samples for SHAP (if None, generated from background_samples)
            n_background_samples: Number of background samples to use (if background_data None)
        """
        super().__init__(predict_fn=predict_fn, **kwargs)
        
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        self.n_background = n_background_samples
        
        # Generate or use provided background data
        if background_data is None:
            raise ValueError("background_data must be provided for SHAP explainer")
        
        # Create background summary using kmeans
        self.background_data = shap.kmeans(background_data, min(n_background_samples, len(background_data)))
        
        # Initialize KernelExplainer
        self.explainer = shap.KernelExplainer(predict_fn, self.background_data)
    
    def compute_attributions(self, instance: np.ndarray) -> np.ndarray:
        """Compute SHAP values for instance."""
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        shap_values = self.explainer(instance)
        values = shap_values.values[0, :, 1] if shap_values.values.ndim == 3 else shap_values.values[0, :]
        return values
    
    def apply(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Explain single instance.
        
        Args:
            instance: Single data point (1D array)
            
        Returns:
            Dict with 'shap_values' and 'base_value'
        """
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        shap_values = self.explainer(instance)
        values = shap_values.values[0, :, 1] if shap_values.values.ndim == 3 else shap_values.values[0, :]
        base_value = shap_values.base_values[0, 1] if hasattr(shap_values.base_values, '__getitem__') else shap_values.base_values
        
        return {
            'shap_values': values,
            'base_value': float(base_value),
            'feature_importance': values
        }
    
    def apply_batch(self, instances: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Explain multiple instances.
        
        Args:
            instances: List of data points or 2D array
            
        Returns:
            List of explanation dicts
        """
        if isinstance(instances, list):
            instances = np.array(instances)
        
        if len(instances.shape) == 1:
            instances = instances.reshape(1, -1)
        
        shap_values = self.explainer(instances)
        values = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values
        base_values = shap_values.base_values[:, 1] if shap_values.base_values.ndim == 2 else shap_values.base_values
        
        results = []
        for i in range(len(instances)):
            results.append({
                'shap_values': values[i],
                'base_value': float(base_values[i]),
                'feature_importance': values[i]
            })
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get explainer metadata."""
        return {
            'name': 'shap',
            'version': '0.1',
            'type': 'model_agnostic',
            'framework': 'shap',
            'description': 'Kernel-based SHAP values for feature importance',
            'requires_background': True,
            'parameters': {
                'n_background_samples': self.n_background
            }
        }
