"""LIME explainer - Local Interpretable Model-agnostic Explanations."""

import numpy as np
from typing import List, Dict, Any, Optional
from .attribution_explainer import AttributionExplainer


class LIMEExplainer(AttributionExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations).
    
    Explains individual predictions by fitting local linear models around instances.
    Model-agnostic: works with any black-box predictor.
    
    From: src/coax/feature_importance/lime_explainer.py
    
    Requirements:
        pip install lime
    """
    
    def __init__(self, predict_fn, training_data: np.ndarray, 
                 categorical_features: List[int] = None, 
                 feature_names: List[str] = None,
                 kernel_width: float = 1.5,
                 num_samples: int = 5000,
                 n_bins: int = 4,
                 **kwargs):
        """
        Initialize LIME explainer.
        
        Args:
            predict_fn: Callable model prediction function
            training_data: Training data for discretization and reference
            categorical_features: Indices of categorical features
            feature_names: Names of features
            kernel_width: Kernel width for local model weighting
            num_samples: Number of perturbed samples to generate
            n_bins: Number of bins for continuous feature discretization
        """
        try:
            import lime.lime_tabular
            from lime.discretize import BaseDiscretizer
        except ImportError:
            raise ImportError("LIME not installed. Install with: pip install lime")
        
        self.predict_fn = predict_fn
        self.training_data = training_data
        self.categorical_features = categorical_features or []
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(training_data.shape[1])]
        self.kernel_width = kernel_width
        self.num_samples = num_samples
        self.n_bins = n_bins
        
        # Create custom discretizer
        class CustomDiscretizer(BaseDiscretizer):
            def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None, bins=4):
                self.num_bins = bins
                super().__init__(data, categorical_features, feature_names, labels=labels, random_state=random_state)
            
            def bins(self, data, labels):
                bins = []
                for feature in self.to_discretize:
                    qts = np.percentile(data[:, feature], np.linspace(0, 100, self.num_bins + 1))
                    bins.append(qts)
                return bins
        
        custom_discretizer = CustomDiscretizer(
            training_data,
            categorical_features=self.categorical_features,
            feature_names=self.feature_names,
            bins=n_bins
        )
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            mode='classification',
            categorical_features=self.categorical_features,
            feature_names=self.feature_names,
            feature_selection='auto',
            kernel_width=kernel_width,
            discretizer=custom_discretizer,
            discretize_continuous=True,
            sample_around_instance=True
        )
    
    def apply(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Explain single instance.
        
        Args:
            instance: Single data point (1D array)
            
        Returns:
            Dict with LIME explanation
        """
        if len(instance.shape) == 1:
            instance = instance.reshape(-1)
        
        explanation = self.explainer.explain_instance(
            instance,
            self.predict_fn,
            num_features=min(50, len(self.feature_names)),
            num_samples=self.num_samples
        )
        
        # Extract importance values
        importance_values = np.zeros(len(self.feature_names))
        explanation_map = dict(explanation.as_map().get(1, {}))
        
        for feature_index, importance in explanation_map.items():
            importance_values[feature_index] = importance
        
        intercept = explanation.intercept.get(1, 0.0) if hasattr(explanation, 'intercept') else 0.0
        
        return {
            'lime_coefficients': importance_values,
            'intercept': float(intercept),
            'feature_importance': importance_values,
            'explanation_object': explanation
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
        
        results = []
        for i in range(len(instances)):
            results.append(self.apply(instances[i]))
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get explainer metadata."""
        return {
            'name': 'lime',
            'version': '0.1',
            'type': 'model_agnostic',
            'framework': 'lime',
            'description': 'Local Interpretable Model-agnostic Explanations',
            'parameters': {
                'kernel_width': self.kernel_width,
                'num_samples': self.num_samples,
                'n_bins': self.n_bins
            }
        }
