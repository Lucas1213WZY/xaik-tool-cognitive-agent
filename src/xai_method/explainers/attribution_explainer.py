"""
Base class for all attribution-based explainers.

Consolidates common patterns for SHAP, LIME, LOFO, Gradient×Input, DeepLIFT, 
Integrated Gradients and other attribution methods.
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from ..base import BaseExplainer


class AttributionExplainer(BaseExplainer):
    """
    Abstract base class for attribution-based explainers.
    
    Attribution methods assign numerical importance scores to features, typically
    representing how much each feature contributes to the model's prediction for
    a given instance.
    
    Common patterns:
    - Features ranked by attribution magnitude
    - Relative to a baseline (e.g., average, zero)
    - Satisfies some desirable axioms (additivity, sensitivity, etc.)
    
    Subclasses should implement:
    - compute_attributions(): Core attribution algorithm
    - get_baseline(): How to compute baseline/reference values
    """
    
    def __init__(self, 
                 model: Optional[Any] = None,
                 predict_fn: Optional[Callable] = None,
                 baseline_type: str = 'mean',
                 baseline_data: Optional[np.ndarray] = None,
                 normalize_attributions: bool = False,
                 aggregation_method: str = 'sum',
                 **kwargs):
        """
        Initialize attribution explainer with common parameters.
        
        Args:
            model: Optional model object (for gradient-based methods)
            predict_fn: Callable model prediction function
            baseline_type: How to compute baseline ('mean', 'median', 'zeros', 'custom')
            baseline_data: Data to compute baseline from
            normalize_attributions: Whether to normalize attributions (e.g., L2 norm)
            aggregation_method: How to aggregate multi-output attributions ('sum', 'max', 'mean')
        """
        super().__init__(source_type='attribution')
        
        self.model = model
        self.predict_fn = predict_fn
        self.baseline_type = baseline_type
        self.baseline_data = baseline_data
        self.normalize_attributions = normalize_attributions
        self.aggregation_method = aggregation_method
        
        # Computed baseline/intercept
        self.baseline = None
        self.intercept = None
        
        # Compute baseline if data provided
        if baseline_data is not None:
            self.baseline = self._compute_baseline(baseline_data, baseline_type)
            if predict_fn is not None:
                self.intercept = self._compute_intercept()
    
    # ========================================================================
    # ABSTRACT METHODS - Subclasses must implement
    # ========================================================================
    
    @abstractmethod
    def compute_attributions(self, instance: np.ndarray) -> np.ndarray:
        """
        Core attribution computation.
        
        Args:
            instance: Single instance (1D or 2D array)
            
        Returns:
            Attribution scores per feature (1D array)
        """
        pass
    
    # ========================================================================
    # COMMON BASELINE COMPUTATION
    # ========================================================================
    
    def _compute_baseline(self, data: np.ndarray, baseline_type: str) -> np.ndarray:
        """
        Compute baseline/reference values from data.
        
        Args:
            data: Training or reference data
            baseline_type: Type of baseline ('mean', 'median', 'zeros', 'custom')
            
        Returns:
            Baseline array (same dimensionality as data features)
        """
        if baseline_type == 'mean':
            return np.mean(data, axis=0)
        elif baseline_type == 'median':
            return np.median(data, axis=0)
        elif baseline_type == 'zeros':
            return np.zeros(data.shape[1])
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")
    
    def _compute_intercept(self) -> float:
        """Compute intercept: prediction at baseline."""
        if self.baseline is None or self.predict_fn is None:
            return 0.0
        
        baseline_reshaped = self.baseline.reshape(1, -1) if len(self.baseline.shape) == 1 else self.baseline
        probs = self.predict_fn(baseline_reshaped)
        
        # Handle different prediction output shapes
        if isinstance(probs, np.ndarray):
            if probs.ndim == 2 and probs.shape[1] > 1:
                return float(probs[0, 1])  # Binary classification, class 1
            else:
                return float(probs[0])
        else:
            return float(probs)
    
    # ========================================================================
    # COMMON ATTRIBUTION PROCESSING
    # ========================================================================
    
    def _normalize_attributions(self, attributions: np.ndarray) -> np.ndarray:
        """
        Normalize attributions (e.g., L2 norm).
        
        Args:
            attributions: Attribution scores (shape: n_features or n_instances x n_features)
            
        Returns:
            Normalized attributions
        """
        if not self.normalize_attributions:
            return attributions
        
        if attributions.ndim == 1:
            norm = np.linalg.norm(attributions)
            return attributions / (norm + 1e-10)
        else:  # 2D
            norms = np.linalg.norm(attributions, axis=1, keepdims=True)
            return attributions / (norms + 1e-10)
    
    def _aggregate_attributions(self, attributions: np.ndarray) -> np.ndarray:
        """
        Aggregate multi-output attributions to single vector.
        
        Args:
            attributions: Shape (n_features, n_outputs) or (n_features,)
            
        Returns:
            Aggregated attributions (n_features,)
        """
        if attributions.ndim == 1:
            return attributions
        
        if self.aggregation_method == 'sum':
            return np.sum(attributions, axis=1)
        elif self.aggregation_method == 'mean':
            return np.mean(attributions, axis=1)
        elif self.aggregation_method == 'max':
            return np.max(np.abs(attributions), axis=1)
        else:
            raise ValueError(f"Unknown aggregation_method: {self.aggregation_method}")
    
    def _ensure_2d(self, instance: np.ndarray) -> np.ndarray:
        """Ensure instance is 2D (n_samples, n_features)."""
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        return instance
    
    def _ensure_1d(self, instance: np.ndarray) -> np.ndarray:
        """Extract single instance if needed."""
        if instance.ndim == 2 and instance.shape[0] == 1:
            instance = instance[0]
        return instance
    
    # ========================================================================
    # STANDARD INTERFACE IMPLEMENTATION
    # ========================================================================
    
    def apply(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Explain single instance.
        
        Args:
            instance: Single data point
            
        Returns:
            Dict with 'attributions', 'baseline', 'intercept', 'feature_importance'
        """
        instance = self._ensure_1d(instance)
        attributions = self.compute_attributions(instance)
        attributions = self._normalize_attributions(attributions)
        attributions = self._aggregate_attributions(attributions)
        
        return {
            'attributions': attributions,
            'feature_importance': attributions,
            'baseline': self.baseline,
            'intercept': self.intercept,
        }
    
    def apply_batch(self, instances: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Explain multiple instances.
        
        Args:
            instances: List of instances or 2D array
            
        Returns:
            List of explanation dicts
        """
        if isinstance(instances, list):
            instances = np.array(instances)
        
        instances = self._ensure_2d(instances)
        
        results = []
        for i in range(len(instances)):
            results.append(self.apply(instances[i]))
        
        return results
    
    # ========================================================================
    # COMMON METADATA
    # ========================================================================
    
    def get_info(self) -> Dict[str, Any]:
        """Get explainer metadata."""
        return {
            'base_class': 'AttributionExplainer',
            'baseline_type': self.baseline_type,
            'normalize': self.normalize_attributions,
            'aggregation': self.aggregation_method,
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_top_features(self, instance: np.ndarray, k: int = 5) -> Tuple[List[int], np.ndarray]:
        """
        Get top-k most important features for an instance.
        
        Args:
            instance: Instance to explain
            k: Number of top features
            
        Returns:
            (feature_indices, scores)
        """
        explanation = self.apply(instance)
        attributions = explanation['attributions']
        
        top_indices = np.argsort(np.abs(attributions))[-k:][::-1]
        top_scores = attributions[top_indices]
        
        return top_indices.tolist(), top_scores
    
    def explain_difference(self, instance1: np.ndarray, instance2: np.ndarray) -> Dict[str, Any]:
        """
        Explain the difference in predictions between two instances.
        
        Args:
            instance1: First instance
            instance2: Second instance
            
        Returns:
            Dict with:
            - attributions1, attributions2: Explanations for each
            - difference: Attributions difference (inst1 - inst2)
            - pred_diff: Difference in model predictions
        """
        exp1 = self.apply(instance1)
        exp2 = self.apply(instance2)
        
        attr_diff = exp1['attributions'] - exp2['attributions']
        
        pred1 = self.predict_fn(instance1.reshape(1, -1))[0, 1] if self.predict_fn else None
        pred2 = self.predict_fn(instance2.reshape(1, -1))[0, 1] if self.predict_fn else None
        pred_diff = pred1 - pred2 if pred1 is not None and pred2 is not None else None
        
        return {
            'attributions1': exp1['attributions'],
            'attributions2': exp2['attributions'],
            'difference': attr_diff,
            'prediction_diff': pred_diff,
        }
    
    def attribution_summary(self, instances: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute summary statistics over multiple instances.
        
        Args:
            instances: List of instances
            
        Returns:
            Dict with mean, std, min, max attributions per feature
        """
        if isinstance(instances, (list, tuple)):
            instances = np.array(instances)
        
        instances = self._ensure_2d(instances)
        explanations = self.apply_batch(instances)
        
        attributions_array = np.array([exp['attributions'] for exp in explanations])
        
        return {
            'mean': np.mean(attributions_array, axis=0),
            'std': np.std(attributions_array, axis=0),
            'min': np.min(attributions_array, axis=0),
            'max': np.max(attributions_array, axis=0),
            'median': np.median(attributions_array, axis=0),
        }
