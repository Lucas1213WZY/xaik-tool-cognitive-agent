"""Gradient-based explainers for neural network models."""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from .attribution_explainer import AttributionExplainer


class GradientInputExplainer(AttributionExplainer):
    """
    Gradient × Input attribution method for differentiable models.
    
    Attribution_j = (∂y/∂x_j) * x_j
    Simple but effective: combines gradient direction with input magnitude.
    
    From: src/coax/feature_importance/gradient_input_explainer.py
    
    Requirements:
        pip install torch
    """
    
    def __init__(self, model, predict_fn: Callable, device: str = 'cpu', **kwargs):
        """
        Initialize Gradient×Input explainer.
        
        Args:
            model: PyTorch model with forward method
            predict_fn: Callable that returns class probabilities
            device: Device to run model on ('cpu' or 'cuda')
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        self.model = model
        self.predict_fn = predict_fn
        self.device = device
        self.torch = torch
        
        self.model.eval()
        self.model.to(device)
    
    def apply(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Explain single instance.
        
        Args:
            instance: Single data point (1D array)
            
        Returns:
            Dict with gradient×input attributions
        """
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        X = self.torch.tensor(instance, dtype=self.torch.float32, requires_grad=True, device=self.device)
        
        # Forward pass
        probs = self.predict_fn(X)
        if probs.ndim == 2:
            y1 = probs[:, 1]
        else:
            y1 = probs
        
        # Backward
        self.model.zero_grad()
        grads = self.torch.autograd.grad(
            outputs=y1,
            inputs=X,
            grad_outputs=self.torch.ones_like(y1),
            retain_graph=False,
            create_graph=False
        )[0]
        
        attributions = (grads * X).detach().cpu().numpy()[0]
        
        return {
            'attributions': attributions,
            'feature_importance': attributions,
            'method': 'gradient_×_input'
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
            'name': 'gradient_input',
            'version': '0.1',
            'type': 'gradient_based',
            'framework': 'torch',
            'description': 'Gradient × Input: element-wise product of gradients and input values',
            'requires_differentiable_model': True
        }


class DeepLIFTExplainer(AttributionExplainer):
    """
    DeepLIFT (reference-based attribution).
    
    More stable than raw gradients; uses reference/baseline as anchor.
    Attribution based on difference-from-baseline rather than slopes.
    
    From: src/coax/feature_importance/deeplift_explainer.py
    
    Requirements:
        pip install torch captum
    """
    
    def __init__(self, model, predict_fn: Callable, 
                 baseline_data: np.ndarray = None,
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize DeepLIFT explainer.
        
        Args:
            model: PyTorch model
            predict_fn: Callable model prediction function
            baseline_data: Data to compute baseline from (mean by default)
            device: Device to run model on
        """
        try:
            import torch
            from captum.attr import DeepLift
        except ImportError:
            raise ImportError("PyTorch and Captum not installed. Install with: pip install torch captum")
        
        self.model = model
        self.predict_fn = predict_fn
        self.device = device
        self.torch = torch
        
        self.model.eval()
        self.model.to(device)
        
        self.attr = DeepLift(self.model)
        
        # Compute baseline
        if baseline_data is not None:
            baseline_vec = self.torch.tensor(baseline_data, dtype=self.torch.float32, device=device)
            self.baseline_vec = baseline_vec.mean(dim=0, keepdim=True)
        else:
            self.baseline_vec = None
        
        # Intercept: model prob at baseline
        if self.baseline_vec is not None:
            with self.torch.no_grad():
                base_prob = self.predict_fn(self.baseline_vec.cpu().numpy())[0, 1]
            self.intercept_value = float(base_prob)
        else:
            self.intercept_value = 0.0
    
    def apply(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain single instance."""
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        X = self.torch.tensor(instance, dtype=self.torch.float32, device=self.device)
        baselines = self.baseline_vec.repeat(X.shape[0], 1) if self.baseline_vec is not None else self.torch.zeros_like(X)
        
        attributions = self.attr.attribute(X, baselines=baselines, target=1).detach().cpu().numpy()[0]
        
        return {
            'deeplift_attributions': attributions,
            'feature_importance': attributions,
            'baseline': self.baseline_vec.cpu().numpy()[0] if self.baseline_vec is not None else None
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
            'name': 'deeplift',
            'version': '0.1',
            'type': 'gradient_based',
            'framework': 'captum',
            'description': 'DeepLIFT: reference-based attribution with reduced noise',
            'requires_differentiable_model': True,
            'parameters': {
                'baseline_type': 'mean',
                'target_class': 1
            }
        }


class IntegratedGradientsExplainer(AttributionExplainer):
    """
    Integrated Gradients: path integral of gradients from baseline to input.
    
    Theoretically grounded; satisfies completeness and sensitivity axioms.
    Accumulates gradients along the straight-line path from baseline to input.
    
    From: src/coax/feature_importance/integrated_gradients_explainer.py
    
    Requirements:
        pip install torch captum
    """
    
    def __init__(self, model, predict_fn: Callable,
                 baseline_data: np.ndarray = None,
                 n_steps: int = 50,
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize Integrated Gradients explainer.
        
        Args:
            model: PyTorch model
            predict_fn: Callable model prediction function
            baseline_data: Data to compute baseline from
            n_steps: Number of steps along integration path
            device: Device to run model on
        """
        try:
            import torch
            from captum.attr import IntegratedGradients
        except ImportError:
            raise ImportError("PyTorch and Captum not installed. Install with: pip install torch captum")
        
        self.model = model
        self.predict_fn = predict_fn
        self.device = device
        self.n_steps = n_steps
        self.torch = torch
        
        self.model.eval()
        self.model.to(device)
        
        self.ig = IntegratedGradients(self.model)
        
        # Compute baseline
        if baseline_data is not None:
            baseline_vec = self.torch.tensor(baseline_data, dtype=self.torch.float32, device=device)
            self.baseline = baseline_vec.mean(dim=0, keepdim=True)
        else:
            self.baseline = None
        
        # Intercept
        if self.baseline is not None:
            with self.torch.no_grad():
                intercept_prob = self.predict_fn(self.baseline.cpu().numpy())[0, 1]
            self.intercept = float(intercept_prob)
        else:
            self.intercept = 0.0
    
    def apply(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain single instance."""
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        X = self.torch.tensor(instance, dtype=self.torch.float32, device=self.device)
        baselines = self.baseline.repeat(X.shape[0], 1) if self.baseline is not None else self.torch.zeros_like(X)
        
        attributions = self.ig.attribute(
            X,
            baselines=baselines,
            target=1,
            n_steps=self.n_steps
        ).detach().cpu().numpy()[0]
        
        return {
            'integrated_gradients': attributions,
            'feature_importance': attributions,
            'baseline': self.baseline.cpu().numpy()[0] if self.baseline is not None else None,
            'n_steps': self.n_steps
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
            'name': 'integrated_gradients',
            'version': '0.1',
            'type': 'gradient_based',
            'framework': 'captum',
            'description': 'Integrated Gradients: path integral of gradients from baseline',
            'requires_differentiable_model': True,
            'parameters': {
                'n_steps': self.n_steps,
                'target_class': 1
            }
        }
