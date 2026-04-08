"""
Examples: Using Extended Explainers in Unified Data Loader

This demonstrates all available explainers:
- Model-agnostic: SHAP, LIME, LOFO
- Gradient-based: Gradient×Input, DeepLIFT, Integrated Gradients
- Model-based: DecisionTree, LogisticRegression (existing)
"""

import numpy as np
from src.data_loaders import UnifiedDataLoader
from src.xai_method import get_registry

# ==============================================================================
# PART 1: Model-Agnostic Explainers (No Special Model Required)
# ==============================================================================

def example_lofo_explainer():
    """LOFO: Leave-One-Feature-Out - Simplest & Most Interpretable."""
    print("\n" + "="*70)
    print("EXAMPLE 1: LOFO (Leave-One-Feature-Out)")
    print("="*70)
    
    # Quick prediction function (e.g., trained sklearn model)
    def predict_fn(X):
        """Dummy predictor: returns probabilities"""
        return np.hstack([
            (1 - 0.3 * X[:, 0] - 0.2 * X[:, 1]).reshape(-1, 1),
            (0.3 * X[:, 0] + 0.2 * X[:, 1]).reshape(-1, 1)
        ])
    
    # Create LOFO explainer
    registry = get_registry()
    lofo = registry.create('lofo',
                           predict_fn=predict_fn,
                           baseline_data=np.random.randn(100, 5),
                           baseline_type='mean')
    
    # Explain an instance
    instance = np.array([0.5, 1.2, -0.3, 0.8, 0.1])
    explanation = lofo.apply(instance)
    
    print(f"\nInstance: {instance}")
    print(f"LOFO Importances:\n{explanation['feature_importance']}")
    print(f"Method: {explanation}")
    print(f"\n✓ LOFO: Takes ~N_features model calls (predictable, fast)")


def example_shap_explainer():
    """SHAP: Model-agnostic Shapley values."""
    print("\n" + "="*70)
    print("EXAMPLE 2: SHAP (Kernel-based)")
    print("="*70)
    
    try:
        import shap
        
        # Dummy prediction function
        def predict_fn(X):
            """Dummy predictor"""
            return np.hstack([
                (1 - 0.3 * X[:, 0]).reshape(-1, 1),
                (0.3 * X[:, 0]).reshape(-1, 1)
            ])
        
        registry = get_registry()
        background_data = np.random.randn(50, 5)
        
        shap_exp = registry.create('shap',
                                   predict_fn=predict_fn,
                                   background_data=background_data,
                                   n_background_samples=45)
        
        instance = np.array([0.5, 1.2, -0.3, 0.8, 0.1])
        explanation = shap_exp.apply(instance)
        
        print(f"\nInstance: {instance}")
        print(f"SHAP Values:\n{explanation['shap_values']}")
        print(f"Base Value: {explanation['base_value']}")
        print(f"\n✓ SHAP: Theoretically grounded Shapley values")
        
    except ImportError:
        print("\n⚠ SHAP not installed. Install with: pip install shap")


def example_lime_explainer():
    """LIME: Local linear model explanations."""
    print("\n" + "="*70)
    print("EXAMPLE 3: LIME (Local Interpretable Model-agnostic)")
    print("="*70)
    
    try:
        import lime.lime_tabular
        
        def predict_fn(X):
            """Dummy predictor"""
            return np.hstack([
                (1 - 0.3 * X[:, 0] - 0.2 * X[:, 1]).reshape(-1, 1),
                (0.3 * X[:, 0] + 0.2 * X[:, 1]).reshape(-1, 1)
            ])
        
        registry = get_registry()
        training_data = np.random.randn(100, 5)
        feature_names = [f"Feature_{i}" for i in range(5)]
        
        lime_exp = registry.create('lime',
                                   predict_fn=predict_fn,
                                   training_data=training_data,
                                   feature_names=feature_names,
                                   kernel_width=1.5)
        
        instance = np.array([0.5, 1.2, -0.3, 0.8, 0.1])
        explanation = lime_exp.apply(instance)
        
        print(f"\nInstance: {instance}")
        print(f"LIME Coefficients:\n{explanation['lime_coefficients']}")
        print(f"Intercept: {explanation['intercept']}")
        print(f"\n✓ LIME: Local linear model (highly interpretable)")
        
    except ImportError:
        print("\n⚠ LIME not installed. Install with: pip install lime")


# ==============================================================================
# PART 2: Gradient-Based Explainers (Require Differentiable Models)
# ==============================================================================

def example_gradient_based_explainers():
    """Gradient-based methods: Gradient×Input, DeepLIFT, Integrated Gradients."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Gradient-Based Explainers")
    print("="*70)
    
    try:
        import torch
        from torch import nn
        
        # Create a simple differentiable model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(5, 16)
                self.fc2 = nn.Linear(16, 2)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return torch.softmax(self.fc2(x), dim=1)
        
        model = SimpleModel()
        model.eval()
        
        def predict_fn(X):
            """Wrapper around model"""
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                return model(X_tensor).numpy()
        
        training_data = np.random.randn(100, 5)
        registry = get_registry()
        
        # -------- Gradient×Input --------
        print("\n▸ Gradient×Input Explainer")
        grad_input_exp = registry.create('gradient_input',
                                         model=model,
                                         predict_fn=predict_fn,
                                         device='cpu')
        
        instance = np.array([[0.5, 1.2, -0.3, 0.8, 0.1]])
        explanation = grad_input_exp.apply(instance[0])
        print(f"  Attributions: {explanation['attributions']}")
        
        # -------- DeepLIFT --------
        try:
            from captum.attr import DeepLift
            
            print("\n▸ DeepLIFT Explainer")
            deeplift_exp = registry.create('deeplift',
                                           model=model,
                                           predict_fn=predict_fn,
                                           baseline_data=training_data,
                                           device='cpu')
            
            explanation = deeplift_exp.apply(instance[0])
            print(f"  DeepLIFT Attributions: {explanation['deeplift_attributions']}")
        except ImportError:
            print("\n▸ DeepLIFT (skipped - requires captum)")
        
        # -------- Integrated Gradients --------
        try:
            from captum.attr import IntegratedGradients
            
            print("\n▸ Integrated Gradients Explainer")
            ig_exp = registry.create('integrated_gradients',
                                     model=model,
                                     predict_fn=predict_fn,
                                     baseline_data=training_data,
                                     n_steps=50,
                                     device='cpu')
            
            explanation = ig_exp.apply(instance[0])
            print(f"  IG Attributions: {explanation['integrated_gradients']}")
        except ImportError:
            print("\n▸ Integrated Gradients (skipped - requires captum)")
    
    except ImportError:
        print("\n⚠ PyTorch not installed. Install with: pip install torch")


# ==============================================================================
# PART 3: Using with UnifiedDataLoader
# ==============================================================================

def example_with_unified_loader():
    """Integrate explainers with UnifiedDataLoader."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Explainers with UnifiedDataLoader")
    print("="*70)
    
    try:
        # Load CoAX data from the standardized assets directory
        loader = UnifiedDataLoader.from_assets(
            source="coax",
            assets_root="assets",
            app_id="wine_quality",
            coax_explanation_type="importance",
        )
        
        # Get available explainers
        registry = get_registry()
        available = registry.list_available()
        
        print(f"\nAvailable Explainers in Registry:")
        print(f"  {', '.join(sorted(available))}")
        
        # Get features for instance
        instances = loader.get_instances([0, 1, 2])
        print(f"\nLoaded {len(instances)} instances from CoAX data")
        
        # Use LOFO explainer
        lofo = registry.create('lofo',
                               predict_fn=lambda X: np.hstack([
                                   (1 - 0.5 * X[:, 0]).reshape(-1, 1),
                                   (0.5 * X[:, 0]).reshape(-1, 1)
                               ]),
                               baseline_data=np.array(instances),
                               baseline_type='mean')
        
        explanation = lofo.apply(instances[0])
        print(f"\nLOFO explanation for instance 0:")
        print(f"  Importances: {explanation['feature_importance']}")
        
    except Exception as e:
        print(f"\n⚠ Could not run with CoAX data: {e}")


# ==============================================================================
# PART 4: Listing All Available Explainers
# ==============================================================================

def example_list_explainers():
    """Show all available explainers."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Available Explainers")
    print("="*70)
    
    registry = get_registry()
    available = registry.list_available()
    
    print("\n📋 All Registered Explainers:")
    print("\nCoXAM-Based (Always Available):")
    print("  • decision_tree (aliases: dt)")
    print("  • logistic_regression (aliases: lr)")
    
    print("\nModel-Agnostic (No Special Model):")
    print("  • lofo (Leave-One-Feature-Out)")
    if 'shap' in available:
        print("  • shap (Kernel-based SHAP values)")
    if 'lime' in available:
        print("  • lime (Local Interpretable Model-agnostic)")
    
    print("\nGradient-Based (Requires Differentiable Model):")
    if 'gradient_input' in available:
        print("  • gradient_input (Gradient × Input)")
    if 'deeplift' in available:
        print("  • deeplift (Reference-based DeepLIFT)")
    if 'integrated_gradients' in available:
        print("  • integrated_gradients (Integrated Gradients)")
    
    print(f"\nTotal Registered: {len(available)} explainer aliases")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXTENDED EXPLAINERS EXAMPLES")
    print("="*70)
    
    example_lofo_explainer()
    example_shap_explainer()
    example_lime_explainer()
    example_gradient_based_explainers()
    example_with_unified_loader()
    example_list_explainers()
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
