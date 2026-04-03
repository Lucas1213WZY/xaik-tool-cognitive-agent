"""
Dependencies Summary for Extended Explainers

This module documents all external dependencies and provides installation instructions.
"""

# ==============================================================================
# REQUIRED vs OPTIONAL DEPENDENCIES
# ==============================================================================

# Core (always required)
CORE_REQUIREMENTS = [
    "numpy",
    "pandas",  # Used by data loaders
]

# Optional by explainer type
OPTIONAL_REQUIREMENTS = {
    "shap": {
        "package": "shap",
        "version": ">=0.42.0",
        "explainers": ["SHAPExplainer"],
        "install": "pip install shap",
        "description": "Kernel-based SHAP values for feature importance"
    },
    "lime": {
        "package": "lime",
        "version": ">=0.2.0",
        "explainers": ["LIMEExplainer"],
        "install": "pip install lime",
        "description": "Local Interpretable Model-agnostic Explanations"
    },
    "torch": {
        "package": "torch",
        "version": ">=1.9.0",
        "explainers": ["GradientInputExplainer"],
        "install": "pip install torch",
        "description": "PyTorch - deep learning framework for gradient-based attribution"
    },
    "captum": {
        "package": "captum",
        "version": ">=0.4.0",
        "explainers": ["DeepLIFTExplainer", "IntegratedGradientsExplainer"],
        "requires": ["torch"],
        "install": "pip install captum",
        "description": "Model Interpretability for PyTorch - gradient-based attribution methods"
    },
}

# ==============================================================================
# LIBRARY USAGE MATRIX
# ==============================================================================

EXPLAINER_DEPENDENCIES = {
    "decision_tree": {
        "core": ["numpy"],
        "optional": [],
        "type": "model_based"
    },
    "logistic_regression": {
        "core": ["numpy"],
        "optional": [],
        "type": "model_based"
    },
    "lofo": {
        "core": ["numpy"],
        "optional": [],
        "type": "model_agnostic"
    },
    "shap": {
        "core": ["numpy"],
        "optional": ["shap"],
        "type": "model_agnostic"
    },
    "lime": {
        "core": ["numpy"],
        "optional": ["lime"],
        "type": "model_agnostic"
    },
    "gradient_input": {
        "core": ["numpy"],
        "optional": ["torch"],
        "type": "gradient_based"
    },
    "deeplift": {
        "core": ["numpy"],
        "optional": ["torch", "captum"],
        "type": "gradient_based"
    },
    "integrated_gradients": {
        "core": ["numpy"],
        "optional": ["torch", "captum"],
        "type": "gradient_based"
    },
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_installation_instructions():
    """Get installation instructions for all optional dependencies."""
    instructions = []
    instructions.append("# Install core dependencies:")
    instructions.append("pip install numpy pandas")
    instructions.append("")
    instructions.append("# Install optional explainers as needed:")
    instructions.append("")
    
    for lib, info in OPTIONAL_REQUIREMENTS.items():
        instructions.append(f"# {info['description']}")
        instructions.append(f"{info['install']}")
        instructions.append("")
    
    return "\n".join(instructions)

def get_explainer_requirements(explainer_name: str) -> dict:
    """Get requirements for a specific explainer."""
    if explainer_name not in EXPLAINER_DEPENDENCIES:
        raise ValueError(f"Unknown explainer: {explainer_name}")
    
    return EXPLAINER_DEPENDENCIES[explainer_name]

def check_requirements(explainer_name: str) -> tuple:
    """
    Check if required packages are installed for an explainer.
    
    Returns:
        (available: bool, missing: list, description: str)
    """
    requirements = get_explainer_requirements(explainer_name)
    missing = []
    
    for lib in requirements["optional"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    
    available = len(missing) == 0
    
    if available:
        description = f"✓ {explainer_name} - all dependencies available"
    else:
        missing_info = [OPTIONAL_REQUIREMENTS[m]["install"] for m in missing if m in OPTIONAL_REQUIREMENTS]
        description = f"✗ {explainer_name} - missing: {', '.join(missing)}\n  Install: {' && '.join(missing_info)}"
    
    return available, missing, description

def print_dependency_report():
    """Print a report of all explainers and their dependency status."""
    print("\n" + "="*80)
    print("EXPLAINERS DEPENDENCY REPORT")
    print("="*80)
    
    for explainer_name in sorted(EXPLAINER_DEPENDENCIES.keys()):
        available, missing, description = check_requirements(explainer_name)
        print(f"\n{description}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXTENDED EXPLAINERS - INSTALLATION GUIDE")
    print("="*80)
    print(get_installation_instructions())
    print_dependency_report()
