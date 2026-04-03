"""
Experiments - Example workflows for synthetic human simulation and evaluation.

This module contains example scripts and utilities for:
1. Running user simulation experiments
2. Evaluating XAI effectiveness
3. Comparing CoAX vs CoXAM reasoning paths
4. Parameter sensitivity analysis

Main entry points:
- experiment_runner.py: Base class for running simulation experiments
- evaluation.py: Evaluation metrics and analysis tools
- coax_evaluation.py: CoAX-specific experiments
- coxam_evaluation.py: CoXAM-specific experiments (with CR agent)

Typical Usage:

```python
from experiments import ExperimentRunner, EvaluationMetrics

# Setup
runner = ExperimentRunner(config={
    'dataset': 'wine_quality',
    'n_participants': 50,
    'n_trials': 40,
    'reasoning_model': 'coxam'  # or 'coax'
})

# Run
results = runner.run()

# Evaluate
metrics = EvaluationMetrics(results)
print(metrics.summary())
```
"""

from . import experiment_runner
from . import evaluation

__version__ = "0.1.0"

__all__ = [
    "experiment_runner",
    "evaluation",
]
