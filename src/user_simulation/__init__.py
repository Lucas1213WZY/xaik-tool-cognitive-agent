"""
User Simulation API - Generate human-like synthetic participant data.

This module provides tools for creating realistic simulated participant responses
using CoAX reasoning strategies and fitted cognitive parameters.

Main Components:

1. **ParameterEstimator** - Extract parameter distributions from fitted data
   ```python
   estimator = ParameterEstimator()
   estimator.load_fitted_data("fitted_params.csv")
   distributions = estimator.estimate_distributions()
   estimator.save_distributions("distributions.json")
   ```

2. **ParameterSampler** - Sample parameters for synthetic participants
   ```python
   sampler = ParameterSampler()
   sampler.load_distributions("distributions.json")
   params = sampler.sample(dataset="wine_quality", strategy="sensitive_features")
   ```

3. **TrialSimulator** - Simulate per-trial responses using CoAX strategies
   ```python
   simulator = TrialSimulator()
   config = TrialConfig(
       participant_id="p001",
       dataset_name="wine_quality",
       strategy_name="sensitive_features",
       cognitive_params={"sensitivity": 76.5, "k": 1, "retrieval_threshold": -2.97},
       n_trials=40
   )
   results = simulator.simulate(config)
   df = simulator.results_to_dataframe(results)
   ```

4. **Utilities** - Helper functions for scheduling, sampling, and metrics

Typical Workflow:

1. Fit human data using CoAX to get participant parameters → fitted_params.csv
2. Extract distributions: EditorialEstimator → distributions.json
3. Sample parameters for new participants: ParameterSampler
4. Simulate trials: TrialSimulator with sampled parameters
5. Export results to CSV for analysis

Example End-to-End:

```python
from src.user_simulation import (
    ParameterEstimator,
    ParameterSampler,
    TrialSimulator,
    TrialConfig
)

# 1. Extract distributions
estimator = ParameterEstimator()
estimator.load_fitted_data("fitted_data.csv")
estimator.estimate_distributions()
estimator.save_distributions("distributions.json")

# 2. Sample parameters for new participants
sampler = ParameterSampler(seed=42)
sampler.load_distributions("distributions.json")

# 3. Simulate trials
simulator = TrialSimulator()
all_results = []

for i in range(10):  # 10 synthetic participants
    params = sampler.sample(
        dataset="wine_quality",
        strategy="sensitive_features",
        xai_type="Importance",
        tested_with_xai="w/ XAI"
    )
    
    config = TrialConfig(
        participant_id=f"synthetic_{i:03d}",
        dataset_name="wine_quality",
        strategy_name="sensitive_features",
        cognitive_params=params,
        n_trials=40
    )
    
    results = simulator.simulate(config)
    all_results.extend(results)

# 4. Export to CSV
df = simulator.results_to_dataframe(all_results)
df.to_csv("synthetic_human_data.csv", index=False)
```

Supported Datasets:
- adult
- wine_quality
- forest_cover
- mushrooms

Supported Strategies (by XAI type):
- Importance: sensitive_features, salient_features, importance_categorization
- Attribution: attribution_sum
- None: (no explanation dependency)

Parameters:
- Importance-based: sensitivity (float), k (int), retrieval_threshold (float)
- Attribution-based: scaling_factor (float)
"""

from .parameter_estimator import (
    ParameterEstimator,
    ParameterStats,
    DistributionKey,
)

from .parameter_sampler import ParameterSampler

from .distribution_loader import DistributionLoader

from .trial_simulator import (
    TrialSimulator,
    TrialConfig,
    TrialResult,
)

from .session_generator import (
    SessionGenerator,
    SessionConfig,
    StrategyConfig,
)

from .forward_trial_generator import (
    TrialSchedule,
    ExperimentalDesign,
    ForwardTrialDatasetGenerator,
    generate_forward_trials,
)

from .utils import (
    normalize_probabilities,
    apply_lapse_rate,
    add_response_noise,
    add_response_time_jitter,
    create_trial_schedule,
    stratify_instances,
    generate_participant_id,
    compute_accuracy,
    compute_agreement,
    compute_response_time_stats,
)

__version__ = "0.2.0"

__all__ = [
    # Parameter estimation and sampling
    "ParameterEstimator",
    "ParameterStats",
    "DistributionKey",
    "ParameterSampler",
    "DistributionLoader",
    
    # Trial simulation
    "TrialSimulator",
    "TrialConfig",
    "TrialResult",
    
    # Session generation (high-level API)
    "SessionGenerator",
    "SessionConfig",
    "StrategyConfig",
    
    # Forward trial generation (RL-based)
    "TrialSchedule",
    "ExperimentalDesign",
    "ForwardTrialDatasetGenerator",
    "generate_forward_trials",
    
    # Utilities
    "normalize_probabilities",
    "apply_lapse_rate",
    "add_response_noise",
    "add_response_time_jitter",
    "create_trial_schedule",
    "stratify_instances",
    "generate_participant_id",
    "compute_accuracy",
    "compute_agreement",
    "compute_response_time_stats",
]
