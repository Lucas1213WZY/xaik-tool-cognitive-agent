# User Simulation API

Generate realistic human-like synthetic participant data using CoAX reasoning strategies and fitted cognitive parameters.

## Quick Start (High-Level API)

The easiest way to generate synthetic data:

```python
from src.user_simulation import SessionGenerator, SessionConfig, StrategyConfig

# Define strategy distribution
generator = SessionGenerator()

config = SessionConfig(
    dataset_name="wine_quality",
    n_participants=50,  # 50 participants
    n_trials_per_participant=40,  # 40 trials each
    distribution_file="distributions.json",  # Load parameter distributions
    strategy_configs=[
        StrategyConfig(
            strategy_name="sensitive_features",
            percentage=50.0,  # 50% of participants
            xai_type="importance",
            tested_with_xai=True,  # Show explanations
        ),
        StrategyConfig(
            strategy_name="salient_features",
            percentage=50.0,  # 50% of participants
            xai_type="importance",
            tested_with_xai=False,  # No explanations
        ),
    ],
    random_seed=42,
)

# Generate session
results = generator.generate(config)

# Export to CSV
generator.export_to_csv("session_data.csv")
generator.export_summary("session_summary.json")
```

## Overview

The `user_simulation` module provides a complete pipeline for creating realistic simulated participant responses:

1. **Extract** parameter distributions from empirically fitted cognitive models
2. **Sample** parameters for synthetic participants from distributions
3. **Simulate** per-trial responses using CoAX reasoning strategies
4. **Export** results to CSV for analysis and machine learning

## Components

### 1. ParameterEstimator

Extracts cognitive parameter distributions from fitted CoAX data.

**Input:** CSV file with fitted parameters (from human cognitive modeling)
**Output:** JSON file with parameter statistics (mean, std, range, etc.)

```python
from src.user_simulation import ParameterEstimator

estimator = ParameterEstimator()
estimator.load_fitted_data("fitted_data.csv")
distributions = estimator.estimate_distributions()
estimator.save_distributions("distributions.json")

# Get summary
summary = estimator.get_distribution_summary()
```

**Expected CSV columns:**
- `Strategy`: Strategy name (e.g., "Sensitive-features categorization")
- `Participant Id`: Unique participant identifier
- `appId`: Dataset name (adult, wine_quality, forest_cover)
- `Tested w/ XAI`: "w/ XAI" or "w/o XAI"
- `XAIType`: "Importance", "Attribution", or "None"
- Cognitive parameters (varies by strategy):
  - **Importance-based:** `sensitivity`, `k`, `retrieval_threshold`
  - **Attribution-based:** `scaling_factor`

### 2. ParameterSampler

Samples cognitive parameters for new participants from estimated distributions.

```python
from src.user_simulation import ParameterSampler

sampler = ParameterSampler(seed=42)
sampler.load_distributions("distributions.json")

# Sample parameters for one participant
params = sampler.sample(
    dataset="wine_quality",
    strategy="sensitive_features",
    xai_type="Importance",
    tested_with_xai="w/ XAI",
    method="truncated_normal"  # or "normal", "uniform"
)

# Sample batch (multiple participants)
batch = sampler.sample_batch(
    dataset="wine_quality",
    strategy="sensitive_features",
    n_samples=10
)

# List available distributions
available = sampler.list_strategies_for_dataset("wine_quality")
```

**Sampling methods:**
- `"normal"`: Sample from Gaussian (may go outside observed range)
- `"uniform"`: Sample uniformly from observed range
- `"truncated_normal"`: Sample from truncated Gaussian (stays within range)

### 3. TrialSimulator

Simulates per-trial participant responses using CoAX strategies and sampled parameters.

```python
from src.user_simulation import TrialSimulator, TrialConfig

simulator = TrialSimulator()

# Optional: connect to strategy registry for full CoAX integration
# simulator.setup_dependencies(strategy_registry=registry, memory_system=memory)

config = TrialConfig(
    participant_id="p001",
    dataset_name="wine_quality",
    strategy_name="sensitive_features",
    xai_type="Importance",
    tested_with_xai=True,
    cognitive_params={
        "sensitivity": 76.5,
        "k": 1,
        "retrieval_threshold": -2.97
    },
    n_trials=40,
    ai_dataset_loader=loader,  # Optional: AIDatasetLoader instance
    explainer=explainer,        # Optional: Explanation model
)

# Simulate trials
results = simulator.simulate(config)

# Export to DataFrame
df = simulator.results_to_dataframe(results)

# Export to CSV
simulator.export_to_csv("participant_trials.csv", results)
```

**TrialConfig parameters:**
- `participant_id`: Unique participant identifier
- `dataset_name`: Dataset (adult, wine_quality, forest_cover, mushrooms)
- `strategy_name`: Strategy name (sensitive_features, salient_features, etc.)
- `xai_type`: XAI type (Importance, Attribution, None)
- `tested_with_xai`: Boolean, whether explanations shown
- `cognitive_params`: Dict of parameters (sensitivity, k, retrieval_threshold, etc.)
- `n_trials`: Number of trials to simulate
- `random_seed`: For reproducibility
- `ai_dataset_loader`: AIDatasetLoader for loading instances
- `explainer`: Explanation model (DecisionTreeInterpreter, etc.)

**Output (TrialResult):**
Each trial produces:
- Trial identifiers (participant_id, trial_idx, instance_id)
- Trial conditions (tested_with_xai, strategy, xai_type)
- Predictions (ai_prediction, explainer_prediction)
- Participant response (response, response_prob, response_time)
- Performance metrics (response_matches_ai, response_matches_explainer, correct)

### 4. Utilities

Helper functions for scheduling, statistics, and data manipulation.

```python
from src.user_simulation import (
    normalize_probabilities,
    apply_lapse_rate,
    add_response_noise,
    create_trial_schedule,
    compute_accuracy,
    compute_agreement,
    compute_response_time_stats,
)

# Probability normalization
probs = normalize_probabilities({0: 1.5, 1: 0.5})  # {0: 0.75, 1: 0.25}

# Apply lapse rate (random guessing)
noisy_probs = apply_lapse_rate({0: 0.8, 1: 0.2}, lapse=0.05)

# Add response noise
response_with_noise = add_response_noise(1, noise_rate=0.1)

# Create XAI trial schedule
schedule = create_trial_schedule(n_trials=40, n_xai_trials=20, randomize=True)

# Compute metrics
acc = compute_accuracy(responses, true_labels)
agreement = compute_agreement(responses1, responses2)
stats = compute_response_time_stats(response_times)
```

## End-to-End Example

```python
from src.user_simulation import (
    ParameterEstimator,
    ParameterSampler,
    TrialSimulator,
    TrialConfig,
)
import os

# Step 1: Extract distributions from fitted human data
print("Extracting parameter distributions...")
estimator = ParameterEstimator()
estimator.load_fitted_data("fitted_data.csv")
estimator.estimate_distributions()
estimator.save_distributions("distributions.json")

# Step 2: Initialize sampler
print("Loading distributions...")
sampler = ParameterSampler(seed=42)
sampler.load_distributions("distributions.json")

# Step 3: Generate synthetic participants
print("Generating synthetic participants...")
simulator = TrialSimulator()
all_results = []

n_participants = 10
for i in range(n_participants):
    # Sample parameters
    params = sampler.sample(
        dataset="wine_quality",
        strategy="sensitive_features",
        xai_type="Importance",
        tested_with_xai="w/ XAI",
        method="truncated_normal"
    )
    
    # Create trial config
    config = TrialConfig(
        participant_id=f"synthetic_{i:03d}",
        dataset_name="wine_quality",
        strategy_name="sensitive_features",
        xai_type="Importance",
        tested_with_xai=True,
        cognitive_params=params,
        n_trials=40,
        random_seed=42,
    )
    
    # Simulate trials
    results = simulator.simulate(config)
    all_results.extend(results)

# Step 4: Export results
print(f"Exporting {len(all_results)} trials to CSV...")
df = simulator.results_to_dataframe(all_results)
df.to_csv("synthetic_human_data.csv", index=False)

print(f"Complete! Generated {len(df)} trials")
print(f"Mean response-AI agreement: {df['Response==AI'].mean():.3f}")
print(f"Mean response time: {df['Response Time (s)'].mean():.2f}s")
```

## Command-Line Usage

Generate synthetic data from fitted parameters:

```bash
python src/user_simulation/example_generate_synthetic_data.py \
    --fitted-data fitted_data.csv \
    --output-dir synthetic_output/ \
    --n-participants 10 \
    --n-trials 40 \
    --dataset wine_quality \
    --strategy sensitive_features \
    --xai-type Importance \
    --with-xai \
    --seed 42
```

## Supported Configurations

### Datasets
- `adult`
- `wine_quality`
- `forest_cover`
- `mushrooms`

### Strategies

**Importance-based (require `sensitivity`, `k`, `retrieval_threshold`):**
- `sensitive_features`: Focus on discriminative features (t-test based)
- `salient_features`: Focus on high-magnitude explanation components
- `importance_categorization`: Use explanation vectors for categorization

**Attribution-based (require `scaling_factor`):**
- `attribution_sum`: Sum top-k attribution values

### XAI Types
- `"Importance"`: Importance/saliency explanations
- `"Attribution"`: Attribution/contribution explanations
- `"None"`: No explanation (memory-based retrieval only)

### With/Without XAI
- `"w/ XAI"`: Show explanations to participant
- `"w/o XAI"`: Hide explanations (memory-based only)

## Data Format

### Input: Fitted Parameters CSV

```csv
Strategy,Participant Id,appId,Tested w/ XAI,XAIType,sensitivity,k,retrieval_threshold,scaling_factor
Sensitive-features categorization,p001,wine_quality,w/ XAI,Importance,76.703638943,1,-2.967453087,
Sensitive-features categorization,p002,wine_quality,w/ XAI,Importance,65.440994687,2,-1.423552897,
...
```

### Output: Parameter Distributions JSON

```json
{
  "wine_quality/sensitive_features/importance/with_xai": {
    "dataset": "wine_quality",
    "strategy": "sensitive_features",
    "xai_type": "Importance",
    "tested_with_xai": "w/ XAI",
    "n_samples": 25,
    "parameters": {
      "sensitivity": {
        "param_name": "sensitivity",
        "count": 25,
        "mean": 70.5,
        "std": 15.2,
        "min": 40.0,
        "max": 100.0,
        "median": 72.0,
        "q25": 60.0,
        "q75": 85.0
      },
      "k": { ... },
      "retrieval_threshold": { ... }
    }
  },
  ...
}
```

### Output: Trial Results CSV

```csv
Participant ID,Trial Index,Instance Id,Tested w/ XAI,Strategy,XAI Type,AI Prediction,Explainer Prediction,Response,Response Prob 0,Response Prob 1,Response Time (s),Response==AI,Response==Explainer
synthetic_000,0,123,w/ XAI,sensitive_features,Importance,1,1,1,0.25,0.75,0.53,1,1
synthetic_000,1,456,w/o XAI,sensitive_features,Importance,0,0,0,0.80,0.20,0.48,1,1
...
```

## Integration with CoAX Strategies

The TrialSimulator can be integrated with the full CoAX reasoning strategy system:

```python
from src.cognitive_models import StrategyRegistry
from src.user_simulation import TrialSimulator

# Initialize registry and load strategies
registry = StrategyRegistry()
registry.initialize()

# Connect to simulator
simulator = TrialSimulator()
simulator.setup_dependencies(strategy_registry=registry)

# Now simulate() will use full CoAX inference pipeline
config = TrialConfig(...)
results = simulator.simulate(config)
```

Without registry integration, the simulator falls back to simple heuristic inference.

## Key Features

✅ **Realistic parameter distributions** - Extracts from empirically fitted models
✅ **Flexible sampling** - Multiple sampling methods (normal, uniform, truncated)
✅ **Full CoAX integration** - Can use complete reasoning strategies
✅ **Comprehensive output** - per-trial metrics, accuracy, agreement, timing
✅ **Reproducible** - Seed support for exact reproducibility
✅ **Scalable** - Efficient batch simulation
✅ **Well-documented** - Extensive docstrings and examples

## Troubleshooting

**KeyError: Distribution not found**
- Ensure the dataset/strategy/xai_type combination exists in your fitted data
- Use `sampler.list_strategies_for_dataset(dataset)` to see available combinations

**AttributeError: 'TrialSimulator' object has no attribute...**
- The simulator requires `ai_dataset_loader` to load instances
- Pass it via `TrialConfig` or to `setup_dependencies()`

**Strategy inference failed, falling back to heuristic**
- Full CoAX integration requires `strategy_registry` to be set up
- Call `simulator.setup_dependencies(strategy_registry=registry)` before simulating

## Version

- **v0.1.0** - Initial release
  - Parameter estimation and sampling
  - Trial simulation with heuristic inference
  - CSV export and basic statistics

## License

Part of the xaik-tool-cognitive-agent project.
